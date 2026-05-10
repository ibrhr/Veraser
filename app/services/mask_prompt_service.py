import logging
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageDraw

from app.schemas.prompts import (
    BoxPrompt,
    ObjectPrompt,
    ObjectPromptListResponse,
    ObjectPromptResponse,
    PointPrompt,
    StoredObjectPrompt,
    SubmitObjectPromptsRequest,
    SubmitObjectPromptsResponse,
)
from app.services.errors import InvalidPromptError, ObjectPromptNotFoundError
from app.services.json_store import read_json, write_json
from app.services.mask_artifact_service import MaskArtifactService
from app.services.session_service import VideoSessionService


logger = logging.getLogger(__name__)


class MaskPromptService:
    def __init__(
        self,
        *,
        session_service: VideoSessionService,
        artifact_service: MaskArtifactService,
    ) -> None:
        self.session_service = session_service
        self.artifact_service = artifact_service

    def submit_prompts(
        self,
        *,
        session_id: str,
        request: SubmitObjectPromptsRequest,
    ) -> SubmitObjectPromptsResponse:
        metadata = self.session_service.get_metadata(session_id)
        width = metadata["first_frame"]["width"]
        height = metadata["first_frame"]["height"]
        logger.info(
            "Submitting object prompts session_id=%s object_count=%s frame_width=%s frame_height=%s",
            session_id,
            len(request.objects),
            width,
            height,
        )
        self._validate_prompt_bounds(request, width=width, height=height)

        objects = [
            StoredObjectPrompt(
                object_id=str(uuid4()),
                client_object_id=object_prompt.client_object_id,
                prompts=object_prompt.prompts,
            )
            for object_prompt in request.objects
        ]

        prompt_path = self.session_service.get_prompts_path(session_id)
        stored_objects = [
            StoredObjectPrompt.model_validate(item)
            for item in read_json(prompt_path, default=[])
        ]
        stored_objects.extend(objects)
        self._write_objects(prompt_path, stored_objects)
        logger.info(
            "Stored object prompts session_id=%s new_object_count=%s total_object_count=%s",
            session_id,
            len(objects),
            len(stored_objects),
        )
        for stored_object in objects:
            self._write_preview_mask(session_id=session_id, object_prompt=stored_object)

        return SubmitObjectPromptsResponse(session_id=session_id, objects=stored_objects)

    def list_objects(self, session_id: str) -> ObjectPromptListResponse:
        self.session_service.get_metadata(session_id)
        return ObjectPromptListResponse(session_id=session_id, objects=self._read_objects(session_id))

    def update_object(
        self,
        *,
        session_id: str,
        object_id: str,
        request: ObjectPrompt,
    ) -> ObjectPromptResponse:
        metadata = self.session_service.get_metadata(session_id)
        logger.info(
            "Updating object prompt session_id=%s object_id=%s prompt_count=%s",
            session_id,
            object_id,
            len(request.prompts),
        )
        self._validate_prompt_bounds(
            SubmitObjectPromptsRequest(objects=[request]),
            width=metadata["first_frame"]["width"],
            height=metadata["first_frame"]["height"],
        )
        objects = self._read_objects(session_id)
        for index, stored_object in enumerate(objects):
            if stored_object.object_id == object_id:
                updated = StoredObjectPrompt(
                    object_id=object_id,
                    client_object_id=request.client_object_id,
                    prompts=request.prompts,
                )
                objects[index] = updated
                self._write_objects(self.session_service.get_prompts_path(session_id), objects)
                preview_path = self._write_preview_mask(session_id=session_id, object_prompt=updated)
                logger.info("Updated object prompt session_id=%s object_id=%s preview_path=%s", session_id, object_id, preview_path)
                return ObjectPromptResponse(
                    session_id=session_id,
                    object=updated,
                    preview_mask_url=self._preview_mask_url(session_id, object_id),
                )
        raise ObjectPromptNotFoundError(f"Object prompt {object_id} was not found.")

    def delete_object(self, *, session_id: str, object_id: str) -> None:
        self.session_service.get_metadata(session_id)
        logger.info("Deleting object prompt session_id=%s object_id=%s", session_id, object_id)
        objects = self._read_objects(session_id)
        remaining_objects = [item for item in objects if item.object_id != object_id]
        if len(remaining_objects) == len(objects):
            raise ObjectPromptNotFoundError(f"Object prompt {object_id} was not found.")
        self._write_objects(self.session_service.get_prompts_path(session_id), remaining_objects)
        self.artifact_service.object_preview_mask_path(session_id, object_id).unlink(missing_ok=True)
        logger.info(
            "Deleted object prompt session_id=%s object_id=%s remaining_object_count=%s",
            session_id,
            object_id,
            len(remaining_objects),
        )

    def get_preview_mask_path(self, *, session_id: str, object_id: str) -> Path:
        self.session_service.get_metadata(session_id)
        objects = self._read_objects(session_id)
        object_prompt = next((item for item in objects if item.object_id == object_id), None)
        if object_prompt is None:
            raise ObjectPromptNotFoundError(f"Object prompt {object_id} was not found.")
        logger.info("Regenerating preview mask session_id=%s object_id=%s", session_id, object_id)
        return self._write_preview_mask(session_id=session_id, object_prompt=object_prompt)

    def _validate_prompt_bounds(
        self,
        request: SubmitObjectPromptsRequest,
        *,
        width: int,
        height: int,
    ) -> None:
        for object_index, object_prompt in enumerate(request.objects):
            for prompt_index, prompt in enumerate(object_prompt.prompts):
                if isinstance(prompt, PointPrompt):
                    if prompt.x >= width or prompt.y >= height:
                        raise InvalidPromptError(
                            f"objects[{object_index}].prompts[{prompt_index}] point is outside the first frame"
                        )
                if isinstance(prompt, BoxPrompt):
                    if prompt.x2 > width or prompt.y2 > height:
                        raise InvalidPromptError(
                            f"objects[{object_index}].prompts[{prompt_index}] box is outside the first frame"
                        )

    def _read_objects(self, session_id: str) -> list[StoredObjectPrompt]:
        return [
            StoredObjectPrompt.model_validate(item)
            for item in read_json(self.session_service.get_prompts_path(session_id), default=[])
        ]

    def _write_objects(self, prompt_path: Path, objects: list[StoredObjectPrompt]) -> None:
        write_json(prompt_path, [stored_object.model_dump(mode="json") for stored_object in objects])

    def _write_preview_mask(self, *, session_id: str, object_prompt: StoredObjectPrompt) -> Path:
        metadata = self.session_service.get_metadata(session_id)
        width = metadata["first_frame"]["width"]
        height = metadata["first_frame"]["height"]
        preview_path = self.artifact_service.object_preview_mask_path(session_id, object_prompt.object_id)
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Writing preview mask session_id=%s object_id=%s prompt_count=%s path=%s",
            session_id,
            object_prompt.object_id,
            len(object_prompt.prompts),
            preview_path,
        )

        image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        for prompt_index, prompt in enumerate(object_prompt.prompts):
            logger.debug(
                "Drawing preview prompt session_id=%s object_id=%s prompt_index=%s prompt_type=%s",
                session_id,
                object_prompt.object_id,
                prompt_index,
                prompt.type,
            )
            if isinstance(prompt, BoxPrompt):
                draw.rectangle((prompt.x1, prompt.y1, prompt.x2, prompt.y2), fill=(23, 105, 170, 180))
            if isinstance(prompt, PointPrompt) and prompt.label == "foreground":
                radius = max(4, min(width, height) // 100)
                draw.ellipse(
                    (
                        prompt.x - radius,
                        prompt.y - radius,
                        prompt.x + radius,
                        prompt.y + radius,
                    ),
                    fill=(49, 196, 141, 220),
                )
        image.save(preview_path)
        return preview_path

    def _preview_mask_url(self, session_id: str, object_id: str) -> str:
        return f"/api/v1/video-sessions/{session_id}/objects/{object_id}/preview-mask"
