from uuid import uuid4

from app.models.base import VideoMaskingModel
from app.schemas.prompts import (
    BoxPrompt,
    PointPrompt,
    StoredObjectPrompt,
    SubmitObjectPromptsRequest,
    SubmitObjectPromptsResponse,
)
from app.services.errors import InvalidPromptError
from app.services.json_store import read_json, write_json
from app.services.session_service import VideoSessionService


class MaskPromptService:
    def __init__(
        self,
        *,
        session_service: VideoSessionService,
        model: VideoMaskingModel,
    ) -> None:
        self.session_service = session_service
        self.model = model

    def submit_prompts(
        self,
        *,
        session_id: str,
        request: SubmitObjectPromptsRequest,
    ) -> SubmitObjectPromptsResponse:
        metadata = self.session_service.get_metadata(session_id)
        width = metadata["first_frame"]["width"]
        height = metadata["first_frame"]["height"]
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
        write_json(
            prompt_path,
            [stored_object.model_dump(mode="json") for stored_object in stored_objects],
        )

        self._try_model_first_frame_segmentation(
            session_id=session_id,
            first_frame_path=self.session_service.get_first_frame_path(session_id),
            objects=objects,
        )

        return SubmitObjectPromptsResponse(session_id=session_id, objects=stored_objects)

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

    def _try_model_first_frame_segmentation(
        self,
        *,
        session_id: str,
        first_frame_path: str,
        objects: list[StoredObjectPrompt],
    ) -> None:
        try:
            self.model.segment_first_frame(
                session_id=session_id,
                first_frame_path=first_frame_path,
                objects=objects,
            )
        except NotImplementedError:
            return
