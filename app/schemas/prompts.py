from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator


class PointPrompt(BaseModel):
    type: Literal["point"] = "point"
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    label: Literal["foreground", "background"] = "foreground"


class BoxPrompt(BaseModel):
    type: Literal["box"] = "box"
    x1: int = Field(ge=0)
    y1: int = Field(ge=0)
    x2: int = Field(ge=0)
    y2: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_box_order(self) -> "BoxPrompt":
        if self.x2 <= self.x1:
            raise ValueError("x2 must be greater than x1")
        if self.y2 <= self.y1:
            raise ValueError("y2 must be greater than y1")
        return self


Prompt = Annotated[PointPrompt | BoxPrompt, Field(discriminator="type")]


class ObjectPrompt(BaseModel):
    client_object_id: str | None = Field(default=None, max_length=128)
    prompts: list[Prompt] = Field(min_length=1)


class SubmitObjectPromptsRequest(BaseModel):
    objects: list[ObjectPrompt] = Field(min_length=1)


class StoredObjectPrompt(ObjectPrompt):
    object_id: str


class SubmitObjectPromptsResponse(BaseModel):
    session_id: str
    objects: list[StoredObjectPrompt]
    model_status: Literal["pending_model_integration"] = "pending_model_integration"
