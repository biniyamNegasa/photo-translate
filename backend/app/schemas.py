from enum import Enum
from typing import Annotated, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

ColorChannel = Annotated[int, Field(ge=0, le=255)]
RGBColor = Tuple[ColorChannel, ColorChannel, ColorChannel]


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ItemStatus(str, Enum):
    QUEUED = "queued"
    DETECTING = "detecting"
    INPAINTING = "inpainting"
    TRANSLATING = "translating"
    RENDERING = "rendering"
    COMPLETED = "completed"
    NO_TEXT = "no_text"
    FAILED = "failed"


class TextStyle(BaseModel):
    fg_color: RGBColor = (255, 255, 255)
    bg_color: RGBColor = (0, 0, 0)
    has_stroke: bool = False
    stroke_color: RGBColor = (0, 0, 0)
    stroke_width: int = Field(0, ge=0, le=32)
    is_bold: bool = False
    is_uppercase: bool = False
    angle: float = Field(0.0, ge=-360.0, le=360.0)
    rotation: float = Field(0.0, ge=-360.0, le=360.0)
    font_size_hint: int = Field(24, ge=6, le=512)
    alignment: Literal["left", "center", "right"] = "center"
    uniform_bg: bool = False
    comic_style: bool = False


class TextBlock(BaseModel):
    id: str
    polygon: List[List[float]]          # 4+ corner points [[x, y], ...]
    bbox: Tuple[int, int, int, int]     # x1, y1, x2, y2
    expanded_bbox: Tuple[int, int, int, int]  # optional uniform-bg expanded bbox
    source_text: str
    translated_text: str
    source_lang: str = "en"
    confidence: float = 1.0
    enabled: bool = True
    line_count: int = 1
    is_vertical: bool = False
    style: TextStyle = Field(default_factory=TextStyle)


class JobItemInfo(BaseModel):
    item_id: str
    filename: str
    width: int
    height: int
    format: str
    status: ItemStatus = ItemStatus.QUEUED
    stage_message: str = "Queued"
    error: Optional[str] = None
    blocks: List[TextBlock] = Field(default_factory=list)
    original_url: str = ""
    translated_url: str = ""


class JobInfo(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.QUEUED
    source_lang: str = "auto"
    target_lang: str = "en"
    translator: str = "google"
    created_at: float
    expires_at: float
    total_items: int = 0
    completed_items: int = 0
    items: List[JobItemInfo] = Field(default_factory=list)


class BlockUpdateItem(BaseModel):
    id: str = Field(..., min_length=1, max_length=64)
    translated_text: Optional[str] = Field(default=None, max_length=2000)
    enabled: Optional[bool] = None


class RerenderRequest(BaseModel):
    blocks: List[BlockUpdateItem] = Field(..., max_length=250)


class LanguageDTO(BaseModel):
    code: str
    name: str
    native_name: str
    script: str
    rtl: bool


class TranslatorInfo(BaseModel):
    id: str
    name: str
    description: str
    available: bool
    is_local: bool


class SystemStatus(BaseModel):
    max_upload_mb: int
    max_batch_images: int
    translators: List[TranslatorInfo]
