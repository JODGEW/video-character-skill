"""Transfer a reference person's appearance onto a driving video's motion."""

from video_character_skill.masks import (
    MaskDecodeError,
    decode_object_mask,
    decode_rle,
    mask_area_ratio,
    mask_bbox,
)
from video_character_skill.schemas import (
    CharacterOrientation,
    DrivingVideo,
    IdentityElement,
    Job,
    JobStatus,
    MaskFrame,
    ObjectMask,
    ReferenceImage,
    ResultVideo,
    SegmentationRequest,
    TransferRequest,
    VideoEditRequest,
    VideoMaskTrack,
)

__all__ = [
    "CharacterOrientation",
    "MaskDecodeError",
    "DrivingVideo",
    "IdentityElement",
    "Job",
    "JobStatus",
    "MaskFrame",
    "ObjectMask",
    "ReferenceImage",
    "ResultVideo",
    "SegmentationRequest",
    "TransferRequest",
    "VideoEditRequest",
    "VideoMaskTrack",
    "decode_object_mask",
    "decode_rle",
    "mask_area_ratio",
    "mask_bbox",
]

__version__ = "0.0.1"
