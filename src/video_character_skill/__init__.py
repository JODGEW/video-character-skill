"""Transfer a reference person's appearance onto a driving video's motion."""

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
]

__version__ = "0.0.1"
