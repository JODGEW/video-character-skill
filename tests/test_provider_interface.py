from __future__ import annotations

from typing import ClassVar

import pytest

from video_character_skill.providers import CharacterTransferProvider, ProviderError
from video_character_skill.schemas import (
    DrivingVideo,
    Job,
    JobStatus,
    ReferenceImage,
    ResultVideo,
    TransferRequest,
)


class FakeProvider(CharacterTransferProvider):
    """In-memory provider used to pin down the interface contract."""

    name: ClassVar[str] = "fake"

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}

    def submit(self, request: TransferRequest) -> Job:
        job = Job(job_id=f"job-{len(self._jobs) + 1}", provider=self.name)
        self._jobs[job.job_id] = job
        return job

    def get_status(self, job_id: str) -> Job:
        try:
            return self._jobs[job_id]
        except KeyError:
            raise ProviderError(f"unknown job: {job_id}") from None

    def get_result(self, job_id: str) -> ResultVideo:
        job = self.get_status(job_id)
        if job.status is not JobStatus.SUCCEEDED:
            raise ProviderError(f"job {job_id} is {job.status.value}, not succeeded")
        return ResultVideo(uri=f"https://example.com/{job_id}.mp4")

    def _finish(self, job_id: str) -> None:
        self._jobs[job_id] = self._jobs[job_id].model_copy(
            update={"status": JobStatus.SUCCEEDED}
        )


@pytest.fixture
def request_() -> TransferRequest:
    return TransferRequest(
        reference_image=ReferenceImage(uri="./reference_image.png"),
        driving_video=DrivingVideo(uri="./driving_video.mov"),
    )


def test_interface_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        CharacterTransferProvider()  # type: ignore[abstract]


def test_submit_then_poll_then_fetch_result(request_: TransferRequest) -> None:
    provider = FakeProvider()

    job = provider.submit(request_)
    assert job.provider == "fake"
    assert not job.is_terminal

    with pytest.raises(ProviderError):
        provider.get_result(job.job_id)

    provider._finish(job.job_id)
    assert provider.get_status(job.job_id).status is JobStatus.SUCCEEDED
    assert provider.get_result(job.job_id).uri.endswith(".mp4")


def test_unknown_job_raises_provider_error() -> None:
    with pytest.raises(ProviderError):
        FakeProvider().get_status("nope")
