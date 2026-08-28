"""Unit tests for the VEED matting provider. No network call is ever made: the
fal client is replaced by an in-memory fake, but the real ``fal_client``
status/error types are used so the mapping is checked against the actual
library."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import fal_client
import pytest

from fal_fakes import COMPLETED_OK, FakeFalClient
from video_character_skill.providers.base import ProviderError
from video_character_skill.providers.fal_veed_matting import (
    APP_ID,
    MAX_VIDEO_URL_CHARS,
    FalVeedMattingProvider,
)
from video_character_skill.schemas import (
    JobStatus,
    MatteCodec,
    MattingRequest,
    SourceVideo,
)

# Shaped exactly like fal's documented GeneralRembgOutput for vp9.
VP9_PAYLOAD: dict[str, Any] = {
    "video": [
        {
            "content_type": "video/webm",
            "url": "https://v3b.fal.media/files/b/0a847713/abc_output.webm",
            "file_name": "output.webm",
            "file_size": 4404019,
        }
    ]
}

# What h264 returns: two files, rgb and alpha, in an order fal does not document.
H264_PAYLOAD: dict[str, Any] = {
    "video": [
        {"content_type": "video/mp4", "url": "https://v3b.fal.media/files/b/x/rgb.mp4"},
        {"content_type": "video/mp4", "url": "https://v3b.fal.media/files/b/x/alpha.mp4"},
    ]
}


@pytest.fixture
def local_video(tmp_path: Path) -> Path:
    video = tmp_path / "o1_strict_prompt.mp4"
    video.write_bytes(b"fake mp4")
    return video


def remote_request(**overrides: Any) -> MattingRequest:
    fields: dict[str, Any] = {
        "source_video": SourceVideo(uri="https://example.com/o1_strict_prompt.mp4")
    }
    fields.update(overrides)
    return MattingRequest(**fields)


def vp9_client(**overrides: Any) -> FakeFalClient:
    return FakeFalClient(result_response=VP9_PAYLOAD, **overrides)


# -- uploads ------------------------------------------------------------


def test_local_video_is_uploaded(local_video: Path) -> None:
    fake = FakeFalClient()
    provider = FalVeedMattingProvider(client=fake)

    arguments = provider.build_arguments(
        MattingRequest(source_video=SourceVideo(uri=str(local_video)))
    )

    assert fake.uploads == [str(local_video)]
    assert arguments["video_url"] == "https://v3.fal.media/files/o1_strict_prompt.mp4"


def test_remote_video_is_passed_through_untouched() -> None:
    fake = FakeFalClient()
    provider = FalVeedMattingProvider(client=fake)

    arguments = provider.build_arguments(remote_request())

    assert fake.uploads == []
    assert arguments["video_url"] == "https://example.com/o1_strict_prompt.mp4"


def test_missing_local_video_fails_before_upload() -> None:
    fake = FakeFalClient()
    provider = FalVeedMattingProvider(client=fake)

    with pytest.raises(ProviderError, match="local file not found"):
        provider.build_arguments(MattingRequest(source_video=SourceVideo(uri="./nope.mp4")))
    assert fake.uploads == []


def test_upload_failure_becomes_provider_error(local_video: Path) -> None:
    fake = FakeFalClient(error=fal_client.FalClientError("upload exploded"))
    provider = FalVeedMattingProvider(client=fake)

    with pytest.raises(ProviderError, match="fal upload failed"):
        provider.build_arguments(
            MattingRequest(source_video=SourceVideo(uri=str(local_video)))
        )


def test_over_long_video_url_is_refused() -> None:
    uri = "https://example.com/" + "a" * MAX_VIDEO_URL_CHARS + ".mp4"
    provider = FalVeedMattingProvider(client=FakeFalClient())

    with pytest.raises(ProviderError, match="the maximum is 2083"):
        provider.build_arguments(MattingRequest(source_video=SourceVideo(uri=uri)))


# -- payload ------------------------------------------------------------


def test_person_matte_payload_is_exactly_the_documented_four_fields() -> None:
    provider = FalVeedMattingProvider(client=FakeFalClient())

    assert provider.build_arguments(remote_request()) == {
        "video_url": "https://example.com/o1_strict_prompt.mp4",
        "output_codec": "vp9",
        "refine_foreground_edges": True,
        "subject_is_person": True,
    }


def test_every_field_is_sent_explicitly_even_when_it_matches_fals_default() -> None:
    """We never rely on the endpoint's own defaults."""
    arguments = FalVeedMattingProvider(client=FakeFalClient()).build_arguments(
        remote_request()
    )

    assert set(arguments) == {
        "video_url",
        "output_codec",
        "refine_foreground_edges",
        "subject_is_person",
    }


def test_flags_and_codec_are_forwarded() -> None:
    provider = FalVeedMattingProvider(client=FakeFalClient())

    arguments = provider.build_arguments(
        remote_request(
            subject_is_person=False,
            refine_foreground_edges=False,
            output_codec=MatteCodec.H264,
        )
    )

    assert arguments["subject_is_person"] is False
    assert arguments["refine_foreground_edges"] is False
    assert arguments["output_codec"] == "h264"


def test_submit_posts_to_the_veed_app_and_returns_a_queued_job() -> None:
    fake = FakeFalClient()
    provider = FalVeedMattingProvider(client=fake)

    job = provider.submit(remote_request())

    assert APP_ID == "veed/video-background-removal"
    application, arguments = fake.submissions[0]
    assert application == APP_ID
    assert arguments["output_codec"] == "vp9"
    assert job.job_id == "req-1"
    assert job.provider == "fal-veed-video-background-removal"
    assert job.status is JobStatus.QUEUED


# -- queue status mapping ----------------------------------------------


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (fal_client.Queued(position=2), JobStatus.QUEUED),
        (fal_client.InProgress(logs=[]), JobStatus.RUNNING),
        (COMPLETED_OK, JobStatus.SUCCEEDED),
    ],
)
def test_queue_states_map_onto_job_statuses(
    status: fal_client.Status, expected: JobStatus
) -> None:
    provider = FalVeedMattingProvider(client=FakeFalClient(status_response=status))

    job = provider.get_status("req-1")

    assert job.status is expected
    assert job.provider == "fal-veed-video-background-removal"
    assert job.error is None


def test_completed_with_error_maps_to_failed() -> None:
    failed = fal_client.Completed(
        logs=[], metrics={}, error="unsupported container", error_type="ValueError"
    )
    provider = FalVeedMattingProvider(client=FakeFalClient(status_response=failed))

    job = provider.get_status("req-1")

    assert job.status is JobStatus.FAILED
    assert job.error == "ValueError: unsupported container"


def test_status_failure_becomes_provider_error() -> None:
    fake = FakeFalClient(error=fal_client.FalClientError("status exploded"))
    provider = FalVeedMattingProvider(client=fake)

    with pytest.raises(ProviderError, match="fal status failed"):
        provider.get_status("req-1")


# -- result parsing -----------------------------------------------------


def test_vp9_result_parses_into_one_matte_video() -> None:
    fake = vp9_client(status_response=COMPLETED_OK)
    provider = FalVeedMattingProvider(client=fake)

    matte = provider.get_result("req-1")

    assert matte.uri == "https://v3b.fal.media/files/b/0a847713/abc_output.webm"
    assert matte.content_type == "video/webm"
    assert matte.is_remote
    assert fake.result_calls == [(APP_ID, "req-1")]


def test_missing_content_type_is_left_unset() -> None:
    payload = {"video": [{"url": "https://example.com/out.webm"}]}
    fake = FakeFalClient(status_response=COMPLETED_OK, result_response=payload)

    assert FalVeedMattingProvider(client=fake).get_result("req-1").content_type is None


def test_two_file_h264_result_is_refused_rather_than_guessed() -> None:
    fake = FakeFalClient(status_response=COMPLETED_OK, result_response=H264_PAYLOAD)
    provider = FalVeedMattingProvider(client=fake)

    with pytest.raises(ProviderError, match="holds 2 files"):
        provider.get_result("req-1")


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"video": []},
        {"video": {"url": "https://example.com/out.webm"}},
        {"videos": [{"url": "https://example.com/out.webm"}]},
        "not a dict",
    ],
)
def test_malformed_results_become_provider_errors(payload: Any) -> None:
    fake = FakeFalClient(status_response=COMPLETED_OK, result_response=payload)
    provider = FalVeedMattingProvider(client=fake)

    with pytest.raises(ProviderError, match="no video list"):
        provider.get_result("req-1")


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"video": ["https://example.com/out.webm"]}, "not a file object"),
        ({"video": [{"content_type": "video/webm"}]}, "no video url"),
        ({"video": [{"url": "   "}]}, "no video url"),
    ],
)
def test_malformed_file_entries_become_provider_errors(payload: Any, message: str) -> None:
    fake = FakeFalClient(status_response=COMPLETED_OK, result_response=payload)
    provider = FalVeedMattingProvider(client=fake)

    with pytest.raises(ProviderError, match=message):
        provider.get_result("req-1")


# -- result failures ----------------------------------------------------


def test_result_before_success_is_refused() -> None:
    fake = vp9_client(status_response=fal_client.Queued(position=1))
    provider = FalVeedMattingProvider(client=fake)

    with pytest.raises(ProviderError, match="is queued, not succeeded"):
        provider.get_result("req-1")
    assert fake.result_calls == []


def test_failed_job_reports_its_error_from_get_result() -> None:
    failed = fal_client.Completed(
        logs=[], metrics={}, error="out of memory", error_type="RuntimeError"
    )
    provider = FalVeedMattingProvider(client=vp9_client(status_response=failed))

    with pytest.raises(ProviderError, match="RuntimeError: out of memory"):
        provider.get_result("req-1")


def test_result_fetch_failure_becomes_provider_error() -> None:
    fake = vp9_client(
        status_response=COMPLETED_OK,
        result_error=fal_client.FalClientError("500 upstream"),
    )
    provider = FalVeedMattingProvider(client=fake)

    with pytest.raises(ProviderError, match="fal result failed"):
        provider.get_result("req-1")


def test_submit_failure_becomes_provider_error() -> None:
    fake = FakeFalClient(error=fal_client.FalClientError("submit exploded"))
    provider = FalVeedMattingProvider(client=fake)

    with pytest.raises(ProviderError, match="fal submit failed"):
        provider.submit(remote_request())
