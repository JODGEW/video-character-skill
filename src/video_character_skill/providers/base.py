"""Provider interface for character-transfer backends.

Providers are asynchronous in the job sense (submit now, poll later) but the
methods are plain synchronous calls: the caller owns the polling loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from video_character_skill.schemas import Job, ResultVideo, TransferRequest


class ProviderError(RuntimeError):
    """A provider call failed (transport, auth, or an upstream job failure)."""


class CharacterTransferProvider(ABC):
    """Submit a transfer job, poll it, then fetch its result video."""

    name: ClassVar[str]

    @abstractmethod
    def submit(self, request: TransferRequest) -> Job:
        """Start a job. The returned job carries the id used by the other calls."""

    @abstractmethod
    def get_status(self, job_id: str) -> Job:
        """Return the current state of a previously submitted job."""

    @abstractmethod
    def get_result(self, job_id: str) -> ResultVideo:
        """Return the output of a succeeded job.

        Raises:
            ProviderError: if the job is not in ``JobStatus.SUCCEEDED``.
        """
