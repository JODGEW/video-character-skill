# Video Character Skill

POC for an open-source **video character transfer** skill.

Given:

1. **one reference image** — a person, with their appearance and outfit
2. **one driving video** — another person performing the motion

the goal is to generate a video that follows the driving video's motion while
resembling the reference person's identity, hairstyle, accessories, and overall
clothing style.

## Status

One provider implemented: **Kling V3 Standard Motion Control** on fal.ai
(`fal-ai/kling-video/v3/standard/motion-control`). Submit / status / result are
wired up, local files are uploaded to fal storage on submit, and every test runs
against a fake client — **no real inference call has been made yet**. There is
no polling loop and no retry logic; the caller polls `get_status` itself.

## Layout

```
src/video_character_skill/
  schemas.py              # ReferenceImage, DrivingVideo, TransferRequest, Job, JobStatus, ResultVideo
  providers/base.py       # CharacterTransferProvider: submit / get_status / get_result
  providers/fal_kling.py  # FalKlingProvider (fal.ai queue API)
tests/                    # schema validation, provider contract, fal payload/mapping
```

Media is referenced by URI only (an `http(s)` URL or a local path). Local paths
are uploaded through `fal_client.upload_file` at submit time; remote URLs are
passed through untouched. Nothing here decodes or re-encodes video — no FFmpeg,
no storage layer of our own.

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env   # set FAL_KEY before any real run
```

## Checks

```bash
pytest
ruff check .
mypy
```

## Sample assets

`reference_image.png` and `driving_video.mov` in the repo root are local test
inputs and are git-ignored — bring your own.
