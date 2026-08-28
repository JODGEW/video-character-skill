# Video Character Skill

POC for an open-source **video character transfer** skill.

Given:

1. **one reference image** — a person, with their appearance and outfit
2. **one driving video** — another person performing the motion

the goal is to generate a video that follows the driving video's motion while
resembling the reference person's identity, hairstyle, accessories, and overall
clothing style.

## Status

Prompt-only editing is no longer the target architecture. A Kling O1 POC
(`fal-ai/kling-video/o1/video-to-video/edit`) did replace the person, but it
materially regenerated the original room background even under a strict prompt
demanding a frame-faithful background — so background preservation has to be
*enforced*, with a temporal person mask, not asked for.

Four providers are implemented, all submit / status / result against the fal
queue, all covered in tests by fakes only:

| Provider | fal endpoint | Role |
| --- | --- | --- |
| `FalKlingProvider` | `fal-ai/kling-video/v3/standard/motion-control` | regenerates the scene; superseded |
| `FalKlingO1EditProvider` | `fal-ai/kling-video/o1/video-to-video/edit` | prompt-only edit; the POC above |
| `FalSam3VideoMaskProvider` | `fal-ai/sam-3/video-rle-objects` | per-frame RLE masks; **rejected as the matte source, see below** |
| `FalVeedMattingProvider` | `veed/video-background-removal` | the person matte: one VP9/WebM with an alpha channel |

Nothing composites yet: the providers only fetch a matte. There is no polling
loop and no retry logic; the caller polls `get_status` itself.

Temporal alignment between `out/source_aligned_24fps.mp4` and
`out/o1_strict_prompt.mp4` has been QA'd and passes — both are 1080x1920, 24 fps,
225 frames, 9.375 s, and same-index compositing is safe (head-top trajectories
correlate at r = 0.9886 at lag 0, with no drift across the clip).

Two real `video-rle-objects` runs over `driving_video_o1.mp4` confirmed the
endpoint: 285/285 frames returned, exactly one stable `track_id` (0), and mask
dimensions of 1080x1920 matching the source exactly. Tightening the prompt
("young man", threshold 0.8) changed the masks only negligibly, so the defaults
stand — `prompt="person"`, `detection_threshold=0.5`.

## The RLE format

fal documents `rle` only as "Run-length encoding (Kaggle/COCO order)", which
names two incompatible formats. Those same runs settled it: the strings are
**1-based start/length pairs flattened row-major (C order)**, not COCO
alternating runs. Consecutive starts sit ~1080 apart — one frame width — and
decoding that way puts the mask on the subject; read as COCO the values sum to
1,065,761,049 against a 2,073,600-pixel frame. `masks.decode_rle` implements
exactly that contract and rejects anything else rather than clipping it.

## Why the matte comes from VEED, not SAM 3

SAM 3 decoded correctly but segmented the wrong thing. Across two real runs its
masks covered 53–82 % of the frame, with a bounding box pinned to the left,
right and bottom borders in all 285 frames and a horizontal hole on only 0.36 %
of rows — a blob plus the whole floor, not a person. Prompt and threshold
changes did not help (per-frame IoU between the two runs was 0.9985).

`veed/video-background-removal` returns an alpha channel directly, so there is
no RLE to decode and no mask semantics to infer. We ask for `vp9`, which
returns **exactly one** WebM carrying the alpha. The `h264` option returns two
files, "rgb and alpha", and fal documents which is which nowhere — not in the
queue OpenAPI, the API docs, the model page, or the generated
`@fal-ai/client` typings. Feeding RGB to a compositor as a matte fails silently,
so the provider refuses a two-file result rather than guessing. fal's stated
reason to prefer h264 is "better RGB quality", which does not apply: we
composite the clip we sent in, not VEED's RGB.

## Why `video-rle-objects` was the SAM 3 choice

Of fal's SAM 3 endpoints, only this one returns mask *data* rather than a
rendered video. `sam-3/video` and `sam-3/video-rle` both declare
`SAM3VideoOutput` (a `video` File) — `apply_mask` only changes what is drawn
into that video, and `video-rle` "collapses every tracked object into a single
union mask per frame". `sam-3/image-rle` returns real RLE but per image, with
no tracking. `sam-3/video-rle-objects` returns, per frame, one RLE mask per
tracked object plus a track id that is stable across the whole clip. See
`providers/fal_sam3.py` for the full comparison and the verbatim schemas.

## Layout

```
src/video_character_skill/
  schemas.py                 # media assets, TransferRequest, SegmentationRequest,
                             # Job/JobStatus, ResultVideo, VideoMaskTrack
  providers/base.py          # CharacterTransferProvider: submit / get_status / get_result
  providers/_fal_queue.py    # FalQueueBase (queue + uploads), FalQueueProvider
  providers/fal_kling.py     # Kling V3 Standard Motion Control
  providers/fal_kling_o1.py  # Kling O1 video-to-video edit
  providers/fal_sam3.py      # SAM 3 per-object video segmentation (superseded)
  providers/fal_veed_matting.py  # VEED background removal -> alpha matte
  masks.py                   # decode SAM 3 RLE into (height, width) bool arrays
tests/                       # schemas, provider contract, fal payload/mapping, RLE decoding
```

Media is referenced by URI only (an `http(s)` URL or a local path). Local paths
are uploaded through `fal_client.upload_file` at submit time; remote URLs are
passed through untouched. Nothing here decodes or re-encodes video — no FFmpeg,
no storage layer of our own.

Masks are the one thing that becomes an array: `masks.py` turns an `ObjectMask`
into a NumPy boolean mask for the compositing step to come.

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

`reference_image.png`, `driving_video.mov` and `driving_video_o1.mp4` in the
repo root are local test inputs and are git-ignored — bring your own.
`driving_video_o1.mp4` is 1080x1920, 30 fps, 285 frames (9.5 s).
