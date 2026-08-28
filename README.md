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

`compositor.py` puts the original background back; it has not been run on the
real clips yet. There is no polling loop and no retry logic; the caller polls
`get_status` itself.

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

## Compositing

`compositor.composite_video` streams three clips through ffmpeg pipes and
writes one MP4:

```
output = alpha * replacement + (1 - alpha) * source
```

Pixels are partitioned by matte alpha rather than blended uniformly. The output
buffer starts as a copy of the source frame, so **`alpha == 0` pixels are
preserved by construction** — no arithmetic touches them. `alpha == 255` pixels
are copied from the replacement. Only the soft edge (1.5–2.35 % of each frame
on the real matte) is computed, with integer arithmetic:
`(alpha * rep + (255 - alpha) * src + 127) // 255`.

That guarantee is **pre-encode**, and only pre-encode: it describes the RGB24
frames handed to the encoder, not the `.mp4`. The pipeline is
`RGB24 → libx264 → yuv420p`, and the RGB→YUV 4:2:0 conversion runs *before*
x264 — it rounds through a colour-space matrix and discards three quarters of
the chroma resolution, which decoding back to RGB cannot undo. Decoded
background pixels of the output are therefore **not** guaranteed to match the
source byte-for-byte at any CRF. `crf=0` is lossless only with respect to the
yuv420p frames x264 was handed, which are already not the composited RGB.
Verify the invariant on the composited frames, not on the encoded file.

One trap worth knowing: FFmpeg's native `vp9` decoder silently drops alpha —
`ffprobe` reports `yuv420p` for `out/o1_matte.webm` despite `ALPHA_MODE=1`.
Only `-c:v libvpx-vp9` yields `yuva420p`. The compositor forces that decoder
and refuses to run if the matte decodes without an alpha channel.

### Dual-matte union (v2 POC)

`composite_video` keys every pixel off the replacement matte alone, so the
original person leaks through wherever the old silhouette extends past the new
one. A read-only overlap analysis of `out/source_matte.webm` against
`out/o1_matte.webm` (foreground = alpha ≥ 128, 225 frames) measured that
`source_only` region at a mean 0.372 % of the frame (max 0.954 %, frame 41),
`replacement_only` at 1.401 %, and foreground IoU at 97.02 %. Inside
`source_only` the source alpha averages 219/255 and the replacement alpha
17/255 — solid old-person interior, not edge noise.

`compositor.composite_video_union` composites under the pixel-wise maximum of
the two mattes:

```
effective_alpha = max(source_alpha, replacement_alpha)
output          = composite_frame(source, replacement, effective_alpha)
```

Wherever either matte sees a person, the replacement wins. The cost is
explicit: `source_only` pixels are now drawn from the O1 clip, so O1's
regenerated background can show there instead of the old person. This is a
cheap POC to judge whether that region is small enough to live with, not the
final architecture — background recovery/inpainting is the fallback if it is
not. The mattes are used exactly as decoded (no threshold, dilation, feather
or inpainting) and `composite_frame` is reused unchanged, so the `alpha == 0` /
`alpha == 255` byte-copy guarantee holds for the effective alpha. All four
streams are validated up front (one size, one frame count, clips at one fps,
both mattes decoded with alpha through `libvpx-vp9`) and streamed frame by
frame. The report's `union_lift_ratio` is the mean fraction of pixels per frame
the union actually re-routed to the replacement.

To produce the v2 composite locally:

```bash
.venv/bin/python -c "
from pathlib import Path
from video_character_skill import composite_video_union
print(composite_video_union(
    Path('out/source_aligned_24fps.mp4'),
    Path('out/o1_strict_prompt.mp4'),
    Path('out/source_matte.webm'),
    Path('out/o1_matte.webm'),
    Path('out/composite_v2_union.mp4'),
))"
```

### Source-matte hardening (v3 POC)

Visual QA of `composite_v2_union.mp4` still showed the old person's hair
ghosting through. The union keeps the *source* matte's partial alpha inside
`source_only`, and a read-only sweep found that alpha is not near-opaque: the
mean residual source contribution there is 14.0 %, spread over a broad tail
(13 % of those pixels at alpha 128–159, 13 % at 160–191, only 20 % at exactly
255). Hardening only near-opaque pixels buys little — thresholds of 208–240
leave 11–13 % residual — so the v3 rule hardens at **160**:

```
effective = max(source_alpha, replacement_alpha)
harden    = (source_alpha >= 160) & (source_alpha > replacement_alpha)
effective[harden] = 255
```

`compositor.composite_video_hardened_union` applies exactly that on top of the
v2 pipeline (same four inputs, same fail-closed validation, same `libvpx-vp9`
alpha decoding, same streaming). On the real mattes it hardens 87 % of
`source_only` (residual 14.0 % → 5.7 %), changes 0.33 % of the frame on average
(max 0.89 %), and overrides about a fifth of the replacement matte's soft
edge, nearly all of it where the source matte is ≥ 240 anyway. What survives
is the source matte's own 1–2 px anti-aliased rim — a thin outline, not a
patch. The replacement matte is never thresholded, the source matte is only
hardened where it is *more confident than the replacement*, and there is still
no dilation, feathering or inpainting; `source_only` pixels still come from
the O1 clip. The report adds `hardened_ratio`: the mean fraction of the frame
the rule actually changed to 255 (pixels already at 255 are not counted).

To produce the v3 composite locally:

```bash
.venv/bin/python -c "
from pathlib import Path
from video_character_skill import composite_video_hardened_union
print(composite_video_hardened_union(
    Path('out/source_aligned_24fps.mp4'),
    Path('out/o1_strict_prompt.mp4'),
    Path('out/source_matte.webm'),
    Path('out/o1_matte.webm'),
    Path('out/composite_v3_hardened.mp4'),
))"
```

### Source-removal mask (v4 POC)

Visual QA of `composite_v3_hardened.mp4` still showed the old person's hair
around the replacement. A read-only morphology sweep explained why: below
alpha 128 the source matte is not a blending weight worth honouring. 69 % of
those pixels sit at alpha ≤ 7 and 10–30 px from the silhouette (VP9 alpha
compression haze), while the hair you can see (alpha ≥ 32) lies within 4 px
(p90) of the `alpha >= 64` core — and blending with *any* partial source alpha
keeps most of a strand, because the source pixel *is* the strand.

v4 stops treating the source matte as an alpha. It is a binary support — "the
old person must not survive here" — from threshold **64** and a Euclidean-disk
dilation of **4 px**:

```
source_core = source_alpha >= 64
removal     = dilate_disk(source_core, 4)      # offsets with dy² + dx² <= 16
effective   = replacement_alpha.copy()
effective[removal] = 255
output      = composite_frame(source, replacement, effective)
```

Outside the mask the replacement matte behaves exactly as in v1. Inside it the
source clip contributes nothing: where the replacement person is, the O1
person is copied; where only the old person was, O1's background is copied
instead of the old person leaking through. Measured on the real mattes the
mask covers ~96 % of visible source hair, costs 0.208 % of the frame per frame
of true background (the dilation band; 0.694 % removal in total) and shows no
flicker signature; radius 2 left too much hair, radius 6 replaced markedly
more background for little gain. No `max()`, no partial source alpha, no
feathering, temporal smoothing or inpainting. The dilation is a deterministic
NumPy shifted-OR over the integer disk offsets (no scipy/OpenCV), clipped at
the image border. The report carries `removal_ratio`, `dilation_only_ratio`
(added by dilation beyond the core) and `replacement_override_ratio` (where the
mask forced a replacement alpha below 255 to 255).

To produce the v4 composite locally:

```bash
.venv/bin/python -c "
from pathlib import Path
from video_character_skill import composite_video_source_removal
print(composite_video_source_removal(
    Path('out/source_aligned_24fps.mp4'),
    Path('out/o1_strict_prompt.mp4'),
    Path('out/source_matte.webm'),
    Path('out/o1_matte.webm'),
    Path('out/composite_v4_removal.mp4'),
))"
```

### Temporal real-background recovery (v5 POC)

Visual QA of `composite_v4_removal.mp4` showed a background-coloured outline
around the subject: v4 copies O1 pixels across the whole removal mask, and
where the replacement person is absent those are O1's *regenerated*
background. A read-only feasibility analysis found the camera is static (every
6-frame background pair aligns at (0, 0) ± 1 px; whole-clip pairs agree within
2–3 luma levels once a brightness offset is removed), that within ±24 frames
81.5 % of the recovery region is real background at the same coordinates in
some other frame (92.9 % over the whole clip; nearest observation a median
2–3 frames away), that 19.8 % of it is already real background in its own
frame, and that frames ~183–222 carry a white-balance/exposure shift rather
than motion.

`temporal_recovery.composite_video_temporal_recovery` therefore rebuilds the
background plate from real source pixels before compositing (new module
`temporal_recovery.py`; v1–v4 untouched):

```
removal           = source_removal_mask(source_alpha_i, 64, 4)          # v4
recovery_region   = removal & (replacement_alpha_i < 128)
own_background    = recovery_region & (source_alpha_i < 32)              # keep the source pixel
needs_temporal    = recovery_region & ~own_background
force_replacement = removal & (replacement_alpha_i >= 128)               # v4 protection

for j in i-1, i+1, i-2, i+2, …, i-24, i+24:                             # nearest first, clipped
    usable_j = source_alpha_j < 32
    offset   = clamp(median over shared background of source_i - source_j, ±64)   # per channel
    each needs_temporal pixel with usable_j takes clip(source_j + offset), until it holds 5

background = source_i;  background[recovered]   = per-channel median of observations
                        background[unrecovered] = replacement_i          # O1 fallback, never the old person
effective_alpha = replacement_alpha_i;  effective_alpha[force_replacement] = 255
output = composite_frame(background, replacement_i, effective_alpha)
```

Precedence inside the recovery region: own-frame real background, then
temporally borrowed real background, then O1 background. Outside the removal
mask nothing differs from plain replacement-matte compositing. No
registration, optical flow, gain fitting, spatial inpainting or feathering.
The source clip and matte are read 24 frames ahead and held in a window of at
most 49 frames (~8.3 MB per 1080x1920 frame, ≈ 410 MB peak). A donor whose
shared background is too small for a trustworthy offset (< 256 samples on a
stride-8 grid) is used with zero offset, and the report counts those fits.
The report also carries the recovery-region, own-background, recovered and
unrecovered ratios (of the full frame), the O1-fallback share of the recovery
region, median and p90 donor distance, mean observations per recovered pixel,
and the peak cache size.

To produce the v5 composite locally:

```bash
.venv/bin/python -c "
from pathlib import Path
from video_character_skill import composite_video_temporal_recovery
print(composite_video_temporal_recovery(
    Path('out/source_aligned_24fps.mp4'),
    Path('out/o1_strict_prompt.mp4'),
    Path('out/source_matte.webm'),
    Path('out/o1_matte.webm'),
    Path('out/composite_v5_recovery.mp4'),
))"
```

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
  compositor.py              # alpha-composite the replacement onto the original;
                             # v1 single matte, v2 union, v3 hardened union,
                             # v4 source-removal mask (binary support + disk dilation)
  temporal_recovery.py       # v5: real background borrowed from nearby source frames
                             # inside the v4 mask, with additive photometric correction
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
