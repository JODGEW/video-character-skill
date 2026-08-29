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

### Spatial real-background fill (v6 POC)

Read-only attribution of `composite_v5_recovery.mp4` showed that temporal
recovery works — recovered pixels sit against the surrounding real background
as well as real background does (seam p90 11.8 vs 10.6 for the control) — and
that the residual light patches are the pixels v5 could not recover at all:
`temporal_unrecovered`, 0.12 % of the frame (~18 % of the recovery region),
which fell back to O1's regenerated background (seam p90 103.8, systematically
lighter than the room). A wider temporal search cannot fix them: ±48 frames
recovers another 24 %, the whole clip 53 %, and 47 % are never real
background at that coordinate because the old person's silhouette never moves
off them.

`spatial_recovery.composite_video_spatial_recovery` keeps v5 unchanged through
temporal recovery and then fills only those pixels from real background next
to them (new module `spatial_recovery.py`; v1–v5 untouched):

```
plate   = source_i with v5's recovered pixels written in                # no O1 yet
trusted = (source_alpha_i < 32) | temporal_recovered                    # real background only
target  = temporal_unrecovered                                          # the only pixels touched

for each 8-connected component of target, inside its bbox + 1 px:
    resolved = trusted pixels 8-adjacent to the component                # seeds
    repeat (synchronous waves):
        every unresolved component pixel with >= 1 resolved 8-neighbour
            takes, per channel, the mean of those neighbours, rounded half up:
            (2 * sum + count) // (2 * count)
        all of them become resolved together
    until no unresolved pixel has a resolved neighbour

background = plate;  background[filled]   = wave values
                     background[residual] = replacement_i               # O1 only here
effective_alpha = replacement_alpha_i;  effective_alpha[force_replacement] = 255
output = composite_frame(background, replacement_i, effective_alpha)
```

Every wave reads only the resolved state left by the previous wave, so the
result does not depend on traversal order. Replacement-person, old-person and
O1 pixels are never resolved, so nothing propagates through or from them; a
component with no trusted neighbour stays unfilled and keeps the O1 fallback.
Precedence inside the recovery region is now: own-frame real background, ±24
temporal real background, spatially propagated real background, O1. The old
person is never a fallback. No optical flow, registration, gain fitting,
feathering, external inpainting or new dependency; the cache, donor search
and photometric correction are v5's.

The report carries v5's diagnostics plus the spatially recovered and
unrecovered ratios (of the full frame), the O1-fallback share of the recovery
region, the component count, the components without a seed, and the median,
p90 and maximum propagation depth in waves.

To produce the v6 composite locally:

```bash
.venv/bin/python -c "
from pathlib import Path
from video_character_skill import composite_video_spatial_recovery
print(composite_video_spatial_recovery(
    Path('out/source_aligned_24fps.mp4'),
    Path('out/o1_strict_prompt.mp4'),
    Path('out/source_matte.webm'),
    Path('out/o1_matte.webm'),
    Path('out/composite_v6_spatial.mp4'),
))"
```

### Hard-inset replacement foreground (v8 diagnostic POC)

With source removal at 32/4 (v7) the old person no longer leaks, yet a thin
hair-like contour remained around the head. Read-only attribution showed it is
not the soft replacement alpha (its final seam equals the real-background
control) and not our alpha forcing (no forcing/zeroing counterfactual removes
more than ~6 % of the edge-light pixels): Kling O1 rendered the new person over
its *own* regenerated background, so O1's anti-aliased edge RGB — the pixels
VEED gives alpha 128–254 — already carries O1 background at any opacity.

`hard_inset_recovery.composite_video_hard_inset_recovery` is a deliberately
diagnostic composite that discards that edge entirely (new module
`hard_inset_recovery.py`; v1–v7 untouched):

```
removal          = source_removal_mask(source_alpha_i, 32, 4)            # v7 parameters
replacement_core = replacement_alpha_i >= 250
hard_foreground  = erode_disk(replacement_core, 2)                        # outside the frame = background

recovery_region  = removal & ~hard_foreground     # every old-person pixel the core will not cover
own_background   = recovery_region & (source_alpha_i < 32)
needs_temporal   = recovery_region & ~own_background
… v5 temporal recovery, v6 spatial fill, O1 only for the unreachable residual …

effective_alpha  = 255 where hard_foreground, 0 everywhere else           # binary
output           = composite_frame(reconstructed_background, replacement_i, effective_alpha)
```

`erode_disk` uses the same integer Euclidean disk as `dilate_disk`
(`dy² + dx² ≤ r²`); it is `NOT dilate_disk(NOT mask)` on a copy padded with
background, so foreground touching the image border erodes normally and
nothing wraps around. The recovery region is no longer
`removal & (replacement_alpha < 128)`: the old person under the dropped alpha
128–249 ring inside the removal mask is rebuilt from real background, so
dropping the ring never exposes it. There is no `force_replacement`, no soft
alpha and no feathering; the silhouette is harder and inset by 2 px on
purpose. If the contour disappears, edge-RGB contamination is proven.

The report carries the hard-foreground and dropped-alpha ratios, the v5/v6
recovery and propagation diagnostics, and — to prove the old person was not
exposed — the share of the dropped ring inside the removal mask resolved by
own-frame, temporal, spatial and O1 background.

To produce the v8 composite locally:

```bash
.venv/bin/python -c "
from pathlib import Path
from video_character_skill import composite_video_hard_inset_recovery
print(composite_video_hard_inset_recovery(
    Path('out/source_aligned_24fps.mp4'),
    Path('out/o1_strict_prompt.mp4'),
    Path('out/source_matte.webm'),
    Path('out/o1_matte.webm'),
    Path('out/composite_v8_hard_inset.mp4'),
))"
```

### Local rim-tone correction (v9 POC)

v8 (a hard, 2-px-inset binary foreground) removed no contour and added ~20x
more static-pixel edge flicker, so v9 returns to the v7 compositing path
unchanged (soft replacement alpha, `force_replacement` at 128, source removal
32/4, temporal + spatial recovery) and changes only the *replacement RGB*
before it is composited. Attribution had excluded every compositing-side
cause; what remains is a rim rendered into O1's own RGB inside
`replacement_alpha >= 250` (head region: +15 luma at the outermost ring,
decaying over ~6 px). Because the silhouette is three very different
materials in equal thirds, a global per-ring offset over-corrects the dark
ones and misses the bright ones; the read-only feasibility study picked a
local box-window offset at half strength (`rim_correction.py`; v1–v8
untouched, the v6 streaming loop gained an optional `replacement_filter`
hook that defaults to `None`):

```
core      = replacement_alpha >= 250
band_k    = erode_disk(core, lo) & ~erode_disk(core, hi)      (0,1) (1,2) (2,3) (3,4) (4,6)
reference = erode_disk(core, 8) & ~erode_disk(core, 12)
per band, per channel, at each band pixel (32x32 box, clipped at the frame; integral images):
    offset = mean(reference rgb in box) - mean(band rgb in box)
    valid  = >= 8 band px and >= 16 reference px in the box
    rgb    = clip(rint(rgb + 0.5 * offset), 0, 255)          only where band & valid
```

Nothing else changes: alpha, masks and recovery are v7's. The report adds
the rim's target/corrected share of the frame, the valid share, the mean and
maximum applied offset and the clipping share.

```bash
.venv/bin/python -c "
from pathlib import Path
from video_character_skill import composite_video_rim_corrected
print(composite_video_rim_corrected(
    Path('out/source_aligned_24fps.mp4'),
    Path('out/o1_strict_prompt.mp4'),
    Path('out/source_matte.webm'),
    Path('out/o1_matte.webm'),
    Path('out/composite_v9_local32_s05.mp4'),
))"
```

### Two-tier residual recovery (v11 POC)

v10 (the v9 code run with `background_threshold=1`: only `source_alpha == 0`
is own-frame background, a temporal donor or a first-pass spatial seed)
removed the old person's hair/skin ghosts, but the enclosed regions the
replacement matte carves out — the hoop earring's interior in frames
~206–220 — then had no trusted donor or seed at all, fell back to O1's light
regenerated background and flashed. A read-only measurement showed that every
such component holds, or borders, source pixels with alpha 1–31 whose colour
matches the surroundings. v11 keeps tier 1 exactly as v10 and adds an
optional lower-confidence pass over the tier-1 residual only, before the O1
fallback (`spatial_recovery.recover_residual`, threaded through the v6 loop
and the v9 orchestrator as `residual_threshold`; `None`, the default, keeps
every earlier version byte for byte):

```
residual_1 = temporal_unrecovered & ~filled          # tier 1 as before: whole seedless components
low        = background_threshold <= source_alpha_i < residual_threshold      (v11: 1 <= a < 32)
S_inside   = residual_1 & low       # own-frame soft pixels: their source RGB is copied in
T          = residual_1 & ~low      # old-person pixels: filled by the same synchronous waves
seeds      = trusted | filled | low # tier-1 resolved, S_inside, the low ring outside residual_1
residual_2 = T & ~filled_2          # only this falls back to O1
```

Temporal donor eligibility is not relaxed; tier-1 own, temporal and spatial
pixels are never rewritten; nothing outside `residual_1` changes; a residual
component no seed touches stays O1; a threshold not above
`background_threshold` is an exact no-op. `spatial_unrecovered_ratio` and the
O1-fallback share now describe `residual_2`, and the report adds the tier-1
residual, the accepted `S_inside`, the tier-2 filled and unresolved shares,
the residual component count, the components without a tier-2 seed and the
tier-2 propagation depth p50/p90/max.

```bash
.venv/bin/python -c "
from pathlib import Path
from video_character_skill import composite_video_rim_corrected
print(composite_video_rim_corrected(
    Path('out/source_aligned_24fps.mp4'),
    Path('out/o1_strict_prompt.mp4'),
    Path('out/source_matte.webm'),
    Path('out/o1_matte.webm'),
    Path('out/composite_v11_tier2.mp4'),
    background_threshold=1,
    residual_threshold=32,
))"
```

**Known limitation.** Source-matte pixels with alpha 1–31 that lie outside
the 32/4 removal mask are composited as they are, so low-alpha source content
the matte rates as background can survive — the see-through interior of the
source person's glasses lens shows as a bright sliver beside the replacement
cheek in frames ~34–49. Closing the source core globally before the dilation
(a padded radius-12 closing) removed it but was rejected: its additions along
the whole silhouette caused foreground/background edge adhesion at the
replacement's soft edge, and neither a replacement-alpha gate nor a
geometry-only pocket rule cleared that without side effects. v11 remains the
accepted POC base.

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
  spatial_recovery.py        # v6: v5's unrecovered pixels filled by synchronous-wave
                             # propagation from trusted real background, O1 last;
                             # v11: optional tier-2 pass over the residual (residual_threshold)
  hard_inset_recovery.py     # v8: eroded opaque replacement core only, dropped edge
                             # rebuilt as real background (diagnostic)
  rim_correction.py          # v9: local box-window rim-tone correction of O1's edge RGB,
                             # applied through the v6/v7 replacement_filter hook
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
