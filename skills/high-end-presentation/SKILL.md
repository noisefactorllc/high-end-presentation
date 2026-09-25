---
name: high-end-presentation
description: Use when an artist wants a still image or video of their artwork presented in a synthetic scene (a gallery, a home wall, a storefront window at night, a modern frame, a glass display) as one high-end long-exposure still, with the life around the piece shown as light trails and ghosted passers-by, shallow depth of field, and a palette-driven grade. Triggers include "present my piece", "show this artwork on a wall", "gallery mockup", "in-situ render", "time-lapse presentation". Needs FAL_KEY.
---

# High-end presentation

Turn one artwork (a still image or a video) plus a short prompt into one still image. The still shows the piece on display in a synthetic scene, photographed as a multiple long exposure: people appear several times along their paths as translucent figures, people who stop to look build up towards solid, and bright moving lights leave trails. People may cross in front of the piece, as they would in life. Depth of field keeps the piece as the only sharp plane. A subtle grade drawn from the piece's own palette ties the scene together.

## The rule you must not break

The piece's features never change. Shapes, marks, composition, detail, and proportions are the artist's. The scene may light the piece (color of light, falloff, glare, reflections in glass, passing shadows), and people may pass in front of it; nothing may redraw it. The scripts enforce this:

- The reference still is gated. The piece must be found, with its proportions, composition, and colours intact. Fine texture is not gated: image models redraw it, and the repair puts the original pixels back. A failed gate means regenerate, never "use it anyway".
- The final image carries the artist's own pixels in the piece region, relit by the scene. Inside the frame, only figures that enter from outside it are kept; changes confined to the frame are the video model distorting the artwork and are dropped. `finish` checks the uncovered part of the piece and refuses to write a result that does not verify.

Do not work around a failed gate by loosening thresholds or editing the piece.

## Inputs

Ask only for what is missing:

| Input | Required | Default | Notes |
|-------|----------|---------|-------|
| piece | yes | | still (png, jpg, tiff, webp) or video (mp4, mov, webm, gif); up to 6 pieces, placed left to right in the order given |
| prompt | yes | | basic guidance: setting, mood, time of day |
| energy | no | calm | quiet, calm, lively, bustling |
| aspect | no | 4:5 | 1:1 2:3 3:2 3:4 4:3 4:5 5:4 9:16 16:9 21:9 |
| resolution | no | 2K | 1K, 2K, 4K (long side of the output) |
| echo | no | off | moving pieces only: a faint echo of the piece's motion |
| frame | no | automatic | moving pieces only: the frame index to present |

Energy sets the life in the scene:

| Energy | People | Exposures layered | Figure opacity each | Light trails |
|--------|--------|-------------------|---------------------|--------------|
| quiet | one or two, one lingers | 5 | 0.65 | soft |
| calm | a few, some stop | 7 | 0.55 | gentle |
| lively | steady flow | 10 | 0.45 | clear |
| bustling | crowd | 14 | 0.38 | dense |

Under the figures, a streak layer blurs each path; its density is solved per clip, so a busy clip does not turn into a veil.

**Glazing.** For `storefront`, `glass-display`, and `screens` the piece is behind glass: the reference still's reflections stay over the piece, and moving light (headlights, lit passers-by) reflects across it. Set `"glazed": true` in the brief for any other scene with glass in front of the piece (a framed print under glass), or `false` to turn it off.

## Setup (once per machine)

```bash
PY="$(<skill-dir>/scripts/bootstrap.sh)"      # creates the venv in ~/.cache, prints its python
R="<skill-dir>/scripts/hep.py"
export FAL_KEY=...                             # the operator's fal key; never print or log it
```

`<skill-dir>` is the directory that contains this file. Python 3.12 or newer is required (set `HEP_PYTHON` to choose an interpreter).

## Procedure

Every stage prints one JSON object. Exit codes: 0 ok, 1 error, 2 fidelity gate failed, 3 fal request still pending (rerun the same command; it resumes the same request and does not pay twice). Use one job directory per presentation, outside any source repository.

1. **Init.**
   `$PY $R init --job JOB --piece PIECE [--piece PIECE ...] --prompt "PROMPT" --scene SCENE --energy ENERGY [--aspect 4:5] [--resolution 2K] [--echo] [--frame N] [--name NAME]`
   Choose `SCENE` from `references/scenes.md` (gallery, home, storefront, modern-frame, glass-display, screens, freeform). Repeat `--piece` to show several pieces in one scene; they are placed left to right in the order given, and every rule below applies to each of them.

2. **Analyze.** `$PY $R analyze --job JOB`
   The output holds, per piece, the palette (colors with weights), tone (key, contrast, warmth, saturation), and for a video piece the chosen frame, plus a merged palette for the grade. Then **look at each `JOB/piece-N-preview.jpg` yourself.** You are the art director: name the themes and the mood.

3. **Brief.** Write a JSON file and run `$PY $R brief --job JOB --brief-file BRIEF.json`.
   Required keys: `themes` (list), `mood` (string), `scene` (the scene paragraph), `motion` (the activity paragraph).
   Optional keys: `glazed` (glass in front of the piece; see Glazing), `focus` (depth-of-field strength, default 1.0; 0 disables), `vignette` (0.22), `tone` (split-tone strength, 0.15), `contrast` (0.08), `grain` (0.012), `piece_grade` (share of the grade applied to the piece, 0.5).
   Build `scene` and `motion` from the archetype in `references/scenes.md`. Let the analysis drive the choices: light that flatters the palette (a warm spot on a cool, dark piece; soft daylight on a pale one), materials that echo the themes, and a time of day that suits the mood. Follow the rules at the top of `scenes.md`. The runner adds the fidelity and locked-camera clauses itself.

4. **Reference still.** `$PY $R still --job JOB` (about 30 s and USD 0.15 per attempt; up to 3 attempts).
   With several pieces, every piece must pass, each in its own place (`pieces-overlap` means two pieces matched the same spot). On exit 2, read each attempt's `reason` (prefixed `piece N:` when several pieces are given):
   - `not-found:*` or `area`: the piece is too small or hidden. Make it larger in the scene text, rewrite the brief, rerun.
   - `aspect`: the model changed the piece's proportions. Rerun; if it repeats, use a straight-on view.
   - `composition` or `local-composition`: the model changed what the piece shows, or part of it. Rerun; simpler scenes help.
   - `color`: the region found has the wrong colours, so it is probably not this piece (or the scene light is far too strong). Rerun, or soften coloured light in the scene text.
   Rerunning `still` with an unchanged brief first re-checks the earlier attempts, then renders new ones.
   Then **look at `JOB/still/plate-preview.jpg`** (the repaired plate). Check composition, light, and that the frame and wall read naturally. If the scene is wrong, change the brief and rerun `still`.

5. **Draft (optional, recommended).** `$PY $R draft --job JOB` writes `JOB/out/NAME-draft.jpg` with lens and grade but no activity. Show it to the operator before the paid video when the operator wants to approve the scene first.

6. **Video.** `$PY $R video --job JOB [--timeout 900]` (Kling v3 Pro, 10 s clip: about 4 minutes and about USD 1.40). Exit 3 means it is still rendering; rerun the same command. Run it in the background when your harness allows.

7. **Develop.** `$PY $R develop --job JOB` (about 70 s). Stacks the clip into the long exposure and writes `JOB/develop/exposure-preview.jpg`. Read the report:
   - `piece_covered`: share of the piece that figures cover. Above about 0.5 the piece reads poorly; rerun `video` or lower the energy.
   - `camera-moved`: the video model moved the camera; registration compensated. Look for doubled edges.
   - `background:median`: the clip's ends did not match the plate; viewers who stood still may vanish.
   - `no-motion`: nothing happened in the clip. Rerun `video` with a more active `motion` text.

8. **Finish.** `$PY $R finish --job JOB` writes `JOB/out/NAME-presentation.png` (16-bit), a JPEG copy, and `manifest.json`. Inspect the JPEG: some image viewers render 16-bit PNGs with false banding. `final_verify` (one score per piece) must be at least 0.85 for every piece or nothing is written. **Look at the result** before you hand it over, and report the path, the energy, the warnings, and the cost of the calls you made.

`$PY $R status --job JOB` shows completed stages, warnings, and pending requests.

## Changing course

- New scene or light: edit the brief, rerun `brief`, `still`, then continue. Each `still` run adds attempts; it never reuses a failed one.
- Same scene, new activity: rerun `video`, `develop`, `finish`.
- Different look only (focus, grade): edit the look keys in the brief, rerun `brief`, then `finish`. No model calls.

## Models

Defaults (all on fal): image `fal-ai/nano-banana-pro/edit`, video `fal-ai/kling-video/v3/pro/image-to-video`, depth `fal-ai/image-preprocessors/depth-anything/v2`. `--image-endpoint openai/gpt-image-2.5/flare/edit` (on `init`, or on `still` to switch an existing job) selects GPT Image 2.5 Flare, which kept several pieces' colours and composition closer in testing. `--video-endpoint bytedance/seedance-2.5/image-to-video` selects Seedance 2.5 (1080p, billed per token, higher cost). The piece and the scene images are sent to fal; nothing else leaves the machine.

## Known limits

- For a steep side view, where one pair of the frame's edges stays parallel in the image, the proportion check depends on a normal-lens assumption. Keep views straight on or gently angled.
- A person who stands entirely inside the frame's outline (never overlapping the wall around it) is treated as distortion and dropped.
- Scene resolution is the image model's (2K by default). The video only contributes the activity layer, so it does not limit sharpness.
