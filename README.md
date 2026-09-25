<sub>Open source from <a href="https://noisefactor.io">Noise Factor</a> · <a href="https://github.com/noisefactorllc">more projects</a></sub>

# High-end presentation

An agent skill that presents an artwork in a synthetic scene. You give it a still image or a video of your work and a short prompt. It returns one still image of the work in a gallery, a home, a storefront window, a public space, or a glass case.

The still is a long exposure. People who walk past appear as translucent figures. People who stop to look become more solid. Bright moving lights leave trails. Depth of field keeps the work in focus, and a subtle grade from the work's own palette ties the scene together.

The skill works in Claude Code, Codex, and Hermes. It uses image, video, and depth models on [fal](https://fal.ai).

## The rule

The skill never changes the features of your work. Shapes, marks, composition, detail, and proportions stay yours. The scene can light the work, reflect in glass in front of it, and send people past it. No model redraws it.

The skill enforces this rule in three steps:

1. An image model makes a reference scene with your work in place. The skill finds the work in that scene and checks its composition, colors, and proportions. If a check fails, the skill makes a new reference.
2. The skill puts your original pixels back into the scene, under the scene's light. Image models redraw fine texture, so the skill does not keep their version.
3. Before it writes the result, the skill compares each work in the final image with the original. It excludes the areas where a person stands in front of the work. If the match is too low, the skill writes no image.

## What you can make

- One piece on a wall, in a frame, or behind glass.
- Several pieces in one scene, placed left to right in the order you give them.
- One piece on many screens, such as a wall of televisions. The skill finds every screen and puts the original on each one.
- Screens as picture tubes. The skill renders the curvature, scanlines, and glow itself, from your original pixels.

A video piece shows one frame. The skill selects a frame that is sharp and typical of the clip, or you can name one.

## Requirements

- Python 3.12 or later.
- A fal API key in the `FAL_KEY` environment variable.
- An agent harness that can read a skill file and run shell commands.

The first run creates a Python environment in `~/.cache/high-end-presentation/`. It installs NumPy, OpenCV (headless), and Pillow from pinned versions.

## Install

### Claude Code

```
/plugin marketplace add noisefactorllc/high-end-presentation
/plugin install high-end-presentation@noisefactor
```

### Codex and Hermes

Clone the repository, then copy the skill folder into the harness's skills directory:

```bash
git clone https://github.com/noisefactorllc/high-end-presentation.git
cp -R high-end-presentation/skills/high-end-presentation ~/.codex/skills/
cp -R high-end-presentation/skills/high-end-presentation ~/.hermes/skills/creative/
```

## Use

Ask your agent for a presentation, and give it the file and the setting:

> Present `bloom.mp4` in a quiet gallery, calm energy.

> Show these three clips on screens in a busy airport concourse at night. High energy, 16:9.

The agent reads `skills/high-end-presentation/SKILL.md` and runs the stages. It looks at your work, writes the scene brief, and checks each result before the next paid step.

### Options

| Option | Values | Default |
|--------|--------|---------|
| Energy | `quiet`, `calm`, `lively`, `bustling` | `calm` |
| Aspect ratio | `1:1`, `2:3`, `3:2`, `3:4`, `4:3`, `4:5`, `5:4`, `9:16`, `16:9`, `21:9` | `4:5` |
| Resolution | `1K`, `2K`, `4K` | `2K` |
| Repeat on screens | 2 to 24 | off |
| Screen treatment | `flat`, `crt` | `flat` |
| Motion echo (video pieces) | on or off | off |
| Frame (video pieces) | a frame number | automatic |

Energy controls how many people pass, how many exposures the skill layers, and how strong the light trails are. The skill adapts the layer count to how crowded the clip is. A dense crowd therefore stays a set of distinct figures and does not become a haze.

### Scenes

`references/scenes.md` holds a template for each scene type:

| Scene | Setting |
|-------|---------|
| `gallery` | A white wall, a float frame, and a spotlight |
| `home` | A living-room wall above a sideboard |
| `storefront` | A gallery window at night, seen from the sidewalk |
| `modern-frame` | A shadow-box frame under raking light |
| `glass-display` | A glowing piece in a block of optical glass |
| `screens` | Bright screens in a busy public space |
| `freeform` | Your own description |

## How it works

1. **Analyze.** The skill measures the palette and tone of each piece. For a video, it selects a frame. The agent looks at the work and writes a brief: themes, mood, scene, and motion.
2. **Reference still.** An image model (GPT Image 2.5 Flare by default) makes an empty scene with the work in place. The skill forbids text in the scene. Image models add lettering to signs and boards when a scene invites it.
3. **Gate and repair.** The skill finds each piece and checks it. Then it puts the original pixels back under the scene's light.
4. **Draft.** An optional preview with lens and grade, but no people. It costs one image call.
5. **Video.** A video model (Kling v3 Pro) animates the scene with a locked camera. The clip starts and ends on the empty scene. The skill uses those end frames as the clean background.
6. **Develop.** The skill stacks the frames into a long exposure. It uses streaks for motion, layered exposures for figures, and the brightest frames for light trails.
7. **Finish.** The skill adds depth of field from a depth model, a vignette, a tone map, a split tone from the piece's palette, and grain. Then it checks each piece and writes the image.

Each stage writes its state to a job folder. If a fal request times out, you run the same stage again. The skill continues the same request and does not pay twice.

## Output

Each job writes to its own folder:

- `out/NAME-presentation.png`: the result, 16-bit.
- `out/NAME-presentation.jpg`: a copy for viewing. Some viewers show false banding in 16-bit PNG files.
- `out/manifest.json`: the inputs, the brief, the models, the request IDs, the check scores, and any warnings.

## Cost and time

These numbers come from test runs. fal sets the prices, and they can change.

- Reference still: about 40 seconds per attempt.
- Video: about 4 minutes and about USD 1.40 for a 10-second Kling v3 Pro clip.
- Develop and finish: about 2 minutes on a laptop.

## Limits

- For a steep side view, the proportion check assumes a normal lens. A straight or gentle view gives the most accurate check.
- A person who stays completely inside a frame's outline looks like a distortion of the work to the skill. The skill removes that person.
- The scene resolution comes from the image model. The video adds the people and light, so it does not set the sharpness.

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -r tests/requirements.txt
.venv/bin/python -m pytest tests
```

The tests use synthetic images and a local fake of the fal API. They make no network calls.

## License

MIT
