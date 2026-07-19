# Photo2Relief

Turn a single 2D photograph into a CNC-machinable 2.5D relief mesh. Locally hosted, Docker-based web app: upload a photo, tune the relief interactively with live 2D/3D previews, export a watertight binary STL (or OBJ) in millimeters, and import it straight into Autodesk Fusion for CAM.

Under the hood: monocular depth estimation (Depth Anything family) → heightmap shaping → grid-triangulated solid mesh. No cloud, no accounts, no paid licenses. See `SPEC.md` for the full design; `CLAUDE.md` for build conventions and status.

> **Status:** Under construction, built milestone-by-milestone per `SPEC.md`. Sections below describe target behavior; anything not yet true should be flagged in `CLAUDE.md`.

## Quick start

Prerequisites: Docker + Docker Compose. For GPU mode (recommended): an NVIDIA GPU and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
# GPU (primary mode — target machine: RTX 3080)
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build

# CPU fallback (slower inference; auto-selects the lightweight model)
docker compose up --build
```

Open **http://localhost:8090**. First run downloads the default depth model into `./data/models/` (one-time, ~1–2 GB); after that the app works fully offline. Port and other settings are env-configurable — see `docker-compose.yml`.

## Using the app

1. **Upload** a JPEG/PNG/WebP/HEIC (up to ~30 MP). Depth estimation runs once and is cached — on the GPU this takes a few seconds.
2. **Tune.** Parameters are grouped as Size / Depth shaping / Surface / Export. The 2D hillshade preview updates instantly; the interactive 3D preview refreshes a moment later. Nothing you do here re-runs the AI model, so twist the knobs freely.
3. **Export.** Generates the full-resolution watertight mesh and downloads it. The filename embeds the physical dimensions in mm.

### Parameter cheat sheet

- **Size:** model width (mm), relief height, base slab thickness, optional flat border frame for clamping.
- **Depth shaping:** depth model selection (see below), invert (intaglio/mold mode), gamma (push the subject forward / flatten the background), percentile clipping, flatten-background threshold.
- **Surface:** smoothing amount, edge-preserving (bilateral) mode, luminance detail blend (adds lithophane-style fine texture — hair, fabric — that depth models smooth over), edge taper.
- **Export:** grid resolution (512/1024/2048 long side), optional decimation, STL or OBJ.

### Choosing a depth model

| Model | Character | Speed | License* |
|---|---|---|---|
| **DA3MONO-Large** (default) | Predicts true depth → most geometrically faithful relief. Best for objects, scenes, pets. | Fast on GPU | Verify HF page (CC-BY-NC family expected) |
| **Depth Anything V2 Large** | Disparity-style output exaggerates the near field and compresses the background — often the more *artistic* bas-relief look for portraits. | Fast on GPU | CC-BY-NC 4.0 |
| **Depth Anything V2 Small** | Lightweight fallback; the only fully Apache-2.0 option. Auto-selected when no GPU is present. | Fast even on CPU | Apache-2.0 |

*This is a personal, non-commercial tool. The CC-BY-NC-licensed weights are fine for that use; don't redistribute this app with those weights bundled or use outputs commercially without checking the model licenses. The build records each model's license as listed on its Hugging Face page.

Tip: run the same photo through DA3MONO and DA2-Large and compare the hillshades — the "right" one is an aesthetic call and differs by subject.

## Importing into Autodesk Fusion

1. **Insert → Insert Mesh**, select the exported `.stl`.
2. **Set units to millimeters** — STL is unitless and this is the classic gotcha. The correct physical size is embedded in the filename (e.g., `photo2relief_ab12_150x100mm.stl`); sanity-check the bounding box after import.
3. Flip/orient as needed; the mesh is a closed solid (relief + base slab), so Fusion's **Manufacture** workspace can toolpath it directly as a mesh body — no BRep conversion required.
4. Typical CAM recipe for wood: 3D Adaptive Clearing with a 1/4" end mill leaving ~0.5 mm stock, then Parallel finishing with a ball nose (1/8" or smaller for fine portraits, ~8–10% stepover). The base slab gives you material to hold in the vise or screw to a spoilboard; add a border frame in the app if you want dedicated clamp real estate.
5. Undercuts are impossible by construction (single-viewpoint relief), so everything is reachable by a 3-axis machine.

## Development (outside Docker)

```bash
uv sync                                  # env + deps (Python 3.12)
uv run uvicorn app.main:app --reload --port 8090
uv run pytest                            # unit tests (fast; model smoke tests are @slow)
uv run pytest -m slow                    # includes one real-inference smoke test
uv run ruff check . && uv run ruff format --check .
```

Layout: `app/` (FastAPI backend — `depth.py` model adapters, `heightmap.py` pure transforms, `meshing.py` mesh build/export), `static/` (vanilla JS + vendored three.js frontend, no build step), `tests/`, `data/` (gitignored: sessions + model cache). Full API contract and architecture: `SPEC.md`.

## Troubleshooting

- **GPU compose fails / falls back to CPU:** confirm `nvidia-smi` works on the host and the NVIDIA Container Toolkit is installed; check `/api/health` — it reports the active device and model.
- **Port conflict:** change the published port in `docker-compose.yml` (container listens per `P2R_PORT`, default 8090).
- **First model download fails:** the container needs internet for the initial Hugging Face fetch only; retry, or pre-populate `./data/models/`.
- **Fusion imports a tiny/huge model:** you imported with the wrong units — re-import as millimeters.

## Licenses

Application code: MIT (or your preference — set before publishing). Third-party: FastAPI/three.js/trimesh et al. are MIT/BSD/Apache. Model weights carry their own licenses per the table above and are downloaded at runtime, not distributed with this repo.
