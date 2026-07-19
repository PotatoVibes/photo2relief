# Photo2Relief — Implementation Plan (SPEC.md)

**Purpose:** Locally-hosted Dockerized web app that converts a single 2D photograph into a 2.5D relief mesh (STL/OBJ) suitable for import into Autodesk Fusion and CNC machining on a 3-axis mill/router.

**Audience:** This document is the execution spec for Claude (Sonnet) working in Claude Code. Build milestone by milestone. Each milestone has acceptance criteria; do not proceed to the next milestone until the current one passes.

**Licensing constraint:** All tools and libraries must be free for personal use with no license purchase. Preferred: MIT / Apache-2.0 / BSD. Depth models: this is a personal, non-commercial tool, so **CC-BY-NC-licensed model weights are acceptable and expected** (owner decision, 2026-07-18). The app ships a **model registry** (§5.1.1) with three backends; default is `da3mono-large`. Record each model's license (as shown on its Hugging Face page at build time) in the README.

---

## 1. Product Overview

### 1.1 User workflow

1. User opens `http://localhost:8090` in a browser.
2. Uploads a photo (JPEG/PNG/WebP/HEIC; up to ~30 MP).
3. App runs monocular depth estimation once and caches the raw depth map for the session.
4. User adjusts tuning parameters (relief height, size, smoothing, detail blend, etc.).
5. App shows two previews that update on parameter change:
   - **Heightmap preview** (2D grayscale + hillshade) — near-instant.
   - **3D mesh preview** (interactive three.js viewport, orbit/zoom) — regenerated at reduced resolution, target < 2 s.
6. When satisfied, user clicks **Export** → app generates the full-resolution watertight mesh and serves a download: **binary STL** (default) or **OBJ**.
7. User imports the STL into Fusion (Insert → Insert Mesh) and toolpaths it in the Manufacture workspace.

### 1.2 Non-goals

- No true 3D (back-side) generation, no photogrammetry, no multi-image input.
- No user accounts, no auth, no HTTPS — single-user LAN/localhost tool.
- No persistent database. Session state lives on disk in a working directory; a restart may clear sessions. (A simple on-disk session folder is fine; do not add SQLite unless genuinely needed.)
- No toolpath generation. Fusion handles CAM.

### 1.3 Definitions

- **Depth map:** float32 array from the model, *relative* depth (unitless, arbitrary scale).
- **Heightmap:** normalized 0–1 float32 array after all user-controlled processing; 1.0 = max relief height.
- **Relief mesh:** watertight manifold solid: displaced top surface + flat back + side walls.

---

## 2. Architecture

### 2.1 Stack

| Layer | Choice | License | Notes |
|---|---|---|---|
| Backend | Python 3.12, FastAPI + Uvicorn | MIT/BSD | Single service |
| Depth inference | PyTorch; per-backend adapters: HF `transformers` pipeline (DA2 models) + `depth_anything_3` package (DA3 models) | Apache-2.0 libs; model weights Apache or CC-BY-NC per registry | GPU default on target machine (RTX 3080); CPU fallback |
| Image I/O | Pillow, pillow-heif (HEIC), numpy | MIT-ish | EXIF orientation must be honored |
| Heightmap ops | numpy + scipy.ndimage + OpenCV (`opencv-python-headless`) | BSD/Apache | Blur, bilateral, morphology, resize |
| Meshing | numpy grid triangulation (own code) + `fast-simplification` for decimation | MIT | Avoid heavyweight open3d unless needed |
| STL/OBJ export | `trimesh` | MIT | Binary STL writer, watertight checks |
| Frontend | Vanilla JS + three.js (vendored locally, no CDN at runtime) | MIT | No build step, no npm framework. Single `index.html` + `app.js` + `style.css` |
| Container | Docker + docker compose | — | One service; optional GPU profile |
| Env mgmt (dev outside Docker) | `uv` | MIT | Matches team convention |

**Rationale for grid-triangulation over marching cubes:** the input is a heightmap; a structured grid mesh is simpler, faster, produces cleaner topology for CAM, and trivially guarantees manifoldness once the base/walls are stitched.

**Model download:** at image build time OR first run, download the **default model's** weights into a mounted `models/` volume (`HF_HOME` pointed there) so subsequent container restarts are offline-capable. Non-default registry models download lazily on first selection, into the same volume. The app must work with no internet after a model has been fetched once; selecting a not-yet-downloaded model while offline must produce a clear error, not a hang.

### 2.2 Repo layout

```
photo2relief/
├── SPEC.md                  # this file
├── CLAUDE.md                # working notes, conventions, milestone status
├── README.md                # user-facing: run, use, Fusion import guide
├── docker-compose.yml
├── docker-compose.gpu.yml   # override adding nvidia runtime
├── Dockerfile
├── pyproject.toml           # uv-managed
├── app/
│   ├── main.py              # FastAPI app, routes
│   ├── config.py            # env-driven settings (port, dirs, model id, device)
│   ├── depth.py             # model load (lazy singleton), inference, raw depth cache
│   ├── heightmap.py         # pure functions: normalize, invert, gamma, blur, blend, taper, background flatten
│   ├── meshing.py           # heightmap → watertight mesh; decimation; STL/OBJ writers
│   ├── sessions.py          # session dir management, param persistence (JSON), cache invalidation keys
│   └── schemas.py           # pydantic models for params + API contracts
├── static/
│   ├── index.html
│   ├── app.js
│   ├── style.css
│   └── vendor/three/        # vendored three.js + OrbitControls (pinned version)
├── tests/
│   ├── test_heightmap.py
│   ├── test_meshing.py      # watertightness, dimensions, triangle counts
│   └── fixtures/            # 2–3 small test images + a synthetic gradient PNG
└── data/                    # bind-mounted: sessions/, models/ (gitignored)
```

### 2.3 Processing pipeline (data flow)

```
photo ──► [EXIF-correct, downscale to MAX_INFER_PX] ──► depth model ──► raw_depth.npy (CACHED per image)
                                                                              │
        user params ──────────────────────────────────────────────────────────┤
                                                                              ▼
                                    heightmap stage (fast, re-runs on every param change):
                                    normalize → optional invert → gamma/contrast →
                                    background flatten (threshold) → smoothing (gaussian or bilateral) →
                                    luminance detail blend → edge taper (border ramp) → clamp 0–1
                                                                              │
                              ┌───────────────────────────────────────────────┤
                              ▼                                               ▼
                    preview mesh (grid ≤ 256×256,                   export mesh (grid up to
                    glTF/binary → three.js)                         EXPORT_MAX_GRID, default 1024 long side)
                                                                              │
                                                                              ▼
                                                            base + walls stitch → optional decimation →
                                                            watertight validation → binary STL / OBJ
```

**Caching rule:** the expensive step (depth inference) keys on `(image hash, model_id, inference resolution)`. Everything downstream is cheap numpy and re-runs freely. Never re-run the model on a parameter change; switching depth model re-runs once per model, then hits cache.

---

## 3. Tuning Parameters (the product surface)

All parameters live in one pydantic model (`ReliefParams`), persisted per session as JSON, sent whole on every preview/export call. Defaults chosen for a portrait-photo relief carved in wood.

| Param | Type / Range | Default | Effect |
|---|---|---|---|
| `model_width_mm` | 10–1000 | 150 | Physical X size. Y derived from image aspect (display derived value in UI). |
| `relief_height_mm` | 0.5–100 | 8 | Z range of the displaced surface above the base. |
| `base_thickness_mm` | 0.5–50 | 3 | Flat slab under the relief (gives Fusion stock to hold onto). |
| `depth_model` | enum: registry ids (§5.1.1) | `da3mono-large` (`da2-small` if no CUDA) | Which depth backend generates the raw depth. Changing it triggers cached re-inference. |
| `invert_depth` | bool | false | Flip near/far (also enables intaglio/mold mode). Applied after backend normalization. |
| `gamma` | 0.2–5.0 | 1.0 | Nonlinear depth remap; >1 compresses background, emphasizes foreground. |
| `depth_floor` / `depth_ceiling` | 0–1 each | 0 / 1 | Clip percentile window of depth before normalize (kills far-background noise). |
| `flatten_background` | bool | false | Everything below `background_threshold` (0–1, default 0.15) is set to 0 → clean flat field around subject. |
| `smoothing` | 0–10 (σ px at working res) | 1.5 | Gaussian blur on heightmap. |
| `edge_preserve` | bool | false | Use bilateral filter instead of Gaussian (slower, keeps silhouettes crisp). |
| `detail_blend` | 0–1 | 0.15 | Adds high-pass of image luminance to heightmap (lithophane-style fine texture: hair, fabric). Implemented as `h += detail_blend * highpass(luma) * detail_scale`; re-clamped. |
| `edge_taper_mm` | 0–50 | 0 | Linear ramp of heightmap → 0 over N mm at the outer border (avoids cliffs at the panel edge). |
| `border_frame_mm` | 0–50 | 0 | Flat margin at full-zero height around the image (mounting/clamping flange). |
| `resolution` | enum: preview auto; export `512 / 1024 / 2048` (long side) | 1024 | Grid vertices along the long side for export. |
| `decimate_ratio` | 0–0.95 | 0.0 | Optional post-decimation (0 = off). Note in UI: Fusion + CAM handle 2M triangles fine; decimation mainly for file size. |
| `output_format` | `stl` \| `obj` | stl | Binary STL default. |

UI groups: **Size** (width, relief height, base, frame) · **Depth shaping** (invert, gamma, clip, flatten bg) · **Surface** (smoothing, edge-preserve, detail blend, taper) · **Export** (resolution, decimate, format). Sliders with live numeric inputs; a **Reset to defaults** button; params auto-saved to the session.

---

## 4. API Contract

Base URL `/api`. All responses JSON unless noted. Errors: RFC-ish `{ "error": str, "detail": str }` with proper 4xx/5xx.

| Method & path | Purpose | Notes |
|---|---|---|
| `POST /api/sessions` (multipart: `image`) | Upload photo, create session | Validates type/size (cap 40 MB). Runs depth inference synchronously if fast, else kicks background task. Returns `{session_id, image: {w,h}, status}` |
| `GET /api/sessions/{id}/status` | Poll depth-inference status | `{status: "processing"|"ready"|"error", device: "cpu"|"cuda", elapsed_s}` — frontend polls until ready |
| `GET /api/sessions/{id}/params` / `PUT ...` | Load/save `ReliefParams` | PUT validates via pydantic, persists JSON |
| `GET /api/sessions/{id}/preview/heightmap?{params-hash}` | 2D preview | Returns PNG: left = grayscale heightmap, or single composite hillshaded render (see M3). ≤ 1024 px |
| `POST /api/sessions/{id}/preview/mesh` (body: params) | 3D preview | Returns **binary glTF (.glb)** of the reduced-res relief (grid long side ≤ 256, includes base). Target < 2 s CPU |
| `POST /api/sessions/{id}/export` (body: params) | Full-res export | Background task; returns `{job_id}` |
| `GET /api/jobs/{job_id}` | Poll export | `{status, progress?, download_url?}` |
| `GET /api/jobs/{job_id}/download` | Download file | `application/sla` or OBJ; filename `photo2relief_{session}_{w}x{h}mm.stl` |
| `GET /api/health` | Liveness + device info | Include torch version, cuda availability, model loaded bool |

Static frontend served at `/` by FastAPI (`StaticFiles`).

---

## 5. Key Implementation Details

### 5.1 Depth inference (`depth.py`)

- **Adapter pattern:** a `DepthBackend` protocol (`load()`, `infer(image) -> float32 HxW`, `depth_convention` property) with two implementations: `TransformersDA2Backend` (HF pipeline) and `DA3Backend` (`depth_anything_3` pip package, installed from PyPI or the ByteDance-Seed GitHub repo — pin a tag). Lazy-load on first use; only one model resident at a time (unload/`torch.cuda.empty_cache()` on switch).
- Device = `cuda` if `torch.cuda.is_available()` and env `P2R_DEVICE != cpu`, else CPU. Log the chosen device and active model loudly at startup and expose both in `/api/health`.
- Downscale input so long side ≤ `MAX_INFER_PX` (default 1024; env-tunable) before inference.
- Output: float32 raw depth at inference resolution, saved as `raw_depth_{model_id}.npy` in the session dir. **Cache key is `(image_hash, model_id, MAX_INFER_PX)`** — switching models re-runs inference once, then both are cached; parameter changes never re-run the model.
- **Depth convention differs by backend:** DA2 relative models output disparity-like values (larger = closer); DA3MONO outputs direct depth (verify direction empirically in M2 for EACH backend using the test fixture, normalize all backends to a single internal convention "1.0 = closest" inside the adapter, and document per-backend findings in CLAUDE.md). The user-facing `invert_depth` param remains as an artistic control on top.
- Thread-safety: guard inference and model switching with an `asyncio.Lock` / threadpool executor — one inference at a time is fine for a single-user tool.

### 5.1.1 Model registry

`config.py` defines the registry; the active model is a user-facing dropdown in the UI (Depth shaping group) and a field on `ReliefParams` (`depth_model`). Selecting a model triggers (cached) re-inference, then the normal heightmap pipeline.

| `model_id` | Weights | Backend | License (verify HF page at build time) | Role |
|---|---|---|---|---|
| `da3mono-large` | `depth-anything/DA3MONO-LARGE` | DA3 | expected CC-BY-NC family | **Default.** Monocular-specialized; predicts depth directly (not disparity) → best geometric fidelity |
| `da2-large` | `depth-anything/Depth-Anything-V2-Large-hf` | transformers | CC-BY-NC 4.0 | Disparity-style output: exaggerated near-field / compressed background — often the better *artistic* bas-relief look for portraits |
| `da2-small` | `depth-anything/Depth-Anything-V2-Small-hf` | transformers | Apache-2.0 | CPU-safe fallback; only Apache option; auto-selected default if no CUDA device found |

UI copy should include a one-line hint per model (geometric vs. artistic vs. fast/CPU). README gets a short "which model when" section plus the license table. If `da3mono-large`'s HF license or availability is problematic at build time, fall back to `DA3-LARGE-1.1` (HF page currently lists Apache-2.0) and note the substitution in CLAUDE.md.

### 5.2 Heightmap stage (`heightmap.py`)

- Pure functions, `raw_depth (H×W float32) + ReliefParams → heightmap (H×W float32 in [0,1])`. No I/O. Fully unit-testable.
- Order of operations exactly as in the pipeline diagram (§2.3). Percentile clip before normalize. Luminance high-pass = `luma − gaussian(luma, σ≈8 px)`, scaled so `detail_blend=1.0` contributes ≈15% of relief range.
- Resize (`cv2.resize`, INTER_AREA down / INTER_CUBIC up) to target grid at the **end**, then apply `edge_taper` and `border_frame` in mm-space using `model_width_mm` to convert mm → px.

### 5.3 Meshing (`meshing.py`)

- Vertex grid: `(nx × ny)` top vertices at `z = base_thickness_mm + h * relief_height_mm`; 4 wall strips; back face as a coarse quad (2 triangles) — back does not need the full grid.
- Deterministic winding: all outward normals. `trimesh` assembles + `mesh.is_watertight` must be `True`; if not, fail the export with a clear error rather than shipping a leaky mesh.
- Units: **millimeters**, and say so in README — Fusion's mesh import will ask; STL is unitless so document "select millimeters on import."
- Y-flip: image row 0 is top; mesh +Y must be image top so the relief isn't mirrored. Add a unit test with an asymmetric synthetic heightmap.
- Decimation: `fast_simplification.simplify` on the top surface only (keep walls/base intact), then re-stitch — OR decimate the whole solid and re-validate watertightness. Choose whichever is robust; test both in M4 and document the decision in CLAUDE.md.
- Export: `trimesh.export` binary STL / OBJ. For the 3D preview endpoint, export `.glb` (trimesh supports glTF) — three.js loads it natively with GLTFLoader.

### 5.4 Frontend (`static/`)

- Layout: left panel = parameter controls + upload; right = tabbed or stacked previews (2D heightmap image, 3D viewport). Show session status states (uploading → estimating depth → ready).
- 3D viewport: three.js + OrbitControls, neutral studio lighting (hemisphere + one directional), matcap or standard material in a wood-ish tone. Add a scale grid floor sized in mm.
- Debounce param changes 300 ms → refresh 2D preview immediately, 3D preview on a slower debounce (800 ms) or an explicit "Update 3D" button if regeneration proves janky — decide in M5 based on feel.
- No framework, no bundler. ES modules, vendored three.js pinned version in `static/vendor/`.
- Export flow: button → job poll → progress → browser download. Show final triangle count + file size + physical dimensions in a summary line.

### 5.5 Docker

- Base `python:3.12-slim`. Two run modes; **GPU is the primary/expected mode on the target machine (RTX 3080)**, CPU image kept for portability. Install torch CPU wheels in the base image; `docker-compose.gpu.yml` overrides with CUDA-enabled torch (build arg) + `deploy.resources.reservations.devices` for the nvidia runtime. README documents both: `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up` (primary) and plain `docker compose up` (CPU fallback). Requires NVIDIA Container Toolkit on the host — include the one-line install pointer in README.
- Port env-configurable, default **8090**. Bind mounts: `./data/sessions`, `./data/models` (`HF_HOME` pointed there).
- Healthcheck hits `/api/health`.
- `.dockerignore` the data dir; keep image rebuildable offline once the pip/model caches exist.

---

## 6. Milestones

Work strictly in order. Update CLAUDE.md milestone status table after each.

**M1 — Skeleton & plumbing.** Repo layout, pyproject (uv), FastAPI app serving static index, `/api/health`, session create with image upload + validation + EXIF-corrected re-encode, Docker builds and runs, tests scaffolded.
*Accept:* `docker compose up` → page loads at :8090, uploading a JPEG creates a session dir containing the normalized image; health shows device.

**M2 — Depth inference.** Backend adapters (DA2 via transformers, DA3 via `depth_anything_3`), model registry, lazy load + model switching, download/caching, inference to per-model `raw_depth_{model_id}.npy`, status polling endpoint, CUDA + CPU paths.
*Accept:* Upload of a 12 MP portrait produces a plausible depth map from **each** registry model (debug PNGs; subject clearly nearer than background after convention normalization) — < 5 s on the 3080, < 60 s CPU for `da2-small`. Re-upload of the same file per model reuses cache; switching models mid-session works without OOM. Per-backend depth convention documented in CLAUDE.md. Works offline after first fetch.

**M3 — Heightmap engine + 2D preview.** All `heightmap.py` transforms, `ReliefParams` schema, params persistence, heightmap PNG preview endpoint returning a **hillshaded** render (simple lambertian shade from the gradient — reads far better than raw grayscale) with grayscale toggle via query param.
*Accept:* Unit tests for each transform (invert, gamma, clip, flatten, taper geometry) pass; changing params in curl/HTTP returns visibly correct previews in < 300 ms.

**M4 — Meshing & export.** Grid mesh, walls/base stitch, watertight validation, decimation, binary STL + OBJ, export job endpoints.
*Accept:* Tests prove: watertight = true at 512/1024/2048; bounding box matches `model_width_mm`/derived Y/`base+relief` Z within 0.01 mm; Y-orientation test passes; 1024-grid export completes < 15 s CPU. Manually confirmed: STL opens in Fusion (Insert Mesh, mm) at correct size. *(Bill does the Fusion check; provide the file + expected dimensions.)*

**M5 — Frontend.** Full parameter UI, 2D + 3D previews with debounce, glb preview endpoint, export/download flow, status/progress states, empty-state and error handling.
*Accept:* End-to-end in browser: upload → tune with live feedback → export → file downloads. 3D preview refresh ≤ 2 s CPU at 256 grid. No console errors; works in Chrome and Edge.

**M6 — Polish & docs.** README (quick start ≤ 5 min, Fusion import walkthrough incl. mm units and Manufacture-workspace note, GPU instructions, model license note re: V2 Small vs Base/Large), CLAUDE.md finalized, parameter tooltips in UI, sensible defaults re-tuned against 3 real test photos, tag `v1.0`.
*Accept:* Fresh clone → `docker compose up` → working app following only the README.

### Stretch (post-v1, do not build now)
- Region masking (paint a mask to force background to zero).
- Additional registry backends behind the same `DepthBackend` protocol: Marigold (diffusion; best-in-class fine detail, ~10× slower — fine on the 3080) and/or DA3METRIC-LARGE (metric-consistent proportions).
- Side-by-side model comparison view (same photo, two heightmaps/meshes).
- Preset library (Portrait wood carve / Coin-style / Lithophane).
- Direct .3mf export.

---

## 7. Testing & Quality Bar

- `pytest` in CI-style local run (`uv run pytest`); heightmap and meshing modules ≥ 90% branch coverage — they're pure functions, no excuse.
- Golden-file test: synthetic radial-gradient PNG → deterministic depth bypass (inject fake depth) → exact expected mesh stats (vertex count, bbox, watertight). This isolates meshing from model nondeterminism.
- Never test through the model in unit tests; the model gets one smoke test marked `@pytest.mark.slow`.
- Type hints throughout; `ruff` for lint/format.

## 8. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Depth convention differs per backend (DA2 disparity vs DA3 depth) | Normalize inside each adapter to "1.0 = closest" in M2 with per-backend tests; `invert_depth` remains a user param on top |
| `depth_anything_3` package instability (new project, moving deps) | Pin exact tag/commit; if integration fights back > half a day, ship M2 with DA2 backends only and add DA3 as M2.5 — do not stall the pipeline |
| Huge meshes choke browser preview | Hard cap preview grid at 256; export runs server-side only |
| CUDA torch bloats image / breaks on non-NVIDIA | CPU-only base image; GPU strictly via compose override (primary mode on target 3080 box) |
| Bilateral filter too slow at full res | Apply at working res before final upsample; cap kernel; it's opt-in |
| STL import confusion in Fusion (units) | README walkthrough + filename embeds mm dimensions |
| Model license ambiguity (e.g., DA3-LARGE vs -1.1 listings disagree) | Personal NC use accepted by owner; record HF license field per model in README at build time; keep `da2-small` (Apache) in registry as the clean fallback |
