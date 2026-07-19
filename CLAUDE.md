# CLAUDE.md — Photo2Relief

Working notes and conventions for Claude Code. **`SPEC.md` is the source of truth** for architecture, API contract, parameters, and milestone acceptance criteria. If this file and SPEC.md conflict, SPEC.md wins; if reality and SPEC.md conflict, update the Decisions Log below and note it here.

## What this project is

Single-user, locally-hosted Docker web app: photo → monocular depth (Depth Anything family, swappable via model registry) → tunable heightmap → watertight relief mesh → binary STL/OBJ in **millimeters** for Autodesk Fusion CAM. Owner's machine: Windows desktop, RTX 3080, Docker; GPU compose is the primary run mode. Port 8090.

## How to work

- Build strictly in milestone order (SPEC §6, M1→M6). Do not start a milestone until the previous one's acceptance criteria pass. Update the status table below as you go.
- **Commit as you go, not just at milestone boundaries.** Make small, reviewable commits per logical unit *within* a milestone (e.g. registry, then adapters, then endpoint wiring, then tests) — don't batch a whole milestone into one commit. Conventional-commit style messages (`feat:`, `fix:`, `test:`, `docs:`, `chore:`). Run `ruff check .` + `ruff format .` and the fast test suite before each commit.
- Write the tests alongside the code, not after. `heightmap.py` and `meshing.py` are pure functions — keep them that way (no I/O, no globals) and hold them to ≥90% branch coverage.
- Never mark a milestone done on "it should work" — run the acceptance checks and paste evidence (test output, timing numbers, debug PNG paths) into the status table notes.
- Items needing the owner (e.g., the Fusion import check in M4) → add to **Owner checklist** below and continue with what's unblocked.

## Commands

```bash
uv sync                                          # deps
uv run uvicorn app.main:app --reload --port 8090 # dev server
uv run pytest                                    # fast tests
uv run pytest -m slow                            # + real-inference smoke test
uv run ruff check . && uv run ruff format .      # lint/format (run before every commit)
docker compose up --build                                                   # CPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build   # GPU (primary)
```

## Conventions & hard rules

- Python 3.12, `uv`-managed. Type hints everywhere; pydantic models in `schemas.py` are the single definition of `ReliefParams` and all API payloads.
- **Never re-run depth inference on a parameter change.** Raw depth caches on `(image_hash, model_id, MAX_INFER_PX)` as `raw_depth_{model_id}.npy` in the session dir.
- Depth backends implement the `DepthBackend` protocol (`load`, `infer`, `depth_convention`) and normalize output to the internal convention **1.0 = closest to camera** before anything downstream sees it. One model resident at a time; free CUDA memory on switch.
- Meshes must pass `trimesh` watertight validation before export — fail loudly, never ship a leaky mesh. Units are mm end-to-end. +Y = image top (there is a unit test for this; keep it passing).
- Frontend: vanilla JS + vendored three.js only. No npm, no bundler, no framework, no CDN at runtime.
- No database. Sessions are directories under `data/sessions/` with a `params.json`.
- Config via env vars with `P2R_` prefix, defined in `config.py` only.
- Model licenses: registry defaults include CC-BY-NC weights — accepted by owner for personal use (see Decisions Log). Keep `da2-small` (Apache-2.0) working as the clean fallback at all times.

## Gotchas (update as discovered)

- **Per-backend depth convention (verified empirically in M2, RTX 3080):**
  - **`da2-small` / `da2-large` (transformers DA2)** → native output is **disparity, larger = closer**. Adapter uses `convention="disparity"` (NO flip), then min-max normalize to [0,1]. Verified on a synthetic shaded-sphere "portrait": subject center ≈0.92, background corners ≈0.26 → subject correctly nearer. Debug PNG (`depth_da2-large.png`) shows sphere bright/near, background dark/far. ✅
  - **`da3mono-large` (DA3MONO)** → native output is **direct depth, larger = farther** — verified empirically in M2.5 (sphere test: raw center 0.366 vs corners 1.038). Adapter uses `convention="depth"` (negate before normalize). ✅
  - Internal convention everywhere downstream: **1.0 = closest to camera**. `invert_depth` (M3 param) is an artistic control layered on top.
- **DA3 runs via subprocess isolation** (`da3worker/`): `depth_anything_3` pins `numpy<2` + open3d/xformers/pycolmap, so it lives in its own uv project/venv and `DA3Backend` shells out per inference. Setup: `cd da3worker && uv sync --extra cu128` (or `--extra cpu`). Costs **~15 s per new photo** on the dev box (venv import + model load dominate; forward pass 0.64 s) — paid once per (image, model) thanks to the cache. If that ever hurts, the upgrade path is a persistent `--serve` worker (model resident, stdin/stdout protocol); deliberately not built — single-shot is simpler, can't hang-leak, and idles at zero VRAM.
- `depth_anything_3==0.1.1` uses `addict` but doesn't declare it — `da3worker/pyproject.toml` pins it explicitly. Its logger also chatters on stdout ignoring `logging.disable`; harmless for the single-shot protocol (we only parse the exit code and the output file).
- Warm GPU inference (model resident, weights cached) is **~0.08 s (da2-small) / ~0.20 s (da2-large)** at 768×1024 infer res — model *load* is the ~2–7 s one-time cost. First-ever run also downloads weights (~28 s small / ~13 s large including download).
- torch is a uv **extra**, not a core dep: `cpu` and `cu128` are mutually-exclusive (`[tool.uv] conflicts`). Dev + GPU image use `--extra cu128`; CPU base image uses `--extra cpu` (Docker build arg `TORCH_EXTRA`). Pinned `torch==2.9.1` (exists in both indexes). Never add torch to `[project.dependencies]`.
- DA3-LARGE vs DA3-LARGE-1.1 license listings disagree upstream (CC-BY-NC vs Apache-2.0). Record whatever the HF license field says at build time in README; prefer `-1.1` checkpoints if that fallback is used (upstream recommends them post-bugfix).
- STL is unitless — the Fusion mm-import instruction in README is a real user-facing failure mode, don't drop it.
- EXIF orientation: normalize at upload; phone photos will otherwise produce rotated reliefs.
- Torch CUDA wheels only in the GPU image (build arg), never in the base image.

## Milestone status

| Milestone | Status | Evidence / notes |
|---|---|---|
| M1 Skeleton & plumbing | ✅ done | `uv run pytest` → 7 passed (health, static index, session create JPEG/PNG, oversized-MP rejection, bad-data rejection, normalized-image-on-disk). `docker compose build && docker compose up -d` → container reports `healthy`; `GET /api/health` → `{"status":"ok","device":"cpu","cuda_available":false,...}`; `GET /` serves index.html; `POST /api/sessions` with a 400×600 JPEG → `{"session_id":...,"image":{"w":400,"h":600},"status":"ready"}` and `data/sessions/{id}/{source.png,meta.json}` present on the host bind mount. `ruff check .` / `ruff format --check .` clean. |
| M2 Depth inference (registry + adapters) | ✅ done (DA2 backends; DA3 → M2.5) | `uv run pytest` → 19 passed (10 M2 unit tests: registry, convention normalize/flip, cache-reuse-no-reinference, downscale cap, unavailable-DA3 error, status endpoint). `uv run pytest -m slow` → 1 passed (real da2-small inference). **GPU verified on RTX 3080** (torch 2.9.1+cu128): 12 MP synthetic portrait → both `da2-small` & `da2-large` produce plausible depth (subject nearer than background, see debug PNGs). Warm infer **0.078 s / 0.203 s** (≪ 5 s target). Model switch da2-small→da2-large freed the first model (1.35 GB resident after, no OOM). Cache reuse: 2nd inference of same image = np.load only, no model call. **Offline verified** (`HF_HUB_OFFLINE=1` loads from `data/models`). Live HTTP E2E: `POST /api/sessions` → `processing`(da2-large,cuda) → poll → `ready` in 6.3 s → health `model_loaded:true`. `ruff` clean. DA3/`da3mono-large` deferred to M2.5 (dependency conflict — see gotchas). |
| **M2.5 DA3 backend (`da3mono-large`)** | ✅ done | `da3worker/` isolated uv project (own lock; `depth-anything-3==0.1.1` + undeclared-upstream `addict`; torch cpu/cu128 extras). `DA3Backend` shells out per inference (temp PNG → raw .npy → normalize in main app; 600 s timeout; clear venv-missing error). **Convention verified empirically on the 3080:** raw center 0.366 < corners 1.038 → larger = farther = direct depth, `convention="depth"` flip correct; normalized sphere center 0.971 vs corners 0.455. Forward pass 0.64 s; **~15 s wall per new photo** (worker venv import + model load; once per (image, model) via cache). `available=True`, GPU default is `da3mono-large` again. **License: Apache-2.0 per HF field (2026-07-18)** — better than the CC-BY-NC the SPEC expected. Fast suite 23 passed; `-m slow` 2 passed (DA2 + real DA3 subprocess round-trip). Live E2E native: upload → `ready` (da3mono-large, cuda) in 15.4 s, offline (`HF_HUB_OFFLINE=1`). **GPU-compose E2E (primary run mode): container healthy, upload → in-container DA3 on cuda → `ready` in 31.9 s, depth convention correct (center 0.975 vs corners 0.487)** — first attempt failed on missing `libGL.so.1` (worker's transitive opencv-python), fixed by adding `libgl1`/`libglib2.0-0` to the image apt layer; the failure surfaced as a clean `status:error` with full stderr, proving the no-hang error path in anger. CPU image ships worker source but no venv (slim, by design); GPU image builds the worker venv via `TORCH_EXTRA=cu128` conditional + uv cache mounts. |
| M3 Heightmap engine + 2D preview | ☐ not started | |
| M4 Meshing & export | ☐ not started | Fusion check → Owner checklist |
| M5 Frontend | ☐ not started | |
| M6 Polish & docs, tag v1.0 | ☐ not started | Re-tune defaults on 3 real photos; A/B models per photo |

## Owner checklist (Bill)

- [ ] M4: open the provided test STL in Fusion (Insert Mesh, **mm**), confirm bounding box matches stated dimensions, confirm Manufacture workspace accepts the mesh body.
- [ ] M6: provide 2–3 real photos (a portrait, an object, a pet?) for default-tuning and the model A/B comparison.

## Decisions log

| Date | Decision | Why |
|---|---|---|
| 2026-07-18 | 2.5D relief pipeline (heightmap → grid mesh), not generative 3D or marching cubes | Matches 3-axis CNC constraints; cleaner topology for CAM; trivially watertight |
| 2026-07-18 | Model registry: `da3mono-large` (default), `da2-large`, `da2-small` (CPU/Apache fallback) | Owner accepts CC-BY-NC for personal use; DA3MONO = best geometry, DA2-L = artistic disparity look, dropdown enables A/B |
| 2026-07-18 | GPU compose is primary run mode; CPU base image retained | Target machine has RTX 3080; keeps image portable |
| 2026-07-18 | Port 8090; binary STL default; mm everywhere | Owner-confirmed port; Fusion workflow |
| 2026-07-18 | Preview = instant 2D hillshade + ≤256-grid .glb in three.js; full res only at export | Keeps param-tweaking loop < 2 s |
| 2026-07-18 | M2 ships DA2 backends only; `depth_anything_3` (DA3) deferred to M2.5 | DA3 pkg pins `numpy<2` + open3d/xformers/pycolmap — hard conflict with our numpy-2/trimesh/opencv stack; exactly the SPEC risk-table trigger. DA2-L/DA2-S fully working & GPU-verified; pipeline not stalled. |
| 2026-07-18 | Effective GPU default model is `da2-large` (was `da3mono-large`) until M2.5 | `da3mono-large` marked `available=False`; `resolve_default_model()` falls back to best available. CPU default stays `da2-small`. |
| 2026-07-18 | torch via mutually-exclusive uv extras `cpu`/`cu128`, pinned `torch==2.9.1` | Honors "no CUDA torch in base image"; one lockfile serves CPU Docker + GPU dev/image. cu128 wheels run fine under the box's CUDA 13.1 driver (backward compat). |
| 2026-07-18 | M2.5 done now (not post-M4): DA3 via single-shot subprocess worker | Spike proved install+inference on this box, retiring the deferral's risk rationale; M3+ heightmap defaults should tune against the real default model. Single-shot over persistent worker: simpler, no hang risk, zero idle VRAM; ~15 s once per photo is acceptable (cached thereafter). |
| 2026-07-18 | `da3mono-large` license recorded as **Apache-2.0** (HF field, 2026-07-18) | SPEC expected CC-BY-NC family; actual HF listing is Apache-2.0. Recorded in README table. `da2-large` remains the only CC-BY-NC weight. |
| | *(append new decisions here — one line each, never delete)* | |
