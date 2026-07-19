// Photo2Relief frontend — vanilla ES modules + vendored three.js. No build step.

import * as THREE from "three";
import { OrbitControls } from "/vendor/three/OrbitControls.js";
import { GLTFLoader } from "/vendor/three/GLTFLoader.js";

// --- state ---------------------------------------------------------------------------

const state = {
  sessionId: null,
  image: null, // {w, h}
  params: null, // ReliefParams, mirrors the sliders
  defaults: null, // snapshot of the session's server-seeded defaults (Reset button)
  depthReady: false,
  readyModel: null, // the depth_model whose raw depth is confirmed cached
  models: [], // registry from /api/models
};

const DEBOUNCE_2D_MS = 300; // SPEC §5.4
const DEBOUNCE_3D_MS = 800;

// --- parameter table (ranges mirror ReliefParams in app/schemas.py) --------------------

const PARAM_GROUPS = {
  "group-size": [
    { id: "model_width_mm", label: "Width (mm)", min: 10, max: 1000, step: 1 },
    { id: "relief_height_mm", label: "Relief height (mm)", min: 0.5, max: 100, step: 0.5 },
    { id: "base_thickness_mm", label: "Base thickness (mm)", min: 0.5, max: 50, step: 0.5 },
    { id: "border_frame_mm", label: "Border frame (mm)", min: 0, max: 50, step: 0.5 },
  ],
  "group-depth": [
    { id: "invert_depth", label: "Invert depth (intaglio / mold)", type: "bool" },
    { id: "gamma", label: "Gamma (>1 favors foreground)", min: 0.2, max: 5, step: 0.05 },
    { id: "depth_floor", label: "Depth floor (clip far)", min: 0, max: 1, step: 0.01 },
    { id: "depth_ceiling", label: "Depth ceiling (clip near)", min: 0, max: 1, step: 0.01 },
    { id: "flatten_background", label: "Flatten background", type: "bool" },
    { id: "background_threshold", label: "Background threshold", min: 0, max: 1, step: 0.01 },
  ],
  "group-surface": [
    { id: "smoothing", label: "Smoothing (σ px)", min: 0, max: 10, step: 0.1 },
    { id: "edge_preserve", label: "Edge-preserving (bilateral)", type: "bool" },
    { id: "detail_blend", label: "Detail blend (luminance)", min: 0, max: 1, step: 0.01 },
    { id: "edge_taper_mm", label: "Edge taper (mm)", min: 0, max: 50, step: 0.5 },
  ],
  "group-export": [
    { id: "resolution", label: "Resolution (grid long side)", type: "enum", options: [512, 1024, 2048] },
    { id: "decimate_ratio", label: "Decimation ratio", min: 0, max: 0.95, step: 0.05 },
    { id: "output_format", label: "Format", type: "enum", options: ["stl", "obj"] },
  ],
};

// --- tiny helpers ----------------------------------------------------------------------

const $ = (id) => document.getElementById(id);

function debounce(fn, ms) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

function setPill(kind, text) {
  const pill = $("status-pill");
  pill.className = `pill pill-${kind}`;
  pill.textContent = text;
}

function showError(message) {
  const banner = $("error-banner");
  banner.textContent = message;
  banner.hidden = !message;
}

async function api(path, options = {}) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || body.error || detail;
    } catch {
      /* non-JSON error body */
    }
    const err = new Error(detail);
    err.status = res.status;
    throw err;
  }
  return res;
}

// --- parameter controls ------------------------------------------------------------------

function buildControls() {
  for (const [groupId, specs] of Object.entries(PARAM_GROUPS)) {
    const container = $(groupId);
    for (const spec of specs) {
      container.appendChild(
        spec.type === "bool"
          ? buildCheckbox(spec)
          : spec.type === "enum"
            ? buildSelect(spec)
            : buildSlider(spec)
      );
    }
  }
}

function buildSlider(spec) {
  const wrap = document.createElement("div");
  wrap.className = "control";
  const name = document.createElement("label");
  name.className = "name";
  name.textContent = spec.label;
  const row = document.createElement("div");
  row.className = "slider-row";
  const range = document.createElement("input");
  range.type = "range";
  const num = document.createElement("input");
  num.type = "number";
  for (const el of [range, num]) {
    el.min = spec.min;
    el.max = spec.max;
    el.step = spec.step;
    el.dataset.param = spec.id;
  }
  range.id = spec.id;
  const commit = (value) => {
    const v = clampParam(spec, Number(value));
    range.value = v;
    num.value = v;
    onParamChanged(spec.id, v);
  };
  range.addEventListener("input", () => commit(range.value));
  num.addEventListener("change", () => commit(num.value));
  row.append(range, num);
  wrap.append(name, row);
  return wrap;
}

function buildCheckbox(spec) {
  const row = document.createElement("label");
  row.className = "checkbox-row";
  const box = document.createElement("input");
  box.type = "checkbox";
  box.id = spec.id;
  box.addEventListener("change", () => onParamChanged(spec.id, box.checked));
  row.append(box, document.createTextNode(spec.label));
  return row;
}

function buildSelect(spec) {
  const wrap = document.createElement("div");
  wrap.className = "control";
  const name = document.createElement("label");
  name.className = "name";
  name.textContent = spec.label;
  const sel = document.createElement("select");
  sel.id = spec.id;
  for (const opt of spec.options) {
    const o = document.createElement("option");
    o.value = opt;
    o.textContent = opt;
    sel.appendChild(o);
  }
  sel.addEventListener("change", () => {
    const raw = sel.value;
    onParamChanged(spec.id, typeof spec.options[0] === "number" ? Number(raw) : raw);
  });
  wrap.append(name, sel);
  return wrap;
}

// Keep the clip window valid client-side (server 422s on ceiling <= floor).
function clampParam(spec, v) {
  if (spec.id === "depth_floor") return Math.min(v, state.params.depth_ceiling - 0.01);
  if (spec.id === "depth_ceiling") return Math.max(v, state.params.depth_floor + 0.01);
  return v;
}

function syncControlsFromParams() {
  for (const specs of Object.values(PARAM_GROUPS)) {
    for (const spec of specs) {
      const el = $(spec.id);
      if (spec.type === "bool") el.checked = state.params[spec.id];
      else if (spec.type === "enum") el.value = state.params[spec.id];
      else {
        el.value = state.params[spec.id];
        el.parentElement.querySelector("input[type=number]").value = state.params[spec.id];
      }
    }
  }
  $("depth_model").value = state.params.depth_model;
  updateModelHint();
  updateDerivedSize();
}

function updateDerivedSize() {
  if (!state.image) return;
  const h = (state.params.model_width_mm * state.image.h) / state.image.w;
  $("derived-size").textContent =
    `Physical size ≈ ${state.params.model_width_mm} × ${h.toFixed(1)} mm ` +
    `(height follows the photo's aspect ratio)`;
}

// --- model dropdown -----------------------------------------------------------------------

async function loadModels() {
  const body = await (await api("/api/models")).json();
  state.models = body.models;
  const sel = $("depth_model");
  for (const m of body.models) {
    const o = document.createElement("option");
    o.value = m.model_id;
    o.textContent = m.available ? m.model_id : `${m.model_id} (unavailable)`;
    o.disabled = !m.available;
    sel.appendChild(o);
  }
  sel.addEventListener("change", () => {
    updateModelHint();
    onParamChanged("depth_model", sel.value);
  });
}

function updateModelHint() {
  const m = state.models.find((m) => m.model_id === $("depth_model").value);
  $("model-hint").textContent = m ? `${m.role} License: ${m.license}.` : "";
}

// --- upload & session lifecycle -------------------------------------------------------------

function wireUpload() {
  const input = $("file-input");
  const zone = $("dropzone");
  input.addEventListener("change", () => input.files[0] && createSession(input.files[0]));
  zone.addEventListener("dragover", (e) => {
    e.preventDefault();
    zone.classList.add("dragover");
  });
  zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("dragover");
    if (e.dataTransfer.files[0]) createSession(e.dataTransfer.files[0]);
  });
}

async function createSession(file) {
  showError("");
  setPill("busy", "Uploading…");
  $("dropzone-text").textContent = file.name;
  const form = new FormData();
  form.append("image", file);
  try {
    const body = await (await api("/api/sessions", { method: "POST", body: form })).json();
    state.sessionId = body.session_id;
    state.image = body.image;
    state.depthReady = false;
    $("image-info").textContent = `${file.name} — ${body.image.w} × ${body.image.h} px`;

    // The server seeds params.json with this device's real defaults at creation.
    state.params = await (await api(`/api/sessions/${state.sessionId}/params`)).json();
    state.defaults = { ...state.params };
    $("params-card").hidden = false;
    $("export-card").hidden = false;
    syncControlsFromParams();
    resetExportUi();
    await waitForDepth();
  } catch (err) {
    setPill("error", "Upload failed");
    showError(`Upload failed: ${err.message}`);
  }
}

async function waitForDepth() {
  state.depthReady = false;
  $("export-btn").disabled = true;
  let s;
  for (;;) {
    s = await (await api(`/api/sessions/${state.sessionId}/status`)).json();
    if (s.status === "ready") break;
    if (s.status === "error") {
      setPill("error", "Depth failed");
      showError(`Depth inference failed: ${s.error}`);
      return;
    }
    setPill("busy", `Estimating depth (${s.model_id ?? "…"} on ${s.device ?? "…"})…`);
    await new Promise((r) => setTimeout(r, 1000));
  }
  state.depthReady = true;
  state.readyModel = s.model_id ?? state.params.depth_model;
  $("export-btn").disabled = false;
  setPill("ready", "Ready");
  showError("");
  refresh2d();
  refresh3d();
}

// --- param changes → save + preview refresh ---------------------------------------------------

function onParamChanged(id, value) {
  state.params[id] = value;
  if (id === "depth_floor" || id === "depth_ceiling") syncControlsFromParams();
  if (id === "model_width_mm") updateDerivedSize();
  saveAndRefresh();
}

const saveAndRefresh = debounce(async () => {
  if (!state.sessionId) return;
  try {
    await api(`/api/sessions/${state.sessionId}/params`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(state.params),
    });
  } catch (err) {
    showError(`Saving parameters failed: ${err.message}`);
    return;
  }
  // A depth_model switch flips the session back to processing; wait it out.
  if (!state.depthReady || state.params.depth_model !== state.readyModel) {
    await waitForDepth(); // refreshes both previews when done
    return;
  }
  refresh2d();
  refresh3dDebounced();
}, DEBOUNCE_2D_MS);

$("reset-btn")?.addEventListener("click", () => {
  if (!state.defaults) return;
  state.params = { ...state.defaults };
  syncControlsFromParams();
  saveAndRefresh();
});

// --- 2D preview -------------------------------------------------------------------------------

let preview2dSeq = 0;
let preview2dObjectUrl = null;

async function refresh2d() {
  if (!state.sessionId || !state.depthReady) return;
  const seq = ++preview2dSeq;
  const grayscale = $("grayscale-toggle").checked;
  const img = $("preview2d");
  img.classList.add("stale");
  let blob;
  try {
    const res = await api(
      `/api/sessions/${state.sessionId}/preview/heightmap?grayscale=${grayscale}`
    );
    blob = await res.blob();
  } catch (err) {
    if (seq === preview2dSeq) img.classList.remove("stale");
    if (err.status !== 409) showError(`2D preview failed: ${err.message}`);
    return;
  }
  if (seq !== preview2dSeq) return; // a newer request superseded this one
  if (preview2dObjectUrl) URL.revokeObjectURL(preview2dObjectUrl);
  preview2dObjectUrl = URL.createObjectURL(blob);
  img.src = preview2dObjectUrl;
  img.hidden = false;
  img.classList.remove("stale");
  $("preview2d-placeholder").hidden = true;
}

$("grayscale-toggle").addEventListener("change", refresh2d);

// --- 3D preview (three.js) ---------------------------------------------------------------------

const viewer = {
  scene: null,
  camera: null,
  renderer: null,
  controls: null,
  mesh: null,
  grid: null,
  fittedDiag: 0,
};

function initViewer() {
  const el = $("viewport");
  viewer.scene = new THREE.Scene();
  viewer.scene.background = new THREE.Color(0x262a33);

  viewer.camera = new THREE.PerspectiveCamera(45, el.clientWidth / el.clientHeight, 1, 5000);
  viewer.renderer = new THREE.WebGLRenderer({ antialias: true });
  viewer.renderer.setPixelRatio(window.devicePixelRatio);
  viewer.renderer.setSize(el.clientWidth, el.clientHeight);
  el.appendChild(viewer.renderer.domElement);

  viewer.controls = new OrbitControls(viewer.camera, viewer.renderer.domElement);
  viewer.controls.enableDamping = true;

  // Neutral studio lighting: hemisphere + one key directional (SPEC §5.4).
  viewer.scene.add(new THREE.HemisphereLight(0xffffff, 0x445566, 1.1));
  const key = new THREE.DirectionalLight(0xffffff, 1.8);
  key.position.set(150, 300, 200);
  viewer.scene.add(key);

  new ResizeObserver(() => {
    const w = el.clientWidth;
    const h = el.clientHeight;
    if (!w || !h) return;
    viewer.camera.aspect = w / h;
    viewer.camera.updateProjectionMatrix();
    viewer.renderer.setSize(w, h);
  }).observe(el);

  (function animate() {
    requestAnimationFrame(animate);
    viewer.controls.update();
    viewer.renderer.render(viewer.scene, viewer.camera);
  })();
}

const WOOD = new THREE.MeshStandardMaterial({ color: 0xa9835a, roughness: 0.82, metalness: 0.0 });

let preview3dSeq = 0;

async function refresh3d() {
  if (!state.sessionId || !state.depthReady) return;
  const seq = ++preview3dSeq;
  let buffer;
  try {
    const res = await api(`/api/sessions/${state.sessionId}/preview/mesh`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(state.params),
    });
    buffer = await res.arrayBuffer();
  } catch (err) {
    if (err.status !== 409) showError(`3D preview failed: ${err.message}`);
    return;
  }
  if (seq !== preview3dSeq) return;

  new GLTFLoader().parse(buffer, "", (gltf) => {
    if (seq !== preview3dSeq) return;
    if (!viewer.scene) initViewer();
    $("preview3d-placeholder").hidden = true;

    if (viewer.mesh) {
      viewer.scene.remove(viewer.mesh);
      viewer.mesh.traverse((o) => o.geometry?.dispose());
    }
    const root = gltf.scene;
    root.traverse((o) => {
      if (o.isMesh) o.material = WOOD;
    });
    // Backend meshes are Z-up (relief toward +Z); three.js is Y-up.
    root.rotation.x = -Math.PI / 2;

    // Center on the origin with the base resting on the y=0 grid plane.
    const box = new THREE.Box3().setFromObject(root);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    root.position.set(-center.x, -box.min.y, -center.z);
    viewer.scene.add(root);
    viewer.mesh = root;

    // mm scale grid floor, sized to the model, 10 mm cells.
    const gridSpan = Math.ceil((Math.max(size.x, size.z) * 1.6) / 100) * 100;
    if (!viewer.grid || viewer.grid.userData.span !== gridSpan) {
      if (viewer.grid) viewer.scene.remove(viewer.grid);
      viewer.grid = new THREE.GridHelper(gridSpan, gridSpan / 10, 0x4a5160, 0x343946);
      viewer.grid.userData.span = gridSpan;
      viewer.scene.add(viewer.grid);
    }

    // Fit the camera on first load or when the model size changes substantially;
    // never yank it around mid-orbit for ordinary param tweaks.
    const diag = size.length();
    if (Math.abs(diag - viewer.fittedDiag) / diag > 0.25) {
      viewer.fittedDiag = diag;
      viewer.camera.position.set(diag * 0.05, diag * 0.9, diag * 1.1);
      viewer.controls.target.set(0, size.y / 2, 0);
      viewer.controls.update();
    }
  },
  (err) => showError(`3D preview failed to parse: ${err.message ?? err}`));
}

const refresh3dDebounced = debounce(refresh3d, DEBOUNCE_3D_MS - DEBOUNCE_2D_MS);

// --- export flow ---------------------------------------------------------------------------------

function resetExportUi() {
  $("export-status").textContent = "";
  $("export-summary").hidden = true;
  $("download-link").hidden = true;
  $("export-btn").disabled = false;
}

$("export-btn").addEventListener("click", async () => {
  if (!state.sessionId) return;
  const btn = $("export-btn");
  btn.disabled = true;
  $("export-summary").hidden = true;
  $("download-link").hidden = true;
  $("export-status").textContent = "Building full-resolution mesh…";
  try {
    const { job_id } = await (
      await api(`/api/sessions/${state.sessionId}/export`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(state.params),
      })
    ).json();

    let job;
    for (;;) {
      job = await (await api(`/api/jobs/${job_id}`)).json();
      if (job.status !== "processing") break;
      await new Promise((r) => setTimeout(r, 500));
    }
    if (job.status === "error") throw new Error(job.error);

    const mb = (job.file_bytes / 1024 / 1024).toFixed(1);
    $("export-summary").textContent =
      `${job.triangles.toLocaleString()} triangles · ${mb} MB · ` +
      `${job.width_mm} × ${job.height_mm} mm`;
    $("export-summary").hidden = false;
    const link = $("download-link");
    link.href = job.download_url;
    link.hidden = false;
    link.click(); // trigger the browser download immediately
    $("export-status").textContent = "Done — check your downloads.";
  } catch (err) {
    $("export-status").textContent = "";
    showError(`Export failed: ${err.message}`);
  } finally {
    btn.disabled = false;
  }
});

// --- boot ------------------------------------------------------------------------------------------

buildControls();
wireUpload();
initViewer();
loadModels().catch((err) => showError(`Could not load model registry: ${err.message}`));
