"use strict";
const $ = (id) => document.getElementById(id);
const api = async (path, opts) => {
  const r = await fetch("/api" + path, opts);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail || j));
  return j;
};
const fmt = (n, d = 0) => Number(n).toLocaleString("en-IN", { maximumFractionDigits: d, minimumFractionDigits: d });
function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  for (const k of kids) e.append(k instanceof Node ? k : document.createTextNode(k));
  return e;
}
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

let OPT = null;
const charts = {};
const FIELDS = ["rainfall_mm", "temp_avg_c", "temp_min_c", "temp_max_c", "soil_ph",
                "organic_carbon_gkg", "soil_texture", "area_kha"];

function fill(select, items) {
  select.replaceChildren(...items.map((v) => el("option", { value: v }, v)));
}

async function init() {
  try {
    OPT = await api("/options");
  } catch (e) {
    $("empty").textContent = "Backend not ready: " + e.message;
    return;
  }
  fill($("state"), Object.keys(OPT.states).sort());
  fill($("crop"), OPT.crops);
  const names = { xgboost: "XGBoost", linear: "Linear Regression", lstm: "LSTM" };
  $("price_model").replaceChildren(...OPT.price_models.map((m) => el("option", { value: m }, names[m])));
  $("year").max = 2040;
  $("state").addEventListener("change", onState);
  $("district").addEventListener("change", loadDefaults);
  $("crop").addEventListener("change", loadDefaults);
  $("reset").addEventListener("click", loadDefaults);
  $("form").addEventListener("submit", run);
  document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => switchTab(t.dataset.tab)));
  $("state").value = OPT.states["Punjab"] ? "Punjab" : $("state").options[0].value;
  $("crop").value = OPT.crops.includes("Wheat") ? "Wheat" : OPT.crops[0];
  onState();
}

function onState() {
  fill($("district"), OPT.states[$("state").value]);
  loadDefaults();
}

async function loadDefaults() {
  const q = new URLSearchParams({ state: $("state").value, district: $("district").value, crop: $("crop").value });
  $("season").value = OPT.crop_season[$("crop").value];
  try {
    const d = await api("/defaults?" + q);
    FIELDS.forEach((f) => ($(f).value = d[f]));
    const lvl = { district_crop: "district–crop average", state_crop: "state–crop average",
                  district: "district average (crop not recorded here)", state: "state average" }[d.level];
    $("defnote").textContent = `(defaults: ${lvl}, n=${d.n})`;
  } catch (e) { showErr(e.message); }
}

function showErr(msg) { const p = $("err"); p.textContent = msg; p.hidden = !msg; }

async function run(ev) {
  ev.preventDefault(); showErr("");
  const body = { state: $("state").value, district: $("district").value, crop: $("crop").value,
                 year: +$("year").value, price_model: $("price_model").value };
  FIELDS.forEach((f) => (body[f] = f === "soil_texture" ? $(f).value : +$(f).value));
  $("go").disabled = true;
  try {
    const r = await api("/predict", { method: "POST", headers: { "Content-Type": "application/json" },
                                      body: JSON.stringify(body) });
    render(r);
  } catch (e) { showErr(e.message); }
  finally { $("go").disabled = false; }
}

function band(f, d = 0) {
  return f.low == null ? "" : `${fmt(f.low, d)} – ${fmt(f.high, d)}  (${f.band_note})`;
}

function render(r) {
  $("empty").hidden = true; $("out").hidden = false;
  $("k-yield").textContent = fmt(r.yield_forecast.value, 1);
  $("k-yield-r").textContent = band(r.yield_forecast, 1) || `Random Forest · XGBoost says ${fmt(r.yield_xgboost_qha, 1)}`;
  $("k-price").textContent = fmt(r.price_forecast.value);
  $("k-price-r").textContent = `${r.price_forecast.model}` + (r.price_forecast.low != null ? ` · ${band(r.price_forecast)}` : "");
  $("k-rev").textContent = fmt(r.revenue_per_ha);
  const eff = r.cascade_effect_rs;
  $("k-link").textContent = (eff >= 0 ? "+" : "−") + fmt(Math.abs(eff));
  $("k-link-r").textContent = `vs. ₹${fmt(r.price_without_yield)} with no yield feature (XGBoost)`;
  $("warn").replaceChildren(...r.warnings.map((w) => el("li", {}, w)));

  const ink = css("--mute"), grid = css("--line"), green = css("--green"), blue = css("--blue"), amber = css("--amber");
  Chart.defaults.color = ink; Chart.defaults.borderColor = grid;
  const make = (id, cfg) => { charts[id]?.destroy(); charts[id] = new Chart($(id), { ...cfg, options: { responsive: true, maintainAspectRatio: false, ...cfg.options } }); };

  const h = r.district_history;
  $("n-hist").hidden = !!h; $("c-hist").parentElement.hidden = !h;
  if (h) make("c-hist", { data: { labels: h.year, datasets: [
      { type: "line", label: "Yield (q/ha)", data: h.yield, borderColor: green, backgroundColor: green, yAxisID: "y", tension: .25 },
      { type: "bar", label: "Rainfall (mm)", data: h.rainfall, backgroundColor: blue + "66", yAxisID: "y2" }] },
    options: { scales: { y: { title: { display: true, text: "q/ha" } }, y2: { position: "right", grid: { drawOnChartArea: false }, title: { display: true, text: "mm" } } } } });

  const years = r.price_trend.map((p) => p.year);
  const ph = r.price_history, histMap = ph ? Object.fromEntries(ph.year.map((y, i) => [y, ph.price[i]])) : {};
  make("c-price", { type: "line", data: { labels: years, datasets: [
      { label: "Historical mean (crop, region)", data: years.map((y) => histMap[y] ?? null), borderColor: amber, backgroundColor: amber, pointRadius: 3, spanGaps: false, showLine: true },
      { label: `${r.price_forecast.model} (this district's yield)`, data: r.price_trend.map((p) => p.price), borderColor: green, borderDash: [5, 4], pointRadius: 0 },
      { label: "Forecast", data: years.map((y) => (y === +$("year").value ? r.price_forecast.value : null)), borderColor: green, backgroundColor: green, pointRadius: 7, showLine: false }] },
    options: { scales: { y: { title: { display: true, text: "₹/quintal" } } } } });

  make("c-link", { type: "line", data: { labels: r.price_vs_yield.map((p) => fmt(p.yield, 1)), datasets: [
      { label: "Price", data: r.price_vs_yield.map((p) => p.price), borderColor: blue, backgroundColor: blue, tension: .2 }] },
    options: { plugins: { legend: { display: false } }, scales: { x: { title: { display: true, text: "Predicted yield (q/ha), ±40% around forecast" } }, y: { title: { display: true, text: "₹/quintal" } } } } });
}

/* ---------------------------- Model evidence tab -------------------------- */
let evLoaded = false;
function switchTab(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("on", t.dataset.tab === name));
  $("tab-forecast").hidden = name !== "forecast"; $("tab-evidence").hidden = name !== "evidence";
  if (name === "evidence" && !evLoaded) loadEvidence();
}

function table(cols, rows, bestCol) {
  const t = el("table"), thead = el("thead"), tb = el("tbody");
  thead.append(el("tr", {}, ...cols.map((c) => el("th", {}, c))));
  let best = Infinity;
  if (bestCol != null) rows.forEach((r) => { if (r._k < best) best = r._k; });
  rows.forEach((r) => { const tr = el("tr", bestCol != null && r._k === best ? { class: "best" } : {}, ...r.cells.map((c) => el("td", {}, c))); tb.append(tr); });
  t.append(thead, tb); return t;
}
const ms = (o, d = 2) => `${fmt(o.mean, d)} ± ${fmt(o.std, d)}`;
function card(title, node, note) { const c = el("div", { class: "card" }, el("h3", {}, title), node); if (note) c.append(el("p", {}, note)); return c; }
function canvasCard(id, title, note) { const cv = el("div", { class: "cv" }, el("canvas", { id })); return card(title, cv, note); }

async function loadEvidence() {
  let M;
  try { M = await api("/metrics"); } catch (e) { $("ev-msg").textContent = e.message; return; }
  evLoaded = true; $("ev-msg").hidden = true;
  const root = $("ev"); root.replaceChildren();
  const cfg = M.config;
  const g = el("div", { class: "ev-grid" });
  const rowsY = Object.entries(M.stage1_cv).map(([m, v]) => ({ _k: v.rmse.mean, cells: [m, ms(v.rmse), ms(v.mae), ms(v.r2, 3)] }));
  g.append(card("Stage 1 · yield, grouped 5-fold CV (q/ha)", table(["Model", "RMSE", "MAE", "R²"], rowsY, 1),
    `${fmt(cfg.n_yield_rows)} district-year rows; whole districts are held out in each fold.`));
  const cc = M.cascade_cv;
  const rowsP = Object.entries(cc.price).filter(([m]) => !m.includes("yield")).map(([m, v]) => ({ _k: v.rmse.mean, cells: [m, ms(v.rmse, 1), ms(v.mae, 1), ms(v.r2, 3)] }));
  g.append(card("Stage 2 · price, grouped 5-fold CV (₹/quintal)", table(["Model", "RMSE", "MAE", "R²"], rowsP, 1),
    `${fmt(cfg.n_price_rows)} synthetic price rows; R² is high partly because price is crop-driven by construction.`));
  const ab = ["XGBoost, no yield", "XGBoost", "XGBoost, true yield"].map((m, i) => ({ _k: i, cells: [["No yield feature", "Predicted yield", "True yield"][i], ms(cc.price[m].rmse, 1), ms(cc.price[m].mae, 1), ms(cc.price[m].r2, 3)] }));
  g.append(card("Ablation of the yield link (XGBoost price model)", table(["Setting", "RMSE", "MAE", "R²"], ab),
    "Does feeding the Stage-1 prediction into Stage 2 reduce price error?"));
  g.append(card("Per-fold results", table(["Fold", "Yield RMSE (XGB)", "Yield R²", "Price RMSE with yield", "without", "LR"],
    cc.per_fold.map((f) => ({ cells: [f.fold, fmt(f.yield_rmse_xgb, 2), fmt(f.yield_r2_xgb, 3), fmt(f.price_rmse_with_yield, 1), fmt(f.price_rmse_no_yield, 1), fmt(f.price_rmse_lr, 1)] })))));
  root.append(g);

  const g2 = el("div", { class: "ev-grid" });
  g2.append(canvasCard("e-noise", "Robustness: noise added to the yield input",
    cc.noise_breakeven_sigma != null ? `Price RMSE overtakes the no-yield level (dashed) at σ ≈ ${fmt(cc.noise_breakeven_sigma, 1)} q/ha.` : "Price RMSE never reaches the no-yield level in the tested range."));
  g2.append(canvasCard("e-imp", "Stage-1 feature importance (gain, one-hot groups summed)", "For interpretation only."));
  root.append(g2);

  const T = M.temporal;
  const tr = [];
  Object.entries(T.yield).forEach(([m, v]) => tr.push({ cells: ["Yield", m, fmt(v.rmse, 2), fmt(v.mae, 2), fmt(v.r2, 3)] }));
  Object.entries(T.price).forEach(([m, v]) => tr.push({ cells: ["Price", m, fmt(v.rmse, 1), fmt(v.mae, 1), fmt(v.r2, 3)] }));
  const g3 = el("div", { class: "ev-grid" });
  g3.append(card(`Temporal holdout · train ${T.train_years}, test ${T.test_years}`, table(["Stage", "Model", "RMSE", "MAE", "R²"], tr),
    "Train on earlier years, test on the last two. The paper found the yield link did not help tree models in this setting; compare the XGBoost rows here to see whether that holds on this data. The LSTM is evaluated only here."));
  const byCrop = M.breakdown.crop.slice().sort((a, b) => b.yield_rmse - a.yield_rmse);
  g3.append(card("Yield error by crop (XGBoost, CV)", table(["Crop", "n", "Mean yield", "RMSE"],
    byCrop.map((c) => ({ cells: [c.name, fmt(c.n), fmt(c.mean_yield, 1), fmt(c.yield_rmse, 2)] })))));
  root.append(g3);

  const g4 = el("div", { class: "ev-grid" });
  g4.append(card("Learning curve (XGBoost yield)", table(["Training districts", "RMSE", "R²"],
    M.learning_curve.map((l) => ({ cells: [`${l.fraction * 100}%`, fmt(l.rmse, 2), fmt(l.r2, 3)] })))));
  g4.append(card("Hyperparameter sensitivity (XGBoost depth)", table(["max_depth", "RMSE (mean ± std)"],
    M.depth_sensitivity.map((d) => ({ cells: [d.max_depth, `${fmt(d.rmse_mean, 2)} ± ${fmt(d.rmse_std, 2)}`] }))),
    "A sensitivity check, not a tuned result."));
  root.append(g4);

  const green = css("--green"), amber = css("--amber"), blue = css("--blue");
  const sig = Object.keys(cc.noise_rmse);
  const noy = cc.price["XGBoost, no yield"].rmse.mean;
  new Chart($("e-noise"), { type: "line", data: { labels: sig, datasets: [
    { label: "Price RMSE", data: sig.map((s) => cc.noise_rmse[s]), borderColor: blue, backgroundColor: blue },
    { label: "No-yield RMSE", data: sig.map(() => noy), borderColor: amber, borderDash: [6, 4], pointRadius: 0 }] },
    options: { responsive: true, maintainAspectRatio: false, scales: { x: { title: { display: true, text: "σ of noise on yield (q/ha)" } }, y: { title: { display: true, text: "₹/quintal" } } } } });
  const imp = Object.entries(M.feature_importance).slice(0, 10);
  new Chart($("e-imp"), { type: "bar", data: { labels: imp.map((i) => i[0]), datasets: [{ data: imp.map((i) => i[1] * 100), backgroundColor: green }] },
    options: { indexAxis: "y", responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { title: { display: true, text: "% of total gain" } } } } });
}

init();
