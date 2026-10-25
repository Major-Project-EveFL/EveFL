// Mission Control: what EveFL is, the measured numbers, and a live system check that runs the real code.
import { h, mount } from "../lib/dom.js";
import { icon } from "../lib/icons.js";
import { getJSON, stream } from "../lib/api.js";
import { countUp } from "../lib/anim.js";

const KIND_LABEL = { measured: "MEASURED", simulated: "SIMULATED", recorded: "RECORDED", "real code": "REAL CODE", library: "LIBRARY, NOT IN TRAINING", project: "FINAL-YEAR PROJECT", stub: "STUB" };
const badge = (kind) => h("span", { class: `badge ${kind.replace(" ", "-")}` }, KIND_LABEL[kind] || kind.toUpperCase());

function heroArt() {
  // Three hospitals send photons to the server; one link has an eavesdropper. Decorative: it shows the idea, not a result.
  const svg = h("svg", { class: "hero-art", viewBox: "0 0 440 300", role: "img", "aria-label": "Three hospitals connected to a server by quantum links, one link watched by an eavesdropper" });
  svg.append(h("defs", {},
    h("linearGradient", { id: "lg", x1: "0", y1: "0", x2: "1", y2: "1" }, h("stop", { offset: "0", "stop-color": "#22d3ee" }), h("stop", { offset: "1", "stop-color": "#a78bfa" }))));
  const hospitals = [[70, 60], [70, 150], [70, 240]];
  const server = [350, 150];
  hospitals.forEach(([x, y], i) => {
    const d = `M${x + 34} ${y} C 190 ${y}, 250 ${server[1]}, ${server[0] - 40} ${server[1]}`;
    const watched = i === 2;
    svg.append(h("path", { d, fill: "none", stroke: watched ? "#f87171" : "#22d3ee", "stroke-opacity": watched ? ".75" : ".5", "stroke-width": "2", class: "flow" }));
    svg.append(h("circle", { r: "4.5", fill: watched ? "#fca5a5" : "#67e8f9" }, h("animateMotion", { dur: `${3 + i * 0.7}s`, repeatCount: "indefinite", path: d })));
    svg.append(h("rect", { x: x - 34, y: y - 28, width: 68, height: 56, rx: 14, fill: "#101830", stroke: "url(#lg)", "stroke-width": "1.6" }));
    svg.append(h("path", { d: `M${x - 9} ${y}h18M${x} ${y - 9}v18`, stroke: "#e8edff", "stroke-width": "3.2", "stroke-linecap": "round" }));
    svg.append(h("text", { x, y: y + 46, "text-anchor": "middle", fill: "#94a3c7", "font-size": "11" }, `Hospital ${i + 1}`));
  });
  svg.append(h("rect", { x: server[0] - 40, y: server[1] - 46, width: 80, height: 92, rx: 16, fill: "#101830", stroke: "url(#lg)", "stroke-width": "1.8" }));
  [-24, -8, 8, 24].forEach((dy) => svg.append(h("rect", { x: server[0] - 26, y: server[1] + dy - 5, width: 52, height: 10, rx: 4, fill: "none", stroke: "#60a5fa", "stroke-opacity": ".7" })));
  svg.append(h("text", { x: server[0], y: server[1] + 66, "text-anchor": "middle", fill: "#94a3c7", "font-size": "11" }, "Server"));
  const eve = [196, 214];
  svg.append(h("circle", { cx: eve[0], cy: eve[1], r: 15, fill: "#2a1220", stroke: "#f87171", "stroke-width": "1.6" }));
  svg.append(h("text", { x: eve[0], y: eve[1] + 5, "text-anchor": "middle", fill: "#f87171", "font-size": "15", "font-weight": "800" }, "E"));
  return svg;
}

function kpi(item) {
  // `item.values` is a list of numbers shown side by side (two seeds); otherwise a single `item.value`.
  const numbers = item.values ?? [item.value];
  const spans = numbers.map(() => h("span", {}, "0"));
  const value = h("div", { class: "value" }, ...spans.flatMap((s, i) => (i ? [h("span", { class: "sep" }, " / "), s] : [s])));
  const card = h("div", { class: "card kpi reveal" }, h("div", { class: "label" }, item.label), value, h("div", { class: "note" }, item.note), badge(item.kind));
  numbers.forEach((n, i) => countUp(spans[i], n, { decimals: item.decimals ?? 0, suffix: item.suffix ?? "" }));
  return card;
}

function checkRow(plan) {
  const mark = h("div", { class: "mark" }, "");
  const detail = h("div", { class: "detail" });
  const time = h("div", { class: "time" }, "queued");
  const row = h("div", { class: "check", id: `check-${plan.name}` }, mark,
    h("div", {}, h("div", { class: "title" }, plan.title, " ", badge(plan.kind)), h("div", { class: "what" }, plan.what), detail), time);
  return { row, mark, detail, time };
}

const doneText = (p) => (!p.healthy ? `${p.failed} of ${p.total} checks FAILED. See the red lines.`
  : `${p.passed} of ${p.total} checks passed${p.skipped ? `, ${p.skipped} skipped (see the amber line)` : ""} in ${p.seconds.toFixed(1)} s.`);

export async function render(view, ctx) {
  const data = await getJSON("/api/summary");
  const hl = data.headline;
  const [seed0, seed1] = hl.runs;

  const checksBox = h("div", { class: "checks" });
  const bannerBox = h("div", {});
  const runButton = h("button", { class: "btn primary", id: "run-check" }, icon("play"), "Run live system check");
  let closeStream = null;

  async function prepare() {
    const plan = await getJSON("/api/selfcheck/plan");
    const rows = new Map(plan.checks.map((c) => [c.name, checkRow(c)]));
    mount(checksBox, [...rows.values()].map((r) => r.row));
    return rows;
  }
  let rows = await prepare();

  async function run() {
    if (closeStream) closeStream();
    runButton.disabled = true;
    mount(bannerBox);
    rows = await prepare();
    ctx.setHealth("running checks…", "run");
    const first = rows.values().next().value;
    if (first) { first.row.classList.add("running"); first.time.textContent = "running…"; }
    const order = [...rows.keys()];
    let finished = 0;
    closeStream = stream("/api/selfcheck/stream", {
      onEvent(kind, payload) {
        if (kind === "result") {
          const r = rows.get(payload.name);
          if (!r) return;
          r.row.classList.remove("running");
          r.row.classList.add(payload.status);
          r.mark.textContent = { pass: "✓", fail: "✕", skip: "–" }[payload.status];
          r.detail.textContent = payload.detail;
          r.time.textContent = `${Math.round(payload.seconds * 1000)} ms`;
          finished += 1;
          const next = rows.get(order[finished]);
          if (next) { next.row.classList.add("running"); next.time.textContent = "running…"; }
        } else {
          runButton.disabled = false;
          const level = !payload.healthy ? "bad" : payload.skipped ? "warn" : "ok";
          mount(bannerBox, h("div", { class: `banner ${level}` }, doneText(payload)));
          ctx.setHealth(!payload.healthy ? `${payload.failed} check(s) failed` : `${payload.passed}/${payload.total} checks passed`, level);
        }
      },
      onError(err) {
        runButton.disabled = false;
        mount(bannerBox, h("div", { class: "banner bad" }, `The live check stopped: ${err.message}. Press the button to try again.`));
        ctx.setHealth("check interrupted", "bad");
      },
    });
  }
  runButton.addEventListener("click", run);

  const steps = [["Hospitals train locally", "Each hospital trains a ResNet-18 on its own X-rays. Images never leave the hospital."],
    ["A quantum link is checked", "A simulated BB84 exchange on each connection reads the error rate (QBER). An eavesdropper raises it."],
    ["A controller decides", "Below 5% the link is SECURE, up to 11% CAUTION, from 11% LOCKDOWN. The response is graduated, not a blunt pause."],
    ["The server aggregates", "Only trusted updates are averaged into the shared model. A suspicious link is set aside; the others keep training."]];

  mount(view,
    h("section", { class: "hero reveal" },
      h("div", {},
        h("h1", {}, data.title),
        h("p", { class: "lead" }, data.tagline),
        h("div", { class: "chips" }, badge("project"), badge("simulated"), badge("measured")),
        h("div", { class: "cta" }, runButton)),
      heroArt()),
    h("div", { class: "grid c4", style: { marginTop: "1rem" } },
      kpi({ label: "Served model, test AUC", value: hl.served_model_test_auc, decimals: 3, kind: "measured", note: `Round ${hl.served_model_round} checkpoint, chosen on validation, one test evaluation, 14 chest findings` }),
      kpi({ label: "Two training runs", values: [seed0.test_auc_round_50, seed1.test_auc_round_50], decimals: 3, kind: "measured", note: "Test AUC at round 50 for seed 0 and seed 1. Two runs, no confidence interval" }),
      kpi({ label: "Hospitals", value: hl.n_hospitals, kind: "measured", note: "Patient-level split, non-IID Dirichlet(0.5); the audit finds no shared patient or image" }),
      kpi({ label: "LOCKDOWN bound", value: hl.lockdown_bound * 100, suffix: "%", kind: "real code", note: "The BB84 asymptotic limit; the code refuses any lower setting" })),
    h("h2", { class: "section" }, "How it works"),
    h("p", { class: "section-sub" }, "One federated round, in four steps"),
    h("div", { class: "grid c4" }, ...steps.map(([title, text], i) =>
      h("div", { class: "card step reveal", style: { animationDelay: `${i * 90}ms` } }, h("div", { class: "num" }, i + 1), h("h4", {}, title), h("p", {}, text)))),
    h("h2", { class: "section" }, "Is it working right now?"),
    h("p", { class: "section-sub" }, "These checks run the real code on this machine, in front of you. Nothing here is a recording. The keys in the checks are random test bytes; the simulator's own keys are not secret."),
    h("div", { class: "card" }, h("div", { class: "panel-head" }, h("strong", {}, "Live system check")), checksBox, bannerBox),
    h("p", { class: "honest" }, "Final-year major project, Atria Institute of Technology (VTU). The quantum layer is a simulator and its keys are not secret; the classifier is not diagnostic; every number is labelled measured, simulated, recorded or real code. The evidence behind each one is in docs/DECISIONS.md."));
  return () => { if (closeStream) closeStream(); };
}
