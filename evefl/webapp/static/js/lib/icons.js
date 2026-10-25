import { h } from "./dom.js";

// 24x24 outline icons (stroke = currentColor).
const PATHS = {
  home: ["M3 11.5 12 4l9 7.5", "M5.5 10v9.5h13V10", "M10 19.5v-5h4v5"],
  play: ["M7 5v14l12-7z"],
  check: ["M5 12.5l4.5 4.5L19 7.5"],
};

export function icon(name) {
  const svg = h("svg", { viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", "stroke-width": "1.9", "stroke-linecap": "round", "stroke-linejoin": "round", "aria-hidden": "true" });
  for (const d of PATHS[name] || []) svg.append(h("path", { d }));
  return svg;
}
