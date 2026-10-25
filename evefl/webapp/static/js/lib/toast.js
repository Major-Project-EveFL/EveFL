import { h } from "./dom.js";

export function toast(message, ms = 6000) {
  const box = document.getElementById("toasts");
  const el = h("div", { class: "toast" }, message);
  box.append(el);
  setTimeout(() => el.remove(), ms);
}
