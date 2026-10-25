// Router, navigation and the shell. Each page is a module with `render(view, ctx)` that may return a cleanup function.
import { h, mount, clear } from "./lib/dom.js";
import { icon } from "./lib/icons.js";
import { startField } from "./lib/field.js";
import { toast } from "./lib/toast.js";

const PAGES = [
  { path: "/", group: "Start here", title: "Mission Control", icon: "home", load: () => import("./pages/home.js") },
];

const view = document.getElementById("view");
const nav = document.getElementById("nav");
const pageTitle = document.getElementById("page-title");
const health = document.getElementById("health");
let cleanup = null;
let token = 0;

const ctx = {
  // The top-right pill shows the result of the last live system check on any page.
  setHealth(text, level) { health.textContent = text; health.className = `pill pill-${level}`; },
  toast,
};

function renderNav(active) {
  let group = null;
  const items = [];
  for (const page of PAGES) {
    if (page.group !== group) { group = page.group; items.push(h("div", { class: "nav-group" }, group)); }
    items.push(h("a", { class: "nav-item" + (page.path === active ? " active" : ""), href: "#" + page.path, title: page.title }, icon(page.icon), h("span", {}, page.title)));
  }
  mount(nav, items);
}

async function route() {
  const path = location.hash.replace(/^#/, "") || "/";
  const page = PAGES.find((p) => p.path === path) || PAGES[0];
  const mine = ++token;
  if (cleanup) { try { cleanup(); } catch (err) { console.error(err); } cleanup = null; }
  renderNav(page.path);
  mount(pageTitle, page.title);
  document.title = `${page.title} · EveFL`;
  clear(view);
  try {
    const module = await page.load();
    if (mine !== token) return;
    const result = await module.render(view, ctx);
    if (mine === token && typeof result === "function") cleanup = result;
    window.scrollTo({ top: 0 });
  } catch (err) {
    console.error(err);
    if (mine !== token) return;
    mount(view, h("div", { class: "card err-card" }, h("h3", {}, "This page could not be shown"), h("p", { class: "section-sub" }, String(err.message || err)),
      h("button", { class: "btn", onclick: () => route() }, "Try again")));
  }
}

addEventListener("hashchange", route);
startField(document.getElementById("field"));
route();
