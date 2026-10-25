export const reducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// Count a number up from zero when it appears. Writes text only; the final value is always exact.
export function countUp(el, to, { decimals = 0, duration = 900, prefix = "", suffix = "" } = {}) {
  const format = (v) => prefix + v.toFixed(decimals) + suffix;
  if (reducedMotion() || !Number.isFinite(to)) { el.textContent = format(to); return; }
  const start = performance.now();
  const tick = (now) => {
    const t = Math.min(1, (now - start) / duration);
    const eased = 1 - Math.pow(1 - t, 3);
    el.textContent = format(to * eased);
    if (t < 1) requestAnimationFrame(tick); else el.textContent = format(to);
  };
  requestAnimationFrame(tick);
}
