// A quiet animated background: faint drifting points and a rare photon streak. Pauses when the tab is hidden.
import { reducedMotion } from "./anim.js";

export function startField(canvas) {
  if (reducedMotion()) return;
  const ctx = canvas.getContext("2d");
  let w = 0, h = 0, raf = 0, last = 0;
  const dots = Array.from({ length: 56 }, () => ({ x: Math.random(), y: Math.random(), r: 0.6 + Math.random() * 1.4, vx: (Math.random() - 0.5) * 0.00004, vy: (Math.random() - 0.5) * 0.00004, a: 0.2 + Math.random() * 0.5 }));
  const streaks = [];
  const resize = () => { const dpr = Math.min(2, window.devicePixelRatio || 1); w = canvas.width = innerWidth * dpr; h = canvas.height = innerHeight * dpr; };
  resize(); addEventListener("resize", resize);

  const frame = (now) => {
    const dt = Math.min(64, now - last); last = now;
    ctx.clearRect(0, 0, w, h);
    for (const d of dots) {
      d.x = (d.x + d.vx * dt + 1) % 1; d.y = (d.y + d.vy * dt + 1) % 1;
      ctx.fillStyle = `rgba(160,190,255,${d.a * 0.5})`; ctx.beginPath(); ctx.arc(d.x * w, d.y * h, d.r * (w / 1400 + 0.6), 0, 6.283); ctx.fill();
    }
    if (Math.random() < 0.004 && streaks.length < 3) streaks.push({ x: Math.random() * 0.6, y: Math.random(), t: 0 });
    for (let i = streaks.length - 1; i >= 0; i--) {
      const s = streaks[i]; s.t += dt / 1400;
      if (s.t >= 1) { streaks.splice(i, 1); continue; }
      const x = (s.x + s.t * 0.35) * w, y = (s.y - s.t * 0.05) * h;
      const g = ctx.createLinearGradient(x - 120, y, x, y);
      g.addColorStop(0, "rgba(34,211,238,0)"); g.addColorStop(1, `rgba(34,211,238,${0.5 * (1 - s.t)})`);
      ctx.strokeStyle = g; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(x - 120, y); ctx.lineTo(x, y); ctx.stroke();
    }
    raf = requestAnimationFrame(frame);
  };
  const toggle = () => { cancelAnimationFrame(raf); if (!document.hidden) { last = performance.now(); raf = requestAnimationFrame(frame); } };
  document.addEventListener("visibilitychange", toggle); toggle();
}
