/**
 * AttractorMap — canvas-based landscape view.
 *
 * Renders a HeatherDB collection's attractor population as soft glowing dots
 * in a normalized -1..1 coord space. Pannable + zoomable with mouse + wheel.
 * Highlighted points (e.g. the activations from the last `analyze` call) get
 * a ring + halo treatment so the operator can see "what fired" against the
 * full population.
 *
 * Designed for ~1k-100k points. Above that we'd want WebGL; this canvas
 * impl is plenty for the MVP and stays photo-friendly.
 */

import { useEffect, useRef, useState } from "react";
import type { ProjectionPoint, ActivatedLocation } from "@/lib/heather";

type Props = {
  points: ProjectionPoint[];
  activations?: ActivatedLocation[];
  accent?: string;
  className?: string;
};

export default function AttractorMap({
  points, activations = [], accent = "#a855f7", className = "",
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [view, setView] = useState({ scale: 1, panX: 0, panY: 0 });
  const dragRef = useRef<{ x: number; y: number } | null>(null);
  const [hover, setHover] = useState<ProjectionPoint | null>(null);

  // Quick lookup of activated cells for ring overlay.
  const lit = new Map<number, number>();
  for (const a of activations) lit.set(a.id, a.weight);

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));

    const resize = () => {
      const { width, height } = wrap.getBoundingClientRect();
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(wrap);

    const draw = () => {
      const W = canvas.width;
      const H = canvas.height;
      const cx = W / 2 + view.panX * dpr;
      const cy = H / 2 + view.panY * dpr;
      const radius = Math.min(W, H) * 0.42 * view.scale;

      ctx.clearRect(0, 0, W, H);

      // Soft frame circle so the boundary of the projection is visible.
      ctx.strokeStyle = `${accent}15`;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.stroke();

      // crosshair guide — very faint
      ctx.strokeStyle = "rgba(255,255,255,0.04)";
      ctx.beginPath();
      ctx.moveTo(cx - radius, cy); ctx.lineTo(cx + radius, cy);
      ctx.moveTo(cx, cy - radius); ctx.lineTo(cx, cy + radius);
      ctx.stroke();

      // Base population — small white dots, alpha scaled by weight.
      for (const p of points) {
        const px = cx + p.x * radius;
        const py = cy + p.y * radius;
        if (px < -10 || px > W + 10 || py < -10 || py > H + 10) continue;
        const a = 0.25 + Math.min(1, Math.max(0, p.weight)) * 0.55;
        ctx.fillStyle = `rgba(255,255,255,${a})`;
        ctx.beginPath();
        ctx.arc(px, py, 1.4 * dpr, 0, Math.PI * 2);
        ctx.fill();
      }

      // Activated cells — accent halo + bigger core.
      for (const p of points) {
        const w = lit.get(p.id);
        if (w == null) continue;
        const px = cx + p.x * radius;
        const py = cy + p.y * radius;
        const haloR = (8 + w * 22) * dpr;
        const grad = ctx.createRadialGradient(px, py, 0, px, py, haloR);
        const alpha = 0.25 + w * 0.6;
        grad.addColorStop(0, hexA(accent, alpha));
        grad.addColorStop(1, hexA(accent, 0));
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(px, py, haloR, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = "#fff";
        ctx.beginPath();
        ctx.arc(px, py, (2 + w * 1.5) * dpr, 0, Math.PI * 2);
        ctx.fill();
      }

      // Hover ring
      if (hover) {
        const px = cx + hover.x * radius;
        const py = cy + hover.y * radius;
        ctx.strokeStyle = "rgba(255,255,255,0.7)";
        ctx.lineWidth = 1 * dpr;
        ctx.beginPath();
        ctx.arc(px, py, 6 * dpr, 0, Math.PI * 2);
        ctx.stroke();
      }
    };

    let raf = 0;
    const loop = () => { draw(); raf = requestAnimationFrame(loop); };
    loop();

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
    // We intentionally redraw on every frame so view/hover changes don't need
    // to be tracked in deps.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [points, activations, view, accent, hover]);

  /* ─── interaction ──────────────────────────────────────────────────────── */

  const onMouseDown = (e: React.MouseEvent) => {
    dragRef.current = { x: e.clientX, y: e.clientY };
  };
  const onMouseMove = (e: React.MouseEvent) => {
    const wrap = wrapRef.current;
    if (!wrap) return;
    const r = wrap.getBoundingClientRect();
    const W = r.width, H = r.height;
    const cx = W / 2 + view.panX;
    const cy = H / 2 + view.panY;
    const radius = Math.min(W, H) * 0.42 * view.scale;

    if (dragRef.current) {
      const dx = e.clientX - dragRef.current.x;
      const dy = e.clientY - dragRef.current.y;
      dragRef.current = { x: e.clientX, y: e.clientY };
      setView((v) => ({ ...v, panX: v.panX + dx, panY: v.panY + dy }));
      return;
    }

    // Hover detection — closest point within 6px.
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    let nearest: ProjectionPoint | null = null;
    let bestD = 6 * 6;
    for (const p of points) {
      const px = cx + p.x * radius;
      const py = cy + p.y * radius;
      const d = (mx - px) ** 2 + (my - py) ** 2;
      if (d < bestD) { bestD = d; nearest = p; }
    }
    setHover(nearest);
  };
  const onMouseUp   = () => { dragRef.current = null; };
  const onWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
    setView((v) => ({ ...v, scale: clamp(v.scale * factor, 0.3, 12) }));
  };
  const reset = () => setView({ scale: 1, panX: 0, panY: 0 });

  return (
    <div
      ref={wrapRef}
      className={`relative h-full w-full select-none ${className}`}
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
      onMouseLeave={onMouseUp}
      onWheel={onWheel}
    >
      <canvas
        ref={canvasRef}
        className="block h-full w-full cursor-grab active:cursor-grabbing"
      />
      {/* Top-left meta */}
      <div className="absolute top-2 left-2 font-mono text-10 uppercase tracking-ops text-ink-muted">
        {points.length.toLocaleString()} attractors
        {activations.length > 0 && (
          <> · <span className="text-white">{activations.length} lit</span></>
        )}
      </div>
      {/* Top-right controls */}
      <div className="absolute top-2 right-2 flex items-center gap-1">
        <button onClick={() => setView((v) => ({ ...v, scale: clamp(v.scale * 1.2, 0.3, 12) }))} className="btn !h-7 !px-2">+</button>
        <button onClick={() => setView((v) => ({ ...v, scale: clamp(v.scale / 1.2, 0.3, 12) }))} className="btn !h-7 !px-2">−</button>
        <button onClick={reset} className="btn !h-7 !px-2">reset</button>
      </div>
      {/* Hover tooltip */}
      {hover && (
        <div className="absolute bottom-2 left-2 font-mono text-10 uppercase tracking-ops text-ink-muted bg-black/80 border border-ink-line px-2 py-1">
          id <span className="text-white">{hover.id}</span> ·
          weight <span className="text-white">{hover.weight.toFixed(3)}</span> ·
          ({hover.x.toFixed(2)}, {hover.y.toFixed(2)})
          {lit.has(hover.id) && (
            <> · <span className="text-accent">w={lit.get(hover.id)!.toFixed(3)}</span></>
          )}
        </div>
      )}
    </div>
  );
}

function clamp(v: number, lo: number, hi: number) { return Math.max(lo, Math.min(hi, v)); }

// "#a855f7" + alpha float → "#a855f7XX"
function hexA(hex: string, a: number): string {
  const v = Math.round(Math.max(0, Math.min(1, a)) * 255).toString(16).padStart(2, "0");
  return `${hex}${v}`;
}
