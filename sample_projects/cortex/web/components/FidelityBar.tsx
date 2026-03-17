"use client";

interface FidelityBarProps {
  value: number;
  label?: string;
  size?: "sm" | "md" | "lg";
}

function getColor(v: number): string {
  if (v >= 0.7) return "bg-emerald-500";
  if (v >= 0.4) return "bg-amber-500";
  return "bg-rose-500";
}

function getGlow(v: number): string {
  if (v >= 0.7) return "shadow-emerald-500/30";
  if (v >= 0.4) return "shadow-amber-500/30";
  return "shadow-rose-500/30";
}

export default function FidelityBar({ value, label, size = "md" }: FidelityBarProps) {
  const pct = Math.round(value * 100);
  const heights = { sm: "h-1.5", md: "h-2.5", lg: "h-4" };

  return (
    <div className="w-full">
      {label && (
        <div className="flex justify-between items-center mb-1">
          <span className="text-xs text-zinc-400 uppercase tracking-wider">{label}</span>
          <span className="text-xs font-mono text-zinc-300">{pct}%</span>
        </div>
      )}
      <div className={`w-full ${heights[size]} bg-zinc-800 rounded-full overflow-hidden`}>
        <div
          className={`${heights[size]} ${getColor(value)} rounded-full transition-all duration-700 ease-out shadow-lg ${getGlow(value)}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
