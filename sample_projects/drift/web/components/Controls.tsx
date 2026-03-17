"use client";

interface ControlsProps {
  alpha: number;
  beta: number;
  gamma: number;
  steps: number;
  onChange: (params: { alpha: number; beta: number; gamma: number; steps: number }) => void;
}

export default function Controls({ alpha, beta, gamma, steps, onChange }: ControlsProps) {
  return (
    <div className="space-y-3">
      <Slider
        label="α drift"
        sublabel="follow the reconstruction"
        value={alpha}
        onChange={(v) => onChange({ alpha: v, beta, gamma, steps })}
      />
      <Slider
        label="β anchor"
        sublabel="pull back to stimulus"
        value={beta}
        onChange={(v) => onChange({ alpha, beta: v, gamma, steps })}
      />
      <Slider
        label="γ context"
        sublabel="accumulate trajectory"
        value={gamma}
        onChange={(v) => onChange({ alpha, beta, gamma: v, steps })}
      />
      <div className="flex items-center justify-between">
        <span className="text-xs text-zinc-500">max steps</span>
        <select
          value={steps}
          onChange={(e) => onChange({ alpha, beta, gamma, steps: Number(e.target.value) })}
          className="bg-zinc-900 border border-zinc-800 rounded text-xs text-zinc-300 px-2 py-1"
        >
          {[6, 8, 12, 16, 20].map((n) => (
            <option key={n} value={n}>{n}</option>
          ))}
        </select>
      </div>
    </div>
  );
}

function Slider({
  label,
  sublabel,
  value,
  onChange,
}: {
  label: string;
  sublabel: string;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between items-baseline">
        <span className="text-xs text-zinc-400 font-[family-name:var(--font-geist-mono)]">
          {label}
          <span className="text-zinc-600 ml-1.5 font-sans">{sublabel}</span>
        </span>
        <span className="text-xs text-zinc-500 font-[family-name:var(--font-geist-mono)]">
          {value.toFixed(2)}
        </span>
      </div>
      <input
        type="range"
        min={0}
        max={1}
        step={0.05}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full h-1 bg-zinc-800 rounded-full appearance-none cursor-pointer accent-violet-500"
      />
    </div>
  );
}
