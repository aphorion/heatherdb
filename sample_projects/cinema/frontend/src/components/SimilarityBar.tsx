interface SimilarityBarProps {
  value: number
  label?: string
  color?: 'gold' | 'cream' | 'crimson'
}

export default function SimilarityBar({ value, label, color = 'gold' }: SimilarityBarProps) {
  const pct = Math.max(0, Math.min(100, value * 100))
  const barColor = {
    gold: 'bg-gold',
    cream: 'bg-cream/50',
    crimson: 'bg-crimson',
  }

  return (
    <div className="flex items-center gap-3">
      {label && (
        <span className="font-serif text-[12px] text-cream-dim italic shrink-0 w-28 truncate">{label}</span>
      )}
      <div className="flex-1 h-[2px] bg-cream/[0.04] rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-1000 ease-out ${barColor[color]}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="font-mono text-[10px] text-cream-muted w-8 text-right tabular-nums">
        {(value * 100).toFixed(0)}
      </span>
    </div>
  )
}
