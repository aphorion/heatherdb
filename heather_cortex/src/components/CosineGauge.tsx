interface CosineGaugeProps {
  value: number;
  size?: number;
  label?: string;
}

export function CosineGauge({ value, size = 120, label }: CosineGaugeProps) {
  const radius = (size - 12) / 2;
  const circumference = Math.PI * radius; // semicircle
  const clampedValue = Math.max(-1, Math.min(1, value));
  const normalized = (clampedValue + 1) / 2; // map [-1,1] to [0,1]
  const offset = circumference * (1 - normalized);

  const color = clampedValue > 0.9 ? '#22c55e' : clampedValue > 0.5 ? '#eab308' : '#ef4444';

  return (
    <div className="flex flex-col items-center gap-1">
      <svg width={size} height={size / 2 + 10} viewBox={`0 0 ${size} ${size / 2 + 10}`}>
        {/* Background arc */}
        <path
          d={`M 6 ${size / 2 + 4} A ${radius} ${radius} 0 0 1 ${size - 6} ${size / 2 + 4}`}
          fill="none"
          stroke="#222222"
          strokeWidth="6"
          strokeLinecap="round"
        />
        {/* Value arc */}
        <path
          d={`M 6 ${size / 2 + 4} A ${radius} ${radius} 0 0 1 ${size - 6} ${size / 2 + 4}`}
          fill="none"
          stroke={color}
          strokeWidth="6"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          className="transition-all duration-500"
        />
        <text
          x={size / 2}
          y={size / 2}
          textAnchor="middle"
          className="fill-text-primary text-lg font-semibold"
          style={{ fontSize: size / 6 }}
        >
          {clampedValue.toFixed(4)}
        </text>
      </svg>
      {label && <p className="label">{label}</p>}
    </div>
  );
}
