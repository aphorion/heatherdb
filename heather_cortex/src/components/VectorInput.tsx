import { useState } from 'react';

interface VectorInputProps {
  dimensions?: number;
  onSubmit: (vector: number[]) => void;
  label?: string;
}

type InputMode = 'json' | 'random';

export function VectorInput({ dimensions, onSubmit, label }: VectorInputProps) {
  const [mode, setMode] = useState<InputMode>('random');
  const [jsonValue, setJsonValue] = useState('');
  const [randomDims, setRandomDims] = useState(dimensions?.toString() || '64');
  const [error, setError] = useState('');

  const generateRandom = () => {
    const d = parseInt(randomDims) || 64;
    const vec = Array.from({ length: d }, () => {
      // Gaussian via Box-Muller
      const u1 = Math.random();
      const u2 = Math.random();
      return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    });
    // Normalize
    const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0));
    return vec.map(v => v / norm);
  };

  const handleSubmit = () => {
    setError('');
    try {
      if (mode === 'random') {
        onSubmit(generateRandom());
      } else {
        const parsed = JSON.parse(jsonValue);
        if (!Array.isArray(parsed) || !parsed.every((v: unknown) => typeof v === 'number')) {
          throw new Error('Must be an array of numbers');
        }
        onSubmit(parsed);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Invalid input');
    }
  };

  return (
    <div className="space-y-3">
      {label && <p className="label">{label}</p>}
      <div className="flex gap-1">
        {(['random', 'json'] as const).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
              mode === m
                ? 'bg-white text-black'
                : 'bg-bg-tertiary text-text-secondary hover:text-text-primary'
            }`}
          >
            {m.toUpperCase()}
          </button>
        ))}
      </div>

      {mode === 'random' && (
        <div className="flex items-center gap-2">
          <input
            type="number"
            value={randomDims}
            onChange={(e) => setRandomDims(e.target.value)}
            className="input-field w-24"
            placeholder="dims"
            min="1"
            disabled={!!dimensions}
          />
          <span className="text-text-muted text-xs">dimensions</span>
        </div>
      )}

      {mode === 'json' && (
        <textarea
          value={jsonValue}
          onChange={(e) => setJsonValue(e.target.value)}
          className="input-field w-full h-24 resize-y"
          placeholder="[0.1, -0.3, 0.5, ...]"
        />
      )}

      {error && <p className="text-status-error text-xs">{error}</p>}

      <button onClick={handleSubmit} className="btn-primary text-xs">
        {mode === 'random' ? 'Generate' : 'Use Vector'}
      </button>
    </div>
  );
}
