import { useEffect, useState, useCallback } from 'react';
import { api } from '../api/client';
import { VectorInput } from '../components/VectorInput';
import { CosineGauge } from '../components/CosineGauge';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length || a.length === 0) return 0;
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom < 1e-12 ? 0 : dot / denom;
}

function generateRandomVector(d: number): number[] {
  const vec = Array.from({ length: d }, () => {
    const u1 = Math.random();
    const u2 = Math.random();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  });
  const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0));
  return vec.map(v => v / norm);
}

interface FidelityResult {
  index: number;
  similarity: number;
}

export function SimilarityExplorer() {
  const [collections, setCollections] = useState<string[]>([]);
  const [selected, setSelected] = useState('');

  // Pair similarity
  const [vecA, setVecA] = useState<number[] | null>(null);
  const [vecB, setVecB] = useState<number[] | null>(null);

  // Fidelity test
  const [fidelityDims, setFidelityDims] = useState('64');
  const [fidelityCount, setFidelityCount] = useState('20');
  const [fidelityResults, setFidelityResults] = useState<FidelityResult[]>([]);
  const [fidelityRunning, setFidelityRunning] = useState(false);
  const [fidelityError, setFidelityError] = useState('');

  // Write-readback
  const [writebackResult, setWritebackResult] = useState<number | null>(null);
  const [writebackLoading, setWritebackLoading] = useState(false);

  const loadCollections = useCallback(async () => {
    try {
      const res = await api.listCollections();
      setCollections(res.collections);
      if (res.collections.length > 0 && !selected) {
        setSelected(res.collections[0]);
      }
    } catch { /* ignore */ }
  }, [selected]);

  useEffect(() => {
    loadCollections();
  }, [loadCollections]);

  const pairSim = vecA && vecB && vecA.length === vecB.length
    ? cosineSimilarity(vecA, vecB)
    : null;

  const handleWriteReadback = async (vec: number[]) => {
    if (!selected) return;
    setWritebackLoading(true);
    setWritebackResult(null);
    try {
      await api.write(selected, [vec]);
      const res = await api.read(selected, vec, 'iterative');
      setWritebackResult(cosineSimilarity(vec, res.result));
    } catch { /* ignore */ }
    setWritebackLoading(false);
  };

  const runFidelityTest = async () => {
    if (!selected) return;
    setFidelityRunning(true);
    setFidelityError('');
    setFidelityResults([]);

    const d = parseInt(fidelityDims) || 64;
    const n = parseInt(fidelityCount) || 20;
    const results: FidelityResult[] = [];

    try {
      for (let i = 0; i < n; i++) {
        const vec = generateRandomVector(d);
        await api.write(selected, [vec]);
        const res = await api.read(selected, vec, 'iterative');
        const sim = cosineSimilarity(vec, res.result);
        results.push({ index: i, similarity: sim });
        setFidelityResults([...results]);
      }
    } catch (e) {
      setFidelityError(e instanceof Error ? e.message : 'Test failed');
    } finally {
      setFidelityRunning(false);
    }
  };

  const avgFidelity = fidelityResults.length > 0
    ? fidelityResults.reduce((s, r) => s + r.similarity, 0) / fidelityResults.length
    : null;

  return (
    <div className="space-y-10">
      <h1 className="text-2xl font-bold tracking-tight">Similarity Explorer</h1>

      {/* Pair Similarity */}
      <div>
        <p className="label mb-4">COSINE SIMILARITY</p>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div>
            <VectorInput label="VECTOR A" onSubmit={setVecA} />
            {vecA && <p className="text-xs text-text-muted mt-1">{vecA.length}d vector set</p>}
          </div>
          <div>
            <VectorInput label="VECTOR B" onSubmit={setVecB} />
            {vecB && <p className="text-xs text-text-muted mt-1">{vecB.length}d vector set</p>}
          </div>
          <div className="flex items-center justify-center">
            {pairSim !== null ? (
              <CosineGauge value={pairSim} size={160} label="Cosine Similarity" />
            ) : (
              <p className="text-text-muted text-xs">Set both vectors to compare</p>
            )}
          </div>
        </div>
      </div>

      {/* Write-Readback */}
      <div>
        <p className="label mb-4">WRITE-READBACK FIDELITY</p>
        <div className="flex flex-wrap items-end gap-4 mb-4">
          <div>
            <p className="label mb-2">COLLECTION</p>
            <select
              value={selected}
              onChange={(e) => setSelected(e.target.value)}
              className="input-field"
            >
              <option value="">Select...</option>
              {collections.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div>
            <VectorInput label="WRITE & READ BACK" onSubmit={handleWriteReadback} />
            {writebackLoading && <p className="text-xs text-text-muted mt-1">Testing...</p>}
          </div>
          <div className="flex items-center justify-center">
            {writebackResult !== null && (
              <CosineGauge value={writebackResult} size={160} label="Fidelity" />
            )}
          </div>
        </div>
      </div>

      {/* Batch Fidelity Test */}
      <div>
        <p className="label mb-4">BATCH FIDELITY TEST</p>
        <div className="flex flex-wrap items-end gap-4 mb-4">
          <div>
            <p className="label mb-2">DIMENSIONS</p>
            <input
              type="number"
              value={fidelityDims}
              onChange={(e) => setFidelityDims(e.target.value)}
              className="input-field w-24"
              min="1"
            />
          </div>
          <div>
            <p className="label mb-2">VECTOR COUNT</p>
            <input
              type="number"
              value={fidelityCount}
              onChange={(e) => setFidelityCount(e.target.value)}
              className="input-field w-24"
              min="1"
              max="100"
            />
          </div>
          <button
            onClick={runFidelityTest}
            disabled={fidelityRunning || !selected}
            className="btn-primary text-xs"
          >
            {fidelityRunning ? `Running... (${fidelityResults.length}/${fidelityCount})` : 'Run Test'}
          </button>
        </div>

        {fidelityError && <p className="text-status-error text-xs mb-2">{fidelityError}</p>}

        {fidelityResults.length > 0 && (
          <div className="card">
            <div className="flex items-center justify-between mb-4">
              <p className="text-sm text-text-secondary">
                {fidelityResults.length} vectors tested
              </p>
              {avgFidelity !== null && (
                <p className="text-sm">
                  <span className="text-text-muted">avg fidelity</span>{' '}
                  <span className="font-mono font-semibold">{avgFidelity.toFixed(4)}</span>
                </p>
              )}
            </div>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={fidelityResults}>
                <XAxis
                  dataKey="index"
                  tick={{ fill: '#555', fontSize: 10 }}
                  axisLine={{ stroke: '#333' }}
                  tickLine={false}
                />
                <YAxis
                  domain={[0, 1]}
                  tick={{ fill: '#555', fontSize: 10 }}
                  axisLine={{ stroke: '#333' }}
                  tickLine={false}
                />
                <Tooltip
                  contentStyle={{ backgroundColor: '#111', border: '1px solid #333', borderRadius: 4, fontSize: 12 }}
                  labelStyle={{ color: '#888' }}
                  itemStyle={{ color: '#fff' }}
                />
                <Bar dataKey="similarity" fill="#ffffff" radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  );
}
