import { useEffect, useState, useCallback } from 'react';
import { api } from '../api/client';
import type { AnalyzeResponse } from '../api/types';
import { VectorInput } from '../components/VectorInput';
import { CosineGauge } from '../components/CosineGauge';

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

export function ReadPlayground() {
  const [collections, setCollections] = useState<string[]>([]);
  const [selected, setSelected] = useState('');
  const [strategy, setStrategy] = useState<'iterative' | 'fast'>('iterative');
  const [analyzeMode, setAnalyzeMode] = useState(false);
  const [compareMode, setCompareMode] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const [query, setQuery] = useState<number[] | null>(null);
  const [resultIterative, setResultIterative] = useState<number[] | null>(null);
  const [resultFast, setResultFast] = useState<number[] | null>(null);
  const [analysis, setAnalysis] = useState<AnalyzeResponse | null>(null);

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

  const handleRead = async (vec: number[]) => {
    if (!selected) return;
    setError('');
    setLoading(true);
    setQuery(vec);
    setResultIterative(null);
    setResultFast(null);
    setAnalysis(null);

    try {
      if (analyzeMode) {
        const res = await api.analyze(selected, vec, strategy);
        setAnalysis(res);
        if (strategy === 'iterative') setResultIterative(res.result);
        else setResultFast(res.result);
      } else if (compareMode) {
        const [iter, fast] = await Promise.all([
          api.read(selected, vec, 'iterative'),
          api.read(selected, vec, 'fast'),
        ]);
        setResultIterative(iter.result);
        setResultFast(fast.result);
      } else {
        const res = await api.read(selected, vec, strategy);
        if (strategy === 'iterative') setResultIterative(res.result);
        else setResultFast(res.result);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Read failed');
    } finally {
      setLoading(false);
    }
  };

  const activeResult = strategy === 'iterative' ? resultIterative : resultFast;

  return (
    <div className="space-y-8">
      <h1 className="text-2xl font-bold tracking-tight">Read Playground</h1>

      {/* Controls */}
      <div className="flex flex-wrap items-end gap-6">
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

        <div>
          <p className="label mb-2">STRATEGY</p>
          <div className="flex gap-1">
            {(['iterative', 'fast'] as const).map((s) => (
              <button
                key={s}
                onClick={() => setStrategy(s)}
                className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${
                  strategy === s
                    ? 'bg-white text-black'
                    : 'bg-bg-tertiary text-text-secondary hover:text-text-primary'
                }`}
              >
                {s.toUpperCase()}
              </button>
            ))}
          </div>
        </div>

        <div>
          <p className="label mb-2">MODE</p>
          <div className="flex gap-1">
            <button
              onClick={() => { setAnalyzeMode(false); setCompareMode(false); }}
              className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${
                !analyzeMode && !compareMode
                  ? 'bg-white text-black'
                  : 'bg-bg-tertiary text-text-secondary hover:text-text-primary'
              }`}
            >
              READ
            </button>
            <button
              onClick={() => { setCompareMode(true); setAnalyzeMode(false); }}
              className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${
                compareMode
                  ? 'bg-white text-black'
                  : 'bg-bg-tertiary text-text-secondary hover:text-text-primary'
              }`}
            >
              COMPARE
            </button>
            <button
              onClick={() => { setAnalyzeMode(true); setCompareMode(false); }}
              className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${
                analyzeMode
                  ? 'bg-white text-black'
                  : 'bg-bg-tertiary text-text-secondary hover:text-text-primary'
              }`}
            >
              ANALYZE
            </button>
          </div>
        </div>
      </div>

      {error && <p className="text-status-error text-xs">{error}</p>}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Input */}
        <div>
          <VectorInput label="QUERY VECTOR" onSubmit={handleRead} />
          {loading && <p className="text-xs text-text-muted mt-2">Reading...</p>}
        </div>

        {/* Results */}
        <div className="space-y-6">
          {/* Similarity gauges */}
          {query && (activeResult || compareMode) && (
            <div>
              <p className="label mb-3">SIMILARITY (QUERY vs RESULT)</p>
              <div className="flex gap-8">
                {resultIterative && (
                  <CosineGauge
                    value={cosineSimilarity(query, resultIterative)}
                    label={compareMode ? 'Iterative' : undefined}
                  />
                )}
                {resultFast && (
                  <CosineGauge
                    value={cosineSimilarity(query, resultFast)}
                    label={compareMode ? 'Fast' : undefined}
                  />
                )}
              </div>
            </div>
          )}

          {/* Result vector preview */}
          {activeResult && !compareMode && (
            <div>
              <p className="label mb-2">RESULT VECTOR ({activeResult.length}d)</p>
              <div className="card font-mono text-xs text-text-secondary overflow-x-auto max-h-32 overflow-y-auto">
                [{activeResult.map((v) => v.toFixed(6)).join(', ')}]
              </div>
            </div>
          )}

          {/* Compare results */}
          {compareMode && resultIterative && resultFast && (
            <div>
              <p className="label mb-2">INTER-STRATEGY SIMILARITY</p>
              <CosineGauge
                value={cosineSimilarity(resultIterative, resultFast)}
                label="Iterative vs Fast"
              />
            </div>
          )}

          {/* Analyze details */}
          {analysis && (
            <div className="space-y-4">
              <p className="label">ANALYSIS</p>
              <div className="grid grid-cols-2 gap-4">
                <div className="card py-3">
                  <p className="label mb-1">Iterations</p>
                  <p className="text-lg font-semibold font-mono">{analysis.iterations}</p>
                </div>
                <div className="card py-3">
                  <p className="label mb-1">Converged</p>
                  <p className="text-lg font-semibold">
                    <span className={`w-2 h-2 rounded-full inline-block mr-2 ${analysis.converged ? 'bg-status-ok' : 'bg-status-error'}`} />
                    {analysis.converged ? 'Yes' : 'No'}
                  </p>
                </div>
              </div>

              <div>
                <p className="label mb-2">ACTIVATED LOCATIONS ({analysis.activated_locations.length})</p>
                <div className="overflow-hidden rounded-lg border border-border-subtle">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="border-b border-border-subtle bg-bg-tertiary">
                        <th className="text-left px-3 py-2 label">ID</th>
                        <th className="text-right px-3 py-2 label">Similarity</th>
                        <th className="text-right px-3 py-2 label">Weight</th>
                      </tr>
                    </thead>
                    <tbody>
                      {analysis.activated_locations.map((al) => (
                        <tr key={al.id} className="border-b border-border-subtle last:border-0">
                          <td className="px-3 py-1.5 font-mono">{al.id}</td>
                          <td className="px-3 py-1.5 text-right font-mono">{al.similarity.toFixed(6)}</td>
                          <td className="px-3 py-1.5 text-right font-mono">
                            <div className="flex items-center justify-end gap-2">
                              <div className="w-16 bg-bg-tertiary rounded-full h-1">
                                <div
                                  className="h-1 rounded-full bg-white"
                                  style={{ width: `${Math.min(100, al.weight * 100)}%` }}
                                />
                              </div>
                              {al.weight.toFixed(4)}
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
