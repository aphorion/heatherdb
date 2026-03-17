import { useEffect, useState, useCallback } from 'react';
import { api } from '../api/client';
import { VectorInput } from '../components/VectorInput';

interface WriteRecord {
  id: number;
  collection: string;
  count: number;
  vectorCount: number;
  timestamp: Date;
  docIds?: number[];
}

export function WritePlayground() {
  const [collections, setCollections] = useState<string[]>([]);
  const [selected, setSelected] = useState('');
  const [batchCount, setBatchCount] = useState('1');
  const [writing, setWriting] = useState(false);
  const [error, setError] = useState('');
  const [history, setHistory] = useState<WriteRecord[]>([]);
  const [nextId, setNextId] = useState(0);

  // JSON batch mode
  const [jsonMode, setJsonMode] = useState(false);
  const [jsonInput, setJsonInput] = useState('');
  const [metadataInput, setMetadataInput] = useState('');

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

  const writeVector = async (vector: number[]) => {
    if (!selected) return;
    setError('');
    setWriting(true);
    try {
      const n = Math.max(1, parseInt(batchCount) || 1);
      const vectors = Array.from({ length: n }, () => vector);
      const res = await api.write(selected, vectors);
      setHistory((prev) => [
        { id: nextId, collection: selected, count: res.count, vectorCount: n, timestamp: new Date() },
        ...prev.slice(0, 19),
      ]);
      setNextId((id) => id + 1);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Write failed');
    } finally {
      setWriting(false);
    }
  };

  const writeJson = async () => {
    if (!selected || !jsonInput.trim()) return;
    setError('');
    setWriting(true);
    try {
      const parsed = JSON.parse(jsonInput);
      const vectors: number[][] = Array.isArray(parsed[0]) ? parsed : [parsed];
      const metadata = metadataInput.trim() ? JSON.parse(metadataInput) : undefined;
      const res = await api.write(selected, vectors, metadata);
      setHistory((prev) => [
        { id: nextId, collection: selected, count: res.count, vectorCount: vectors.length, timestamp: new Date(), docIds: res.ids },
        ...prev.slice(0, 19),
      ]);
      setNextId((id) => id + 1);
      setJsonInput('');
      setMetadataInput('');
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Write failed');
    } finally {
      setWriting(false);
    }
  };

  return (
    <div className="space-y-8">
      <h1 className="text-2xl font-bold tracking-tight">Write Playground</h1>

      {/* Collection Selector */}
      <div>
        <p className="label mb-2">COLLECTION</p>
        <select
          value={selected}
          onChange={(e) => setSelected(e.target.value)}
          className="input-field w-full max-w-xs"
        >
          <option value="">Select collection...</option>
          {collections.map((c) => (
            <option key={c} value={c}>{c}</option>
          ))}
        </select>
      </div>

      {/* Mode Toggle */}
      <div className="flex gap-1">
        <button
          onClick={() => setJsonMode(false)}
          className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
            !jsonMode ? 'bg-white text-black' : 'bg-bg-tertiary text-text-secondary hover:text-text-primary'
          }`}
        >
          VECTOR INPUT
        </button>
        <button
          onClick={() => setJsonMode(true)}
          className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
            jsonMode ? 'bg-white text-black' : 'bg-bg-tertiary text-text-secondary hover:text-text-primary'
          }`}
        >
          JSON BATCH
        </button>
      </div>

      {error && <p className="text-status-error text-xs">{error}</p>}

      {!jsonMode ? (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div className="space-y-4">
            <div>
              <p className="label mb-2">REPEAT COUNT</p>
              <input
                type="number"
                value={batchCount}
                onChange={(e) => setBatchCount(e.target.value)}
                className="input-field w-24"
                min="1"
              />
            </div>
            <VectorInput
              label="VECTOR"
              onSubmit={writeVector}
            />
            {writing && <p className="text-xs text-text-muted">Writing...</p>}
          </div>

          {/* History */}
          <div>
            <p className="label mb-3">RECENT WRITES</p>
            {history.length === 0 ? (
              <p className="text-xs text-text-muted">No writes yet</p>
            ) : (
              <div className="space-y-2">
                {history.map((h) => (
                  <div key={h.id} className="card py-3">
                    <div className="flex justify-between text-xs">
                      <span className="text-text-secondary">
                        <span className="font-mono text-text-primary">{h.collection}</span>
                        {' '}&middot; {h.vectorCount} vector{h.vectorCount > 1 ? 's' : ''}
                        {h.docIds && <span className="text-text-muted"> &middot; docs [{h.docIds.join(', ')}]</span>}
                      </span>
                      <span className="text-text-muted font-mono">
                        {h.count} written
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          <p className="label">VECTORS JSON</p>
          <textarea
            value={jsonInput}
            onChange={(e) => setJsonInput(e.target.value)}
            className="input-field w-full h-48 resize-y"
            placeholder={'[[0.1, -0.3, 0.5, ...], [0.2, 0.4, -0.1, ...]]\nor\n[0.1, -0.3, 0.5, ...] (single vector)'}
          />
          <p className="label">METADATA JSON <span className="text-text-muted font-normal">(optional)</span></p>
          <textarea
            value={metadataInput}
            onChange={(e) => setMetadataInput(e.target.value)}
            className="input-field w-full h-32 resize-y"
            placeholder={'[{"title": "Doc A"}, {"title": "Doc B"}]\nOne object per vector. Enables document storage.'}
          />
          <button
            onClick={writeJson}
            disabled={writing || !selected || !jsonInput.trim()}
            className="btn-primary text-xs"
          >
            {writing ? 'Writing...' : 'Write'}
          </button>
        </div>
      )}
    </div>
  );
}
