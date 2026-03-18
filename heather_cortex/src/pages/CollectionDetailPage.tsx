import { useEffect, useState, useCallback } from 'react';
import { useParams, Link } from 'react-router-dom';
import { api } from '../api/client';
import type { StatsResponse, EAMConfig, DocumentItem } from '../api/types';
import { StatCard } from '../components/StatCard';
import { ConfigTable } from '../components/ConfigTable';

export function CollectionDetailPage() {
  const { name } = useParams<{ name: string }>();
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [config, setConfig] = useState<EAMConfig | null>(null);
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [docsExpanded, setDocsExpanded] = useState(false);
  const [error, setError] = useState('');

  const refresh = useCallback(async () => {
    if (!name) return;
    setError('');
    try {
      const [s, c, d] = await Promise.all([
        api.stats(name),
        api.config(name),
        api.getDocuments(name).catch(() => ({ documents: [] })),
      ]);
      setStats(s);
      setConfig(c.config);
      setDocuments(d.documents);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load');
    }
  }, [name]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  if (!name) return null;

  return (
    <div className="space-y-8">
      <div className="flex items-center gap-3">
        <Link to="/collections" className="text-text-muted hover:text-text-secondary transition-colors">
          Collections
        </Link>
        <span className="text-text-muted">/</span>
        <h1 className="text-2xl font-bold tracking-tight">{name}</h1>
        <button onClick={refresh} className="btn-secondary text-xs ml-auto">
          Refresh
        </button>
      </div>

      {error && (
        <div className="card border-red-900/50">
          <p className="text-status-error text-sm">{error}</p>
        </div>
      )}

      {/* Stats */}
      {stats && (
        <div>
          <p className="label mb-3">STATISTICS</p>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
            <StatCard
              label="Locations"
              value={stats.num_locations}
            />
            <StatCard
              label="Total Writes"
              value={stats.total_writes.toFixed(0)}
            />
            <StatCard
              label="Learning Rate"
              value={stats.current_eta.toFixed(6)}
            />
            <StatCard
              label="Avg Write Count"
              value={stats.avg_write_count.toFixed(2)}
            />
            <StatCard
              label="Max Write Count"
              value={stats.max_write_count.toFixed(2)}
            />
          </div>
        </div>
      )}

      {/* Documents */}
      {documents.length > 0 && (
        <div>
          <div className="flex items-center gap-3 mb-3">
            <p className="label">DOCUMENTS</p>
            <span className="text-xs text-text-muted font-mono">{documents.length} stored</span>
            <button
              onClick={() => setDocsExpanded(!docsExpanded)}
              className="btn-secondary text-xs ml-auto"
            >
              {docsExpanded ? 'Collapse' : 'Expand'}
            </button>
          </div>
          {docsExpanded && (
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {documents.map((doc) => (
                <div key={doc.id} className="card py-3">
                  <div className="flex items-start justify-between gap-4">
                    <span className="text-xs font-mono text-text-muted shrink-0">#{doc.id}</span>
                    <pre className="text-xs text-text-secondary flex-1 overflow-x-auto whitespace-pre-wrap">
                      {JSON.stringify(doc.metadata, null, 2)}
                    </pre>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Location Heatmap Link */}
      {stats && stats.num_locations > 0 && (
        <div>
          <p className="label mb-3">VISUALIZATIONS</p>
          <Link
            to={`/collections/${name}/heatmap`}
            className="card inline-flex items-center gap-3 hover:border-border transition-colors"
          >
            <div className="w-8 h-8 bg-bg-tertiary rounded flex items-center justify-center">
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
                <rect x="1" y="1" width="4" height="4" />
                <rect x="6" y="1" width="4" height="4" />
                <rect x="11" y="1" width="4" height="4" />
                <rect x="1" y="6" width="4" height="4" />
                <rect x="6" y="6" width="4" height="4" />
                <rect x="11" y="6" width="4" height="4" />
              </svg>
            </div>
            <div>
              <p className="text-sm font-medium">Location Heatmap</p>
              <p className="text-xs text-text-muted">Write counts, distributions, scatter plots</p>
            </div>
          </Link>
        </div>
      )}

      {/* Configuration */}
      {config && (
        <div>
          <p className="label mb-3">CONFIGURATION</p>
          <ConfigTable config={config} />
        </div>
      )}
    </div>
  );
}
