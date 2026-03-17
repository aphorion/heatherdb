import { useEffect, useState, useCallback } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../api/client';
import type { StatsResponse } from '../api/types';

interface CollectionInfo {
  name: string;
  stats: StatsResponse | null;
}

export function CollectionBrowserPage() {
  const [collections, setCollections] = useState<CollectionInfo[]>([]);
  const [newName, setNewName] = useState('');
  const [creating, setCreating] = useState(false);
  const [dropping, setDropping] = useState<string | null>(null);
  const [error, setError] = useState('');

  const refresh = useCallback(async () => {
    try {
      const res = await api.listCollections();
      const infos = await Promise.all(
        res.collections.map(async (name) => {
          try {
            const stats = await api.stats(name);
            return { name, stats };
          } catch {
            return { name, stats: null };
          }
        })
      );
      setCollections(infos);
    } catch {
      setCollections([]);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const handleCreate = async () => {
    if (!newName.trim()) return;
    setError('');
    setCreating(true);
    try {
      await api.createCollection(newName.trim());
      setNewName('');
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to create');
    } finally {
      setCreating(false);
    }
  };

  const handleDrop = async (name: string) => {
    setError('');
    try {
      await api.dropCollection(name);
      setDropping(null);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to drop');
    }
  };

  return (
    <div className="space-y-8">
      <h1 className="text-2xl font-bold tracking-tight">Collections</h1>

      {/* Create Collection */}
      <div>
        <p className="label mb-3">CREATE NEW</p>
        <div className="flex gap-2">
          <input
            type="text"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleCreate()}
            className="input-field flex-1"
            placeholder="collection name"
          />
          <button
            onClick={handleCreate}
            disabled={creating || !newName.trim()}
            className="btn-primary text-xs"
          >
            {creating ? 'Creating...' : 'Create'}
          </button>
        </div>
        {error && <p className="text-status-error text-xs mt-2">{error}</p>}
      </div>

      {/* Collection Grid */}
      <div>
        <p className="label mb-3">ALL COLLECTIONS ({collections.length})</p>
        {collections.length === 0 ? (
          <div className="card text-center py-12">
            <p className="text-text-secondary">No collections found</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {collections.map((c) => (
              <div key={c.name} className="card group relative">
                <Link to={`/collections/${c.name}`} className="block">
                  <p className="font-medium text-text-primary group-hover:text-white transition-colors">
                    {c.name}
                  </p>
                  {c.stats && (
                    <div className="mt-3 space-y-1 text-xs text-text-secondary">
                      <div className="flex justify-between">
                        <span className="text-text-muted">Locations</span>
                        <span className="font-mono">{c.stats.num_locations}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-text-muted">Total Writes</span>
                        <span className="font-mono">{c.stats.total_writes.toFixed(0)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-text-muted">Avg Write Count</span>
                        <span className="font-mono">{c.stats.avg_write_count.toFixed(2)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-text-muted">Max Write Count</span>
                        <span className="font-mono">{c.stats.max_write_count.toFixed(2)}</span>
                      </div>
                    </div>
                  )}
                </Link>

                {/* Drop button */}
                <div className="mt-3 pt-3 border-t border-border-subtle">
                  {dropping === c.name ? (
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-status-error">Confirm drop?</span>
                      <button
                        onClick={() => handleDrop(c.name)}
                        className="text-xs text-status-error hover:underline"
                      >
                        Yes
                      </button>
                      <button
                        onClick={() => setDropping(null)}
                        className="text-xs text-text-muted hover:underline"
                      >
                        Cancel
                      </button>
                    </div>
                  ) : (
                    <button
                      onClick={() => setDropping(c.name)}
                      className="text-xs text-text-muted hover:text-status-error transition-colors"
                    >
                      Drop
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
