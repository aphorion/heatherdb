import { useEffect, useState, useCallback } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../api/client';
import type { StatsResponse } from '../api/types';

interface CollectionInfo {
  name: string;
  stats: StatsResponse | null;
}

export function DashboardPage() {
  const [healthy, setHealthy] = useState<boolean | null>(null);
  const [collections, setCollections] = useState<CollectionInfo[]>([]);
  const [autoRefresh, setAutoRefresh] = useState(false);

  const refresh = useCallback(async () => {
    try {
      await api.health();
      setHealthy(true);
    } catch {
      setHealthy(false);
      return;
    }

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

  useEffect(() => {
    if (!autoRefresh) return;
    const id = setInterval(refresh, 3000);
    return () => clearInterval(id);
  }, [autoRefresh, refresh]);

  return (
    <div className="space-y-10 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-white/5 pb-6">
        <div>
          <h1 className="text-3xl font-light tracking-tight text-white mb-1">System Overview</h1>
          <p className="text-xs text-text-secondary uppercase tracking-widest font-mono">HeatherDB Cortex Node v0.1</p>
        </div>

        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-bg-tertiary/50 border border-white/5">
            <div className={`w-2 h-2 rounded-full ${autoRefresh ? 'bg-green-500 animate-pulse' : 'bg-text-muted'}`} />
            <label className="text-[10px] font-medium text-text-secondary cursor-pointer hover:text-white transition-colors uppercase tracking-wider">
              <span className="mr-2">Live Data</span>
              <input
                type="checkbox"
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
                className="hidden"
              />
            </label>
          </div>

          <button onClick={refresh} className="p-2 rounded-full bg-white/5 hover:bg-white/10 text-text-secondary hover:text-white transition-colors">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M1 8a7 7 0 1 0 14 0" />
              <path d="M8 4v4l2 2" />
            </svg>
          </button>
        </div>
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-12 gap-6">
        {/* Status Module - Spans 4 cols */}
        <div className="col-span-12 md:col-span-4 lg:col-span-3 space-y-6">
          <div className="card h-full flex flex-col justify-between relative overflow-hidden group">
            <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
              <svg width="100" height="100" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z" /></svg>
            </div>

            <div>
              <p className="label mb-4">SYSTEM STATUS</p>
              <div className="flex items-center gap-3 mb-6">
                <div className={`w-3 h-3 rounded-full shadow-[0_0_10px] ${healthy ? 'bg-green-500 shadow-green-500/50' : 'bg-red-500 shadow-red-500/50'}`} />
                <span className="text-xl text-white font-medium tracking-tight">{healthy ? 'Operational' : 'System Error'}</span>
              </div>

              <div className="space-y-4">
                <div className="flex justify-between items-center text-xs">
                  <span className="text-text-muted">Uptime</span>
                  <span className="font-mono text-white">99.98%</span>
                </div>
                <div className="w-full bg-white/5 h-1 rounded-full overflow-hidden">
                  <div className="bg-green-500 w-[99%] h-full" />
                </div>

                <div className="flex justify-between items-center text-xs">
                  <span className="text-text-muted">Memory</span>
                  <span className="font-mono text-white">42%</span>
                </div>
                <div className="w-full bg-white/5 h-1 rounded-full overflow-hidden">
                  <div className="bg-blue-500 w-[42%] h-full" />
                </div>
              </div>
            </div>

            <div className="mt-8 pt-4 border-t border-white/5">
              <p className="text-[10px] text-text-muted font-mono">LAST HEARTBEAT: {new Date().toLocaleTimeString()}</p>
            </div>
          </div>
        </div>

        {/* Stats Modules - Spans 8 cols */}
        <div className="col-span-12 md:col-span-8 lg:col-span-9 grid grid-cols-1 sm:grid-cols-3 gap-6">
          <div className="card relative overflow-hidden group">
            <div className="absolute -right-4 -top-4 w-24 h-24 bg-blue-500/10 rounded-full blur-2xl group-hover:bg-blue-500/20 transition-colors" />
            <p className="label mb-2">ACTIVE COLLECTIONS</p>
            <p className="text-4xl font-light text-white tracking-tighter mb-1">{collections.length}</p>
            <p className="text-xs text-text-muted">Managed entities</p>
          </div>

          <div className="card relative overflow-hidden group">
            <div className="absolute -right-4 -top-4 w-24 h-24 bg-purple-500/10 rounded-full blur-2xl group-hover:bg-purple-500/20 transition-colors" />
            <p className="label mb-2">VECTOR LOCATIONS</p>
            <p className="text-4xl font-light text-white tracking-tighter mb-1">
              {collections.reduce((s, c) => s + (c.stats?.num_locations || 0), 0).toLocaleString()}
            </p>
            <p className="text-xs text-text-muted">Indexed points</p>
          </div>

          <div className="card relative overflow-hidden group">
            <div className="absolute -right-4 -top-4 w-24 h-24 bg-orange-500/10 rounded-full blur-2xl group-hover:bg-orange-500/20 transition-colors" />
            <p className="label mb-2">WRITE OPS</p>
            <p className="text-4xl font-light text-white tracking-tighter mb-1">
              {collections.reduce((s, c) => s + (c.stats?.total_writes || 0), 0).toLocaleString(undefined, { maximumFractionDigits: 0, notation: 'compact' })}
            </p>
            <p className="text-xs text-text-muted">Total commands</p>
          </div>

          {/* Expanded collection view if available */}
          <div className="col-span-1 sm:col-span-3 card bg-bg-card/30">
            <p className="label mb-4">DATA TOPOLOGY</p>

            {collections.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {collections.map((c) => (
                  <Link
                    key={c.name}
                    to={`/collections/${c.name}`}
                    className="group flex flex-col p-4 rounded-lg bg-bg-tertiary/30 hover:bg-white/5 border border-transparent hover:border-white/10 transition-all cursor-pointer"
                  >
                    <div className="flex justify-between items-start mb-3">
                      <span className="text-sm font-medium text-white group-hover:text-blue-400 transition-colors truncate">
                        {c.name}
                      </span>
                      <div className="w-1.5 h-1.5 rounded-full bg-green-500 shadow-[0_0_5px_rgba(34,197,94,0.5)]" />
                    </div>

                    {c.stats ? (
                      <div className="grid grid-cols-2 gap-y-2 gap-x-4 text-[10px] text-text-muted font-mono">
                        <div className="flex justify-between">
                          <span>LOCS</span>
                          <span className="text-text-secondary">{c.stats.num_locations}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>WRITES</span>
                          <span className="text-text-secondary">{c.stats.total_writes.toFixed(0)}</span>
                        </div>
                        <div className="col-span-2 mt-1">
                          <div className="w-full bg-white/5 h-0.5 rounded-full overflow-hidden">
                            <div className="bg-white/20 w-1/2 h-full" style={{ width: `${Math.min(100, (c.stats.avg_write_count / 1000) * 100)}%` }} />
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className="mt-auto text-[10px] text-status-error">Offline</div>
                    )}
                  </Link>
                ))}
              </div>
            ) : (
              <div className="text-center py-10 opacity-50">
                <p className="text-sm text-text-secondary">No active data nodes detected.</p>
                <Link to="/collections" className="text-xs text-blue-400 hover:text-blue-300 mt-2 inline-block">Initialize Node +</Link>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
