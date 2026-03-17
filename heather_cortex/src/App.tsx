import { Routes, Route, NavLink, Navigate } from 'react-router-dom';
import { useEffect, useState } from 'react';
import { api } from './api/client';
import { DashboardPage } from './pages/DashboardPage';
import { CollectionBrowserPage } from './pages/CollectionBrowserPage';
import { CollectionDetailPage } from './pages/CollectionDetailPage';
import { WritePlayground } from './pages/WritePlayground';
import { ReadPlayground } from './pages/ReadPlayground';
import { SimilarityExplorer } from './pages/SimilarityExplorer';
import { LocationHeatmap } from './pages/LocationHeatmap';

const NAV_ITEMS = [
  { path: '/', label: 'Dashboard', icon: DashboardIcon },
  { path: '/collections', label: 'Collections', icon: CollectionsIcon },
  { path: '/write', label: 'Write', icon: WriteIcon },
  { path: '/read', label: 'Read', icon: ReadIcon },
  { path: '/similarity', label: 'Similarity', icon: SimilarityIcon },
];

export default function App() {
  const [healthy, setHealthy] = useState<boolean | null>(null);

  useEffect(() => {
    const check = async () => {
      try {
        await api.health();
        setHealthy(true);
      } catch {
        setHealthy(false);
      }
    };
    check();
    const id = setInterval(check, 10000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="flex h-screen overflow-hidden bg-bg-primary text-text-primary selection:bg-accent selection:text-black">
      {/* Sidebar - Glassmorphism style */}
      <aside className="w-64 flex-shrink-0 bg-bg-card/20 backdrop-blur-xl border-r border-white/5 flex flex-col relative z-20">
        
        {/* Glow effect for sidebar */}
        <div className="absolute inset-0 bg-gradient-to-b from-white/5 to-transparent pointer-events-none" />

        {/* Logo */}
        <div className="relative px-6 py-8">
          <div className="flex items-center gap-4">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-white/20 to-transparent border border-white/10 flex items-center justify-center shadow-lg shadow-black/50">
              <div className="w-2 h-2 rounded-full bg-white shadow-[0_0_10px_2px_rgba(255,255,255,0.5)]" />
            </div>
            <div>
              <p className="text-sm font-bold tracking-wide text-white">HEATHER<span className="opacity-50 font-light">DB</span></p>
              <p className="text-[9px] text-text-muted uppercase tracking-[0.2em] mt-0.5">Cortex Control</p>
            </div>
          </div>
        </div>

        {/* Navigation */}
        <nav className="relative flex-1 px-4 space-y-1">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              end={item.path === '/'}
              className={({ isActive }) =>
                `group flex items-center gap-3 px-4 py-3 rounded-lg text-sm transition-all duration-300 ${
                  isActive
                    ? 'bg-white/10 text-white shadow-inner shadow-white/5 border border-white/5'
                    : 'text-text-secondary hover:text-white hover:bg-white/5'
                }`
              }
            >
              <item.icon />
              <span className="font-medium tracking-wide">{item.label}</span>
              {/* Active Indicator */}
              <div className={`ml-auto w-1 h-1 rounded-full bg-white shadow-[0_0_8px_rgba(255,255,255,0.8)] transition-opacity duration-300 ${
                 // We can't easily access isActive state here in the child cleanly without logic, so let's rely on CSS group-hover or cleaner component logic.
                 // Actually the NavLink render prop exposes isActive. Let's simplify.
                 ''
              }`} style={{ opacity: 0 }} /> 
            </NavLink>
          ))}
        </nav>

        {/* Server status */}
        <div className="relative px-6 py-6 mt-auto border-t border-white/5 bg-black/20">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <span className="relative flex h-2 w-2">
                 {healthy && <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-status-ok opacity-75"></span>}
                 <span
                  className={`relative inline-flex rounded-full h-2 w-2 ${
                    healthy === null
                      ? 'bg-text-muted'
                      : healthy
                      ? 'bg-status-ok'
                      : 'bg-status-error'
                  }`}
                />
              </span>
              <span className="text-[10px] font-medium text-text-secondary uppercase tracking-widest">
                {healthy === null ? 'SYNCING' : healthy ? 'ONLINE' : 'OFFLINE'}
              </span>
            </div>
            <span className="text-[10px] text-text-muted font-mono">v0.1.0</span>
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-y-auto relative">
         {/* Background gradients */}
         <div className="fixed inset-0 pointer-events-none z-0">
            <div className="absolute top-[-20%] right-[-10%] w-[600px] h-[600px] bg-white/5 rounded-full blur-[120px] opacity-50" />
            <div className="absolute bottom-[-10%] left-[-10%] w-[500px] h-[500px] bg-blue-500/5 rounded-full blur-[100px] opacity-30" />
         </div>

        <div className="relative z-10 max-w-7xl mx-auto px-10 py-10">
          <Routes>
            <Route path="/" element={<DashboardPage />} />
            <Route path="/collections" element={<CollectionBrowserPage />} />
            <Route path="/collections/:name" element={<CollectionDetailPage />} />
            <Route path="/collections/:name/heatmap" element={<LocationHeatmap />} />
            <Route path="/write" element={<WritePlayground />} />
            <Route path="/read" element={<ReadPlayground />} />
            <Route path="/similarity" element={<SimilarityExplorer />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </div>
      </main>
    </div>
  );
}

// Minimal SVG icons
function DashboardIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
      <rect x="1" y="1" width="6" height="6" rx="1" />
      <rect x="9" y="1" width="6" height="6" rx="1" />
      <rect x="1" y="9" width="6" height="6" rx="1" />
      <rect x="9" y="9" width="6" height="6" rx="1" />
    </svg>
  );
}

function CollectionsIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
      <rect x="2" y="3" width="12" height="10" rx="1" />
      <line x1="2" y1="6" x2="14" y2="6" />
      <line x1="2" y1="9" x2="14" y2="9" />
    </svg>
  );
}

function WriteIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path d="M8 3v10M3 8h10" />
    </svg>
  );
}

function ReadIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
      <circle cx="7" cy="7" r="4" />
      <line x1="10" y1="10" x2="14" y2="14" />
    </svg>
  );
}

function SimilarityIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
      <circle cx="5" cy="8" r="4" />
      <circle cx="11" cy="8" r="4" />
    </svg>
  );
}
