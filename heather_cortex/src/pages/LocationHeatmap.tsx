import { useEffect, useState, useCallback, useRef } from 'react';
import { useParams, Link } from 'react-router-dom';
import { api } from '../api/client';
import type { LocationSummaryItem } from '../api/types';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  ScatterChart, Scatter, Cell,
} from 'recharts';

export function LocationHeatmap() {
  const { name } = useParams<{ name: string }>();
  const [locations, setLocations] = useState<LocationSummaryItem[]>([]);
  const [error, setError] = useState('');
  const [hoveredLoc, setHoveredLoc] = useState<LocationSummaryItem | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const refresh = useCallback(async () => {
    if (!name) return;
    setError('');
    try {
      const res = await api.locations(name);
      setLocations(res.locations);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load');
    }
  }, [name]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // Draw heatmap grid on canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || locations.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const maxWrites = Math.max(...locations.map(l => l.write_count), 1);
    const cols = Math.ceil(Math.sqrt(locations.length));
    const rows = Math.ceil(locations.length / cols);
    const cellSize = Math.max(4, Math.min(20, Math.floor(600 / cols)));

    canvas.width = cols * cellSize;
    canvas.height = rows * cellSize;

    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    locations.forEach((loc, i) => {
      const col = i % cols;
      const row = Math.floor(i / cols);
      const intensity = loc.write_count / maxWrites;
      const brightness = Math.floor(intensity * 220 + 15);
      ctx.fillStyle = `rgb(${brightness}, ${brightness}, ${brightness})`;
      ctx.fillRect(col * cellSize + 1, row * cellSize + 1, cellSize - 2, cellSize - 2);
    });
  }, [locations]);

  const handleCanvasMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || locations.length === 0) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const cols = Math.ceil(Math.sqrt(locations.length));
    const cellSize = Math.max(4, Math.min(20, Math.floor(600 / cols)));
    const col = Math.floor(x / cellSize);
    const row = Math.floor(y / cellSize);
    const idx = row * cols + col;

    if (idx >= 0 && idx < locations.length) {
      setHoveredLoc(locations[idx]);
    } else {
      setHoveredLoc(null);
    }
  };

  // Write count distribution histogram
  const writeCountBins = useCallback(() => {
    if (locations.length === 0) return [];
    const maxWrites = Math.max(...locations.map(l => l.write_count));
    if (maxWrites === 0) return [{ range: '0', count: locations.length }];
    const binCount = Math.min(20, Math.ceil(Math.sqrt(locations.length)));
    const binWidth = maxWrites / binCount;
    const bins = Array.from({ length: binCount }, (_, i) => ({
      range: `${(i * binWidth).toFixed(1)}`,
      count: 0,
    }));
    locations.forEach(l => {
      const idx = Math.min(binCount - 1, Math.floor(l.write_count / binWidth));
      bins[idx].count++;
    });
    return bins;
  }, [locations]);

  // Magnitude distribution
  const magnitudeBins = useCallback(() => {
    if (locations.length === 0) return [];
    const maxMag = Math.max(...locations.map(l => l.avg_counter_magnitude));
    if (maxMag === 0) return [{ range: '0', count: locations.length }];
    const binCount = Math.min(20, Math.ceil(Math.sqrt(locations.length)));
    const binWidth = maxMag / binCount;
    const bins = Array.from({ length: binCount }, (_, i) => ({
      range: `${(i * binWidth).toFixed(3)}`,
      count: 0,
    }));
    locations.forEach(l => {
      const idx = Math.min(binCount - 1, Math.floor(l.avg_counter_magnitude / binWidth));
      bins[idx].count++;
    });
    return bins;
  }, [locations]);

  if (!name) return null;

  return (
    <div className="space-y-8">
      <div className="flex items-center gap-3">
        <Link to={`/collections/${name}`} className="text-text-muted hover:text-text-secondary transition-colors">
          {name}
        </Link>
        <span className="text-text-muted">/</span>
        <h1 className="text-2xl font-bold tracking-tight">Location Heatmap</h1>
        <button onClick={refresh} className="btn-secondary text-xs ml-auto">
          Refresh
        </button>
      </div>

      {error && (
        <div className="card border-red-900/50">
          <p className="text-status-error text-sm">{error}</p>
        </div>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
        {/* Heatmap Canvas */}
        <div>
          <p className="label mb-3">WRITE COUNT HEATMAP ({locations.length} locations)</p>
          <div className="card relative">
            <canvas
              ref={canvasRef}
              onMouseMove={handleCanvasMove}
              onMouseLeave={() => setHoveredLoc(null)}
              className="w-full cursor-crosshair"
              style={{ imageRendering: 'pixelated' }}
            />
            {hoveredLoc && (
              <div className="absolute top-2 right-2 card py-2 px-3 text-xs space-y-1">
                <div className="flex justify-between gap-4">
                  <span className="text-text-muted">Location</span>
                  <span className="font-mono">{hoveredLoc.id}</span>
                </div>
                <div className="flex justify-between gap-4">
                  <span className="text-text-muted">Writes</span>
                  <span className="font-mono">{hoveredLoc.write_count.toFixed(2)}</span>
                </div>
                <div className="flex justify-between gap-4">
                  <span className="text-text-muted">Avg Mag</span>
                  <span className="font-mono">{hoveredLoc.avg_counter_magnitude.toFixed(4)}</span>
                </div>
              </div>
            )}
          </div>
          <p className="text-xs text-text-muted mt-2">
            Brighter = more writes. Hover for details.
          </p>
        </div>

        {/* Scatter plot: write count vs magnitude */}
        <div>
          <p className="label mb-3">WRITE COUNT vs MAGNITUDE</p>
          <div className="card">
            <ResponsiveContainer width="100%" height={300}>
              <ScatterChart margin={{ top: 10, right: 10, bottom: 20, left: 10 }}>
                <XAxis
                  type="number"
                  dataKey="write_count"
                  name="Write Count"
                  tick={{ fill: '#555', fontSize: 10 }}
                  axisLine={{ stroke: '#333' }}
                  tickLine={false}
                  label={{ value: 'Write Count', fill: '#555', fontSize: 10, position: 'bottom' }}
                />
                <YAxis
                  type="number"
                  dataKey="avg_counter_magnitude"
                  name="Avg Magnitude"
                  tick={{ fill: '#555', fontSize: 10 }}
                  axisLine={{ stroke: '#333' }}
                  tickLine={false}
                />
                <Tooltip
                  contentStyle={{ backgroundColor: '#111', border: '1px solid #333', borderRadius: 4, fontSize: 12 }}
                  labelStyle={{ color: '#888' }}
                  itemStyle={{ color: '#fff' }}
                  cursor={{ strokeDasharray: '3 3', stroke: '#333' }}
                />
                <Scatter data={locations}>
                  {locations.map((_, i) => (
                    <Cell key={i} fill="#ffffff" fillOpacity={0.6} />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Distribution Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div>
          <p className="label mb-3">WRITE COUNT DISTRIBUTION</p>
          <div className="card">
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={writeCountBins()}>
                <XAxis
                  dataKey="range"
                  tick={{ fill: '#555', fontSize: 9 }}
                  axisLine={{ stroke: '#333' }}
                  tickLine={false}
                  interval="preserveStartEnd"
                />
                <YAxis
                  tick={{ fill: '#555', fontSize: 10 }}
                  axisLine={{ stroke: '#333' }}
                  tickLine={false}
                />
                <Tooltip
                  contentStyle={{ backgroundColor: '#111', border: '1px solid #333', borderRadius: 4, fontSize: 12 }}
                  labelStyle={{ color: '#888' }}
                  itemStyle={{ color: '#fff' }}
                />
                <Bar dataKey="count" fill="#ffffff" radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div>
          <p className="label mb-3">COUNTER MAGNITUDE DISTRIBUTION</p>
          <div className="card">
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={magnitudeBins()}>
                <XAxis
                  dataKey="range"
                  tick={{ fill: '#555', fontSize: 9 }}
                  axisLine={{ stroke: '#333' }}
                  tickLine={false}
                  interval="preserveStartEnd"
                />
                <YAxis
                  tick={{ fill: '#555', fontSize: 10 }}
                  axisLine={{ stroke: '#333' }}
                  tickLine={false}
                />
                <Tooltip
                  contentStyle={{ backgroundColor: '#111', border: '1px solid #333', borderRadius: 4, fontSize: 12 }}
                  labelStyle={{ color: '#888' }}
                  itemStyle={{ color: '#fff' }}
                />
                <Bar dataKey="count" fill="#888888" radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
