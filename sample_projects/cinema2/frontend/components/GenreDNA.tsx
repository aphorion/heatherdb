"use client";

import type { GenreDNA as GenreDNAType } from "@/lib/types";

export default function GenreDNA({ dna }: { dna: GenreDNAType }) {
  if (!dna) return null;
  const entries = Object.entries(dna)
    .filter(([, v]) => v > 5)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 10);

  if (!entries.length) return null;

  const max = entries[0][1];

  return (
    <div className="space-y-2">
      <h3 className="font-heading text-sm font-bold text-film-gold uppercase tracking-wider">
        Your Taste DNA
      </h3>
      {entries.map(([genre, value]) => (
        <div key={genre} className="flex items-center gap-3">
          <span className="text-sm text-film-cream w-24 text-right">
            {genre}
          </span>
          <div className="flex-1 h-4 bg-film-dark rounded-full overflow-hidden">
            <div
              className="h-full rounded-full bg-gradient-to-r from-film-gold to-film-gold-bright transition-all duration-500"
              style={{ width: `${(value / max) * 100}%` }}
            />
          </div>
          <span className="text-xs text-film-muted w-12">{value}%</span>
        </div>
      ))}
    </div>
  );
}
