"use client";

import { useState } from "react";
import { useProfile } from "@/hooks/useProfile";
import { blendProfiles } from "@/lib/api";
import type { BlendResult } from "@/lib/types";
import CompatibilityMeter from "@/components/CompatibilityMeter";
import GenreDNA from "@/components/GenreDNA";
import MovieRow from "@/components/MovieRow";

export default function BlendPage() {
  const { profiles } = useProfile();
  const [p1, setP1] = useState("");
  const [p2, setP2] = useState("");
  const [result, setResult] = useState<BlendResult | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleBlend() {
    if (!p1 || !p2 || p1 === p2) return;
    setLoading(true);
    try {
      const res = await blendProfiles(p1, p2);
      setResult(res);
    } catch {
      // error
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <h1 className="font-heading text-3xl font-bold text-film-cream mb-2">
        Movie Night
      </h1>
      <p className="text-film-muted mb-8">
        Find movies both of you will enjoy
      </p>

      <div className="flex flex-wrap items-end gap-4 mb-8">
        <div>
          <label className="text-xs text-film-muted block mb-1">Profile 1</label>
          <select
            value={p1}
            onChange={(e) => setP1(e.target.value)}
            className="bg-film-card border border-film-border rounded px-3 py-2 text-sm text-film-cream focus:outline-none focus:border-film-gold"
          >
            <option value="">Select...</option>
            {profiles.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
        </div>

        <span className="text-film-muted text-lg pb-2">&</span>

        <div>
          <label className="text-xs text-film-muted block mb-1">Profile 2</label>
          <select
            value={p2}
            onChange={(e) => setP2(e.target.value)}
            className="bg-film-card border border-film-border rounded px-3 py-2 text-sm text-film-cream focus:outline-none focus:border-film-gold"
          >
            <option value="">Select...</option>
            {profiles
              .filter((p) => p !== p1)
              .map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
          </select>
        </div>

        <button
          onClick={handleBlend}
          disabled={!p1 || !p2 || p1 === p2 || loading}
          className="bg-film-gold text-film-black font-heading font-bold px-6 py-2 rounded hover:bg-film-gold-bright transition disabled:opacity-50"
        >
          {loading ? "Blending..." : "Find Movies"}
        </button>
      </div>

      {result && (
        <div className="space-y-8">
          <div className="bg-film-card border border-film-border rounded-xl p-6">
            <CompatibilityMeter
              profile1={p1}
              profile2={p2}
              compatibility={result.compatibility}
            />
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-film-card border border-film-border rounded-xl p-6">
              <p className="text-sm text-film-muted mb-3">{p1}&apos;s DNA</p>
              <GenreDNA dna={result.profile1_dna} />
            </div>
            <div className="bg-film-card border border-film-border rounded-xl p-6">
              <p className="text-sm text-film-muted mb-3">{p2}&apos;s DNA</p>
              <GenreDNA dna={result.profile2_dna} />
            </div>
          </div>

          <MovieRow
            title="Movies You'll Both Enjoy"
            movies={result.recommendations}
          />
        </div>
      )}
    </div>
  );
}
