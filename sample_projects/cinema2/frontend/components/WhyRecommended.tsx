"use client";

import Image from "next/image";
import type { Influence } from "@/lib/types";

export default function WhyRecommended({
  influences,
}: {
  influences: Influence[];
}) {
  if (!influences.length) return null;

  return (
    <div>
      <h3 className="font-heading text-sm font-bold text-film-gold uppercase tracking-wider mb-3">
        Why This Was Recommended
      </h3>
      <div className="space-y-3">
        {influences.slice(0, 5).map((inf) => {
          const hasPoster =
            inf.poster && inf.poster !== "N/A" && inf.poster.startsWith("http");
          return (
            <div key={inf.imdb_id} className="flex items-center gap-3">
              <div className="w-10 h-14 rounded overflow-hidden bg-film-card flex-shrink-0 relative">
                {hasPoster ? (
                  <Image
                    src={inf.poster}
                    alt={inf.title}
                    fill
                    sizes="40px"
                    className="object-cover"
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center text-[8px] text-film-muted">
                    {inf.title.slice(0, 8)}
                  </div>
                )}
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm text-film-cream truncate">{inf.title}</p>
                <div className="h-2 bg-film-dark rounded-full overflow-hidden mt-1">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-film-gold to-film-gold-bright"
                    style={{ width: `${inf.contribution}%` }}
                  />
                </div>
              </div>
              <span className="text-xs text-film-muted flex-shrink-0">
                {inf.contribution}%
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
