"use client";

import type { JourneyStep } from "@/lib/types";

const stabilityColor: Record<string, string> = {
  Stable: "text-green-400",
  Evolving: "text-film-gold",
  Shifting: "text-orange-400",
};

export default function TasteJourney({ journey }: { journey: JourneyStep[] }) {
  if (!journey.length) return null;

  return (
    <div>
      <h3 className="font-heading text-sm font-bold text-film-gold uppercase tracking-wider mb-3">
        Your Taste Journey
      </h3>
      <div className="space-y-3">
        {journey.map((step) => (
          <div
            key={step.step}
            className="flex items-start gap-3 border-l-2 border-film-border pl-4"
          >
            <span className="text-xs text-film-muted w-4 pt-0.5">
              {step.step}
            </span>
            <div className="flex-1">
              <p className="text-sm text-film-cream">{step.movie}</p>
              <span
                className={`text-xs ${stabilityColor[step.stability] || "text-film-muted"}`}
              >
                {step.stability}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
