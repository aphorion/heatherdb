"use client";

interface Props {
  profile1: string;
  profile2: string;
  compatibility: number;
}

export default function CompatibilityMeter({
  profile1,
  profile2,
  compatibility,
}: Props) {
  const color =
    compatibility >= 70
      ? "from-green-500 to-green-400"
      : compatibility >= 40
        ? "from-film-gold to-film-gold-bright"
        : "from-orange-500 to-orange-400";

  return (
    <div className="flex items-center gap-4">
      <div className="w-10 h-10 rounded-full bg-film-gold/20 flex items-center justify-center text-film-gold font-heading font-bold text-sm">
        {profile1[0].toUpperCase()}
      </div>

      <div className="flex-1">
        <div className="h-3 bg-film-dark rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full bg-gradient-to-r ${color} transition-all duration-700`}
            style={{ width: `${compatibility}%` }}
          />
        </div>
        <p className="text-center text-sm text-film-cream mt-1">
          {compatibility}% compatible
        </p>
      </div>

      <div className="w-10 h-10 rounded-full bg-film-accent/20 flex items-center justify-center text-film-accent font-heading font-bold text-sm">
        {profile2[0].toUpperCase()}
      </div>
    </div>
  );
}
