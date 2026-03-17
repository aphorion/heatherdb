"use client";

import { useEffect, useState } from "react";
import type { LandscapePoint, TrajectoryPoint } from "@/lib/api";

interface LandscapeProps {
  concepts: LandscapePoint[];
  trajectory: TrajectoryPoint[];
  stimulus: string;
}

const SIZE = 500;
const PAD = 40;

function toScreen(val: number): number {
  return PAD + ((val + 1) / 2) * (SIZE - 2 * PAD);
}

export default function Landscape({ concepts, trajectory, stimulus }: LandscapeProps) {
  const [visibleSegments, setVisibleSegments] = useState(0);

  useEffect(() => {
    setVisibleSegments(0);
    if (trajectory.length === 0) return;
    let seg = 0;
    const timer = setInterval(() => {
      seg++;
      setVisibleSegments(seg);
      if (seg >= trajectory.length) clearInterval(timer);
    }, 400);
    return () => clearInterval(timer);
  }, [trajectory]);

  if (concepts.length === 0) return null;

  return (
    <div className="relative">
      <svg
        viewBox={`0 0 ${SIZE} ${SIZE}`}
        className="w-full max-w-[500px] mx-auto"
      >
        {/* background */}
        <rect width={SIZE} height={SIZE} fill="transparent" />

        {/* concept dots */}
        {concepts.map((c) => {
          const x = toScreen(c.x);
          const y = toScreen(c.y);
          const isStimulus = c.name === stimulus.toLowerCase();
          const isOnPath = trajectory.some(
            (t) =>
              Math.abs(toScreen(t.x) - x) < 15 &&
              Math.abs(toScreen(t.y) - y) < 15
          );

          return (
            <g key={c.name}>
              <circle
                cx={x}
                cy={y}
                r={isStimulus ? 5 : isOnPath ? 4 : 2.5}
                fill={
                  isStimulus
                    ? "#fafafa"
                    : isOnPath
                    ? "#8b5cf6"
                    : "#3f3f46"
                }
                opacity={isOnPath || isStimulus ? 1 : 0.5}
              />
              <text
                x={x}
                y={y - 8}
                textAnchor="middle"
                className="text-[9px] font-[family-name:var(--font-geist-mono)]"
                fill={
                  isStimulus
                    ? "#fafafa"
                    : isOnPath
                    ? "#a78bfa"
                    : "#52525b"
                }
              >
                {c.name}
              </text>
            </g>
          );
        })}

        {/* trajectory line */}
        {trajectory.slice(0, visibleSegments).map((point, i) => {
          if (i === 0) return null;
          const prev = trajectory[i - 1];
          return (
            <line
              key={i}
              x1={toScreen(prev.x)}
              y1={toScreen(prev.y)}
              x2={toScreen(point.x)}
              y2={toScreen(point.y)}
              stroke="#8b5cf6"
              strokeWidth={Math.max(1, 3 - i * 0.3)}
              opacity={Math.max(0.3, 1 - i * 0.08)}
              strokeLinecap="round"
            />
          );
        })}

        {/* trajectory points */}
        {trajectory.slice(0, visibleSegments).map((point, i) => {
          const isEnd = i === trajectory.length - 1 && visibleSegments >= trajectory.length;
          return (
            <circle
              key={`tp-${i}`}
              cx={toScreen(point.x)}
              cy={toScreen(point.y)}
              r={i === 0 ? 6 : isEnd ? 6 : 3}
              fill={i === 0 ? "#fafafa" : "#8b5cf6"}
              stroke={isEnd ? "#8b5cf6" : "none"}
              strokeWidth={isEnd ? 2 : 0}
              opacity={isEnd ? 1 : 0.8}
            />
          );
        })}
      </svg>
    </div>
  );
}
