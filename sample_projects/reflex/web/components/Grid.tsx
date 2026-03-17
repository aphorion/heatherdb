"use client";

import { useCallback } from "react";
import type { Grid as GridType } from "@/lib/api";

interface GridProps {
  grid: GridType;
  onChange?: (grid: GridType) => void;
  readOnly?: boolean;
  overlay?: GridType | null;
  confidence?: number[][] | null;
}

const CELL_SIZE = 52;
const GAP = 2;
const SIZE = 8;

export default function Grid({
  grid,
  onChange,
  readOnly,
  overlay,
  confidence,
}: GridProps) {
  const handleClick = useCallback(
    (r: number, c: number) => {
      if (readOnly || !onChange) return;
      const next = grid.map((row) => [...row]);
      next[r][c] = next[r][c] === 1 ? 0 : 1;
      onChange(next);
    },
    [grid, onChange, readOnly]
  );

  const handleDrag = useCallback(
    (r: number, c: number, e: React.MouseEvent) => {
      if (readOnly || !onChange || e.buttons !== 1) return;
      const next = grid.map((row) => [...row]);
      next[r][c] = 1;
      onChange(next);
    },
    [grid, onChange, readOnly]
  );

  return (
    <div
      className="inline-grid select-none"
      style={{
        gridTemplateColumns: `repeat(${SIZE}, ${CELL_SIZE}px)`,
        gap: `${GAP}px`,
      }}
    >
      {grid.map((row, r) =>
        row.map((cell, c) => {
          const isOverlay = overlay?.[r]?.[c] === 1 && cell === 0;
          const conf = confidence?.[r]?.[c] ?? 0;
          const isFilled = cell === 1;

          let bg: string;
          let border: string;

          if (isFilled) {
            bg = "bg-zinc-100";
            border = "border-zinc-300";
          } else if (isOverlay) {
            const opacity = Math.max(0.3, Math.min(1, conf));
            bg = "";
            border = "border-violet-500/50";
            return (
              <div
                key={`${r}-${c}`}
                className={`rounded-md border-2 ${border} transition-all duration-300`}
                style={{
                  width: CELL_SIZE,
                  height: CELL_SIZE,
                  backgroundColor: `rgba(139, 92, 246, ${opacity})`,
                }}
                title={`SDM: ${Math.round(conf * 100)}%`}
              />
            );
          } else {
            bg = "bg-zinc-900";
            border = "border-zinc-800";
          }

          return (
            <div
              key={`${r}-${c}`}
              className={`rounded-md border-2 ${bg} ${border} transition-all duration-150 ${
                readOnly ? "" : "cursor-pointer hover:border-zinc-500"
              }`}
              style={{ width: CELL_SIZE, height: CELL_SIZE }}
              onClick={() => handleClick(r, c)}
              onMouseEnter={(e) => handleDrag(r, c, e)}
            />
          );
        })
      )}
    </div>
  );
}

export function emptyGrid(): GridType {
  return Array.from({ length: SIZE }, () => Array(SIZE).fill(0));
}
