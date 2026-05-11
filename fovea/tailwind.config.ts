import type { Config } from "tailwindcss";

// Same monochrome operator aesthetic as the HeatherDB landing site.
// Inter for prose, JetBrains Mono for tags / numerics. Very dark surface
// ladder, single accent purple, two semantic accents (green = ok,
// red = alarm).
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          DEFAULT: "#fff",
          dim: "rgba(255,255,255,0.65)",
          muted: "rgba(255,255,255,0.45)",
          ghost: "rgba(255,255,255,0.25)",
          line: "rgba(255,255,255,0.10)",
        },
        surface: {
          0: "#000",
          1: "#070708",
          2: "#0c0c0e",
          3: "#141417",
        },
        accent: {
          DEFAULT: "#a855f7",
          cool: "#26c6ff",
          warm: "#f59e0b",
          ok: "#4ade80",
          alarm: "#f87171",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
      fontSize: {
        // Operator UIs read better at small mono sizes — these are the
        // workhorses for tags, latencies, counts.
        "10": ["10px", { lineHeight: "14px" }],
        "11": ["11px", { lineHeight: "15px" }],
      },
      letterSpacing: {
        ops: "0.18em",
      },
      animation: {
        "pulse-soft": "pulse 2.4s cubic-bezier(0.4, 0, 0.6, 1) infinite",
      },
    },
  },
  plugins: [],
} satisfies Config;
