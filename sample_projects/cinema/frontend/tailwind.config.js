/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        film: {
          black: '#0f0e0c',
          dark: '#1a1816',
          card: '#221f1c',
          elevated: '#2c2825',
          border: '#332f2a',
        },
        gold: {
          DEFAULT: '#e8b34b',
          dim: '#b8923f',
          bright: '#f5c96a',
          muted: 'rgba(232,179,75,0.15)',
        },
        crimson: {
          DEFAULT: '#c75c5c',
          dim: '#9e4a4a',
          bright: '#e07070',
        },
        cream: {
          DEFAULT: '#ede8df',
          dim: '#8a8279',
          muted: '#5c5650',
        },
      },
      fontFamily: {
        display: ['Syne', 'system-ui', 'sans-serif'],
        serif: ['Crimson Pro', 'Georgia', 'serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      borderRadius: {
        '2xl': '1rem',
        '3xl': '1.5rem',
        '4xl': '2rem',
      },
    },
  },
  plugins: [],
}
