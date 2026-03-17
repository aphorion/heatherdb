/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: {
          primary: '#0a0a0a',
          secondary: '#111111',
          tertiary: '#1a1a1a',
          card: '#141414',
        },
        border: {
          subtle: '#222222',
          DEFAULT: '#333333',
        },
        text: {
          primary: '#ffffff',
          secondary: '#888888',
          muted: '#555555',
        },
        accent: {
          DEFAULT: '#ffffff',
          dim: '#666666',
        },
        status: {
          ok: '#22c55e',
          error: '#ef4444',
          warn: '#eab308',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-out',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
    },
  },
  plugins: [],
}
