/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#FAFAFA',
        surface: '#FFFFFF',
        'surface-subtle': '#F4F4F6',
        'surface-hover': '#ECECF0',
        border: '#E4E4E7',
        'border-subtle': '#F0F0F2',
        primary: {
          50: '#F0F7FF',
          100: '#E0EFFF',
          200: '#B9DDFF',
          300: '#7CC0FF',
          400: '#389EFF',
          500: '#0070F3',
          600: '#0059C6',
          700: '#004399',
          800: '#00306E',
          900: '#00204A',
        },
        slate: {
          50: '#F8FAFC',
          100: '#F1F5F9',
          200: '#E2E8F0',
          300: '#CBD5E1',
          400: '#94A3B8',
          500: '#64748B',
          600: '#475569',
          700: '#334155',
          800: '#1E293B',
          900: '#0F172A',
        },
      },
      fontFamily: {
        sans: [
          'Inter',
          '-apple-system',
          'BlinkMacSystemFont',
          '"Segoe UI"',
          'Roboto',
          'Helvetica',
          'Arial',
          'sans-serif',
        ],
        mono: ['JetBrains Mono', 'Menlo', 'Monaco', 'Courier New', 'monospace'],
      },
      boxShadow: {
        'subtle': '0 1px 2px 0 rgba(0, 0, 0, 0.03), 0 1px 6px -1px rgba(0, 0, 0, 0.02)',
        'card': '0 1px 3px 0 rgba(0, 0, 0, 0.04), 0 2px 8px -2px rgba(0, 0, 0, 0.04)',
        'elevated': '0 4px 20px -2px rgba(0, 0, 0, 0.06), 0 2px 6px -1px rgba(0, 0, 0, 0.03)',
      },
    },
  },
  plugins: [],
}
