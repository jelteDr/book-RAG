/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./src/**/*.{html,ts}'],
  // Umschaltung per data-theme-Attribut auf <html> (ThemeService + Bootstrap-Script in index.html).
  darkMode: ['selector', '[data-theme="dark"]'],
  theme: {
    extend: {
      // Semantische Farben als CSS-Variablen -> hell/dunkel wechselt in styles.css.
      colors: {
        canvas: 'var(--canvas)',
        surface: { DEFAULT: 'var(--surface)', muted: 'var(--surface-muted)' },
        line: { DEFAULT: 'var(--line)', strong: 'var(--line-strong)', soft: 'var(--line-soft)' },
        ink: { DEFAULT: 'var(--ink)', muted: 'var(--ink-muted)' },
        accent: {
          DEFAULT: 'var(--accent)',
          hover: 'var(--accent-hover)',
          fg: 'var(--accent-fg)',
          soft: 'var(--accent-soft)',
        },
        danger: { DEFAULT: 'var(--danger)', soft: 'var(--danger-soft)' },
        info: { DEFAULT: 'var(--info)', soft: 'var(--info-soft)' },
        warn: { DEFAULT: 'var(--warn)', soft: 'var(--warn-soft)' },
        overlay: 'var(--overlay)',
        bar: 'var(--bar)',
        chart: { 1: 'var(--chart-1)', 2: 'var(--chart-2)' },
      },
      fontFamily: {
        sans: ['-apple-system', 'BlinkMacSystemFont', '"SF Pro Text"', 'system-ui', '"Segoe UI"', 'Roboto', 'sans-serif'],
        display: ['ui-rounded', '"SF Pro Rounded"', '-apple-system', 'BlinkMacSystemFont', 'system-ui', 'sans-serif'],
      },
      // Felder 10 px, Buttons/Banner 12 px (rounded-xl), Karten 16 px
      borderRadius: { sm: '10px', md: '16px' },
      boxShadow: { pop: 'var(--shadow-pop)', thumb: 'var(--shadow-thumb)' },
      animation: {
        rise: 'rise 0.24s var(--ease-out) both',
        fade: 'fade 0.18s ease-out both',
        drawer: 'drawer 0.28s var(--ease-out)',
      },
    },
  },
  plugins: [],
};
