/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./src/**/*.{html,ts}'],
  // Umschaltung per `dark`-Klasse auf <html> (ThemeService).
  darkMode: 'class',
  theme: {
    extend: {
      // Farben als CSS-Variablen -> light/dark wechseln in styles.css,
      // bestehende Utilities (bg-bg, text-ink, border-edge …) färben automatisch um.
      colors: {
        bg: 'var(--c-bg)',
        panel: 'var(--c-panel)',
        panel2: 'var(--c-panel2)',
        ink: 'var(--c-ink)',
        muted: 'var(--c-muted)',
        accent: 'var(--c-accent)',
        onaccent: 'var(--c-on-accent)',
        userbubble: 'var(--c-userbubble)',
        edge: 'var(--c-edge)',
      },
    },
  },
  plugins: [],
};
