/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./src/**/*.{html,ts}'],
  theme: {
    extend: {
      colors: {
        bg: '#0f1115',
        panel: '#171a21',
        panel2: '#1e222b',
        ink: '#e6e8ec',
        muted: '#9aa3b2',
        accent: '#6ea8fe',
        userbubble: '#24405f',
        edge: '#2a2f3a',
      },
    },
  },
  plugins: [],
};
