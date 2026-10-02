/** Spec 9.2: the four IMD warning colours are the ONLY saturated colours on
 *  the MAP. The accent family below is for everything that is not warning
 *  state -- tabs, regime bands, the atlas, charts -- and is deliberately cool
 *  so it cannot be mistaken for a warning at a glance. */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        ink: '#10151C', slate2: '#44505F', mist: '#8593A4',
        paper: '#F4F6F8', card: '#FFFFFF', rule: '#E2E7EC', sunk: '#EDF1F5',
        warn: { green: '#1B8A3F', yellow: '#F2C200', orange: '#F07C00', red: '#D31F26' },
        accent: { DEFAULT: '#2B5CE6', deep: '#1B3FA8', soft: '#EAF0FE' },
        teal: '#0E8F8F', violet: '#6B4EC7', indigo: '#3A4FC4', cyan: '#1B87C9',
      },
      fontFamily: {
        sans: ['"Source Sans 3"', 'system-ui', 'sans-serif'],
        display: ['"Source Serif 4"', 'Georgia', 'serif'],
      },
    },
  },
  plugins: [],
}
