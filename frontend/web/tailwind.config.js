/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        bg: '#080c10',
        bg2: '#0d1117',
        bg3: '#161b22',
        bg4: '#1c2333',
        border: '#21262d',
        border2: '#30363d',
        text: '#e6edf3',
        muted: '#7d8590',
        muted2: '#484f58',
        accent: '#58a6ff',
        accent2: '#1f6feb',
        green: '#3fb950',
        red: '#f85149',
        yellow: '#d29922',
        purple: '#bc8cff',
      },
      fontFamily: {
        sans: ['"IBM Plex Sans"', 'sans-serif'],
        mono: ['"IBM Plex Mono"', 'monospace'],
      },
      fontSize: {
        'xs-custom': '10px',
        'sm-custom': '11px',
        'base-custom': '13px',
      },
    },
  },
  plugins: [],
}
