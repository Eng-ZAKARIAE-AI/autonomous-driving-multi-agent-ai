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
        background: "#0F172A",
        surface: "#1E293B",
        primary: {
          DEFAULT: "#6366F1",
          hover: "#4F46E5",
        },
        safety: {
          red: "#EF4444",
          green: "#10B981",
        },
        telemetry: "#F59E0B",
      },
    },
  },
  plugins: [],
}
