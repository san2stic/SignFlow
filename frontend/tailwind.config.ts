import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        primary: "#6366F1",
        secondary: "#10B981",
        accent: "#F59E0B",
        background: "#0F172A",
        surface: "#1E293B",
        text: "#F8FAFC"
      },
      borderRadius: {
        card: "12px",
        btn: "8px"
      },
      fontFamily: {
        heading: ["'Plus Jakarta Sans'", "sans-serif"],
        body: ["Inter", "sans-serif"],
        mono: ["'JetBrains Mono'", "monospace"]
      }
    }
  },
  plugins: []
} satisfies Config;
