import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Bioluminescent palette
        primary: {
          DEFAULT: "#0EA5E9", // Electric cyan
          dark: "#0284C7",
          light: "#38BDF8",
          glow: "rgba(14, 165, 233, 0.4)"
        },
        secondary: {
          DEFAULT: "#8B5CF6", // Deep violet
          dark: "#7C3AED",
          light: "#A78BFA",
          glow: "rgba(139, 92, 246, 0.3)"
        },
        accent: {
          DEFAULT: "#10B981", // Emerald
          dark: "#059669",
          light: "#34D399",
          glow: "rgba(16, 185, 129, 0.3)"
        },
        background: {
          DEFAULT: "#020617", // Deep space
          elevated: "#0F172A",
          card: "#1E1B4B"
        },
        surface: {
          DEFAULT: "rgba(30, 27, 75, 0.6)", // Glass violet
          secondary: "rgba(15, 23, 42, 0.8)",
          tertiary: "rgba(51, 65, 85, 0.4)"
        },
        text: {
          DEFAULT: "#F1F5F9",
          secondary: "#CBD5E1",
          tertiary: "#94A3B8",
          muted: "#64748B"
        }
      },
      borderRadius: {
        card: "24px",
        btn: "16px",
        blob: "60% 40% 30% 70% / 60% 30% 70% 40%"
      },
      fontFamily: {
        // Ultra-modern display font
        display: ["'Orbitron'", "'Exo 2'", "sans-serif"],
        // Clean body font
        body: ["'DM Sans'", "system-ui", "sans-serif"],
        // Technical mono
        mono: ["'Fira Code'", "'JetBrains Mono'", "monospace"]
      },
      boxShadow: {
        glow: "0 0 20px rgba(14, 165, 233, 0.3), 0 0 40px rgba(139, 92, 246, 0.2)",
        "glow-strong": "0 0 30px rgba(14, 165, 233, 0.5), 0 0 60px rgba(139, 92, 246, 0.3)",
        inner: "inset 0 2px 4px rgba(0, 0, 0, 0.3)",
        card: "0 8px 32px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.05)"
      },
      animation: {
        "pulse-slow": "pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "float": "float 6s ease-in-out infinite",
        "shimmer": "shimmer 2s linear infinite",
        "glow-pulse": "glow-pulse 3s ease-in-out infinite",
        "morph": "morph 8s ease-in-out infinite"
      },
      keyframes: {
        float: {
          "0%, 100%": { transform: "translateY(0px)" },
          "50%": { transform: "translateY(-20px)" }
        },
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" }
        },
        "glow-pulse": {
          "0%, 100%": {
            opacity: "1",
            filter: "brightness(1) drop-shadow(0 0 10px currentColor)"
          },
          "50%": {
            opacity: "0.8",
            filter: "brightness(1.2) drop-shadow(0 0 20px currentColor)"
          }
        },
        morph: {
          "0%, 100%": { borderRadius: "60% 40% 30% 70% / 60% 30% 70% 40%" },
          "50%": { borderRadius: "30% 60% 70% 40% / 50% 60% 30% 60%" }
        }
      },
      backdropBlur: {
        xs: "2px"
      }
    }
  },
  plugins: []
} satisfies Config;
