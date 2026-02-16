import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Premium cyan/teal palette
        primary: {
          DEFAULT: "#14B8A6", // teal
          dark: "#0D9488",
          light: "#5EEAD4",
          glow: "rgba(20, 184, 166, 0.32)"
        },
        secondary: {
          DEFAULT: "#06B6D4", // cyan
          dark: "#0891B2",
          light: "#67E8F9",
          glow: "rgba(6, 182, 212, 0.32)"
        },
        accent: {
          DEFAULT: "#F59E0B",
          dark: "#D97706",
          light: "#FCD34D",
          glow: "rgba(245, 158, 11, 0.3)"
        },
        background: {
          DEFAULT: "#020617",
          elevated: "#0B1224",
          card: "#0E172E"
        },
        surface: {
          DEFAULT: "rgba(15, 23, 42, 0.74)",
          secondary: "rgba(12, 20, 39, 0.88)",
          tertiary: "rgba(30, 41, 59, 0.5)"
        },
        text: {
          DEFAULT: "#F1F5F9",
          secondary: "#D1D5DB",
          tertiary: "#9CA3AF",
          muted: "#6B7280"
        }
      },
      borderRadius: {
        card: "20px",
        btn: "14px",
        blob: "60% 40% 30% 70% / 60% 30% 70% 40%"
      },
      fontFamily: {
        display: ["'Sora'", "'Orbitron'", "sans-serif"],
        heading: ["'Sora'", "'Orbitron'", "sans-serif"],
        body: ["'Manrope'", "'DM Sans'", "system-ui", "sans-serif"],
        mono: ["'JetBrains Mono'", "'Fira Code'", "monospace"]
      },
      boxShadow: {
        glow: "0 0 18px rgba(20, 184, 166, 0.25), 0 0 34px rgba(6, 182, 212, 0.2)",
        "glow-strong": "0 0 28px rgba(20, 184, 166, 0.35), 0 0 50px rgba(6, 182, 212, 0.25)",
        inner: "inset 0 2px 4px rgba(2, 6, 23, 0.45)",
        card: "0 12px 30px rgba(2, 6, 23, 0.45), inset 0 1px 0 rgba(148, 163, 184, 0.08)"
      },
      animation: {
        "pulse-slow": "pulse 3.5s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "float": "float 6s ease-in-out infinite",
        "shimmer": "shimmer 1.8s linear infinite",
        "glow-pulse": "glow-pulse 2.4s ease-in-out infinite",
        "morph": "morph 8s ease-in-out infinite"
      },
      keyframes: {
        float: {
          "0%, 100%": { transform: "translateY(0px)" },
          "50%": { transform: "translateY(-14px)" }
        },
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" }
        },
        "glow-pulse": {
          "0%, 100%": {
            opacity: "1",
            filter: "brightness(1) drop-shadow(0 0 8px currentColor)"
          },
          "50%": {
            opacity: "0.8",
            filter: "brightness(1.1) drop-shadow(0 0 14px currentColor)"
          }
        },
        morph: {
          "0%, 100%": { borderRadius: "60% 40% 30% 70% / 60% 30% 70% 40%" },
          "50%": { borderRadius: "30% 60% 70% 40% / 50% 60% 30% 60%" }
        }
      },
      backdropBlur: {
        xs: "2px"
      },
      maxWidth: {
        app: "1240px"
      }
    }
  },
  plugins: []
} satisfies Config;
