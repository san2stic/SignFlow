# 🎨 SignFlow Design System — Néo-Biomécanique Fluide

## Concept Créatif

**SignFlow** traduit le langage des signes, un langage vivant, gestuel et organique. L'interface reflète cette **fluidité biomécanique** : des formes organiques rencontrant la précision technologique, comme des mains humaines fusionnant avec l'IA.

### Direction Esthétique

- **Formes fluides** : Courbes organiques, morphismes, effets de liquidité
- **Palette bioluminescente** : Bleu électrique, cyan néon, violet profond, vert émeraude
- **Typographie contrastée** : Display ultramoderne (Orbitron) + Body clean (DM Sans) + Mono technique (Fira Code)
- **Mouvements organiques** : Animations fluides, ondulations, effets de particules
- **Profondeur tactile** : Glassmorphism, ombres profondes, effets de lueur

---

## 🎨 Palette de Couleurs

### Primaire — Cyan Électrique
```css
--primary: #0EA5E9
--primary-dark: #0284C7
--primary-light: #38BDF8
--primary-glow: rgba(14, 165, 233, 0.4)
```
Usage : Actions principales, liens, états actifs

### Secondaire — Violet Profond
```css
--secondary: #8B5CF6
--secondary-dark: #7C3AED
--secondary-light: #A78BFA
--secondary-glow: rgba(139, 92, 246, 0.3)
```
Usage : Accents, highlights, états secondaires

### Accent — Émeraude
```css
--accent: #10B981
--accent-dark: #059669
--accent-light: #34D399
--accent-glow: rgba(16, 185, 129, 0.3)
```
Usage : Succès, validations, états positifs

### Backgrounds
```css
--background: #020617        /* Deep space */
--background-elevated: #0F172A
--background-card: #1E1B4B
```

### Surfaces (Glassmorphism)
```css
--surface: rgba(30, 27, 75, 0.6)
--surface-secondary: rgba(15, 23, 42, 0.8)
--surface-tertiary: rgba(51, 65, 85, 0.4)
```

### Texte
```css
--text: #F1F5F9           /* Primary */
--text-secondary: #CBD5E1
--text-tertiary: #94A3B8
--text-muted: #64748B
```

---

## 📝 Typographie

### Display — Orbitron
```css
font-family: 'Orbitron', 'Exo 2', sans-serif
```
- **Usage** : Titres principaux, logos, textes emphase
- **Poids** : 500 (Medium), 600 (SemiBold), 700 (Bold), 800 (ExtraBold), 900 (Black)
- **Caractéristiques** : Futuriste, géométrique, ultra-moderne

### Body — DM Sans
```css
font-family: 'DM Sans', system-ui, sans-serif
```
- **Usage** : Corps de texte, labels, descriptions
- **Poids** : 400 (Regular), 500 (Medium), 600 (SemiBold), 700 (Bold)
- **Caractéristiques** : Clean, lisible, humaniste

### Mono — Fira Code
```css
font-family: 'Fira Code', 'JetBrains Mono', monospace
```
- **Usage** : Code, valeurs numériques, statuts techniques
- **Poids** : 400 (Regular), 500 (Medium), 600 (SemiBold)
- **Caractéristiques** : Technique, précis, avec ligatures

---

## 🔲 Composants de Base

### Cards — Glassmorphic
```css
.card {
  @apply rounded-card backdrop-blur-xl border;
  background: linear-gradient(145deg, rgba(30, 27, 75, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
  border-color: rgba(148, 163, 184, 0.1);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.05);
}
```

**Effet shimmer au hover** :
```css
.card::before {
  content: '';
  position: absolute;
  background: linear-gradient(90deg, transparent, rgba(14, 165, 233, 0.1), transparent);
  /* Animation de gauche à droite au hover */
}
```

### Buttons — Biomechanical Touch
```css
.touch-btn {
  @apply min-h-12 min-w-12 rounded-btn px-5 py-3 text-base font-medium;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1);
  /* Ripple effect au click */
}
```

**Variantes** :
- **Primary** : `bg-gradient-to-br from-primary to-secondary`
- **Accent** : `bg-gradient-to-br from-accent to-accent-dark`
- **Ghost** : `bg-gradient-to-br from-primary/30 to-secondary/30 backdrop-blur-sm`

### Border Radius
```css
--rounded-card: 24px   /* Cards, containers */
--rounded-btn: 16px    /* Buttons, inputs */
--rounded-blob: 60% 40% 30% 70% / 60% 30% 70% 40%  /* Morphing shapes */
```

---

## ✨ Effets Visuels

### Glow Effects
```css
/* Text glow */
.glow-text {
  text-shadow: 0 0 20px currentColor;
  animation: glow-pulse 3s ease-in-out infinite;
}

/* Box shadow glow */
box-shadow: 0 0 20px rgba(14, 165, 233, 0.3), 0 0 40px rgba(139, 92, 246, 0.2);
```

### Neon Border
```css
.neon-border {
  position: relative;
  border: 1px solid transparent;
}

.neon-border::before {
  background: linear-gradient(135deg, #0EA5E9, #8B5CF6, #10B981);
  /* Masking technique pour créer un border gradient */
}
```

### Shimmer Loading
```css
.shimmer {
  background: linear-gradient(90deg, rgba(148, 163, 184, 0.05), rgba(148, 163, 184, 0.15), rgba(148, 163, 184, 0.05));
  background-size: 200% 100%;
  animation: shimmer 2s linear infinite;
}
```

### Cyber Grid
```css
.cyber-grid {
  background-image:
    linear-gradient(rgba(14, 165, 233, 0.05) 1px, transparent 1px),
    linear-gradient(90deg, rgba(14, 165, 233, 0.05) 1px, transparent 1px);
  background-size: 30px 30px;
}
```

---

## 🎬 Animations

### Float
```css
@keyframes float {
  0%, 100% { transform: translateY(0px); }
  50% { transform: translateY(-20px); }
}
animation: float 6s ease-in-out infinite;
```

### Morph
```css
@keyframes morph {
  0%, 100% { border-radius: 60% 40% 30% 70% / 60% 30% 70% 40%; }
  50% { border-radius: 30% 60% 70% 40% / 50% 60% 30% 60%; }
}
animation: morph 8s ease-in-out infinite;
```

### Glow Pulse
```css
@keyframes glow-pulse {
  0%, 100% { opacity: 1; filter: brightness(1) drop-shadow(0 0 10px currentColor); }
  50% { opacity: 0.8; filter: brightness(1.2) drop-shadow(0 0 20px currentColor); }
}
animation: glow-pulse 3s ease-in-out infinite;
```

### Shimmer
```css
@keyframes shimmer {
  0% { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}
animation: shimmer 2s linear infinite;
```

---

## 🌟 Background Effects

### Organic Mesh Gradient
```css
body {
  background:
    radial-gradient(ellipse at 20% 30%, rgba(14, 165, 233, 0.15) 0%, transparent 50%),
    radial-gradient(ellipse at 80% 10%, rgba(139, 92, 246, 0.12) 0%, transparent 50%),
    radial-gradient(circle at 50% 100%, rgba(16, 185, 129, 0.08) 0%, transparent 40%),
    linear-gradient(180deg, #020617 0%, #0F172A 100%);
  background-attachment: fixed;
}
```

### Noise Texture
```css
body::before {
  content: '';
  background: url("data:image/svg+xml,..."); /* Fractal noise SVG */
  opacity: 0.03;
  pointer-events: none;
}
```

### Rotating Orbs
```css
body::after {
  background:
    radial-gradient(circle at 25% 25%, rgba(14, 165, 233, 0.1) 0%, transparent 25%),
    radial-gradient(circle at 75% 75%, rgba(139, 92, 246, 0.08) 0%, transparent 25%);
  animation: rotate-orbs 30s linear infinite;
}
```

---

## 📱 Composants Spécialisés

### Confidence Badge
Indicateur de confiance avec :
- Barre de progression gradiente animée
- Pulse indicator à la fin de la barre
- Labels de qualité (EXCELLENT, BON, MOYEN, FAIBLE)
- Points indicateurs visuels (5 niveaux)
- Couleurs adaptatives selon le niveau

### Camera Viewport
- Overlay cyber-grid semi-transparent
- Status badges avec glow effect (LIVE, IA Connected)
- FPS counter en police mono
- Shimmer loading state
- Neon border effect

### Model Version Cards
- Icon container avec gradient background
- Badge ACTIF pour le modèle actif
- Progress bar de précision
- Hover scale effect

---

## 🎯 Principes de Design

1. **Fluidité** : Toutes les transitions sont smooth et naturelles (300-800ms)
2. **Luminosité** : Effets de glow stratégiques sur les éléments actifs
3. **Profondeur** : Utilisation de glassmorphism et d'ombres multiples
4. **Contraste** : Texte très lisible sur backgrounds sombres
5. **Mouvement** : Animations organiques et subtiles
6. **Hiérarchie** : Typographie contrastée (Display/Body/Mono)

---

## 🚀 Utilisation

### Import des fonts
```html
@import url("https://fonts.googleapis.com/css2?family=Orbitron:wght@500;600;700;800;900&family=DM+Sans:wght@400;500;600;700&family=Fira+Code:wght@400;500;600&display=swap");
```

### Classes utilitaires Tailwind custom
Voir `tailwind.config.ts` pour :
- Couleurs étendues
- Animations custom
- Box shadows avec glow
- Backdrop blur values

### CSS Global
Voir `globals.css` pour :
- Variables CSS custom
- Composants de base (.card, .touch-btn)
- Effets visuels (.glow-text, .shimmer, .neon-border)
- Scrollbar styling
- Selection styling

---

## 📦 Packages requis

```json
{
  "framer-motion": "^11.3.31",  // Animations fluides
  "recharts": "^3.7.0",         // Charts avec styling custom
  "tailwindcss": "^3.4.10"      // Utility-first CSS
}
```

---

## 💡 Exemples d'Utilisation

### Header avec glow
```tsx
<h1 className="font-display text-3xl font-bold">
  <span className="glow-text bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
    SignFlow Live
  </span>
</h1>
```

### Button avec gradient et ripple
```tsx
<button className="touch-btn bg-gradient-to-br from-primary to-secondary text-white">
  <span className="relative z-10 flex items-center gap-2">
    <Icon />
    Action
  </span>
</button>
```

### Card avec neon border
```tsx
<div className="card neon-border p-6">
  <Content />
</div>
```

### Animated background blob
```tsx
<motion.div
  className="absolute h-64 w-64 rounded-full bg-primary blur-3xl"
  animate={{
    x: [0, 100, 0],
    y: [0, 50, 0],
    scale: [1, 1.2, 1]
  }}
  transition={{
    duration: 20,
    repeat: Infinity,
    ease: "easeInOut"
  }}
/>
```

---

## 🎨 Résultat Final

Le design system SignFlow crée une expérience visuelle **unique et mémorable** qui :
- Se distingue complètement des interfaces génériques
- Reflète la nature fluide du langage des signes
- Fusionne esthétique organique et précision technologique
- Offre une expérience premium et professionnelle
- Reste accessible et fonctionnel

**Signature visuelle** : Formes fluides + palette bioluminescente + typographie contrastée + animations organiques = Interface néo-biomécanique distinctive
