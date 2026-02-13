# 🎨 SignFlow Design Snippets

Collection de snippets réutilisables pour maintenir la cohérence du design system.

---

## 🎭 Composants de Base

### Card Standard
```tsx
<div className="card p-6">
  <h2 className="font-display text-xl font-semibold mb-4">Titre</h2>
  <p className="text-text-secondary">Contenu</p>
</div>
```

### Card avec Neon Border
```tsx
<div className="card neon-border p-6">
  <h2 className="font-display text-xl font-semibold mb-4">Titre</h2>
  <p className="text-text-secondary">Contenu</p>
</div>
```

### Card avec Animation d'Entrée
```tsx
<motion.div
  initial={{ opacity: 0, y: 20 }}
  animate={{ opacity: 1, y: 0 }}
  transition={{ delay: 0.2 }}
  className="card p-6"
>
  <Content />
</motion.div>
```

---

## 🔘 Boutons

### Primary Button
```tsx
<button className="touch-btn bg-gradient-to-br from-primary to-secondary text-white">
  <span className="relative z-10 flex items-center gap-2">
    <Icon />
    Action
  </span>
</button>
```

### Secondary Button
```tsx
<button className="touch-btn bg-gradient-to-br from-accent/30 to-accent-dark/30 text-accent backdrop-blur-sm">
  <span className="relative z-10 flex items-center gap-2">
    <Icon />
    Action
  </span>
</button>
```

### Ghost Button
```tsx
<button className="touch-btn bg-gradient-to-br from-primary/30 to-secondary/30 text-primary backdrop-blur-sm">
  <span className="relative z-10 flex items-center gap-2">
    <Icon />
    Action
  </span>
</button>
```

### Danger Button
```tsx
<button className="touch-btn bg-gradient-to-br from-red-500/80 to-red-600 text-white">
  <span className="relative z-10 flex items-center gap-2">
    <Icon />
    Supprimer
  </span>
</button>
```

---

## ✨ Texte avec Effets

### Glow Text avec Gradient
```tsx
<h1 className="font-display text-3xl font-bold">
  <span className="glow-text bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
    Titre Principal
  </span>
</h1>
```

### Texte Accentué
```tsx
<p className="text-lg">
  Texte normal avec <span className="text-primary font-semibold">accent cyan</span>
</p>
```

### Label avec Tracking
```tsx
<p className="text-xs font-medium uppercase tracking-wider text-text-tertiary">
  Label
</p>
```

---

## 🎨 Backgrounds Animés

### Orbe Animé
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

### Blob Morphing
```tsx
<div className="morph-shape absolute -top-20 -right-20 h-64 w-64 bg-gradient-to-br from-primary/20 to-secondary/20 blur-3xl pointer-events-none" />
```

### Grid Overlay
```tsx
<div className="absolute inset-0 cyber-grid opacity-20 pointer-events-none" />
```

---

## 🏷️ Badges et Status

### Badge Actif
```tsx
<span className="rounded-full bg-accent/20 px-3 py-1 text-xs font-medium text-accent">
  ACTIF
</span>
```

### Badge Status
```tsx
<div className="flex items-center gap-2 rounded-full bg-accent/20 px-3 py-1.5 backdrop-blur-sm">
  <div className="h-2 w-2 animate-pulse rounded-full bg-accent shadow-glow" />
  <span className="text-xs font-medium text-accent">LIVE</span>
</div>
```

### Status Badge avec Animation
```tsx
<motion.div
  initial={{ scale: 0 }}
  animate={{ scale: 1 }}
  className="flex items-center gap-2 rounded-full bg-primary/20 px-3 py-1.5 backdrop-blur-sm"
>
  <div className="h-2 w-2 rounded-full bg-primary shadow-glow" />
  <span className="text-xs font-medium text-primary">Connected</span>
</motion.div>
```

---

## 📊 KPI Cards

### KPI Card Simple
```tsx
<div className="card p-6">
  <p className="text-xs font-medium uppercase tracking-wider text-text-tertiary mb-1">
    Label
  </p>
  <p className="font-display text-3xl font-bold">1,234</p>
</div>
```

### KPI Card avec Icon
```tsx
<motion.div
  whileHover={{ scale: 1.03, y: -4 }}
  className="card relative overflow-hidden p-6"
>
  {/* Background blob */}
  <div className="pointer-events-none absolute -top-10 -right-10 h-32 w-32 rounded-full bg-gradient-to-br from-primary to-primary-dark opacity-20 blur-2xl" />

  <div className="relative z-10">
    <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br from-primary to-primary-dark p-[2px]">
      <div className="flex h-full w-full items-center justify-center rounded-xl bg-background-card">
        <Icon className="h-6 w-6" />
      </div>
    </div>
    <p className="mb-1 text-xs font-medium uppercase tracking-wider text-text-tertiary">
      Label
    </p>
    <p className="font-display text-3xl font-bold">1,234</p>
  </div>
</motion.div>
```

---

## 🎯 Icons avec Gradient Container

### Small Icon
```tsx
<div className="h-10 w-10 rounded-lg bg-gradient-to-br from-primary/20 to-secondary/20 flex items-center justify-center">
  <Icon className="h-5 w-5 text-primary" />
</div>
```

### Large Icon
```tsx
<div className="h-12 w-12 rounded-xl bg-gradient-to-br from-primary to-secondary p-[2px]">
  <div className="flex h-full w-full items-center justify-center rounded-xl bg-background-elevated">
    <Icon className="h-6 w-6 text-primary" />
  </div>
</div>
```

---

## 📝 Inputs

### Input Standard
```tsx
<input
  className="w-full rounded-btn border border-primary/30 bg-surface-secondary/80 px-4 py-3 text-base backdrop-blur-sm transition-all focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/50"
  placeholder="Entrez du texte..."
/>
```

### Label + Input
```tsx
<label className="flex flex-col gap-2 text-sm">
  <span className="font-medium text-text-secondary">Label</span>
  <input
    className="rounded-btn border border-primary/30 bg-surface-secondary/80 px-4 py-3 text-base backdrop-blur-sm transition-all focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/50"
    placeholder="Placeholder..."
  />
</label>
```

---

## 🎬 Animations Framer Motion

### Container avec Stagger Children
```tsx
<motion.div
  variants={{
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: { staggerChildren: 0.1 }
    }
  }}
  initial="hidden"
  animate="show"
>
  <motion.div
    variants={{
      hidden: { opacity: 0, y: 20 },
      show: { opacity: 1, y: 0 }
    }}
  >
    Child 1
  </motion.div>
  <motion.div
    variants={{
      hidden: { opacity: 0, y: 20 },
      show: { opacity: 1, y: 0 }
    }}
  >
    Child 2
  </motion.div>
</motion.div>
```

### Modal avec AnimatePresence
```tsx
<AnimatePresence>
  {showModal && (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4"
    >
      <motion.div
        initial={{ scale: 0.9, y: 20 }}
        animate={{ scale: 1, y: 0 }}
        exit={{ scale: 0.9, y: 20 }}
        className="card neon-border w-full max-w-md p-6"
      >
        <h3 className="font-display text-xl font-semibold mb-4">Modal Title</h3>
        <p className="text-text-secondary mb-6">Modal content</p>
        <button
          className="touch-btn bg-gradient-to-br from-primary to-secondary text-white w-full"
          onClick={() => setShowModal(false)}
        >
          Fermer
        </button>
      </motion.div>
    </motion.div>
  )}
</AnimatePresence>
```

### Liste avec Animation d'Entrée
```tsx
{items.map((item, index) => (
  <motion.li
    key={item.id}
    initial={{ opacity: 0, x: -20 }}
    animate={{ opacity: 1, x: 0 }}
    transition={{ delay: index * 0.05 }}
    className="card p-4"
  >
    {item.content}
  </motion.li>
))}
```

---

## 🌊 Progress Bars

### Simple Progress
```tsx
<div className="h-2 w-full rounded-full bg-surface-secondary/80 overflow-hidden">
  <motion.div
    initial={{ width: 0 }}
    animate={{ width: `${progress}%` }}
    transition={{ duration: 0.8, ease: "easeOut" }}
    className="h-full rounded-full bg-gradient-to-r from-primary to-secondary"
  />
</div>
```

### Progress avec Glow
```tsx
<div className="relative h-3 overflow-hidden rounded-full bg-surface-secondary/80">
  <div className="absolute inset-0 shimmer opacity-30" />
  <motion.div
    initial={{ width: 0 }}
    animate={{ width: `${progress}%` }}
    transition={{ duration: 0.8, ease: "easeOut" }}
    className="relative h-full rounded-full bg-gradient-to-r from-primary to-secondary shadow-[0_0_20px_rgba(14,165,233,0.4)]"
  >
    <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent" />
  </motion.div>
</div>
```

---

## 💫 Loading States

### Shimmer Block
```tsx
<div className="shimmer h-8 w-full rounded-btn" />
```

### Skeleton Card
```tsx
<div className="card p-6 space-y-3">
  <div className="shimmer h-6 w-3/4 rounded-btn" />
  <div className="shimmer h-4 w-full rounded-btn" />
  <div className="shimmer h-4 w-5/6 rounded-btn" />
</div>
```

### Spinner
```tsx
<div className="flex items-center justify-center">
  <div className="h-12 w-12 animate-spin rounded-full border-4 border-surface-tertiary border-t-primary" />
</div>
```

---

## 🎨 Tooltips

### Tooltip Simple
```tsx
<div className="group relative inline-block">
  <button className="touch-btn bg-primary/30 text-primary">
    Hover me
  </button>
  <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:block">
    <div className="card px-3 py-2 text-xs whitespace-nowrap">
      Tooltip text
    </div>
  </div>
</div>
```

---

## 📱 Responsive Grids

### 2 Columns → 4 Columns
```tsx
<div className="grid grid-cols-2 gap-4 md:grid-cols-4">
  <Item />
  <Item />
  <Item />
  <Item />
</div>
```

### Auto-fit avec Min Width
```tsx
<div className="grid grid-cols-[repeat(auto-fit,minmax(280px,1fr))] gap-4">
  <Item />
  <Item />
  <Item />
</div>
```

---

## 🎯 Headers de Section

### Header Simple
```tsx
<header className="flex items-center gap-4 mb-6">
  <div className="h-12 w-12 rounded-xl bg-gradient-to-br from-primary to-secondary p-[2px]">
    <div className="flex h-full w-full items-center justify-center rounded-xl bg-background-elevated">
      <Icon className="h-6 w-6 text-primary" />
    </div>
  </div>
  <h1 className="font-display text-3xl font-bold">
    <span className="glow-text bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
      Page Title
    </span>
  </h1>
</header>
```

### Header avec Action
```tsx
<header className="flex items-center justify-between mb-6">
  <div className="flex items-center gap-4">
    <div className="h-12 w-12 rounded-xl bg-gradient-to-br from-primary to-secondary p-[2px]">
      <div className="flex h-full w-full items-center justify-center rounded-xl bg-background-elevated">
        <Icon className="h-6 w-6 text-primary" />
      </div>
    </div>
    <h1 className="font-display text-3xl font-bold">
      <span className="glow-text bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
        Page Title
      </span>
    </h1>
  </div>
  <button className="touch-btn bg-gradient-to-br from-primary to-secondary text-white">
    Action
  </button>
</header>
```

---

## 🔔 Notifications

### Success Toast
```tsx
<motion.div
  initial={{ opacity: 0, y: -10 }}
  animate={{ opacity: 1, y: 0 }}
  exit={{ opacity: 0 }}
  className="card neon-border p-4"
>
  <div className="flex items-center gap-3">
    <div className="h-2 w-2 animate-pulse rounded-full bg-accent shadow-glow" />
    <p className="text-sm text-accent">Opération réussie !</p>
  </div>
</motion.div>
```

### Error Toast
```tsx
<motion.div
  initial={{ opacity: 0, y: -10 }}
  animate={{ opacity: 1, y: 0 }}
  exit={{ opacity: 0 }}
  className="card border-red-500/30 p-4"
>
  <div className="flex items-center gap-3">
    <svg className="h-5 w-5 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
    <p className="text-sm text-red-300">Une erreur s'est produite</p>
  </div>
</motion.div>
```

---

## 💡 Utilisation

1. **Copier le snippet** désiré
2. **Adapter les props** selon besoin
3. **Respecter les classes** Tailwind custom
4. **Maintenir la cohérence** du design system

---

## 📚 Ressources

- **Design System complet** : `DESIGN_SYSTEM.md`
- **Tailwind Config** : `tailwind.config.ts`
- **Global CSS** : `src/styles/globals.css`
- **Exemples réels** : `pages/TranslatePage.tsx`, `pages/DashboardPage.tsx`

---

**Happy coding! 🚀**
