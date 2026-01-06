# Montage AI - Voxel Design System

**Version:** 1.0
**Theme:** Voxel Dark (Neon Technical)

---

## Design Philosophy

Montage AI uses a **Voxel aesthetic** - hard edges, pixel-sharp shadows, and neon accents on deep black backgrounds. The design evokes retro computing while maintaining modern readability.

**Core Principles:**
1. **Monospace Typography** - Technical, precise, readable
2. **Hard Shadows** - No blur, pixel-perfect offsets (4px)
3. **Neon Accents** - Electric blue + orange highlights
4. **Grid Pattern** - Subtle 40px grid overlay on body
5. **Mobile-First** - Fluid typography with `clamp()`

---

## Color Palette

### Primary Colors

| Token | Hex | Usage |
|-------|-----|-------|
| `--bg` | `#050505` | Page background |
| `--fg` | `#e0e0e0` | Default text |
| `--card-bg` | `#0d0d0d` | Card/panel backgrounds |
| `--primary` | `#0055ff` | Electric Blue - CTAs, links, active states |
| `--secondary` | `#ff5500` | Neon Orange - Warnings, highlights |

### Semantic Colors

| Token | Hex | Usage |
|-------|-----|-------|
| `--success` | `#00ff88` | Success states, completed |
| `--warning` | `#ffcc00` | Warnings, attention |
| `--error` | `#ff3333` | Errors, destructive actions |
| `--muted` | `#1a1a1a` | Disabled backgrounds |
| `--muted-fg` | `#888888` | Secondary text, labels |

### Borders & Shadows

| Token | Value | Usage |
|-------|-------|-------|
| `--border` | `#222222` | Default borders |
| `--border-bright` | `#444444` | Active/hover borders |
| `--shd-md` | `4px 4px 0px 0px var(--border)` | Card shadows |
| `--shd-primary` | `4px 4px 0px 0px var(--primary)` | Primary button shadows |

---

## Typography

### Font Stack

```css
--font-main: 'Share Tech Mono', monospace;
```

### Fluid Sizes (Mobile-First)

| Token | Range | Usage |
|-------|-------|-------|
| `--fs-xs` | 0.7-0.8rem | Labels, captions |
| `--fs-sm` | 0.8-0.9rem | Secondary text |
| `--fs-base` | 0.9-1rem | Body text |
| `--fs-lg` | 1.1-1.3rem | Subheadings |
| `--fs-xl` | 1.4-1.8rem | Section headers |
| `--fs-2xl` | 1.8-2.5rem | Page titles |

### Technical Labels

```css
.tech-label {
  font-size: var(--fs-xs);
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--muted-fg);
  font-weight: 700;
}
```

---

## Components

### Cards (`.voxel-card`)

```html
<div class="voxel-card">
  <h3>Card Title</h3>
  <p>Card content</p>
</div>
```

**CSS:**
```css
.voxel-card {
  background: var(--card-bg);
  border: 2px solid var(--border);
  padding: 1.5rem;
  box-shadow: var(--shd-md);
}
```

### Buttons

| Class | Usage |
|-------|-------|
| `.voxel-btn` | Default button (dark) |
| `.voxel-btn-primary` | Primary action (blue) |
| `.voxel-btn-secondary` | Secondary action (orange) |
| `.voxel-btn-ghost` | Subtle button (transparent) |

**Example:**
```html
<button class="voxel-btn voxel-btn-primary">Generate</button>
```

### Inputs

```html
<input type="text" class="voxel-input" placeholder="Enter value">
<select class="voxel-input">
  <option>Option 1</option>
</select>
```

### Progress Bar

```html
<div class="progress-bar">
  <div class="progress-fill" style="width: 65%"></div>
</div>
```

---

## Layout Classes

### Grid System

```html
<!-- Auto-fit columns (min 300px) -->
<div class="grid grid-cols-auto">...</div>

<!-- Responsive columns -->
<div class="grid md:grid-cols-2 lg:grid-cols-3">...</div>
```

### Spacing

| Class | Value |
|-------|-------|
| `.p-2` | padding: 0.5rem |
| `.p-4` | padding: 1rem |
| `.mb-2` | margin-bottom: 0.5rem |
| `.mb-4` | margin-bottom: 1rem |
| `.gap-2` | gap: 0.5rem |
| `.gap-4` | gap: 1rem |

### Flexbox

```html
<div class="flex items-center justify-between gap-4">
  <span>Left</span>
  <span>Right</span>
</div>
```

---

## Animations

### Hover Transitions

```css
transition: all 0.1s steps(2); /* Voxel-style stepped animation */
```

### Progress Shimmer

```css
@keyframes shimmer {
  0% { transform: translateX(-100%); }
  100% { transform: translateX(100%); }
}
```

---

## Best Practices

### DO:
- Use `--primary` for primary CTAs
- Use `--secondary` sparingly for highlights
- Keep shadows consistent (4px offset)
- Use uppercase for labels (`.tech-label`)
- Use monospace for all text

### DON'T:
- Mix rounded corners with sharp edges
- Use gradient backgrounds (except shimmer)
- Use multiple font families
- Use soft/blur shadows
- Use more than 2 accent colors per view

---

## Template Reference

| Template | Primary Action | Secondary Actions |
|----------|----------------|-------------------|
| `index_strategy.html` | Generate Montage | Preview, Settings |
| `shorts.html` | Generate Short | Preview, Safe Zones |
| `transcript.html` | Apply Edits | Preview, Export |

---

## Quick Reference

```html
<!-- Standard page structure -->
<div class="container">
  <h1 class="text-2xl text-primary mb-4">Page Title</h1>

  <div class="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
    <div class="voxel-card">
      <label class="tech-label">Label</label>
      <input class="voxel-input" />
      <button class="voxel-btn voxel-btn-primary mt-4">Action</button>
    </div>
  </div>
</div>
```
