# CSS Responsive Utilities Reference

Quick reference guide for the enhanced responsive CSS system in Montage AI.

## 🎨 Responsive Breakpoints

| Breakpoint | Min Width | Use Case |
|-----------|-----------|----------|
| **None** | 0px | Mobile (default) |
| **sm** | 640px | Small tablets |
| **md** | 768px | Tablets |
| **lg** | 1024px | Laptops (NEW) |
| **xl** | 1280px | Large desktops (NEW) |

**Prefix syntax**: `md:`, `lg:`, `xl:` (e.g., `md:flex-row`, `lg:grid-cols-3`)

---

## 📐 Grid System

### Auto-fit Grid
```html
<div class="grid grid-cols-auto gap-6">
  <!-- 1+ columns with min 300px width -->
</div>
```

### Fixed Columns
```html
<!-- Mobile: 1 col | Tablet: 2 cols | Desktop: 3 cols -->
<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
  ...
</div>

<!-- Mobile: 1 col | Desktop+: 4 cols -->
<div class="grid grid-cols-1 lg:grid-cols-4 gap-8">
  ...
</div>
```

---

## 🔄 Flexbox Utilities

### Direction (Responsive)
```html
<!-- Stacked on mobile, row on tablet -->
<div class="flex flex-col md:flex-row items-center gap-4">
  ...
</div>

<!-- Row on desktop -->
<div class="flex flex-col lg:flex-row justify-between gap-8">
  ...
</div>
```

### Alignment
```html
<div class="flex items-start gap-4">...</div>        <!-- top align -->
<div class="flex items-center gap-4">...</div>       <!-- center align -->
<div class="flex items-end gap-4">...</div>          <!-- bottom align -->
<div class="flex justify-start gap-4">...</div>      <!-- left -->
<div class="flex justify-center gap-4">...</div>     <!-- center -->
<div class="flex justify-between gap-4">...</div>    <!-- space between -->
<div class="flex justify-end gap-4">...</div>        <!-- right -->
```

### Gap Sizes
```html
<div class="flex gap-1">...</div>    <!-- 0.25rem -->
<div class="flex gap-2">...</div>    <!-- 0.5rem -->
<div class="flex gap-3">...</div>    <!-- 0.75rem (NEW) -->
<div class="flex gap-4">...</div>    <!-- 1rem -->
<div class="flex gap-6">...</div>    <!-- 1.5rem (NEW) -->
<div class="flex gap-8">...</div>    <!-- 2rem (NEW) -->
```

---

## 📏 Spacing

### Margin Bottom
```html
<div class="mb-1">...</div>    <!-- 0.25rem -->
<div class="mb-2">...</div>    <!-- 0.5rem -->
<div class="mb-3">...</div>    <!-- 0.75rem (NEW) -->
<div class="mb-4">...</div>    <!-- 1rem -->
<div class="mb-6">...</div>    <!-- 1.5rem (NEW) -->
<div class="mb-8">...</div>    <!-- 2rem -->
<div class="mb-12">...</div>   <!-- 3rem (NEW) -->
```

### Margin Top
```html
<div class="mt-4">...</div>    <!-- 1rem -->
<div class="mt-8">...</div>    <!-- 2rem (NEW) -->
```

### Padding Top
```html
<div class="pt-4">...</div>    <!-- 1rem -->
<div class="pt-8">...</div>    <!-- 2rem (NEW) -->
```

### Vertical Space Between
```html
<div class="space-y-1">
  <div>Item 1</div>
  <div>Item 2</div>  <!-- Gets 0.25rem top margin -->
</div>

<!-- Options: space-y-1, space-y-2, space-y-3 (NEW), space-y-4 (NEW) -->
```

---

## 🔘 Buttons

### Variants
```html
<button class="voxel-btn">Default</button>
<button class="voxel-btn voxel-btn-primary">Primary</button>
<button class="voxel-btn voxel-btn-secondary">Secondary</button>

<!-- Disabled state -->
<button class="voxel-btn" disabled>Disabled</button>
```

### Sizes
```html
<button class="voxel-btn voxel-btn-sm">Small</button>
<button class="voxel-btn">Base (default)</button>
<button class="voxel-btn voxel-btn-lg">Large</button>
```

### Full Width
```html
<button class="voxel-btn voxel-btn-primary w-full">Full Width</button>
```

---

## 🎴 Cards

### Basic Card
```html
<div class="voxel-card">
  <h3 class="mb-4">Title</h3>
  <p class="text-muted">Content</p>
</div>
```

### Workflow Card (NEW)
```html
<a href="/montage" class="voxel-card workflow-card">
  <div class="workflow-icon mb-3">📽️</div>
  <h3 class="text-primary mb-2">MONTAGE CREATOR</h3>
  <p class="text-muted">Description...</p>
  <button class="voxel-btn voxel-btn-primary w-full">LAUNCH →</button>
</a>
```

---

## 🏷️ Badges & Status

### Badges
```html
<span class="voxel-badge">Default</span>
<span class="voxel-badge voxel-badge-primary">Primary</span>
<span class="voxel-badge voxel-badge-secondary">Secondary</span>
```

### Status Badges (NEW)
```html
<span class="badge badge-success">COMPLETED</span>
<span class="badge badge-secondary">PROCESSING</span>
```

---

## 🎨 Text Colors

```html
<span class="text-primary">Electric Blue</span>
<span class="text-secondary">Neon Orange</span>
<span class="text-muted">Muted Gray</span>
```

### Text Sizes
```html
<p class="text-xs">Extra small</p>
<p class="text-sm">Small</p>
<p class="text-center">Centered</p>
```

---

## 🔗 Utility Classes

### Sizing
```html
<div class="w-full">Full width</div>
<div class="flex-1">Flex grow</div>
<div class="flex-shrink-0">No shrink</div>
<div class="min-w-0">Min width 0 (for text truncate)</div>
```

### Text Overflow
```html
<div class="truncate">Long text gets cut off with ellipsis...</div>
```

### Borders
```html
<div class="border-t">Top border</div>
<div class="border-b">Bottom border</div>
<div class="border-l">Left border</div>
<div class="border-r">Right border</div>
```

---

## 📱 Common Responsive Patterns

### Card Grid (Mobile → Desktop)
```html
<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
  <div class="voxel-card">Card 1</div>
  <div class="voxel-card">Card 2</div>
  <div class="voxel-card">Card 3</div>
</div>
```

### Header with Title & Actions
```html
<div class="flex flex-col md:flex-row justify-between items-start gap-6 md:items-end">
  <div>
    <h1 class="voxel-title">Title</h1>
    <p class="text-muted">Subtitle</p>
  </div>
  <div class="flex gap-2">
    <button class="voxel-btn">Action 1</button>
    <button class="voxel-btn">Action 2</button>
  </div>
</div>
```

### Job List Item
```html
<div class="voxel-card flex flex-col md:flex-row md:items-center justify-between gap-4">
  <div class="flex items-center gap-4 flex-1">
    <div style="width: 12px; height: 12px; background: #10b981;"></div>
    <div class="min-w-0">
      <h4 class="truncate">PROJECT_NAME</h4>
      <span class="text-xs text-muted">Style • Resolution</span>
    </div>
  </div>
  <div class="flex items-center gap-3 flex-wrap justify-between md:justify-end">
    <span class="badge badge-success">COMPLETED</span>
    <span class="text-xs text-muted">2h ago</span>
    <button class="voxel-btn">Download</button>
  </div>
</div>
```

### Multi-Column on Desktop
```html
<div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
  <div class="voxel-card">
    <h4 class="text-primary mb-4">Card 1</h4>
    <p>Content...</p>
  </div>
  <!-- More cards... -->
</div>
```

---

## 🎯 Design System Reminders

- **Colors**: Use CSS variables (`var(--primary)`, `var(--secondary)`, etc.)
- **Spacing**: Use predefined gap/margin/padding values (no arbitrary sizes)
- **Typography**: Use `text-xs`, `text-sm`, `text-base` classes
- **Shadows**: Use `var(--shd-sm)`, `var(--shd-md)`, `var(--shd-primary)`
- **Transitions**: Cards/buttons have built-in `steps(2)` transitions

---

## 🔍 Testing Tips

1. **Mobile First**: Design for mobile, add breakpoints for larger screens
2. **Test at Breakpoints**: 320px, 640px, 768px, 1024px, 1280px
3. **Touch Targets**: Ensure buttons/inputs are ≥44px (min touch target)
4. **Read Width**: Text should be <80 characters per line for readability
5. **Performance**: CSS is lightweight, no JS required for responsive behavior

