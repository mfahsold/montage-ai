# Web UI Architecture Guide

## Structure

```
src/montage_ai/web_ui/
├── templates/
│   ├── base.html                 # Main template (inheritance root)
│   ├── index.html               # Dashboard (extends base.html)
│   ├── montage.html             # Montage Creator (extends base.html)
│   ├── shorts.html              # Shorts Generator (extends base.html)
│   ├── gallery.html             # Gallery (extends base.html) ✅
│   ├── transcript.html          # Transcript Editor (extends base.html)
│   ├── features.html            # Features Page (extends base.html)
│   ├── settings.html            # Settings (extends base.html)
│   └── components/
│       ├── macros.html          # Reusable template macros
│       ├── icon.html            # Icon components
│       └── voxel.html           # Voxel UI components
├── static/
│   ├── css/
│   │   ├── voxel-dark.css      # Main theme (CSS vars, shared styles)
│   │   ├── icons.css            # Icon definitions
│   │   └── responsive.css       # Media queries
│   ├── js/
│   │   ├── app.js              # Shared app initialization
│   │   └── components.js        # Reusable JS components
│   └── img/
└── app.py                       # Flask app
```

## Best Practices

### 1. Template Inheritance
**Always** extend `base.html`:
```html
{% extends "base.html" %}
{% import "components/macros.html" as m %}

{% block title %}Page Title - Montage AI{% endblock %}

{% block content %}
    <!-- Your page content here -->
{% endblock %}
```

### 2. Using Macros
Instead of writing repetitive HTML, use macros from `components/macros.html`:

```html
{# Simple card #}
{% call m.voxel_card("Title", "Subtitle") %}
    <p>Card content here</p>
{% endcall %}

{# Workflow cards with buttons #}
{{ m.workflow_card(
    title="MONTAGE CREATOR",
    subtitle="Upload footage. AI assembles cuts.",
    icon_class="icon-camera",
    button_label="Launch Creator",
    button_url="/montage",
    theme="primary"
) }}

{# Job list items #}
{{ m.job_card(
    title="PROJECT_ALPHA_V2",
    status="COMPLETED",
    timestamp="2h ago",
    is_completed=True
) }}

{# Section headers #}
{% call m.section_header("SECTION TITLE", "Optional subtitle") %}{% endcall %}

{# Empty states #}
{{ m.empty_state(
    message="No data available",
    action_label="Go Back",
    action_url="/"
) }}
```

### 3. CSS - Use Variables, Not Inline Styles
In `voxel-dark.css`:
```css
:root {
    --primary: #44ff00;
    --secondary: #ff5500;
    --bg: #0a0a0a;
    --card-bg: #1a1a1a;
    --border: #333;
    --text: #e5e5e5;
    --muted: #666;
}
```

Use in HTML:
```html
<!-- ❌ DON'T: Inline styles -->
<div style="color: #44ff00; background: #1a1a1a;">Bad</div>

<!-- ✅ DO: CSS classes with variables -->
<div class="text-primary bg-card-bg">Good</div>
```

### 4. Consistent Spacing
Use Tailwind-like utility classes:
```html
<div class="mb-8 p-4 flex gap-3">
    <!-- mb-8 = margin-bottom: 2rem -->
    <!-- p-4 = padding: 1rem -->
    <!-- gap-3 = grid-gap / flex-gap: 0.75rem -->
</div>
```

**Spacing Scale:**
- `gap-1`, `gap-2`, `gap-3`, `gap-4`, `gap-6` (0.25rem to 1.5rem)
- `mb-2`, `mb-4`, `mb-8`, `mb-12` (0.5rem to 3rem)
- `p-2`, `p-4`, `p-6` (0.5rem to 1.5rem)

### 5. Component Consistency

| Component | When to Use | Example |
|-----------|-----------|---------|
| `voxel-btn` | Primary action | "Create", "Submit" |
| `voxel-btn voxel-btn-secondary` | Secondary action | "Cancel", "Skip" |
| `voxel-card` | Content container | Any boxed section |
| `badge` | Status indicator | "PROCESSING", "COMPLETED" |
| `section-header` | Section title | "DASHBOARD", "RECENT JOBS" |

### 6. Adding New Pages

1. **Create template file** in `templates/new-page.html`
2. **Extend base.html**:
```html
{% extends "base.html" %}
{% import "components/macros.html" as m %}

{% block title %}New Page - Montage AI{% endblock %}

{% block content %}
    <!-- Use macros for all repeated elements -->
{% endblock %}
```

3. **Add route** in `app.py`:
```python
@app.route('/new-page')
def new_page():
    return render_template('new-page.html')
```

4. **Update navbar** (already in `base.html` - no changes needed!)

### 7. Responsive Design

Use CSS media queries from `responsive.css`:
```html
<!-- Mobile-first classes -->
<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3">
    <!-- 1 col on mobile, 2 on tablet, 3 on desktop -->
</div>

<!-- Conditional visibility -->
<div class="hidden md:flex">Only visible on tablet+</div>
<div class="md:hidden">Only visible on mobile</div>
```

## Migration Checklist

When converting old templates:

- [ ] Change `<!DOCTYPE html>` to `{% extends "base.html" %}`
- [ ] Remove duplicate `<nav>` (already in base)
- [ ] Remove duplicate `<footer>` (already in base)
- [ ] Convert inline `<style>` to `voxel-dark.css`
- [ ] Replace repetitive HTML with macros
- [ ] Use CSS variables instead of hardcoded colors
- [ ] Test navbar highlighting (`active` class)
- [ ] Test responsive layouts on mobile

## Example: Before & After

### ❌ Before (Old Pattern)
```html
<!DOCTYPE html>
<html>
<head>
    <title>Page</title>
    <link rel="stylesheet" href="...voxel-dark.css">
    <style>
        .my-card { border: 2px solid #333; padding: 1rem; }
        .my-card h3 { color: #44ff00; }
    </style>
</head>
<body>
    <nav class="navbar">
        <!-- Duplicate navbar code -->
    </nav>
    
    <div class="container">
        <h1 style="color: #44ff00;">Title</h1>
        <div class="my-card">
            <h3>Card Title</h3>
            <p>Content</p>
            <button>Action</button>
        </div>
    </div>
    
    <footer><!-- Duplicate footer --></footer>
</body>
</html>
```

### ✅ After (New Pattern)
```html
{% extends "base.html" %}
{% import "components/macros.html" as m %}

{% block title %}Page - Montage AI{% endblock %}

{% block content %}
    <h1 class="text-primary">Title</h1>
    
    {% call m.voxel_card("Card Title") %}
        <p>Content</p>
        <button class="voxel-btn">Action</button>
    {% endcall %}
{% endblock %}
```

**Benefits:**
- ✅ 40% less HTML
- ✅ Single navbar to maintain
- ✅ Consistent styling
- ✅ Easier updates

## Development Workflow

```bash
# 1. Modify templates/base.html or voxel-dark.css
# 2. Changes auto-apply to all pages (Flask auto-reload)
# 3. Test all pages: /, /montage, /shorts, /gallery, etc.

# Start local server
python src/montage_ai/web_ui/app.py

# Visit: http://localhost:5000
```

## Known Templates Still to Migrate

- [ ] `montage.html` (in progress)
- [ ] `shorts.html` (in progress)
- [ ] `transcript.html` (in progress)
- [ ] `features.html` (in progress)
- [ ] `settings.html` (in progress)

**Current Status:** `index.html` ✅, `gallery.html` ✅
