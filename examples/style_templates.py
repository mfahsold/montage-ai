"""
Example: Style Templates

Shows how to use predefined style templates.
"""

from montage_ai import list_available_styles, get_style_template

# List all available styles
print("Available Styles:")
for style_name in list_available_styles():
    template = get_style_template(style_name)
    print(f"  - {style_name}: {template['description']}")

# Get specific template
hitchcock = get_style_template("hitchcock")
print(f"\nHitchcock Style Details:")
print(f"  Pacing: {hitchcock['params']['pacing']}")
print(f"  Effects: {hitchcock['params']['effects']}")
