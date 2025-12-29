"""Rich console configuration and helpers."""

from rich.console import Console
from rich.theme import Theme

# Custom theme for consistent styling
custom_theme = Theme(
    {
        "info": "cyan",
        "success": "green",
        "warning": "yellow",
        "error": "red bold",
        "word": "magenta",
        "pos": "blue",
        "dim": "dim",
    }
)

# Main console for output
console = Console(theme=custom_theme)

# Error console for stderr
error_console = Console(theme=custom_theme, stderr=True)
