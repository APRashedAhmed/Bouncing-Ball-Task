"""Place to define constants to be used throughout the repo."""

# Image Extensions (taken from torchvision with some additions)
IMAGE_EXTENSIONS = {
    ".jpeg",
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
}

DEFAULT_COLORS = (
    (255, 0, 0),  # R
    (0, 255, 0),  # G
    (0, 0, 255),  # B
)

CONSTANT_COLOR = (
    (255, 0, 0),  # R
    (255, 0, 0),  # R
    (255, 0, 0),  # R
)

# Experiment constants
default_trial_types = (
    "catch",
    "straight",
    "bounce",
)

default_ball_colors = (
    "red",
    "green",
    "blue",
)

default_idx_to_color_dict = {
    idx + 1 : color
    for idx, color in enumerate(default_ball_colors)
}

default_color_to_idx_dict = {
    color : idx + 1
    for idx, color in enumerate(default_ball_colors)
}
