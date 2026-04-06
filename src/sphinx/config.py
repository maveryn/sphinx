# sphinx/config.py
# Global rendering + dataset config

SUPERSAMPLE = 1
BASE_CELL   = 300
SS_CELL     = BASE_CELL * SUPERSAMPLE
OUT_CELL    = BASE_CELL

BORDER_PX   = 8
SEP_PX      = 80
LABEL_PAD   = 120

CANVAS_BG = (255, 255, 255)
SEP_COLOR    = (255, 255, 255)
BORDER_COLOR = (0, 0, 0)
LABEL_COLOR  = (0, 0, 0)
FONT_SIZE    = 96

# Draw an inner frame around option images
OPTION_INNER_BORDER_PX  = BORDER_PX
OPTION_TILE_OUTER_BORDER_PX = 0

# ---------- GLOBAL: image-similarity & retry knobs ----------
# Minimum fractional pixel difference required to treat two images as "meaningfully different"
IMG_DIFF_MIN = 0.02

# Two OPTION images must be at least this different to both appear
# (stricter to prevent near-duplicates)
OPT_UNIQUENESS_MIN = 0.05

# Treat two images as "the same" if their fractional diff <= this
# (generic equality tolerance; formerly 'sym_tol' in symmetry tasks)
IMG_EQUAL_TOL = 0.010

# --- Perceptual hashing knobs (for uniqueness verification) ---
DHASH_SIZE = 8                 # 8x8 per-channel dHash => 64 bits per channel
HASH_PRE_BLUR = 0.6            # BoxBlur radius (averaging) before hashing
HASH_PRE_DOWNSAMPLE = 64       # downsample to NxN with Image.BOX (average pooling)

# Hamming thresholds are PER CHANNEL (R/G/B)
OPT_HASH_MIN_BITS = 10         # >= to treat two OPTION tiles as distinct
GRID_HASH_MIN_BITS = 10        # >= for base vs. transform (grid rule verification)
EQUAL_HASH_MAX_BITS = 4        # <= to treat two tiles as effectively the same


# Retry budget for building ANY sample (tasks read this)
MAX_BUILD_RETRIES = 100
# -----------------------------------------------------------

# distinct palette
COLORS_NAMES = {
    "#e6194b": "red",
    "#3cb44b": "green",
    "#1976d2": "blue",
    "#ff8c00": "orange",
    "#4b0082": "purple",
    "#42d4f4": "cyan",
    "#ff00ff": "magenta",
    "#edc001": "yellow",
    "#8b4513": "brown",
    "#666666": "gray",
}

# Palette list derived from the mapping above (for convenience).
COLORS = list(COLORS_NAMES.keys())

# Which motifs are disabled globally
DISABLED_MOTIFS = {
    "petals",
}

TASK_WEIGHTS = {
    "symmetry_grid_mirror_fill": 0.5,
    "symmetry_scene_mirror_identify": 0.5,
    "symmetry_wallpaper_groups": 0.5,
    "symmetry_frieze_groups": 0.5,
    "sequence_arithmetic": 0.5,
    "sequence_rotation": 0.5,
    "sequence_multi_column_arithmetic": 0.5,
    "tiles_connected_component": 0.5,
    "tiles_shortest_path": 0.5,
    "tiles_missing_tiles": 0.5,
    "tiles_geometry": 0.5,
    "tiles_recoloring": 0.5,
    "tiles_line_length": 0.5,
    "tiles_line_intersections": 0.5,
    "tiles_decompose_compose": 0.5,
    "transform_result_identify": 0.5,
    "transform_pair_infer": 0.5,
    "transform_similarity_identify": 0.5,
    "charts_pie": 0.5,
    "charts_match_proportions": 0.5,
    "shape_count": 0.5,
    "rect_venn": 0.5,
    "geometric_position": 0.5,
    "geometric_sort": 0.5,
    "geometric_stack_count": 0.5,
}

# Assigned by each task that uses Motifs
MOTIF_WEIGHTS = {
    "arc": 0.5,
    "arrow": 0.0,
    "single_arrow": 0.0,
    "bars": 0.0,
    "bitgrid": 0.0,
    "clock": 0.0,
    "concentric_polygon": 0.0,
    "crescent": 0.0,
    "dot": 0.5,
    "dotgrid2d": 0.0,
    "fractal": 0.0,
    "gear": 0.0,
    "glyph": 0.0,
    "keyhole": 0.0,
    "ladder": 0.0,
    "pictogram": 0.0,
    "pinwheel_triangles": 0.0,
    "polygon": 0.5,
    "polyhex": 0.0,
    "polyiamond": 0.5,
    "polyline": 0.0,
    "polyomino": 0.0,
    "rings": 0.0,
    "segment": 0.0,
    "star_polygon": 0.0,
    "stripes": 0.0,
}


# RNG
DEFAULT_SEED = 42

# Optional pattern filters
TASK_PATTERN_FILTERS = {
    # "symmetry_grid_mirror_fill": {...}
}
