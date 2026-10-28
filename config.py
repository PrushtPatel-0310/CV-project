"""Configuration for deterministic edge detection, tracking, and safety monitoring."""

# --- Edge & Contour Detection ---
EDGE_GAUSSIAN_KERNEL_SIZE = (5, 5)
EDGE_CANNY_LOW_THRESHOLD = 50
EDGE_CANNY_HIGH_THRESHOLD = 150
EDGE_MORPHOLOGY_KERNEL_SIZE = (7, 7)
EDGE_MORPHOLOGY_ITERATIONS = 2
EDGE_DILATION_ENABLED = False
EDGE_DILATION_KERNEL_SIZE = (3, 3)
EDGE_DILATION_ITERATIONS = 1
EDGE_MIN_CONTOUR_AREA = 1_500
EDGE_MIN_BOUNDING_BOX_HEIGHT = 80
EDGE_MIN_BOUNDING_BOX_WIDTH = 30
EDGE_MIN_HEIGHT_WIDTH_RATIO = 1.0
EDGE_MAX_HEIGHT_WIDTH_RATIO = 6.0
EDGE_NMS_IOU_THRESHOLD = 0.4
EDGE_SHOW_DEBUG_VIEWS = False

# --- Candidate Box Merging (Fragmentation Reduction) ---
EDGE_MERGE_ENABLED = True
EDGE_MERGE_MAX_VERTICAL_GAP = 35
EDGE_MERGE_MIN_HORIZONTAL_OVERLAP_RATIO = 0.35
EDGE_MERGE_MAX_ASPECT_RATIO = 6.5
EDGE_MERGE_MIN_ASPECT_RATIO = 0.8

# --- Lightweight Temporal Tracker ---
TRACK_IOU_MATCH_THRESHOLD = 0.2
TRACK_CENTER_DIST_NORM_THRESHOLD = 0.75
TRACK_SMOOTHING_ALPHA = 0.4  # Exponential moving average weight for new detection
TRACK_CONFIRMATION_FRAMES = 3  # Consecutive detections required to confirm track
TRACK_MAX_MISSED_FRAMES = 5  # Missed frames before track expiration
TRACK_DANGER_CLEAR_FRAMES = 3  # Consecutive clean frames required to clear danger

# --- Visualization & Debug ---
SHOW_RAW_CANDIDATE_BOXES = False

