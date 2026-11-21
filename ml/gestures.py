LABELS = (
    "idle",
    "pinch_draw",
    "open_palm_hover",
    "fist_erase",
    "peace_color",
    "swipe_undo",
    "circle_clear",
)
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}

STATIC_LABELS = ("idle", "pinch_draw", "open_palm_hover", "fist_erase", "peace_color")
MOTION_LABELS = ("swipe_undo", "circle_clear")

FPS = 30  
WINDOW = 20  
CLIP_FRAMES = 30  
NUM_LANDMARKS = 21
FEATURES = 126  
FEATURE_VERSION = 1
