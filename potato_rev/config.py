import os

# Paths are overridable through the environment so the same code runs locally and on Colab.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.environ.get("POTATO_DATA_DIR", os.path.join(PROJECT_ROOT, "potato_disease_dataset"))
MANIFEST_PATH = os.environ.get("POTATO_MANIFEST", os.path.join(PROJECT_ROOT, "data_audit", "manifest.json"))
OUT_DIR = os.environ.get("POTATO_OUT_DIR", os.path.join(PROJECT_ROOT, "revision_outputs"))

CLASS_NAMES = [
    "Black Scurf",
    "Blackleg",
    "Common Scab",
    "Dry Rot",
    "Healthy Potatoes",
    "Miscellaneous",
    "Pink Rot",
]
NUM_CLASSES = len(CLASS_NAMES)

ARCHITECTURES = [
    "resnet18",
    "resnet50",
    "densenet121",
    "mobilenet_v2",
    "efficientnet_b0",
    "efficientnet_b3",
    "vgg16",
    "vgg19",
]

ACTIVATIONS = ["relu", "lrelu", "prelu", "mish", "swish", "gelu"]

# Activation each backbone was designed and ImageNet-pretrained with.
NATIVE_ACTIVATION = {
    "resnet18": "relu",
    "resnet50": "relu",
    "densenet121": "relu",
    "mobilenet_v2": "relu6",
    "efficientnet_b0": "swish",
    "efficientnet_b3": "swish",
    "vgg16": "relu",
    "vgg19": "relu",
}

# Cross-validation protocol: in rotation k, fold k is the test set, fold (k+1) % N_FOLDS is the
# validation set (early stopping, LR scheduling, LR selection) and the other folds are training data.
N_FOLDS = 5
SEED = 42

# Data
CACHE_SIZE = 256          # images are cached as CACHE_SIZE x CACHE_SIZE uint8 (short side resize + centre crop)
IMAGE_SIZE = 224
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)
AUG_SCALE = (0.8, 1.0)
AUG_RATIO = (3.0 / 4.0, 4.0 / 3.0)
AUG_ROTATION_DEG = 20.0
AUG_BRIGHTNESS = 0.15
AUG_CONTRAST = 0.15
AUG_SATURATION = 0.15

# Optimisation
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 128
LR_GRID = (3e-5, 1e-4, 3e-4)
# After the first pass, validation selected the top of LR_GRID in more than half of the
# batch-normalised rotations and the bottom of it in 40% of VGG-19 rotations, so the grid is
# extended one half-decade step in that direction: upward for networks with batch normalisation,
# downward for the BN-free VGGs.
LR_GRID_EXTENSION = {
    "resnet18": 1e-3, "resnet50": 1e-3, "densenet121": 1e-3, "mobilenet_v2": 1e-3,
    "efficientnet_b0": 1e-3, "efficientnet_b3": 1e-3, "vgg16": 1e-5, "vgg19": 1e-5,
}


def lr_grid(arch: str) -> tuple:
    return tuple(sorted(LR_GRID + (LR_GRID_EXTENSION[arch],)))
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
WARMUP_EPOCHS = 2
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
LR_PLATEAU_PATIENCE = 5
LR_PLATEAU_FACTOR = 0.5
MIN_LR = 1e-6
GRAD_CLIP_NORM = 1.0
# Full 32-bit training: under float16 autocast some activation swaps overflow to NaN, which would
# be indistinguishable from genuine divergence.
USE_AMP = False
LRELU_SLOPE = 0.1
