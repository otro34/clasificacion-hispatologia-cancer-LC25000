"""Configuracion central: rutas, hiperparametros y deteccion de dispositivo.

  - Mac Apple Silicon  -> backend MPS
  - Google Colab       -> backend CUDA
  - Cualquier maquina  -> fallback CPU
"""
from pathlib import Path
import torch


# ----------------------------------------------------------------------------
# Deteccion de entorno y dispositivo
# ----------------------------------------------------------------------------
def get_device() -> torch.device:
    """Devuelve el mejor dispositivo disponible: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


DEVICE = get_device()

# num_workers: en MPS/Mac conviene 0-2 para evitar problemas de fork;
# en CUDA/Colab se puede subir.
NUM_WORKERS = 2 if DEVICE.type == "cuda" else 0
PIN_MEMORY = DEVICE.type == "cuda"


# ----------------------------------------------------------------------------
# Rutas (se resuelven relativas a la raiz del proyecto)
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# En Colab el dataset suele descargarse a /content; localmente a ./data
if in_colab():
    DATA_DIR = Path("/content/lung_colon_image_set")
else:
    DATA_DIR = PROJECT_ROOT / "data" / "lung_colon_image_set"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
FIGURES_DIR = OUTPUTS_DIR / "figures"
for _d in (CHECKPOINTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# Clases del dataset LC25000 (5 clases)
# ----------------------------------------------------------------------------
# Nombres de carpeta -> etiqueta legible
CLASS_NAMES = {
    "colon_n": "Colon benigno",
    "colon_aca": "Colon adenocarcinoma",
    "lung_n": "Pulmon benigno",
    "lung_aca": "Pulmon adenocarcinoma",
    "lung_scc": "Pulmon carcinoma escamoso",
}
CLASSES = list(CLASS_NAMES.keys())
NUM_CLASSES = len(CLASSES)


# ----------------------------------------------------------------------------
# Hiperparametros
# ----------------------------------------------------------------------------
SEED = 42
IMG_SIZE = 224

# Normalizacion estandar de ImageNet (necesaria para modelos preentrenados)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Split estratificado
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Entrenamiento
BATCH_SIZE = 32 if DEVICE.type != "cpu" else 16
EPOCHS_HEAD = 4        # Etapa A: solo la cabeza (feature extraction)
EPOCHS_FINETUNE = 8    # Etapa B: fine-tuning de capas superiores
LR_HEAD = 1e-3
LR_FINETUNE = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 4

# Modelo base por defecto (timm). Alternativa de comparacion: "resnet50"
MODEL_NAME = "efficientnet_b0"


def summary() -> str:
    """Resumen legible de la configuracion activa."""
    return (
        f"Dispositivo : {DEVICE}\n"
        f"En Colab    : {in_colab()}\n"
        f"num_workers : {NUM_WORKERS}\n"
        f"Data dir    : {DATA_DIR}\n"
        f"Modelo      : {MODEL_NAME}\n"
        f"Clases      : {NUM_CLASSES} -> {CLASSES}\n"
        f"Batch size  : {BATCH_SIZE}\n"
    )


if __name__ == "__main__":
    print(summary())
