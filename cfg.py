from pathlib import Path

# ----------------------------
# Configurações gerais
# ----------------------------
WIDTH = 900
HEIGHT = 600
DISPLAY_WIDTH = 600
DISPLAY_HEIGHT = 600
FPS = 24

NUM_IMAGES_PER_PERSON = 50
CAMERA_INDEX = 0  # Índice padrão da webcam

# ----------------------------
# Diretórios e arquivos
# ----------------------------
CONFIG_FILE = Path("config.json")
ASSETS_DIR = Path("assets")
DATASETS_DIR = Path("datasets")
DATASETS_DIR.mkdir(exist_ok=True)

MODEL_FILE = Path("modelo_lbph.yml")
LABELS_FILE = Path("labels.pkl")

# ----------------------------
# Cores
# ----------------------------
COLORS = {
    "bg": (30, 30, 30),
    "panel": (45, 45, 45),
    "button": (70, 70, 70),
    "button_hover": (90, 90, 90),
    "text": (230, 230, 230),
    "correct": (0, 200, 0),
    "incorrect": (200, 0, 0)
}
