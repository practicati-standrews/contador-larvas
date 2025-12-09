from pathlib import Path

# --- Información General ---
COMPANY_NAME = "St. Andrews"
APP_NAME = "Contador de Larvas de Chorito"
VERSION = "1.0.0 Web"

# --- Rutas de Archivos ---
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "best yolov11S.pt"
LOGO_PATH = BASE_DIR / "assets" / "logo-st-andrews.png"

# --- Parametros de Detección ---
DEFAULT_CONFIDENCE = 0.25
MIN_CONFIDENCE = 0.1
MAX_CONFIDENCE = 0.9

# --- Colores ---
COLOR_PRIMARY = "#1B4D89"
COLOR_SECONDARY = "#D4A446"
