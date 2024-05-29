import os
import os.path as op
from pathlib import Path

from Misc.config import WORK_DIR, IS_CRYPTO

def get_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
    return dir

DATA_DIR = get_dir(op.join(WORK_DIR, "data"))
PROCESSED_DATA_DIR = get_dir(op.join(DATA_DIR, "processed_data"))
STOCKS_SAVEPATH = os.path.join(DATA_DIR, "stocks_dataset")
CRYPTOS_SAVEPATH = os.path.join(DATA_DIR, "cryptos_dataset")
RAW_DATA_DIR = op.join(CRYPTOS_SAVEPATH, "raw_data")

CACHE_DIR = Path("./CACHE_DIR")
PORTFOLIO = Path("./CACHE_DIR/PORTFOLIO")
if not os.path.isdir(PORTFOLIO):
    os.makedirs(PORTFOLIO, exist_ok=True)

BAR_WIDTH = 3
LINE_WIDTH = 1
if IS_CRYPTO:
    IMAGE_WIDTH = {1: BAR_WIDTH * 1, 7: BAR_WIDTH * 7, 30: BAR_WIDTH *30, 90: BAR_WIDTH * 90}
    IMAGE_HEIGHT = {1: 16, 7: 32, 30: 64, 90: 96}
else:
    IMAGE_WIDTH = {5: BAR_WIDTH * 5, 20: BAR_WIDTH * 20, 60: BAR_WIDTH * 60}
    IMAGE_HEIGHT = {5: 32, 20: 64, 60: 96}
VOLUME_CHART_GAP = 1
BACKGROUND_COLOR = 0
CHART_COLOR = 255

"""
week := mini
month := normal
quarter := large
"""
FREQ_DICT = {5: "week", 20: "month", 60: "quarter", 65: "quarter", 260: "year"}
# FREQ_DICT = {5: "month", 20: "month", 60: "month", 65: "month", 260: "month"}
if IS_CRYPTO:
    FREQ_DICT = {1: "day", 7: "week", 30: "month", 90: "quarter", 360: "year", 365: "year"}
else:
    FREQ_DICT = {5: "week", 20: "week", 60: "week", 65: "week", 260: "week"}



INTERNATIONAL_COUNTRIES = [
    "Japan",
    "UnitedKingdom",
    "China",
    "SouthKorea",
    "India",
    "Canada",
    "Germany",
    "Australia",
    "HongKong",
    "France",
    "Singapore",
    "Italy",
    "Sweden",
    "Switzerland",
    "Netherlands",
    "Norway",
    "Spain",
    "Belgium",
    "Greece",
    "Denmark",
    "Russia",
    "Finland",
    "NewZealand",
    "Austria",
    "Portugal",
    "Ireland",
]
