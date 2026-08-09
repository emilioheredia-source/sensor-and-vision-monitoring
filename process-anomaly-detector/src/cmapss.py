"""Loading and basic handling for the NASA C-MAPSS turbofan degradation data.

Data format (space separated, no header), one row per engine per cycle:

    col 1      unit number
    col 2      time, in cycles
    col 3-5    operational settings 1..3
    col 6-26   sensor measurements 1..21

Four subsets. FD001 is the simplest: 100 engines, one operating condition,
one fault mode (HPC degradation). Every engine in the training set runs to
failure, so the last cycle of each unit is its failure point.
"""

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "CMAPSS"

SETTING_COLS = [f"setting_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"s_{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle"] + SETTING_COLS + SENSOR_COLS

# What each sensor physically measures. The data file only numbers them; these
# names come from Saxena, Goebel, Simon & Eklund, "Damage Propagation Modeling
# for Aircraft Engine Run-to-Failure Simulation", PHM08 (PDF is in data/CMAPSS).
#
# LPC/HPC = low/high pressure compressor, LPT/HPT = low/high pressure turbine.
SENSOR_NAMES = {
    "s_1": "T2 — total temperature at fan inlet",
    "s_2": "T24 — total temperature at LPC outlet",
    "s_3": "T30 — total temperature at HPC outlet",
    "s_4": "T50 — total temperature at LPT outlet",
    "s_5": "P2 — pressure at fan inlet",
    "s_6": "P15 — total pressure in bypass duct",
    "s_7": "P30 — total pressure at HPC outlet",
    "s_8": "Nf — physical fan speed",
    "s_9": "Nc — physical core speed",
    "s_10": "epr — engine pressure ratio (P50/P2)",
    "s_11": "Ps30 — static pressure at HPC outlet",
    "s_12": "phi — fuel flow / Ps30",
    "s_13": "NRf — corrected fan speed",
    "s_14": "NRc — corrected core speed",
    "s_15": "BPR — bypass ratio",
    "s_16": "farB — burner fuel-air ratio",
    "s_17": "htBleed — bleed enthalpy",
    "s_18": "Nf_dmd — DEMANDED fan speed (setpoint)",
    "s_19": "PCNfR_dmd — DEMANDED corrected fan speed (setpoint)",
    "s_20": "W31 — HPT coolant bleed",
    "s_21": "W32 — LPT coolant bleed",
}

# The six sensors with zero variance in FD001 are not faulty instruments.
# FD001 fixes the operating condition at sea level with one throttle setting,
# so ambient conditions (T2, P2), the derived pressure ratio, the burner
# fuel-air ratio, and the two DEMANDED setpoints cannot move. In FD002, which
# has six operating conditions, these do vary.


def load(subset: str = "FD001", split: str = "train") -> pd.DataFrame:
    """Read one C-MAPSS file into a DataFrame with named columns."""
    path = DATA_DIR / f"{split}_{subset}.txt"
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)
    return df


def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Add remaining useful life, in cycles, for training data.

    Valid only where every unit runs to failure (the training split). RUL is
    then just how many cycles remain before that unit's last recorded cycle.
    """
    last_cycle = df.groupby("unit")["cycle"].transform("max")
    out = df.copy()
    out["rul"] = last_cycle - out["cycle"]
    return out


def constant_sensors(df: pd.DataFrame, tol: float = 1e-9) -> list[str]:
    """Sensors whose value never changes: no information, safe to drop."""
    spread = df[SENSOR_COLS].std()
    return sorted(spread[spread <= tol].index.tolist())
