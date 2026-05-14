#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create fake phase velocity data for each station.

Input station file:
    output/station_with_topo.lst

Output:
    output/fake_data_CHT/{sta}_data/{sta}.ph

Station name is forced to zfill(5), e.g.
    100 -> 00100

Output file format:
    period  vph  unc

Period array:
    0.1 -> 10.0 sec, low to high
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ==================================================
# Parameters
# ==================================================
STATION_FILE = Path("output/station_with_topo.lst")

OUT_ROOT = Path("output/fake_data_CHT")

N_PERIODS = 32
PERIOD_MIN = 0.1
PERIOD_MAX = 10.0

# Fake values
FAKE_VPH = 0.5       # km/s
FAKE_UNC = 0.05      # km/s

# Output format
FMT = "%.6f %.6f %.6f"


# ==================================================
# Read station file
# ==================================================
def read_station_file(station_file):
    df = pd.read_csv(
        station_file,
        sep=r"\s+",
        comment="#",
        dtype={"name": str},
    )

    if "name" not in df.columns:
        raise ValueError(f"Missing column 'name' in station file: {station_file}")

    df["name"] = df["name"].astype(str)

    return df


# ==================================================
# Normalize station name to zfill(5)
# ==================================================
def normalize_station_name(sta_raw):
    sta_raw = str(sta_raw).strip()

    try:
        return f"{int(float(sta_raw)):05d}"
    except Exception:
        return sta_raw.zfill(5)


# ==================================================
# Make fake Vph data
# ==================================================
def make_fake_vph_data():
    # Low to high period:
    # 0.1, ..., 10.0 sec
    periods = np.logspace(
        np.log10(PERIOD_MIN),
        np.log10(PERIOD_MAX),
        N_PERIODS,
    )

    vph = np.full_like(periods, FAKE_VPH, dtype=float)
    unc = np.full_like(periods, FAKE_UNC, dtype=float)

    data = np.column_stack([periods, vph, unc])

    return data


# ==================================================
# Main
# ==================================================
def main():
    if not STATION_FILE.exists():
        raise FileNotFoundError(f"Station file not found: {STATION_FILE}")

    sta_df = read_station_file(STATION_FILE)
    fake_data = make_fake_vph_data()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    n_out = 0

    for _, row in sta_df.iterrows():
        sta = normalize_station_name(row["name"])

        sta_dir = OUT_ROOT / f"{sta}_data"
        sta_dir.mkdir(parents=True, exist_ok=True)

        out_file = sta_dir / f"{sta}.ph"

        np.savetxt(
            out_file,
            fake_data,
            fmt=FMT,
        )

        print(f"[OK] {out_file}")
        n_out += 1

    print("==================================================")
    print("[DONE]")
    print(f"Station file : {STATION_FILE}")
    print(f"Output root  : {OUT_ROOT}")
    print(f"N stations   : {n_out}")
    print(f"N periods    : {N_PERIODS}")
    print(f"Period range : {PERIOD_MIN} - {PERIOD_MAX} s")
    print(f"Period order : low to high")
    print(f"Fake Vph     : {FAKE_VPH} km/s")
    print(f"Fake unc     : {FAKE_UNC} km/s")
    print("==================================================")


if __name__ == "__main__":
    main()