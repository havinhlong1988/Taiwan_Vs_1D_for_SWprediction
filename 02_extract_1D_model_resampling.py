#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract 1D Vs/Vp model at station locations using inverse-distance weighting,
then plot individual 1D profiles and full 2D velocity sections using PyGMT.

Main output model file
----------------------
output/1D_extract_dist_weighting/{sta}.mod

Output columns:
    dep long lat vs vp

Important updates
-----------------
1) Output extension is .mod, not .dat.
2) Each station model is resampled to a regular depth interval.
   Default: RESAMPLE_DZ_KM = 0.5 km.
3) Longitude and latitude are written on every model row:
       dep long lat vs vp

Inputs
------
1) output/station_with_topo.lst
   Expected columns:
       name long lat distance elevation_m idx

2) input/NTW-ANT-L21_interpolated.txt
   Model format:
       Dimension: nlon nlat ndep
       lon lat dep vs dvs uncertainty

Method
------
For each station:
- Find the horizontal model cell covering the station.
- Use the surrounding grid nodes at each depth.
- Estimate Vs by inverse-distance weighting.
- Convert Vp using:
      vp = vs * VpVs
- Resample the vertical profile to RESAMPLE_DZ_KM.
- Save:
      dep long lat vs vp

Outputs
-------
1) 1D model data:
   output/1D_extract_dist_weighting/{sta}.mod

2) 1D figures:
   figures/1D_extract_dist_weighting/{sta}.png

3) Full absolute profile:
   figures/profiles.png

4) Full relative perturbation profile:
   figures/profiles_relative.png

5) Horizontal model-node/station map:
   figures/model_nodes_stations.png
"""

import os
import re
import numpy as np
import pandas as pd
import pygmt
import xarray as xr

from pathlib import Path
from scipy.interpolate import interp1d
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.ndimage import gaussian_filter


# ==================================================
# Parameters
# ==================================================
STATION_FILE = "output/station_with_topo.lst"
MODEL_FILE = "input/NTW-ANT-L21_interpolated.txt"

OUT_DATA_DIR = Path("output/CHT_models")
OUT_FIG_1D_DIR = Path("figures/CHT_models")
OUT_PROFILE_FIG = Path("figures/profiles.png")
OUT_PROFILE_REL_FIG = Path("figures/profiles_relative.png")
OUT_NODE_STA_FIG = Path("figures/model_nodes_stations.png")

OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG_1D_DIR.mkdir(parents=True, exist_ok=True)
OUT_PROFILE_FIG.parent.mkdir(parents=True, exist_ok=True)

# Extract one station only, e.g. "00800".
# Set None for all stations.
extract_sta = None
# extract_sta = "00800"

# Depth range for extraction from the input 3D/2D model.
DEPTH_MIN = -2.0
DEPTH_MAX = None   # None = deepest available depth

# Resampling interval for output .mod file.
# If None or <= 0, keep original extracted model depths.
RESAMPLE_DZ_KM = 0.5

# Output model file extension.
OUTPUT_MODEL_EXT = ".mod"

# Vp/Vs ratio used to build Vp from Vs.
VpVs = 1.73

# IDW power.
IDW_POWER = 1.0

# If station is outside model grid:
# False -> skip station
# True  -> use nearest 4 horizontal nodes
ALLOW_NEAREST_FALLBACK = False

# Multi-core extraction only.
# PyGMT plotting is kept serial because GMT modern mode is not safe in multiprocessing.
NCORE = 20
# NCORE = None   # None means maximum CPU - 1

# Small number.
EPS = 1.0e-12

# Section plotting.
SECTION_DX = 0.05       # km
TOPO_NY = 180           # topographic colored-fill vertical resolution
FIG_DPI = 300

# Map setting.
MAP_PAD_LON = 0.02
MAP_PAD_LAT = 0.02
LABEL_EVERY_N_STATION = 10

# Horizontal map velocity background depth.
# The script will use the closest available model depth.
MAP_VEL_DEPTH = 0.0

# CPTs.
CPT_VS_AUTO = True
CPT_VS = "seis"
CPT_VS_MAN = Path("input/cpt/Vs_abs.cpt")
CPT_TOPO = "geo"

# Relative perturbation profile.
RELATIVE_PERCENT = 2.0
CPT_RELATIVE = "polar"
RELATIVE_CPT_STEP = 0.1

# 1D figure setting.
DEPTH_PLOT_MIN = DEPTH_MIN
DEPTH_PLOT_MAX = None


# ==================================================
# Global variables for multiprocessing
# ==================================================
G_MODEL = None
G_LONS = None
G_LATS = None
G_DEPS = None
G_HORIZONTAL_NODES = None


def init_worker(model, lons, lats, deps, horizontal_nodes):
    global G_MODEL, G_LONS, G_LATS, G_DEPS, G_HORIZONTAL_NODES
    G_MODEL = model
    G_LONS = lons
    G_LATS = lats
    G_DEPS = deps
    G_HORIZONTAL_NODES = horizontal_nodes


# ==================================================
# Utilities
# ==================================================
def get_ncore():
    if NCORE is not None:
        return max(1, int(NCORE))

    ncpu = os.cpu_count()
    if ncpu is None:
        return 1

    return max(1, ncpu - 1)


def haversine_km(lon1, lat1, lon2, lat2):
    """
    Great-circle distance in km.
    Works with scalars or arrays.
    """
    r_earth = 6371.0

    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(a))

    return r_earth * c


def make_xarray_grid(z, x, y, name="grid"):
    """
    Make PyGMT-friendly xarray DataArray.
    z shape must be [len(y), len(x)].
    """
    return xr.DataArray(
        z,
        coords={"y": y, "x": x},
        dims=("y", "x"),
        name=name,
    )


def compute_vs_perturbation_percent(vs_grid):
    """
    Compute relative Vs perturbation in percent.

    Formula:
        dVs(%) = (Vs(x,z) - mean_Vs(z)) / mean_Vs(z) * 100

    Mean is calculated independently for each depth.
    """
    vs_grid = np.asarray(vs_grid, dtype=float)
    mean_vs_depth = np.nanmean(vs_grid, axis=1)
    rel_grid = np.full_like(vs_grid, np.nan, dtype=float)

    for i in range(vs_grid.shape[0]):
        mean_vs = mean_vs_depth[i]
        if not np.isfinite(mean_vs) or mean_vs == 0:
            continue
        rel_grid[i, :] = (vs_grid[i, :] - mean_vs) / mean_vs * 100.0

    return rel_grid, mean_vs_depth


def make_or_get_vs_cpt(vs_min, vs_max):
    """
    Prepare CPT for Vs plotting.

    If CPT_VS_AUTO = True:
        make a new CPT from CPT_VS.

    If CPT_VS_AUTO = False:
        use CPT_VS_MAN if it exists.
        If not found, automatically fall back to auto CPT.
    """
    if not np.isfinite(vs_min) or not np.isfinite(vs_max):
        vs_min, vs_max = 2.0, 5.0

    if CPT_VS_AUTO:
        v0 = np.floor(vs_min / 0.1) * 0.1
        v1 = np.ceil(vs_max / 0.1) * 0.1

        if v0 == v1:
            v0 -= 0.1
            v1 += 0.1

        pygmt.makecpt(
            cmap=CPT_VS,
            series=[v0, v1, 0.05],
            continuous=True,
        )
        return True, None

    if CPT_VS_MAN.exists():
        return False, str(CPT_VS_MAN)

    print(f"[WARN] Manual CPT not found: {CPT_VS_MAN}")
    print("[WARN] Fall back to auto CPT.")

    v0 = np.floor(vs_min / 0.1) * 0.1
    v1 = np.ceil(vs_max / 0.1) * 0.1

    if v0 == v1:
        v0 -= 0.1
        v1 += 0.1

    pygmt.makecpt(
        cmap=CPT_VS,
        series=[v0, v1, 0.05],
        continuous=True,
    )
    return True, None


def make_relative_cpt(rel_min, rel_max):
    """
    Make CPT for relative perturbation.

    The main color range is controlled by RELATIVE_PERCENT, e.g. +/- 2%.
    Values below/above the range will be clipped to the min/max CPT colors.
    """
    cmin = -abs(float(RELATIVE_PERCENT))
    cmax = abs(float(RELATIVE_PERCENT))

    if cmin == cmax:
        cmin = -2.0
        cmax = 2.0

    pygmt.makecpt(
        cmap=CPT_RELATIVE,
        series=[cmin, cmax, RELATIVE_CPT_STEP],
        continuous=True,
        background=True,
    )

    print(f"[INFO] Relative perturbation actual min/max: {rel_min:.3f} / {rel_max:.3f} %")
    print(f"[INFO] Relative perturbation color scale: {cmin:.3f} to {cmax:.3f} %")


def grdimage_with_vs_cpt(fig, grid, use_current_cpt, cpt_file):
    """
    Plot grid using either current CPT or manual CPT file.
    """
    if use_current_cpt:
        fig.grdimage(
            grid=grid,
            cmap=True,
            nan_transparent=True,
        )
    else:
        fig.grdimage(
            grid=grid,
            cmap=cpt_file,
            nan_transparent=True,
        )


def colorbar_with_vs_cpt(fig, use_current_cpt, cpt_file, position, frame):
    """
    Draw colorbar using either current CPT or manual CPT file.
    """
    if use_current_cpt:
        fig.colorbar(
            position=position,
            frame=frame,
        )
    else:
        fig.colorbar(
            cmap=cpt_file,
            position=position,
            frame=frame,
        )


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

    required = ["name", "long", "lat", "distance"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in station file: {station_file}")

    df["name"] = df["name"].astype(str)
    df["long"] = pd.to_numeric(df["long"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["distance"] = pd.to_numeric(df["distance"], errors="coerce")

    if "elevation_m" in df.columns:
        df["elevation_m"] = pd.to_numeric(df["elevation_m"], errors="coerce")
    else:
        df["elevation_m"] = np.nan

    df = df.dropna(subset=["long", "lat", "distance"]).reset_index(drop=True)
    df = df.sort_values("distance").reset_index(drop=True)

    return df


# ==================================================
# Read model file
# ==================================================
def find_model_header(model_file):
    dim = None
    header = None
    data_start_line = None

    with open(model_file, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if line.strip().startswith("Dimension"):
            nums = re.findall(r"[-+]?\d+", line)
            if len(nums) >= 3:
                dim = tuple(map(int, nums[:3]))

            for j in range(i + 1, len(lines)):
                temp = lines[j].strip()
                if temp == "":
                    continue
                header = temp.split()
                data_start_line = j + 1
                break
            break

    if dim is None:
        raise ValueError(f"Cannot find Dimension line in: {model_file}")

    if header is None:
        raise ValueError(f"Cannot find header line after Dimension line in: {model_file}")

    return dim, header, data_start_line


def read_model_file(model_file):
    dim, header, data_start_line = find_model_header(model_file)

    print(f"[INFO] Model dimension: nlon={dim[0]}, nlat={dim[1]}, ndep={dim[2]}")
    print(f"[INFO] Model header: {header}")

    model = pd.read_csv(
        model_file,
        sep=r"\s+",
        skiprows=data_start_line,
        names=header,
        comment="#",
    )

    for col in header:
        model[col] = pd.to_numeric(model[col], errors="coerce")

    required = ["lon", "lat", "dep", "vs"]
    for col in required:
        if col not in model.columns:
            raise ValueError(f"Missing required model column: {col}")

    model = model.dropna(subset=["lon", "lat", "dep", "vs"]).reset_index(drop=True)

    return model, dim, header


# ==================================================
# Grid helpers
# ==================================================
def find_bracketing_values(values, x):
    values = np.asarray(values, dtype=float)
    values = np.sort(np.unique(values))

    if x < values.min() or x > values.max():
        return None, None

    idx_exact = np.where(np.isclose(values, x, atol=1e-10))[0]
    if len(idx_exact) > 0:
        val = values[idx_exact[0]]
        return val, val

    lower = values[values < x]
    upper = values[values > x]

    if len(lower) == 0 or len(upper) == 0:
        return None, None

    return lower.max(), upper.min()


def get_cell_nodes(sta_lon, sta_lat, lons, lats):
    lon1, lon2 = find_bracketing_values(lons, sta_lon)
    lat1, lat2 = find_bracketing_values(lats, sta_lat)

    if lon1 is None or lat1 is None:
        return None

    nodes = [
        (lon1, lat1),
        (lon1, lat2),
        (lon2, lat1),
        (lon2, lat2),
    ]

    nodes_unique = []
    for node in nodes:
        if node not in nodes_unique:
            nodes_unique.append(node)

    return nodes_unique


def get_nearest_4_nodes(sta_lon, sta_lat, horizontal_nodes):
    tmp = horizontal_nodes.copy()
    tmp["dist_km"] = haversine_km(
        sta_lon,
        sta_lat,
        tmp["lon"].values,
        tmp["lat"].values,
    )
    tmp = tmp.sort_values("dist_km").head(4)

    return list(zip(tmp["lon"].values, tmp["lat"].values))


# ==================================================
# IDW and vertical resampling
# ==================================================
def idw_vs(sta_lon, sta_lat, node_df):
    node_lons = node_df["lon"].values
    node_lats = node_df["lat"].values
    node_vs = node_df["vs"].values.astype(float)

    dist = haversine_km(sta_lon, sta_lat, node_lons, node_lats)

    if np.any(dist < EPS):
        idx = np.argmin(dist)
        return float(node_vs[idx])

    weights = 1.0 / np.power(dist, IDW_POWER)
    weights = weights / np.sum(weights)

    return float(np.sum(weights * node_vs))


def resample_1d_profile(df_1d, sta_lon, sta_lat, dz_km=RESAMPLE_DZ_KM):
    """
    Resample extracted 1D model to regular depth spacing.

    Input columns required:
        dep, vs, vp

    Output columns:
        dep, long, lat, vs, vp
    """
    if df_1d is None or len(df_1d) == 0:
        return df_1d

    df = df_1d.copy()
    df = df.sort_values("dep").drop_duplicates(subset=["dep"], keep="last")
    df = df.reset_index(drop=True)

    dep_old = df["dep"].values.astype(float)
    vs_old = df["vs"].values.astype(float)

    ok = np.isfinite(dep_old) & np.isfinite(vs_old)
    dep_old = dep_old[ok]
    vs_old = vs_old[ok]

    if len(dep_old) == 0:
        return pd.DataFrame(columns=["dep", "long", "lat", "vs", "vp"])

    if len(dep_old) < 2:
        vs_new = vs_old.copy()
        dep_new = dep_old.copy()
    else:
        if dz_km is None or float(dz_km) <= 0:
            dep_new = dep_old.copy()
        else:
            dz_km = float(dz_km)
            dep_min = float(dep_old.min())
            dep_max = float(dep_old.max())

            dep_new = np.arange(dep_min, dep_max + dz_km * 0.5, dz_km)
            dep_new = dep_new[dep_new <= dep_max + 1.0e-8]

            # Make sure exact deepest original depth is kept.
            if len(dep_new) == 0 or not np.isclose(dep_new[-1], dep_max):
                dep_new = np.r_[dep_new, dep_max]

        vs_new = np.interp(dep_new, dep_old, vs_old)

    vp_new = vs_new * VpVs

    out = pd.DataFrame(
        {
            "dep": dep_new,
            "long": np.full(len(dep_new), float(sta_lon)),
            "lat": np.full(len(dep_new), float(sta_lat)),
            "vs": vs_new,
            "vp": vp_new,
        }
    )

    return out


# ==================================================
# Extract one station
# ==================================================
def extract_station_1d_core(sta_dict, model, lons, lats, deps, horizontal_nodes):
    sta_name = str(sta_dict["name"])
    sta_lon = float(sta_dict["long"])
    sta_lat = float(sta_dict["lat"])

    nodes = get_cell_nodes(sta_lon, sta_lat, lons, lats)

    if nodes is None:
        if ALLOW_NEAREST_FALLBACK:
            nodes = get_nearest_4_nodes(sta_lon, sta_lat, horizontal_nodes)
            warn = f"[WARN] Station {sta_name} outside grid -> nearest 4 nodes fallback."
        else:
            return sta_name, None, f"[SKIP] Station {sta_name} outside model grid."
    else:
        warn = None

    rows = []

    for dep in deps:
        dep_rows = []

        for lon, lat in nodes:
            sub = model[
                np.isclose(model["lon"], lon)
                & np.isclose(model["lat"], lat)
                & np.isclose(model["dep"], dep)
            ]

            if len(sub) > 0:
                dep_rows.append(sub.iloc[0])

        if len(dep_rows) == 0:
            continue

        dep_df = pd.DataFrame(dep_rows)
        vs = idw_vs(sta_lon, sta_lat, dep_df)
        vp = vs * VpVs

        rows.append(
            {
                "dep": float(dep),
                "vs": float(vs),
                "vp": float(vp),
            }
        )

    if len(rows) == 0:
        return sta_name, None, f"[SKIP] No valid model extracted for station {sta_name}"

    out = pd.DataFrame(rows)
    out = out.sort_values("dep").reset_index(drop=True)
    out = resample_1d_profile(out, sta_lon=sta_lon, sta_lat=sta_lat, dz_km=RESAMPLE_DZ_KM)

    return sta_name, out, warn


def extract_station_worker(sta_dict):
    return extract_station_1d_core(
        sta_dict=sta_dict,
        model=G_MODEL,
        lons=G_LONS,
        lats=G_LATS,
        deps=G_DEPS,
        horizontal_nodes=G_HORIZONTAL_NODES,
    )


# ==================================================
# Station selection
# ==================================================
def select_output_stations(sta_df, extract_sta):
    if extract_sta is None:
        return sta_df.copy()

    extract_sta = str(extract_sta)
    selected = sta_df[sta_df["name"].astype(str) == extract_sta].copy()

    if len(selected) == 0:
        try:
            target_int = int(extract_sta)
            sta_int = pd.to_numeric(sta_df["name"], errors="coerce")
            selected = sta_df[sta_int == target_int].copy()
        except Exception:
            pass

    if len(selected) == 0:
        raise ValueError(f"Cannot find station: {extract_sta}")

    return selected.reset_index(drop=True)


def find_station_row_by_name(sta_df, sta_name):
    sta_name = str(sta_name)

    sub = sta_df[sta_df["name"].astype(str) == sta_name]
    if len(sub) > 0:
        return sub.iloc[0]

    try:
        target_int = int(sta_name)
        sta_int = pd.to_numeric(sta_df["name"], errors="coerce")
        sub = sta_df[sta_int == target_int]
        if len(sub) > 0:
            return sub.iloc[0]
    except Exception:
        pass

    return None


# ==================================================
# Save 1D model data
# ==================================================
def save_1d_data(sta_name, df_1d):
    """
    Save station 1D model as .mod.

    Output columns:
        dep long lat vs vp
    """
    out_file = OUT_DATA_DIR / f"{sta_name}{OUTPUT_MODEL_EXT}"

    df = df_1d.copy()

    if "long" not in df.columns:
        df["long"] = np.nan
    if "lat" not in df.columns:
        df["lat"] = np.nan
    if "vp" not in df.columns:
        df["vp"] = df["vs"] * VpVs

    # cols = ["dep", "long", "lat", "vs", "vp"]
    cols = ["dep", "vs", "vp"]

    np.savetxt(
        out_file,
        df[cols].values,
        # fmt="%.6f %.6f %.6f %.6f %.6f",
        fmt="%.6f %.6f %.6f",
        # header="dep vs vp",
        # comments="# ",
    )

    print(f"[OK] Saved 1D model: {out_file}")


# ==================================================
# PyGMT 1D plot
# ==================================================
def plot_1d_figure_pygmt(sta_row, df_1d):
    sta_name = str(sta_row["name"])
    lon = float(sta_row["long"])
    lat = float(sta_row["lat"])
    dist = float(sta_row["distance"])
    elev = float(sta_row["elevation_m"]) if np.isfinite(sta_row["elevation_m"]) else np.nan

    out_png = OUT_FIG_1D_DIR / f"{sta_name}.png"

    dep = df_1d["dep"].values.astype(float)
    vs = df_1d["vs"].values.astype(float)
    vp = df_1d["vp"].values.astype(float)

    dep_min = DEPTH_PLOT_MIN
    dep_max = dep.max() if DEPTH_PLOT_MAX is None else DEPTH_PLOT_MAX

    vmin = np.nanmin([np.nanmin(vs), np.nanmin(vp)])
    vmax = np.nanmax([np.nanmax(vs), np.nanmax(vp)])
    vpad = 0.08 * (vmax - vmin) if vmax > vmin else 0.2
    region = [vmin - vpad, vmax + vpad, dep_min, dep_max]

    fig = pygmt.Figure()

    pygmt.config(
        FONT_LABEL="12p,Times-Bold,black",
        FONT_ANNOT_PRIMARY="11p,Times-Roman,black",
        FONT_TITLE="14p,Times-Bold,black",
        MAP_FRAME_TYPE="plain",
    )

    fig.basemap(
        region=region,
        projection="X6c/-8c",
        frame=[
            f"WSne+tStation {sta_name}",
            'xagf+l"Velocity (km/s)"',
            'yagf+l"Depth (km)"',
        ],
    )

    fig.plot(
        x=vs,
        y=dep,
        pen="1p,blue",
        label="Vs",
    )

    fig.plot(
        x=vs,
        y=dep,
        style="c0.005c",
        fill="blue",
        pen="1p,blue",
    )

    fig.plot(
        x=vp,
        y=dep,
        pen="1p,red",
        label=f"Vp = Vs x {VpVs:.2f}",
    )

    fig.legend(
        position="JTR+jTR+o0.15c/0.15c",
        box="+gwhite+p0.5p,black",
    )

    info = (
        f"Lon: {lon:.5f}\n"
        f"Lat: {lat:.5f}\n"
        f"Dist: {dist:.2f} km\n"
        f"Elev: {elev:.1f} m\n"
        f"dz: {RESAMPLE_DZ_KM} km"
    )

    fig.text(
        x=region[1],
        y=region[3],
        text=info,
        justify="TR",
        font="8p,Times-Roman,black",
        offset="-0.15c/-0.2c",
        fill="white@20",
        pen="0.4p,black",
    )

    fig.savefig(out_png, dpi=FIG_DPI)
    print(f"[OK] Saved 1D figure: {out_png}")


def plot_1d_worker(args):
    sta_dict, df_1d = args
    plot_1d_figure_pygmt(pd.Series(sta_dict), df_1d)
    return str(sta_dict["name"])


# ==================================================
# Build profile section grid
# ==================================================
def build_section_grid(section_profiles, sta_df_all):
    valid_rows = []

    for _, row in sta_df_all.iterrows():
        sta = str(row["name"])
        if sta in section_profiles and section_profiles[sta] is not None:
            valid_rows.append(row)

    if len(valid_rows) < 2:
        raise ValueError("Need at least 2 valid stations to build the full 2D profile.")

    sta_valid = pd.DataFrame(valid_rows).copy()
    sta_valid = sta_valid.sort_values("distance").reset_index(drop=True)

    first_sta = str(sta_valid.iloc[0]["name"])
    deps = section_profiles[first_sta]["dep"].values.copy()

    x_sta = sta_valid["distance"].values.astype(float)
    topo_sta = sta_valid["elevation_m"].values.astype(float)
    names_sta = sta_valid["name"].astype(str).values

    x_min = np.nanmin(x_sta)
    x_max = np.nanmax(x_sta)
    x_grid = np.arange(x_min, x_max + SECTION_DX * 0.5, SECTION_DX)

    if np.all(~np.isfinite(topo_sta)):
        topo_sta = np.zeros_like(x_sta)

    topo_func = interp1d(
        x_sta,
        topo_sta,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    topo_grid = topo_func(x_grid)

    n_sta = len(sta_valid)
    n_dep = len(deps)
    vs_sta_mat = np.full((n_sta, n_dep), np.nan)

    for i, sta in enumerate(names_sta):
        df_1d = section_profiles[sta].sort_values("dep")
        dep_arr = df_1d["dep"].values.astype(float)
        vs_arr = df_1d["vs"].values.astype(float)

        if len(dep_arr) == len(deps) and np.allclose(dep_arr, deps):
            vs_sta_mat[i, :] = vs_arr
        else:
            f = interp1d(
                dep_arr,
                vs_arr,
                kind="linear",
                bounds_error=False,
                fill_value=np.nan,
            )
            vs_sta_mat[i, :] = f(deps)

    vs_grid = np.full((n_dep, len(x_grid)), np.nan)

    for j in range(n_dep):
        y = vs_sta_mat[:, j]
        ok = np.isfinite(y) & np.isfinite(x_sta)

        if np.sum(ok) < 2:
            continue

        f = interp1d(
            x_sta[ok],
            y[ok],
            kind="linear",
            bounds_error=False,
            fill_value=(y[ok][0], y[ok][-1]),
        )
        vs_grid[j, :] = f(x_grid)

    return {
        "sta_valid": sta_valid,
        "deps": deps,
        "x_sta": x_sta,
        "topo_sta": topo_sta,
        "x_grid": x_grid,
        "topo_grid": topo_grid,
        "vs_grid": vs_grid,
    }


# ==================================================
# PyGMT full profile plot
# ==================================================
def plot_topography_panel(fig, title, sta_valid, x_sta, topo_sta, x_grid, topo_grid, extract_sta=None):
    x_min = float(np.nanmin(x_grid))
    x_max = float(np.nanmax(x_grid))

    topo_min = float(np.nanmin(topo_grid))
    topo_max = float(np.nanmax(topo_grid))
    topo_pad = max(50.0, 0.15 * (topo_max - topo_min if topo_max > topo_min else 100.0))

    topo_ymin = topo_min - topo_pad
    topo_ymax = topo_max + 0.25 * topo_pad

    y_topo = np.linspace(topo_ymin, topo_ymax, TOPO_NY)
    _, y_mesh = np.meshgrid(x_grid, y_topo)

    z_topo = np.tile(topo_grid.reshape(1, -1), (TOPO_NY, 1))
    z_topo[y_mesh > z_topo] = np.nan

    topo_da = make_xarray_grid(
        z=z_topo,
        x=x_grid,
        y=y_topo,
        name="topography",
    )

    pygmt.makecpt(
        cmap=CPT_TOPO,
        series=[np.floor(topo_min / 50) * 50, np.ceil(topo_max / 50) * 50, 10],
        continuous=True,
    )

    fig.basemap(
        region=[x_min, x_max, topo_ymin, topo_ymax],
        projection="X18c/3.0c",
        frame=[
            f"WSne+t{title}",
            'xaf+l"Distance along profile (km)"',
            'yaf+l"Elevation (m)"',
        ],
    )

    fig.grdimage(
        grid=topo_da,
        cmap=True,
        nan_transparent=True,
    )

    fig.plot(
        x=x_grid,
        y=topo_grid,
        pen="1.2p,black",
    )

    fig.plot(
        x=x_sta,
        y=topo_sta,
        style="c0.12c",
        fill="white",
        pen="0.4p,black",
    )

    sta_label = sta_valid.iloc[::LABEL_EVERY_N_STATION].copy()
    fig.text(
        x=sta_label["distance"],
        y=sta_label["elevation_m"],
        text=sta_label["name"].astype(str),
        font="6p,Times-Bold,black",
        justify="CB",
        offset="0c/0.12c",
        no_clip=True,
    )

    fig.colorbar(
        position="JMR+jML+w2.4c/0.25c+o0.5c/0c",
        frame='xaf+l"Topo (m)"',
    )

    if extract_sta is not None:
        mark_row = find_station_row_by_name(sta_valid, extract_sta)
        if mark_row is not None:
            x_mark = float(mark_row["distance"])
            sta_mark = str(mark_row["name"])

            fig.plot(
                x=[x_mark, x_mark],
                y=[topo_ymin, topo_ymax],
                pen="1.2p,red,--",
            )

            fig.text(
                x=x_mark,
                y=topo_ymax,
                text=sta_mark,
                font="9p,Times-Bold,red",
                justify="TL",
                offset="0.05c/-0.05c",
                no_clip=True,
            )


def plot_full_profiles_pygmt(section_dict, extract_sta=None):
    sta_valid = section_dict["sta_valid"]
    deps = section_dict["deps"]
    x_sta = section_dict["x_sta"]
    topo_sta = section_dict["topo_sta"]
    x_grid = section_dict["x_grid"]
    topo_grid = section_dict["topo_grid"]
    vs_grid = section_dict["vs_grid"]

    dep_min = float(np.nanmin(deps))
    dep_max = float(np.nanmax(deps))
    x_min = float(np.nanmin(x_grid))
    x_max = float(np.nanmax(x_grid))

    vs_grid_smooth = gaussian_filter(vs_grid, sigma=(0.5, 1.0))
    vs_da = make_xarray_grid(
        z=vs_grid_smooth,
        x=x_grid,
        y=deps,
        name="vs",
    )

    fig = pygmt.Figure()

    pygmt.config(
        FONT_LABEL="13p,Times-Bold,black",
        FONT_ANNOT_PRIMARY="11p,Times-Roman,black",
        FONT_TITLE="15p,Times-Bold,black",
        MAP_FRAME_TYPE="plain",
    )

    plot_topography_panel(
        fig=fig,
        title="Topography and Vs profile",
        sta_valid=sta_valid,
        x_sta=x_sta,
        topo_sta=topo_sta,
        x_grid=x_grid,
        topo_grid=topo_grid,
        extract_sta=extract_sta,
    )

    fig.shift_origin(yshift="-7.3c")

    vs_min = float(np.nanmin(vs_grid))
    vs_max = float(np.nanmax(vs_grid))
    use_current_cpt, cpt_file = make_or_get_vs_cpt(vs_min, vs_max)

    fig.basemap(
        region=[x_min, x_max, dep_min, dep_max],
        projection="X18c/-7.0c",
        frame=[
            "WSne",
            'xaf+l"Distance along profile (km)"',
            'yaf+l"Depth (km)"',
        ],
    )

    grdimage_with_vs_cpt(
        fig=fig,
        grid=vs_da,
        use_current_cpt=use_current_cpt,
        cpt_file=cpt_file,
    )

    fig.plot(
        x=x_sta,
        y=np.full_like(x_sta, dep_min),
        style="t0.18c",
        fill="white",
        pen="0.4p,black",
    )

    if extract_sta is not None:
        mark_row = find_station_row_by_name(sta_valid, extract_sta)
        if mark_row is not None:
            x_mark = float(mark_row["distance"])
            fig.plot(
                x=[x_mark, x_mark],
                y=[dep_min, dep_max],
                pen="1.2p,red,--",
            )

    colorbar_with_vs_cpt(
        fig=fig,
        use_current_cpt=use_current_cpt,
        cpt_file=cpt_file,
        position="JMR+jML+w6.4c/0.35c+o0.65c/0c",
        frame='xaf+l"Vs (km/s)"',
    )

    fig.savefig(OUT_PROFILE_FIG, dpi=FIG_DPI)
    print(f"[OK] Saved profile figure: {OUT_PROFILE_FIG}")


def plot_full_profiles_relative_pygmt(section_dict, extract_sta=None):
    sta_valid = section_dict["sta_valid"]
    deps = section_dict["deps"]
    x_sta = section_dict["x_sta"]
    topo_sta = section_dict["topo_sta"]
    x_grid = section_dict["x_grid"]
    topo_grid = section_dict["topo_grid"]
    vs_grid = section_dict["vs_grid"]

    rel_grid, _ = compute_vs_perturbation_percent(vs_grid)

    dep_min = float(np.nanmin(deps))
    dep_max = float(np.nanmax(deps))
    x_min = float(np.nanmin(x_grid))
    x_max = float(np.nanmax(x_grid))

    rel_grid_smooth = gaussian_filter(rel_grid, sigma=(0.5, 1.0))
    rel_da = make_xarray_grid(
        z=rel_grid_smooth,
        x=x_grid,
        y=deps,
        name="dvs_percent",
    )

    fig = pygmt.Figure()

    pygmt.config(
        FONT_LABEL="13p,Times-Bold,black",
        FONT_ANNOT_PRIMARY="11p,Times-Roman,black",
        FONT_TITLE="15p,Times-Bold,black",
        MAP_FRAME_TYPE="plain",
    )

    plot_topography_panel(
        fig=fig,
        title="Topography and relative Vs perturbation profile",
        sta_valid=sta_valid,
        x_sta=x_sta,
        topo_sta=topo_sta,
        x_grid=x_grid,
        topo_grid=topo_grid,
        extract_sta=extract_sta,
    )

    fig.shift_origin(yshift="-7.3c")

    rel_min = float(np.nanmin(rel_grid))
    rel_max = float(np.nanmax(rel_grid))
    make_relative_cpt(rel_min, rel_max)

    fig.basemap(
        region=[x_min, x_max, dep_min, dep_max],
        projection="X18c/-7.0c",
        frame=[
            "WSne",
            'xaf+l"Distance along profile (km)"',
            'yaf+l"Depth (km)"',
        ],
    )

    fig.grdimage(
        grid=rel_da,
        cmap=True,
        nan_transparent=True,
    )

    fig.plot(
        x=x_sta,
        y=np.full_like(x_sta, dep_min),
        style="t0.18c",
        fill="white",
        pen="0.4p,black",
    )

    if extract_sta is not None:
        mark_row = find_station_row_by_name(sta_valid, extract_sta)
        if mark_row is not None:
            x_mark = float(mark_row["distance"])
            fig.plot(
                x=[x_mark, x_mark],
                y=[dep_min, dep_max],
                pen="1.2p,red,--",
            )

    fig.colorbar(
        position="JMR+jML+w6.4c/0.35c+o0.65c/0c",
        frame='xaf+l"dVs/Vs (%)"',
    )

    fig.savefig(OUT_PROFILE_REL_FIG, dpi=FIG_DPI)
    print(f"[OK] Saved relative profile figure: {OUT_PROFILE_REL_FIG}")


# ==================================================
# Horizontal velocity grid for map
# ==================================================
def get_horizontal_vs_grid_for_map(model, region, target_depth):
    """
    Build a horizontal Vs grid at the closest available depth.

    Region:
        [lon_min, lon_max, lat_min, lat_max]

    Returns:
        vs_da, plane_crop, actual_depth
    """
    deps = np.sort(model["dep"].unique())
    actual_depth = float(deps[np.argmin(np.abs(deps - target_depth))])

    lon_min, lon_max, lat_min, lat_max = region

    plane = model[np.isclose(model["dep"], actual_depth)].copy()
    plane_crop = plane[
        (plane["lon"] >= lon_min)
        & (plane["lon"] <= lon_max)
        & (plane["lat"] >= lat_min)
        & (plane["lat"] <= lat_max)
    ].copy()

    if len(plane_crop) == 0:
        return None, plane_crop, actual_depth

    pivot = plane_crop.pivot_table(
        index="lat",
        columns="lon",
        values="vs",
        aggfunc="mean",
    )

    pivot = pivot.sort_index(axis=0).sort_index(axis=1)

    lats = pivot.index.values.astype(float)
    lons = pivot.columns.values.astype(float)
    z = pivot.values.astype(float)

    vs_da = make_xarray_grid(
        z=z,
        x=lons,
        y=lats,
        name="vs_map",
    )

    return vs_da, plane_crop, actual_depth


def plot_model_nodes_and_stations_pygmt(model, horizontal_nodes, sta_df):
    """
    Plot horizontal Vs background, model nodes, and station locations.

    Region is cropped by station distribution + padding,
    not by the full model-node area.
    """
    lon_min = sta_df["long"].min() - MAP_PAD_LON
    lon_max = sta_df["long"].max() + MAP_PAD_LON
    lat_min = sta_df["lat"].min() - MAP_PAD_LAT
    lat_max = sta_df["lat"].max() + MAP_PAD_LAT

    region = [
        float(lon_min),
        float(lon_max),
        float(lat_min),
        float(lat_max),
    ]

    nodes_crop = horizontal_nodes[
        (horizontal_nodes["lon"] >= lon_min)
        & (horizontal_nodes["lon"] <= lon_max)
        & (horizontal_nodes["lat"] >= lat_min)
        & (horizontal_nodes["lat"] <= lat_max)
    ].copy()

    vs_da, plane_crop, actual_depth = get_horizontal_vs_grid_for_map(
        model=model,
        region=region,
        target_depth=MAP_VEL_DEPTH,
    )

    print(f"[INFO] Horizontal map region: {region}")
    print(f"[INFO] Model nodes in cropped region: {len(nodes_crop)}")
    print(f"[INFO] Horizontal map target depth: {MAP_VEL_DEPTH} km")
    print(f"[INFO] Horizontal map actual depth: {actual_depth} km")
    print(f"[INFO] Velocity nodes in cropped region: {len(plane_crop)}")

    fig = pygmt.Figure()

    pygmt.config(
        FONT_LABEL="13p,Times-Bold,black",
        FONT_ANNOT_PRIMARY="11p,Times-Roman,black",
        FONT_TITLE="15p,Times-Bold,black",
        MAP_FRAME_TYPE="plain",
        FORMAT_GEO_MAP="ddd.xx",
    )

    title = f"Model nodes and stations, Vs at {actual_depth:.1f} km"

    fig.basemap(
        region=region,
        projection="M15c",
        frame=[
            f"WSne+t{title}",
            'xaf+l"Longitude"',
            'yaf+l"Latitude"',
        ],
    )

    if vs_da is not None and len(plane_crop) > 0:
        vs_min = float(np.nanmin(plane_crop["vs"].values))
        vs_max = float(np.nanmax(plane_crop["vs"].values))
        use_current_cpt, cpt_file = make_or_get_vs_cpt(vs_min, vs_max)

        grdimage_with_vs_cpt(
            fig=fig,
            grid=vs_da,
            use_current_cpt=use_current_cpt,
            cpt_file=cpt_file,
        )
    else:
        print("[WARN] No velocity grid available for horizontal map background.")
        use_current_cpt, cpt_file = True, None

    fig.coast(
        region=region,
        projection="M15c",
        shorelines="0.5p,black",
        borders=["1/0.5p,black", "2/0.25p,gray"],
        water="lightblue@30",
        resolution="i",
    )

    if len(nodes_crop) > 0:
        fig.plot(
            x=nodes_crop["lon"],
            y=nodes_crop["lat"],
            style="c0.07c",
            fill="gray20",
            pen="0.1p,white",
            label="Model nodes",
        )

    fig.plot(
        x=sta_df["long"],
        y=sta_df["lat"],
        style="a0.26c",
        fill="red",
        pen="0.45p,black",
        label="Stations",
    )

    sta_label = sta_df.iloc[::LABEL_EVERY_N_STATION].copy()
    fig.text(
        x=sta_label["long"],
        y=sta_label["lat"],
        text=sta_label["name"].astype(str),
        font="6p,Times-Bold,black",
        justify="CB",
        offset="0c/0.13c",
        no_clip=True,
    )

    fig.legend(
        position="JTR+jTR+o0.15c/0.15c",
        box="+gwhite+p0.5p,black",
    )

    if vs_da is not None and len(plane_crop) > 0:
        colorbar_with_vs_cpt(
            fig=fig,
            use_current_cpt=use_current_cpt,
            cpt_file=cpt_file,
            position="JMR+jML+w7c/0.35c+o0.65c/0c",
            frame='xaf+l"Vs (km/s)"',
        )

    fig.savefig(OUT_NODE_STA_FIG, dpi=FIG_DPI)
    print(f"[OK] Saved node/station velocity map: {OUT_NODE_STA_FIG}")


# ==================================================
# Main
# ==================================================
def main():
    if not os.path.exists(STATION_FILE):
        raise FileNotFoundError(f"Station file not found: {STATION_FILE}")

    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")

    print(f"[INFO] Read station file: {STATION_FILE}")
    sta_df = read_station_file(STATION_FILE)

    print(f"[INFO] Read model file: {MODEL_FILE}")
    model, dim, header = read_model_file(MODEL_FILE)

    lons = np.sort(model["lon"].unique())
    lats = np.sort(model["lat"].unique())
    deps_all = np.sort(model["dep"].unique())

    print(f"[INFO] Model lon count: {len(lons)}")
    print(f"[INFO] Model lat count: {len(lats)}")
    print(f"[INFO] Model dep count: {len(deps_all)}")
    print(f"[INFO] Model depth range: {deps_all.min()} to {deps_all.max()} km")

    dep_min = DEPTH_MIN
    dep_max = deps_all.max() if DEPTH_MAX is None else DEPTH_MAX
    deps = deps_all[(deps_all >= dep_min) & (deps_all <= dep_max)]

    if len(deps) == 0:
        raise ValueError(
            f"No depth values found in requested range: {dep_min} to {dep_max} km"
        )

    print(f"[INFO] Extract depth range: {deps.min()} to {deps.max()} km")
    print(f"[INFO] Output vertical resampling dz: {RESAMPLE_DZ_KM} km")
    print(f"[INFO] Output model extension: {OUTPUT_MODEL_EXT}")
    print(f"[INFO] Vp/Vs ratio = {VpVs:.2f}")

    horizontal_nodes = model[["lon", "lat"]].drop_duplicates().reset_index(drop=True)

    # --------------------------------------------------
    # Horizontal plane plot with Vs background
    # --------------------------------------------------
    plot_model_nodes_and_stations_pygmt(
        model=model,
        horizontal_nodes=horizontal_nodes,
        sta_df=sta_df,
    )

    # --------------------------------------------------
    # Multi-core extraction for all stations
    # --------------------------------------------------
    ncore = get_ncore()
    print(f"[INFO] Use CPU cores for extraction: {ncore}")

    station_dicts = sta_df.to_dict(orient="records")
    section_profiles = {}
    n_ok_all = 0
    n_skip_all = 0

    with ProcessPoolExecutor(
        max_workers=ncore,
        initializer=init_worker,
        initargs=(model, lons, lats, deps, horizontal_nodes),
    ) as executor:
        futures = {
            executor.submit(extract_station_worker, sta_dict): sta_dict["name"]
            for sta_dict in station_dicts
        }

        for future in as_completed(futures):
            sta_name = str(futures[future])

            try:
                sta_name_out, df_1d, msg = future.result()

                if msg is not None:
                    print(msg)

                if df_1d is None or len(df_1d) == 0:
                    section_profiles[sta_name_out] = None
                    n_skip_all += 1
                else:
                    section_profiles[sta_name_out] = df_1d
                    n_ok_all += 1

            except Exception as e:
                print(f"[ERROR] Station {sta_name}: {e}")
                section_profiles[sta_name] = None
                n_skip_all += 1

    print(f"[INFO] Valid extracted station profiles for section: {n_ok_all}")
    print(f"[INFO] Skipped station profiles: {n_skip_all}")

    # --------------------------------------------------
    # Save and plot requested station outputs
    # --------------------------------------------------
    sta_out = select_output_stations(sta_df, extract_sta)

    output_plot_tasks = []
    n_saved = 0

    for _, sta_row in sta_out.iterrows():
        sta_name = str(sta_row["name"])
        df_1d = section_profiles.get(sta_name, None)

        if df_1d is None or len(df_1d) == 0:
            print(f"[SKIP] No output for station {sta_name}")
            continue

        save_1d_data(sta_name, df_1d)
        output_plot_tasks.append((sta_row.to_dict(), df_1d))
        n_saved += 1

    print(f"[INFO] Saved output 1D station count: {n_saved}")

    # --------------------------------------------------
    # Serial 1D PyGMT plotting
    # --------------------------------------------------
    if len(output_plot_tasks) > 0:
        print("[INFO] Plot 1D figures by PyGMT serially...")

        for task in output_plot_tasks:
            sta_name = str(task[0]["name"])
            try:
                plot_1d_worker(task)
            except Exception as e:
                print(f"[ERROR] Plot station {sta_name}: {e}")

    # --------------------------------------------------
    # Full profile plots by PyGMT
    # --------------------------------------------------
    if n_ok_all >= 2:
        section_dict = build_section_grid(section_profiles, sta_df)
        plot_full_profiles_pygmt(section_dict, extract_sta=extract_sta)
        plot_full_profiles_relative_pygmt(section_dict, extract_sta=extract_sta)
    else:
        print("[WARN] Not enough valid stations to build 2D profile figures.")

    print("==================================================")
    print("[DONE]")
    print(f"1D model dir      : {OUT_DATA_DIR}")
    print(f"1D figure dir     : {OUT_FIG_1D_DIR}")
    print(f"Profile figure    : {OUT_PROFILE_FIG}")
    print(f"Relative profile  : {OUT_PROFILE_REL_FIG}")
    print(f"Node-station map  : {OUT_NODE_STA_FIG}")
    print("Output .mod format:")
    print("    # dep long lat vs vp")
    print("==================================================")


if __name__ == "__main__":
    main()

