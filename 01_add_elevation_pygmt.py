#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Read station file: name long lat distance
Download PyGMT 01s topography, extract elevation at stations,
trace topography along the station trend, and plot station map + elevation profile.

Input columns:
    name long lat distance

Outputs:
    output_topo/station_with_topo.lst
    output_topo/profile_topo_trace.csv
    output_topo/station_topo_map_profile.png
    output_topo/station_topo_map_profile.pdf
"""

import os
import numpy as np
import pandas as pd
import pygmt
from pathlib import Path


# ==================================================
# Parameters
# ==================================================
STATION_FILE = "input/station_use.lst"

OUT_DIR = Path("output")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG_DIR = Path("figures")
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_STATION_FILE = OUT_DIR / "station_with_topo.lst"
OUT_TRACE_FILE = OUT_DIR / "profile_topo_trace.csv"

OUT_FIG_PNG = OUT_FIG_DIR / "station_topo_map_profile.png"
# OUT_FIG_PDF = OUT_FIG_DIR / "station_topo_map_profile.pdf"

# PyGMT topography resolution
TOPO_RESOLUTION = "01s"

# Map padding in degree
PAD_LON = 0.01
PAD_LAT = 0.01

# Profile trace spacing in km
TRACE_SPACING_KM = 0.02

# Label every N stations on map/profile
LABEL_EVERY_N_STATION = 10

# Figure setting
MAP_PROJECTION = "M15c"
PROFILE_WIDTH = "15c"
PROFILE_HEIGHT = "5c"

# CPT
TOPO_CMAP = "geo"
STATION_CMAP = "turbo"


# ==================================================
# Functions
# ==================================================
def read_station_file(station_file):
    """
    Read station file with columns:
    name long lat distance

    name is kept as text/string.
    """
    df = pd.read_csv(
        station_file,
        sep=r"\s+",
        header=None,
        comment="#",
        names=["name", "long", "lat", "distance"],
        dtype={"name": str},
    )

    df["long"] = pd.to_numeric(df["long"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["distance"] = pd.to_numeric(df["distance"], errors="coerce")

    df = df.dropna(subset=["long", "lat", "distance"]).reset_index(drop=True)

    # Sort along network trend by distance
    df = df.sort_values("distance").reset_index(drop=True)

    # Add station index
    df["idx"] = np.arange(len(df))

    return df


def get_region(df):
    """
    Get map region with padding.
    """
    lon_min = df["long"].min() - PAD_LON
    lon_max = df["long"].max() + PAD_LON
    lat_min = df["lat"].min() - PAD_LAT
    lat_max = df["lat"].max() + PAD_LAT

    return [
        float(lon_min),
        float(lon_max),
        float(lat_min),
        float(lat_max),
    ]


def load_topography(region):
    """
    Load PyGMT Earth relief.
    For first use, PyGMT will download the 01s tile automatically.
    """
    grid = pygmt.datasets.load_earth_relief(
        resolution=TOPO_RESOLUTION,
        region=region,
        registration="gridline",
    )
    return grid


def sample_grid_at_points(grid, df, lon_col="long", lat_col="lat"):
    """
    Sample grid elevation at station points using pygmt.grdtrack.
    """
    points = df[[lon_col, lat_col]].copy()
    points.columns = ["x", "y"]

    sampled = pygmt.grdtrack(
        points=points,
        grid=grid,
        newcolname="elevation_m",
    )

    df_out = df.copy()
    df_out["elevation_m"] = sampled["elevation_m"].values

    return df_out


def make_profile_trace(df):
    """
    Trace lon-lat along station trend.

    The path follows stations sorted by distance.
    Between each station pair, interpolate lon, lat, and distance.
    """
    rows = []

    for i in range(len(df) - 1):
        lon1 = df.loc[i, "long"]
        lat1 = df.loc[i, "lat"]
        dis1 = df.loc[i, "distance"]

        lon2 = df.loc[i + 1, "long"]
        lat2 = df.loc[i + 1, "lat"]
        dis2 = df.loc[i + 1, "distance"]

        seg_len = abs(dis2 - dis1)

        if seg_len <= 0:
            npts = 2
        else:
            npts = max(2, int(np.ceil(seg_len / TRACE_SPACING_KM)) + 1)

        lons = np.linspace(lon1, lon2, npts)
        lats = np.linspace(lat1, lat2, npts)
        diss = np.linspace(dis1, dis2, npts)

        # Avoid duplicate points at segment boundaries
        if i > 0:
            lons = lons[1:]
            lats = lats[1:]
            diss = diss[1:]

        for lo, la, di in zip(lons, lats, diss):
            rows.append([lo, la, di])

    trace = pd.DataFrame(rows, columns=["long", "lat", "distance"])
    return trace


def sample_profile_topography(grid, trace):
    """
    Sample topography along profile trace.
    """
    points = trace[["long", "lat"]].copy()
    points.columns = ["x", "y"]

    sampled = pygmt.grdtrack(
        points=points,
        grid=grid,
        newcolname="elevation_m",
    )

    trace_out = trace.copy()
    trace_out["elevation_m"] = sampled["elevation_m"].values

    return trace_out


def save_outputs(df_station, trace):
    """
    Save station and topo trace outputs.
    """
    df_station_out = df_station[
        ["name", "long", "lat", "distance", "elevation_m", "idx"]
    ].copy()

    df_station_out.to_csv(
        OUT_STATION_FILE,
        sep=" ",
        index=False,
        float_format="%.6f",
    )

    trace.to_csv(
        OUT_TRACE_FILE,
        index=False,
        float_format="%.6f",
    )

    print(f"[OK] Saved station elevation file: {OUT_STATION_FILE}")
    print(f"[OK] Saved topo profile trace:    {OUT_TRACE_FILE}")


def plot_map_profile(df, trace, grid, region):
    """
    Plot topographic map and elevation profile.
    """
    fig = pygmt.Figure()

    pygmt.config(
        FONT_LABEL="13p,Times-Bold,black",
        FONT_ANNOT_PRIMARY="11p,Times-Roman,black",
        FONT_TITLE="15p,Times-Bold,black",
        MAP_FRAME_TYPE="plain",
        FORMAT_GEO_MAP="ddd.xx",
    )

    elev_min = float(np.nanmin(df["elevation_m"]))
    elev_max = float(np.nanmax(df["elevation_m"]))

    trace_min = float(np.nanmin(trace["elevation_m"]))
    trace_max = float(np.nanmax(trace["elevation_m"]))

    all_elev_min = min(elev_min, trace_min)
    all_elev_max = max(elev_max, trace_max)

    cpt_min = np.floor(all_elev_min / 50.0) * 50.0
    cpt_max = np.ceil(all_elev_max / 50.0) * 50.0

    if cpt_min == cpt_max:
        cpt_min -= 10
        cpt_max += 10

    pygmt.makecpt(
        cmap=TOPO_CMAP,
        series=[cpt_min, cpt_max, 10],
        continuous=True,
    )

    # ==================================================
    # Panel 1: Map
    # ==================================================
    fig.basemap(
        region=region,
        projection=MAP_PROJECTION,
        frame=[
            "WSne+tCHT array with topography",
            'xaf+l"Longitude"',
            'yaf+l"Latitude"',
        ],
    )

    fig.grdimage(
        grid=grid,
        cmap=True,
        shading=True,
    )

    fig.coast(
        region=region,
        projection=MAP_PROJECTION,
        shorelines="0.5p,black",
        borders=["1/0.5p,black", "2/0.25p,gray"],
        water="lightblue",
        resolution="i",
    )

    # Trace line along network
    fig.plot(
        x=trace["long"],
        y=trace["lat"],
        pen="1.5p,black",
    )

    # Station points colored by elevation
    pygmt.makecpt(
        cmap=STATION_CMAP,
        series=[cpt_min, cpt_max, 10],
        continuous=True,
    )

    fig.plot(
        x=df["long"],
        y=df["lat"],
        style="c0.18c",
        fill=df["elevation_m"],
        cmap=True,
        pen="0.3p,black",
    )

    # Label station index every N stations
    df_label = df.iloc[::LABEL_EVERY_N_STATION].copy()

    fig.text(
        x=df_label["long"],
        y=df_label["lat"],
        text=df_label["name"].astype(int).astype(str),
        font="13p,Times-Bold,black",
        justify="CM",
        offset="0c/0.18c",
        fill="lightgray",
        no_clip=True,
    )

    fig.colorbar(
        position="JMR+jML+w7c/0.35c+o0.7c/0c",
        frame='xaf+l"Elevation (m)"',
    )

    # ==================================================
    # Panel 2: Topographic profile
    # ==================================================
    fig.shift_origin(yshift="-6.8c")

    x_min = float(df["distance"].min())
    x_max = float(df["distance"].max())

    y_min = float(np.nanmin(trace["elevation_m"]))
    y_max = float(np.nanmax(trace["elevation_m"]))

    y_pad = max(10.0, 0.15 * (y_max - y_min))
    y0 = np.floor((y_min - y_pad) / 10.0) * 10.0
    y1 = np.ceil((y_max + y_pad) / 10.0) * 10.0

    profile_region = [x_min, x_max, y0, y1]

    fig.basemap(
        region=profile_region,
        projection=f"X-{PROFILE_WIDTH}/{PROFILE_HEIGHT}",
        frame=[
            "WSne+tTopographic profile along station trend",
            'xaf+l"Distance along profile (km)"',
            'yaf+l"Elevation (m)"',
        ],
    )

    # Fill elevation polygon
    poly_x = np.r_[trace["distance"].values, trace["distance"].values[::-1]]
    poly_y = np.r_[
        trace["elevation_m"].values,
        np.full(len(trace), y0)[::-1],
    ]

    fig.plot(
        x=poly_x,
        y=poly_y,
        fill="gray80",
        pen="0.5p,gray40",
    )

    # Topo line
    fig.plot(
        x=trace["distance"],
        y=trace["elevation_m"],
        pen="1.2p,black",
    )

    # Station points on profile
    pygmt.makecpt(
        cmap=STATION_CMAP,
        series=[cpt_min, cpt_max, 10],
        continuous=True,
    )

    fig.plot(
        x=df["distance"],
        y=df["elevation_m"],
        style="c0.16c",
        fill=df["elevation_m"],
        cmap=True,
        pen="0.25p,black",
    )

    # Station index labels on profile
    fig.text(
        x=df_label["distance"],
        y=df_label["elevation_m"],
        text=df_label["name"].astype(int).astype(str),
        font="7p,Times-Bold,black",
        justify="CM",
        offset="0c/0.18c",
        no_clip=True,
    )

    fig.colorbar(
        position="JMR+jML+w5c/0.35c+o0.7c/0c",
        frame='xaf+l"Station elevation (m)"',
    )

    fig.savefig(OUT_FIG_PNG, dpi=300)
    # fig.savefig(OUT_FIG_PDF)

    print(f"[OK] Saved figure: {OUT_FIG_PNG}")
    # print(f"[OK] Saved figure: {OUT_FIG_PDF}")


# ==================================================
# Main
# ==================================================
def main():
    if not os.path.exists(STATION_FILE):
        raise FileNotFoundError(f"Cannot find station file: {STATION_FILE}")

    print(f"[INFO] Reading station file: {STATION_FILE}")
    df = read_station_file(STATION_FILE)

    print(f"[INFO] Number of stations: {len(df)}")
    print(f"[INFO] First station: {df.iloc[0]['name']}")
    print(f"[INFO] Last station:  {df.iloc[-1]['name']}")

    region = get_region(df)
    print(f"[INFO] Topography region: {region}")

    print(f"[INFO] Loading PyGMT Earth relief: {TOPO_RESOLUTION}")
    grid = load_topography(region)

    print("[INFO] Sampling station elevations...")
    df = sample_grid_at_points(grid, df)

    print("[INFO] Making profile trace along station trend...")
    trace = make_profile_trace(df)

    print("[INFO] Sampling topography along profile trace...")
    trace = sample_profile_topography(grid, trace)

    save_outputs(df, trace)

    print("[INFO] Plotting map and profile...")
    plot_map_profile(df, trace, grid, region)

    print("[DONE]")


if __name__ == "__main__":
    main()