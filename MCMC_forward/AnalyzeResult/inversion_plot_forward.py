#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot starting forward responses and observed data for one station.

This script is a cleaned/reworked Python 3 version of inversion_plot_forward.py.
It keeps the old 5-panel layout:
    1) Full Vs model
    2) Shallow Vs model
    3) Receiver function
    4) Surface-wave dispersion: Vph and optional GV
    5) H/V ellipticity

Usage
-----
python inversion_plot_forward_reworked.py STA MCMC_DIR

Example
-------
python inversion_plot_forward_reworked.py 00000 /path/to/Project/MonterCarlo

Expected directory structure
----------------------------
Run this script from AnalyzeResult/ or the plotting script directory, so that:
    ../data/{STA}_data/in.data_{STA}
    ../data/{STA}_data/{STA}.ph
    ../data/{STA}_data/{STA}.gv
    ../data/{STA}_data/{STA}.HV
    ../data/{STA}_data/{STA}.RF
    ../data/{STA}_data/in.connector
    ../Vel_mod/{VELMODFILENAME}
exist relative to the current working directory.

The station MCMC directory is expected at:
    {MCMC_DIR}/{STA}/Initial.mod
    {MCMC_DIR}/{STA}/Initial_left.mod
    {MCMC_DIR}/{STA}/Initial_right.mod
    {MCMC_DIR}/{STA}/Initial.ph
    {MCMC_DIR}/{STA}/Initial.gv     optional
    {MCMC_DIR}/{STA}/Initial.hv
    {MCMC_DIR}/{STA}/Initial.rf

Output
------
    {MCMC_DIR}/{STA}/{STA}_forward_data.png
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


# =============================================================================
# USER PARAMETERS
# =============================================================================
VELMODFILENAME = "NTW_1D_Liu21_modified.dat"  # file in Project/Vel_mod

FIGSIZE_MAIN = (18, 20)
DPI = 300

VS_XLIM = (0.0, 7.0)
VS_YLIM_FULL = (0.0, 15.0)
VS_YLIM_SHALLOW = (0.0, 5.0)

RF_YLIM = (-0.25, 0.60)
HV_YLIM = (0.0, 2.0)
DISP_YLIM = (0.0, 3.0)

AX_FACE = "lightyellow"

FONT_LABEL = {
    "family": "serif",
    "color": "darkblue",
    "size": 20,
    "weight": "bold",
}
FONT_TITLE_BIG = {
    "family": "serif",
    "color": "darkgreen",
    "size": 30,
    "weight": "bold",
}
FONT_TITLE_PANEL = {
    "family": "serif",
    "color": "black",
    "size": 25,
    "weight": "bold",
}

# If True, cap non-mantle right-side model-space Vs at 4.9 km/s.
CAP_NON_MANTLE_RIGHT_VS = True
NON_MANTLE_VS_CAP = 4.9
MANTLE_ID = 4

# Connector line can be different between old/new scripts.
# The old forward script used line 7; the newer cleaned script uses line 8.
# This script tries line 8 first, then line 7.
PLOT_TYPE_LINES_TO_TRY = (8, 7)


# =============================================================================
# SMALL HELPERS
# =============================================================================
def die(msg):
    print("[ERROR] {}".format(msg))
    sys.exit(1)


def warn(msg):
    print("[WARN] {}".format(msg))


def require_file(path, label=None):
    if not os.path.isfile(path):
        if label is None:
            label = "required file"
        die("Missing {}: {}".format(label, path))


def optional_file(path):
    return os.path.isfile(path)


def read_table(path, names, usecols=None, required=True):
    if not os.path.isfile(path):
        if required:
            die("Missing file: {}".format(path))
        return None
    try:
        return pd.read_csv(
            path,
            sep=r"\s+",
            usecols=usecols,
            names=names,
            header=None,
            comment="#",
        )
    except Exception as exc:
        if required:
            die("Cannot read {}: {}".format(path, exc))
        warn("Cannot read optional file {}: {}".format(path, exc))
        return None


def read_inversion_flags(indatafile):
    """Read iph, igv, ihv, irf from in.data_STA."""
    require_file(indatafile, "in.data file")
    data = np.genfromtxt(indatafile, dtype=int, comments="#")
    data = np.atleast_1d(data)
    if len(data) < 4:
        die("Bad format in {}. Need at least 4 integer values: iph igv ihv irf".format(indatafile))
    return int(data[0]), int(data[1]), int(data[2]), int(data[3])


def parse_plot_type(connectorfile):
    """Try to read model plot type from in.connector.

    plot_type = 0: layer-cake style
    plot_type = 1: gradient style
    """
    require_file(connectorfile, "in.connector")
    with open(connectorfile, "r") as fobj:
        lines = fobj.readlines()

    for line_number in PLOT_TYPE_LINES_TO_TRY:
        idx = line_number - 1
        if idx < 0 or idx >= len(lines):
            continue
        parts = lines[idx].strip().split()
        if not parts:
            continue
        try:
            plot_type = int(parts[0])
        except ValueError:
            continue
        if plot_type in (0, 1):
            print("Plot type read from {} line {}: {}".format(connectorfile, line_number, plot_type))
            return plot_type

    die("Cannot read valid plot_type from {}. Tried lines {}.".format(connectorfile, PLOT_TYPE_LINES_TO_TRY))


def layer_bottom_to_step_model(df):
    """Convert layer-bottom style model to step-style profile.

    Expected columns:
        thick/depth-like column, vs, id

    The newer MCMC output commonly stores the first column as layer bottom depth.
    This function inserts duplicated boundary rows when the layer group id changes,
    then creates a true depth column named 'dep'.
    """
    df = df.copy()
    df.columns = ["thick", "vs", "id"]

    if df.empty:
        return df.assign(dep=[], thickness=[])

    # Insert a copy of boundary rows at group transitions.
    group_change = df["id"].ne(df["id"].shift())
    group_change.iloc[0] = False

    ins = df.loc[group_change, ["thick", "vs", "id"]].copy()
    ins["thick"] = df["thick"].shift().loc[group_change].values

    df2 = pd.concat([ins, df]).sort_index(kind="mergesort").reset_index(drop=True)
    df2 = pd.concat([df2, df2.tail(1)], ignore_index=True)

    df2["dep"] = df2["thick"].shift(fill_value=0.0)
    df2["thickness"] = df2["thick"] - df2["dep"]
    return df2


def add_zero_depth_if_needed(df):
    """For old depth-style model files, duplicate first row at depth 0 if needed."""
    if df is None or df.empty:
        return df
    df = df.copy()
    if "dep" not in df.columns:
        return df
    if float(df["dep"].iloc[0]) == 0.0:
        return df
    first_row = df.iloc[0].copy()
    first_row["dep"] = 0.0
    return pd.concat([pd.DataFrame([first_row]), df], ignore_index=True)


def read_initial_model(path, plot_type):
    """Read Initial.mod / Initial_left.mod / Initial_right.mod.

    For gradient style from the newer workflow, use the layer-bottom-to-step converter.
    For layer-cake/older style, fall back to direct depth plotting with a zero-depth row.
    """
    require_file(path, os.path.basename(path))
    raw = read_table(path, names=["col0", "vs", "id"], usecols=[0, 1, 7], required=True)

    if plot_type == 1:
        return layer_bottom_to_step_model(raw[["col0", "vs", "id"]])

    out = raw.rename(columns={"col0": "dep"})[["dep", "vs", "id"]]
    return add_zero_depth_if_needed(out)


def read_reference_model(path):
    """Read Project/Vel_mod reference model.

    Expected columns follow old script: depth, Vs, Vp at columns 0, 1, 7.
    """
    if not os.path.isfile(path):
        warn("Reference velocity model not found: {}".format(path))
        return None
    return read_table(path, names=["dep", "vs", "vp"], usecols=[0, 1, 7], required=False)


def estimate_moho_depth(model_df):
    """Estimate Moho as the depth just above the largest positive Vs jump."""
    if model_df is None or model_df.empty:
        return None
    if len(model_df) < 2:
        return None
    tmp = model_df.copy()
    tmp["dvs"] = tmp["vs"].diff()
    if not np.isfinite(tmp["dvs"]).any():
        return None
    idx = tmp["dvs"].idxmax() - 1
    if idx not in tmp.index:
        return None
    return float(tmp.loc[idx, "dep"])


def split_model_space_by_mantle(df):
    """Split model-space lines into non-mantle and mantle parts."""
    if df is None or df.empty or "id" not in df.columns:
        return df, pd.DataFrame(columns=df.columns if df is not None else [])
    non_mantle = df[df["id"] != MANTLE_ID].copy()
    mantle = df[df["id"] == MANTLE_ID].copy()
    return non_mantle, mantle


def plot_model_space(ax, left_model, right_model, label=True):
    """Plot left/right model-space lines with mantle segment dotted."""
    right_non, right_man = split_model_space_by_mantle(right_model)
    left_non, left_man = split_model_space_by_mantle(left_model)

    if right_non is not None and not right_non.empty:
        ax.plot(
            right_non["vs"], right_non["dep"],
            dashes=[8, 8, 16, 8], lw=2, color="g",
            label="Model Space" if label else None,
        )
    if left_non is not None and not left_non.empty:
        ax.plot(left_non["vs"], left_non["dep"], dashes=[8, 8, 16, 8], lw=2, color="g")

    # connect non-mantle to mantle where both exist
    if right_non is not None and right_man is not None and (not right_non.empty) and (not right_man.empty):
        ax.plot(
            [right_non["vs"].iloc[-1], right_man["vs"].iloc[0]],
            [right_non["dep"].iloc[-1], right_man["dep"].iloc[0]],
            dashes=[8, 8, 16, 8], lw=2, color="g",
        )
    if left_non is not None and left_man is not None and (not left_non.empty) and (not left_man.empty):
        ax.plot(
            [left_non["vs"].iloc[-1], left_man["vs"].iloc[0]],
            [left_non["dep"].iloc[-1], left_man["dep"].iloc[0]],
            dashes=[8, 8, 16, 8], lw=2, color="g",
        )

    if right_man is not None and not right_man.empty:
        ax.plot(right_man["vs"], right_man["dep"], ":", lw=2, color="g")
    if left_man is not None and not left_man.empty:
        ax.plot(left_man["vs"], left_man["dep"], ":", lw=2, color="g")


def set_depth_axis(ax, ylim, yticks=None, labeltop=False):
    ax.set_xlabel("Vs (km/s)", fontdict=FONT_LABEL)
    ax.set_ylabel("Depth (km)", fontdict=FONT_LABEL)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.tick_params(labeltop=labeltop, labelsize=20)
    ax.set_xlim(VS_XLIM)
    ax.set_ylim(ylim)
    ax.set_ylim(ax.get_ylim()[::-1])
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.grid(color="k", axis="both", linestyle="-", linewidth=2, alpha=0.1, zorder=2)


def finite_minmax(values, pad_fraction=0.05, default=None):
    vals = []
    for v in values:
        if v is None:
            continue
        arr = np.asarray(v, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            vals.append(arr)
    if not vals:
        return default
    allv = np.concatenate(vals)
    vmin = float(np.min(allv))
    vmax = float(np.max(allv))
    if vmin == vmax:
        pad = 0.5 if vmax == 0 else abs(vmax) * 0.1
    else:
        pad = (vmax - vmin) * pad_fraction
    return vmin - pad, vmax + pad


def build_paths(sta, mdir, pwd):
    project_root = os.path.dirname(pwd)
    data_root = os.path.join(project_root, "data")
    sta_data_dir = os.path.join(data_root, "{}_data".format(sta))
    sta_dir = os.path.join(mdir, sta)

    return {
        "project_root": project_root,
        "data_root": data_root,
        "sta_data_dir": sta_data_dir,
        "sta_dir": sta_dir,
        "indatafile": os.path.join(sta_data_dir, "in.data_{}".format(sta)),
        "connectorfile": os.path.join(sta_data_dir, "in.connector"),
        # Fake data to get the period output
        "phdatafile": os.path.join(sta_data_dir, "{}.ph".format(sta)),
        "gvdatafile": os.path.join(sta_data_dir, "{}.gv".format(sta)),
        "hvdatafile": os.path.join(sta_data_dir, "{}.HV".format(sta)),
        "rfdatafile": os.path.join(sta_data_dir, "{}.RF".format(sta)),
        # Real data to compare
        "phdatafiler": os.path.join(os.path.dirname(os.path.dirname(sta_data_dir)),"real_data","{}_data".format(sta),"{}.ph".format(sta)),
        "gvdatafiler": os.path.join(os.path.dirname(os.path.dirname(sta_data_dir)),"real_data","{}_data".format(sta),"{}.gv".format(sta)),
        "hvdatafiler": os.path.join(os.path.dirname(os.path.dirname(sta_data_dir)),"real_data","{}_data".format(sta),"{}.HV".format(sta)),
        "rfdatafiler": os.path.join(os.path.dirname(os.path.dirname(sta_data_dir)),"real_data","{}_data".format(sta),"{}.RF".format(sta)),
        #
        "modfile": os.path.join(sta_data_dir, "mod.{}".format(sta)),
        "velmodfile": os.path.join(project_root, "Vel_mod", VELMODFILENAME),
        "inputmodelfile": os.path.join(sta_dir, "Initial.mod"),
        "inputmodelfileleft": os.path.join(sta_dir, "Initial_left.mod"),
        "inputmodelfileright": os.path.join(sta_dir, "Initial_right.mod"),
        "startfileph": os.path.join(sta_dir, "Initial.ph"),
        "startfilegv": os.path.join(sta_dir, "Initial.gv"),
        "startfilehv": os.path.join(sta_dir, "Initial.hv"),
        "startfilerf": os.path.join(sta_dir, "Initial.rf"),
        "out_png": os.path.join(sta_dir, "{}_forward_data.png".format(sta)),
    }


# =============================================================================
# MAIN PLOTTING
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Plot starting forward model responses and observed data for one station."
    )
    parser.add_argument("station", help="Station name, e.g., 00000")
    parser.add_argument("mcmc_dir", help="Path to MonteCarlo directory containing station subdirectory")
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output PNG path. Default: MCMC_DIR/STA/STA_forward_data.png",
    )
    parser.add_argument(
        "--velmod",
        default=None,
        help="Optional reference velocity model path. Overrides VELMODFILENAME.",
    )
    parser.add_argument(
        "--full-depth",
        type=float,
        default=VS_YLIM_FULL[1],
        help="Maximum depth for full Vs panel. Default: {} km".format(VS_YLIM_FULL[1]),
    )
    parser.add_argument(
        "--shallow-depth",
        type=float,
        default=VS_YLIM_SHALLOW[1],
        help="Maximum depth for shallow Vs panel. Default: {} km".format(VS_YLIM_SHALLOW[1]),
    )
    args = parser.parse_args()

    sta = str(args.station)
    mdir = os.path.abspath(args.mcmc_dir)
    pwd = os.getcwd()

    print("Current directory: {}".format(pwd))
    print("Station: {}".format(sta))
    print("MCMC dir: {}".format(mdir))

    paths = build_paths(sta, mdir, pwd)
    if args.out is not None:
        paths["out_png"] = os.path.abspath(args.out)
    if args.velmod is not None:
        paths["velmodfile"] = os.path.abspath(args.velmod)

    require_file(paths["indatafile"], "in.data file")
    iph, igv, ihv, irf = read_inversion_flags(paths["indatafile"])
    print("Data flags: iph={} igv={} ihv={} irf={}".format(iph, igv, ihv, irf))

    plot_type = parse_plot_type(paths["connectorfile"])
    if plot_type == 1:
        print("Plot the model in gradient style")
    else:
        print("Plot the model in layer-cake style")

    # -------------------------------------------------------------------------
    # Read models
    # -------------------------------------------------------------------------
    inputmodel0 = read_reference_model(paths["velmodfile"])
    inputmodel = read_initial_model(paths["inputmodelfile"], plot_type)
    inputmodelleft = read_initial_model(paths["inputmodelfileleft"], plot_type)
    inputmodelright = read_initial_model(paths["inputmodelfileright"], plot_type)

    if CAP_NON_MANTLE_RIGHT_VS and inputmodelright is not None and "id" in inputmodelright.columns:
        mask = inputmodelright["id"] != MANTLE_ID
        inputmodelright.loc[mask & (inputmodelright["vs"] > NON_MANTLE_VS_CAP), "vs"] = NON_MANTLE_VS_CAP

    moho_depth = estimate_moho_depth(inputmodel)
    if moho_depth is not None:
        print("Estimated Moho depth: {:.3f} km".format(moho_depth))
    else:
        warn("Cannot estimate Moho depth from Initial.mod")

    # -------------------------------------------------------------------------
    # Read starting forward predictions and observed data
    # -------------------------------------------------------------------------
    inputph = read_table(paths["startfileph"], names=["per", "vel"], usecols=[0, 1], required=(iph == 1))
    inputgv = read_table(paths["startfilegv"], names=["per", "vel"], usecols=[0, 1], required=False)
    inputhv = read_table(paths["startfilehv"], names=["per", "hv"], usecols=[0, 1], required=(ihv == 1))
    inputrf = read_table(paths["startfilerf"], names=["time", "amp"], usecols=[0, 1], required=(irf == 1))

    dataph = read_table(paths["phdatafile"], names=["per", "vel", "unc"], usecols=[0, 1, 2], required=(iph == 1))
    datagv = read_table(paths["gvdatafile"], names=["per", "vel", "unc"], usecols=[0, 1, 2], required=False)
    datahv = read_table(paths["hvdatafile"], names=["per", "hv", "unc"], usecols=[0, 1, 2], required=(ihv == 1))
    datarf = read_table(paths["rfdatafile"], names=["time", "amp", "unc"], usecols=[0, 1, 2], required=(irf == 1))

    dataphr = read_table(paths["phdatafiler"], names=["per", "vel", "unc"], usecols=[0, 1, 2], required=(iph == 1))
    datagvr = read_table(paths["gvdatafiler"], names=["per", "vel", "unc"], usecols=[0, 1, 2], required=False)
    datahvr = read_table(paths["hvdatafiler"], names=["per", "hv", "unc"], usecols=[0, 1, 2], required=(ihv == 1))
    datarfr = read_table(paths["rfdatafiler"], names=["time", "amp", "unc"], usecols=[0, 1, 2], required=(irf == 1))

    if igv == 1 and inputgv is None:
        warn("igv=1 but Initial.gv was not found: {}".format(paths["startfilegv"]))
    if igv == 1 and datagv is None:
        warn("igv=1 but observed GV file was not found: {}".format(paths["gvdatafile"]))

    # -------------------------------------------------------------------------
    # Create figure
    # -------------------------------------------------------------------------
    print("Making forward-data figure...")
    fig = plt.figure(figsize=FIGSIZE_MAIN)

    ax_vs_full = plt.subplot2grid((3, 5), (1, 0), rowspan=2, colspan=2)
    ax_rf = plt.subplot2grid((3, 5), (0, 2), colspan=3)
    ax_disp = plt.subplot2grid((3, 5), (1, 2), colspan=3)
    ax_vs_shallow = plt.subplot2grid((3, 5), (0, 0), colspan=2)
    ax_hv = plt.subplot2grid((3, 5), (2, 2), colspan=3)

    for ax in (ax_vs_full, ax_rf, ax_disp, ax_vs_shallow, ax_hv):
        ax.set_facecolor(AX_FACE)

    fig.suptitle(
        "Forward data of station: [{}]".format(sta),
        x=0.5,
        y=0.985,
        ha="center",
        va="top",
        fontfamily="serif",
        color="darkgreen",
        fontsize=30,
        fontweight="bold",
    )

    # -------------------------------------------------------------------------
    # Vs full-depth panel
    # -------------------------------------------------------------------------
    plot_model_space(ax_vs_full, inputmodelleft, inputmodelright, label=True)
    ax_vs_full.plot(inputmodel["vs"], inputmodel["dep"], "r^-", lw=2.5, ms=10, label="Start Vs")

    if inputmodel0 is not None and not inputmodel0.empty:
        ax_vs_full.plot(inputmodel0["vs"], inputmodel0["dep"], "bo-", lw=2.5, ms=10, alpha=0.5, label="Input model")

    if moho_depth is not None:
        ax_vs_full.hlines(moho_depth, xmin=VS_XLIM[0], xmax=VS_XLIM[1], colors="k", linestyles="--", linewidth=2.5, label="Moho depth")
        ax_vs_full.text(1.5, moho_depth * 1.05, "{:.1f} km".format(moho_depth), va="center", fontsize=28)

    ax_vs_full.legend(loc="lower left", fontsize=15)
    full_ticks = np.r_[np.arange(0, min(args.full_depth, 5.0) + 0.1, 1.0), np.arange(5, args.full_depth + 0.1, 5.0)]
    set_depth_axis(ax_vs_full, (0.0, args.full_depth), yticks=full_ticks, labeltop=False)

    # -------------------------------------------------------------------------
    # Vs shallow panel
    # -------------------------------------------------------------------------
    ax_vs_shallow.set_title("Predicted model", y=1.05, loc="left", fontdict=FONT_TITLE_BIG)
    plot_model_space(ax_vs_shallow, inputmodelleft, inputmodelright, label=True)
    ax_vs_shallow.plot(inputmodel["vs"], inputmodel["dep"], "r^-", lw=2.5, ms=10, label="Start Vs")

    if inputmodel0 is not None and not inputmodel0.empty:
        ax_vs_shallow.plot(inputmodel0["vs"], inputmodel0["dep"], "bo-", lw=2.5, ms=10, alpha=0.5, label="Input model")

    shallow_ticks = np.arange(0, args.shallow_depth + 0.001, 1.0)
    set_depth_axis(ax_vs_shallow, (0.0, args.shallow_depth), yticks=shallow_ticks, labeltop=True)

    # -------------------------------------------------------------------------
    # Dispersion: Phase velocity and optional Group velocity
    # -------------------------------------------------------------------------
    ax_disp.set_title("Surface-wave dispersion", loc="left", fontdict=FONT_TITLE_PANEL)

    x_for_xlim = []
    y_for_ylim = []

    if inputph is not None and not inputph.empty:
        ax_disp.plot(inputph["per"], inputph["vel"], "r^", label="Predicted Vph", ms=10, zorder=5)
        x_for_xlim.append(inputph["per"].values)
        y_for_ylim.append(inputph["vel"].values)
    if dataph is not None and not dataph.empty:
        ax_disp.errorbar(
            dataph["per"], dataph["vel"], yerr=dataph["unc"],
            fmt="o", color="k", ecolor="k", ms=7,
            elinewidth=1.5, capthick=1.5, label="Period capture", zorder=6,
        )
        x_for_xlim.append(dataph["per"].values)
        y_for_ylim.append(dataph["vel"].values)
    if dataphr is not None and not dataphr.empty:
        ax_disp.errorbar(
            dataphr["per"], dataphr["vel"], yerr=dataphr["unc"],
            fmt="s", color="b", ecolor="b", ms=7,
            elinewidth=1.5, capthick=1.5, label="Observed Vph", zorder=6,
        )


    if inputgv is not None and not inputgv.empty:
        ax_disp.plot(inputgv["per"], inputgv["vel"], "b^", label="Predicted GV", ms=10, zorder=5)
        x_for_xlim.append(inputgv["per"].values)
        y_for_ylim.append(inputgv["vel"].values)
    if datagv is not None and not datagv.empty:
        ax_disp.errorbar(
            datagv["per"], datagv["vel"], yerr=datagv["unc"],
            fmt="s", color="0.25", ecolor="0.25", ms=7,
            elinewidth=1.5, capthick=1.5, label="Data GV", zorder=6,
        )
        x_for_xlim.append(datagv["per"].values)
        y_for_ylim.append(datagv["vel"].values)

    # xlim = finite_minmax(x_for_xlim, pad_fraction=0.08, default=None)
    xlim = 0, 5;
    if xlim is not None:
        ax_disp.set_xlim(xlim)
    # ylim = finite_minmax(y_for_ylim, pad_fraction=0.12, default=DISP_YLIM)
    ylim = 0, 3;
    if ylim is None:
        ylim = DISP_YLIM
    # Keep old broad velocity range unless data are outside it.
    ax_disp.set_ylim(min(DISP_YLIM[0], ylim[0]), max(DISP_YLIM[1], ylim[1]))
    ax_disp.set_xlabel("Period (s)", fontdict=FONT_LABEL)
    ax_disp.set_ylabel("Velocity (km/s)", fontdict=FONT_LABEL)
    ax_disp.grid(color="k", axis="both", linestyle="-", linewidth=2, alpha=0.1, zorder=2)
    ax_disp.tick_params(labelsize=20)
    ax_disp.yaxis.tick_right()
    ax_disp.yaxis.set_label_position("right")
    ax_disp.legend(loc="best", fontsize=15, frameon=False)

    # -------------------------------------------------------------------------
    # H/V panel
    # -------------------------------------------------------------------------
    ax_hv.set_title("H/V", loc="left", fontdict=FONT_TITLE_PANEL)
    if inputhv is not None and not inputhv.empty:
        ax_hv.plot(inputhv["per"], inputhv["hv"], "r^", ms=10, label="Predicted H/V", zorder=5)
    if datahv is not None and not datahv.empty:
        ax_hv.errorbar(
            datahv["per"], datahv["hv"], yerr=datahv["unc"],
            fmt="o", color="k", ecolor="k", ms=7,
            elinewidth=1.5, capthick=1.5, label="Data H/V", zorder=6,
        )

    hxlim = finite_minmax(
        [None if inputhv is None else inputhv["per"].values, None if datahv is None else datahv["per"].values],
        pad_fraction=0.08,
        default=None,
    )
    if hxlim is not None:
        ax_hv.set_xlim(hxlim)
    hylim = finite_minmax(
        [None if inputhv is None else inputhv["hv"].values, None if datahv is None else datahv["hv"].values],
        pad_fraction=0.12,
        default=HV_YLIM,
    )
    if hylim is None:
        hylim = HV_YLIM
    ax_hv.set_ylim(min(HV_YLIM[0], hylim[0]), max(HV_YLIM[1], hylim[1]))
    ax_hv.set_xlabel("Period (s)", fontdict=FONT_LABEL)
    ax_hv.set_ylabel("H/V", fontdict=FONT_LABEL)
    ax_hv.tick_params(labelsize=20)
    ax_hv.grid(color="k", axis="both", linestyle="-", linewidth=2, alpha=0.1, zorder=2)
    ax_hv.yaxis.tick_right()
    ax_hv.yaxis.set_label_position("right")
    ax_hv.legend(loc="upper right", fontsize=15)

    # -------------------------------------------------------------------------
    # Receiver function panel
    # -------------------------------------------------------------------------
    ax_rf.set_title("Receiver function", loc="left", fontdict=FONT_TITLE_PANEL)
    if inputrf is not None and not inputrf.empty:
        ax_rf.plot(inputrf["time"], inputrf["amp"], "r-", lw=2.0, label="Predicted RF", zorder=5)
    if datarf is not None and not datarf.empty:
        ax_rf.errorbar(
            datarf["time"], datarf["amp"], yerr=datarf["unc"],
            fmt="o", color="k", ecolor="k", ms=3,
            elinewidth=1.0, capthick=0.5, label="Data RF", zorder=6, alpha=0.75,
        )

    rxlim = finite_minmax(
        [None if inputrf is None else inputrf["time"].values, None if datarf is None else datarf["time"].values],
        pad_fraction=0.03,
        default=None,
    )
    if rxlim is not None:
        ax_rf.set_xlim(rxlim)
    rylim = finite_minmax(
        [None if inputrf is None else inputrf["amp"].values, None if datarf is None else datarf["amp"].values],
        pad_fraction=0.18,
        default=RF_YLIM,
    )
    if rylim is None:
        rylim = RF_YLIM
    ax_rf.set_ylim(min(RF_YLIM[0], rylim[0]), max(RF_YLIM[1], rylim[1]))
    ax_rf.set_xlabel("Time (s)", fontdict=FONT_LABEL)
    ax_rf.set_ylabel("RF Amp", fontdict=FONT_LABEL)
    ax_rf.grid(color="k", axis="both", linestyle="-", linewidth=2, alpha=0.1, zorder=2)
    ax_rf.tick_params(labelsize=20)
    ax_rf.yaxis.tick_right()
    ax_rf.yaxis.set_label_position("right")
    ax_rf.legend(loc="upper right", fontsize=15)

    plt.tight_layout(rect=[0, 0, 1, 0.965])

    out_dir = os.path.dirname(paths["out_png"])
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(paths["out_png"], bbox_inches="tight", transparent=False, pad_inches=0.1, dpi=DPI)
    plt.close(fig)

    print("Saved: {}".format(paths["out_png"]))
    print("Done.")


if __name__ == "__main__":
    main()

