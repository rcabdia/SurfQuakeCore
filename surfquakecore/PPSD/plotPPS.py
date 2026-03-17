#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plotPPS.py  –  Plotting utilities for PPSD databases produced by PPSDSurf.

Key features
------------
* plot_ppsds_from_pickle  : plot a single page (N stations, 3 channels each).
* plot_all_pages          : iterate over ALL stations and save/show one figure
                            per group of `stations_per_page` stations.
* plot_statistics         : overlay mean, mode, NHNM, NLNM, earthquake models.

Usage example (bottom of file).
"""

import os
import pickle
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patheffects import withStroke

from obspy.signal.spectral_estimation import earthquake_models
import obspy
import obspy.imaging.cm
from obspy import UTCDateTime
import obspy.signal.spectral_estimation as osse


# ---------------------------------------------------------------------------
# Statistics overlay helper
# ---------------------------------------------------------------------------

def plot_statistics(
    axis,
    ppsd,
    show_mean=False,
    show_mode=False,
    show_nhnm=False,
    show_nlnm=False,
    show_earthquakes=False,
    min_mag=0.0,
    max_mag=10.0,
    min_dist=0.0,
    max_dist=10000.0,
):
    """Overlay statistical curves on a PPSD axis."""

    if show_mean:
        mean = ppsd.get_mean()
        axis.plot(mean[0], mean[1],
                  color="black", linewidth=1, linestyle="--", label="Mean")

    if show_mode:
        mode = ppsd.get_mode()
        axis.plot(mode[0], mode[1],
                  color="green", linewidth=1, linestyle="--", label="Mode")

    if show_nhnm:
        nhnm = osse.get_nhnm()
        axis.plot(nhnm[0], nhnm[1],
                  color="gray", linewidth=2, linestyle="-",
                  label="NHNM (Peterson et al., 2003)")

    if show_nlnm:
        nlnm = osse.get_nlnm()
        axis.plot(nlnm[0], nlnm[1],
                  color="gray", linewidth=2, linestyle="-",
                  label="NLNM (Peterson et al., 2003)")

    if show_earthquakes:
        for key, data in earthquake_models.items():
            magnitude, distance = key
            frequencies, accelerations = data
            accelerations = np.array(accelerations)
            frequencies = np.array(frequencies)
            periods = 1.0 / frequencies

            # Eq.1 from Clinton and Cauzzi (2013)
            ydata = accelerations / (periods ** (-0.5))
            ydata = 20 * np.log10(ydata / 2)

            if not (
                min_mag <= magnitude <= max_mag
                and min_dist <= distance <= max_dist
                and min(ydata) < ppsd.db_bin_edges[-1]
            ):
                continue

            axis.plot(periods, ydata, linewidth=2, color="black")
            leftpoint = np.argsort(periods)[0]
            if not ydata[leftpoint] < ppsd.db_bin_edges[-1]:
                continue

            axis.text(
                periods[leftpoint], ydata[leftpoint],
                "M%.1f\n%dkm" % (magnitude, distance),
                ha="right", va="top", color="w", weight="bold",
                fontsize="x-small",
                path_effects=[withStroke(linewidth=3, foreground="0.4")],
            )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_and_build_query(pickle_path, selection=None,
                          net_pattern="*", sta_pattern="*", chn_pattern="*"):
    """
    Load pickle and return (ppsd_db, db_query, selected_stations).

    If *selection* is provided it is used as-is (explicit dict, legacy path).
    Otherwise the wildcard patterns net_pattern / sta_pattern / chn_pattern
    are applied to every entry in the database, so CLI filters like -ch SHZ
    or -st OBS* are honoured by all three plot functions.
    """
    with open(pickle_path, "rb") as f:
        ppsd_db = pickle.load(f)

    if selection is not None:
        db_query = selection
    else:
        db_query = {}
        nets = ppsd_db.get("nets", {})
        for ntwk, st_dict in nets.items():
            if not _wildcard_filter([ntwk], net_pattern):
                continue
            for stnm, ch_dict in st_dict.items():
                if not _wildcard_filter([stnm], sta_pattern):
                    continue
                channels = _wildcard_filter(list(ch_dict.keys()), chn_pattern)
                if not channels:
                    continue
                db_query.setdefault(stnm, {"network": ntwk, "channels": []})
                db_query[stnm]["channels"].extend(
                    c for c in channels
                    if c not in db_query[stnm]["channels"]
                )

    selected_stations = sorted(db_query.keys())
    if not selected_stations:
        raise ValueError(
            f"No stations found after filtering "
            f"net='{net_pattern}' sta='{sta_pattern}' chn='{chn_pattern}'."
        )

    return ppsd_db, db_query, selected_stations


def _normalise_times(starttime, endtime):
    """Convert strings to UTCDateTime and handle None values."""
    if isinstance(starttime, str):
        starttime = UTCDateTime(starttime)
    if isinstance(endtime, str):
        endtime = UTCDateTime(endtime)
    if starttime is None and endtime is not None:
        starttime = endtime
    if endtime is None and starttime is not None:
        endtime = starttime
    if starttime is None and endtime is None:
        starttime = endtime = UTCDateTime(0)
    return starttime, endtime


def _figure_height(n_stations):
    """Return a sensible figure height for N station rows."""
    if n_stations == 1:
        return 4
    elif n_stations == 2:
        return 6
    else:
        return max(8, n_stations * 2.8)


# ---------------------------------------------------------------------------
# Single-page plot
# ---------------------------------------------------------------------------

def plot_ppsds_from_pickle(
    pickle_path,
    stations_per_page=3,
    page=0,
    plot_mode="pdf",          # "pdf" | "variation"
    variation="Diurnal",      # "Diurnal" | "Seasonal"
    starttime=None,
    endtime=None,
    selection=None,           # {stnm: {"network": ntwk, "channels": [...]}, ...}
    net_pattern="*",          # wildcard filter for networks
    sta_pattern="*",          # wildcard filter for stations
    chn_pattern="*",          # wildcard filter for channels
    # ---- statistics options ----
    show_mean=False,
    show_mode=False,
    show_nhnm=False,
    show_nlnm=False,
    show_earthquakes=False,
    min_mag=0.0,
    max_mag=10.0,
    min_dist=0.0,
    max_dist=10000.0,
    # ----------------------------
    show=True,
    save_path=None,
    fig_title=None,           # optional suptitle for the figure
):
    """
    Plot one page of PPSD results (up to `stations_per_page` stations,
    3 channels each).

    Parameters
    ----------
    pickle_path : str
        Path to the .pkl file produced by PPSDSurf.
    stations_per_page : int
        How many stations to show on this figure.
    page : int
        Zero-based page index.
    plot_mode : str
        "pdf" for the probability density plot, "variation" for diurnal/seasonal.
    variation : str
        "Diurnal" or "Seasonal" (used only when plot_mode="variation").
    starttime, endtime : None | str | UTCDateTime
        Time window for histogram calculation.
    selection : dict | None
        Subset of stations/channels to use.  None = use everything.
    show_mean, show_mode, show_nhnm, show_nlnm, show_earthquakes : bool
        Statistical overlays.
    min_mag, max_mag, min_dist, max_dist : float
        Magnitude / distance filters for earthquake model lines.
    show : bool
        Whether to call plt.show().
    save_path : str | None
        If provided, save the figure to this path.
    fig_title : str | None
        Optional super-title placed at the top of the figure.

    Returns
    -------
    fig, axes
    """

    ppsd_db, db_query, selected_stations = _load_and_build_query(
        pickle_path, selection,
        net_pattern=net_pattern, sta_pattern=sta_pattern, chn_pattern=chn_pattern)
    starttime, endtime = _normalise_times(starttime, endtime)

    # Clamp stations_per_page
    stations_per_page = min(stations_per_page, len(selected_stations))
    pages = max(1, math.ceil(len(selected_stations) / stations_per_page))

    if page < 0 or page >= pages:
        raise ValueError(f"Page {page} out of range [0, {pages - 1}].")

    st1 = stations_per_page * page
    st2 = min(st1 + stations_per_page, len(selected_stations))
    page_stations = selected_stations[st1:st2]
    n_rows = len(page_stations)

    # ---- build figure -------------------------------------------------------
    fig = plt.figure(figsize=(11, _figure_height(n_rows)))
    if fig_title:
        fig.suptitle(fig_title, fontsize=13, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(n_rows, 3, figure=fig)
    gs.update(left=0.10, right=0.95,
              top=0.93 if fig_title else 0.95,
              bottom=0.075,
              hspace=0.45, wspace=0.40)
    axes = [fig.add_subplot(gs[i]) for i in range(n_rows * 3)]

    # ---- plot loop ----------------------------------------------------------
    for row_idx, stnm in enumerate(page_stations):
        ntwk = db_query[stnm]["network"]
        channels = sorted(db_query[stnm]["channels"])
        ax_base = row_idx * 3  # first axis index for this station row

        for col_idx, chnm in enumerate(channels[:3]):
            try:
                ppsd = ppsd_db["nets"][ntwk][stnm][chnm][1]
            except Exception as e:
                print(f"  [WARN] Cannot retrieve PPSD for {ntwk}.{stnm}.{chnm}: {e}")
                axes[ax_base + col_idx].set_axis_off()
                continue

            if starttime == endtime:
                ppsd.calculate_histogram()
            else:
                ppsd.calculate_histogram(starttime=starttime, endtime=endtime)

            ax = axes[ax_base + col_idx]

            # ---- PDF mode --------------------------------------------------
            if plot_mode == "pdf":
                zdata = (ppsd.current_histogram * 100
                         / (ppsd.current_histogram_count or 1))

                xedges = ppsd.period_xedges
                yedges = ppsd.db_bin_edges
                mg = np.meshgrid(xedges, yedges)

                pcolor = ax.pcolormesh(mg[0], mg[1], zdata.T,
                                       cmap=obspy.imaging.cm.pqlx)
                ax.set_xscale("log")
                ax.set_xlabel("Period (s)", fontsize=8)
                ax.set_ylabel("Amplitude [$m^2/s^4/Hz$] [dB]", fontsize=8)
                ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)
                ax.set_xlim(0.02, 120)

                plot_statistics(
                    ax, ppsd,
                    show_mean=show_mean, show_mode=show_mode,
                    show_nhnm=show_nhnm, show_nlnm=show_nlnm,
                    show_earthquakes=show_earthquakes,
                    min_mag=min_mag, max_mag=max_mag,
                    min_dist=min_dist, max_dist=max_dist,
                )

                # Colorbar on the rightmost channel of each station row
                if col_idx == min(len(channels), 3) - 1:
                    cbar = fig.colorbar(pcolor, ax=ax,
                                        orientation="vertical",
                                        fraction=0.05, pad=0.05)
                    cbar.set_label("Probability [%]", fontsize=8)

            # ---- variation mode --------------------------------------------
            elif plot_mode == "variation":
                num_period_bins = len(ppsd.period_bin_centers)
                num_db_bins = len(ppsd.db_bin_centers)

                if variation == "Diurnal":
                    n_bins, x_label = 24, "GMT Hour"
                    x_values = np.arange(1, 25)
                    time_key_fn = lambda t: t.hour
                elif variation == "Seasonal":
                    n_bins, x_label = 12, "Month"
                    x_values = np.arange(1, 13)
                    time_key_fn = lambda t: t.month - 1
                else:
                    raise ValueError(f"Unknown variation type: {variation!r}")

                hist_dict = {
                    k: np.zeros((num_period_bins, num_db_bins), dtype=np.uint64)
                    for k in range(n_bins)
                }

                for i, time in enumerate(ppsd.times_processed):
                    no_filter = (starttime == endtime)  # True when user passed no -t0/-t1
                    if not no_filter:
                        if not (starttime <= time <= endtime):
                            continue
                    key = time_key_fn(time)
                    inds = ppsd._binned_psds[i]
                    inds = ppsd.db_bin_edges.searchsorted(inds, side="left") - 1
                    inds[inds == -1] = 0
                    inds[inds == num_db_bins] -= 1
                    for pi, inds_ in enumerate(inds):
                        hist_dict[key][pi, inds_] += 1

                modes = [ppsd.db_bin_centers[hist_dict[k].argmax(axis=1)]
                         for k in sorted(hist_dict)]

                contour = ax.contourf(x_values,
                                      ppsd.period_bin_centers,
                                      np.array(modes).T,
                                      cmap=obspy.imaging.cm.pqlx,
                                      levels=200)
                ax.set_xlabel(x_label, fontsize=8)
                ax.set_ylabel("Period (s)", fontsize=8)
                ax.set_ylim(0.02, 120)
                if variation == "Seasonal":
                    ax.set_xlim(1, 12)

                if col_idx == min(len(channels), 3) - 1:
                    cbar = fig.colorbar(contour, ax=ax,
                                        orientation="vertical",
                                        fraction=0.05, pad=0.05)
                    #cbar.set_label("Amplitude [dB]", fontsize=8)
                    cbar.set_label("Amplitude [$m^2/s^4/Hz$] [dB]", fontsize=8)

            # ---- titles ----------------------------------------------------
            ax.set_title(chnm, fontsize=9, fontweight="medium")
            ax.tick_params(labelsize=7)
            if col_idx == 0:
                ax.set_title(f"{stnm}  –  {chnm}", fontsize=9, fontweight="bold")

        # Hide unused axes in this row (< 3 channels)
        for empty in range(len(channels), 3):
            axes[ax_base + empty].set_axis_off()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")

    if show:
        plt.show()

    return fig, axes


# ---------------------------------------------------------------------------
# Multi-page / all-stations iterator
# ---------------------------------------------------------------------------

def plot_all_pages(
    pickle_path,
    stations_per_page=3,
    plot_mode="pdf",
    variation="Diurnal",
    starttime=None,
    endtime=None,
    selection=None,
    net_pattern="*",
    sta_pattern="*",
    chn_pattern="*",
    # statistics
    show_mean=False,
    show_mode=False,
    show_nhnm=False,
    show_nlnm=False,
    show_earthquakes=False,
    min_mag=0.0,
    max_mag=10.0,
    min_dist=0.0,
    max_dist=10000.0,
    # output
    show=True,
    save_dir=None,            # directory to save figures (one per page)
    save_prefix="ppsd_page",  # filename prefix; files will be <prefix>_01.png etc.
    save_format="png",        # "png", "pdf", "svg" …
):
    """
    Iterate over ALL stations and produce one figure per group of
    `stations_per_page` stations.

    Parameters
    ----------
    pickle_path : str
        Path to the .pkl file produced by PPSDSurf.
    stations_per_page : int
        Number of station rows per figure.
    plot_mode : str
        "pdf" or "variation".
    variation : str
        "Diurnal" or "Seasonal".
    starttime, endtime : None | str | UTCDateTime
        Time window passed to ppsd.calculate_histogram().
    selection : dict | None
        Restrict to a subset of stations/channels.
    net_pattern, sta_pattern, chn_pattern : str
        Wildcard filters applied when selection=None.
    show_mean, show_mode, show_nhnm, show_nlnm, show_earthquakes : bool
        Statistical overlays forwarded to each panel.
    min_mag, max_mag, min_dist, max_dist : float
        Earthquake model filters.
    show : bool
        Call plt.show() after each figure.  Set False for batch saving.
    save_dir : str | None
        Directory for saving figures.  Created automatically if needed.
        Pass None to skip saving.
    save_prefix : str
        Prefix for saved filenames.
    save_format : str
        File extension / format understood by matplotlib.

    Returns
    -------
    list of (fig, axes) tuples – one per page.
    """

    # Determine total number of pages
    _ppsd_db, _db_query, selected_stations = _load_and_build_query(
        pickle_path, selection,
        net_pattern=net_pattern, sta_pattern=sta_pattern, chn_pattern=chn_pattern)
    stations_per_page = min(stations_per_page, len(selected_stations))
    n_pages = max(1, math.ceil(len(selected_stations) / stations_per_page))

    print(f"Total stations : {len(selected_stations)}")
    print(f"Stations/page  : {stations_per_page}")
    print(f"Total pages    : {n_pages}")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    results = []
    for page in range(n_pages):
        st1 = stations_per_page * page
        st2 = min(st1 + stations_per_page, len(selected_stations))
        page_label = f"Page {page + 1}/{n_pages}  –  stations: " \
                     f"{', '.join(selected_stations[st1:st2])}"
        print(f"\n[Page {page + 1}/{n_pages}] {selected_stations[st1:st2]}")

        save_path = None
        if save_dir is not None:
            fname = f"{save_prefix}_{page + 1:02d}.{save_format}"
            save_path = os.path.join(save_dir, fname)

        fig, axes = plot_ppsds_from_pickle(
            pickle_path=pickle_path,
            stations_per_page=stations_per_page,
            page=page,
            plot_mode=plot_mode,
            variation=variation,
            starttime=starttime,
            endtime=endtime,
            selection=selection,
            net_pattern=net_pattern,
            sta_pattern=sta_pattern,
            chn_pattern=chn_pattern,
            show_mean=show_mean,
            show_mode=show_mode,
            show_nhnm=show_nhnm,
            show_nlnm=show_nlnm,
            show_earthquakes=show_earthquakes,
            min_mag=min_mag,
            max_mag=max_mag,
            min_dist=min_dist,
            max_dist=max_dist,
            show=show,
            save_path=save_path,
            fig_title=page_label,
        )
        results.append((fig, axes))

        if not show:
            plt.close(fig)  # free memory when batch-saving

    print(f"\nDone. {n_pages} figure(s) produced.")
    return results


# ---------------------------------------------------------------------------
# Comparison plot  –  overlay stat curves across stations / channels
# ---------------------------------------------------------------------------

# Colour cycle used for the comparison lines (one colour per station)
_COMPARISON_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#fffac8", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#000000",
]

_STAT_LINESTYLES = {
    "mean":   ("--", 1.5),
    "median": ("-",  1.5),
    "mode":   (":",  1.5),
}


def _wildcard_filter(names, pattern):
    """
    Return the subset of *names* that match *pattern*.

    *pattern* may be:
      - None or ""  → match everything
      - "*"         → match everything
      - "OBS1,OBS2" → exact comma-separated list
      - "OBS*"      → single wildcard glob
      - "OBS?,ES*"  → comma-separated list of globs
    """
    from fnmatch import fnmatch

    if pattern is None or str(pattern).strip() in ("", "*"):
        return list(names)

    patterns = [p.strip() for p in str(pattern).split(",") if p.strip()]
    return [n for n in names if any(fnmatch(n, p) for p in patterns)]


def _compute_stat(ppsd, stat):
    """
    Return (periods, values) for the requested statistic.

    stat : "mean" | "median" | "mode"
    """
    stat = stat.lower()
    if stat == "mean":
        result = ppsd.get_mean()
        return result[0], result[1]
    elif stat == "median":
        # ObsPy ≥ 1.4 exposes get_percentile; use 50th percentile as median
        try:
            result = ppsd.get_percentile(percentile=50)
            return result[0], result[1]
        except AttributeError:
            # Fall back to mean if get_percentile not available
            result = ppsd.get_mean()
            return result[0], result[1]
    elif stat == "mode":
        result = ppsd.get_mode()
        return result[0], result[1]
    else:
        raise ValueError(f"Unknown statistic '{stat}'. Choose mean, median, or mode.")


def plot_comparison(
    pickle_path,
    sta_pattern="*",          # wildcard(s) for station names, e.g. "OBS*" or "OBS1,OBS3"
    chn_pattern="*",          # wildcard(s) for channel codes, e.g. "BH?" or "HHZ,BHZ"
    net_pattern="*",          # wildcard(s) for network codes
    stats=("mean",),          # tuple/list of statistics: any subset of mean, median, mode
    starttime=None,
    endtime=None,
    # reference curves
    show_nhnm=True,
    show_nlnm=True,
    # layout
    layout="by_component",    # "by_component" (one panel per component) | "single"
    period_lim=(0.02, 120),   # x-axis limits
    db_lim=None,              # y-axis limits or None for auto
    # output
    show=True,
    save_path=None,
    fig_title=None,
):
    """
    Comparison plot: overlay statistical summary curves (mean / median / mode)
    for a user-defined selection of stations and channels on shared axes.

    Parameters
    ----------
    pickle_path : str
        Path to the .pkl file produced by PPSDSurf.
    sta_pattern : str
        Wildcard pattern for station names.
        Examples: "*", "OBS*", "OBS1,OBS3", "ES?,WM*"
    chn_pattern : str
        Wildcard pattern for channel codes.
        Examples: "*", "BH?", "HHZ", "BHZ,HHZ"
    net_pattern : str
        Wildcard pattern for network codes.
        Examples: "*", "IU", "II,IU"
    stats : tuple of str
        Which statistics to plot.  Any combination of "mean", "median", "mode".
    starttime, endtime : None | str | UTCDateTime
        Time window for histogram calculation.  None = full dataset.
    show_nhnm, show_nlnm : bool
        Draw Peterson (1993) noise model bounds as reference.
    layout : str
        "by_component" → one panel per unique channel component (last letter),
                          all matching stations overlaid.
        "single"        → everything on one shared panel.
    period_lim : tuple
        (xmin, xmax) period axis limits in seconds.
    db_lim : tuple | None
        (ymin, ymax) dB axis limits.  None = matplotlib auto.
    show : bool
        Whether to call plt.show().
    save_path : str | None
        If provided, save figure here.
    fig_title : str | None
        Overall figure title.

    Returns
    -------
    fig, axes
    """

    # ---- load data ----------------------------------------------------------
    with open(pickle_path, "rb") as f:
        ppsd_db = pickle.load(f)

    starttime, endtime = _normalise_times(starttime, endtime)

    # ---- collect all matching (net, sta, chn, ppsd) tuples ------------------
    nets_dict = ppsd_db.get("nets", {})
    all_nets   = list(nets_dict.keys())
    matched_nets = _wildcard_filter(all_nets, net_pattern)

    entries = []   # list of (net, sta, chn, ppsd_obj)
    for ntwk in matched_nets:
        matched_stas = _wildcard_filter(list(nets_dict[ntwk].keys()), sta_pattern)
        for stnm in matched_stas:
            matched_chns = _wildcard_filter(
                list(nets_dict[ntwk][stnm].keys()), chn_pattern)
            for chnm in matched_chns:
                try:
                    ppsd_obj = ppsd_db["nets"][ntwk][stnm][chnm][1]
                    if ppsd_obj is None:
                        continue
                    entries.append((ntwk, stnm, chnm, ppsd_obj))
                except Exception as e:
                    print(f"  [WARN] Skipping {ntwk}.{stnm}.{chnm}: {e}")

    if not entries:
        raise ValueError(
            f"No PPSD data found for sta='{sta_pattern}', "
            f"chn='{chn_pattern}', net='{net_pattern}'."
        )

    print(f"Matched {len(entries)} channel(s):")
    for ntwk, stnm, chnm, _ in entries:
        print(f"  {ntwk}.{stnm}.{chnm}")

    # ---- compute histograms -------------------------------------------------
    for _ntwk, _stnm, _chnm, ppsd_obj in entries:
        try:
            if starttime == endtime:
                ppsd_obj.calculate_histogram()
            else:
                ppsd_obj.calculate_histogram(starttime=starttime,
                                             endtime=endtime)
        except Exception as e:
            print(f"  [WARN] calculate_histogram failed for "
                  f"{_ntwk}.{_stnm}.{_chnm}: {e}")

    # ---- determine panel layout --------------------------------------------
    if layout == "by_component":
        # Group by last character of channel code (Z, N, E, 1, 2, 3, …)
        components = sorted({chnm[-1] for _, _, chnm, _ in entries})
        n_panels = len(components)
        panel_map = {comp: [] for comp in components}
        for entry in entries:
            panel_map[entry[2][-1]].append(entry)
    else:  # "single"
        components = ["all"]
        n_panels = 1
        panel_map = {"all": entries}

    # ---- build figure -------------------------------------------------------
    fig_width  = max(6, n_panels * 5.5)
    fig_height = 5.5

    fig, axes_raw = plt.subplots(
        1, n_panels,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )
    axes_list = axes_raw[0]   # shape (n_panels,)

    fig.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.12,
                        wspace=0.35)

    title_str = fig_title or (
        f"PPSD comparison  |  sta: {sta_pattern}  chn: {chn_pattern}"
        f"  |  stats: {', '.join(stats)}"
    )
    fig.suptitle(title_str, fontsize=11, fontweight="bold")

    # Reference curves (drawn once per panel, behind all lines)
    nhnm_data = osse.get_nhnm() if show_nhnm else None
    nlnm_data = osse.get_nlnm() if show_nlnm else None

    # Assign one colour per *station* (consistent across panels)
    all_stations = sorted({stnm for _, stnm, _, _ in entries})
    sta_color = {
        stnm: _COMPARISON_COLORS[i % len(_COMPARISON_COLORS)]
        for i, stnm in enumerate(all_stations)
    }

    # ---- plot ---------------------------------------------------------------
    for panel_idx, comp in enumerate(components):
        ax = axes_list[panel_idx]
        panel_entries = panel_map[comp]

        # Reference noise models
        if nlnm_data is not None:
            ax.plot(nlnm_data[0], nlnm_data[1],
                    color="gray", linewidth=1.8, linestyle="-", zorder=1,
                    label="NLNM")
        if nhnm_data is not None:
            ax.plot(nhnm_data[0], nhnm_data[1],
                    color="gray", linewidth=1.8, linestyle="-", zorder=1,
                    label="NHNM")

        # One curve per (station × stat)
        plotted_labels = set()
        for ntwk, stnm, chnm, ppsd_obj in panel_entries:
            color = sta_color[stnm]
            for stat in stats:
                ls, lw = _STAT_LINESTYLES.get(stat, ("-", 1.5))
                try:
                    periods, values = _compute_stat(ppsd_obj, stat)
                except Exception as e:
                    print(f"  [WARN] {stat} failed for "
                          f"{ntwk}.{stnm}.{chnm}: {e}")
                    continue

                # Build legend label: show stat only if more than one requested
                if len(stats) > 1:
                    label = f"{stnm} ({stat})"
                else:
                    label = stnm

                # Avoid duplicate legend entries for same station
                legend_key = f"{stnm}_{stat}"
                ax.plot(periods, values,
                        color=color, linewidth=lw, linestyle=ls,
                        label=label if legend_key not in plotted_labels else "_nolegend_",
                        zorder=3)
                plotted_labels.add(legend_key)

        # Axes formatting
        ax.set_xscale("log")
        ax.set_xlim(*period_lim)
        if db_lim is not None:
            ax.set_ylim(*db_lim)
        ax.set_xlabel("Period (s)", fontsize=9)
        ax.set_ylabel("Amplitude [$m^2/s^4/Hz$] [dB]", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)

        panel_title = (f"Component: {comp}" if layout == "by_component"
                       else "All channels")
        ax.set_title(panel_title, fontsize=10, fontweight="medium")

        # Legend: place outside panel on the last panel to avoid overlap
        if panel_idx == n_panels - 1:
            ax.legend(loc="upper left",
                      bbox_to_anchor=(1.01, 1.0),
                      borderaxespad=0,
                      fontsize=7,
                      framealpha=0.85)
        else:
            ax.legend(loc="upper right", fontsize=7, framealpha=0.85)

        # Stat linestyle legend (only if multiple stats)
        if len(stats) > 1 and panel_idx == 0:
            from matplotlib.lines import Line2D
            stat_handles = [
                Line2D([0], [0], color="k",
                       linestyle=_STAT_LINESTYLES[s][0],
                       linewidth=_STAT_LINESTYLES[s][1],
                       label=s.capitalize())
                for s in stats if s in _STAT_LINESTYLES
            ]
            ax.legend(handles=stat_handles,
                      title="Statistic", loc="lower left",
                      fontsize=7, framealpha=0.85)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")

    if show:
        plt.show()

    return fig, axes_list


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # plot_mode = "pdf",  # "pdf" or "variation"
    # variation = "Diurnal",  # "Diurnal" or "Seasonal" (only if plot_mode="variation")

    # fig, axes = plot_ppsds_from_pickle(
    #     "/Volumes/LaCie/gibraltar/ppsds/ppsd.pkl", stations_per_page=4,
    #     page=0, plot_mode="pdf",
    #     variation="Seasonal", starttime=None, endtime=None,
    #     show_mean=True, show_mode=True, show_nhnm=True, show_nlnm=True,
    #     show_earthquakes=True, min_mag=1.0, max_mag=3.0, min_dist=10.0, max_dist=100.0)

    # ── Option A: single page ───────────────────────────────────────────────
    # fig, axes = plot_all_pages(
    #     pickle_path="/Volumes/LaCie/datosOBS/surfquake_test/output/test.pkl",
    #     stations_per_page=2,
    #     plot_mode="pdf",
    #     show_mean=True,
    #     show_mode=True,
    #     show_nhnm=True,
    #     show_nlnm=True,
    #     show_earthquakes=True,
    #     min_mag=1.0, max_mag=3.0,
    #     min_dist=10.0, max_dist=100.0,
    #     show=True,
    # )

    # ── Option B: all pages at once, save to a folder ───────────────────────
    # plot_all_pages(
    #     pickle_path="/Volumes/LaCie/datosOBS/surfquake_test/output/test.pkl",
    #     stations_per_page=3,          # 3 stations per figure
    #     plot_mode="pdf",
    #     show_mean=True,
    #     show_mode=True,
    #     show_nhnm=True,
    #     show_nlnm=True,
    #     show=False,                   # don't pop up windows
    #     save_dir="/tmp/ppsd_figures", # save all figures here
    #     save_prefix="gibraltar",
    #     save_format="png",
    # )

    # Example 2 – compare two specific stations across all BH? channels
    fig, axes = plot_comparison(
        pickle_path="/Volumes/LaCie/datosOBS/surfquake_test/output/test.pkl",
        sta_pattern="OBS01,OBS02,OBS05",      # exact list
        chn_pattern="SH?",            # BHZ, BHN, BHE
        stats=("median",),
        layout="by_component",        # panels: Z | N | E
        show=True)