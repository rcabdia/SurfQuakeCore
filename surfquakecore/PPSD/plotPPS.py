import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patheffects import withStroke
from obspy.signal.spectral_estimation import earthquake_models
import obspy
from obspy import UTCDateTime
import obspy.signal.spectral_estimation as osse


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
    """
    plot_statistics.

    Parameters control what to show instead of GUI checkboxes/spinboxes.
    """

    if show_mean:
        mean = ppsd.get_mean()
        axis.plot(
            mean[0],
            mean[1],
            color="black",
            linewidth=1,
            linestyle="--",
            label="Mean",
        )

    if show_mode:
        mode = ppsd.get_mode()
        axis.plot(
            mode[0],
            mode[1],
            color="green",
            linewidth=1,
            linestyle="--",
            label="Mode",
        )

    if show_nhnm:
        nhnm = osse.get_nhnm()
        axis.plot(
            nhnm[0],
            nhnm[1],
            color="gray",
            linewidth=2,
            linestyle="-",
            label="NHNM (Peterson et al., 2003)",
        )

    if show_nlnm:
        nlnm = osse.get_nlnm()
        axis.plot(
            nlnm[0],
            nlnm[1],
            color="gray",
            linewidth=2,
            linestyle="-",
            label="NLNM (Peterson et al., 2003)",
        )

    if show_earthquakes:
        # same logic as your original method
        for key, data in earthquake_models.items():
            magnitude, distance = key
            frequencies, accelerations = data
            accelerations = np.array(accelerations)
            frequencies = np.array(frequencies)
            periods = 1.0 / frequencies

            # Eq.1 from Clinton and Cauzzi (2013) converts power to density
            ydata = accelerations / (periods ** (-0.5))
            ydata = 20 * np.log10(ydata / 2)

            if not (
                min_mag <= magnitude <= max_mag
                and min_dist <= distance <= max_dist
                and min(ydata) < ppsd.db_bin_edges[-1]
            ):
                continue

            xdata = periods
            axis.plot(xdata, ydata, linewidth=2, color="black")

            leftpoint = np.argsort(xdata)[0]
            if not ydata[leftpoint] < ppsd.db_bin_edges[-1]:
                continue

            axis.text(
                xdata[leftpoint],
                ydata[leftpoint],
                "M%.1f\n%dkm" % (magnitude, distance),
                ha="right",
                va="top",
                color="w",
                weight="bold",
                fontsize="x-small",
                path_effects=[withStroke(linewidth=3, foreground="0.4")],
            )


def plot_ppsds_from_pickle(
    pickle_path,
    stations_per_page=3,
    page=0,
    plot_mode="pdf",              # "pdf" or "variation"
    variation="Diurnal",          # "Diurnal" or "Seasonal" (only if plot_mode="variation")
    starttime=None,               # None or UTCDateTime / ISO string
    endtime=None,                 # None or UTCDateTime / ISO string
    selection=None,               # None or dict: {stnm: {"network": ntwk, "channels": [chn1, ...]}, ...}
    # ---- statistics options (replacing checkboxes/spinboxes) ----
    show_mean=False,
    show_mode=False,
    show_nhnm=False,
    show_nlnm=False,
    show_earthquakes=False,
    min_mag=0.0,
    max_mag=10.0,
    min_dist=0.0,
    max_dist=10000.0,
    # -------------------------------------------------------------
    show=True,                    # show the figure
    save_path=None                # if not None, save figure here
):
    """
    Non-GUI version of your plot_ppsds method with integrated statistics.
    """

    # -------------------------------------------------------------------------
    # Load pickle and build db_query dict like in your GUI method
    # -------------------------------------------------------------------------
    with open(pickle_path, "rb") as f:
        ppsd_db = pickle.load(f)
    for station, station_dict in ppsd_db['nets']['SG'].items():
        station_dict.pop('CDH', None)
        # Save to pickle
        #with open("/Volumes/LaCie 1/gibraltar/ppsds/ppsd.pkl", "wb") as f:
        #    pickle.dump(ppsd_db, f)

    # Convert start/end times
    if isinstance(starttime, str):
        starttime = UTCDateTime(starttime)
    if isinstance(endtime, str):
        endtime = UTCDateTime(endtime)

    # if only one of start/end given, use it as a point time (like your code)
    if starttime is None and endtime is not None:
        starttime = endtime
    if endtime is None and starttime is not None:
        endtime = starttime

    # If still both None: treat like full range -> use the "starttime == endtime" case
    if starttime is None and endtime is None:
        starttime = endtime = UTCDateTime(0)  # any dummy equal values

    # Build db_query from selection or from ppsd_db
    if selection is not None:
        db_query = selection
    else:
        # Auto-select everything in ppsd_db['nets']
        db_query = {}
        nets = ppsd_db.get("nets", {})
        for ntwk, st_dict in nets.items():
            for stnm, ch_dict in st_dict.items():
                db_query.setdefault(stnm, {"network": ntwk, "channels": []})
                for chnm in ch_dict.keys():
                    db_query[stnm]["channels"].append(chnm)

    selected_stations = sorted(list(db_query.keys()))

    if not selected_stations:
        raise ValueError("No stations found / selected.")

    # -------------------------------------------------------------------------
    # Stations per page and pagination logic
    # -------------------------------------------------------------------------
    if len(selected_stations) < stations_per_page:
        stations_per_page = len(selected_stations)

    pages = max(1, int(np.ceil(len(selected_stations) / stations_per_page)))

    if page < 0 or page >= pages:
        raise ValueError(f"Page {page} out of range [0, {pages - 1}]")

    st1 = stations_per_page * page
    st2 = min(st1 + stations_per_page, len(selected_stations))

    # -------------------------------------------------------------------------
    # Figure + axes
    # -------------------------------------------------------------------------
    fig = plt.figure()
    gs = gridspec.GridSpec(stations_per_page, 3, figure=fig)
    gs.update(
        left=0.10,
        right=0.95,
        top=0.95,
        bottom=0.075,
        hspace=0.35,
        wspace=0.35,
    )
    axes = [fig.add_subplot(gs[i]) for i in range(stations_per_page * 3)]

    # -------------------------------------------------------------------------
    # Plot loop
    # -------------------------------------------------------------------------
    j = 0  # axis index for first channel of a station

    for si in range(st1, st2):
        stnm = selected_stations[si]
        ntwk = db_query[stnm]["network"]
        channels = sorted(db_query[stnm]["channels"])

        c = 0  # channel index for this station

        for chnm in channels:
            try:
                ppsd = ppsd_db["nets"][ntwk][stnm][chnm][1]
            except Exception as e:
                print(f"Error getting PPSDs for {ntwk}.{stnm}.{chnm}: {e}")
                continue

            if starttime == endtime:
                ppsd.calculate_histogram()
            else:
                ppsd.calculate_histogram(starttime=starttime, endtime=endtime)

            ax = axes[j + c]
            if plot_mode == "pdf":
                try:
                    zdata = (
                        ppsd.current_histogram * 100
                        / (ppsd.current_histogram_count or 1)
                    )
                except Exception:
                    raise RuntimeError("Some data channel is not valid.")

                xedges = ppsd.period_xedges
                yedges = ppsd.db_bin_edges
                meshgrid = np.meshgrid(xedges, yedges)

                pcolor = ax.pcolormesh(
                    meshgrid[0],
                    meshgrid[1],
                    zdata.T,
                    cmap=obspy.imaging.cm.pqlx,
                )
                ax.set_xscale("log")
                ax.set_xlabel("Period (s)")
                ax.set_ylabel("Amplitude [$m^2/s^4/Hz$] [dB]")

                # --- statistics overlays ---
                plot_statistics(
                    ax,
                    ppsd,
                    show_mean=show_mean,
                    show_mode=show_mode,
                    show_nhnm=show_nhnm,
                    show_nlnm=show_nlnm,
                    show_earthquakes=show_earthquakes,
                    min_mag=min_mag,
                    max_mag=max_mag,
                    min_dist=min_dist,
                    max_dist=max_dist,
                )
                # ---------------------------

                ax.set_xlim(0.02, 120)

                sum_check = j + c
                if sum_check % 3 == 2:
                    cbar = fig.colorbar(
                        pcolor,
                        ax=axes[sum_check],
                        orientation="vertical",
                        fraction=0.05,
                        pad=0.05,
                    )
                    cbar.set_label("Probability [%]")

            elif plot_mode == "variation":
                if variation == "Diurnal":
                    hist_dict = {}
                    num_period_bins = len(ppsd.period_bin_centers)
                    num_db_bins = len(ppsd.db_bin_centers)
                    for h in range(24):
                        hist_dict.setdefault(
                            h,
                            np.zeros(
                                (num_period_bins, num_db_bins),
                                dtype=np.uint64,
                            ),
                        )

                    for i, time in enumerate(ppsd.times_processed):
                        if starttime != time:
                            if not (starttime < time < endtime):
                                continue

                        hour = time.hour
                        inds = ppsd._binned_psds[i]
                        inds = (
                            ppsd.db_bin_edges.searchsorted(
                                inds, side="left"
                            )
                            - 1
                        )
                        inds[inds == -1] = 0
                        inds[inds == num_db_bins] -= 1
                        for pi, inds_ in enumerate(inds):
                            hist_dict[hour][pi, inds_] += 1

                    modes = []
                    for h in sorted(hist_dict.keys()):
                        current_hist = hist_dict[h]
                        mode = ppsd.db_bin_centers[current_hist.argmax(axis=1)]
                        modes.append(mode)

                    x = ppsd.period_bin_centers
                    y = np.arange(1, 25, 1)

                    contour = ax.contourf(
                        y,
                        x,
                        np.array(modes).T,
                        cmap=obspy.imaging.cm.pqlx,
                        levels=200,
                    )
                    ax.set_xlabel("GMT Hour")
                    ax.set_ylabel("Period (s)")
                    ax.set_ylim(0.02, 120)

                    sum_check = j + c
                    if sum_check in (2, 5, 8):
                        cbar = fig.colorbar(
                            contour,
                            ax=axes[sum_check],
                            orientation="vertical",
                            fraction=0.05,
                            pad=0.05,
                        )
                        cbar.set_label("Probability [%]")

                elif variation == "Seasonal":
                    hist_dict = {}
                    num_period_bins = len(ppsd.period_bin_centers)
                    num_db_bins = len(ppsd.db_bin_centers)
                    for m in range(12):
                        hist_dict.setdefault(
                            m,
                            np.zeros(
                                (num_period_bins, num_db_bins),
                                dtype=np.uint64,
                            ),
                        )

                    for i, time in enumerate(ppsd.times_processed):
                        if starttime != time:
                            if not (starttime < time < endtime):
                                continue

                        month = time.month
                        inds = ppsd._binned_psds[i]
                        inds = (
                            ppsd.db_bin_edges.searchsorted(
                                inds, side="left"
                            )
                            - 1
                        )
                        inds[inds == -1] = 0
                        inds[inds == num_db_bins] -= 1
                        for pi, inds_ in enumerate(inds):
                            hist_dict[month - 1][pi, inds_] += 1

                    modes = []
                    for m in sorted(hist_dict.keys()):
                        current_hist = hist_dict[m]
                        mode = ppsd.db_bin_centers[current_hist.argmax(axis=1)]
                        modes.append(mode)

                    x = ppsd.period_bin_centers
                    y = np.arange(1, 13, 1)

                    contour = ax.contourf(
                        y,
                        x,
                        np.array(modes).T,
                        cmap=obspy.imaging.cm.pqlx,
                        levels=200,
                    )
                    ax.set_xlabel("Month")
                    ax.set_ylabel("Period (s)")
                    ax.set_ylim(0.02, 120)
                    ax.set_xlim(1, 12)

                    sum_check = j + c
                    if sum_check in (2, 5, 8):
                        cbar = fig.colorbar(
                            contour,
                            ax=axes[sum_check],
                            orientation="vertical",
                            fraction=0.05,
                            pad=0.05,
                        )
                        cbar.set_label("Probability [%]")

            ax.set_title(chnm, fontsize=9, fontweight="medium")
            if c == 0:
                axes[j].set_title(
                    stnm,
                    loc="left",
                    fontsize=11,
                    fontweight="bold",
                )

            c += 1

        if c < 3:
            for l in range(c, 3):
                axes[j + l].set_axis_off()
            c = 3

        j += c

    if j < stations_per_page * 3:
        for m in range(j, stations_per_page * 3):
            axes[m].set_axis_off()

    # Figure size (no constrained_layout to avoid the previous bug)
    if stations_per_page == 1:
        fig.set_size_inches(11, 4)
    elif stations_per_page == 2:
        fig.set_size_inches(11, 6)
    else:
        fig.set_size_inches(11, 8)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show()

    return fig, axes


if __name__ == "__main__":
    # plot_mode = "pdf",  # "pdf" or "variation"
    # variation = "Diurnal",  # "Diurnal" or "Seasonal" (only if plot_mode="variation")

    fig, axes = plot_ppsds_from_pickle(
        "/Volumes/LaCie/gibraltar/ppsds/ppsd.pkl", stations_per_page=4,
        page=0, plot_mode="pdf",
        variation="Seasonal", starttime=None, endtime=None,
        show_mean=True, show_mode=True, show_nhnm=True, show_nlnm=True,
        show_earthquakes=True, min_mag=1.0, max_mag=3.0, min_dist=10.0, max_dist=100.0)

