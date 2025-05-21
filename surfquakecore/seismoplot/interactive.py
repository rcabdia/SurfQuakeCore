#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
interactive



:param : 
:type : 
:return: 
:rtype: 
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from obspy import UTCDateTime
import matplotlib.dates as mdt
matplotlib.use("TkAgg")

class SeismicPicker:
    def __init__(self, traces):
        self.traces = traces
        self.fig = None
        self.axs = None
        self.picks = {}  # {trace_id: [pick_times]}
        self.current_pick = None
        self.pick_lines = []  # Reference to vertical pick lines
        self.vspace = 0.3  # Vertical spacing between subplots

    def plot_interactive(self, traces_per_fig=4, sort_by='distance'):
        """Plot traces with interactive picking capability."""
        traces = sorted(self.traces, key=self._sort_key(sort_by))

        for i in range(0, len(traces), traces_per_fig):
            self._create_figure(traces[i:i + traces_per_fig])
            self._setup_pick_interaction()
            plt.show(block=True)

        return self.picks

    def _sort_key(self, sort_by):
        """Generate sorting key function."""
        if sort_by == 'distance':
            return lambda tr: self._compute_distance_azimuth(tr)[0]
        elif sort_by == 'backazimuth':
            return lambda tr: self._compute_distance_azimuth(tr)[1]
        return lambda tr: tr.id

    def _create_figure(self, sub_traces):
        """Create the figure and plot traces."""
        self.fig, self.axs = plt.subplots(
            len(sub_traces), 1,
            figsize=(12, 2 * len(sub_traces)),
            sharex=True,
            gridspec_kw={'hspace': self.vspace}
        )

        if len(sub_traces) == 1:
            self.axs = [self.axs]

        for ax, tr in zip(self.axs, sub_traces):
            t = tr.times("matplotlib")
            ax.plot(t, tr.data, 'k-', linewidth=0.75)
            ax.set_title(f"{tr.id}", fontsize=10)
            self.picks.setdefault(tr.id, [])

        plt.tight_layout()

        # Add clear button
        ax_clear = plt.axes([0.81, 0.01, 0.1, 0.05])
        btn_clear = Button(ax_clear, 'Clear Picks')
        btn_clear.on_clicked(self._clear_picks)

    def _setup_pick_interaction(self):
        """Set up mouse click and key press events."""
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        plt.gcf().text(0.1, 0.95,
                       "Click to place picks | 'd' to delete last | 'c' to clear",
                       fontsize=9)

    def _on_click(self, event):
        """Handle mouse clicks to place picks."""
        if event.inaxes not in self.axs:
            return

        # Find which trace was clicked
        tr_idx = self.axs.index(event.inaxes)
        trace = self.traces[tr_idx]
        pick_time = mdt.num2date(event.xdata)

        # Store pick
        self.picks[trace.id].append(pick_time)
        self.current_pick = (trace.id, pick_time)

        # Draw pick line on all subplots for visibility
        self._draw_pick_lines(pick_time)
        self.fig.canvas.draw()

    def _draw_pick_lines(self, pick_time):
        """Draw vertical lines at pick time on all subplots."""
        # Remove old lines
        for line in self.pick_lines:
            line.remove()
        self.pick_lines = []

        # Convert to matplotlib time format
        x = mdt.date2num(pick_time)

        # Draw new lines
        for ax in self.axs:
            line = ax.axvline(x=x, color='r', linestyle='--', alpha=0.7)
            self.pick_lines.append(line)

    def _on_key_press(self, event):
        """Handle key presses for pick management."""
        if event.key == 'd':  # Delete last pick
            if self.current_pick:
                trace_id, pick_time = self.current_pick
                if pick_time in self.picks[trace_id]:
                    self.picks[trace_id].remove(pick_time)
                self._redraw_picks()

        elif event.key == 'c':  # Clear all picks
            self._clear_picks()

    def _redraw_picks(self):
        """Redraw all pick lines."""
        for line in self.pick_lines:
            line.remove()
        self.pick_lines = []

        # Redraw remaining picks
        for picks in self.picks.values():
            for pick in picks:
                x = mdt.date2num(pick)
                for ax in self.axs:
                    line = ax.axvline(x=x, color='r', linestyle='--', alpha=0.7)
                    self.pick_lines.append(line)

        self.fig.canvas.draw()

    def _clear_picks(self, event=None):
        """Clear all picks from current figure."""
        for trace_id in self.picks:
            self.picks[trace_id] = []
        self.current_pick = None
        self._redraw_picks()

    def get_picks(self):
        """Return picks as {trace_id: [UTCDateTime]}."""
        return {k: [UTCDateTime(p) for p in v]
                for k, v in self.picks.items() if v}