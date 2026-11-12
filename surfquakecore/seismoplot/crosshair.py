from matplotlib.lines import Line2D

CROSSHAIR_MODES = ("single", "vertical", "horizontal", "all")


class MultiAxesCrosshair:
    """
    Crosshair shared by all the axes of one figure.

    mode:
        single     : vertical line in all axes + horizontal line only in the axes
                     under the mouse (default)
        vertical   : vertical line in all axes
        horizontal : horizontal line in all axes, at the same y value
                     (meaningful when the axes share the Y scale, sharey: true)
        all        : vertical and horizontal lines in all axes
    """

    def __init__(self, axes, mode="single", color="k", lw=0.8, ls="--", useblit=True):
        if mode not in CROSSHAIR_MODES:
            raise ValueError(f"crosshair mode must be one of {CROSSHAIR_MODES}, got '{mode}'")

        self.axes = list(axes)
        self._axset = set(self.axes)
        self.mode = mode
        self.canvas = self.axes[0].figure.canvas
        self.useblit = useblit and getattr(self.canvas, "supports_blit", False)
        self.background = None

        # animated lines are skipped in normal draws and in savefig (blit mode)
        style = dict(color=color, lw=lw, ls=ls, animated=self.useblit, visible=False)

        self.vlines = {}
        self.hlines = {}
        for ax in self.axes:
            if mode != "horizontal":
                # x in data units, y in axes fraction -> full-height line
                v = Line2D([0, 0], [0, 1], transform=ax.get_xaxis_transform(), **style)
                ax.add_artist(v)  # add_artist does not change the data limits
                self.vlines[ax] = v
            if mode != "vertical":
                # x in axes fraction, y in data units -> full-width line
                h = Line2D([0, 1], [0, 0], transform=ax.get_yaxis_transform(), **style)
                ax.add_artist(h)
                self.hlines[ax] = h

        self._cids = [
            self.canvas.mpl_connect("motion_notify_event", self.on_move),
            self.canvas.mpl_connect("figure_leave_event", lambda event: self._hide()),
        ]
        if self.useblit:
            self._cids.append(self.canvas.mpl_connect("draw_event", self._on_draw))

    def _lines(self):
        return list(self.vlines.values()) + list(self.hlines.values())

    def _on_draw(self, event):
        # new clean background after every full redraw (zoom, picks, resize...)
        self.background = self.canvas.copy_from_bbox(self.canvas.figure.bbox)

    def on_move(self, event):
        ax = event.inaxes
        if ax not in self._axset or event.xdata is None:
            self._hide()
            return

        x, y = event.xdata, event.ydata
        for v in self.vlines.values():
            v.set_xdata([x, x])
            v.set_visible(True)
        for a, h in self.hlines.items():
            h.set_ydata([y, y])
            h.set_visible(self.mode != "single" or a is ax)
        self._update()

    def _hide(self):
        changed = False
        for line in self._lines():
            if line.get_visible():
                line.set_visible(False)
                changed = True
        if changed:
            self._update()

    def _update(self):
        if self.useblit:
            if self.background is None:  # figure not drawn yet
                return
            self.canvas.restore_region(self.background)
            for line in self._lines():
                if line.get_visible():
                    line.axes.draw_artist(line)
            self.canvas.blit(self.canvas.figure.bbox)
        else:
            self.canvas.draw_idle()

    def disconnect(self):
        for cid in self._cids:
            self.canvas.mpl_disconnect(cid)
        self._cids = []