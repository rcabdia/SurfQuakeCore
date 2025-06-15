from typing import Callable, List, Set

from matplotlib.axes import Axes
from matplotlib.widgets import SpanSelector


# class ExtendSpanSelector(SpanSelector):
#
#     def __init__(self, ax: Axes, onselect: Callable, direction: str, sharex=False, **kwargs):
#         # Keep track of all axes
#         self.__on_move_callback = kwargs.pop("onmove_callback", None)
#         super().__init__(ax, onselect, direction, onmove_callback=self.__on_move, **kwargs)
#
#         self.__kwargs = kwargs
#         self.sharex: bool = sharex
#         self.__sub_axes: List[Axes] = []
#         self.__sub_selectors: Set[SpanSelector] = set()
#
#     def _release(self, event):
#         self.clear_subplot()
#         return super()._release(event)
#
#     def __on_sub_select(self, xmin, xmax):
#         # this is just a placeholder for sub selectors
#         pass
#
#     def clear_subplot(self):
#         all([ss.clear() for ss in self.__sub_selectors])
#
#     def create_sub_selectors(self):
#         self.__sub_selectors = {
#             SpanSelector(axe, self.__on_sub_select, self.direction, **self.__kwargs)
#             for axe in self.__sub_axes
#         }
#
#     def remove_sub_selectors(self):
#         all([s.disconnect_events() for s in self.__sub_selectors])
#         self.__sub_selectors.clear()
#
#     @staticmethod
#     def draw_selector(selector: SpanSelector, vmin, vmax):
#         selector._draw_shape(vmin, vmax)
#         selector.set_visible(True)
#         selector.update()
#
#     def __on_move(self, vmin, vmax):
#         all([self.draw_selector(ss, vmin, vmax) for ss in self.__sub_selectors])
#         if self.__on_move_callback:
#             return self.__on_move_callback(vmin, vmax)
#
#     def set_sub_axes(self, axes: List[Axes]):
#         # TODO: this is a temporary fix due to a bug (possible in matplotlib) selector is duplicated for current ax
#         if self.sharex:
#             self.__sub_axes = axes
#         else:
#             self.__sub_axes = [self.ax]
#
#         self.remove_sub_selectors()
#         self.create_sub_selectors()
#         self.canvas.draw()

class ExtendSpanSelector(SpanSelector):

    def __init__(self, ax: Axes, onselect: Callable, direction: str, sharex=False, **kwargs):
        # Keep track of all axes
        self.__on_move_callback = kwargs.pop("onmove_callback", None)
        super().__init__(ax, onselect, direction, onmove_callback=self.__on_move, **kwargs)

        self.__kwargs = kwargs
        self.sharex: bool = sharex
        self.__sub_axes: List[Axes] = []
        self.__sub_selectors: Set[SpanSelector] = set()

    def _release(self, event):
        # Call original release behavior
        result = super()._release(event)

        # If onselect is set (from parent SpanSelector), manually trigger it
        if self.extents and self.onselect is not None:
            vmin, vmax = self.extents
            self.onselect(vmin, vmax)

        return result

    def __on_sub_select(self, xmin, xmax):
        # this is just a placeholder for sub selectors
        pass

    def clear_subplot(self):
        all([ss.clear() for ss in self.__sub_selectors])

    def create_sub_selectors(self):
        self.__sub_selectors = {
            SpanSelector(axe, self.__on_sub_select, self.direction, **self.__kwargs)
            for axe in self.__sub_axes
        }

    def remove_sub_selectors(self):
        all([s.disconnect_events() for s in self.__sub_selectors])
        self.__sub_selectors.clear()

    @staticmethod
    def draw_selector(selector: SpanSelector, vmin, vmax):
        try:
            selector._selection_completed = False
            selector._draw_shape(vmin, vmax)
            selector.set_visible(True)
            selector.update()
        except Exception as e:
            print(f"[DEBUG] Failed to draw selector: {e}")

    def __on_move(self, vmin, vmax):
        for ss in self.__sub_selectors:
            self.draw_selector(ss, vmin, vmax)

        self.canvas.draw_idle()  # Force redraw

        if self.__on_move_callback:
            return self.__on_move_callback(vmin, vmax)

    def set_sub_axes(self, axes: List[Axes]):
        # TODO: this is a temporary fix due to a bug (possible in matplotlib) selector is duplicated for current ax
        if self.sharex:
            self.__sub_axes = axes
        else:
            self.__sub_axes = [self.ax]

        self.remove_sub_selectors()
        self.create_sub_selectors()
        self.canvas.draw()

