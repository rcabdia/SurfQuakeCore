#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
crosshair
"""
class Cursor:
    """
    A cross hair cursor.
    """
    def __init__(self, ax):
        self.ax = ax
        self.horizontal_line = ax.axhline(color='k', lw=0.8, ls='--')
        self.vertical_line = ax.axvline(color='k', lw=0.8, ls='--')
        # text location in axes coordinates
        #self.text = ax.text(0.72, 0.9, '', transform=ax.transAxes)

    def set_cross_hair_visible(self, visible):
        need_redraw = self.horizontal_line.get_visible() != visible
        self.horizontal_line.set_visible(visible)
        self.vertical_line.set_visible(visible)
        #self.text.set_visible(visible)
        return need_redraw

    def on_mouse_move(self, event):
        if not event.inaxes:
            need_redraw = self.set_cross_hair_visible(False)
            if need_redraw:
                self.ax.figure.canvas.draw()
        else:
            self.set_cross_hair_visible(True)
            x, y = event.xdata, event.ydata
            # update the line positions
            self.horizontal_line.set_ydata([y])
            self.vertical_line.set_xdata([x])
            #self.text.set_text(f'x={x:1.2f}, y={y:1.2f}')
            self.ax.figure.canvas.draw()
class BlittedCursor:
    def __init__(self, ax, all_axes):
        self.ax = ax
        self.all_axes = all_axes
        self.background = None
        self.horizontal_line = ax.axhline(color='k', lw=0.8, ls='--')
        #self.text = ax.text(0.72, 0.9, '', transform=ax.transAxes)

        # Vertical lines in all axes
        self.vertical_lines = []
        for other_ax in all_axes:
            line = other_ax.axvline(color='k', lw=0.8, ls='--')
            self.vertical_lines.append(line)

        self._creating_background = False
        ax.figure.canvas.mpl_connect('draw_event', self.on_draw)

    def on_draw(self, event):
        self.create_new_background()

    def set_horizontal_visible(self, visible):
        changed = self.horizontal_line.get_visible() != visible
        self.horizontal_line.set_visible(visible)
        #self.text.set_visible(visible)
        return changed

    def create_new_background(self):
        if self._creating_background:
            return
        self._creating_background = True
        self.set_horizontal_visible(False)
        for vline in self.vertical_lines:
            vline.set_visible(False)
        self.ax.figure.canvas.draw()
        self.background = self.ax.figure.canvas.copy_from_bbox(self.ax.figure.bbox)
        self.set_horizontal_visible(True)
        for vline in self.vertical_lines:
            vline.set_visible(True)
        self._creating_background = False

    def on_mouse_move(self, event):
        if self.background is None:
            self.create_new_background()

        x, y = event.xdata, event.ydata
        if not event.inaxes:
            if self.set_horizontal_visible(False):
                self.ax.figure.canvas.restore_region(self.background)
                for vline in self.vertical_lines:
                    vline.set_visible(False)
                self.ax.figure.canvas.blit(self.ax.figure.bbox)
            return

        # Update lines
        for vline in self.vertical_lines:
            vline.set_xdata([x])
            vline.set_visible(True)

        if event.inaxes == self.ax:
            self.set_horizontal_visible(True)
            self.horizontal_line.set_ydata([y])
            #self.text.set_text(f'x={x:1.2f}, y={y:1.2f}')
        else:
            self.set_horizontal_visible(False)

        # Redraw
        self.ax.figure.canvas.restore_region(self.background)
        for vline in self.vertical_lines:
            self.ax.draw_artist(vline)
        if self.horizontal_line.get_visible():
            self.ax.draw_artist(self.horizontal_line)
            #self.ax.draw_artist(self.text)
        self.ax.figure.canvas.blit(self.ax.figure.bbox)

# class BlittedCursor:
#     """
#     A cross-hair cursor using blitting for faster redraw.
#     """
#
#     def __init__(self, ax):
#         self.ax = ax
#         self.background = None
#         self.horizontal_line = ax.axhline(color='k', lw=0.8, ls='--')
#         self.vertical_line = ax.axvline(color='k', lw=0.8, ls='--')
#         # text location in axes coordinates
#         #self.text = ax.text(0.72, 0.9, '', transform=ax.transAxes)
#         self._creating_background = False
#         ax.figure.canvas.mpl_connect('draw_event', self.on_draw)
#
#     def on_draw(self, event):
#         self.create_new_background()
#
#     def set_cross_hair_visible(self, visible):
#         need_redraw = self.horizontal_line.get_visible() != visible
#         self.horizontal_line.set_visible(visible)
#         self.vertical_line.set_visible(visible)
#         #self.text.set_visible(visible)
#         return need_redraw
#
#     def create_new_background(self):
#         if self._creating_background:
#             # discard calls triggered from within this function
#             return
#         self._creating_background = True
#         self.set_cross_hair_visible(False)
#         self.ax.figure.canvas.draw()
#         self.background = self.ax.figure.canvas.copy_from_bbox(self.ax.bbox)
#         self.set_cross_hair_visible(True)
#         self._creating_background = False
#
#     def on_mouse_move(self, event):
#         if self.background is None:
#             self.create_new_background()
#
#         # Case: mouse outside any axes
#         if not event.inaxes or event.inaxes != self.ax:
#             need_redraw = self.set_cross_hair_visible(False)
#             if need_redraw:
#                 self.ax.figure.canvas.restore_region(self.background)
#                 self.ax.figure.canvas.blit(self.ax.bbox)
#             return
#
#         # Inside the active axes
#         self.set_cross_hair_visible(True)
#         x, y = event.xdata, event.ydata
#         self.horizontal_line.set_ydata([y])
#         self.vertical_line.set_xdata([x])
#         #self.text.set_text(f'x={x:1.2f}, y={y:1.2f}')
#
#         self.ax.figure.canvas.restore_region(self.background)
#         self.ax.draw_artist(self.horizontal_line)
#         self.ax.draw_artist(self.vertical_line)
#         #self.ax.draw_artist(self.text)
#         self.ax.figure.canvas.blit(self.ax.bbox)
