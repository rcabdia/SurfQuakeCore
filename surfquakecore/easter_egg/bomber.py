"""BOMBER (1984) — an Amstrad CPC-style Easter egg for PyQt5 applications.

A recreation of Dave Town's AMSOFT / Locomotive BASIC game, featuring a
CPC-inspired palette, pixel font, the original SYMBOL 241/242 aircraft,
block buildings and a deliberately gentle starting speed.

Requires PyQt5; uses QtMultimedia for sound when available, with an
optional fallback to a system WAV player. No extra Python packages needed.

From the CLI (hidden command):
    surfquake play [--level 0-5] [--speed 0-100]

From an ISP window:
    from surfquakecore.easter_egg.bomber import BombarderoDialog
    self._bombardero = BombarderoDialog(self)
    self._bombardero.show()

Standalone: python -m surfquakecore.easter_egg.bomber

Controls:
    ENTER       start / play again
    SPACE       drop a bomb (only one in flight at a time)
    P           pause / resume
    M           mute / unmute
    R           restart
    ESC         close

The speed parameter follows the BASIC convention: 0 = fastest, 100 = slowest.
Default speed=30 gives 210 ms per aircraft step at stage 1. Bombs use a
separate, faster timer, so several may be dropped during one aircraft pass.
"""
from __future__ import annotations

import math
import random
import shutil
import struct
import subprocess
import tempfile
import wave
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple

from PyQt5.QtCore import QPoint, QRect, Qt, QTimer, QUrl
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import QApplication, QDialog


def _best_score_path() -> Path:
    """Store the high score next to the user's surfquake.ini."""
    try:
        from surfquakecore.config.global_config import user_config_path
        return user_config_path().parent / "bomber_best.txt"
    except Exception:
        return Path.home() / ".surfquake_bomber_best.txt"


def _load_best() -> int:
    try:
        return max(0, int(_best_score_path().read_text().strip()))
    except (OSError, ValueError):
        return 0


def _save_best(score: int) -> None:
    try:
        path = _best_score_path()
        if score > _load_best():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{score}\n")
    except OSError:
        pass  # A read-only home must never break the game.


@dataclass
class Bomb:
    col: int
    row: int


class BombarderoGame:
    """Game mechanics independent of Qt widgets."""

    def __init__(self, level: int = 3, speed: int = 30) -> None:
        # Level 0 = hardest; level 5 = easiest (as in the BASIC original).
        self.initial_level = max(0, min(5, int(level)))
        self.initial_speed = max(0, min(100, int(speed)))
        self.best_score = _load_best()
        self.restart()

    def restart(self) -> None:
        self.score = 0
        self.difficulty = self.initial_level + 10
        self.speed = self.initial_speed
        self.stage = 1
        self.status = "ready"
        self.new_city()

    def new_city(self) -> None:
        self.plane_x, self.plane_y = 2, 2
        self.bomb: Optional[Bomb] = None
        self.explosion: Optional[Tuple[int, int]] = None
        self.explosion_ticks = 0
        self.buildings: Dict[int, Set[int]] = {}
        for col in range(5, 16):
            # Each building grows upward from ground level (row 21).
            top = self.difficulty + random.randrange(8)
            self.buildings[col] = set(range(top, 22))

    def _update_best(self) -> None:
        if self.score > self.best_score:
            self.best_score = self.score
            _save_best(self.score)

    def start(self) -> None:
        if self.status in ("ready", "paused"):
            self.status = "running"

    def drop_bomb(self) -> Optional[str]:
        """Launch a bomb; return None if no bomb was launched.

        Returns "drop", or "explosion" when the bomb spawns directly on a
        rooftop (otherwise that first cell would be skipped and the bomb
        would carve the floor *below* it, leaving a floating storey).
        """
        if self.status == "running" and self.bomb is None and self.plane_y < 20:
            self.bomb = Bomb(self.plane_x, self.plane_y + 1)
            return self._resolve_bomb() or "drop"
        return None

    def tick(self) -> None:
        """Advance the plane by one horizontal step (not the falling bomb)."""
        if self.status != "running":
            return

        # The plane moves one cell right each step, dropping one row when
        # it wraps to the left, just as in the original game.
        self.plane_x += 1
        if self.plane_x == 18:
            self.plane_x = 2
            self.plane_y += 1

        # The BASIC game treats x=15, y=21 as a successful landing.
        if self.plane_x == 15 and self.plane_y == 21:
            self.score += 100 - 2 * self.difficulty
            self._update_best()
            self.difficulty = max(10, self.difficulty - 1)
            self.stage += 1
            self.new_city()
            return

        if self.plane_y > 21 or any(
            self.plane_y in self.buildings.get(col, ())
            for col in (self.plane_x, self.plane_x + 1)
        ):
            self.status = "crashed"
            self._update_best()
            self.bomb = None

    def advance_bomb(self) -> Optional[str]:
        """Move a bomb; report 'explosion' or 'ground' to the sound system.

        There is still *only one active bomb* at a time, matching the
        instruction in the original BASIC game. The next bomb may be
        dropped as soon as the first hits a building or the ground.
        """
        if self.status != "running":
            return None

        if self.explosion_ticks:
            self.explosion_ticks -= 1
            if self.explosion_ticks == 0:
                self.explosion = None

        if self.bomb is None:
            return None

        self.bomb.row += 1
        return self._resolve_bomb()

    def _resolve_bomb(self) -> Optional[str]:
        """Check the bomb's current cell for ground or building impact."""
        col, row = self.bomb.col, self.bomb.row
        if row > 21:
            self.bomb = None
            return "ground"
        elif row in self.buildings.get(col, ()):
            # Remove 1–3 floors, awarding 5 points per removed floor.
            for r in range(row, min(22, row + random.randint(1, 3))):
                if r in self.buildings[col]:
                    self.buildings[col].remove(r)
                    self.score += 5
            self.explosion = (col, row)
            self.explosion_ticks = 3
            self.bomb = None
            return "explosion"
        return None


# Approximation of the 27 hardware colors of the Amstrad CPC.
# The BASIC uses INK 0,0 : INK 1,18 : INK 2,6, and so on.
CPC_INKS = (
    "#000000", "#000080", "#0000FF", "#800000", "#800080",
    "#8000FF", "#FF0000", "#FF0080", "#FF00FF", "#008000",
    "#008080", "#0080FF", "#808000", "#808080", "#8080FF",
    "#FF8000", "#FF8080", "#FF80FF", "#00FF00", "#00FF80",
    "#00FFFF", "#80FF00", "#80FF80", "#80FFFF", "#FFFF00",
    "#FFFF80", "#FFFFFF",
)

# Sprite bytes from the original BASIC's SYMBOL 241 and SYMBOL 242
# (8 pixel rows per character, 2 characters).
PLANE_LEFT  = (0x40, 0x60, 0x70, 0x7F, 0x7F, 0xEF, 0x07, 0x00)
PLANE_RIGHT = (0x00, 0x32, 0x7A, 0xFE, 0xFA, 0xF2, 0xE0, 0x00)

# Small 5x7 bitmap font: draw pixels manually to avoid font smoothing.
FONT_5X7 = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01111", "10000", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "J": ("00111", "00010", "00010", "00010", "10010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01111", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "11110"),
    " ": ("00000",) * 7,
    ":": ("00000", "00100", "00100", "00000", "00100", "00100", "00000"),
    ".": ("00000", "00000", "00000", "00000", "00000", "00100", "00100"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    "/": ("00001", "00001", "00010", "00100", "01000", "10000", "10000"),
    "!": ("00100", "00100", "00100", "00100", "00100", "00000", "00100"),
    "+": ("00000", "00100", "00100", "11111", "00100", "00100", "00000"),
    "=": ("00000", "11111", "00000", "11111", "00000", "00000", "00000"),
    ">": ("10000", "01000", "00100", "00010", "00100", "01000", "10000"),
    "?": ("01110", "10001", "00001", "00010", "00100", "00000", "00100"),
}


class RetroAudio:
    """Tiny synthesised 8-bit-style WAV effects without third-party audio libs.

    Preferred backend: PyQt5.QtMultimedia.QSoundEffect (non-blocking).
    Fallback: an already installed system WAV player (pw-play/paplay/aplay).
    If neither is usable, the game continues silently.
    """

    SAMPLE_RATE = 22050
    _VOLUMES = {
        "drop": 0.28,
        "explosion": 0.48,
        "ground": 0.22,
        "crash": 0.54,
        "land": 0.32,
    }

    def __init__(self, parent) -> None:
        self.enabled = True
        self._dir = tempfile.TemporaryDirectory(prefix="isp_bomber_sound_")
        self._effects = {}
        self._players = {}
        self._backend = None
        self._external = None

        for kind in self._VOLUMES:
            path = Path(self._dir.name) / (kind + ".wav")
            self._write_wave(path, kind)
            self._players[kind] = str(path)

        # Attempt Qt audio first; this does not require another pip library,
        # but some Linux distributions package QtMultimedia separately.
        try:
            from PyQt5.QtMultimedia import QSoundEffect
            for kind, path in self._players.items():
                effect = QSoundEffect(parent)
                effect.setSource(QUrl.fromLocalFile(path))
                effect.setLoopCount(1)
                effect.setVolume(self._VOLUMES[kind])
                self._effects[kind] = effect
            self._backend = "qt"
        except (ImportError, RuntimeError, OSError):
            self._effects.clear()

        # Also detect an external player as a fallback for RHEL/Ubuntu.
        for exe, extra in (
            ("pw-play", ()),
            ("paplay", ()),
            ("aplay", ("-q",)),
            ("afplay", ()),
            ("ffplay", ("-nodisp", "-autoexit", "-loglevel", "error")),
        ):
            resolved = shutil.which(exe)
            if resolved:
                self._external = (resolved, extra)
                break

    @staticmethod
    def _write_wave(path: Path, kind: str) -> None:
        """Generate small PCM effects inspired by a CPC/AY square-wave chip."""
        duration = {
            "drop": 0.17,
            "explosion": 0.28,
            "ground": 0.10,
            "crash": 0.65,
            "land": 0.36,
        }[kind]
        rate = RetroAudio.SAMPLE_RATE
        count = round(duration * rate)
        rng = random.Random({
            "drop": 10, "explosion": 20, "ground": 30,
            "crash": 40, "land": 50,
        }[kind])
        phase = 0.0
        samples = bytearray()
        for i in range(count):
            t = i / rate
            u = i / max(1, count - 1)

            if kind == "drop":
                # Short, falling square-wave whistle.
                freq = 1100 - 650 * u
                amp = 0.35 * (1 - u) ** 1.3
                phase += freq / rate
                value = amp * (1 if phase % 1 < 0.5 else -1)
            elif kind == "explosion":
                # Crunchy pseudo-noise over a descending bass oscillator.
                freq = 160 - 95 * u
                phase += freq / rate
                pulse = 1 if phase % 1 < 0.5 else -1
                grain = rng.uniform(-1, 1)
                amp = 0.68 * (1 - u) ** 2.1
                value = amp * (0.72 * grain + 0.28 * pulse)
            elif kind == "ground":
                phase += (180 - 110 * u) / rate
                amp = 0.25 * (1 - u) ** 2
                value = amp * (1 if phase % 1 < 0.5 else -1)
            elif kind == "crash":
                freq = 420 * (1 - u) + 65 + 45 * math.sin(t * 45)
                phase += freq / rate
                square = 1 if phase % 1 < 0.5 else -1
                amp = 0.67 * (1 - u) ** 1.5
                value = amp * (0.55 * square + 0.45 * rng.uniform(-1, 1))
            else:  # land: a short upward arpeggio
                notes = (392, 523, 659, 784)
                freq = notes[min(3, int(u * 4))]
                phase += freq / rate
                amp = 0.24 * (1 - u) ** 0.4
                value = amp * (1 if phase % 1 < 0.5 else -1)

            sample = int(max(-1, min(1, value)) * 32767)
            samples.extend(struct.pack("<h", sample))

        with wave.open(str(path), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(rate)
            out.writeframes(samples)

    def play(self, kind: str) -> None:
        if not self.enabled or kind not in self._players:
            return
        if self._backend == "qt":
            effect = self._effects.get(kind)
            if effect is not None:
                # A Qt audio backend may be missing even when the Python
                # QtMultimedia binding itself imports successfully.
                from PyQt5.QtMultimedia import QSoundEffect
                if effect.status() != QSoundEffect.Error:
                    effect.stop()
                    effect.play()
                    return
        if self._external is not None:
            exe, extra = self._external
            try:
                subprocess.Popen(
                    [exe, *extra, self._players[kind]],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except OSError:
                pass

    def close(self) -> None:
        if self._dir is None:
            return
        for effect in self._effects.values():
            effect.stop()
            effect.setSource(QUrl())
        self._effects.clear()
        self._dir.cleanup()
        self._dir = None


class BombarderoDialog(QDialog):
    """Self-contained, non-modal dialog; existing ISP integration is unchanged."""

    CW, CH = 32, 20
    MX, MY = 24, 38
    WIDTH = 20 * CW + 2 * MX
    HEIGHT = 25 * CH + 2 * MY

    def __init__(self, parent=None, level: int = 3, speed: int = 30) -> None:
        super().__init__(parent)
        self.game = BombarderoGame(level=level, speed=speed)
        self.setWindowTitle("BOMBER - AMSOFT 1984 | CPC 6128")
        self.setFixedSize(self.WIDTH, self.HEIGHT)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self._ink = [QColor(code) for code in CPC_INKS]
        self._ticks = 0
        self.audio = RetroAudio(self)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(self._interval())

        # Bombs descend independently of the much slower aircraft.
        # Roughly 55 ms per floor makes multiple drops possible per pass.
        self.bomb_timer = QTimer(self)
        self.bomb_timer.timeout.connect(self._advance_bomb)
        self.bomb_timer.start(55)

        # Free the temp WAV files when the dialog is really destroyed, not
        # on close: the GUI keeps one instance and may show() it again.
        self.destroyed.connect(lambda *_: self.audio.close())

    def showEvent(self, event) -> None:
        if not self.timer.isActive():
            self.timer.start(self._interval())
        if not self.bomb_timer.isActive():
            self.bomb_timer.start(55)
        super().showEvent(event)

    def closeEvent(self, event) -> None:
        self.timer.stop()
        self.bomb_timer.stop()
        if self.game.status == "running":
            self.game.status = "paused"
        super().closeEvent(event)

    def focusOutEvent(self, event) -> None:
        # Auto-pause when the user switches back to real work.
        if self.game.status == "running":
            self.game.status = "paused"
            self.update()
        super().focusOutEvent(event)

    def _interval(self) -> int:
        # Start at 210 ms per step at speed=30, with gradual acceleration.
        base = 150 + self.game.speed * 2
        return max(75, base - 11 * (self.game.stage - 1))

    def _tick(self) -> None:
        self._ticks += 1
        stage_before = self.game.stage
        status_before = self.game.status
        self.game.tick()
        if self.game.stage != stage_before:
            self.timer.setInterval(self._interval())
            self.audio.play("land")
        elif status_before == "running" and self.game.status == "crashed":
            self.audio.play("crash")
        self.update()

    def _advance_bomb(self) -> None:
        # Avoid repainting the entire window when no bomb is active.
        if self.game.status != "running":
            return
        if self.game.bomb is None and not self.game.explosion_ticks:
            return
        self._ticks += 1
        event = self.game.advance_bomb()
        if event:
            self.audio.play(event)
        self.update()

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()
        elif key == Qt.Key_R:
            self.game.restart()
            self.timer.setInterval(self._interval())
        elif key in (Qt.Key_Return, Qt.Key_Enter):
            if self.game.status == "crashed":
                self.game.restart()
                self.timer.setInterval(self._interval())
            self.game.start()
        elif key == Qt.Key_P:
            if self.game.status == "running":
                self.game.status = "paused"
            elif self.game.status == "paused":
                self.game.start()
        elif key == Qt.Key_M:
            self.audio.enabled = not self.audio.enabled
        elif key == Qt.Key_Space:
            if self.game.status == "ready":
                self.game.start()
            else:
                event_name = self.game.drop_bomb()
                if event_name:
                    self.audio.play(event_name)
        else:
            super().keyPressEvent(event)
        self.update()

    def _xy(self, col: int, row: int) -> Tuple[int, int]:
        return self.MX + (col - 1) * self.CW, self.MY + (row - 1) * self.CH

    def _pixel_text(self, p: QPainter, x: int, y: int, message: str,
                    ink: int = 18, scale: int = 2) -> None:
        p.setPen(Qt.NoPen)
        p.setBrush(self._ink[ink])
        left = x
        for char in message.upper():
            if char == "\n":
                x = left
                y += 9 * scale
                continue
            glyph = FONT_5X7.get(char, FONT_5X7["?"])
            for iy, line in enumerate(glyph):
                for ix, active in enumerate(line):
                    if active == "1":
                        p.fillRect(x + ix * scale, y + iy * scale,
                                   scale, scale, self._ink[ink])
            x += 6 * scale

    def _pixel_sprite(self, p: QPainter, x: int, y: int,
                      bitmap: Tuple[int, ...], ink: int,
                      pixel_w: int = 4, pixel_h: int = 2) -> None:
        for row, bits in enumerate(bitmap):
            for col in range(8):
                if bits & (0x80 >> col):
                    p.fillRect(x + col * pixel_w, y + row * pixel_h,
                               pixel_w, pixel_h, self._ink[ink])

    def _draw_border(self, p: QPainter) -> None:
        # CPC-style screen frame with a blue border and black playfield.
        p.fillRect(self.rect(), self._ink[1])
        p.fillRect(self.MX - 9, self.MY - 9, 20 * self.CW + 18,
                   25 * self.CH + 18, self._ink[2])
        p.fillRect(self.MX - 5, self.MY - 5, 20 * self.CW + 10,
                   25 * self.CH + 10, self._ink[0])
        self._pixel_text(p, self.MX, 13, "AMSTRAD CPC 6128", ink=24, scale=2)
        self._pixel_text(p, self.WIDTH - 208, 13, "AMSOFT 1984", ink=26, scale=2)
        self._pixel_text(p, self.MX + 12, self.HEIGHT - 27,
                         "SPACE=BOMB  P=PAUSE  M=SOUND  ESC=EXIT", ink=26, scale=2)

    def _draw_city(self, p: QPainter) -> None:
        building_inks = (4, 15, 2, 24, 10, 20, 12, 14, 16, 21, 26)
        for col, rows in self.game.buildings.items():
            if not rows:
                continue
            ink = building_inks[col - 5]
            for row in rows:
                x, y = self._xy(col, row)
                # Solid pixel blocks with no gradients or anti-aliasing.
                p.fillRect(x, y + 2, self.CW - 1, self.CH - 1, self._ink[ink])
                # Two dark windows per floor.
                p.fillRect(x + 6, y + 7, 5, 6, self._ink[0])
                p.fillRect(x + 21, y + 7, 5, 6, self._ink[0])
                # Horizontal line separating floors.
                p.fillRect(x + 2, y + self.CH - 2,
                           self.CW - 5, 2, self._ink[0])
        _, ground_y = self._xy(1, 22)
        p.fillRect(self.MX, ground_y, self.CW * 20, 3, self._ink[18])
        # Runway on the right, at the end of the final pass.
        runway_x, runway_y = self._xy(16, 21)
        p.fillRect(runway_x, runway_y + 17, self.CW * 4, 3, self._ink[24])

    def _draw_plane(self, p: QPainter) -> None:
        x, y = self._xy(self.game.plane_x, self.game.plane_y)
        # Exact reconstruction of both 8-byte BASIC SYMBOL sprites.
        self._pixel_sprite(p, x, y + 2, PLANE_LEFT, ink=18)
        self._pixel_sprite(p, x + self.CW, y + 2, PLANE_RIGHT, ink=18)

    def _draw_bomb(self, p: QPainter) -> None:
        if self.game.bomb is None:
            return
        bx, by = self._xy(self.game.bomb.col, self.game.bomb.row)
        # Pixelated bomb with a short trail.
        p.fillRect(bx + 14, by + 1, 4, 4, self._ink[26])
        p.fillRect(bx + 11, by + 5, 10, 8, self._ink[24])
        p.fillRect(bx + 14, by + 13, 4, 4, self._ink[6])

    def _draw_explosion(self, p: QPainter) -> None:
        if self.game.explosion is None:
            return
        x, y = self._xy(*self.game.explosion)
        center_x, center_y = x + 16, y + 10
        ink = (24, 6, 26)[self._ticks % 3]
        p.fillRect(center_x - 4, center_y - 16, 8, 32, self._ink[ink])
        p.fillRect(center_x - 16, center_y - 4, 32, 8, self._ink[ink])
        p.fillRect(center_x - 9, center_y - 9, 18, 18, self._ink[26])

    def _draw_hud(self, p: QPainter) -> None:
        x, y = self._xy(1, 1)
        self._pixel_text(p, x + 7, y + 3, "BOMBER", ink=18, scale=2)
        self._pixel_text(p, x + 360, y + 3,
                         f"LEVEL {self.game.stage:02d}", ink=24, scale=2)
        _, footer_y = self._xy(1, 25)
        self._pixel_text(p, self.MX + 7, footer_y + 2,
                         f"SCORE {self.game.score:05d}", ink=6, scale=2)
        self._pixel_text(p, self.MX + 370, footer_y + 2,
                         f"BEST {self.game.best_score:05d}", ink=24, scale=2)

    def _draw_intro(self, p: QPainter) -> None:
        """Full yellow CPC-style title screen, with dark blue pixel text."""
        p.fillRect(self.rect(), self._ink[24])  # CPC bright yellow
        p.setPen(QPen(self._ink[1], 4))        # CPC dark blue
        p.drawRect(QRect(14, 14, self.WIDTH - 28, self.HEIGHT - 28))
        p.drawLine(30, 108, self.WIDTH - 30, 108)
        p.drawLine(30, self.HEIGHT - 76, self.WIDTH - 30, self.HEIGHT - 76)

        def centered(message: str, y: int, scale: int = 2) -> None:
            x = (self.WIDTH - len(message) * 6 * scale) // 2
            self._pixel_text(p, x, y, message, ink=1, scale=scale)

        centered("AMSTRAD CPC 6128", 45)
        centered("AMSOFT 1984", 76)
        centered("BOMBER", 145, scale=5)
        centered("BY DAVE TOWN", 213)
        # Authentic sprite bytes, rendered in the same dark-blue ink.
        sprite_x = (self.WIDTH - 2 * self.CW) // 2
        self._pixel_sprite(p, sprite_x, 252, PLANE_LEFT, ink=1)
        self._pixel_sprite(p, sprite_x + self.CW, 252, PLANE_RIGHT, ink=1)
        centered("ONE BOMB IN FLIGHT AT A TIME", 311)
        centered("SPACE TO DROP A BOMB", 347)
        centered("ENTER TO START", 383)
        centered("P TO PAUSE  R TO RESTART", 419)
        centered("M TO MUTE  ESC TO EXIT", 455)
        centered("PYQT5 EDITION", 520)

    def _draw_overlay(self, p: QPainter) -> None:
        if self.game.status not in ("paused", "crashed"):
            return
        width, height = 510, 200
        left = (self.WIDTH - width) // 2
        top = self.MY + 128
        box = QRect(left, top, width, height)
        p.fillRect(box, self._ink[0])
        p.setPen(QPen(self._ink[24], 3))
        p.drawRect(box)
        title = "PAUSED" if self.game.status == "paused" else "GAME OVER"
        self._pixel_text(p, left + (width - len(title) * 18) // 2,
                         top + 22, title, ink=18, scale=3)
        if self.game.status == "paused":
            lines = (
                "GAME PAUSED",
                "P TO RESUME",
                "R TO RESTART - ESC TO EXIT",
            )
        else:
            lines = (
                f"SCORE {self.game.score:05d}  BEST {self.game.best_score:05d}",
                "ENTER TO PLAY AGAIN",
                "R TO RESTART - ESC TO EXIT",
            )
        for i, line in enumerate(lines):
            self._pixel_text(p, left + (width - len(line) * 12) // 2,
                             top + 83 + i * 28, line,
                             ink=26 if i != 2 else 24, scale=2)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        p.setRenderHint(QPainter.TextAntialiasing, False)
        if self.game.status == "ready":
            self._draw_intro(p)
            p.end()
            return
        self._draw_border(p)
        self._draw_city(p)
        if self.game.status != "crashed":
            self._draw_plane(p)
        self._draw_bomb(p)
        self._draw_explosion(p)
        self._draw_hud(p)
        self._draw_overlay(p)
        p.end()


def run(level: int = 3, speed: int = 30) -> int:
    """Open BOMBER as a standalone window (used by ``surfquake play``)."""
    import os
    import sys

    if (sys.platform.startswith("linux")
            and not os.environ.get("DISPLAY")
            and not os.environ.get("WAYLAND_DISPLAY")
            and os.environ.get("QT_QPA_PLATFORM", "") != "offscreen"):
        print("No display found: BOMBER needs a graphical session. "
              "(Try ssh -X, or play it on your desktop.)")
        return 1

    app = QApplication.instance() or QApplication(sys.argv[:1])
    dialog = BombarderoDialog(level=level, speed=speed)
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
    return app.exec_()


if __name__ == "__main__":
    import sys
    sys.exit(run())
