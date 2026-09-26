"""Legend drawing shared by the world map and the region view.

A legend is ``(kind, entries)``: ``kind`` is ``"gradient"`` or ``"categorical"``
and ``entries`` a list of ``(label, rgba)`` with rgba floats in ``[0, 1]``.  For a
gradient, entries run from the lowest value (bottom of the bar) to the highest.
"""
from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFontMetrics, QLinearGradient, QPainter, QPen

_PAD = 6
_FONT_PT = 9.0

# Gradient bar.
_BAR_W = 18
_LABEL_GAP = 6

# Categorical rows.
_SWATCH = 13
_SWATCH_GAP = 5


def _set_font(painter: QPainter) -> QFontMetrics:
    font = painter.font()
    font.setPointSizeF(_FONT_PT)
    painter.setFont(font)
    return painter.fontMetrics()


def _gradient_bar_h(avail_h: float) -> int:
    return min(220, max(120, int(avail_h * 0.38)))


def _categorical_row_h(fm: QFontMetrics) -> int:
    return max(_SWATCH, fm.height()) + 3


def legend_size(painter: QPainter, kind: str, entries: list, avail_h: float) -> tuple[int, int]:
    """``(width, height)`` of the legend box, background padding included."""
    painter.save()
    fm = _set_font(painter)
    max_lw = max(fm.horizontalAdvance(lbl) for lbl, _ in entries)
    painter.restore()
    if kind == "gradient":
        return _PAD + _BAR_W + _LABEL_GAP + max_lw + _PAD, _PAD + _gradient_bar_h(avail_h) + _PAD
    return (_PAD + _SWATCH + _SWATCH_GAP + max_lw + _PAD,
            _PAD + len(entries) * _categorical_row_h(fm) + _PAD)


def draw_legend(
    painter: QPainter, kind: str, entries: list, x: float, y_bottom: float, avail_h: float
) -> None:
    """Draw a legend whose box has its bottom-left corner at ``(x, y_bottom)``.

    ``avail_h`` is the height the legend sits against; a gradient bar scales with it.
    """
    if len(entries) < 2:
        return
    painter.save()
    fm = _set_font(painter)
    w, h = legend_size(painter, kind, entries, avail_h)
    x0, y0 = int(x), int(y_bottom) - h
    painter.fillRect(x0, y0, w, h, QColor(20, 20, 20, 160))
    if kind == "gradient":
        _draw_gradient(painter, fm, entries, x0 + _PAD, y0 + _PAD, h - 2 * _PAD)
    else:
        _draw_categorical(painter, fm, entries, x0 + _PAD, y0 + _PAD)
    painter.restore()


def _draw_gradient(
    painter: QPainter, fm: QFontMetrics, entries: list, x0: int, y_top: int, bar_h: int
) -> None:
    y_bot = y_top + bar_h
    gradient = QLinearGradient(x0, y_top, x0, y_bot)
    n = len(entries)
    for i, (_, rgba) in enumerate(entries):
        # entries[0] = lowest = bottom (pos 1.0); entries[-1] = highest = top (pos 0.0)
        pos = 1.0 - i / (n - 1)
        gradient.setColorAt(pos, QColor(int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255), 255))
    painter.fillRect(QRectF(x0, y_top, _BAR_W, bar_h), QBrush(gradient))
    painter.setPen(QPen(QColor(120, 120, 120, 180), 0.5))
    painter.drawRect(QRectF(x0, y_top, _BAR_W, bar_h))

    painter.setPen(QColor(240, 240, 240))
    for i, (label, _) in enumerate(entries):
        cy = y_bot - i / (n - 1) * bar_h
        ty = int(cy + (fm.ascent() - fm.descent()) // 2)
        painter.drawText(int(x0 + _BAR_W + _LABEL_GAP), ty, label)


def _draw_categorical(painter: QPainter, fm: QFontMetrics, entries: list, x0: int, y0: int) -> None:
    row_h = _categorical_row_h(fm)
    for i, (label, rgba) in enumerate(entries):
        row_y = y0 + i * row_h
        sw_y = row_y + (row_h - _SWATCH) // 2
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))))
        painter.drawRect(x0, sw_y, _SWATCH, _SWATCH)
        painter.setPen(QColor(240, 240, 240))
        cy = row_y + row_h / 2
        ty = int(cy + (fm.ascent() - fm.descent()) // 2)
        painter.drawText(x0 + _SWATCH + _SWATCH_GAP, ty, label)
