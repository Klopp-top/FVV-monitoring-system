import cv2
from ultralytics import YOLO
import numpy as np
import os
import sys
import math
import json
import time
import tkinter as tk
from tkinter import messagebox
from datetime import datetime


def get_base_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(get_base_dir(), relative_path)


BASE_DIR = get_base_dir()

HEADER_BANNER_PATH = resource_path("header_banner.png")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
SAVE_INTERVAL_SEC = 60
FLASH_INTERVAL_SEC = 0.4


SMOKE_MINOR_THRESHOLD = 0.08
SMOKE_HIGH_THRESHOLD = 0.18
FIRE_MINOR_PERCENT = 1.5
FIRE_HIGH_PERCENT = 6.0
SMOKE_MAX_Y_RATIO = 0.7
FRAME_AREA_M2 = 500
FIRE_CONF_THRESHOLD = 0.45
SMOKE_CONF_THRESHOLD = 0.55


model_person = YOLO(resource_path('yolov8n.pt'))

fire_model_path = resource_path('best.pt')
if os.path.exists(fire_model_path):
    model_fire = YOLO(fire_model_path)
    use_fire_model = True
    print("Модель огня загружена")
else:
    use_fire_model = False

drone_model_path = resource_path('drone.pt')
if os.path.exists(drone_model_path):
    model_drone = YOLO(drone_model_path)
    use_drone_model = True
    print("Модель дрона загружена")
else:
    use_drone_model = False

banner_img = None
if os.path.exists(HEADER_BANNER_PATH):
    banner_img = cv2.imread(HEADER_BANNER_PATH, cv2.IMREAD_UNCHANGED)


PANEL_W = 760

BG_TOP    = (70, 40, 30)
BG_BOTTOM = (175, 130, 115)

CARD_BORDER  = (255, 255, 255)
CARD_ALPHA   = 0.10
SHADOW_ALPHA = 0.28

TXT_PRIMARY   = (255, 255, 255)
TXT_SECONDARY = (215, 210, 225)
TXT_MUTED     = (170, 165, 185)

COL_OK     = (150, 235, 165)
COL_WATCH  = (110, 200, 250)
COL_DANGER = (95, 95, 250)

FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_LIGHT = cv2.FONT_HERSHEY_SIMPLEX


def draw_gradient_bg(w, h, top_color, bottom_color):
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        t = (y / max(h - 1, 1)) ** 0.85
        color = tuple(int(top_color[i] * (1 - t) + bottom_color[i] * t) for i in range(3))
        bg[y, :] = color
    return bg


def draw_rounded_rect(img, top_left, bottom_right, color, thickness=-1, radius=14):
    x1, y1 = top_left
    x2, y2 = bottom_right
    if thickness < 0:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, -1, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, -1, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, -1, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, -1, cv2.LINE_AA)
    else:
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)
    return img


def draw_glass_card(panel, top_left, bottom_right, radius=22, alpha=CARD_ALPHA,
                     border_color=CARD_BORDER, shadow=True):
    x1, y1 = top_left
    x2, y2 = bottom_right

    if shadow:
        sh = panel.copy()
        off = 6
        draw_rounded_rect(sh, (x1, y1 + off), (x2, y2 + off), (10, 8, 15), -1, radius=radius)
        cv2.addWeighted(sh, SHADOW_ALPHA, panel, 1 - SHADOW_ALPHA, 0, dst=panel)

    overlay = panel.copy()
    draw_rounded_rect(overlay, top_left, bottom_right, (255, 255, 255), -1, radius=radius)
    cv2.addWeighted(overlay, alpha, panel, 1 - alpha, 0, dst=panel)
    draw_rounded_rect(panel, top_left, bottom_right, border_color, 1, radius=radius)


def text_centered(panel, text, cx, y, font, scale, color, thickness):
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.putText(panel, text, (int(cx - tw / 2), y), font, scale, color, thickness, cv2.LINE_AA)
    return tw, th


def overlay_image(panel, img, x, y, target_w=None, target_h=None):
    if img is None:
        return panel, 0
    h, w = img.shape[:2]
    if target_w and not target_h:
        target_h = int(h * (target_w / w))
    if target_h and not target_w:
        target_w = int(w * (target_h / h))
    resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    if resized.shape[2] == 4:
        alpha = resized[:, :, 3] / 255.0
        for c in range(3):
            panel[y:y + target_h, x:x + target_w, c] = (
                alpha * resized[:, :, c] + (1 - alpha) * panel[y:y + target_h, x:x + target_w, c]
            )
    else:
        panel[y:y + target_h, x:x + target_w] = resized[:, :, :3]

    return panel, target_h


def icon_flame(panel, cx, cy, size, color):
    outer = np.array([
        [cx, cy - size], [cx - int(size * 0.62), cy - int(size * 0.15)],
        [cx - int(size * 0.55), cy + int(size * 0.55)], [cx - int(size * 0.18), cy + size],
        [cx + int(size * 0.05), cy + int(size * 0.85)], [cx + int(size * 0.3), cy + size],
        [cx + int(size * 0.6), cy + int(size * 0.5)], [cx + int(size * 0.35), cy - int(size * 0.05)],
        [cx + int(size * 0.15), cy - int(size * 0.35)],
    ], np.int32)
    cv2.fillPoly(panel, [outer], color, cv2.LINE_AA)
    cv2.circle(panel, (cx, cy + int(size * 0.35)), max(2, int(size * 0.28)), (255, 255, 255), -1, cv2.LINE_AA)


def icon_smoke(panel, cx, cy, r, color):
    offsets = [(-0.15, 0.55, 0.62), (0.28, -0.15, 0.5), (-0.22, -0.85, 0.4)]
    for dx, dy, rs in offsets:
        cv2.circle(panel, (cx + int(r * dx), cy + int(r * dy)), max(2, int(r * rs)), color, 2, cv2.LINE_AA)


def icon_sun(panel, cx, cy, r, color):
    cv2.circle(panel, (cx, cy), r, color, -1, cv2.LINE_AA)
    for angle in range(0, 360, 45):
        rad = math.radians(angle)
        x1, y1 = int(cx + (r + 4) * math.cos(rad)), int(cy + (r + 4) * math.sin(rad))
        x2, y2 = int(cx + (r + 9) * math.cos(rad)), int(cy + (r + 9) * math.sin(rad))
        cv2.line(panel, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)


def icon_cloud(panel, cx, cy, r, color):
    cv2.circle(panel, (cx - r, cy + 2), int(r * 0.7), color, -1, cv2.LINE_AA)
    cv2.circle(panel, (cx + r, cy + 2), int(r * 0.7), color, -1, cv2.LINE_AA)
    cv2.circle(panel, (cx, cy - 3), r, color, -1, cv2.LINE_AA)
    cv2.rectangle(panel, (cx - r, cy), (cx + r, cy + int(r * 0.7)), color, -1)


def icon_drop(panel, cx, cy, r, color):
    cv2.circle(panel, (cx, cy + r // 3), r, color, -1, cv2.LINE_AA)
    pts = np.array([[cx, cy - r], [cx - r, cy + r // 3], [cx + r, cy + r // 3]], np.int32)
    cv2.fillPoly(panel, [pts], color, cv2.LINE_AA)


def icon_warning(panel, cx, cy, size, color):
    pts = np.array([[cx, cy - size], [cx - size, cy + size], [cx + size, cy + size]], np.int32)
    cv2.fillPoly(panel, [pts], color, cv2.LINE_AA)
    cv2.polylines(panel, [pts], True, color, 1, cv2.LINE_AA)
    cv2.line(panel, (cx, cy - size // 5), (cx, cy + size // 3), (25, 20, 30), 2, cv2.LINE_AA)
    cv2.circle(panel, (cx, cy + size - size // 6), 1, (25, 20, 30), 2, cv2.LINE_AA)


def icon_person(panel, cx, cy, size, color):
    cv2.circle(panel, (cx, cy - size), max(2, size // 2), color, -1, cv2.LINE_AA)
    pts = np.array([
        [cx - size, cy + size], [cx + size, cy + size],
        [cx + int(size * 0.6), cy - size // 4], [cx - int(size * 0.6), cy - size // 4]
    ], np.int32)
    cv2.fillPoly(panel, [pts], color, cv2.LINE_AA)


def icon_earthquake(panel, cx, cy, size, color):
    pts = np.array([
        [cx - size, cy], [cx - int(size * 0.5), cy - size],
        [cx, cy + size], [cx + int(size * 0.5), cy - size], [cx + size, cy],
    ], np.int32)
    cv2.polylines(panel, [pts], False, color, 2, cv2.LINE_AA)


def icon_radiation(panel, cx, cy, size, color):
    cv2.circle(panel, (cx, cy), max(2, size // 4), color, -1, cv2.LINE_AA)
    for angle in range(0, 360, 120):
        a1 = math.radians(angle - 25)
        a2 = math.radians(angle + 25)
        p2 = (int(cx + size * math.cos(a1)), int(cy + size * math.sin(a1)))
        p3 = (int(cx + size * math.cos(a2)), int(cy + size * math.sin(a2)))
        pts = np.array([(cx, cy), p2, p3], np.int32)
        cv2.fillPoly(panel, [pts], color, cv2.LINE_AA)


def icon_drone(panel, cx, cy, size, color):
    cv2.line(panel, (cx - size, cy - size), (cx + size, cy + size), color, 2, cv2.LINE_AA)
    cv2.line(panel, (cx - size, cy + size), (cx + size, cy - size), color, 2, cv2.LINE_AA)
    r = max(2, size // 3)
    for dx, dy in [(-size, -size), (size, -size), (-size, size), (size, size)]:
        cv2.circle(panel, (cx + dx, cy + dy), r, color, -1, cv2.LINE_AA)
    cv2.circle(panel, (cx, cy), max(2, size // 4), color, -1, cv2.LINE_AA)


ALERT_RED = (55, 55, 255)


def draw_status_row_card(panel, top_left, bottom_right, title, status_text, color, icon_fn,
                          alert=False, extra_text=None):
    x1, y1 = top_left
    x2, y2 = bottom_right

    if alert:
        glow = panel.copy()
        draw_rounded_rect(glow, top_left, bottom_right, ALERT_RED, -1, radius=16)
        cv2.addWeighted(glow, 0.30, panel, 0.70, 0, dst=panel)
        draw_glass_card(panel, top_left, bottom_right, radius=16, alpha=0.05,
                         border_color=ALERT_RED, shadow=False)
        draw_rounded_rect(panel, top_left, bottom_right, ALERT_RED, 3, radius=16)
        text_color = ALERT_RED
        dot_color = ALERT_RED
        dot_r = 15
    else:
        draw_glass_card(panel, top_left, bottom_right, radius=16, border_color=color)
        text_color = color
        dot_color = color
        dot_r = 12

    icon_fn(panel, x1 + 30, y1 + 36, 16, text_color)
    cv2.putText(panel, title, (x1 + 62, y1 + 44), FONT, 0.68, TXT_SECONDARY, 1, cv2.LINE_AA)
    cv2.putText(panel, status_text, (x1 + 30, y2 - 22), FONT, 0.72, text_color, 2, cv2.LINE_AA)

    if extra_text:
        (tw, th), _ = cv2.getTextSize(extra_text, FONT_LIGHT, 0.42, 1)
        cv2.putText(panel, extra_text, (x2 - tw - 30, y2 - 24), FONT_LIGHT, 0.42, TXT_MUTED, 1, cv2.LINE_AA)

    dot_cx = x2 - 44
    dot_cy = y1 + 42
    cv2.circle(panel, (dot_cx, dot_cy), dot_r, dot_color, -1, cv2.LINE_AA)
    cv2.circle(panel, (dot_cx, dot_cy), dot_r, (255, 255, 255), 1, cv2.LINE_AA)


def draw_section_label(panel, x, y, w, text):
    cv2.putText(panel, text, (x, y), FONT_LIGHT, 0.38, TXT_MUTED, 1, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(text, FONT_LIGHT, 0.38, 1)
    line_x = x + tw + 12
    cv2.line(panel, (line_x, y - 4), (x + w, y - 4), (255, 255, 255), 1, cv2.LINE_AA)


def draw_mini_status_card(panel, top_left, bottom_right, title, status_text, color, icon_fn):
    x1, y1 = top_left
    x2, y2 = bottom_right
    draw_glass_card(panel, top_left, bottom_right, radius=14, alpha=0.07, border_color=color, shadow=False)

    icon_fn(panel, x1 + 22, y1 + 24, 9, color)
    cv2.putText(panel, title, (x1 + 40, y1 + 28), FONT_LIGHT, 0.32, TXT_SECONDARY, 1, cv2.LINE_AA)
    cv2.putText(panel, status_text, (x1 + 16, y2 - 14), FONT, 0.42, color, 1, cv2.LINE_AA)

    cv2.circle(panel, (x2 - 16, y1 + 18), 5, color, -1, cv2.LINE_AA)


def draw_count_card(panel, top_left, bottom_right, title, value, color, icon_fn):
    x1, y1 = top_left
    x2, y2 = bottom_right
    draw_glass_card(panel, top_left, bottom_right, radius=16, border_color=color)

    icon_fn(panel, x1 + 30, y1 + 36, 16, color)
    cv2.putText(panel, title, (x1 + 62, y1 + 44), FONT, 0.68, TXT_SECONDARY, 1, cv2.LINE_AA)

    (tw, th), _ = cv2.getTextSize(value, FONT, 1.35, 3)
    cv2.putText(panel, value, (x2 - tw - 30, (y1 + y2) // 2 + th // 2 + 6), FONT, 1.35, color, 3, cv2.LINE_AA)


def draw_peak_card(panel, top_left, bottom_right, title, value_text, peak_time, color):
    x1, y1 = top_left
    x2, y2 = bottom_right
    draw_glass_card(panel, top_left, bottom_right, radius=16, border_color=color)

    cv2.putText(panel, title, (x1 + 30, y1 + 44), FONT, 0.68, TXT_SECONDARY, 1, cv2.LINE_AA)
    cv2.putText(panel, value_text, (x1 + 30, y2 - 22), FONT, 0.86, color, 2, cv2.LINE_AA)

    time_text = f"Vaqti: {peak_time}"
    (tw, th), _ = cv2.getTextSize(time_text, FONT_LIGHT, 0.46, 1)
    cv2.putText(panel, time_text, (x2 - tw - 30, y2 - 24), FONT_LIGHT, 0.46, TXT_MUTED, 1, cv2.LINE_AA)


def draw_weather_card(panel, top_left, bottom_right, weather_label, humidity_label, icon_fn):
    x1, y1 = top_left
    x2, y2 = bottom_right
    color = (140, 190, 235)
    draw_glass_card(panel, top_left, bottom_right, radius=16, border_color=color)

    icon_fn(panel, x1 + 30, y1 + 36, 14, color)
    cv2.putText(panel, "OB-HAVO", (x1 + 62, y1 + 44), FONT, 0.68, TXT_SECONDARY, 1, cv2.LINE_AA)
    cv2.putText(panel, weather_label, (x1 + 30, y2 - 22), FONT, 0.78, color, 2, cv2.LINE_AA)

    hum_text = f"Namlik: {humidity_label}"
    (tw, th), _ = cv2.getTextSize(hum_text, FONT_LIGHT, 0.46, 1)
    cv2.putText(panel, hum_text, (x2 - tw - 30, y2 - 24), FONT_LIGHT, 0.46, TXT_MUTED, 1, cv2.LINE_AA)


def create_info_window(fire_severity, smoke_severity, person_count, peak_count, peak_time,
                        fire_area_percent=0.0, fire_peak_area=0.0, fire_peak_time="-",
                        drone_detected=False,
                        weather_label="-", humidity_label="-", flash_on=False):
    W = PANEL_W

    margin = 26
    gap = 12
    card_h_big = 104
    card_h_small = 92
    card_h_mini = 68
    status_h = 66

    CANVAS_H = 1600

    banner_h = 0
    if banner_img is not None:
        bh, bw = banner_img.shape[:2]
        banner_h = int(bh * (W / bw))
        top = banner_h + 20
    else:
        top = 96

    info = draw_gradient_bg(W, CANVAS_H, BG_TOP, BG_BOTTOM)

    if banner_img is not None:
        info, banner_h = overlay_image(info, banner_img, 0, 0, target_w=W)
    else:
        cv2.putText(info, "FAVQULODDA VAZIYATLAR VAZIRLIGI", (24, 38), FONT, 0.46, TXT_PRIMARY, 1, cv2.LINE_AA)
        cv2.putText(info, "Monitoring tizimi", (24, 60), FONT_LIGHT, 0.35, TXT_SECONDARY, 1, cv2.LINE_AA)

    SEV_COLOR = {"none": COL_OK, "minor": COL_WATCH, "high": COL_DANGER}
    FIRE_LABEL = {"none": "Aniqlanmadi", "minor": "Kam xavfli", "high": "ANIQLANDI!"}
    SMOKE_LABEL = {"none": "Aniqlanmadi", "minor": "Kam xavfli", "high": "ANIQLANDI!"}

    fire_alert = flash_on and fire_severity != "none"
    smoke_alert = flash_on and smoke_severity != "none"
    drone_alert = flash_on and drone_detected
    drone_color = COL_DANGER if drone_detected else COL_OK
    drone_label = "ANIQLANDI!" if drone_detected else "Aniqlanmadi"

    fire_area_m2 = fire_area_percent / 100.0 * FRAME_AREA_M2
    fire_area_text = f"~{fire_area_m2:.0f} m2" if fire_severity != "none" else None

    y = top
    draw_section_label(info, margin, y, W - margin * 2, "ASOSIY XAVFLAR")
    y += 22

    draw_status_row_card(info, (margin, y), (W - margin, y + card_h_big), "OLOV",
                          FIRE_LABEL[fire_severity], SEV_COLOR[fire_severity], icon_flame,
                          alert=fire_alert, extra_text=fire_area_text)
    y += card_h_big + gap

    draw_status_row_card(info, (margin, y), (W - margin, y + card_h_big), "TUTUN",
                          SMOKE_LABEL[smoke_severity], SEV_COLOR[smoke_severity], icon_smoke, alert=smoke_alert)
    y += card_h_big + gap

    draw_status_row_card(info, (margin, y), (W - margin, y + card_h_big), "DRON",
                          drone_label, drone_color, icon_drone, alert=drone_alert)
    y += card_h_big + 26

    draw_section_label(info, margin, y, W - margin * 2, "QO'SHIMCHA NAZORAT")
    y += 22

    col_w2 = (W - margin * 2 - gap) // 2
    draw_mini_status_card(info, (margin, y), (margin + col_w2, y + card_h_mini),
                           "ZILZILA", "Xavf yo'q", COL_OK, icon_earthquake)
    draw_mini_status_card(info, (margin + col_w2 + gap, y), (W - margin, y + card_h_mini),
                           "RADIATSION", "Xavf yo'q", COL_OK, icon_radiation)
    y += card_h_mini + 26

    draw_section_label(info, margin, y, W - margin * 2, "STATISTIKA")
    y += 22

    person_color = COL_OK if person_count > 0 else TXT_MUTED
    draw_count_card(info, (margin, y), (W - margin, y + card_h_small), "ODAMLAR SONI",
                     str(person_count), person_color, icon_person)
    y += card_h_small + gap

    draw_peak_card(info, (margin, y), (W - margin, y + card_h_small), "ENG KO'P ODAM (PIK)",
                    f"{peak_count} kishi", peak_time, COL_WATCH)
    y += card_h_small + gap

    draw_peak_card(info, (margin, y), (W - margin, y + card_h_small), "ENG YUQORI YONG'IN (PIK)",
                    f"~{fire_peak_area:.0f} m2", fire_peak_time, COL_DANGER)
    y += card_h_small + 26

    overall_danger = fire_severity == "high" or smoke_severity == "high" or drone_detected
    overall_watch = (not overall_danger) and (fire_severity == "minor" or smoke_severity == "minor")
    if overall_danger:
        status_color, status_word = COL_DANGER, "XAVFLI"
    elif overall_watch:
        status_color, status_word = COL_WATCH, "NAZORATDA"
    else:
        status_color, status_word = COL_OK, "NORMAL"

    draw_rounded_rect(info, (margin, y), (W - margin, y + status_h), status_color, -1, radius=16)
    cv2.putText(info, f"HOLAT: {status_word}", (margin + 24, y + status_h // 2 + 12),
                FONT, 0.9, (25, 20, 20), 2, cv2.LINE_AA)
    y += status_h + gap

    weather_icon = icon_sun if "Quyoshli" in weather_label else icon_cloud
    draw_weather_card(info, (margin, y), (W - margin, y + card_h_small), weather_label, humidity_label, weather_icon)
    y += card_h_small + 24

    timestamp = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(info, timestamp, (margin, y), FONT_LIGHT, 0.4, TXT_MUTED, 1, cv2.LINE_AA)
    y += 20

    return info[:min(y, CANVAS_H), :]


def detect_smoke(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    lower_smoke_light = np.array([0, 0, 150])
    upper_smoke_light = np.array([180, 45, 255])
    lower_smoke_dark = np.array([0, 0, 40])
    upper_smoke_dark = np.array([180, 45, 150])

    mask_light = cv2.inRange(hsv, lower_smoke_light, upper_smoke_light)
    mask_dark = cv2.inRange(hsv, lower_smoke_dark, upper_smoke_dark)
    smoke_mask = cv2.bitwise_or(mask_light, mask_dark)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_OPEN, kernel)
    smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_CLOSE, kernel)

    smoke_pixels = cv2.countNonZero(smoke_mask)
    total_pixels = frame.shape[0] * frame.shape[1]
    smoke_ratio = smoke_pixels / total_pixels

    contours, _ = cv2.findContours(smoke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    smoke_regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 1500:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        if not (0.3 < aspect_ratio < 4.0):
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        if circularity > 0.75:
            continue

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 1.0
        if solidity > 0.92:
            continue

        center_y = y + h / 2
        if center_y > frame.shape[0] * SMOKE_MAX_Y_RATIO:
            continue

        roi = gray[y:y + h, x:x + w]
        if roi.size == 0:
            continue
        sharpness = cv2.Laplacian(roi, cv2.CV_64F).var()
        if sharpness > 250:
            continue

        smoke_regions.append((x, y, w, h, smoke_ratio))

    return smoke_regions, smoke_mask, smoke_ratio


ENABLE_FIRE_COLOR_FALLBACK = False


def detect_fire_color(frame):
    if not ENABLE_FIRE_COLOR_FALLBACK:
        return []

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_fire1 = np.array([0, 140, 180])
    upper_fire1 = np.array([18, 255, 255])
    lower_fire2 = np.array([170, 140, 180])
    upper_fire2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_fire1, upper_fire1)
    mask2 = cv2.inRange(hsv, lower_fire2, upper_fire2)
    fire_mask = cv2.bitwise_or(mask1, mask2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)
    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel)
    fire_mask = cv2.dilate(fire_mask, kernel, iterations=2)

    contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fire_regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 500:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        region_mask = fire_mask[y:y + h, x:x + w]
        region_hsv = hsv[y:y + h, x:x + w]
        mask_bool = region_mask > 0
        if mask_bool.sum() == 0:
            continue

        hue_rad = region_hsv[:, :, 0][mask_bool].astype(np.float64) * (np.pi / 90.0)
        r_len = np.sqrt(np.mean(np.cos(hue_rad)) ** 2 + np.mean(np.sin(hue_rad)) ** 2)
        hue_circ_std = math.sqrt(-2 * math.log(r_len)) if r_len > 1e-6 else 999.0

        v_all = region_hsv[:, :, 2].astype(np.float64)
        s_all = region_hsv[:, :, 1].astype(np.float64)
        hot_white_ratio = float(np.mean((v_all > 190) & (s_all < 140)))

        if hue_circ_std < 0.35 and hot_white_ratio < 0.015:
            continue

        fire_regions.append((x, y, w, h))

    return fire_regions


def estimate_weather_humidity(frame):
    h, w = frame.shape[:2]
    sky_region = frame[0:int(h * 0.28), :]

    hsv_sky = cv2.cvtColor(sky_region, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([90, 40, 90])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv_sky, lower_blue, upper_blue)
    blue_ratio = cv2.countNonZero(blue_mask) / (sky_region.shape[0] * sky_region.shape[1])

    mean_v = float(np.mean(hsv_sky[:, :, 2]))
    mean_s = float(np.mean(hsv_sky[:, :, 1]))

    if blue_ratio > 0.35 and mean_v > 130:
        weather = "Quyoshli"
        humidity = "~15-30%"
    elif mean_v < 90:
        weather = "Bulutli / Qorong'i"
        humidity = "~40-55%"
    elif mean_s < 40 and mean_v > 130:
        weather = "Bulutli"
        humidity = "~30-45%"
    else:
        weather = "Aralash (qisman bulutli)"
        humidity = "~25-40%"

    return weather, humidity


def classify_fire_scale(fire_boxes, frame_shape):
    frame_area = frame_shape[0] * frame_shape[1]
    total_area = sum(w * h for (x, y, w, h) in fire_boxes)
    percent = (total_area / frame_area) * 100 if frame_area > 0 else 0.0

    if percent <= 0:
        label, severity = "Aniqlanmadi", "none"
    elif percent < FIRE_MINOR_PERCENT:
        label, severity = "Kichik o'choq", "minor"
    elif percent < FIRE_HIGH_PERCENT:
        label, severity = "O'rta o'choq", "high"
    else:
        label, severity = "Katta o'choq", "high"

    return label, percent, severity


def classify_smoke_severity(smoke_ratio):
    if smoke_ratio < SMOKE_MINOR_THRESHOLD:
        return "none"
    elif smoke_ratio < SMOKE_HIGH_THRESHOLD:
        return "minor"
    else:
        return "high"


def estimate_hazard_zone(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    edge_density = cv2.countNonZero(edges) / (frame.shape[0] * frame.shape[1])

    is_hazard = edge_density > 0.16
    return is_hazard, edge_density


PEAK_STATE_FILE = os.path.join(OUTPUT_DIR, "peak_state.json")


def reset_peak_state():
    if os.path.exists(PEAK_STATE_FILE):
        try:
            os.remove(PEAK_STATE_FILE)
        except Exception:
            pass


def load_peak_state():
    default = {"peak_count": 0, "peak_time": "-", "fire_peak_area": 0.0, "fire_peak_time": "-"}
    if os.path.exists(PEAK_STATE_FILE):
        try:
            with open(PEAK_STATE_FILE, "r") as f:
                loaded = json.load(f)
                default.update(loaded)
        except Exception:
            pass
    return default


def update_peak_state(person_count, fire_area_m2):
    state = load_peak_state()
    changed = False

    if person_count > state.get("peak_count", 0):
        state["peak_count"] = person_count
        state["peak_time"] = datetime.now().strftime("%H:%M:%S")
        changed = True

    if fire_area_m2 > state.get("fire_peak_area", 0.0):
        state["fire_peak_area"] = fire_area_m2
        state["fire_peak_time"] = datetime.now().strftime("%H:%M:%S")
        changed = True

    if changed:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(PEAK_STATE_FILE, "w") as f:
            json.dump(state, f)

    return state


def process_frame(frame):
    results_person = model_person(frame)

    person_count = 0
    fire_boxes = []
    smoke_boxes = []

    for result in results_person[0].boxes:
        if int(result.cls[0].item()) == 0:
            person_count += 1
            x1, y1, x2, y2 = map(int, result.xyxy[0].cpu().numpy())
            conf = result.conf[0].item()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Odam {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if use_fire_model:
        results_fire = model_fire(frame)
        for result in results_fire[0].boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0].cpu().numpy())
            conf = result.conf[0].item()
            cls_id = int(result.cls[0].item())
            w, h = x2 - x1, y2 - y1
            if cls_id == 1:
                if conf < FIRE_CONF_THRESHOLD:
                    continue
                fire_boxes.append((x1, y1, w, h))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f'Olov {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                if conf < SMOKE_CONF_THRESHOLD:
                    continue
                smoke_boxes.append((x1, y1, w, h))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 2)
                cv2.putText(frame, f'Tutun {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)
        smoke_area = sum(w * h for x, y, w, h in smoke_boxes)
        smoke_ratio = smoke_area / (frame.shape[0] * frame.shape[1])
    else:
        fire_regions = detect_fire_color(frame)
        for x, y, w, h in fire_regions:
            fire_boxes.append((x, y, w, h))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, 'Olov 0.75', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        smoke_regions, smoke_mask, smoke_ratio = detect_smoke(frame)
        for x, y, w, h, confidence in smoke_regions:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (128, 128, 128), 2)
            cv2.putText(frame, f'Tutun {confidence:.2f}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)

    fire_scale_label, fire_area_percent, fire_severity = classify_fire_scale(fire_boxes, frame.shape)
    smoke_severity = classify_smoke_severity(smoke_ratio)
    weather_label, humidity_label = estimate_weather_humidity(frame)

    drone_detected = False
    if use_drone_model:
        results_drone = model_drone(frame)
        for result in results_drone[0].boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0].cpu().numpy())
            conf = result.conf[0].item()
            drone_detected = True
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 2)
            cv2.putText(frame, f'Dron {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)

    return (person_count, fire_severity, fire_area_percent, smoke_severity,
            drone_detected, weather_label, humidity_label)


def select_camera_source():
    result = {"source": None}

    root = tk.Tk()
    root.title("Kamera tanlash")
    root.geometry("360x260")
    root.resizable(False, False)

    choice = tk.StringVar(value="0")
    ip_var = tk.StringVar()

    tk.Label(root, text="Kamera manbasini tanlang", font=("Segoe UI", 12, "bold")).pack(pady=(16, 10))

    tk.Radiobutton(root, text="Kamera 0 (asosiy)", variable=choice, value="0",
                   font=("Segoe UI", 10)).pack(anchor="w", padx=28, pady=2)
    tk.Radiobutton(root, text="Kamera 1", variable=choice, value="1",
                   font=("Segoe UI", 10)).pack(anchor="w", padx=28, pady=2)
    tk.Radiobutton(root, text="IP kamera (brauzer orqali)", variable=choice, value="ip",
                   font=("Segoe UI", 10)).pack(anchor="w", padx=28, pady=2)

    ip_entry = tk.Entry(root, textvariable=ip_var, width=34, state="disabled")
    ip_entry.pack(pady=(10, 2), padx=28)
    tk.Label(root, text="masalan: http://192.168.1.50:8080/video", fg="gray",
             font=("Segoe UI", 8)).pack()

    def on_choice_change(*_):
        ip_entry.config(state="normal" if choice.get() == "ip" else "disabled")

    choice.trace_add("write", on_choice_change)

    def on_start():
        if choice.get() == "ip":
            ip = ip_var.get().strip()
            if not ip:
                messagebox.showwarning("Xato", "IP kamera manzilini kiriting")
                return
            result["source"] = ip
        else:
            result["source"] = int(choice.get())
        root.destroy()

    tk.Button(root, text="Boshlash", command=on_start, width=18,
              font=("Segoe UI", 10, "bold")).pack(pady=18)

    root.protocol("WM_DELETE_WINDOW", lambda: (result.update(source=None), root.destroy()))
    root.mainloop()
    return result["source"]


def main():
    camera_source = select_camera_source()
    if camera_source is None:
        print("Kamera tanlanmadi, chiqilmoqda")
        return

    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print(f"Kamerani ochib bo'lmadi: {camera_source}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    MAX_DISPLAY_H = 760

    def to_display(panel):
        if panel.shape[0] > MAX_DISPLAY_H:
            scale = MAX_DISPLAY_H / panel.shape[0]
            return cv2.resize(panel, (int(panel.shape[1] * scale), MAX_DISPLAY_H))
        return panel

    cv2.namedWindow('Kamera', cv2.WINDOW_NORMAL)
    cv2.namedWindow("Ma'lumotlar Paneli", cv2.WINDOW_NORMAL)

    flash_on = True
    last_flash_toggle = time.time()
    last_periodic_save = 0.0
    prev_alert = False

    print("Kamera ishga tushdi. Chiqish uchun 'q' tugmasini bosing")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kadr o'qib bo'lmadi, kamera uzildi")
            break

        person_count, fire_severity, fire_area_percent, smoke_severity, drone_detected, \
            weather_label, humidity_label = process_frame(frame)

        fire_area_m2 = fire_area_percent / 100.0 * FRAME_AREA_M2
        peak_state = update_peak_state(person_count, fire_area_m2)

        need_blink = fire_severity != "none" or smoke_severity != "none" or drone_detected
        now = time.time()
        if now - last_flash_toggle > FLASH_INTERVAL_SEC:
            flash_on = not flash_on
            last_flash_toggle = now
        current_flash = flash_on if need_blink else False

        info_window = create_info_window(
            fire_severity, smoke_severity, person_count, peak_state["peak_count"], peak_state["peak_time"],
            fire_area_percent=fire_area_percent, fire_peak_area=peak_state["fire_peak_area"],
            fire_peak_time=peak_state["fire_peak_time"], drone_detected=drone_detected,
            weather_label=weather_label, humidity_label=humidity_label, flash_on=current_flash
        )

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if need_blink and not prev_alert:
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"hodisa_{stamp}.jpg"), frame)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"panel_{stamp}.jpg"), info_window)
        elif now - last_periodic_save > SAVE_INTERVAL_SEC:
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"kamera_{stamp}.jpg"), frame)
            last_periodic_save = now
        prev_alert = need_blink

        cv2.imshow('Kamera', frame)
        cv2.imshow("Ma'lumotlar Paneli", to_display(info_window))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    reset_peak_state()
    main()
