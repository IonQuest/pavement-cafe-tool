import io
import re
import cv2
import fitz
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
from rapidocr_onnxruntime import RapidOCR
import streamlit as st

st.set_page_config(page_title="Pavement Cafe Plan Generator", layout="wide")
st.title("Pavement Cafe Dimensioned Polygon Generator")


@st.cache_resource
def load_ocr():
    return RapidOCR()


ocr = load_ocr()


def draw_dimension_arrow(draw, p1, p2, text, offset=25, color="black"):
    x1, y1 = p1
    x2, y2 = p2
    if y1 == y2:
        yd = y1 + offset
        draw.line([(x1, y1), (x1, yd)], fill="#64748b", width=1)
        draw.line([(x2, y2), (x2, yd)], fill="#64748b", width=1)
        draw.line([(x1, yd), (x2, yd)], fill=color, width=2)
        draw.line([(x1 - 4, yd + 4), (x1 + 4, yd - 4)], fill=color, width=2)
        draw.line([(x2 - 4, yd + 4), (x2 + 4, yd - 4)], fill=color, width=2)
        draw.text(((x1 + x2) // 2 - 18, yd + 5), text, fill=color)
    else:
        xd = x1 + offset
        draw.line([(x1, y1), (xd, y1)], fill="#64748b", width=1)
        draw.line([(x2, y2), (xd, y2)], fill="#64748b", width=1)
        draw.line([(xd, y1), (xd, y2)], fill=color, width=2)
        draw.line([(xd - 4, y1 + 4), (xd + 4, y1 - 4)], fill=color, width=2)
        draw.line([(xd - 4, y2 + 4), (xd + 4, y2 - 4)], fill=color, width=2)
        draw.text((xd + 8, (y1 + y2) // 2 - 6), text, fill=color)


def draw_zone_with_entrance(draw, rect, fill_color, entrance_side="None", gap_ratio=0.32):
    x0, y0, x1, y1 = rect
    draw.rectangle([x0, y0, x1, y1], fill=fill_color, outline=None)

    w = x1 - x0
    h = y1 - y0
    tick = 10

    if entrance_side == "top":
        gx0 = x0 + int(w * (0.5 - gap_ratio / 2))
        gx1 = x0 + int(w * (0.5 + gap_ratio / 2))
        draw.line([(x0, y0), (gx0, y0)], fill="black", width=3)
        draw.line([(gx1, y0), (x1, y0)], fill="black", width=3)
        draw.line([(gx0, y0), (gx0, y0 + tick)], fill="black", width=2)
        draw.line([(gx1, y0), (gx1, y0 + tick)], fill="black", width=2)
        draw.text(((gx0 + gx1) // 2 - 26, y0 + 4), "ENTRANCE", fill="#0f172a")
    else:
        draw.line([(x0, y0), (x1, y0)], fill="black", width=3)

    if entrance_side == "bottom":
        gx0 = x0 + int(w * (0.5 - gap_ratio / 2))
        gx1 = x0 + int(w * (0.5 + gap_ratio / 2))
        draw.line([(x0, y1), (gx0, y1)], fill="black", width=3)
        draw.line([(gx1, y1), (x1, y1)], fill="black", width=3)
        draw.line([(gx0, y1), (gx0, y1 - tick)], fill="black", width=2)
        draw.line([(gx1, y1), (gx1, y1 - tick)], fill="black", width=2)
        draw.text(((gx0 + gx1) // 2 - 26, y1 - 16), "ENTRANCE", fill="#0f172a")
    else:
        draw.line([(x0, y1), (x1, y1)], fill="black", width=3)

    if entrance_side == "left":
        gy0 = y0 + int(h * (0.5 - gap_ratio / 2))
        gy1 = y0 + int(h * (0.5 + gap_ratio / 2))
        draw.line([(x0, y0), (x0, gy0)], fill="black", width=3)
        draw.line([(x0, gy1), (x0, y1)], fill="black", width=3)
        draw.line([(x0, gy0), (x0 + tick, gy0)], fill="black", width=2)
        draw.line([(x0, gy1), (x0 + tick, gy1)], fill="black", width=2)
        draw.text((x0 + 6, (gy0 + gy1) // 2 - 6), "ENTRANCE", fill="#0f172a")
    else:
        draw.line([(x0, y0), (x0, y1)], fill="black", width=3)

    if entrance_side == "right":
        gy0 = y0 + int(h * (0.5 - gap_ratio / 2))
        gy1 = y0 + int(h * (0.5 + gap_ratio / 2))
        draw.line([(x1, y0), (x1, gy0)], fill="black", width=3)
        draw.line([(x1, gy1), (x1, y1)], fill="black", width=3)
        draw.line([(x1, gy0), (x1 - tick, gy0)], fill="black", width=2)
        draw.line([(x1, gy1), (x1 - tick, gy1)], fill="black", width=2)
        draw.text((x1 - 58, (gy0 + gy1) // 2 - 6), "ENTRANCE", fill="#0f172a")
    else:
        draw.line([(x1, y0), (x1, y1)], fill="black", width=3)


def box_area(box):
    bx, by, bw, bh = box
    return bw * bh


def box_top_y(box):
    bx, by, bw, bh = box
    return by


def detect_entrance_opening(thresh, box):
    bx, by, bw, bh = box
    h, w = thresh.shape
    band = max(8, int(min(bw, bh) * 0.14))

    edges = {
        "top": thresh[max(0, by - band):min(h, by + band), bx + int(bw * 0.15):bx + int(bw * 0.85)],
        "bottom": thresh[max(0, by + bh - band):min(h, by + bh + band), bx + int(bw * 0.15):bx + int(bw * 0.85)],
        "left": thresh[by + int(bh * 0.15):by + int(bh * 0.85), max(0, bx - band):min(w, bx + band)],
        "right": thresh[by + int(bh * 0.15):by + int(bh * 0.85), max(0, bx + bw - band):min(w, bx + bw + band)],
    }

    best_side = "None"
    best_gap_ratio = 0.0

    for side, strip in edges.items():
        if strip.size == 0:
            continue
        axis = 0 if side in ("top", "bottom") else 1
        profile = (strip > 0).any(axis=axis).astype(np.uint8)
        if profile.size < 10:
            continue

        max_run = 0
        cur_run = 0
        for val in profile:
            if val == 0:
                cur_run += 1
                if cur_run > max_run:
                    max_run = cur_run
            else:
                cur_run = 0

        gap_ratio = max_run / float(profile.size)
        if 0.15 <= gap_ratio <= 0.65 and gap_ratio > best_gap_ratio:
            best_gap_ratio = gap_ratio
            best_side = side

    return best_side


def detect_cafe_rectangles(img_cv):
    h, w = img_cv.shape[:2]
    page_area = h * w
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    bg = cv2.medianBlur(gray, 51)
    norm = 255 - cv2.absdiff(gray, bg)
    blurred = cv2.GaussianBlur(norm, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 12
    )

    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, w // 14), 3))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, max(25, h // 14)))
    bridged = cv2.bitwise_or(
        cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, horiz_kernel),
        cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, vert_kernel),
    )
    bridged = cv2.morphologyEx(bridged, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

    contours, _ = cv2.findContours(bridged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    raw_boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < (page_area * 0.0012) or area > (page_area * 0.45):
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        rect_fill = area / float(bw * bh) if (bw * bh) > 0 else 0
        aspect = bw / float(bh) if bh > 0 else 0
        if bw >= 30 and bh >= 22 and rect_fill > 0.62 and 0.15 <= aspect <= 7.5:
            raw_boxes.append((bx, by, bw, bh))

    raw_boxes.sort(key=box_area, reverse=True)
    unique_boxes = []
    for bx, by, bw, bh in raw_boxes:
        is_dup = False
        for ux, uy, uw, uh in unique_boxes:
            if abs(bx - ux) < 30 and abs(by - uy) < 30 and abs(bw - uw) < 45 and abs(bh - uh) < 45:
                is_dup = True
                break
        if not is_dup:
            unique_boxes.append((bx, by, bw, bh))

    inner_boxes = []
    for i, (bx, by, bw, bh) in enumerate(unique_boxes):
        contains_others = 0
        for j, (ox, oy, ow, oh) in enumerate(unique_boxes):
            if i != j and bx <= ox and by <= oy and (bx + bw) >= (ox + ow) and (by + bh) >= (oy + oh):
                contains_others += 1
        if contains_others == 0:
            inner_boxes.append((bx, by, bw, bh))

    inner_boxes.sort(key=box_top_y)
    results = []
    for box in inner_boxes:
        entrance = detect_entrance_opening(thresh, box)
        results.append((box, entrance))
    return results


def extract_dimensions(pil_img):
    gray = np.array(pil_img.convert("L"))
    bg = cv2.medianBlur(gray, 51)
    norm = 255 - cv2.absdiff(gray, bg)
    norm_pil = Image.fromarray(norm).convert("RGB")
    enhanced = ImageEnhance.Contrast(norm_pil).enhance(2.2)
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(2.0)

    dimensions = []
    ocr_results, _ = ocr(np.array(enhanced))
    if ocr_results:
        for item in ocr_results:
            bbox, text, _ = item
            clean_text = (
                str(text)
                .replace(",", ".")
                .replace("O", "0")
                .replace("o", "0")
                .replace("l", "1")
                .replace("I", "1")
            )
            for match in re.finditer(r"(\d+\.?\d*)", clean_text):
                val = float(match.group(1))
                if val > 100:
                    val /= 1000.0
                if 0.2 <= val <= 50.0:
                    center_x, center_y = np.array(bbox, dtype=float).mean(axis=0)
                    dimensions.append({"val": round(val, 2), "x": int(center_x), "y": int(center_y)})
    return dimensions


def pick_best_dim(dimensions, target_x, target_y, fallback):
    if not dimensions:
        return fallback
    best = min(dimensions, key=lambda d: (d["x"] - target_x) ** 2 + (d["y"] - target_y) ** 2)
    return best["val"]


uploaded_file = st.file_uploader(
    "Upload Site Plan (JPG, PNG, WEBP, or PDF)",
    type=["jpg", "jpeg", "png", "webp", "pdf"],
)

if uploaded_file:
    file_bytes = uploaded_file.read()

    if uploaded_file.name.lower().endswith(".pdf"):
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pix = doc.load_page(0).get_pixmap(dpi=200)
        pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    else:
        pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    w, h = pil_img.size
    if w > 1400:
        pil_img = pil_img.resize((1400, int(h * (1400 / w))))

    with st.spinner("Detecting pavement cafe zones, entrances, and dimensions..."):
        img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        zone_results = detect_cafe_rectangles(img_cv)
        dimensions = extract_dimensions(pil_img)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Uploaded Plan & Detection Overlay")
        debug_img = pil_img.copy()
        debug_draw = ImageDraw.Draw(debug_img)
        for idx, ((bx, by, bw, bh), ent) in enumerate(zone_results):
            debug_draw.rectangle([bx, by, bx + bw, by + bh], outline="#16a34a", width=4)
            tag = f"Zone {idx + 1}" + (f" (Entrance: {ent})" if ent != "None" else "")
            debug_draw.text((bx + 5, max(5, by - 18)), tag, fill="#16a34a")
        for d in dimensions:
            dx, dy = d["x"], d["y"]
            debug_draw.ellipse([dx - 5, dy - 5, dx + 5, dy + 5], fill="#dc2626")
            debug_draw.text((dx + 8, dy - 8), f"{d['val']}m", fill="#dc2626")
        st.image(debug_img, use_container_width=True)
        st.caption("Green boxes = detected PC zones | Red dots = numbers read by OCR")

    adj_w, adj_d, adj_ent = 5.0, 2.0, "None"
    has_remote, rem_w, rem_d, rem_ent, gap = False, 2.0, 2.0, "None", 2.0

    if zone_results:
        (bx, by, bw, bh), adj_ent = zone_results[-1]
        adj_w = pick_best_dim(dimensions, bx + bw / 2, by - 25, adj_w)
        adj_d = pick_best_dim(dimensions, bx - 30, by + bh / 2, adj_d)

        if len(zone_results) >= 2:
            has_remote = True
            (rx, ry, rw, rh), rem_ent = next(iter(zone_results))
            rem_w = pick_best_dim(dimensions, rx + rw / 2, ry - 25, rem_w)
            rem_d = pick_best_dim(dimensions, rx + rw + 30, ry + rh / 2, rem_d)
            gap_y = (ry + rh + by) / 2
            gap = pick_best_dim(dimensions, min(bx, rx) - 35, gap_y, gap)

    with col2:
        st.subheader("Dimensioned Polygon Output")
        detected_vals = [f"{d['val']}m" for d in dimensions]
        st.caption(
            f"Detected {len(zone_results)} zone(s) | Numbers read: {', '.join(detected_vals) if detected_vals else 'None'}"
        )

        ent_options = ["None", "top", "bottom", "left", "right"]
        c1, c2, c3 = st.columns(3)
        adj_w = c1.number_input("Adjacent Width (m)", value=float(adj_w), min_value=0.1, step=0.1)
        adj_d = c2.number_input("Adjacent Depth (m)", value=float(adj_d), min_value=0.1, step=0.1)
        adj_ent = c3.selectbox("Adjacent Entrance", ent_options, index=ent_options.index(adj_ent))

        has_remote = st.checkbox("Include Remote PC", value=has_remote)
        if has_remote:
            r1, r2, r3, r4 = st.columns(4)
            rem_w = r1.number_input("Remote Width (m)", value=float(rem_w), min_value=0.1, step=0.1)
            rem_d = r2.number_input("Remote Depth (m)", value=float(rem_d), min_value=0.1, step=0.1)
            gap = r3.number_input("Separation Gap (m)", value=float(gap), min_value=0.1, step=0.1)
            rem_ent = r4.selectbox("Remote Entrance", ent_options, index=ent_options.index(rem_ent))

        total_w_m = max(adj_w, rem_w) if has_remote else adj_w
        total_h_m = (adj_d + gap + rem_d) if has_remote else adj_d
        scale = min(500 / total_w_m, 500 / total_h_m)

        out = Image.new("RGB", (800, 800), "white")
        draw = ImageDraw.Draw(out)

        ax0 = 130
        ay1 = 680
        ay0 = ay1 - int(adj_d * scale)
        ax1 = ax0 + int(adj_w * scale)

        draw.line([(60, ay1), (740, ay1)], fill="black", width=4)
        draw.text((65, ay1 + 12), "PREMISES / BUILDING FRONTAGE", fill="black")

        draw_zone_with_entrance(draw, (ax0, ay0, ax1, ay1), "#dbeafe", entrance_side=adj_ent)
        adj_area = round(adj_w * adj_d, 2)
        draw.text((ax0 + 12, (ay0 + ay1) // 2 - 8), f"Adjacent PC ({adj_area} m²)", fill="black")

        draw_dimension_arrow(draw, (ax0, ay1), (ax1, ay1), f"{adj_w}m", offset=35)
        draw_dimension_arrow(draw, (ax1, ay0), (ax1, ay1), f"{adj_d}m", offset=25)

        if has_remote:
            ry1 = ay0 - int(gap * scale)
            ry0 = ry1 - int(rem_d * scale)
            rx1 = ax0 + int(rem_w * scale)

            draw_zone_with_entrance(draw, (ax0, ry0, rx1, ry1), "#fef3c7", entrance_side=rem_ent)
            rem_area = round(rem_w * rem_d, 2)
            draw.text((ax0 + 12, (ry0 + ry1) // 2 - 8), f"Remote PC ({rem_area} m²)", fill="black")

            draw_dimension_arrow(draw, (ax0, ry0), (rx1, ry0), f"{rem_w}m", offset=-25)
            draw_dimension_arrow(draw, (rx1, ry0), (rx1, ry1), f"{rem_d}m", offset=25)

            gap_x = ax0 + int(min(adj_w, rem_w) * scale * 0.5)
            draw.line([(gap_x, ry1), (gap_x, ay0)], fill="#dc2626", width=2)
            draw.text((gap_x + 8, (ry1 + ay0) // 2 - 6), f"Clear Gap: {gap}m", fill="#dc2626")

            total_area = round(adj_area + rem_area, 2)
            draw.text((60, 30), f"Total Pavement Cafe Area: {total_area} m²", fill="black")
        else:
            draw.text((60, 30), f"Total Pavement Cafe Area: {adj_area} m²", fill="black")

        st.image(out, use_container_width=True)

        buf = io.BytesIO()
        out.save(buf, format="JPEG", quality=95)
        st.download_button(
            label="Download Dimensioned JPEG",
            data=buf.getvalue(),
            file_name="pavement_cafe_plan.jpg",
            mime="image/jpeg",
        )
