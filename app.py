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


def normalize_sketch(img_cv):
    """Removes warm lighting/shadow gradients and isolates dark ink lines."""
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    bg = cv2.GaussianBlur(gray, (51, 51), 0)
    norm = cv2.divide(gray, bg, scale=255)
    thresh = cv2.adaptiveThreshold(
        norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 18
    )
    return thresh


def detect_entrance_opening(thresh, bx, by, bw, bh):
    """Checks the 4 edges of a detected zone for an intentional customer entrance gap."""
    h, w = thresh.shape
    band = max(12, int(min(bw, bh) * 0.14))

    def find_gap_on_edge(strip, is_horizontal):
        # Project ink onto 1D line along the edge
        proj = np.sum(strip > 0, axis=0 if is_horizontal else 1)
        length = len(proj)
        if length < 30:
            return None
        # Ignore outer 15% corners; look for a clear gap in the middle 70%
        margin = int(length * 0.15)
        mid = proj[margin : length - margin]
        empty = (mid == 0).astype(np.uint8)
        # Find longest contiguous run of empty pixels
        best_len = 0
        best_start = 0
        cur_len = 0
        cur_start = 0
        for idx, val in enumerate(empty):
            if val:
                if cur_len == 0:
                    cur_start = idx
                cur_len += 1
                if cur_len > best_len:
                    best_len = cur_len
                    best_start = cur_start
            else:
                cur_len = 0
        # An intentional entrance gap is typically 12% to 55% of the edge length
        if int(length * 0.12) <= best_len <= int(length * 0.55):
            start_ratio = (margin + best_start) / float(length)
            end_ratio = (margin + best_start + best_len) / float(length)
            return (round(start_ratio, 2), round(end_ratio, 2))
        return None

    y_top0, y_top1 = max(0, by - band), min(h, by + band)
    y_bot0, y_bot1 = max(0, by + bh - band), min(h, by + bh + band)
    x_lft0, x_lft1 = max(0, bx - band), min(w, bx + band)
    x_rgt0, x_rgt1 = max(0, bx + bw - band), min(w, bx + bw + band)

    top_gap = find_gap_on_edge(thresh[y_top0:y_top1, bx : bx + bw], True)
    if top_gap:
        return ("top", top_gap[0], top_gap[1])

    bot_gap = find_gap_on_edge(thresh[y_bot0:y_bot1, bx : bx + bw], True)
    if bot_gap:
        return ("bottom", bot_gap[0], bot_gap[1])

    lft_gap = find_gap_on_edge(thresh[by : by + bh, x_lft0:x_lft1], False)
    if lft_gap:
        return ("left", lft_gap[0], lft_gap[1])

    rgt_gap = find_gap_on_edge(thresh[by : by + bh, x_rgt0:x_rgt1], False)
    if rgt_gap:
        return ("right", rgt_gap[0], rgt_gap[1])

    return None


def detect_cafe_zones(img_cv):
    """Detects both closed zones and zones with customer entrance gaps."""
    h, w = img_cv.shape[:2]
    page_area = h * w
    thresh = normalize_sketch(img_cv)

    # Bridge customer entrance gaps temporarily so we get the full outer footprint
    bridge_w = max(25, int(w * 0.14))
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (bridge_w, 5))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, max(20, int(h * 0.08))))
    bridged = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, horiz_kernel)
    bridged = cv2.morphologyEx(bridged, cv2.MORPH_CLOSE, vert_kernel)

    contours, _ = cv2.findContours(bridged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    raw_boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < (page_area * 0.01) or area > (page_area * 0.55):
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        aspect = bw / float(bh) if bh > 0 else 0
        # Filter out thin horizontal dimension arrows (aspect > 5)
        if bw > 60 and bh > 45 and 0.35 <= aspect <= 4.8:
            raw_boxes.append((bx, by, bw, bh))

    raw_boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    unique_boxes = []
    for bx, by, bw, bh in raw_boxes:
        is_dup = False
        for ux, uy, uw, uh in unique_boxes:
            if abs(bx - ux) < 35 and abs(by - uy) < 35 and abs(bw - uw) < 50 and abs(bh - uh) < 50:
                is_dup = True
                break
        if not is_dup:
            unique_boxes.append((bx, by, bw, bh))

    # Remove any outer container box that encloses other boxes
    inner_boxes = []
    for i, (bx, by, bw, bh) in enumerate(unique_boxes):
        contains = 0
        for j, (ox, oy, ow, oh) in enumerate(unique_boxes):
            if i != j and bx <= ox + 10 and by <= oy + 10 and (bx + bw) >= (ox + ow - 10) and (by + bh) >= (oy + oh - 10):
                contains += 1
        if contains == 0:
            inner_boxes.append((bx, by, bw, bh))

    inner_boxes.sort(key=lambda b: b[1])

    zones = []
    for bx, by, bw, bh in inner_boxes:
        entrance = detect_entrance_opening(thresh, bx, by, bw, bh)
        zones.append({"box": (bx, by, bw, bh), "entrance": entrance})
    return zones


def extract_dimensions(pil_img):
    """Reads dimension numbers on original and high-contrast grayscale images."""
    gray = pil_img.convert("L")
    boosted = ImageEnhance.Contrast(gray).enhance(2.4)
    boosted = ImageEnhance.Sharpness(boosted).enhance(2.0)

    dimensions = []
    for variant in [np.array(boosted.convert("RGB")), np.array(pil_img)]:
        ocr_results, _ = ocr(variant)
        if ocr_results:
            for item in ocr_results:
                bbox, text, _ = item
                clean = (
                    str(text)
                    .replace(",", ".")
                    .replace("O", "0")
                    .replace("o", "0")
                    .replace("Z", "2")
                    .replace("z", "2")
                    .replace("S", "5")
                    .replace("s", "5")
                )
                for match in re.finditer(r"(\d+\.?\d*)", clean):
                    val = float(match.group(1))
                    if val > 100:
                        val /= 1000.0
                    if 0.3 <= val <= 40.0:
                        cx, cy = np.array(bbox, dtype=float).mean(axis=0)
                        cx, cy = int(cx), int(cy)
                        if not any(abs(d["x"] - cx) < 25 and abs(d["y"] - cy) < 25 for d in dimensions):
                            dimensions.append({"val": round(val, 2), "x": cx, "y": cy})
    return dimensions


def draw_zone_with_entrance(draw, x0, y0, x1, y1, fill_color, entrance, line_w=3):
    """Draws a shaded zone polygon with an explicit open customer entrance gap."""
    draw.rectangle([x0, y0, x1, y1], fill=fill_color, outline=None)

    # Top edge
    if entrance and entrance[0] == "top":
        g0 = x0 + int((x1 - x0) * entrance[1])
        g1 = x0 + int((x1 - x0) * entrance[2])
        draw.line([(x0, y0), (g0, y0)], fill="black", width=line_w)
        draw.line([(g1, y0), (x1, y0)], fill="black", width=line_w)
        # Inward return ticks at entrance opening
        draw.line([(g0, y0), (g0, y0 + 8)], fill="black", width=2)
        draw.line([(g1, y0), (g1, y0 + 8)], fill="black", width=2)
        draw.text(((g0 + g1) // 2 - 24, y0 + 6), "ENTRANCE", fill="#1e3a8a")
    else:
        draw.line([(x0, y0), (x1, y0)], fill="black", width=line_w)

    # Bottom edge
    if entrance and entrance[0] == "bottom":
        g0 = x0 + int((x1 - x0) * entrance[1])
        g1 = x0 + int((x1 - x0) * entrance[2])
        draw.line([(x0, y1), (g0, y1)], fill="black", width=line_w)
        draw.line([(g1, y1), (x1, y1)], fill="black", width=line_w)
        draw.line([(g0, y1), (g0, y1 - 8)], fill="black", width=2)
        draw.line([(g1, y1), (g1, y1 - 8)], fill="black", width=2)
        draw.text(((g0 + g1) // 2 - 24, y1 - 18), "ENTRANCE", fill="#1e3a8a")
    else:
        draw.line([(x0, y1), (x1, y1)], fill="black", width=line_w)

    # Left edge
    if entrance and entrance[0] == "left":
        g0 = y0 + int((y1 - y0) * entrance[1])
        g1 = y0 + int((y1 - y0) * entrance[2])
        draw.line([(x0, y0), (x0, g0)], fill="black", width=line_w)
        draw.line([(x0, g1), (x0, y1)], fill="black", width=line_w)
        draw.line([(x0, g0), (x0 + 8, g0)], fill="black", width=2)
        draw.line([(x0, g1), (x0 + 8, g1)], fill="black", width=2)
        draw.text((x0 + 10, (g0 + g1) // 2 - 6), "ENTRANCE", fill="#1e3a8a")
    else:
        draw.line([(x0, y0), (x0, y1)], fill="black", width=line_w)

    # Right edge
    if entrance and entrance[0] == "right":
        g0 = y0 + int((y1 - y0) * entrance[1])
        g1 = y0 + int((y1 - y0) * entrance[2])
        draw.line([(x1, y0), (x1, g0)], fill="black", width=line_w)
        draw.line([(x1, g1), (x1, y1)], fill="black", width=line_w)
        draw.line([(x1, g0), (x1 - 8, g0)], fill="black", width=2)
        draw.line([(x1, g1), (x1 - 8, g1)], fill="black", width=2)
        draw.text((x1 - 65, (g0 + g1) // 2 - 6), "ENTRANCE", fill="#1e3a8a")
    else:
        draw.line([(x1, y0), (x1, y1)], fill="black", width=line_w)


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
    if w > 1200:
        pil_img = pil_img.resize((1200, int(h * (1200 / w))))

    with st.spinner("Detecting pavement cafe zones, entrances, and dimensions..."):
        img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        zones = detect_cafe_zones(img_cv)
        dimensions = extract_dimensions(pil_img)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Uploaded Plan & Detected Features")
        debug_img = pil_img.copy()
        debug_draw = ImageDraw.Draw(debug_img)
        for idx, z in enumerate(zones):
            bx, by, bw, bh = z["box"]
            ent = z["entrance"]
            debug_draw.rectangle([bx, by, bx + bw, by + bh], outline="#16a34a", width=4)
            lbl = f"Zone {idx + 1}" + (f" (Entrance: {ent[0]})" if ent else "")
            debug_draw.text((bx + 5, by - 18), lbl, fill="#16a34a")
        for d in dimensions:
            dx, dy = d["x"], d["y"]
            debug_draw.ellipse([dx - 5, dy - 5, dx + 5, dy + 5], fill="#dc2626")
            debug_draw.text((dx + 8, dy - 8), f"{d['val']}m", fill="#dc2626")
        st.image(debug_img, use_container_width=True)

    adj_w, adj_d, has_remote, rem_w, rem_d, gap = 5.0, 2.0, False, 2.0, 2.0, 2.0
    adj_entrance = None
    rem_entrance = None

    if zones:
        adj_zone = zones[-1]
        bx, by, bw, bh = adj_zone["box"]
        adj_entrance = adj_zone["entrance"]

        if dimensions:
            horiz_cands = [d for d in dimensions if (bx - 40) <= d["x"] <= (bx + bw + 40)]
            if horiz_cands:
                adj_w = min(horiz_cands, key=lambda d: min(abs(d["y"] - by), abs(d["y"] - (by + bh))))["val"]
            vert_cands = [d for d in dimensions if (by - 30) <= d["y"] <= (by + bh + 30)]
            if vert_cands:
                adj_d = min(vert_cands, key=lambda d: min(abs(d["x"] - bx), abs(d["x"] - (bx + bw))))["val"]

        if len(zones) >= 2:
            has_remote = True
            rem_zone = zones[0]
            rx, ry, rw, rh = rem_zone["box"]
            rem_entrance = rem_zone["entrance"]

            if dimensions:
                r_horiz = [d for d in dimensions if (rx - 40) <= d["x"] <= (rx + rw + 40)]
                if r_horiz:
                    rem_w = min(r_horiz, key=lambda d: min(abs(d["y"] - ry), abs(d["y"] - (ry + rh))))["val"]
                r_vert = [d for d in dimensions if (ry - 30) <= d["y"] <= (ry + rh + 30)]
                if r_vert:
                    rem_d = min(r_vert, key=lambda d: min(abs(d["x"] - rx), abs(d["x"] - (rx + rw))))["val"]
                gap_y_top = ry + rh
                gap_y_bot = by
                gap_cands = [d for d in dimensions if (gap_y_top - 25) <= d["y"] <= (gap_y_bot + 25)]
                if gap_cands:
                    gap = min(gap_cands, key=lambda d: abs(d["y"] - (gap_y_top + gap_y_bot) / 2))["val"]

    with col2:
        st.subheader("Dimensioned Polygon Output")
        detected_vals = [f"{d['val']}m" for d in dimensions]
        st.caption(
            f"Detected {len(zones)} zone(s) | Numbers read:"
            f" {', '.join(detected_vals) if detected_vals else 'None'}"
        )

        c1, c2, c3 = st.columns(3)
        adj_w = c1.number_input("Adjacent Width (m)", value=float(adj_w), min_value=0.1, step=0.1)
        adj_d = c2.number_input("Adjacent Depth (m)", value=float(adj_d), min_value=0.1, step=0.1)
        ent_options = ["None", "top", "bottom", "left", "right"]
        default_adj_ent = adj_entrance[0] if adj_entrance else "None"
        adj_ent_choice = c3.selectbox(
            "Adjacent Entrance", ent_options, index=ent_options.index(default_adj_ent)
        )
        if adj_ent_choice == "None":
            adj_entrance = None
        elif not adj_entrance or adj_entrance[0] != adj_ent_choice:
            adj_entrance = (adj_ent_choice, 0.35, 0.65)

        has_remote = st.checkbox("Include Remote PC", value=has_remote)
        if has_remote:
            c4, c5, c6, c7 = st.columns(4)
            rem_w = c4.number_input("Remote Width (m)", value=float(rem_w), min_value=0.1, step=0.1)
            rem_d = c5.number_input("Remote Depth (m)", value=float(rem_d), min_value=0.1, step=0.1)
            gap = c6.number_input("Clear Gap (m)", value=float(gap), min_value=0.1, step=0.1)
            default_rem_ent = rem_entrance[0] if rem_entrance else "None"
            rem_ent_choice = c7.selectbox(
                "Remote Entrance", ent_options, index=ent_options.index(default_rem_ent)
            )
            if rem_ent_choice == "None":
                rem_entrance = None
            elif not rem_entrance or rem_entrance[0] != rem_ent_choice:
                rem_entrance = (rem_ent_choice, 0.35, 0.65)

        total_w_m = max(adj_w, rem_w) if has_remote else adj_w
        total_h_m = (adj_d + gap + rem_d) if has_remote else adj_d
        scale = min(500 / total_w_m, 500 / total_h_m)

        out = Image.new("RGB", (800, 800), "white")
        draw = ImageDraw.Draw(out)

        ax0 = 120
        ay1 = 680
        ay0 = ay1 - int(adj_d * scale)
        ax1 = ax0 + int(adj_w * scale)

        draw.line([(60, ay1), (740, ay1)], fill="black", width=4)
        draw.text((65, ay1 + 12), "PREMISES / BUILDING FRONTAGE", fill="black")

        draw_zone_with_entrance(draw, ax0, ay0, ax1, ay1, "#dbeafe", adj_entrance, line_w=3)
        adj_area = round(adj_w * adj_d, 2)
        draw.text(
            (ax0 + 10, (ay0 + ay1) // 2 - 6),
            f"Adjacent PC ({adj_area} m²)",
            fill="black",
        )

        draw_dimension_arrow(draw, (ax0, ay1), (ax1, ay1), f"{adj_w}m", offset=35)
        draw_dimension_arrow(draw, (ax1, ay0), (ax1, ay1), f"{adj_d}m", offset=25)

        if has_remote:
            ry1 = ay0 - int(gap * scale)
            ry0 = ry1 - int(rem_d * scale)
            rx1 = ax0 + int(rem_w * scale)

            draw_zone_with_entrance(draw, ax0, ry0, rx1, ry1, "#fef3c7", rem_entrance, line_w=3)
            rem_area = round(rem_w * rem_d, 2)
            draw.text(
                (ax0 + 10, (ry0 + ry1) // 2 - 6),
                f"Remote PC ({rem_area} m²)",
                fill="black",
            )

            draw_dimension_arrow(draw, (ax0, ry0), (rx1, ry0), f"{rem_w}m", offset=-25)
            draw_dimension_arrow(draw, (rx1, ry0), (rx1, ry1), f"{rem_d}m", offset=25)

            gap_x = ax0 + int(min(adj_w, rem_w) * scale * 0.5)
            draw.line([(gap_x, ry1), (gap_x, ay0)], fill="#dc2626", width=2)
            draw.text(
                (gap_x + 8, (ry1 + ay0) // 2 - 6),
                f"Clear Gap: {gap}m",
                fill="#dc2626",
            )

            total_area = round(adj_area + rem_area, 2)
            draw.text(
                (60, 30), f"Total Pavement Cafe Area: {total_area} m²", fill="black"
            )
        else:
            draw.text(
                (60, 30), f"Total Pavement Cafe Area: {adj_area} m²", fill="black"
            )

        st.image(out, use_container_width=True)

        buf = io.BytesIO()
        out.save(buf, format="JPEG", quality=95)
        st.download_button(
            label="Download Dimensioned JPEG",
            data=buf.getvalue(),
            file_name="pavement_cafe_plan.jpg",
            mime="image/jpeg",
        )
