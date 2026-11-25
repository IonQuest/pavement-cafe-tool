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


def box_area(box):
    bx, by, bw, bh = box
    return bw * bh


def box_top_y(box):
    bx, by, bw, bh = box
    return by


def detect_cafe_rectangles(img_cv):
    """Detects both closed boxes and open C-shaped boxes with customer entrances without merging nearby dimension arrows."""
    h, w = img_cv.shape[:2]
    page_area = h * w
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold captures hand-drawn pen/pencil lines evenly across warm lighting
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
    )
    # Very small 3x3 close just connects slight pen breaks without bridging to nearby arrows
    small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean_lines = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, small_kernel)

    raw_boxes = []

    # Pass 1: Connected-component strokes (captures full C-shaped open boxes from bottom to top returns)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean_lines, connectivity=8)
    for i in range(1, num_labels):
        bx = int(stats[i, cv2.CC_STAT_LEFT])
        by = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        bbox_area = bw * bh

        # Filter out tiny text/dots, giant page borders, and thin 1D arrow lines
        if bbox_area < (page_area * 0.004) or bbox_area > (page_area * 0.45):
            continue
        if bw < w * 0.09 or bh < h * 0.04:
            continue
        # A 2D rectangle or C-shape has low ink density relative to its bounding box (unlike solid blobs)
        # while a single vertical/horizontal arrow shaft is very skinny (aspect ratio > 6)
        aspect = max(bw / float(bh), bh / float(bw))
        if aspect > 5.5:
            continue
        # Verify the stroke actually spans both left/right walls and top/bottom edges (not just an L-corner)
        roi = clean_lines[by : by + bh, bx : bx + bw]
        top_band = np.any(roi[: max(4, bh // 6), :] > 0)
        bot_band = np.any(roi[-max(4, bh // 6) :, :] > 0)
        left_band = np.any(roi[:, : max(4, bw // 6)] > 0)
        right_band = np.any(roi[:, -max(4, bw // 6) :] > 0)
        if top_band and bot_band and left_band and right_band:
            raw_boxes.append((bx, by, bw, bh))

    # Pass 2: Standard closed contours (backup for closed boxes touching other lines)
    contours, _ = cv2.findContours(clean_lines, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if area < (page_area * 0.004) or area > (page_area * 0.45):
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        bx, by, bw, bh = cv2.boundingRect(approx)
        rect_fill = area / float(bw * bh) if (bw * bh) > 0 else 0
        if 4 <= len(approx) <= 8 and bw > w * 0.09 and bh > h * 0.04 and rect_fill > 0.70:
            raw_boxes.append((bx, by, bw, bh))

    # Deduplicate overlapping detections of the same zone
    raw_boxes.sort(key=box_area, reverse=True)
    unique_boxes = []
    for bx, by, bw, bh in raw_boxes:
        is_dup = False
        for ux, uy, uw, uh in unique_boxes:
            inter_x1 = max(bx, ux)
            inter_y1 = max(by, uy)
            inter_x2 = min(bx + bw, ux + uw)
            inter_y2 = min(by + bh, uy + uh)
            if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                if inter_area / float(min(bw * bh, uw * uh)) > 0.50:
                    is_dup = True
                    break
        if not is_dup:
            unique_boxes.append((bx, by, bw, bh))

    # Remove any outer container box that wraps around inner boxes
    inner_boxes = []
    for i, (bx, by, bw, bh) in enumerate(unique_boxes):
        contains_others = 0
        for j, (ox, oy, ow, oh) in enumerate(unique_boxes):
            if i != j and bx <= ox + 10 and by <= oy + 10 and (bx + bw) >= (ox + ow - 10) and (by + bh) >= (oy + oh - 10):
                contains_others += 1
        if contains_others == 0:
            inner_boxes.append((bx, by, bw, bh))

    inner_boxes.sort(key=box_top_y)
    return inner_boxes


def extract_dimensions(pil_img):
    """Runs dual-pass OCR with handwritten S->5 recovery and unit priority."""
    enhanced = ImageEnhance.Contrast(pil_img).enhance(1.9)
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.6)
    img_np = np.array(enhanced)

    dimensions = []
    ocr_results, _ = ocr(img_np)
    if ocr_results:
        for item in ocr_results:
            bbox, text, _ = item
            raw_str = str(text).strip()
            has_unit = "m" in raw_str.lower()
            clean_text = (
                raw_str.replace(",", ".")
                .replace("O", "0")
                .replace("o", "0")
                .replace("S", "5")
                .replace("s", "5")
            )
            for match in re.finditer(r"(\d+\.?\d*)", clean_text):
                val = float(match.group(1))
                if val > 100:
                    val /= 1000.0
                if val == 1.0 and not has_unit:
                    continue
                if 0.2 <= val <= 50.0:
                    pts = np.array(bbox, dtype=float)
                    center_x, center_y = pts.mean(axis=0)
                    dimensions.append(
                        {
                            "val": round(val, 2),
                            "x": int(center_x),
                            "y": int(center_y),
                            "has_unit": has_unit,
                            "used": False,
                        }
                    )
    return dimensions


def pick_best_dim(candidates, target_x, target_y, default_val):
    unused = [d for d in candidates if not d["used"]]
    if not unused:
        return default_val
    best = min(
        unused,
        key=lambda d: ((d["x"] - target_x) ** 2 + (d["y"] - target_y) ** 2) ** 0.5
        - (60 if d["has_unit"] else 0),
    )
    best["used"] = True
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

    with st.spinner("Detecting pavement cafe zones and reading dimensions..."):
        img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        boxes = detect_cafe_rectangles(img_cv)
        dimensions = extract_dimensions(pil_img)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Uploaded Plan & Detection Overlay")
        debug_img = pil_img.copy()
        debug_draw = ImageDraw.Draw(debug_img)
        for idx, (bx, by, bw, bh) in enumerate(boxes):
            debug_draw.rectangle([bx, by, bx + bw, by + bh], outline="#16a34a", width=4)
            debug_draw.text((bx + 6, by + 6), f"Zone {idx + 1}", fill="#16a34a")
        for d in dimensions:
            dx, dy = d["x"], d["y"]
            debug_draw.ellipse([dx - 5, dy - 5, dx + 5, dy + 5], fill="#dc2626")
            debug_draw.text((dx + 8, dy - 8), f"{d['val']}m", fill="#dc2626")
        st.image(debug_img, use_container_width=True)
        st.caption("Green boxes = detected PC zones | Red dots = numbers read by OCR")

    adj_w, adj_d, has_remote, rem_w, rem_d, gap = 4.0, 1.5, False, 4.0, 1.5, 2.0

    if boxes and dimensions:
        bx, by, bw, bh = boxes[-1]  # Bottom box = Adjacent PC

        if len(boxes) >= 2:
            has_remote = True
            rx, ry, rw, rh = next(iter(boxes))  # Top box = Remote PC

            # 1. Claim Clear Gap strictly on the outer left or right margin between the two zones
            gap_candidates = [
                d
                for d in dimensions
                if (ry + rh - 25) <= d["y"] <= (by + 25)
                and (d["x"] <= min(bx, rx) + 25 or d["x"] >= max(bx + bw, rx + rw) - 25)
            ]
            gap = pick_best_dim(
                gap_candidates,
                min(bx, rx) - 40,
                (ry + rh + by) / 2,
                2.0,
            )

            # 2. Claim Adjacent PC width centered in the walkway above or below the bottom box
            adj_w_candidates = [
                d
                for d in dimensions
                if (bx + 20) <= d["x"] <= (bx + bw - 20)
                and (ry + rh - 20) <= d["y"] <= (by + bh + 120)
            ]
            adj_w = pick_best_dim(adj_w_candidates, bx + bw / 2, by - 40, 4.0)

            # 3. Claim Adjacent PC depth beside the bottom box's vertical midpoint
            adj_d_candidates = [
                d
                for d in dimensions
                if (by - 30) <= d["y"] <= (by + bh + 30)
                and (d["x"] <= bx + 40 or d["x"] >= bx + bw - 40)
            ]
            adj_d = pick_best_dim(adj_d_candidates, bx - 40, by + bh / 2, 1.5)

            # 4. Claim Remote PC width centered above or below the top box
            rem_w_candidates = [
                d
                for d in dimensions
                if (rx - 30) <= d["x"] <= (rx + rw + 30)
                and (ry - 140) <= d["y"] <= (ry + rh + 40)
            ]
            rem_w = pick_best_dim(rem_w_candidates, rx + rw / 2, ry - 45, 4.0)

            # 5. Claim Remote PC depth beside the top box's vertical midpoint
            rem_d_candidates = [
                d
                for d in dimensions
                if (ry - 30) <= d["y"] <= (ry + rh + 30)
                and (d["x"] <= rx + 40 or d["x"] >= rx + rw - 40)
            ]
            rem_d = pick_best_dim(rem_d_candidates, rx + rw + 45, ry + rh / 2, 1.5)

            # Visual orientation safeguard for wide zones
            if bw > bh * 1.3 and adj_d > adj_w:
                adj_w, adj_d = adj_d, adj_w
            if rw > rh * 1.3 and rem_d > rem_w:
                rem_w, rem_d = rem_d, rem_w
        else:
            adj_w_candidates = [
                d
                for d in dimensions
                if (bx - 30) <= d["x"] <= (bx + bw + 30)
                and (by - 140) <= d["y"] <= (by + bh + 140)
            ]
            adj_w = pick_best_dim(adj_w_candidates, bx + bw / 2, by - 40, 4.0)

            adj_d_candidates = [
                d
                for d in dimensions
                if (by - 30) <= d["y"] <= (by + bh + 30)
                and (d["x"] <= bx + 40 or d["x"] >= bx + bw - 40)
            ]
            adj_d = pick_best_dim(adj_d_candidates, bx - 40, by + bh / 2, 1.5)

            if bw > bh * 1.3 and adj_d > adj_w:
                adj_w, adj_d = adj_d, adj_w

    with col2:
        st.subheader("Dimensioned Polygon Output")
        detected_vals = [f"{d['val']}m" for d in dimensions]
        st.caption(
            f"Detected {len(boxes)} zone(s) | Numbers read:"
            f" {', '.join(detected_vals) if detected_vals else 'None'}"
        )

        c1, c2 = st.columns(2)
        adj_w = c1.number_input(
            "Adjacent PC Width (m)", value=float(adj_w), min_value=0.1, step=0.1
        )
        adj_d = c2.number_input(
            "Adjacent PC Depth (m)", value=float(adj_d), min_value=0.1, step=0.1
        )

        has_remote = st.checkbox("Include Remote PC", value=has_remote)
        if has_remote:
            c3, c4, c5 = st.columns(3)
            rem_w = c3.number_input(
                "Remote Width (m)", value=float(rem_w), min_value=0.1, step=0.1
            )
            rem_d = c4.number_input(
                "Remote Depth (m)", value=float(rem_d), min_value=0.1, step=0.1
            )
            gap = c5.number_input(
                "Separation Gap (m)", value=float(gap), min_value=0.1, step=0.1
            )

        total_w_m = max(adj_w, rem_w) if has_remote else adj_w
        total_h_m = (adj_d + gap + rem_d) if has_remote else adj_d
        scale = min(520 / total_w_m, 520 / total_h_m)

        out = Image.new("RGB", (800, 800), "white")
        draw = ImageDraw.Draw(out)

        ax0 = 120
        ay1 = 680
        ay0 = ay1 - int(adj_d * scale)
        ax1 = ax0 + int(adj_w * scale)

        draw.line([(60, ay1), (740, ay1)], fill="black", width=4)
        draw.text((65, ay1 + 12), "PREMISES / BUILDING FRONTAGE", fill="black")

        draw.rectangle([ax0, ay0, ax1, ay1], outline="black", fill="#dbeafe", width=3)
        adj_area = round(adj_w * adj_d, 2)
        adj_label = f"Adjacent PC ({adj_area} sq.m)"
        if (ax1 - ax0) >= 150 and (ay1 - ay0) >= 35:
            draw.text((ax0 + 10, (ay0 + ay1) // 2 - 6), adj_label, fill="black")
        else:
            draw.text((ax0, ay0 - 18), adj_label, fill="black")

        draw_dimension_arrow(draw, (ax0, ay1), (ax1, ay1), f"{adj_w}m", offset=35)
        draw_dimension_arrow(draw, (ax1, ay0), (ax1, ay1), f"{adj_d}m", offset=25)

        if has_remote:
            ry1 = ay0 - int(gap * scale)
            ry0 = ry1 - int(rem_d * scale)
            rx1 = ax0 + int(rem_w * scale)

            draw.rectangle([ax0, ry0, rx1, ry1], outline="black", fill="#fef3c7", width=3)
            rem_area = round(rem_w * rem_d, 2)
            rem_label = f"Remote PC ({rem_area} sq.m)"
            if (rx1 - ax0) >= 150 and (ry1 - ry0) >= 35:
                draw.text((ax0 + 10, (ry0 + ry1) // 2 - 6), rem_label, fill="black")
            else:
                draw.text((ax0, ry1 + 6), rem_label, fill="black")

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
                (60, 30), f"Total Pavement Cafe Area: {total_area} sq.m", fill="black"
            )
        else:
            draw.text(
                (60, 30), f"Total Pavement Cafe Area: {adj_area} sq.m", fill="black"
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
