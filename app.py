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
        text_y = yd + 6 if offset > 0 else yd - 18
        draw.text(((x1 + x2) // 2 - 16, text_y), text, fill=color)
    else:
        xd = x1 + offset
        draw.line([(x1, y1), (xd, y1)], fill="#64748b", width=1)
        draw.line([(x2, y2), (xd, y2)], fill="#64748b", width=1)
        draw.line([(xd, y1), (xd, y2)], fill=color, width=2)
        draw.line([(xd - 4, y1 + 4), (xd + 4, y1 - 4)], fill=color, width=2)
        draw.line([(xd - 4, y2 + 4), (xd + 4, y2 - 4)], fill=color, width=2)
        draw.text((xd + 8, (y1 + y2) // 2 - 6), text, fill=color)


def normalize_paper(img_cv):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    bg = cv2.GaussianBlur(gray, (55, 55), 0)
    norm = cv2.divide(gray, bg, scale=255)
    return norm


def box_area(box):
    bx, by, bw, bh = box
    return bw * bh


def box_top_y(box):
    bx, by, bw, bh = box
    return by


def detect_cafe_rectangles(img_cv):
    h, w = img_cv.shape[:2]
    page_area = h * w
    norm = normalize_paper(img_cv)
    blurred = cv2.GaussianBlur(norm, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 12
    )

    # Multi-pass morphological closing:
    # Pass 1 catches closed smaller boxes (Remote PC)
    # Pass 2 bridges wide customer access gaps (Adjacent PC)
    kernels = [
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (55, 7)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 55)),
    ]

    raw_boxes = []
    for k in kernels:
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k)
        contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < (page_area * 0.002) or area > (page_area * 0.45):
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            bx, by, bw, bh = cv2.boundingRect(approx)
            rect_fill = area / float(bw * bh) if (bw * bh) > 0 else 0
            if 4 <= len(approx) <= 8 and bw > 35 and bh > 25 and rect_fill > 0.65:
                raw_boxes.append((bx, by, bw, bh))

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
                if inter_area > 0.5 * min(bw * bh, uw * uh):
                    is_dup = True
                    break
        if not is_dup:
            unique_boxes.append((bx, by, bw, bh))

    # Remove any outer container box that wraps around other detected zones
    inner_boxes = []
    for i, (bx, by, bw, bh) in enumerate(unique_boxes):
        wraps_others = False
        for j, (ox, oy, ow, oh) in enumerate(unique_boxes):
            if i != j and bx <= ox and by <= oy and (bx + bw) >= (ox + ow) and (by + bh) >= (oy + oh):
                wraps_others = True
                break
        if not wraps_others:
            inner_boxes.append((bx, by, bw, bh))

    inner_boxes.sort(key=box_top_y)
    return inner_boxes


def extract_dimensions(pil_img, img_cv):
    norm = normalize_paper(img_cv)
    norm_rgb = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)
    enhanced = ImageEnhance.Contrast(Image.fromarray(norm_rgb)).enhance(1.6)
    img_np = np.array(enhanced)

    dimensions = []
    ocr_results, _ = ocr(img_np)
    if ocr_results:
        for item in ocr_results:
            bbox, text, conf = item
            pts = np.array(bbox, dtype=float)
            min_x, min_y = pts.min(axis=0)
            max_x, max_y = pts.max(axis=0)
            box_w = max_x - min_x
            box_h = max_y - min_y

            # Reject tall, skinny vertical arrow shafts misread as digits (e.g. "0.6" or "1.6")
            if box_w > 0 and (box_h / box_w) > 2.0:
                continue

            raw_str = str(text).strip()
            clean_text = (
                raw_str.replace(",", ".")
                .replace("O", "0")
                .replace("o", "0")
                .replace("S", "5")
                .replace("s", "5")
            )
            has_unit = "m" in clean_text.lower()

            for match in re.finditer(r"(\d+\.?\d*)", clean_text):
                token = match.group(1)
                val = float(token)
                if val > 100:
                    val /= 1000.0
                # If no explicit decimal point was written, snap noisy OCR decimals to integer
                if "." not in token:
                    val = round(val)
                # Filter out stray single-digit "1" without a unit unless valid decimal
                if val == 1.0 and not has_unit and "." not in token:
                    continue
                if 0.5 <= val <= 50.0:
                    center_x, center_y = pts.mean(axis=0)
                    dimensions.append({
                        "val": round(val, 2),
                        "x": int(center_x),
                        "y": int(center_y),
                        "has_unit": has_unit,
                        "used": False,
                    })
    return dimensions


def claim_closest(dimensions, target_x, target_y, max_dist=350, prefer_unit=True):
    candidates = [
        d for d in dimensions
        if not d["used"] and ((d["x"] - target_x) ** 2 + (d["y"] - target_y) ** 2) <= max_dist ** 2
    ]
    if not candidates:
        return None
    best = min(
        candidates,
        key=lambda d: ((d["x"] - target_x) ** 2 + (d["y"] - target_y) ** 2) * (0.6 if (prefer_unit and d["has_unit"]) else 1.0),
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
        dimensions = extract_dimensions(pil_img, img_cv)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Uploaded Plan & Detected Zones")
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
        # Bottom box = Adjacent PC
        bx, by, bw, bh = boxes[-1]

        if len(boxes) >= 2:
            has_remote = True
            rx, ry, rw, rh = next(iter(boxes))

            # 1. Adjacent PC width: centered horizontally over the bottom box (inside walkway)
            val_adj_w = claim_closest(dimensions, bx + bw / 2, by - 35)
            if val_adj_w is not None:
                adj_w = val_adj_w

            # 2. Clear gap: located along the left/right outer margin between the two boxes
            gap_mid_y = (ry + rh + by) / 2
            left_margin_x = min(bx, rx) - 35
            val_gap = claim_closest(dimensions, left_margin_x, gap_mid_y)
            if val_gap is not None:
                gap = val_gap

            # 3. Remote PC width: centered above (or below) the top box
            val_rem_w = claim_closest(dimensions, rx + rw / 2, ry - 35)
            if val_rem_w is not None:
                rem_w = val_rem_w

            # 4. Remote PC depth: beside the vertical midpoint of the top box
            rem_side_dims = [
                d for d in dimensions
                if not d["used"] and abs(d["y"] - (ry + rh / 2)) < rh * 0.75
            ]
            if rem_side_dims:
                best_rem_d = min(
                    rem_side_dims,
                    key=lambda d: min(abs(d["x"] - (rx + rw)), abs(d["x"] - rx)),
                )
                best_rem_d["used"] = True
                rem_d = best_rem_d["val"]

            # 5. Adjacent PC depth: beside the vertical midpoint of the bottom box
            adj_side_dims = [
                d for d in dimensions
                if not d["used"] and abs(d["y"] - (by + bh / 2)) < bh * 0.75
            ]
            if adj_side_dims:
                best_adj_d = min(
                    adj_side_dims,
                    key=lambda d: min(abs(d["x"] - bx), abs(d["x"] - (bx + bw))),
                )
                best_adj_d["used"] = True
                adj_d = best_adj_d["val"]
        else:
            val_adj_w = claim_closest(dimensions, bx + bw / 2, by - 35)
            if val_adj_w is not None:
                adj_w = val_adj_w
            adj_side_dims = [
                d for d in dimensions
                if not d["used"] and abs(d["y"] - (by + bh / 2)) < bh * 0.75
            ]
            if adj_side_dims:
                best_adj_d = min(
                    adj_side_dims,
                    key=lambda d: min(abs(d["x"] - bx), abs(d["x"] - (bx + bw))),
                )
                best_adj_d["used"] = True
                adj_d = best_adj_d["val"]

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
        scale = min(500 / total_w_m, 500 / total_h_m)

        out = Image.new("RGB", (800, 800), "white")
        draw = ImageDraw.Draw(out)

        ax0 = 120
        ay1 = 680
        ay0 = ay1 - int(adj_d * scale)
        ax1 = ax0 + int(adj_w * scale)

        draw.line([(60, ay1), (740, ay1)], fill="black", width=4)
        draw.text((65, ay1 + 12), "PREMISES / BUILDING FRONTAGE", fill="black")

        # Solid, fully enclosed Adjacent PC polygon
        draw.rectangle([ax0, ay0, ax1, ay1], outline="black", fill="#dbeafe", width=3)
        adj_area = round(adj_w * adj_d, 2)
        adj_label = f"Adjacent PC ({adj_area} sq.m)"
        if (ax1 - ax0) > 150 and (ay1 - ay0) > 35:
            draw.text((ax0 + 10, (ay0 + ay1) // 2 - 6), adj_label, fill="black")
        else:
            draw.text((ax0, ay0 - 18), adj_label, fill="black")

        draw_dimension_arrow(draw, (ax0, ay1), (ax1, ay1), f"{adj_w}m", offset=38)
        draw_dimension_arrow(draw, (ax1, ay0), (ax1, ay1), f"{adj_d}m", offset=25)

        if has_remote:
            ry1 = ay0 - int(gap * scale)
            ry0 = ry1 - int(rem_d * scale)
            rx1 = ax0 + int(rem_w * scale)

            # Solid, fully enclosed Remote PC polygon
            draw.rectangle([ax0, ry0, rx1, ry1], outline="black", fill="#fef3c7", width=3)
            rem_area = round(rem_w * rem_d, 2)
            rem_label = f"Remote PC ({rem_area} sq.m)"
            if (rx1 - ax0) > 150 and (ry1 - ry0) > 35:
                draw.text((ax0 + 10, (ry0 + ry1) // 2 - 6), rem_label, fill="black")
            else:
                draw.text((rx1 + 10, ry0 - 18), rem_label, fill="black")

            draw_dimension_arrow(draw, (ax0, ry0), (rx1, ry0), f"{rem_w}m", offset=-28)
            draw_dimension_arrow(draw, (rx1, ry0), (rx1, ry1), f"{rem_d}m", offset=25)

            gap_x = ax0 + int(min(adj_w, rem_w) * scale * 0.5)
            draw.line([(gap_x, ry1), (gap_x, ay0)], fill="#dc2626", width=2)
            draw.text((gap_x + 8, (ry1 + ay0) // 2 - 6), f"Clear Gap: {gap}m", fill="#dc2626")

            total_area = round(adj_area + rem_area, 2)
            draw.text((60, 30), f"Total Pavement Cafe Area: {total_area} sq.m", fill="black")
        else:
            draw.text((60, 30), f"Total Pavement Cafe Area: {adj_area} sq.m", fill="black")

        st.image(out, use_container_width=True)

        buf = io.BytesIO()
        out.save(buf, format="JPEG", quality=95)
        st.download_button(
            label="Download Dimensioned JPEG",
            data=buf.getvalue(),
            file_name="pavement_cafe_plan.jpg",
            mime="image/jpeg",
        )
