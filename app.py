import io
import re
import cv2
import fitz
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageEnhance
from rapidocr_onnxruntime import RapidOCR

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
    h, w = img_cv.shape[:2]
    page_area = h * w
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Normalize uneven phone shadow / paper lighting
    bg = cv2.GaussianBlur(gray, (51, 51), 0)
    norm = cv2.divide(gray, bg, scale=255)
    blurred = cv2.GaussianBlur(norm, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 12
    )

    # Wide closing bridges customer access openings in boundary walls
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 45))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw_boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < (page_area * 0.002) or area > (page_area * 0.55):
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        rect_fill = area / float(bw * bh) if (bw * bh) > 0 else 0
        if bw > 40 and bh > 25 and rect_fill > 0.55:
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

    unique_boxes.sort(key=box_top_y)
    return unique_boxes


def extract_dimensions(pil_img):
    gray_pil = pil_img.convert("L")
    enhanced = ImageEnhance.Contrast(gray_pil).enhance(2.2)
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(2.0)
    img_np = np.array(enhanced.convert("RGB"))

    dimensions = []
    ocr_results, _ = ocr(img_np)
    if ocr_results:
        for item in ocr_results:
            bbox, text, _ = item
            clean_text = (
                str(text)
                .replace(",", ".")
                .replace("O", "0")
                .replace("o", "0")
                .replace("S", "5")
                .replace("s", "5")
            )
            for match in re.finditer(r"(\d+\.?\d*)", clean_text):
                val = float(match.group(1))
                if val > 100:
                    val /= 1000.0
                if 0.2 <= val <= 50.0:
                    center_x, center_y = np.array(bbox, dtype=float).mean(axis=0)
                    dimensions.append(
                        {"val": round(val, 2), "x": int(center_x), "y": int(center_y), "used": False}
                    )
    return dimensions


def pick_best_dimension(dimensions, target_x, target_y, x_weight=1.0, y_weight=1.0, filter_fn=None):
    candidates = [d for d in dimensions if not d["used"] and (filter_fn is None or filter_fn(d))]
    if not candidates:
        candidates = [d for d in dimensions if filter_fn is None or filter_fn(d)]
    if not candidates:
        candidates = dimensions
    if not candidates:
        return None
    best = min(
        candidates,
        key=lambda d: x_weight * (d["x"] - target_x) ** 2 + y_weight * (d["y"] - target_y) ** 2,
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
        if len(boxes) >= 2:
            has_remote = True
            rx, ry, rw, rh = boxes[0]
            bx, by, bw, bh = boxes[-1]

            # 1. Claim Gap dimension first (sits vertically between Remote bottom and Adjacent top)
            gap_mid_y = (ry + rh + by) / 2.0
            gap_val = pick_best_dimension(
                dimensions,
                target_x=rx,
                target_y=gap_mid_y,
                x_weight=0.5,
                y_weight=2.0,
                filter_fn=lambda d: (ry + rh - 20) <= d["y"] <= (by + 20) and d["x"] < (bx + bw * 0.4),
            )
            if gap_val is not None:
                gap = gap_val

            # 2. Remote PC Width (centered horizontally near Remote box center, above or below it)
            rem_w_val = pick_best_dimension(
                dimensions,
                target_x=rx + rw / 2.0,
                target_y=ry - 25,
                x_weight=2.5,
                y_weight=1.0,
                filter_fn=lambda d: abs(d["x"] - (rx + rw / 2.0)) < max(rw * 0.8, 80) and d["y"] < (ry + rh * 0.5),
            )
            if rem_w_val is not None:
                rem_w = rem_w_val

            # 3. Remote PC Depth (centered vertically near Remote box middle, to left or right)
            rem_d_val = pick_best_dimension(
                dimensions,
                target_x=rx + rw + 30,
                target_y=ry + rh / 2.0,
                x_weight=0.5,
                y_weight=2.5,
                filter_fn=lambda d: abs(d["y"] - (ry + rh / 2.0)) < max(rh * 0.7, 60),
            )
            if rem_d_val is not None:
                rem_d = rem_d_val

            # 4. Adjacent PC Width (centered horizontally near Adjacent box center)
            adj_w_val = pick_best_dimension(
                dimensions,
                target_x=bx + bw / 2.0,
                target_y=by - 25,
                x_weight=2.5,
                y_weight=1.0,
                filter_fn=lambda d: abs(d["x"] - (bx + bw / 2.0)) < max(bw * 0.6, 100),
            )
            if adj_w_val is not None:
                adj_w = adj_w_val

            # 5. Adjacent PC Depth (centered vertically near Adjacent box middle)
            adj_d_val = pick_best_dimension(
                dimensions,
                target_x=bx - 30,
                target_y=by + bh / 2.0,
                x_weight=0.5,
                y_weight=2.5,
                filter_fn=lambda d: abs(d["y"] - (by + bh / 2.0)) < max(bh * 0.7, 60),
            )
            if adj_d_val is not None:
                adj_d = adj_d_val
        else:
            bx, by, bw, bh = boxes[0]
            adj_w_val = pick_best_dimension(
                dimensions,
                target_x=bx + bw / 2.0,
                target_y=by - 25,
                x_weight=2.5,
                y_weight=1.0,
            )
            if adj_w_val is not None:
                adj_w = adj_w_val

            adj_d_val = pick_best_dimension(
                dimensions,
                target_x=bx - 30,
                target_y=by + bh / 2.0,
                x_weight=0.5,
                y_weight=2.5,
            )
            if adj_d_val is not None:
                adj_d = adj_d_val

    with col2:
        st.subheader("Dimensioned Polygon Output")
        detected_vals = [f"{d['val']}m" for d in dimensions]
        st.caption(
            f"Detected {len(boxes)} zone(s) | Numbers read:"
            f" {', '.join(detected_vals) if detected_vals else 'None'}"
        )

        c1, c2 = st.columns(2)
        adj_w = c1.number_input("Adjacent PC Width (m)", value=float(adj_w), min_value=0.1, step=0.1)
        adj_d = c2.number_input("Adjacent PC Depth (m)", value=float(adj_d), min_value=0.1, step=0.1)

        has_remote = st.checkbox("Include Remote PC", value=has_remote)
        if has_remote:
            c3, c4, c5 = st.columns(3)
            rem_w = c3.number_input("Remote Width (m)", value=float(rem_w), min_value=0.1, step=0.1)
            rem_d = c4.number_input("Remote Depth (m)", value=float(rem_d), min_value=0.1, step=0.1)
            gap = c5.number_input("Separation Gap (m)", value=float(gap), min_value=0.1, step=0.1)

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

        # Solid enclosed Adjacent PC polygon
        draw.rectangle([ax0, ay0, ax1, ay1], outline="black", fill="#dbeafe", width=3)
        adj_area = round(adj_w * adj_d, 2)
        draw.text((ax0 + 10, (ay0 + ay1) // 2 - 8), f"Adjacent PC ({adj_area} m²)", fill="black")

        draw_dimension_arrow(draw, (ax0, ay1), (ax1, ay1), f"{adj_w}m", offset=35)
        draw_dimension_arrow(draw, (ax1, ay0), (ax1, ay1), f"{adj_d}m", offset=25)

        if has_remote:
            ry1 = ay0 - int(gap * scale)
            ry0 = ry1 - int(rem_d * scale)
            rx1 = ax0 + int(rem_w * scale)

            # Solid enclosed Remote PC polygon
            draw.rectangle([ax0, ry0, rx1, ry1], outline="black", fill="#fef3c7", width=3)
            rem_area = round(rem_w * rem_d, 2)
            draw.text((ax0 + 10, (ry0 + ry1) // 2 - 8), f"Remote PC ({rem_area} m²)", fill="black")

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
