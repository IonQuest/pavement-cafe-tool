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
    """Multi-pass detector: finds both closed boxes and open-entrance boxes."""
    h, w = img_cv.shape[:2]
    page_area = h * w
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Normalize warm phone-photo lighting
    bg = cv2.medianBlur(gray, 51)
    norm = cv2.divide(gray, bg, scale=255)
    blurred = cv2.GaussianBlur(norm, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
    )

    raw_boxes = []
    # Pass 1: Light close for intact boxes (e.g. Remote PC)
    # Pass 2: Wide close to bridge customer access gaps (e.g. Adjacent PC)
    kernels = [
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (45, 7)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 45)),
    ]

    for kernel in kernels:
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < (page_area * 0.0012) or area > (page_area * 0.45):
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            bx, by, bw, bh = cv2.boundingRect(approx)
            rect_fill = area / float(bw * bh) if (bw * bh) > 0 else 0
            if 4 <= len(approx) <= 8 and bw > 30 and bh > 20 and rect_fill > 0.65:
                raw_boxes.append((bx, by, bw, bh))

    raw_boxes.sort(key=box_area, reverse=True)
    unique_boxes = []
    for bx, by, bw, bh in raw_boxes:
        is_dup = False
        for ux, uy, uw, uh in unique_boxes:
            inter_x0 = max(bx, ux)
            inter_y0 = max(by, uy)
            inter_x1 = min(bx + bw, ux + uw)
            inter_y1 = min(by + bh, uy + uh)
            if inter_x1 > inter_x0 and inter_y1 > inter_y0:
                inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
                if inter_area > 0.45 * min(bw * bh, uw * uh):
                    is_dup = True
                    break
        if not is_dup:
            unique_boxes.append((bx, by, bw, bh))

    inner_boxes = []
    for i, (bx, by, bw, bh) in enumerate(unique_boxes):
        contains_others = 0
        for j, (ox, oy, ow, oh) in enumerate(unique_boxes):
            if (
                i != j
                and bx <= ox
                and by <= oy
                and (bx + bw) >= (ox + ow)
                and (by + bh) >= (oy + oh)
            ):
                contains_others += 1
        if contains_others == 0:
            inner_boxes.append((bx, by, bw, bh))

    inner_boxes.sort(key=box_top_y)
    return inner_boxes


def extract_dimensions(pil_img):
    """Dual-pass OCR that blocks vertical line '1' hallucinations and prioritizes 'm'-labeled values."""
    gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    bg = cv2.medianBlur(gray, 51)
    norm_rgb = cv2.cvtColor(cv2.divide(gray, bg, scale=255), cv2.COLOR_GRAY2RGB)

    boosted = ImageEnhance.Contrast(pil_img).enhance(1.9)
    boosted = ImageEnhance.Sharpness(boosted).enhance(1.6)

    dimensions = []
    for img_variant in (np.array(boosted), norm_rgb):
        ocr_results, _ = ocr(img_variant)
        if not ocr_results:
            continue
        for item in ocr_results:
            bbox, text, _ = item
            raw_str = str(text).strip()
            clean_text = raw_str.replace(",", ".").replace("O", "0").replace("o", "0")

            pts = np.array(bbox, dtype=float)
            box_w = float(pts[:, 0].max() - pts[:, 0].min())
            box_h = float(pts[:, 1].max() - pts[:, 1].min())
            # Skip tall skinny vertical arrow shaft artifacts misread as digits
            if box_w > 0 and (box_h / box_w) > 2.3:
                continue

            has_unit = bool(re.search(r"[mM]", clean_text))
            for match in re.finditer(r"(\d+\.?\d*)", clean_text):
                num_str = match.group(1)
                val = float(num_str)
                if val > 100:
                    val /= 1000.0
                # Block standalone '1' or '1.0' without an 'm' unit (almost always a vertical arrow line)
                if val == 1.0 and not has_unit:
                    continue
                if 0.3 <= val <= 50.0:
                    cx, cy = pts.mean(axis=0)
                    is_dup = any(
                        abs(d["x"] - cx) < 25 and abs(d["y"] - cy) < 25
                        for d in dimensions
                    )
                    if not is_dup:
                        dimensions.append(
                            {
                                "val": round(val, 2),
                                "x": int(cx),
                                "y": int(cy),
                                "has_unit": has_unit,
                            }
                        )
    return dimensions


def pick_best_dimension(candidates, target_x, target_y, x_weight=1.0, y_weight=1.0):
    """Picks closest candidate, giving a strong preference to labels with an explicit 'm' unit."""
    def score(d):
        dist = ((d["x"] - target_x) * x_weight) ** 2 + ((d["y"] - target_y) * y_weight) ** 2
        unit_bonus = 0.25 if d.get("has_unit") else 1.0
        return dist * unit_bonus

    return min(candidates, key=score)


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
            debug_draw.text((bx + 5, by + 5), f"Zone {idx + 1}", fill="#16a34a")
        for d in dimensions:
            dx, dy = d["x"], d["y"]
            debug_draw.ellipse([dx - 5, dy - 5, dx + 5, dy + 5], fill="#dc2626")
            debug_draw.text((dx + 8, dy - 8), f"{d['val']}m", fill="#dc2626")
        st.image(debug_img, use_container_width=True)
        st.caption("Green boxes = detected PC zones | Red dots = dimensions read")

    adj_w, adj_d, has_remote, rem_w, rem_d, gap = 5.0, 2.0, False, 2.0, 2.0, 2.0
    available = list(dimensions)

    if boxes and available:
        bx, by, bw, bh = boxes[-1]  # Bottom zone = Adjacent PC

        if len(boxes) >= 2:
            has_remote = True
            rx, ry, rw, rh = next(iter(boxes))  # Top zone = Remote PC
            gap_top = ry + rh
            gap_bot = by

            # 1. Adjacent PC Width: centered horizontally over bottom box inside walkway gap
            adj_w_cands = [
                d for d in available
                if abs(d["x"] - (bx + bw / 2)) < bw * 0.65 and (gap_top - 10) <= d["y"] <= (by + bh + 60)
            ] or available
            chosen = pick_best_dimension(adj_w_cands, bx + bw / 2, by, x_weight=1.0, y_weight=1.5)
            adj_w = chosen["val"]
            if len(available) > 1:
                available.remove(chosen)

            # 2. Clear Separation Gap: in outer left/right margin vertically between the two zones
            left_bound = min(bx, rx)
            right_bound = max(bx + bw, rx + rw)
            gap_cands = [
                d for d in available
                if (gap_top - 15) <= d["y"] <= (gap_bot + 15)
                and (d["x"] <= left_bound + 30 or d["x"] >= right_bound - 30)
            ] or available
            chosen = pick_best_dimension(gap_cands, left_bound, (gap_top + gap_bot) / 2, x_weight=0.5, y_weight=2.0)
            gap = chosen["val"]
            if len(available) > 1:
                available.remove(chosen)

            # 3. Adjacent PC Depth: alongside vertical midpoint of bottom box
            adj_d_cands = [
                d for d in available
                if abs(d["y"] - (by + bh / 2)) < bh * 0.85
            ] or available
            chosen = pick_best_dimension(adj_d_cands, bx, by + bh / 2, x_weight=0.4, y_weight=2.0)
            adj_d = chosen["val"]
            if len(available) > 1:
                available.remove(chosen)

            # 4. Remote PC Width: centered above/below top box
            rem_w_cands = [
                d for d in available
                if abs(d["x"] - (rx + rw / 2)) < rw * 0.9 and d["y"] <= gap_top + 25
            ] or available
            chosen = pick_best_dimension(rem_w_cands, rx + rw / 2, ry, x_weight=1.0, y_weight=1.5)
            rem_w = chosen["val"]
            if len(available) > 1:
                available.remove(chosen)

            # 5. Remote PC Depth: alongside vertical midpoint of top box
            rem_d_cands = [
                d for d in available
                if abs(d["y"] - (ry + rh / 2)) < rh * 0.95
            ] or available
            chosen = pick_best_dimension(rem_d_cands, rx + rw, ry + rh / 2, x_weight=0.4, y_weight=2.0)
            rem_d = chosen["val"]
        else:
            w_cands = [d for d in available if abs(d["x"] - (bx + bw / 2)) < bw * 0.7] or available
            chosen = pick_best_dimension(w_cands, bx + bw / 2, by)
            adj_w = chosen["val"]
            if len(available) > 1:
                available.remove(chosen)
            chosen = pick_best_dimension(available, bx, by + bh / 2, x_weight=0.4, y_weight=2.0)
            adj_d = chosen["val"]

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
        scale = min(500 / total_w_m, 500 / total_h_m)

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
        adj_label = f"Adjacent PC ({adj_area} sq.m)"
        if (ax1 - ax0) > 140:
            draw.text((ax0 + 10, (ay0 + ay1) // 2 - 6), adj_label, fill="black")
        else:
            draw.text((ax0, ay0 - 18), adj_label, fill="black")

        draw_dimension_arrow(draw, (ax0, ay1), (ax1, ay1), f"{adj_w}m", offset=35)
        draw_dimension_arrow(draw, (ax1, ay0), (ax1, ay1), f"{adj_d}m", offset=25)

        if has_remote:
            ry1 = ay0 - int(gap * scale)
            ry0 = ry1 - int(rem_d * scale)
            rx1 = ax0 + int(rem_w * scale)

            # Solid enclosed Remote PC polygon
            draw.rectangle([ax0, ry0, rx1, ry1], outline="black", fill="#fef3c7", width=3)
            rem_area = round(rem_w * rem_d, 2)
            rem_label = f"Remote PC ({rem_area} sq.m)"
            if (rx1 - ax0) > 140:
                draw.text((ax0 + 10, (ry0 + ry1) // 2 - 6), rem_label, fill="black")
            else:
                draw.text((ax0, ry1 + 6), rem_label, fill="black")

            draw_dimension_arrow(draw, (ax0, ry0), (rx1, ry0), f"{rem_w}m", offset=-25)
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
