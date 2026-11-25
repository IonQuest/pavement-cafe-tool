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


def detect_cafe_rectangles(img_cv, dimensions):
    """Finds actual pavement cafe zones while filtering out dimension arrow boxes."""
    h, w = img_cv.shape[:2]
    page_area = h * w
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Normalize uneven paper lighting so pencil/pen strokes have uniform contrast
    bg = cv2.GaussianBlur(gray, (51, 51), 0)
    norm = cv2.divide(gray, bg, scale=255)
    blurred = cv2.GaussianBlur(norm, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
    )

    # Pass 1: Light closing to detect intact zones like the Remote PC
    k_light = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed_light = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k_light)

    # Pass 2: Horizontal-only closing to bridge customer entrance openings without merging vertically
    k_horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (max(35, int(w * 0.14)), 3))
    closed_horiz = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k_horiz)

    raw_boxes = []
    for mask in (closed_light, closed_horiz):
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < (page_area * 0.004) or area > (page_area * 0.45):
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            bx, by, bw, bh = cv2.boundingRect(approx)
            rect_fill = area / float(bw * bh) if (bw * bh) > 0 else 0
            # Require reasonable height (at least 7% of page height) so thin horizontal arrow boxes are excluded
            if bw > int(w * 0.10) and bh > int(h * 0.07) and rect_fill > 0.65:
                # Discard any box that contains an OCR dimension dot inside its interior (walkway/arrow frame)
                has_text_inside = any(
                    (bx + 15) < d["x"] < (bx + bw - 15) and (by + 15) < d["y"] < (by + bh - 15)
                    for d in dimensions
                )
                if not has_text_inside:
                    raw_boxes.append((bx, by, bw, bh))

    # Deduplicate overlapping boxes
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
                if inter_area > 0.50 * min(bw * bh, uw * uh):
                    is_dup = True
                    break
        if not is_dup:
            unique_boxes.append((bx, by, bw, bh))

    # Remove any outer container box that encloses smaller boxes
    inner_boxes = []
    for i, (bx, by, bw, bh) in enumerate(unique_boxes):
        contains_others = 0
        for j, (ox, oy, ow, oh) in enumerate(unique_boxes):
            if i != j and bx <= ox + 10 and by <= oy + 10 and (bx + bw) >= (ox + ow - 10) and (by + bh) >= (oy + oh - 10):
                contains_others += 1
        if contains_others == 0:
            inner_boxes.append((bx, by, bw, bh))

    # Merge split horizontal halves on the same row (e.g., C-shaped Adjacent PC with wide entrance)
    inner_boxes.sort(key=box_top_y)
    merged_boxes = []
    for bx, by, bw, bh in inner_boxes:
        merged = False
        for idx, (mx, my, mw, mh) in enumerate(merged_boxes):
            if abs(by - my) < 35 and abs((by + bh) - (my + mh)) < 35:
                nx = min(bx, mx)
                ny = min(by, my)
                nw = max(bx + bw, mx + mw) - nx
                nh = max(by + bh, my + mh) - ny
                # Only merge if the merged box still has no OCR text inside it
                has_text_inside = any(
                    (nx + 15) < d["x"] < (nx + nw - 15) and (ny + 15) < d["y"] < (ny + nh - 15)
                    for d in dimensions
                )
                if not has_text_inside:
                    merged_boxes[idx] = (nx, ny, nw, nh)
                    merged = True
                    break
        if not merged:
            merged_boxes.append((bx, by, bw, bh))

    merged_boxes.sort(key=box_top_y)
    return merged_boxes


def extract_dimensions(pil_img):
    """Dual-pass OCR to reliably read handwritten dimensions."""
    dimensions = []

    enhanced = ImageEnhance.Contrast(pil_img).enhance(1.9)
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.6)

    gray = np.array(pil_img.convert("L"))
    bg = cv2.GaussianBlur(gray, (51, 51), 0)
    norm = cv2.divide(gray, bg, scale=255)
    norm_rgb = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)

    for img_np in (np.array(enhanced), norm_rgb):
        ocr_results, _ = ocr(img_np)
        if not ocr_results:
            continue
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
            pts = np.array(bbox, dtype=float)
            box_w = pts[:, 0].max() - pts[:, 0].min()
            box_h = pts[:, 1].max() - pts[:, 1].min()
            if box_w > 0 and (box_h / box_w) > 2.2:
                continue

            for match in re.finditer(r"(\d+\.?\d*)", clean_text):
                raw_num = match.group(1)
                val = float(raw_num)
                if val > 100:
                    val /= 1000.0
                if "." not in raw_num:
                    nearest_int = round(val)
                    if abs(val - nearest_int) <= 0.45:
                        val = float(nearest_int)
                if not has_unit and val == 1.0:
                    continue
                if 0.4 <= val <= 50.0:
                    center_x, center_y = pts.mean(axis=0)
                    cx, cy = int(center_x), int(center_y)
                    is_dup = any(
                        abs(d["x"] - cx) < 25 and abs(d["y"] - cy) < 25
                        for d in dimensions
                    )
                    if not is_dup:
                        dimensions.append({
                            "val": round(val, 2),
                            "x": cx,
                            "y": cy,
                            "has_unit": has_unit,
                            "used": False,
                        })
    return dimensions


def claim_best(candidates, target_x, target_y):
    unused = [d for d in candidates if not d["used"]]
    if not unused:
        return None
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
        dimensions = extract_dimensions(pil_img)
        boxes = detect_cafe_rectangles(img_cv, dimensions)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Uploaded Plan & Detection Check")
        debug_img = pil_img.copy()
        debug_draw = ImageDraw.Draw(debug_img)
        for idx, (bx, by, bw, bh) in enumerate(boxes):
            debug_draw.rectangle([bx, by, bx + bw, by + bh], outline="#16a34a", width=4)
        for d in dimensions:
            dx, dy = d["x"], d["y"]
            debug_draw.ellipse([dx - 5, dy - 5, dx + 5, dy + 5], fill="#dc2626")
        st.image(debug_img, use_container_width=True)
        st.caption("Green boxes = detected PC zones | Red dots = numbers read by OCR")

    adj_w, adj_d, has_remote, rem_w, rem_d, gap = 4.0, 1.5, False, 4.0, 1.5, 2.0

    if boxes and dimensions:
        if len(boxes) >= 2:
            has_remote = True
            rx, ry, rw, rh = boxes[0]
            bx, by, bw, bh = boxes[-1]

            # 1. Clear Gap: strictly in outer left/right margin vertically between Remote PC and Adjacent PC
            gap_candidates = [
                d for d in dimensions
                if (ry + rh - 25) <= d["y"] <= (by + 25)
                and (d["x"] <= min(rx, bx) + 25 or d["x"] >= max(rx + rw, bx + bw) - 25)
            ]
            val = claim_best(gap_candidates, min(rx, bx) - 30, (ry + rh + by) / 2)
            if val is not None:
                gap = val

            # 2. Adjacent PC Width: centered horizontally over/under bottom box
            adj_w_candidates = [
                d for d in dimensions
                if (bx + 20) <= d["x"] <= (bx + bw - 20)
                and (ry + rh - 10) <= d["y"] <= (by + bh + 110)
            ]
            val = claim_best(adj_w_candidates, bx + bw / 2, by)
            if val is not None:
                adj_w = val

            # 3. Adjacent PC Depth: beside vertical midpoint of bottom box
            adj_d_candidates = [
                d for d in dimensions
                if (by - 25) <= d["y"] <= (by + bh + 25)
                and (d["x"] <= bx + 45 or d["x"] >= bx + bw - 45)
            ]
            val = claim_best(adj_d_candidates, bx, by + bh / 2)
            if val is not None:
                adj_d = val

            # 4. Remote PC Width: centered above/below top box
            rem_w_candidates = [
                d for d in dimensions
                if (rx - 25) <= d["x"] <= (rx + rw + 25)
                and (ry - 130) <= d["y"] <= (ry + rh + 40)
            ]
            val = claim_best(rem_w_candidates, rx + rw / 2, ry)
            if val is not None:
                rem_w = val

            # 5. Remote PC Depth: beside vertical midpoint of top box
            rem_d_candidates = [
                d for d in dimensions
                if (ry - 25) <= d["y"] <= (ry + rh + 25)
                and (d["x"] <= rx + 45 or d["x"] >= rx + rw - 45)
            ]
            val = claim_best(rem_d_candidates, rx + rw, ry + rh / 2)
            if val is not None:
                rem_d = val

            # Enforce visual aspect ratio on Adjacent PC if wider than tall
            if bw > bh * 1.3 and adj_w < adj_d:
                adj_w, adj_d = adj_d, adj_w

        else:
            bx, by, bw, bh = boxes[0]
            adj_w_candidates = [
                d for d in dimensions
                if (bx - 20) <= d["x"] <= (bx + bw + 20)
                and (by - 120) <= d["y"] <= (by + bh + 120)
            ]
            val = claim_best(adj_w_candidates, bx + bw / 2, by)
            if val is not None:
                adj_w = val

            adj_d_candidates = [
                d for d in dimensions
                if (by - 25) <= d["y"] <= (by + bh + 25)
                and (d["x"] <= bx + 45 or d["x"] >= bx + bw - 45)
            ]
            val = claim_best(adj_d_candidates, bx, by + bh / 2)
            if val is not None:
                adj_d = val

            if bw > bh * 1.3 and adj_w < adj_d:
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
        draw.text(
            (ax0 + 10, (ay0 + ay1) // 2 - 8),
            f"Adjacent PC ({adj_area} sq.m)",
            fill="black",
        )

        draw_dimension_arrow(draw, (ax0, ay1), (ax1, ay1), f"{adj_w}m", offset=35)
        draw_dimension_arrow(draw, (ax1, ay0), (ax1, ay1), f"{adj_d}m", offset=25)

        if has_remote:
            ry1 = ay0 - int(gap * scale)
            ry0 = ry1 - int(rem_d * scale)
            rx1 = ax0 + int(rem_w * scale)

            draw.rectangle(
                [ax0, ry0, rx1, ry1], outline="black", fill="#fef3c7", width=3
            )
            rem_area = round(rem_w * rem_d, 2)
            draw.text(
                (ax0 + 10, (ry0 + ry1) // 2 - 8),
                f"Remote PC ({rem_area} sq.m)",
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
