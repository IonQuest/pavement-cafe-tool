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


def normalize_lighting(gray):
    bg = cv2.GaussianBlur(gray, (55, 55), 0)
    norm = cv2.divide(gray, bg, scale=255)
    return norm


def detect_cafe_rectangles(img_cv):
    """Detects both intact Remote PC boxes and entrance-gapped Adjacent PC boxes without merging across the walkway."""
    h, w = img_cv.shape[:2]
    page_area = h * w
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    norm = normalize_lighting(gray)
    blurred = cv2.GaussianBlur(norm, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
    )

    # Pass 1: Light closing for intact rectangles (Remote PC)
    k_light = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed_light = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k_light)

    # Pass 2: Horizontal-only wide closing to bridge customer entrance openings without merging vertical zones
    k_horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 3))
    closed_horiz = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k_horiz)
    closed_horiz = cv2.morphologyEx(closed_horiz, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9)))

    raw_boxes = []
    for mask in (closed_light, closed_horiz):
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < (page_area * 0.0012) or area > (page_area * 0.45):
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            bx, by, bw, bh = cv2.boundingRect(approx)
            rect_fill = area / float(bw * bh) if (bw * bh) > 0 else 0
            if 4 <= len(approx) <= 8 and bw > 30 and bh > 22 and rect_fill > 0.65:
                raw_boxes.append((bx, by, bw, bh))

    # Deduplicate overlapping boxes (keep largest clean outline per zone)
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

    # Remove any outer container box that wraps around inner boxes
    inner_boxes = []
    for i, (bx, by, bw, bh) in enumerate(unique_boxes):
        contains_others = False
        for j, (ox, oy, ow, oh) in enumerate(unique_boxes):
            if i != j and bx <= ox + 10 and by <= oy + 10 and (bx + bw) >= (ox + ow - 10) and (by + bh) >= (oy + oh - 10):
                contains_others = True
                break
        if not contains_others:
            inner_boxes.append((bx, by, bw, bh))

    inner_boxes.sort(key=box_top_y)
    return inner_boxes


def parse_ocr_items(ocr_results, dimensions_list):
    if not ocr_results:
        return
    for item in ocr_results:
        bbox, text, _ = item
        pts = np.array(bbox, dtype=float)
        xs = pts[:, 0]
        ys = pts[:, 1]
        box_w = max(xs) - min(xs)
        box_h = max(ys) - min(ys)

        # Reject tall, skinny vertical arrow lines misread as "1" or "1.6"
        if box_h > 0 and (box_h / max(box_w, 1.0)) > 2.4:
            continue

        raw_str = str(text).strip()
        has_unit = bool(re.search(r"[mM]", raw_str))
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
            # Block bare "1" or "1.0" without an "m" unit (vertical line artifact)
            if abs(val - 1.0) < 0.05 and not has_unit:
                continue
            if 0.4 <= val <= 50.0:
                # Snap noisy decimals when no decimal point was written
                if "." not in match.group(1):
                    val = round(val)
                center_x = int(xs.mean())
                center_y = int(ys.mean())
                # Avoid duplicate detections at the exact same spot from dual passes
                already = any(
                    abs(d["x"] - center_x) < 25 and abs(d["y"] - center_y) < 25 and abs(d["val"] - val) < 0.1
                    for d in dimensions_list
                )
                if not already:
                    dimensions_list.append({
                        "val": round(val, 2),
                        "x": center_x,
                        "y": center_y,
                        "has_unit": has_unit,
                        "used": False,
                    })


def extract_dimensions(pil_img):
    dimensions = []
    # Pass 1: Contrast-boosted original image
    enhanced = ImageEnhance.Contrast(pil_img).enhance(2.0)
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.8)
    res1, _ = ocr(np.array(enhanced))
    parse_ocr_items(res1, dimensions)

    # Pass 2: Lighting-normalized grayscale image
    gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    norm = normalize_lighting(gray)
    res2, _ = ocr(cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB))
    parse_ocr_items(res2, dimensions)

    return dimensions


def claim_best(candidates, target_x, target_y, max_dist=320):
    unused = [d for d in candidates if not d["used"]]
    if not unused:
        return None
    best = min(
        unused,
        key=lambda d: ((d["x"] - target_x) ** 2 + (d["y"] - target_y) ** 2) ** 0.5
        - (50 if d["has_unit"] else 0),
    )
    dist = ((best["x"] - target_x) ** 2 + (best["y"] - target_y) ** 2) ** 0.5
    if dist <= max_dist:
        best["used"] = True
        return best["val"]
    return None


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
        st.subheader("Uploaded Plan & Detection Preview")
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
        st.caption("Green boxes = detected PC zones | Red dots = numbers read by OCR")

    adj_w, adj_d, has_remote, rem_w, rem_d, gap = 4.0, 1.5, False, 2.0, 2.0, 2.0

    if boxes and dimensions:
        bx, by, bw, bh = boxes[-1]  # Bottom box = Adjacent PC

        if len(boxes) >= 2:
            has_remote = True
            rx, ry, rw, rh = boxes[0]  # Top box = Remote PC

            # 1. Claim Clear Gap strictly along the left or right outer margin between the two boxes
            margin_limit_left = max(bx, rx) + 45
            margin_limit_right = min(bx + bw, rx + rw) - 45
            gap_cands = [
                d for d in dimensions
                if (ry + rh - 10) <= d["y"] <= (by + 10)
                and (d["x"] <= margin_limit_left or d["x"] >= margin_limit_right)
            ]
            val_gap = claim_best(gap_cands, min(bx, rx), (ry + rh + by) / 2, max_dist=350)
            if val_gap is not None:
                gap = val_gap

            # 2. Claim Adjacent PC Width from the centered walkway label between the boxes (or below bottom box)
            adj_w_cands = [
                d for d in dimensions
                if (ry + rh - 10) <= d["y"] <= (by + bh + 160)
                and (bx - 40) <= d["x"] <= (bx + bw + 40)
            ]
            val_adj_w = claim_best(adj_w_cands, bx + bw / 2, by, max_dist=380)
            if val_adj_w is not None:
                adj_w = val_adj_w

            # 3. Claim Adjacent PC Depth beside the bottom box
            adj_d_cands = [
                d for d in dimensions
                if (by - 40) <= d["y"] <= (by + bh + 40)
                and (d["x"] <= bx + 45 or d["x"] >= bx + bw - 45)
            ]
            val_adj_d = claim_best(adj_d_cands, bx, by + bh / 2, max_dist=320)
            if val_adj_d is not None:
                adj_d = val_adj_d

            # 4. Claim Remote PC Width above/below top box
            rem_w_cands = [
                d for d in dimensions
                if (ry - 160) <= d["y"] <= (ry + rh + 40)
                and (rx - 40) <= d["x"] <= (rx + rw + 40)
            ]
            val_rem_w = claim_best(rem_w_cands, rx + rw / 2, ry, max_dist=320)
            if val_rem_w is not None:
                rem_w = val_rem_w

            # 5. Claim Remote PC Depth beside top box
            rem_d_cands = [
                d for d in dimensions
                if (ry - 40) <= d["y"] <= (ry + rh + 40)
                and (d["x"] <= rx + 45 or d["x"] >= rx + rw - 45)
            ]
            val_rem_d = claim_best(rem_d_cands, rx + rw, ry + rh / 2, max_dist=320)
            if val_rem_d is not None:
                rem_d = val_rem_d

            # Orientation safeguard: if Remote box is clearly wider than tall, ensure width >= depth
            if rw > rh * 1.35 and rem_d > rem_w:
                rem_w, rem_d = rem_d, rem_w
        else:
            # Single Adjacent PC layout
            adj_w_cands = [
                d for d in dimensions
                if (bx - 40) <= d["x"] <= (bx + bw + 40)
            ]
            val_adj_w = claim_best(adj_w_cands, bx + bw / 2, by, max_dist=380)
            if val_adj_w is not None:
                adj_w = val_adj_w

            adj_d_cands = [
                d for d in dimensions
                if (by - 40) <= d["y"] <= (by + bh + 40)
            ]
            val_adj_d = claim_best(adj_d_cands, bx, by + bh / 2, max_dist=320)
            if val_adj_d is not None:
                adj_d = val_adj_d

        # Orientation safeguard: if Adjacent box is clearly wider than tall, ensure width >= depth
        if bw > bh * 1.35 and adj_d > adj_w:
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
        if (ax1 - ax0) >= 150:
            draw.text((ax0 + 10, (ay0 + ay1) // 2 - 8), adj_label, fill="black")
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
            if (rx1 - ax0) >= 150:
                draw.text((ax0 + 10, (ry0 + ry1) // 2 - 8), rem_label, fill="black")
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
