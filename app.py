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
    """Multi-pass detector: finds both intact small boxes and wide boxes with access openings."""
    h, w = img_cv.shape[:2]
    page_area = h * w
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Pass 1: Normalize warm shadows / phone lighting
    bg = cv2.GaussianBlur(gray, (51, 51), 0)
    norm = cv2.divide(gray, bg, scale=255)
    blurred = cv2.GaussianBlur(norm, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Pass 2: Light closing (preserves small intact boxes like 2m x 2m Remote PC)
    k_small = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed_small = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k_small)

    # Pass 3: Wide horizontal/vertical closing (bridges customer access openings)
    k_horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (55, 5))
    k_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 45))
    closed_wide = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k_horiz)
    closed_wide = cv2.morphologyEx(closed_wide, cv2.MORPH_CLOSE, k_vert)

    raw_boxes = []
    for mask in (closed_small, closed_wide):
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
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

    raw_boxes.sort(key=box_area, reverse=True)
    unique_boxes = []
    for bx, by, bw, bh in raw_boxes:
        is_dup = False
        for ux, uy, uw, uh in unique_boxes:
            if (
                abs(bx - ux) < 35
                and abs(by - uy) < 35
                and abs(bw - uw) < 55
                and abs(bh - uh) < 55
            ):
                is_dup = True
                break
        if not is_dup:
            unique_boxes.append((bx, by, bw, bh))

    # Remove any giant outer container box that completely encloses other detected boxes
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


def parse_ocr_items(ocr_results, dimensions):
    if not ocr_results:
        return
    for item in ocr_results:
        bbox, text, _ = item
        pts = np.array(bbox, dtype=float)
        xs = pts[:, 0]
        ys = pts[:, 1]
        box_w = float(xs.max() - xs.min())
        box_h = float(ys.max() - ys.min())
        # Reject tall, skinny vertical arrow lines misread as '1' or '0.6'
        if box_w > 0 and (box_h / box_w) > 2.2:
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
            num_token = match.group(1)
            val = float(num_token)
            if val > 100:
                val /= 1000.0
            # Block standalone '1.0' without 'm' unit (vertical dimension line artifact)
            if val == 1.0 and not has_unit:
                continue
            # Snap noisy OCR decimal artifacts (e.g. '2.4' from '2m') if no decimal was written
            if "." not in num_token and val < 20:
                val = float(round(val))
            elif abs(val - round(val)) in (0.4, 0.6) and not has_unit:
                val = float(round(val))

            if 0.4 <= val <= 50.0:
                cx = int((xs.min() + xs.max()) / 2)
                cy = int((ys.min() + ys.max()) / 2)
                is_dup = any(
                    abs(d["x"] - cx) < 25 and abs(d["y"] - cy) < 25 and d["val"] == round(val, 2)
                    for d in dimensions
                )
                if not is_dup:
                    dimensions.append(
                        {
                            "val": round(val, 2),
                            "x": cx,
                            "y": cy,
                            "has_unit": has_unit,
                            "used": False,
                        }
                    )


def extract_dimensions(pil_img):
    """Dual-pass OCR: contrast-enhanced original + lighting-normalized image."""
    dimensions = []

    enhanced = ImageEnhance.Contrast(pil_img).enhance(1.9)
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.6)
    res1, _ = ocr(np.array(enhanced))
    parse_ocr_items(res1, dimensions)

    gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    bg = cv2.GaussianBlur(gray, (51, 51), 0)
    norm = cv2.divide(gray, bg, scale=255)
    norm_rgb = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)
    res2, _ = ocr(norm_rgb)
    parse_ocr_items(res2, dimensions)

    return dimensions


def claim_best_dim(dimensions, candidates, score_fn, default_val):
    unused = [d for d in candidates if not d["used"]]
    pool = unused if unused else candidates
    if not pool:
        return default_val
    # Prefer detections that explicitly include 'm' unit, then closest spatial match
    best = min(pool, key=lambda d: (0 if d.get("has_unit") else 1, score_fn(d)))
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
        st.subheader("Uploaded Plan & Detection Check")
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
        st.caption("Green boxes = detected PC zones | Red dots = numbers read on plan")

    adj_w, adj_d, has_remote, rem_w, rem_d, gap = 5.0, 2.0, False, 2.0, 2.0, 2.0

    if boxes and dimensions:
        bx, by, bw, bh = boxes[-1]  # Bottom box = Adjacent PC

        if len(boxes) >= 2:
            has_remote = True
            rx, ry, rw, rh = boxes[0]  # Top box = Remote PC

            # 1. Claim centered width label sitting in the walkway gap above Adjacent PC
            walkway_w_cands = [
                d
                for d in dimensions
                if (ry + rh - 10) <= d["y"] <= (by + 25)
                and (bx + bw * 0.15) <= d["x"] <= (bx + bw * 0.85)
            ]
            if not walkway_w_cands:
                walkway_w_cands = [
                    d
                    for d in dimensions
                    if abs(d["x"] - (bx + bw / 2)) < bw * 0.45
                ]
            adj_w = claim_best_dim(
                dimensions,
                walkway_w_cands,
                lambda d: abs(d["x"] - (bx + bw / 2)) + abs(d["y"] - by) * 0.5,
                5.0,
            )

            # 2. Claim Clear Gap label in the left/right outer margin between the two zones
            left_edge = min(bx, rx)
            right_edge = max(bx + bw, rx + rw)
            gap_y = (ry + rh + by) / 2
            gap_cands = [
                d
                for d in dimensions
                if (ry + rh - 25) <= d["y"] <= (by + 25)
                and (d["x"] <= left_edge + 20 or d["x"] >= right_edge - 20)
            ]
            gap = claim_best_dim(
                dimensions,
                gap_cands,
                lambda d: abs(d["y"] - gap_y),
                2.0,
            )

            # 3. Claim Remote PC Width (centered above Remote PC)
            rem_w_cands = [
                d
                for d in dimensions
                if d["y"] <= ry + 20 and abs(d["x"] - (rx + rw / 2)) < rw * 0.75
            ]
            rem_w = claim_best_dim(
                dimensions,
                rem_w_cands,
                lambda d: abs(d["x"] - (rx + rw / 2)) + abs(d["y"] - ry),
                2.0,
            )

            # 4. Claim Remote PC Depth (beside Remote PC vertical midpoint)
            rem_d_cands = [
                d
                for d in dimensions
                if abs(d["y"] - (ry + rh / 2)) < rh * 0.75
                and (d["x"] <= rx + 15 or d["x"] >= rx + rw - 15)
            ]
            rem_d = claim_best_dim(
                dimensions,
                rem_d_cands,
                lambda d: abs(d["y"] - (ry + rh / 2)),
                2.0,
            )

            # 5. Claim Adjacent PC Depth (beside Adjacent PC vertical midpoint)
            adj_d_cands = [
                d
                for d in dimensions
                if abs(d["y"] - (by + bh / 2)) < bh * 0.75
                and (d["x"] <= bx + 15 or d["x"] >= bx + bw - 15)
            ]
            adj_d = claim_best_dim(
                dimensions,
                adj_d_cands,
                lambda d: abs(d["y"] - (by + bh / 2)),
                2.0,
            )
        else:
            adj_w_cands = [
                d for d in dimensions if abs(d["x"] - (bx + bw / 2)) < bw * 0.65
            ]
            adj_w = claim_best_dim(
                dimensions,
                adj_w_cands,
                lambda d: abs(d["x"] - (bx + bw / 2)) + abs(d["y"] - by),
                5.0,
            )
            adj_d_cands = [
                d for d in dimensions if abs(d["y"] - (by + bh / 2)) < bh * 0.75
            ]
            adj_d = claim_best_dim(
                dimensions,
                adj_d_cands,
                lambda d: abs(d["y"] - (by + bh / 2)),
                2.0,
            )

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

        # Solid, fully enclosed Adjacent PC polygon
        draw.rectangle([ax0, ay0, ax1, ay1], outline="black", fill="#dbeafe", width=3)
        adj_area = round(adj_w * adj_d, 2)
        adj_label = f"Adjacent PC ({adj_area} sq.m)"
        if (ax1 - ax0) >= 130:
            draw.text((ax0 + 10, (ay0 + ay1) // 2 - 8), adj_label, fill="black")
        else:
            draw.text((ax0, ay0 - 18), adj_label, fill="black")

        draw_dimension_arrow(draw, (ax0, ay1), (ax1, ay1), f"{adj_w}m", offset=35)
        draw_dimension_arrow(draw, (ax1, ay0), (ax1, ay1), f"{adj_d}m", offset=25)

        if has_remote:
            ry1 = ay0 - int(gap * scale)
            ry0 = ry1 - int(rem_d * scale)
            rx1 = ax0 + int(rem_w * scale)

            # Solid, fully enclosed Remote PC polygon
            draw.rectangle([ax0, ry0, rx1, ry1], outline="black", fill="#fef3c7", width=3)
            rem_area = round(rem_w * rem_d, 2)
            rem_label = f"Remote PC ({rem_area} sq.m)"
            if (rx1 - ax0) >= 130:
                draw.text((ax0 + 10, (ry0 + ry1) // 2 - 8), rem_label, fill="black")
            else:
                draw.text((ax0, ry0 - 18), rem_label, fill="black")

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
