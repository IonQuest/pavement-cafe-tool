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


def normalize_lighting(img_cv):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    bg = cv2.GaussianBlur(gray, (51, 51), 0)
    norm = cv2.divide(gray, bg, scale=255)
    return norm


def extract_dimensions(pil_img, img_cv):
    """Runs OCR on contrast-boosted and lighting-normalized passes, deduplicating nearby hits."""
    enhanced = ImageEnhance.Contrast(pil_img).enhance(1.8)
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.5)
    pass1 = np.array(enhanced)

    norm_gray = normalize_lighting(img_cv)
    pass2 = cv2.cvtColor(norm_gray, cv2.COLOR_GRAY2RGB)

    raw_dims = []
    for img_arr in (pass1, pass2):
        ocr_results, _ = ocr(img_arr)
        if not ocr_results:
            continue
        for item in ocr_results:
            bbox, text, _ = item
            pts = np.array(bbox, dtype=float)
            bw = pts[:, 0].max() - pts[:, 0].min()
            bh = pts[:, 1].max() - pts[:, 1].min()
            # Reject tall vertical arrow shafts misread as digits
            if bw > 0 and (bh / bw) > 2.4:
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
                if not (0.3 <= val <= 50.0):
                    continue
                # Ignore bare single '1' without 'm' (typical vertical tick artifact)
                if val == 1.0 and not has_unit:
                    continue
                cx, cy = pts.mean(axis=0)
                raw_dims.append({
                    "val": round(val, 2),
                    "x": int(cx),
                    "y": int(cy),
                    "has_unit": has_unit,
                    "used": False,
                })

    # Deduplicate overlapping detections from the two passes
    unique_dims = []
    # Prefer detections with explicit 'm' unit first
    raw_dims.sort(key=lambda d: (not d["has_unit"], d["y"]))
    for d in raw_dims:
        is_dup = False
        for u in unique_dims:
            if abs(d["x"] - u["x"]) < 35 and abs(d["y"] - u["y"]) < 35:
                is_dup = True
                break
        if not is_dup:
            unique_dims.append(d)

    return unique_dims


def detect_cafe_rectangles(img_cv, dimensions):
    """Detects exact Pavement Cafe polygons (both intact and open-entrance C-shapes)."""
    h, w = img_cv.shape[:2]
    page_area = h * w
    norm_gray = normalize_lighting(img_cv)
    blurred = cv2.GaussianBlur(norm_gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 12
    )

    candidates = []
    # Pass 1: Light closing for intact boxes (e.g., Remote PC)
    # Pass 2: Horizontal closing to bridge customer entrance openings (e.g., Adjacent PC)
    kernels = [
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, int(w * 0.045)), 5)),
    ]

    for k in kernels:
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k)
        contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < (page_area * 0.002) or area > (page_area * 0.45):
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            # Filter out edge shadows, thin arrow lines, and narrow vertical sidebars
            if bw < int(w * 0.09) or bh < int(h * 0.065):
                continue
            if bw > int(w * 0.85) or by < 10 or (by + bh) > (h - 10):
                continue
            rect_fill = area / float(bw * bh) if (bw * bh) > 0 else 0
            if rect_fill < 0.55:
                continue

            # Discard phantom boxes that enclose an OCR dimension label inside their interior
            has_text_inside = False
            for d in dimensions:
                if (bx + 15) < d["x"] < (bx + bw - 15) and (by + 15) < d["y"] < (by + bh - 15):
                    has_text_inside = True
                    break
            if has_text_inside:
                continue

            candidates.append((bx, by, bw, bh))

    # Deduplicate overlapping candidates, keeping the widest clean bounding box per zone
    candidates.sort(key=lambda b: b[2] * b[3], reverse=True)
    unique_boxes = []
    for bx, by, bw, bh in candidates:
        is_dup = False
        for ux, uy, uw, uh in unique_boxes:
            inter_x1 = max(bx, ux)
            inter_y1 = max(by, uy)
            inter_x2 = min(bx + bw, ux + uw)
            inter_y2 = min(by + bh, uy + uh)
            if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                if inter_area > 0.4 * min(bw * bh, uw * uh):
                    is_dup = True
                    break
        if not is_dup:
            unique_boxes.append((bx, by, bw, bh))

    # Remove any outer box that completely wraps another detected box
    inner_boxes = []
    for i, (bx, by, bw, bh) in enumerate(unique_boxes):
        wraps_other = False
        for j, (ox, oy, ow, oh) in enumerate(unique_boxes):
            if i != j and bx <= ox and by <= oy and (bx + bw) >= (ox + ow) and (by + bh) >= (oy + oh):
                wraps_other = True
                break
        if not wraps_other:
            inner_boxes.append((bx, by, bw, bh))

    # Sort top-to-bottom by Y coordinate
    inner_boxes.sort(key=lambda b: b[1])
    return inner_boxes


def claim_closest(dimensions, filter_fn, target_x, target_y):
    matches = [d for d in dimensions if not d["used"] and filter_fn(d)]
    if not matches:
        return None
    best = min(
        matches,
        key=lambda d: ((d["x"] - target_x) ** 2 + (d["y"] - target_y) ** 2)
        - (3600 if d["has_unit"] else 0),
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
        dimensions = extract_dimensions(pil_img, img_cv)
        boxes = detect_cafe_rectangles(img_cv, dimensions)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Uploaded Plan & Detected Elements")
        debug_img = pil_img.copy()
        debug_draw = ImageDraw.Draw(debug_img)
        for idx, (bx, by, bw, bh) in enumerate(boxes):
            debug_draw.rectangle([bx, by, bx + bw, by + bh], outline="#16a34a", width=4)
        for d in dimensions:
            dx, dy = d["x"], d["y"]
            debug_draw.ellipse([dx - 5, dy - 5, dx + 5, dy + 5], fill="#dc2626")
        st.image(debug_img, use_container_width=True)
        st.caption("Green boxes = detected PC zones | Red dots = dimensions read by OCR")

    adj_w, adj_d, has_remote, rem_w, rem_d, gap = 4.0, 1.5, False, 4.0, 1.5, 2.0

    if boxes and dimensions:
        # Bottom box is Adjacent PC
        bx, by, bw, bh = boxes[-1]
        adj_cx = bx + bw / 2.0
        adj_cy = by + bh / 2.0

        if len(boxes) >= 2:
            has_remote = True
            rx, ry, rw, rh = boxes[0]
            rem_cx = rx + rw / 2.0
            rem_cy = ry + rh / 2.0
            walkway_mid_y = (ry + rh + by) / 2.0
            left_bound = min(rx, bx)
            right_bound = max(rx + rw, bx + bw)

            # 1. Claim Clear Gap along left/right outer margin between the two boxes
            val_gap = claim_closest(
                dimensions,
                lambda d: (ry + rh - 25) <= d["y"] <= (by + 25)
                and (d["x"] <= left_bound + 35 or d["x"] >= right_bound - 35),
                left_bound,
                walkway_mid_y,
            )
            if val_gap is not None:
                gap = val_gap

            # 2. Claim Adjacent PC Width (centered above or below Adjacent PC)
            val_adj_w = claim_closest(
                dimensions,
                lambda d: (bx - 40) <= d["x"] <= (bx + bw + 40)
                and (
                    ((ry + rh - 10) <= d["y"] <= (by + 40))
                    or ((by + bh - 20) <= d["y"] <= (by + bh + 160))
                ),
                adj_cx,
                by - 30,
            )
            if val_adj_w is not None:
                adj_w = val_adj_w

            # 3. Claim Adjacent PC Depth (along left or right side of Adjacent PC)
            val_adj_d = claim_closest(
                dimensions,
                lambda d: (by - 35) <= d["y"] <= (by + bh + 35)
                and (d["x"] <= bx + 40 or d["x"] >= bx + bw - 40),
                bx - 40,
                adj_cy,
            )
            if val_adj_d is not None:
                adj_d = val_adj_d

            # 4. Claim Remote PC Width (directly above Remote PC — covers full top arrow zone)
            val_rem_w = claim_closest(
                dimensions,
                lambda d: (rx - 60) <= d["x"] <= (rx + rw + 60)
                and (ry - 240) <= d["y"] <= (ry + 25),
                rem_cx,
                ry - 60,
            )
            if val_rem_w is not None:
                rem_w = val_rem_w

            # 5. Claim Remote PC Depth (along left or right side of Remote PC)
            val_rem_d = claim_closest(
                dimensions,
                lambda d: (ry - 35) <= d["y"] <= (ry + rh + 35)
                and (d["x"] <= rx + 40 or d["x"] >= rx + rw - 40),
                rx + rw + 40,
                rem_cy,
            )
            if val_rem_d is not None:
                rem_d = val_rem_d

            # Orientation check: ensure visually wide boxes keep larger value as width
            if bw > bh * 1.35 and adj_d > adj_w:
                adj_w, adj_d = adj_d, adj_w
            if rw > rh * 1.35 and rem_d > rem_w:
                rem_w, rem_d = rem_d, rem_w
        else:
            # Single Adjacent PC layout
            val_adj_w = claim_closest(
                dimensions,
                lambda d: (bx - 40) <= d["x"] <= (bx + bw + 40)
                and ((by - 200) <= d["y"] <= (by + 40) or (by + bh - 20) <= d["y"] <= (by + bh + 180)),
                adj_cx,
                by - 40,
            )
            if val_adj_w is not None:
                adj_w = val_adj_w

            val_adj_d = claim_closest(
                dimensions,
                lambda d: (by - 35) <= d["y"] <= (by + bh + 35)
                and (d["x"] <= bx + 40 or d["x"] >= bx + bw - 40),
                bx - 40,
                adj_cy,
            )
            if val_adj_d is not None:
                adj_d = val_adj_d

            if bw > bh * 1.35 and adj_d > adj_w:
                adj_w, adj_d = adj_d, adj_w

    with col2:
        st.subheader("Dimensioned Polygon Output")
        detected_vals = [f"{d['val']}m" for d in dimensions]
        st.caption(
            f"Detected {len(boxes)} zone(s) | Numbers read: "
            f"{', '.join(detected_vals) if detected_vals else 'None'}"
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

        draw.rectangle([ax0, ay0, ax1, ay1], outline="black", fill="#dbeafe", width=3)
        adj_area = round(adj_w * adj_d, 2)
        draw.text((ax0 + 10, (ay0 + ay1) // 2 - 8), f"Adjacent PC ({adj_area} sq.m)", fill="black")

        draw_dimension_arrow(draw, (ax0, ay1), (ax1, ay1), f"{adj_w}m", offset=35)
        draw_dimension_arrow(draw, (ax1, ay0), (ax1, ay1), f"{adj_d}m", offset=25)

        if has_remote:
            ry1 = ay0 - int(gap * scale)
            ry0 = ry1 - int(rem_d * scale)
            rx1 = ax0 + int(rem_w * scale)

            draw.rectangle([ax0, ry0, rx1, ry1], outline="black", fill="#fef3c7", width=3)
            rem_area = round(rem_w * rem_d, 2)
            draw.text((ax0 + 10, (ry0 + ry1) // 2 - 8), f"Remote PC ({rem_area} sq.m)", fill="black")

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
