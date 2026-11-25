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
        draw.text(((x1 + x2) // 2 - 18, yd + 5 if offset > 0 else yd - 18), text, fill=color)
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


def box_area(box):
    bx, by, bw, bh = box
    return bw * bh


def box_top_y(box):
    bx, by, bw, bh = box
    return by


def find_boxes_from_binary(binary_img, page_area):
    contours, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    found = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < (page_area * 0.0012) or area > (page_area * 0.45):
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        bx, by, bw, bh = cv2.boundingRect(approx)
        rect_fill = area / float(bw * bh) if (bw * bh) > 0 else 0
        if 4 <= len(approx) <= 10 and bw > 28 and bh > 22 and rect_fill > 0.55:
            found.append((bx, by, bw, bh))
    return found


def detect_cafe_rectangles(img_cv):
    h, w = img_cv.shape[:2]
    page_area = h * w
    norm = normalize_lighting(img_cv)
    blurred = cv2.GaussianBlur(norm, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
    )

    k_small = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed_small = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k_small)

    k_wide = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 25))
    closed_wide = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k_wide)

    raw_boxes = find_boxes_from_binary(closed_small, page_area) + find_boxes_from_binary(closed_wide, page_area)

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
                min_area = min(bw * bh, uw * uh)
                if inter_area / float(min_area) > 0.55:
                    is_dup = True
                    break
        if not is_dup:
            unique_boxes.append((bx, by, bw, bh))

    inner_boxes = []
    for i, (bx, by, bw, bh) in enumerate(unique_boxes):
        contains_others = 0
        for j, (ox, oy, ow, oh) in enumerate(unique_boxes):
            if i != j and bx <= ox + 5 and by <= oy + 5 and (bx + bw) >= (ox + ow - 5) and (by + bh) >= (oy + oh - 5):
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
        bw_box = pts[:, 0].max() - pts[:, 0].min()
        bh_box = pts[:, 1].max() - pts[:, 1].min()
        # Reject tall skinny vertical arrow shaft artifacts
        if bw_box > 0 and (bh_box / bw_box) > 2.2:
            continue

        raw_str = str(text).strip()
        has_unit = bool(re.search(r"[mM]", raw_str))
        clean_text = raw_str.replace(",", ".").replace("O", "0").replace("o", "0")

        for match in re.finditer(r"(\d+\.?\d*)", clean_text):
            num_str = match.group(1)
            val = float(num_str)
            if val > 100:
                val /= 1000.0
            if not (0.4 <= val <= 50.0):
                continue
            # Reject bare standalone '1' or '1.0' without 'm' unit (arrow tick artifact)
            if abs(val - 1.0) < 0.05 and not has_unit:
                continue
            if "." not in num_str and abs(val - round(val)) <= 0.4:
                val = float(round(val))

            cx, cy = pts.mean(axis=0)
            cx_i, cy_i = int(cx), int(cy)

            is_dup = False
            for d in dimensions:
                if abs(d["x"] - cx_i) < 35 and abs(d["y"] - cy_i) < 35 and abs(d["val"] - val) < 0.2:
                    if has_unit:
                        d["has_unit"] = True
                    is_dup = True
                    break
            if not is_dup:
                dimensions.append({
                    "val": round(val, 2),
                    "x": cx_i,
                    "y": cy_i,
                    "has_unit": has_unit,
                    "used": False,
                })


def extract_dimensions(pil_img, img_cv):
    dimensions = []
    enhanced = ImageEnhance.Contrast(pil_img).enhance(2.0)
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.8)
    res1, _ = ocr(np.array(enhanced))
    parse_ocr_items(res1, dimensions)

    norm_rgb = cv2.cvtColor(normalize_lighting(img_cv), cv2.COLOR_GRAY2RGB)
    res2, _ = ocr(norm_rgb)
    parse_ocr_items(res2, dimensions)
    return dimensions


def pick_closest_unused(dimensions, target_x, target_y, filter_fn=None, fallback=None):
    candidates = [d for d in dimensions if not d["used"] and (filter_fn is None or filter_fn(d))]
    if not candidates:
        candidates = [d for d in dimensions if (filter_fn is None or filter_fn(d))]
    if not candidates:
        return fallback
    best = min(
        candidates,
        key=lambda d: ((d["x"] - target_x) ** 2 + (d["y"] - target_y) ** 2) ** 0.5 - (60 if d["has_unit"] else 0),
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
        st.caption("Green boxes = detected PC zones | Red dots = numbers read by OCR")

    adj_w, adj_d, has_remote, rem_w, rem_d, gap = 4.0, 1.5, False, 2.0, 2.0, 2.0

    if boxes and dimensions:
        bx, by, bw, bh = boxes[-1]  # Bottom box = Adjacent PC

        if len(boxes) >= 2:
            has_remote = True
            rx, ry, rw, rh = boxes[0]  # Top box = Remote PC

            # 1. Adjacent PC width: centered horizontally over/under the bottom box
            adj_w_val = pick_closest_unused(
                dimensions,
                bx + bw / 2,
                by - 20,
                filter_fn=lambda d: abs(d["x"] - (bx + bw / 2)) < bw * 0.45 and (ry + rh - 10) <= d["y"] <= (by + bh + 90),
            )
            if adj_w_val is not None:
                adj_w = adj_w_val

            # 2. Clear gap: sits in the walkway band between top and bottom boxes near left/right outer edges
            left_edge = min(rx, bx)
            right_edge = max(rx + rw, bx + bw)
            gap_y = (ry + rh + by) / 2
            gap_val = pick_closest_unused(
                dimensions,
                left_edge,
                gap_y,
                filter_fn=lambda d: (ry + rh - 15) <= d["y"] <= (by + 15) and (d["x"] <= left_edge + 60 or d["x"] >= right_edge - 60),
            )
            if gap_val is not None:
                gap = gap_val

            # 3. Adjacent PC depth: sits outside left or right vertical edge of bottom box
            adj_d_val = pick_closest_unused(
                dimensions,
                bx,
                by + bh / 2,
                filter_fn=lambda d: abs(d["y"] - (by + bh / 2)) < bh * 0.65 and (d["x"] <= bx + 20 or d["x"] >= bx + bw - 20),
            )
            if adj_d_val is not None:
                adj_d = adj_d_val

            # Enforce orientation consistency: if detected Adjacent PC box is wider than tall, width >= depth
            if bw > bh * 1.15 and adj_w < adj_d:
                adj_w, adj_d = adj_d, adj_w

            # 4. Remote PC width: centered above/below top box
            rem_w_val = pick_closest_unused(
                dimensions,
                rx + rw / 2,
                ry - 20,
                filter_fn=lambda d: abs(d["x"] - (rx + rw / 2)) < rw * 0.75 and d["y"] <= ry + rh + 40,
            )
            if rem_w_val is not None:
                rem_w = rem_w_val

            # 5. Remote PC depth: sits outside left or right vertical edge of top box
            rem_d_val = pick_closest_unused(
                dimensions,
                rx + rw,
                ry + rh / 2,
                filter_fn=lambda d: abs(d["y"] - (ry + rh / 2)) < rh * 0.75 and (d["x"] <= rx + 20 or d["x"] >= rx + rw - 20),
            )
            if rem_d_val is not None:
                rem_d = rem_d_val
        else:
            adj_w_val = pick_closest_unused(
                dimensions,
                bx + bw / 2,
                by,
                filter_fn=lambda d: abs(d["x"] - (bx + bw / 2)) < bw * 0.55,
            )
            if adj_w_val is not None:
                adj_w = adj_w_val
            adj_d_val = pick_closest_unused(
                dimensions,
                bx,
                by + bh / 2,
                filter_fn=lambda d: abs(d["y"] - (by + bh / 2)) < bh * 0.65,
            )
            if adj_d_val is not None:
                adj_d = adj_d_val
            if bw > bh * 1.15 and adj_w < adj_d:
                adj_w, adj_d = adj_d, adj_w

    with col2:
        st.subheader("Dimensioned Polygon Output")
        detected_vals = [f"{d['val']}m" for d in dimensions]
        st.caption(
            f"Detected {len(boxes)} zone(s) | Numbers read: {', '.join(detected_vals) if detected_vals else 'None'}"
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

        # Draw Adjacent PC as a solid, fully enclosed rectangle
        draw.rectangle([ax0, ay0, ax1, ay1], outline="black", fill="#dbeafe", width=3)
        adj_area = round(adj_w * adj_d, 2)
        draw.text((ax0 + 10, (ay0 + ay1) // 2 - 8), f"Adjacent PC ({adj_area} sq.m)", fill="black")

        draw_dimension_arrow(draw, (ax0, ay1), (ax1, ay1), f"{adj_w}m", offset=35)
        draw_dimension_arrow(draw, (ax1, ay0), (ax1, ay1), f"{adj_d}m", offset=25)

        if has_remote:
            ry1 = ay0 - int(gap * scale)
            ry0 = ry1 - int(rem_d * scale)
            rx1 = ax0 + int(rem_w * scale)

            # Draw Remote PC as a solid, fully enclosed rectangle
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
