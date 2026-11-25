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
    h, w = img_cv.shape[:2]
    page_area = h * w
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    bg = cv2.GaussianBlur(gray, (51, 51), 0)
    norm = cv2.divide(gray, bg, scale=255)
    blurred = cv2.GaussianBlur(norm, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
    )

    candidates = []
    for k_size in [(7, 7), (45, 15), (15, 45)]:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, k_size)
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
            if 4 <= len(approx) <= 8 and bw > 30 and bh > 20 and rect_fill > 0.62:
                candidates.append((bx, by, bw, bh))

    candidates.sort(key=box_area, reverse=True)
    unique_boxes = []
    for bx, by, bw, bh in candidates:
        is_dup = False
        for ux, uy, uw, uh in unique_boxes:
            inter_x0 = max(bx, ux)
            inter_y0 = max(by, uy)
            inter_x1 = min(bx + bw, ux + uw)
            inter_y1 = min(by + bh, uy + uh)
            if inter_x1 > inter_x0 and inter_y1 > inter_y0:
                inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
                min_area = min(bw * bh, uw * uh)
                if inter_area / float(min_area) > 0.45:
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
                and bx <= ox + 10
                and by <= oy + 10
                and (bx + bw) >= (ox + ow - 10)
                and (by + bh) >= (oy + oh - 10)
            ):
                contains_others += 1
        if contains_others == 0:
            inner_boxes.append((bx, by, bw, bh))

    inner_boxes.sort(key=box_top_y)
    return inner_boxes


def run_ocr_pass(img_array, dimensions):
    ocr_results, _ = ocr(img_array)
    if not ocr_results:
        return
    for item in ocr_results:
        bbox, text, _ = item
        pts = np.array(bbox, dtype=float)
        min_xy = pts.min(axis=0)
        max_xy = pts.max(axis=0)
        bw = max_xy[0] - min_xy[0]
        bh = max_xy[1] - min_xy[1]
        if bw > 0 and (bh / bw) > 2.4:
            continue

        raw_str = str(text).strip()
        has_m = "m" in raw_str.lower()
        clean_text = (
            raw_str.replace(",", ".")
            .replace("O", "0")
            .replace("o", "0")
            .replace("l", "1")
            .replace("I", "1")
        )
        for match in re.finditer(r"(\d+\.?\d*)", clean_text):
            num_str = match.group(1)
            val = float(num_str)
            if val > 100:
                val /= 1000.0
            if val == 1.0 and not has_m:
                continue
            if "." not in num_str and abs(val - round(val)) <= 0.4:
                val = float(round(val))
            if 0.4 <= val <= 50.0:
                center_x, center_y = pts.mean(axis=0)
                cx, cy = int(center_x), int(center_y)
                is_dup = False
                for d in dimensions:
                    if abs(d["x"] - cx) < 35 and abs(d["y"] - cy) < 35:
                        if has_m and not d["has_m"]:
                            d["val"] = round(val, 2)
                            d["has_m"] = True
                        is_dup = True
                        break
                if not is_dup:
                    dimensions.append(
                        {"val": round(val, 2), "x": cx, "y": cy, "has_m": has_m}
                    )


def extract_dimensions(pil_img):
    dimensions = []
    enhanced = ImageEnhance.Contrast(pil_img).enhance(1.9)
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.6)
    run_ocr_pass(np.array(enhanced), dimensions)

    gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    bg = cv2.GaussianBlur(gray, (51, 51), 0)
    norm = cv2.divide(gray, bg, scale=255)
    norm_rgb = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)
    run_ocr_pass(norm_rgb, dimensions)
    return dimensions


def claim_closest(dimensions, used_ids, target_x, target_y, x_weight=1.0, y_weight=1.0):
    best_idx = None
    best_score = float("inf")
    for idx, d in enumerate(dimensions):
        if idx in used_ids:
            continue
        score = ((d["x"] - target_x) * x_weight) ** 2 + ((d["y"] - target_y) * y_weight) ** 2
        if d["has_m"]:
            score -= 3600
        if score < best_score:
            best_score = score
            best_idx = idx
    if best_idx is not None:
        used_ids.add(best_idx)
        return dimensions[best_idx]["val"]
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
        st.caption("Green boxes = detected PC zones | Red dots = numbers read by OCR")

    adj_w, adj_d, has_remote, rem_w, rem_d, gap = 4.0, 1.5, False, 4.0, 1.5, 2.0
    used_ids = set()

    if boxes and dimensions:
        bx, by, bw, bh = boxes[-1]

        if len(boxes) >= 2:
            has_remote = True
            rx, ry, rw, rh = next(iter(boxes))

            # 1. Claim Clear Gap along the left/right side margin between the two zones
            gap_y = (ry + rh + by) / 2.0
            left_margin_x = min(bx, rx) - 25
            gap_candidates = [
                (idx, d)
                for idx, d in enumerate(dimensions)
                if (ry + rh - 25) <= d["y"] <= (by + 25)
                and (d["x"] <= min(bx, rx) + 25 or d["x"] >= max(bx + bw, rx + rw) - 25)
            ]
            if gap_candidates:
                best_gap_idx, best_gap_d = min(
                    gap_candidates,
                    key=lambda item: (item[1]["x"] - left_margin_x) ** 2 + (item[1]["y"] - gap_y) ** 2,
                )
                used_ids.add(best_gap_idx)
                gap = best_gap_d["val"]
            else:
                g_val = claim_closest(dimensions, used_ids, left_margin_x, gap_y)
                if g_val:
                    gap = g_val

            # 2. Claim Remote PC Width & Depth
            rw_val = claim_closest(dimensions, used_ids, rx + rw / 2.0, ry - 20, x_weight=1.0, y_weight=1.3)
            if rw_val:
                rem_w = rw_val
            rd_val = claim_closest(dimensions, used_ids, rx + rw + 25, ry + rh / 2.0, x_weight=1.2, y_weight=1.0)
            if rd_val:
                rem_d = rd_val

            # Enforce orientation match for Remote PC if clearly rectangular
            if rw > rh * 1.25 and rem_w < rem_d:
                rem_w, rem_d = rem_d, rem_w
            elif rh > rw * 1.25 and rem_d < rem_w:
                rem_w, rem_d = rem_d, rem_w

        # 3. Claim Adjacent PC Width & Depth
        aw_val = claim_closest(dimensions, used_ids, bx + bw / 2.0, by - 20, x_weight=1.0, y_weight=1.3)
        if aw_val:
            adj_w = aw_val
        ad_val = claim_closest(dimensions, used_ids, bx - 25, by + bh / 2.0, x_weight=1.2, y_weight=1.0)
        if ad_val:
            adj_d = ad_val

        # Enforce orientation match for Adjacent PC (wider box -> larger dim is Width)
        if bw > bh * 1.25 and adj_w < adj_d:
            adj_w, adj_d = adj_d, adj_w
        elif bh > bw * 1.25 and adj_d < adj_w:
            adj_w, adj_d = adj_d, adj_w

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

        draw.rectangle([ax0, ay0, ax1, ay1], outline="black", fill="#dbeafe", width=3)
        adj_area = round(adj_w * adj_d, 2)
        adj_label = f"Adjacent PC ({adj_area} sq.m)"
        if (ax1 - ax0) < 150:
            draw.text((ax0, ay0 - 18), adj_label, fill="black")
        else:
            draw.text((ax0 + 10, (ay0 + ay1) // 2 - 8), adj_label, fill="black")

        draw_dimension_arrow(draw, (ax0, ay1), (ax1, ay1), f"{adj_w}m", offset=35)
        draw_dimension_arrow(draw, (ax1, ay0), (ax1, ay1), f"{adj_d}m", offset=25)

        if has_remote:
            ry1 = ay0 - int(gap * scale)
            ry0 = ry1 - int(rem_d * scale)
            rx1 = ax0 + int(rem_w * scale)

            draw.rectangle([ax0, ry0, rx1, ry1], outline="black", fill="#fef3c7", width=3)
            rem_area = round(rem_w * rem_d, 2)
            rem_label = f"Remote PC ({rem_area} sq.m)"
            if (rx1 - ax0) < 150:
                draw.text((ax0, ry1 + 6), rem_label, fill="black")
            else:
                draw.text((ax0 + 10, (ry0 + ry1) // 2 - 8), rem_label, fill="black")

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
