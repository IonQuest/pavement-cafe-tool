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

def normalize_lighting(pil_img):
    gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    bg = cv2.medianBlur(gray, 51)
    norm = cv2.divide(gray, bg, scale=255)
    return norm

def detect_cafe_rectangles(pil_img):
    norm = normalize_lighting(pil_img)
    h, w = norm.shape[:2]
    page_area = h * w

    thresh = cv2.adaptiveThreshold(
        norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 12
    )

    # Pass 1: Light close for intact small Remote PC boxes
    k_small = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed_small = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k_small)

    # Pass 2: Wide H/V close to bridge customer access openings on Adjacent PC boxes
    k_wide_h = cv2.getStructuringElement(cv2.MORPH_RECT, (55, 3))
    k_wide_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 55))
    closed_wide = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k_wide_h)
    closed_wide = cv2.morphologyEx(closed_wide, cv2.MORPH_CLOSE, k_wide_v)

    raw_boxes = []
    for mask in (closed_small, closed_wide):
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < (page_area * 0.0012) or area > (page_area * 0.45):
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            bx, by, bw, bh = cv2.boundingRect(approx)
            rect_fill = area / float(bw * bh) if (bw * bh) > 0 else 0
            if 4 <= len(approx) <= 10 and bw > 30 and bh > 20 and rect_fill > 0.65:
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
                smaller_area = min(bw * bh, uw * uh)
                if inter_area / float(smaller_area) > 0.55:
                    is_dup = True
                    break
        if not is_dup:
            unique_boxes.append((bx, by, bw, bh))

    # Discard any outer container rectangle that encloses smaller detected zones
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

def extract_dimensions(pil_img):
    enhanced = ImageEnhance.Contrast(pil_img).enhance(1.9)
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.6)
    norm_rgb = cv2.cvtColor(normalize_lighting(pil_img), cv2.COLOR_GRAY2RGB)

    all_dims = []
    for img_variant in (np.array(enhanced), norm_rgb):
        ocr_results, _ = ocr(img_variant)
        if not ocr_results:
            continue
        for item in ocr_results:
            bbox, text, _ = item
            pts = np.array(bbox, dtype=float)
            min_x, min_y = pts.min(axis=0)
            max_x, max_y = pts.max(axis=0)
            bw_txt = max_x - min_x
            bh_txt = max_y - min_y

            # Reject tall skinny vertical arrow lines misread as 1 or 0.x
            if bw_txt > 0 and (bh_txt / bw_txt) > 2.4:
                continue

            raw_str = str(text).strip()
            has_unit = "m" in raw_str.lower()
            clean_text = raw_str.replace(",", ".").replace("O", "0").replace("o", "0")

            for match in re.finditer(r"(\d+\.?\d*)", clean_text):
                num_str = match.group(1)
                val = float(num_str)
                if val > 100:
                    val /= 1000.0
                # Filter standalone '1' without 'm' unit (common arrow-line artifact)
                if val == 1.0 and not has_unit:
                    continue
                # Snap noisy decimals when no decimal point was written
                if "." not in num_str and val < 50:
                    val = round(val)
                elif "." in num_str and abs(val - round(val)) <= 0.35 and not has_unit:
                    val = float(round(val))

                if 0.4 <= val <= 50.0:
                    cx = int((min_x + max_x) / 2)
                    cy = int((min_y + max_y) / 2)
                    is_dup = False
                    for existing in all_dims:
                        if abs(existing["x"] - cx) < 25 and abs(existing["y"] - cy) < 25 and existing["val"] == round(val, 2):
                            if has_unit:
                                existing["has_unit"] = True
                            is_dup = True
                            break
                    if not is_dup:
                        all_dims.append({
                            "val": round(val, 2),
                            "x": cx,
                            "y": cy,
                            "has_unit": has_unit,
                            "used": False
                        })
    return all_dims

def claim_best_dim(candidates, target_x, target_y, max_dist=320):
    best = None
    best_score = float("inf")
    for d in candidates:
        if d["used"]:
            continue
        dist = ((d["x"] - target_x) ** 2 + (d["y"] - target_y) ** 2) ** 0.5
        if dist <= max_dist:
            # Prefer dimensions explicitly written with 'm'
            score = dist - (65 if d["has_unit"] else 0)
            if score < best_score:
                best_score = score
                best = d
    if best is not None:
        best["used"] = True
        return best["val"]
    return None

uploaded_file = st.file_uploader(
    "Upload Site Plan (JPG, PNG, WEBP, or PDF)",
    type=["jpg", "jpeg", "png", "webp", "pdf"]
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
        boxes = detect_cafe_rectangles(pil_img)
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
        st.caption("Green boxes = detected PC zones | Red dots = dimensions read by OCR")

    adj_w, adj_d, has_remote, rem_w, rem_d, gap = 4.0, 1.5, False, 4.0, 1.5, 2.0

    if len(boxes) >= 2 and dimensions:
        has_remote = True
        rx, ry, rw, rh = boxes[0]
        bx, by, bw, bh = boxes[-1]

        # 1. Claim Clear Gap strictly on left or right outer margin between the two zones
        gap_y_mid = (ry + rh + by) / 2.0
        left_limit = min(rx, bx) + 60
        right_limit = max(rx + rw, bx + bw) - 60
        margin_gap_cands = [
            d for d in dimensions
            if (ry + rh - 30) <= d["y"] <= (by + 30)
            and (d["x"] <= left_limit or d["x"] >= right_limit)
        ]
        found_gap = claim_best_dim(margin_gap_cands, min(rx, bx), gap_y_mid, max_dist=350)
        if found_gap is not None:
            gap = found_gap

        # 2. Claim Adjacent PC Width (search wide vertically around top & bottom of Adjacent box)
        adj_w_cands = [
            d for d in dimensions
            if (bx - 80) <= d["x"] <= (bx + bw + 80)
            and ((ry + rh - 15) <= d["y"] <= (by + 80) or (by + bh - 40) <= d["y"] <= (by + bh + 160))
        ]
        found_adj_w = claim_best_dim(adj_w_cands, bx + bw / 2.0, by, max_dist=380)
        if found_adj_w is not None:
            adj_w = found_adj_w

        # 3. Claim Adjacent PC Depth (search along left and right sides of Adjacent box)
        adj_d_cands = [
            d for d in dimensions
            if (by - 40) <= d["y"] <= (by + bh + 60)
            and (d["x"] <= bx + 60 or d["x"] >= bx + bw - 60)
        ]
        found_adj_d = claim_best_dim(adj_d_cands, bx, by + bh / 2.0, max_dist=350)
        if found_adj_d is not None:
            adj_d = found_adj_d

        # Ensure wide Adjacent box stays horizontal if both numbers were found
        if bw > bh * 1.3 and adj_w < adj_d:
            adj_w, adj_d = adj_d, adj_w

        # 4. Claim Remote PC Width (search above/below Remote box)
        rem_w_cands = [
            d for d in dimensions
            if (rx - 80) <= d["x"] <= (rx + rw + 80)
            and d["y"] <= (ry + rh + 50)
        ]
        found_rem_w = claim_best_dim(rem_w_cands, rx + rw / 2.0, ry, max_dist=320)
        if found_rem_w is not None:
            rem_w = found_rem_w

        # 5. Claim Remote PC Depth (search left/right of Remote box)
        rem_d_cands = [
            d for d in dimensions
            if (ry - 40) <= d["y"] <= (ry + rh + 50)
            and (d["x"] <= rx + 60 or d["x"] >= rx + rw - 60)
        ]
        found_rem_d = claim_best_dim(rem_d_cands, rx + rw, ry + rh / 2.0, max_dist=320)
        if found_rem_d is not None:
            rem_d = found_rem_d

    elif len(boxes) == 1 and dimensions:
        bx, by, bw, bh = boxes[0]
        w_cands = [d for d in dimensions if (bx - 80) <= d["x"] <= (bx + bw + 80)]
        d_cands = [d for d in dimensions if (by - 40) <= d["y"] <= (by + bh + 60)]
        found_w = claim_best_dim(w_cands, bx + bw / 2.0, by, max_dist=380)
        found_d = claim_best_dim(d_cands, bx + bw, by + bh / 2.0, max_dist=350)
        if found_w is not None:
            adj_w = found_w
        if found_d is not None:
            adj_d = found_d
        if bw > bh * 1.3 and adj_w < adj_d:
            adj_w, adj_d = adj_d, adj_w

    with col2:
        st.subheader("Dimensioned Polygon Output")
        detected_vals = [f"{d['val']}m" for d in dimensions]
        st.caption(f"Detected {len(boxes)} zone(s) | Numbers read: {', '.join(detected_vals) if detected_vals else 'None'}")

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
        if (ax1 - ax0) >= 140:
            draw.text((ax0 + 10, (ay0 + ay1) // 2 - 8), f"Adjacent PC ({adj_area} sq.m)", fill="black")
        else:
            draw.text((ax0, ay0 - 20), f"Adjacent PC ({adj_area} sq.m)", fill="black")

        draw_dimension_arrow(draw, (ax0, ay1), (ax1, ay1), f"{adj_w}m", offset=35)
        draw_dimension_arrow(draw, (ax1, ay0), (ax1, ay1), f"{adj_d}m", offset=25)

        if has_remote:
            ry1 = ay0 - int(gap * scale)
            ry0 = ry1 - int(rem_d * scale)
            rx1 = ax0 + int(rem_w * scale)

            draw.rectangle([ax0, ry0, rx1, ry1], outline="black", fill="#fef3c7", width=3)
            rem_area = round(rem_w * rem_d, 2)
            if (rx1 - ax0) >= 140:
                draw.text((ax0 + 10, (ry0 + ry1) // 2 - 8), f"Remote PC ({rem_area} sq.m)", fill="black")
            else:
                draw.text((ax0, ry1 + 6), f"Remote PC ({rem_area} sq.m)", fill="black")

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
            mime="image/jpeg"
        )
