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
    """Multi-pass zone detector that bridges customer access openings without losing smaller Remote PCs."""
    h, w = img_cv.shape[:2]
    page_area = h * w
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
    )

    # Pass 1: Light closing (catches intact boxes like Remote PC without merging nearby labels)
    # Pass 2: Wide horizontal/vertical closing (bridges customer entrance gaps on Adjacent PC)
    kernels = [
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (45, 7)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 45)),
    ]

    raw_boxes = []
    for k in kernels:
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k)
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

    # Deduplicate overlapping boxes (keep largest bridged version of each distinct zone)
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
                if inter_area > 0.45 * min(bw * bh, uw * uh):
                    is_dup = True
                    break
        if not is_dup:
            unique_boxes.append((bx, by, bw, bh))

    # Remove any outer container box that wraps around other detected boxes
    inner_boxes = []
    for i, (bx, by, bw, bh) in enumerate(unique_boxes):
        contains_others = False
        for j, (ox, oy, ow, oh) in enumerate(unique_boxes):
            if i != j and bx <= ox and by <= oy and (bx + bw) >= (ox + ow) and (by + bh) >= (oy + oh):
                contains_others = True
                break
        if not contains_others:
            inner_boxes.append((bx, by, bw, bh))

    # Sort top-to-bottom by Y coordinate (Remote PC = top, Adjacent PC = bottom)
    inner_boxes.sort(key=box_top_y)
    return inner_boxes


def extract_dimensions(pil_img):
    """Dual-pass OCR that avoids washing out faint lines while filtering vertical arrow-line artifacts."""
    enhanced = ImageEnhance.Contrast(pil_img).enhance(1.9)
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.6)

    gray = np.array(pil_img.convert("L"), dtype=np.float32)
    bg = cv2.GaussianBlur(gray, (61, 61), 0)
    norm = np.clip((gray / (bg + 1e-5)) * 235, 0, 255).astype(np.uint8)
    norm_rgb = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)

    dimensions = []
    for img_variant in [np.array(enhanced), norm_rgb]:
        ocr_results, _ = ocr(img_variant)
        if not ocr_results:
            continue
        for item in ocr_results:
            bbox, text, _ = item
            pts = np.array(bbox, dtype=float)
            min_pt = pts.min(axis=0)
            max_pt = pts.max(axis=0)
            bw = max_pt[0] - min_pt[0]
            bh = max_pt[1] - min_pt[1]
            # Reject tall, skinny vertical arrow lines misread as '1' or '1.6'
            if bw > 0 and (bh / bw) > 2.3:
                continue

            raw_str = str(text).strip()
            has_unit = "m" in raw_str.lower()
            clean_text = raw_str.replace(",", ".").replace("O", "0").replace("o", "0")
            for match in re.finditer(r"(\d+\.?\d*)", clean_text):
                token = match.group(1)
                val = float(token)
                if val > 100:
                    val /= 1000.0
                if not (0.3 <= val <= 50.0):
                    continue
                # Ignore bare single-digit '1' without 'm' (almost always a vertical dimension tick)
                if val == 1.0 and not has_unit:
                    continue
                # Snap noisy OCR decimal hallucinations if no decimal point was written
                if "." not in token and val != round(val):
                    val = round(val)

                cx = int((min_pt[0] + max_pt[0]) / 2)
                cy = int((min_pt[1] + max_pt[1]) / 2)
                val = round(val, 2)

                # Deduplicate detections from the two passes
                already = False
                for existing in dimensions:
                    if abs(existing["x"] - cx) < 35 and abs(existing["y"] - cy) < 35 and existing["val"] == val:
                        if has_unit:
                            existing["has_unit"] = True
                        already = True
                        break
                if not already:
                    dimensions.append({"val": val, "x": cx, "y": cy, "has_unit": has_unit})

    return dimensions


def claim_best_dim(candidates, target_x, target_y, used_ids, prefer_larger=False):
    avail = [d for d in candidates if id(d) not in used_ids]
    if not avail:
        return None
    if prefer_larger:
        best = max(avail, key=lambda d: (d["val"], -((d["x"] - target_x) ** 2 + (d["y"] - target_y) ** 2)))
    else:
        best = min(
            avail,
            key=lambda d: ((d["x"] - target_x) ** 2 + (d["y"] - target_y) ** 2) - (3600 if d["has_unit"] else 0),
        )
    used_ids.add(id(best))
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
        st.caption("Green boxes = detected PC zones | Red dots = numbers read by OCR")

    adj_w, adj_d, has_remote, rem_w, rem_d, gap = 4.0, 1.5, False, 4.0, 1.5, 2.0
    used_ids = set()

    if boxes and dimensions:
        if len(boxes) >= 2:
            has_remote = True
            rx, ry, rw, rh = boxes[0]   # Top box = Remote PC
            bx, by, bw, bh = boxes[-1]  # Bottom box = Adjacent PC

            # 1. Claim Clear Gap strictly on left/right outer margins between the two boxes
            gap_y_mid = (ry + rh + by) / 2
            margin_gap_cands = [
                d for d in dimensions
                if (ry + rh - 10) <= d["y"] <= (by + 15)
                and (d["x"] <= min(rx, bx) + 45 or d["x"] >= max(rx + rw, bx + bw) - 45)
            ]
            val = claim_best_dim(margin_gap_cands, min(rx, bx), gap_y_mid, used_ids)
            if val is not None:
                gap = val

            # 2. Claim Adjacent PC Width (centered horizontally over/under bottom box, including walkway space)
            adj_w_cands = [
                d for d in dimensions
                if (bx + bw * 0.15) <= d["x"] <= (bx + bw * 0.85)
                and (ry + rh - 10) <= d["y"] <= (by + bh + 90)
            ]
            val = claim_best_dim(adj_w_cands, bx + bw / 2, by, used_ids, prefer_larger=(bw > bh))
            if val is not None:
                adj_w = val

            # 3. Claim Adjacent PC Depth (along left or right vertical side of bottom box)
            adj_d_cands = [
                d for d in dimensions
                if (by - 25) <= d["y"] <= (by + bh + 25)
                and (d["x"] <= bx + 55 or d["x"] >= bx + bw - 55)
            ]
            val = claim_best_dim(adj_d_cands, bx, by + bh / 2, used_ids)
            if val is not None:
                adj_d = val

            # 4. Claim Remote PC Width (above or below top box, near its horizontal center)
            rem_w_cands = [
                d for d in dimensions
                if (rx - 30) <= d["x"] <= (rx + rw + 30)
                and (ry - 95) <= d["y"] <= (ry + rh + 45)
            ]
            val = claim_best_dim(rem_w_cands, rx + rw / 2, ry, used_ids)
            if val is not None:
                rem_w = val

            # 5. Claim Remote PC Depth (beside top box's vertical center)
            rem_d_cands = [
                d for d in dimensions
                if (ry - 25) <= d["y"] <= (ry + rh + 25)
                and (d["x"] <= rx + 50 or d["x"] >= rx + rw - 50)
            ]
            val = claim_best_dim(rem_d_cands, rx + rw, ry + rh / 2, used_ids)
            if val is not None:
                rem_d = val

        else:
            bx, by, bw, bh = boxes[0]
            adj_w_cands = [
                d for d in dimensions
                if (bx - 30) <= d["x"] <= (bx + bw + 30)
                and (by - 95) <= d["y"] <= (by + bh + 95)
            ]
            val = claim_best_dim(adj_w_cands, bx + bw / 2, by, used_ids, prefer_larger=(bw > bh))
            if val is not None:
                adj_w = val

            adj_d_cands = [
                d for d in dimensions
                if (by - 25) <= d["y"] <= (by + bh + 25)
                and (d["x"] <= bx + 55 or d["x"] >= bx + bw - 55)
            ]
            val = claim_best_dim(adj_d_cands, bx, by + bh / 2, used_ids)
            if val is not None:
                adj_d = val

        # Orientation guard: if detected box is clearly wider than tall, ensure width >= depth
        bx, by, bw, bh = boxes[-1]
        if bw > bh * 1.35 and adj_w < adj_d:
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

        draw.rectangle([ax0, ay0, ax1, ay1], outline="black", fill="#dbeafe", width=3)
        adj_area = round(adj_w * adj_d, 2)
        adj_label = f"Adjacent PC ({adj_area} sq.m)"
        if (ax1 - ax0) >= 165:
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
            if (rx1 - ax0) >= 165:
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
