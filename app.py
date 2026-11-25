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
    """Detects Pavement Cafe zones, merges split C-shapes, and filters out dimension arrows."""
    h, w = img_cv.shape[:2]
    page_area = h * w
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Normalize warm phone photo shadows
    bg = cv2.GaussianBlur(gray, (51, 51), 0)
    norm = cv2.divide(gray, bg, scale=255)
    blurred = cv2.GaussianBlur(norm, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
    )

    # Light closing to keep distinct walls separate from dimension arrows
    k_light = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k_light)

    raw_boxes = []

    # 1. Connected components catch open C-shaped zones cleanly
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(closed)
    for idx in range(1, num_labels):
        bx = int(stats[idx, cv2.CC_STAT_LEFT])
        by = int(stats[idx, cv2.CC_STAT_TOP])
        bw = int(stats[idx, cv2.CC_STAT_WIDTH])
        bh = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = bw * bh
        # Reject thin arrow strips (bh < 65 or bw < 65) and huge page borders
        if bw >= 65 and bh >= 65 and (page_area * 0.002) <= area <= (page_area * 0.45):
            raw_boxes.append((bx, by, bw, bh))

    # 2. Contours catch closed rectangles
    contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if area < (page_area * 0.002) or area > (page_area * 0.45):
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        bx, by, bw, bh = cv2.boundingRect(approx)
        rect_fill = area / float(bw * bh) if (bw * bh) > 0 else 0
        if 4 <= len(approx) <= 8 and bw >= 65 and bh >= 65 and rect_fill > 0.55:
            raw_boxes.append((bx, by, bw, bh))

    # Filter out any box that wraps around a dimension arrow + red OCR text dot
    clean_boxes = []
    for bx, by, bw, bh in raw_boxes:
        has_ocr_inside = False
        for d in dimensions:
            if (bx + 10) <= d["x"] <= (bx + bw - 10) and (by + 10) <= d["y"] <= (by + bh - 10):
                has_ocr_inside = True
                break
        if not has_ocr_inside:
            clean_boxes.append((bx, by, bw, bh))

    # Merge boxes that sit on the same horizontal row (unites left & right halves of a C-shaped zone)
    merged = True
    while merged:
        merged = False
        next_boxes = []
        used = [False] * len(clean_boxes)
        for i in range(len(clean_boxes)):
            if used[i]:
                continue
            bx, by, bw, bh = clean_boxes[i]
            for j in range(i + 1, len(clean_boxes)):
                if used[j]:
                    continue
                ux, uy, uw, uh = clean_boxes[j]
                y_overlap = max(0, min(by + bh, uy + uh) - max(by, uy))
                min_h = min(bh, uh)
                # Same horizontal band (vertical overlap > 65%) and horizontally overlapping or within 180px gap
                x_gap = max(bx, ux) - min(bx + bw, ux + uw)
                if min_h > 0 and (y_overlap / float(min_h)) > 0.65 and x_gap < 180:
                    nx0 = min(bx, ux)
                    ny0 = min(by, uy)
                    nx1 = max(bx + bw, ux + uw)
                    ny1 = max(by + bh, uy + uh)
                    bx, by, bw, bh = nx0, ny0, nx1 - nx0, ny1 - ny0
                    used[j] = True
                    merged = True
            next_boxes.append((bx, by, bw, bh))
        clean_boxes = next_boxes

    # Remove any outer container box that encloses another distinct zone
    final_boxes = []
    for i, (bx, by, bw, bh) in enumerate(clean_boxes):
        encloses_other = False
        for j, (ox, oy, ow, oh) in enumerate(clean_boxes):
            if i != j and bx <= ox + 10 and by <= oy + 10 and (bx + bw) >= (ox + ow - 10) and (by + bh) >= (oy + oh - 10):
                encloses_other = True
                break
        if not encloses_other:
            final_boxes.append((bx, by, bw, bh))

    final_boxes.sort(key=box_top_y)
    return final_boxes


def extract_dimensions(pil_img):
    """Dual-pass OCR with handwriting normalization and arrow-artifact filtering."""
    enhanced = ImageEnhance.Contrast(pil_img).enhance(1.9)
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.6)
    img_np = np.array(enhanced)

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    bg = cv2.GaussianBlur(gray, (51, 51), 0)
    norm_gray = cv2.divide(gray, bg, scale=255)
    norm_rgb = cv2.cvtColor(norm_gray, cv2.COLOR_GRAY2RGB)

    dimensions = []
    for pass_img in (img_np, norm_rgb):
        ocr_results, _ = ocr(pass_img)
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
            box_w = float(pts[:, 0].max() - pts[:, 0].min())
            box_h = float(pts[:, 1].max() - pts[:, 1].min())
            if box_w > 0 and (box_h / box_w) > 2.3:
                continue

            for match in re.finditer(r"(\d+\.?\d*)", clean_text):
                num_str = match.group(1)
                val = float(num_str)
                if val > 100:
                    val /= 1000.0
                if val == 1.0 and not has_unit:
                    continue
                if "." not in num_str and val < 20:
                    val = float(round(val))
                if 0.4 <= val <= 50.0:
                    cx = int(pts[:, 0].mean())
                    cy = int(pts[:, 1].mean())
                    if not any(abs(d["x"] - cx) < 25 and abs(d["y"] - cy) < 25 for d in dimensions):
                        dimensions.append({
                            "val": round(val, 2),
                            "x": cx,
                            "y": cy,
                            "has_unit": has_unit,
                        })
    return dimensions


def pick_best_dimension(candidates, target_x, target_y, used_indices):
    best_idx = None
    best_score = float("inf")
    for idx, d in candidates:
        if idx in used_indices:
            continue
        dist = ((d["x"] - target_x) ** 2 + (d["y"] - target_y) ** 2) ** 0.5
        if d.get("has_unit"):
            dist -= 60.0
        if dist < best_score:
            best_score = dist
            best_idx = idx
    return best_idx


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
        st.subheader("Uploaded Plan & Detection Overlay")
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
    used_dims = set()

    if boxes and dimensions:
        indexed_dims = list(enumerate(dimensions))
        bx, by, bw, bh = boxes[-1]  # Bottom box = Adjacent PC

        if len(boxes) >= 2:
            has_remote = True
            rx, ry, rw, rh = next(iter(boxes))  # Top box = Remote PC
            left_bound = min(bx, rx)
            right_bound = max(bx + bw, rx + rw)

            # 1. Claim Clear Gap along left or right outer margin between the two zones
            gap_candidates = [
                (i, d) for i, d in indexed_dims
                if (ry + rh - 20) <= d["y"] <= (by + 20)
                and (d["x"] <= left_bound + 45 or d["x"] >= right_bound - 45)
            ]
            g_idx = pick_best_dimension(gap_candidates, left_bound - 40, (ry + rh + by) / 2, used_dims)
            if g_idx is not None:
                gap = dimensions[g_idx]["val"]
                used_dims.add(g_idx)

            # 2. Claim Adjacent PC width centered over/under the bottom zone
            adj_w_candidates = [
                (i, d) for i, d in indexed_dims
                if (bx - 20) <= d["x"] <= (bx + bw + 20)
                and ((ry + rh - 10) <= d["y"] <= (by + 30) or (by + bh - 20) <= d["y"] <= (by + bh + 120))
            ]
            aw_idx = pick_best_dimension(adj_w_candidates, bx + bw / 2, by - 35, used_dims)
            if aw_idx is not None:
                adj_w = dimensions[aw_idx]["val"]
                used_dims.add(aw_idx)

            # 3. Claim Adjacent PC depth beside bottom box
            adj_d_candidates = [
                (i, d) for i, d in indexed_dims
                if (by - 25) <= d["y"] <= (by + bh + 25)
                and (d["x"] <= bx + 45 or d["x"] >= bx + bw - 45)
            ]
            ad_idx = pick_best_dimension(adj_d_candidates, bx - 40, by + bh / 2, used_dims)
            if ad_idx is not None:
                adj_d = dimensions[ad_idx]["val"]
                used_dims.add(ad_idx)

            # 4. Claim Remote PC width above/below top box
            rem_w_candidates = [
                (i, d) for i, d in indexed_dims
                if (rx - 40) <= d["x"] <= (rx + rw + 40)
                and ((ry - 130) <= d["y"] <= (ry + 25) or (ry + rh - 20) <= d["y"] <= (ry + rh + 75))
            ]
            rw_idx = pick_best_dimension(rem_w_candidates, rx + rw / 2, ry - 40, used_dims)
            if rw_idx is not None:
                rem_w = dimensions[rw_idx]["val"]
                used_dims.add(rw_idx)

            # 5. Claim Remote PC depth beside top box
            rem_d_candidates = [
                (i, d) for i, d in indexed_dims
                if (ry - 25) <= d["y"] <= (ry + rh + 25)
                and (d["x"] <= rx + 45 or d["x"] >= rx + rw - 45)
            ]
            rd_idx = pick_best_dimension(rem_d_candidates, rx + rw + 45, ry + rh / 2, used_dims)
            if rd_idx is not None:
                rem_d = dimensions[rd_idx]["val"]
                used_dims.add(rd_idx)

            if rw > rh * 1.3 and rem_w < rem_d:
                rem_w, rem_d = rem_d, rem_w
            elif rh > rw * 1.3 and rem_d < rem_w:
                rem_w, rem_d = rem_d, rem_w
        else:
            adj_w_candidates = [
                (i, d) for i, d in indexed_dims
                if (bx - 30) <= d["x"] <= (bx + bw + 30)
                and ((by - 130) <= d["y"] <= (by + 30) or (by + bh - 20) <= d["y"] <= (by + bh + 130))
            ]
            aw_idx = pick_best_dimension(adj_w_candidates, bx + bw / 2, by - 40, used_dims)
            if aw_idx is not None:
                adj_w = dimensions[aw_idx]["val"]
                used_dims.add(aw_idx)

            adj_d_candidates = [
                (i, d) for i, d in indexed_dims
                if (by - 25) <= d["y"] <= (by + bh + 25)
                and (d["x"] <= bx + 45 or d["x"] >= bx + bw - 45)
            ]
            ad_idx = pick_best_dimension(adj_d_candidates, bx - 40, by + bh / 2, used_dims)
            if ad_idx is not None:
                adj_d = dimensions[ad_idx]["val"]
                used_dims.add(ad_idx)

        if bw > bh * 1.3 and adj_w < adj_d:
            adj_w, adj_d = adj_d, adj_w
        elif bh > bw * 1.3 and adj_d < adj_w:
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
        scale = min(480 / total_w_m, 480 / total_h_m)

        out = Image.new("RGB", (800, 800), "white")
        draw = ImageDraw.Draw(out)

        ax0 = 140
        ay1 = 680
        ay0 = ay1 - int(adj_d * scale)
        ax1 = ax0 + int(adj_w * scale)

        draw.line([(60, ay1), (740, ay1)], fill="black", width=4)
        draw.text((65, ay1 + 12), "PREMISES / BUILDING FRONTAGE", fill="black")

        draw.rectangle([ax0, ay0, ax1, ay1], outline="black", fill="#dbeafe", width=3)
        adj_area = round(adj_w * adj_d, 2)
        label_y = (ay0 + ay1) // 2 - 6 if (ay1 - ay0) >= 36 else ay0 - 20
        draw.text((ax0 + 8, label_y), f"Adjacent PC ({adj_area} sq.m)", fill="black")

        draw_dimension_arrow(draw, (ax0, ay1), (ax1, ay1), f"{adj_w}m", offset=38)
        draw_dimension_arrow(draw, (ax1, ay0), (ax1, ay1), f"{adj_d}m", offset=25)

        if has_remote:
            ry1 = ay0 - int(gap * scale)
            ry0 = ry1 - int(rem_d * scale)
            rx1 = ax0 + int(rem_w * scale)

            draw.rectangle([ax0, ry0, rx1, ry1], outline="black", fill="#fef3c7", width=3)
            rem_area = round(rem_w * rem_d, 2)
            r_label_y = (ry0 + ry1) // 2 - 6 if (ry1 - ry0) >= 36 else ry0 - 20
            draw.text((ax0 + 8, r_label_y), f"Remote PC ({rem_area} sq.m)", fill="black")

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
