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
    """Finds inner pavement cafe zones, bridging customer access openings."""
    h, w = img_cv.shape[:2]
    page_area = h * w
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    norm = normalize_lighting(gray)
    blurred = cv2.GaussianBlur(norm, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
    )

    candidate_boxes = []
    # Multi-pass morphological closing:
    # Pass 1 (5x5): captures closed boxes like Remote PC cleanly
    # Pass 2 (45x15): bridges wide horizontal customer access openings on Adjacent PC
    # Pass 3 (15x45): bridges vertical access openings
    for kw, kh in [(5, 5), (45, 15), (15, 45)]:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
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
            if 4 <= len(approx) <= 8 and bw > 30 and bh > 25 and rect_fill > 0.65:
                candidate_boxes.append((bx, by, bw, bh))

    candidate_boxes.sort(key=box_area, reverse=True)
    unique_boxes = []
    for bx, by, bw, bh in candidate_boxes:
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

    # Remove giant outer container boxes that wrap around multiple inner zones
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


def extract_dimensions(pil_img):
    """Dual-pass OCR with artifact filtering and 'm'-unit prioritization."""
    enhanced = ImageEnhance.Contrast(pil_img).enhance(1.9)
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.6)
    img_np = np.array(enhanced)

    gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    norm_np = cv2.cvtColor(normalize_lighting(gray), cv2.COLOR_GRAY2RGB)

    dimensions = []
    for pass_img in [img_np, norm_np]:
        ocr_results, _ = ocr(pass_img)
        if not ocr_results:
            continue
        for item in ocr_results:
            bbox, text, _ = item
            pts = np.array(bbox, dtype=float)
            min_x, min_y = pts.min(axis=0)
            max_x, max_y = pts.max(axis=0)
            box_w = max_x - min_x
            box_h = max_y - min_y

            # Filter tall skinny arrow-line artifacts (e.g. vertical bar misread as 1 or 0.6)
            if box_w > 0 and (box_h / box_w) > 2.3:
                continue

            raw_str = str(text).strip()
            has_unit = "m" in raw_str.lower()
            clean_text = (
                raw_str.replace(",", ".")
                .replace("O", "0")
                .replace("o", "0")
                .replace("l", "1")
                .replace("I", "1")
            )
            for match in re.finditer(r"(\d+\.?\d*)", clean_text):
                val = float(match.group(1))
                if val > 100:
                    val /= 1000.0
                # Block standalone '1' / '1.0' without an 'm' unit (common vertical-line artifact)
                if val == 1.0 and not has_unit:
                    continue
                if 0.4 <= val <= 50.0:
                    cx = int((min_x + max_x) / 2)
                    cy = int((min_y + max_y) / 2)
                    # Deduplicate across both OCR passes
                    already_found = False
                    for existing in dimensions:
                        if abs(existing["x"] - cx) < 25 and abs(existing["y"] - cy) < 25:
                            if has_unit and not existing["has_unit"]:
                                existing["val"] = round(val, 2)
                                existing["has_unit"] = True
                            already_found = True
                            break
                    if not already_found:
                        dimensions.append(
                            {
                                "val": round(val, 2),
                                "x": cx,
                                "y": cy,
                                "has_unit": has_unit,
                                "used": False,
                            }
                        )
    return dimensions


def pick_best_dimension(candidates, target_x, target_y, max_dist=320):
    """Picks nearest unused dimension with a proximity bonus for 'm'-labeled values."""
    best = None
    best_score = float("inf")
    for d in candidates:
        if d["used"]:
            continue
        dist = ((d["x"] - target_x) ** 2 + (d["y"] - target_y) ** 2) ** 0.5
        if dist > max_dist:
            continue
        score = dist - (60.0 if d["has_unit"] else 0.0)
        if score < best_score:
            best_score = score
            best = d
    if best is not None:
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

    if boxes and dimensions:
        # Bottom box = Adjacent PC
        bx, by, bw, bh = boxes[-1]

        if len(boxes) >= 2:
            has_remote = True
            rx, ry, rw, rh = next(iter(boxes))

            # 1. Claim Clear Gap strictly along the left or right outer margin between the two boxes
            gap_top = ry + rh
            gap_bot = by
            gap_mid_y = (gap_top + gap_bot) / 2
            left_margin = min(rx, bx) + 45
            right_margin = max(rx + rw, bx + bw) - 45

            margin_cands = [
                d
                for d in dimensions
                if (gap_top - 30) <= d["y"] <= (gap_bot + 30)
                and (d["x"] <= left_margin or d["x"] >= right_margin)
            ]
            g_val = pick_best_dimension(
                margin_cands, min(rx, bx), gap_mid_y, max_dist=260
            )
            if g_val is not None:
                gap = g_val

            # 2. Claim Adjacent PC Width from centered label above or below the bottom box
            adj_w_cands = [
                d
                for d in dimensions
                if abs(d["x"] - (bx + bw / 2)) < (bw * 0.55)
                and (abs(d["y"] - by) < 150 or abs(d["y"] - (by + bh)) < 150)
            ]
            aw_val = pick_best_dimension(adj_w_cands, bx + bw / 2, by, max_dist=250)
            if aw_val is None:
                aw_val = pick_best_dimension(dimensions, bx + bw / 2, by, max_dist=320)
            if aw_val is not None:
                adj_w = aw_val

            # 3. Claim Adjacent PC Depth beside its vertical midpoint
            adj_d_cands = [
                d
                for d in dimensions
                if abs(d["y"] - (by + bh / 2)) < (bh * 0.75)
            ]
            ad_val = pick_best_dimension(adj_d_cands, bx, by + bh / 2, max_dist=280)
            if ad_val is None:
                ad_val = pick_best_dimension(dimensions, bx + bw, by + bh / 2, max_dist=320)
            if ad_val is not None:
                adj_d = ad_val

            # Enforce physical box orientation for Adjacent PC:
            # If the detected rectangle is wider than tall (bw > bh * 1.15), width must be >= depth
            if bw > (bh * 1.15) and adj_w < adj_d:
                adj_w, adj_d = adj_d, adj_w

            # 4. Claim Remote PC Width (above/below top box center) & Depth (beside top box midpoint)
            rem_w_cands = [
                d
                for d in dimensions
                if abs(d["x"] - (rx + rw / 2)) < max(rw * 0.85, 110)
                and d["y"] <= (ry + rh + 40)
            ]
            rw_val = pick_best_dimension(rem_w_cands, rx + rw / 2, ry, max_dist=250)
            if rw_val is None:
                rw_val = pick_best_dimension(dimensions, rx + rw / 2, ry, max_dist=300)
            if rw_val is not None:
                rem_w = rw_val

            rem_d_cands = [
                d
                for d in dimensions
                if abs(d["y"] - (ry + rh / 2)) < max(rh * 0.85, 100)
            ]
            rd_val = pick_best_dimension(rem_d_cands, rx + rw, ry + rh / 2, max_dist=260)
            if rd_val is None:
                rd_val = pick_best_dimension(dimensions, rx, ry + rh / 2, max_dist=280)
            if rd_val is not None:
                rem_d = rd_val
            elif rw_val is not None and abs(rw - rh) < max(rw, rh) * 0.25:
                # If Remote PC is drawn as a square and only one side label was readable, match square side
                rem_d = rem_w
        else:
            aw_val = pick_best_dimension(dimensions, bx + bw / 2, by, max_dist=320)
            if aw_val is not None:
                adj_w = aw_val
            ad_val = pick_best_dimension(dimensions, bx, by + bh / 2, max_dist=320)
            if ad_val is not None:
                adj_d = ad_val
            if bw > (bh * 1.15) and adj_w < adj_d:
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
        scale = min(500 / total_w_m, 500 / total_h_m)

        out = Image.new("RGB", (800, 800), "white")
        draw = ImageDraw.Draw(out)

        ax0 = 120
        ay1 = 680
        ay0 = ay1 - int(adj_d * scale)
        ax1 = ax0 + int(adj_w * scale)

        draw.line([(60, ay1), (740, ay1)], fill="black", width=4)
        draw.text((65, ay1 + 12), "PREMISES / BUILDING FRONTAGE", fill="black")

        # Solid, fully enclosed Adjacent PC rectangle
        draw.rectangle([ax0, ay0, ax1, ay1], outline="black", fill="#dbeafe", width=3)
        adj_area = round(adj_w * adj_d, 2)
        adj_label = f"Adjacent PC ({adj_area} sq.m)"
        if (ax1 - ax0) < 175:
            draw.text((ax0, ay0 - 16), adj_label, fill="black")
        else:
            draw.text((ax0 + 10, (ay0 + ay1) // 2 - 8), adj_label, fill="black")

        draw_dimension_arrow(draw, (ax0, ay1), (ax1, ay1), f"{adj_w}m", offset=35)
        draw_dimension_arrow(draw, (ax1, ay0), (ax1, ay1), f"{adj_d}m", offset=25)

        if has_remote:
            ry1 = ay0 - int(gap * scale)
            ry0 = ry1 - int(rem_d * scale)
            rx1 = ax0 + int(rem_w * scale)

            # Solid, fully enclosed Remote PC rectangle
            draw.rectangle([ax0, ry0, rx1, ry1], outline="black", fill="#fef3c7", width=3)
            rem_area = round(rem_w * rem_d, 2)
            rem_label = f"Remote PC ({rem_area} sq.m)"
            if (rx1 - ax0) < 175:
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
