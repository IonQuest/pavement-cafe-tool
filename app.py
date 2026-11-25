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


def extract_dimensions(pil_img):
    """Dual-pass OCR (enhanced color + normalized grayscale) to catch all handwritten labels."""
    dimensions = []
    # Pass 1: Contrast & sharpness enhanced
    enh = ImageEnhance.Contrast(pil_img).enhance(1.9)
    enh = ImageEnhance.Sharpness(enh).enhance(1.6)
    res1, _ = ocr(np.array(enh))

    # Pass 2: Lighting normalized grayscale
    gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    bg = cv2.GaussianBlur(gray, (51, 51), 0)
    norm = cv2.divide(gray, bg, scale=255)
    res2, _ = ocr(cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB))

    all_items = (res1 or []) + (res2 or [])
    for item in all_items:
        bbox, text, _ = item
        raw_str = str(text).strip()
        has_m = "m" in raw_str.lower()
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
            # Reject bare '1' without 'm' (vertical dimension arrow tick artifact)
            if val == 1.0 and not has_m:
                continue
            if 0.2 <= val <= 50.0:
                pts = np.array(bbox, dtype=float)
                min_pt = pts.min(axis=0)
                max_pt = pts.max(axis=0)
                bw_t = max_pt[0] - min_pt[0]
                bh_t = max_pt[1] - min_pt[1]
                # Reject tall skinny arrow-line misreads
                if bw_t > 0 and (bh_t / bw_t) > 2.4:
                    continue
                cx = int((min_pt[0] + max_pt[0]) / 2)
                cy = int((min_pt[1] + max_pt[1]) / 2)
                # Deduplicate across the two OCR passes
                is_dup = False
                for existing in dimensions:
                    if abs(existing["x"] - cx) < 35 and abs(existing["y"] - cy) < 35:
                        if has_m and not existing["has_m"]:
                            existing["val"] = round(val, 2)
                            existing["has_m"] = True
                        is_dup = True
                        break
                if not is_dup:
                    dimensions.append({
                        "val": round(val, 2),
                        "x": cx,
                        "y": cy,
                        "has_m": has_m,
                        "used": False,
                    })
    return dimensions


def detect_cafe_rectangles(img_cv, dimensions):
    """Exact v20/v22 zone detector: finds both closed boxes and C-shaped open-entrance boxes."""
    h, w = img_cv.shape[:2]
    page_area = h * w
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    bg = cv2.GaussianBlur(gray, (51, 51), 0)
    norm = cv2.divide(gray, bg, scale=255)
    blurred = cv2.GaussianBlur(norm, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
    )

    candidate_boxes = []

    # Pass A: Light closing (catches intact boxes like 2m x 2m Remote PC)
    k_light = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed_light = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k_light)
    contours_a, _ = cv2.findContours(closed_light, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours_a:
        area = cv2.contourArea(c)
        if area < (page_area * 0.002) or area > (page_area * 0.45):
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        if bw > w * 0.08 and bh > h * 0.06 and bw < w * 0.85:
            candidate_boxes.append((bx, by, bw, bh))

    # Pass B: Horizontal closing (bridges wide customer entrance openings like 5m x 2m Adjacent PC)
    bridge_w = max(45, int(w * 0.14))
    k_horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (bridge_w, 3))
    closed_horiz = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k_horiz)
    contours_b, _ = cv2.findContours(closed_horiz, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours_b:
        area = cv2.contourArea(c)
        if area < (page_area * 0.003) or area > (page_area * 0.45):
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        if bw > w * 0.12 and bh > h * 0.06 and bw < w * 0.85:
            candidate_boxes.append((bx, by, bw, bh))

    # Filter out boxes touching page edges or enclosing an OCR label inside them
    clean_boxes = []
    for bx, by, bw, bh in candidate_boxes:
        if by < 10 or (by + bh) > (h - 10):
            continue
        has_text_inside = False
        for d in dimensions:
            if (bx + 15) < d["x"] < (bx + bw - 15) and (by + 15) < d["y"] < (by + bh - 15):
                has_text_inside = True
                break
        if not has_text_inside:
            clean_boxes.append((bx, by, bw, bh))

    # Deduplicate overlapping detections (keep largest bounding box for each zone)
    clean_boxes.sort(key=box_area, reverse=True)
    unique_boxes = []
    for bx, by, bw, bh in clean_boxes:
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

    # Remove any outer container box that wraps around other detected boxes
    inner_boxes = []
    for i, (bx, by, bw, bh) in enumerate(unique_boxes):
        wraps_other = False
        for j, (ox, oy, ow, oh) in enumerate(unique_boxes):
            if i != j and bx <= ox and by <= oy and (bx + bw) >= (ox + ow) and (by + bh) >= (oy + oh):
                wraps_other = True
                break
        if not wraps_other:
            inner_boxes.append((bx, by, bw, bh))

    inner_boxes.sort(key=box_top_y)
    return inner_boxes


def claim_closest(candidates, target_x, target_y):
    """Picks the closest unused dimension (giving priority to 'm'-labeled numbers)."""
    unused = [d for d in candidates if not d["used"]]
    if not unused:
        return None
    best = min(
        unused,
        key=lambda d: ((d["x"] - target_x) ** 2 + (d["y"] - target_y) ** 2) ** 0.5
        - (60 if d["has_m"] else 0),
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

    if boxes and dimensions:
        bx, by, bw, bh = boxes[-1]  # Bottom box = Adjacent PC

        if len(boxes) >= 2:
            has_remote = True
            rx, ry, rw, rh = boxes[0]  # Top box = Remote PC

            # 1. Claim Clear Gap strictly on the left/right outer margins between the two boxes
            left_bound = min(bx, rx)
            right_bound = max(bx + bw, rx + rw)
            gap_candidates = [
                d
                for d in dimensions
                if (ry + rh - 25) <= d["y"] <= (by + 25)
                and (d["x"] <= left_bound + 25 or d["x"] >= right_bound - 25)
            ]
            g_val = claim_closest(gap_candidates, left_bound, (ry + rh + by) / 2)
            if g_val is not None:
                gap = g_val

            # 2. Claim Remote PC Width strictly above the top edge (generous 280px ceiling)
            rem_w_candidates = [
                d
                for d in dimensions
                if (ry - 280) <= d["y"] <= (ry + 10)
                and (rx - 40) <= d["x"] <= (rx + rw + 40)
            ]
            rw_val = claim_closest(rem_w_candidates, rx + rw / 2, ry - 40)
            if rw_val is not None:
                rem_w = rw_val

            # 3. Claim Remote PC Depth beside its left/right edges
            rem_d_candidates = [
                d
                for d in dimensions
                if (ry - 25) <= d["y"] <= (ry + rh + 25)
                and (d["x"] <= rx + 10 or d["x"] >= rx + rw - 10)
            ]
            rd_val = claim_closest(rem_d_candidates, rx + rw + 30, ry + rh / 2)
            if rd_val is not None:
                rem_d = rd_val

        # 4. Claim Adjacent PC Width (centered above or below the bottom box)
        adj_w_candidates = [
            d
            for d in dimensions
            if (by - 180) <= d["y"] <= (by + bh + 180)
            and (bx + 25) <= d["x"] <= (bx + bw - 25)
        ]
        aw_val = claim_closest(adj_w_candidates, bx + bw / 2, by - 30)
        if aw_val is not None:
            adj_w = aw_val

        # 5. Claim Adjacent PC Depth (beside its left/right edges)
        adj_d_candidates = [
            d
            for d in dimensions
            if (by - 25) <= d["y"] <= (by + bh + 25)
            and (d["x"] <= bx + 15 or d["x"] >= bx + bw - 15)
        ]
        ad_val = claim_closest(adj_d_candidates, bx - 30, by + bh / 2)
        if ad_val is not None:
            adj_d = ad_val

        # Enforce visual aspect-ratio orientation so wide boxes stay wide
        if bw > bh * 1.35 and adj_w < adj_d:
            adj_w, adj_d = adj_d, adj_w
        if has_remote and rw > rh * 1.35 and rem_w < rem_d:
            rem_w, rem_d = rem_d, rem_w

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
