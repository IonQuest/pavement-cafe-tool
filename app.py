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


def normalize_paper_lighting(gray):
    """Removes warm shadows/gradients on phone photos while preserving crisp lines."""
    bg = cv2.GaussianBlur(gray, (51, 51), 0)
    norm = cv2.divide(gray, bg, scale=255)
    return norm


def detect_cafe_zones(img_cv):
    """
    Multi-pass detector:
    Pass 1 (light closing): catches intact smaller boxes like Remote PC (2m x 2m).
    Pass 2 (wide closing): bridges customer access gaps like Adjacent PC (5m x 2m).
    """
    h, w = img_cv.shape[:2]
    page_area = h * w
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    norm = normalize_paper_lighting(gray)
    blurred = cv2.GaussianBlur(norm, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
    )

    candidate_boxes = []
    # Two closing kernels: small for intact boxes, wide for boxes with entrance gaps
    kernels = [
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (55, 15)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (15, 55)),
    ]

    for kernel in kernels:
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
                candidate_boxes.append((bx, by, bw, bh))

    # Deduplicate overlapping boxes (keep the cleaner/wider bounding box when merged)
    candidate_boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    unique_boxes = []
    for bx, by, bw, bh in candidate_boxes:
        is_dup = False
        for ux, uy, uw, uh in unique_boxes:
            # Check significant overlap (IoU > 0.45)
            ix1 = max(bx, ux)
            iy1 = max(by, uy)
            ix2 = min(bx + bw, ux + uw)
            iy2 = min(by + bh, uy + uh)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                smaller_area = min(bw * bh, uw * uh)
                if inter / float(smaller_area) > 0.45:
                    is_dup = True
                    break
        if not is_dup:
            unique_boxes.append((bx, by, bw, bh))

    # Remove any giant outer border that encloses multiple inner boxes
    final_boxes = []
    for i, (bx, by, bw, bh) in enumerate(unique_boxes):
        encloses = 0
        for j, (ox, oy, ow, oh) in enumerate(unique_boxes):
            if i != j and bx <= ox and by <= oy and (bx + bw) >= (ox + ow) and (by + bh) >= (oy + oh):
                encloses += 1
        if encloses == 0:
            final_boxes.append((bx, by, bw, bh))

    # Sort top-to-bottom by Y coordinate (Remote PC on top, Adjacent PC on bottom)
    final_boxes.sort(key=lambda b: b[1])
    return final_boxes


def extract_all_dimensions(pil_img):
    """
    Runs dual-pass OCR (enhanced original + normalized) and filters out
    vertical arrow-shaft hallucinations like standalone '1' without 'm'.
    """
    enhanced = ImageEnhance.Contrast(pil_img).enhance(1.9)
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.6)
    img_np_1 = np.array(enhanced)

    gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    norm = normalize_paper_lighting(gray)
    img_np_2 = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)

    raw_dims = []
    for pass_idx, img_input in enumerate([img_np_1, img_np_2]):
        ocr_results, _ = ocr(img_input)
        if not ocr_results:
            continue
        for item in ocr_results:
            bbox, text, conf = item
            raw_str = str(text).strip()
            pts = np.array(bbox, dtype=float)
            bw = pts[:, 0].max() - pts[:, 0].min()
            bh = pts[:, 1].max() - pts[:, 1].min()

            # Reject tall skinny arrow shafts misread as digits
            if bw > 0 and (bh / bw) > 2.4:
                continue

            has_unit_m = "m" in raw_str.lower()
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
                if not (0.3 <= val <= 50.0):
                    continue
                # Block standalone '1' or '1.0' without an 'm' unit (arrow tick artifact)
                if abs(val - 1.0) < 0.05 and not has_unit_m:
                    continue
                # Snap noisy decimals (e.g. 2.4 -> 2.0) when no decimal point was written
                if "." not in clean_text and abs(val - round(val)) <= 0.4:
                    val = float(round(val))

                cx, cy = pts.mean(axis=0)
                raw_dims.append({
                    "val": round(val, 2),
                    "x": int(cx),
                    "y": int(cy),
                    "has_m": has_unit_m,
                    "used": False,
                })

    # Deduplicate same reading found by both OCR passes at the same spot
    unique_dims = []
    for d in raw_dims:
        dup = False
        for u in unique_dims:
            if abs(d["x"] - u["x"]) < 35 and abs(d["y"] - u["y"]) < 35 and abs(d["val"] - u["val"]) < 0.15:
                if d["has_m"]:
                    u["has_m"] = True
                dup = True
                break
        if not dup:
            unique_dims.append(d)

    return unique_dims


def claim_closest_dimension(dims, target_x, target_y, x_weight=1.0, y_weight=1.0, max_dist=350, default_val=None):
    """Finds the closest unclaimed dimension using weighted X/Y distance and marks it used."""
    best = None
    best_score = float("inf")
    for d in dims:
        if d["used"]:
            continue
        dist = (((d["x"] - target_x) * x_weight) ** 2 + ((d["y"] - target_y) * y_weight) ** 2) ** 0.5
        if dist > max_dist:
            continue
        # Slight priority bonus if 'm' unit was explicitly read
        score = dist - (30.0 if d["has_m"] else 0.0)
        if score < best_score:
            best_score = score
            best = d

    if best is not None:
        best["used"] = True
        return best["val"]
    return default_val


def assign_zone_dimensions(boxes, dims, img_w, img_h):
    """
    Strictly separates:
    1. Adjacent PC (bottom box) -> width claimed near its horizontal center, depth near its vertical side
    2. Clear Gap -> claimed along the left/right outer margin between the two boxes
    3. Remote PC (top box) -> width claimed near its horizontal center, depth near its vertical side
    """
    adj_w, adj_d = 4.0, 1.5
    has_remote = False
    rem_w, rem_d, gap = 2.0, 2.0, 2.0

    if not boxes:
        return adj_w, adj_d, has_remote, rem_w, rem_d, gap

    # Bottom-most box is Adjacent PC (against building frontage)
    ax, ay, aw, ah = boxes[-1]

    if len(boxes) >= 2:
        has_remote = True
        rx, ry, rw, rh = boxes[0]

        # Step 1: Claim Clear Gap ONLY along left/right margins between Remote bottom (ry+rh) and Adjacent top (ay)
        gap_mid_y = (ry + rh + ay) / 2.0
        left_margin_x = min(rx, ax) - 20
        right_margin_x = max(rx + rw, ax + aw) + 20

        margin_candidates = []
        for d in dims:
            if d["used"]:
                continue
            # Must be vertically between the two boxes (with small tolerance)
            if (ry + rh - 25) <= d["y"] <= (ay + 25):
                # Must sit near the left or right edge, NOT centered over the wide Adjacent box
                if d["x"] <= (min(rx, ax) + 45) or d["x"] >= (max(rx + rw, ax + aw) - 45):
                    dist = min(abs(d["x"] - left_margin_x), abs(d["x"] - right_margin_x)) + abs(d["y"] - gap_mid_y)
                    margin_candidates.append((dist, d))

        if margin_candidates:
            margin_candidates.sort(key=lambda item: item[0])
            gap_dim = margin_candidates[0][1]
            gap_dim["used"] = True
            gap = gap_dim["val"]

        # Step 2: Claim Adjacent PC Width (centered above or below Adjacent PC box)
        adj_center_x = ax + aw / 2.0
        adj_top_val = claim_closest_dimension(dims, adj_center_x, ay - 25, x_weight=1.8, y_weight=1.0, max_dist=320)
        if adj_top_val is None:
            adj_top_val = claim_closest_dimension(dims, adj_center_x, ay + ah + 25, x_weight=1.8, y_weight=1.0, max_dist=320)
        if adj_top_val is not None:
            adj_w = adj_top_val

        # Step 3: Claim Adjacent PC Depth (along left or right side of Adjacent PC)
        adj_mid_y = ay + ah / 2.0
        d_left = claim_closest_dimension(dims, ax - 30, adj_mid_y, x_weight=1.0, y_weight=1.8, max_dist=280)
        if d_left is not None:
            adj_d = d_left
        else:
            d_right = claim_closest_dimension(dims, ax + aw + 30, adj_mid_y, x_weight=1.0, y_weight=1.8, max_dist=280)
            if d_right is not None:
                adj_d = d_right

        # Enforce visual orientation for Adjacent PC: if box is clearly wider than tall, width >= depth
        if aw > ah * 1.35 and adj_w < adj_d:
            adj_w, adj_d = adj_d, adj_w

        # Step 4: Claim Remote PC Width (centered above or below Remote PC)
        rem_center_x = rx + rw / 2.0
        r_top = claim_closest_dimension(dims, rem_center_x, ry - 25, x_weight=1.8, y_weight=1.0, max_dist=300)
        if r_top is None:
            r_top = claim_closest_dimension(dims, rem_center_x, ry + rh + 25, x_weight=1.8, y_weight=1.0, max_dist=300)
        if r_top is not None:
            rem_w = r_top

        # Step 5: Claim Remote PC Depth (along right or left side of Remote PC)
        rem_mid_y = ry + rh / 2.0
        r_right = claim_closest_dimension(dims, rx + rw + 30, rem_mid_y, x_weight=1.0, y_weight=1.8, max_dist=280)
        if r_right is not None:
            rem_d = r_right
        else:
            r_left = claim_closest_dimension(dims, rx - 30, rem_mid_y, x_weight=1.0, y_weight=1.8, max_dist=280)
            if r_left is not None:
                rem_d = r_left

        # Enforce visual orientation for Remote PC if clearly rectangular
        if rw > rh * 1.35 and rem_w < rem_d:
            rem_w, rem_d = rem_d, rem_w
        elif rh > rw * 1.35 and rem_d < rem_w:
            rem_w, rem_d = rem_d, rem_w

    else:
        # Single Adjacent PC zone only
        adj_center_x = ax + aw / 2.0
        w_val = claim_closest_dimension(dims, adj_center_x, ay - 25, x_weight=1.8, y_weight=1.0, max_dist=350)
        if w_val is None:
            w_val = claim_closest_dimension(dims, adj_center_x, ay + ah + 25, x_weight=1.8, y_weight=1.0, max_dist=350)
        if w_val is not None:
            adj_w = w_val

        adj_mid_y = ay + ah / 2.0
        d_val = claim_closest_dimension(dims, ax - 30, adj_mid_y, x_weight=1.0, y_weight=1.8, max_dist=300)
        if d_val is None:
            d_val = claim_closest_dimension(dims, ax + aw + 30, adj_mid_y, x_weight=1.0, y_weight=1.8, max_dist=300)
        if d_val is not None:
            adj_d = d_val

        if aw > ah * 1.35 and adj_w < adj_d:
            adj_w, adj_d = adj_d, adj_w

    return adj_w, adj_d, has_remote, rem_w, rem_d, gap


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
        boxes = detect_cafe_zones(img_cv)
        dimensions = extract_all_dimensions(pil_img)
        adj_w, adj_d, has_remote, rem_w, rem_d, gap = assign_zone_dimensions(
            boxes, dimensions, pil_img.width, pil_img.height
        )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Uploaded Plan & Detection Overlay")
        debug_img = pil_img.copy()
        debug_draw = ImageDraw.Draw(debug_img)
        for idx, (bx, by, bw, bh) in enumerate(boxes):
            zone_name = "Remote PC" if (len(boxes) >= 2 and idx == 0) else "Adjacent PC"
            debug_draw.rectangle([bx, by, bx + bw, by + bh], outline="#16a34a", width=4)
            debug_draw.text((bx + 6, by + 6), zone_name, fill="#16a34a")
        for d in dimensions:
            dx, dy = d["x"], d["y"]
            debug_draw.ellipse([dx - 5, dy - 5, dx + 5, dy + 5], fill="#dc2626")
            debug_draw.text((dx + 8, dy - 8), f"{d['val']}m", fill="#dc2626")
        st.image(debug_img, use_container_width=True)
        st.caption("Green boxes = detected PC zones | Red dots = numbers read by OCR")

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

        # Solid enclosed Adjacent PC polygon
        draw.rectangle([ax0, ay0, ax1, ay1], outline="black", fill="#dbeafe", width=3)
        adj_area = round(adj_w * adj_d, 2)
        adj_label = f"Adjacent PC ({adj_area} sq.m)"
        if (ax1 - ax0) >= 160:
            draw.text((ax0 + 10, (ay0 + ay1) // 2 - 8), adj_label, fill="black")
        else:
            draw.text((ax0, ay0 - 20), adj_label, fill="black")

        draw_dimension_arrow(draw, (ax0, ay1), (ax1, ay1), f"{adj_w}m", offset=35)
        draw_dimension_arrow(draw, (ax1, ay0), (ax1, ay1), f"{adj_d}m", offset=25)

        if has_remote:
            ry1 = ay0 - int(gap * scale)
            ry0 = ry1 - int(rem_d * scale)
            rx1 = ax0 + int(rem_w * scale)

            # Solid enclosed Remote PC polygon
            draw.rectangle([ax0, ry0, rx1, ry1], outline="black", fill="#fef3c7", width=3)
            rem_area = round(rem_w * rem_d, 2)
            rem_label = f"Remote PC ({rem_area} sq.m)"
            if (rx1 - ax0) >= 160:
                draw.text((ax0 + 10, (ry0 + ry1) // 2 - 8), rem_label, fill="black")
            else:
                draw.text((ax0, ry1 + 8), rem_label, fill="black")

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
