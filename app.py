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
    """Draws a crisp CAD-style architectural dimension line with extension ticks."""
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


def detect_all_cafe_zones(img_cv):
    """
    Combines two passes so BOTH closed small boxes (Remote PC) and open-entrance
    wide boxes (Adjacent PC) are reliably detected without dropping either one.
    """
    h, w = img_cv.shape[:2]
    page_area = h * w

    # 1. Normalize warm/uneven phone lighting
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    bg = cv2.GaussianBlur(gray, (55, 55), 0)
    norm = cv2.divide(gray, bg, scale=255)
    blurred = cv2.GaussianBlur(norm, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
    )

    candidate_boxes = []

    # Pass A: Light closing to catch intact/closed rectangles (such as Remote PC)
    # Pass B: Wide horizontal/vertical closing to bridge customer entrance openings (such as Adjacent PC)
    kernels = [
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (55, 5)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 55)),
    ]

    for kernel in kernels:
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            # Low minimum (0.12% page area) so small 2m x 2m Remote PCs are never dropped
            if area < (page_area * 0.0012) or area > (page_area * 0.55):
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            if bw < 35 or bh < 25 or bw > (w * 0.90) or bh > (h * 0.85):
                continue

            # Ignore skinny dimension arrows/text lines (aspect ratio check)
            aspect = max(bw / float(bh), bh / float(bw))
            if aspect > 8.5:
                continue

            # Verify rectangular solidity
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity = area / float(hull_area) if hull_area > 0 else 0
            rect_fill = area / float(bw * bh)
            if solidity > 0.72 and rect_fill > 0.62:
                candidate_boxes.append((bx, by, bw, bh))

    # Deduplicate overlapping boxes (keep largest clean bounding box per zone)
    candidate_boxes.sort(key=box_area, reverse=True)
    unique_boxes = []
    for bx, by, bw, bh in candidate_boxes:
        is_dup = False
        bcx, bcy = bx + bw / 2.0, by + bh / 2.0
        for ux, uy, uw, uh in unique_boxes:
            ucx, ucy = ux + uw / 2.0, uy + uh / 2.0
            # Same zone if centers are close or one box is an inner wall of the other
            if abs(bcx - ucx) < max(bw, uw) * 0.45 and abs(bcy - ucy) < max(bh, uh) * 0.45:
                is_dup = True
                break
        if not is_dup:
            unique_boxes.append((bx, by, bw, bh))

    # Remove any giant outer border that completely wraps around 2+ separate zones
    filtered_boxes = []
    for i, (bx, by, bw, bh) in enumerate(unique_boxes):
        wrapped = 0
        for j, (ox, oy, ow, oh) in enumerate(unique_boxes):
            if i != j and bx <= ox and by <= oy and (bx + bw) >= (ox + ow) and (by + bh) >= (oy + oh):
                wrapped += 1
        if wrapped == 0:
            filtered_boxes.append((bx, by, bw, bh))

    # Sort top-to-bottom by vertical Y position
    filtered_boxes.sort(key=box_top_y)
    return filtered_boxes


def extract_dimensions(pil_img):
    """Reads all numeric dimensions on the plan with their exact (x, y) pixel centers."""
    gray_pil = pil_img.convert("L")
    enhanced = ImageEnhance.Contrast(gray_pil).enhance(2.0)
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.6)
    img_np = np.array(enhanced.convert("RGB"))

    dimensions = []
    ocr_results, _ = ocr(img_np)
    if ocr_results:
        for item in ocr_results:
            bbox, text, _ = item
            clean_text = (
                str(text)
                .replace(",", ".")
                .replace("O", "0")
                .replace("o", "0")
                .replace("S", "5")
                .replace("s", "5")
            )
            for match in re.finditer(r"(\d+\.?\d*)", clean_text):
                val = float(match.group(1))
                if val > 100:  # Convert mm (e.g. 5000) to meters (5.0)
                    val /= 1000.0
                if 0.2 <= val <= 50.0:
                    cx, cy = np.array(bbox, dtype=float).mean(axis=0)
                    dimensions.append({"val": round(val, 2), "x": int(cx), "y": int(cy), "used": False})
    return dimensions


def claim_best_dimension(dimensions, target_x, target_y, max_dist, x_weight=1.0, y_weight=1.0, fallback=2.0):
    """
    Finds the closest unused dimension to a target point using weighted X/Y distance,
    then marks it used so numbers aren't stolen or double-counted across zones.
    """
    best_idx = None
    best_score = float("inf")
    for idx, d in enumerate(dimensions):
        if d["used"]:
            continue
        dx = abs(d["x"] - target_x) * x_weight
        dy = abs(d["y"] - target_y) * y_weight
        dist = (dx ** 2 + dy ** 2) ** 0.5
        if dist < best_score and dist <= max_dist:
            best_score = dist
            best_idx = idx

    if best_idx is not None:
        dimensions[best_idx]["used"] = True
        return dimensions[best_idx]["val"]
    return fallback


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
        boxes = detect_all_cafe_zones(img_cv)
        dimensions = extract_dimensions(pil_img)

    # Default values
    adj_w, adj_d = 5.0, 2.0
    has_remote = False
    rem_w, rem_d, gap = 2.0, 2.0, 2.0
    rem_offset_x = 0.0

    # --- ONE-TO-ONE SPATIAL DIMENSION PAIRING ---
    if len(boxes) >= 2:
        has_remote = True
        # Top box = Remote PC, Bottom box = Adjacent PC
        rx, ry, rw, rh = boxes[0]
        ax, ay, aw, ah = boxes[-1]

        # Estimate relative horizontal shift between Remote PC and Adjacent PC
        px_per_m_est = max(aw, rw) / 5.0
        if px_per_m_est > 0:
            rem_offset_x = round(max(0.0, (rx - ax) / px_per_m_est), 1)

        # 1. Claim Separation Gap first (located vertically between ry+rh and ay)
        gap_target_x = min(ax, rx) - 40
        gap_target_y = (ry + rh + ay) / 2.0
        gap = claim_best_dimension(
            dimensions, gap_target_x, gap_target_y, max_dist=350, x_weight=0.5, y_weight=1.5, fallback=2.0
        )

        # 2. Claim Remote PC Width (centered horizontally over rx + rw/2, above ry or below ry+rh)
        rem_w = claim_best_dimension(
            dimensions, rx + rw / 2.0, ry - 45, max_dist=300, x_weight=1.4, y_weight=0.8, fallback=2.0
        )

        # 3. Claim Remote PC Depth (vertically centered at ry + rh/2, beside left or right edge)
        rem_d = claim_best_dimension(
            dimensions, rx + rw + 55, ry + rh / 2.0, max_dist=300, x_weight=0.5, y_weight=1.5, fallback=2.0
        )

        # 4. Claim Adjacent PC Width (centered horizontally over ax + aw/2, above ay or below ay+ah)
        adj_w = claim_best_dimension(
            dimensions, ax + aw / 2.0, ay - 45, max_dist=350, x_weight=1.4, y_weight=0.8, fallback=5.0
        )

        # 5. Claim Adjacent PC Depth (vertically centered at ay + ah/2, beside left or right edge)
        adj_d = claim_best_dimension(
            dimensions, ax - 55, ay + ah / 2.0, max_dist=350, x_weight=0.5, y_weight=1.5, fallback=2.0
        )

    elif len(boxes) == 1:
        has_remote = False
        ax, ay, aw, ah = boxes[0]
        adj_w = claim_best_dimension(
            dimensions, ax + aw / 2.0, ay - 45, max_dist=350, x_weight=1.4, y_weight=0.8, fallback=5.0
        )
        adj_d = claim_best_dimension(
            dimensions, ax - 55, ay + ah / 2.0, max_dist=350, x_weight=0.5, y_weight=1.5, fallback=2.0
        )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Uploaded Plan & Detection Preview")
        preview = pil_img.copy()
        pdraw = ImageDraw.Draw(preview)
        for idx, (bx, by, bw, bh) in enumerate(boxes):
            zone_name = "Remote PC" if (len(boxes) >= 2 and idx == 0) else "Adjacent PC"
            pdraw.rectangle([bx, by, bx + bw, by + bh], outline="#16a34a", width=4)
            pdraw.text((bx + 6, by + 6), zone_name, fill="#16a34a")
        for d in dimensions:
            dx, dy = d["x"], d["y"]
            pdraw.ellipse([dx - 5, dy - 5, dx + 5, dy + 5], fill="#dc2626")
            pdraw.text((dx + 8, dy - 8), f"{d['val']}m", fill="#dc2626")
        st.image(preview, use_container_width=True)
        detected_list = [f"{d['val']}m" for d in dimensions]
        st.caption(
            f"Zones detected: {len(boxes)} | Numbers read: {', '.join(detected_list) if detected_list else 'None'}"
        )

    with col2:
        st.subheader("Dimensioned Polygon Output")
        c1, c2 = st.columns(2)
        adj_w = c1.number_input("Adjacent PC Width (m)", value=float(adj_w), min_value=0.1, step=0.1)
        adj_d = c2.number_input("Adjacent PC Depth (m)", value=float(adj_d), min_value=0.1, step=0.1)

        has_remote = st.checkbox("Include Remote PC", value=has_remote)
        if has_remote:
            c3, c4, c5 = st.columns(3)
            rem_w = c3.number_input("Remote PC Width (m)", value=float(rem_w), min_value=0.1, step=0.1)
            rem_d = c4.number_input("Remote PC Depth (m)", value=float(rem_d), min_value=0.1, step=0.1)
            gap = c5.number_input("Separation Gap (m)", value=float(gap), min_value=0.1, step=0.1)

        # Auto-scale so both zones and external dimension lines fit neatly on an 800x800 canvas
        total_w_m = max(adj_w, rem_w) if has_remote else adj_w
        total_h_m = (adj_d + gap + rem_d) if has_remote else adj_d
        scale = min(500 / total_w_m, 500 / total_h_m)

        out = Image.new("RGB", (800, 800), "white")
        draw = ImageDraw.Draw(out)

        # Premises / Building Frontage reference line
        ax0 = 130
        ay1 = 680
        ay0 = ay1 - int(adj_d * scale)
        ax1 = ax0 + int(adj_w * scale)

        draw.line([(60, ay1), (740, ay1)], fill="black", width=4)
        draw.text((65, ay1 + 12), "PREMISES / BUILDING FRONTAGE", fill="black")

        # Solid enclosed Adjacent PC rectangle
        draw.rectangle([ax0, ay0, ax1, ay1], outline="black", fill="#dbeafe", width=3)
        adj_area = round(adj_w * adj_d, 2)
        draw.text((ax0 + 10, (ay0 + ay1) // 2 - 8), f"Adjacent PC ({adj_area} m²)", fill="black")

        # External dimension lines for Adjacent PC
        draw_dimension_arrow(draw, (ax0, ay1), (ax1, ay1), f"{adj_w}m", offset=38)
        draw_dimension_arrow(draw, (ax1, ay0), (ax1, ay1), f"{adj_d}m", offset=25)

        if has_remote:
            ry1 = ay0 - int(gap * scale)
            ry0 = ry1 - int(rem_d * scale)
            rx0 = ax0
            rx1 = rx0 + int(rem_w * scale)

            # Solid enclosed Remote PC rectangle
            draw.rectangle([rx0, ry0, rx1, ry1], outline="black", fill="#fef3c7", width=3)
            rem_area = round(rem_w * rem_d, 2)
            draw.text((rx0 + 10, (ry0 + ry1) // 2 - 8), f"Remote PC ({rem_area} m²)", fill="black")

            # External dimension lines for Remote PC
            draw_dimension_arrow(draw, (rx0, ry0), (rx1, ry0), f"{rem_w}m", offset=-28)
            draw_dimension_arrow(draw, (rx1, ry0), (rx1, ry1), f"{rem_d}m", offset=25)

            # Separation gap arrow between Adjacent and Remote PCs
            gap_x = ax0 + int(min(adj_w, rem_w) * scale * 0.5)
            draw.line([(gap_x, ry1), (gap_x, ay0)], fill="#dc2626", width=2)
            draw.line([(gap_x - 5, ry1 + 5), (gap_x, ry1), (gap_x + 5, ry1 + 5)], fill="#dc2626", width=2)
            draw.line([(gap_x - 5, ay0 - 5), (gap_x, ay0), (gap_x + 5, ay0 - 5)], fill="#dc2626", width=2)
            draw.text((gap_x + 8, (ry1 + ay0) // 2 - 6), f"Clear Gap: {gap}m", fill="#dc2626")

            total_area = round(adj_area + rem_area, 2)
            draw.text((60, 30), f"Total Pavement Cafe Area: {total_area} m²", fill="black")
        else:
            draw.text((60, 30), f"Total Pavement Cafe Area: {adj_area} m²", fill="black")

        st.image(out, use_container_width=True)

        buf = io.BytesIO()
        out.save(buf, format="JPEG", quality=95)
        st.download_button(
            label="Download Dimensioned JPEG",
            data=buf.getvalue(),
            file_name="pavement_cafe_plan.jpg",
            mime="image/jpeg",
        )
