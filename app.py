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

def extract_dimensions(pil_img):
    enhanced = ImageEnhance.Contrast(pil_img).enhance(1.9)
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.6)
    img_np = np.array(enhanced)

    gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    bg = cv2.GaussianBlur(gray, (51, 51), 0)
    norm = cv2.divide(gray, bg, scale=245)
    norm_rgb = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)

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
            bw_ocr = float(pts[:, 0].max() - pts[:, 0].min())
            bh_ocr = float(pts[:, 1].max() - pts[:, 1].min())
            if bw_ocr > 0 and (bh_ocr / bw_ocr) > 2.4 and not has_unit:
                continue

            for match in re.finditer(r"(\d+\.?\d*)", clean_text):
                raw_num = match.group(1)
                val = float(raw_num)
                if val > 100:
                    val /= 1000.0
                if not has_unit and abs(val - 1.0) < 0.05:
                    continue
                if "." not in raw_num and val < 100:
                    val = round(val)
                if 0.3 <= val <= 50.0:
                    cx = int(pts[:, 0].mean())
                    cy = int(pts[:, 1].mean())
                    is_dup = any(
                        abs(d["x"] - cx) < 25 and abs(d["y"] - cy) < 25
                        for d in dimensions
                    )
                    if not is_dup:
                        dimensions.append({
                            "val": round(float(val), 2),
                            "x": cx,
                            "y": cy,
                            "has_unit": has_unit,
                        })
    return dimensions

def detect_cafe_rectangles(img_cv, dimensions):
    h, w = img_cv.shape[:2]
    page_area = h * w
    min_w = int(w * 0.12)
    min_h = int(h * 0.07)

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    bg = cv2.GaussianBlur(gray, (51, 51), 0)
    norm = cv2.divide(gray, bg, scale=255)
    blurred = cv2.GaussianBlur(norm, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
    )

    k_light = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    k_horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (max(45, int(w * 0.14)), 3))

    candidates = []
    for k in (k_light, k_horiz):
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k)
        contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < (page_area * 0.004) or area > (page_area * 0.45):
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            if bw < min_w or bh < min_h:
                continue
            has_text_inside = any(
                (bx + 15) <= d["x"] <= (bx + bw - 15)
                and (by + 15) <= d["y"] <= (by + bh - 15)
                for d in dimensions
            )
            if has_text_inside:
                continue
            candidates.append((bx, by, bw, bh))

    candidates.sort(key=lambda b: b[2] * b[3], reverse=True)
    unique = []
    for bx, by, bw, bh in candidates:
        if any(
            abs(bx - ux) < 35 and abs(by - uy) < 35 and abs(bw - uw) < 45 and abs(bh - uh) < 45
            for ux, uy, uw, uh in unique
        ):
            continue
        unique.append((bx, by, bw, bh))

    inner = []
    for i, (bx, by, bw, bh) in enumerate(unique):
        wraps_other = any(
            i != j
            and bx <= ox + 10
            and by <= oy + 10
            and (bx + bw) >= (ox + ow - 10)
            and (by + bh) >= (oy + oh - 10)
            for j, (ox, oy, ow, oh) in enumerate(unique)
        )
        if not wraps_other:
            inner.append((bx, by, bw, bh))

    inner.sort(key=lambda b: b[1])
    return inner

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
    used = set()

    def pick_best(cands, target_x, target_y):
        avail = [(i, d) for i, d in enumerate(dimensions) if i not in used and cands(d)]
        if not avail:
            return None
        best_i, best_d = min(
            avail,
            key=lambda item: ((item[1]["x"] - target_x) ** 2 + (item[1]["y"] - target_y) ** 2)
            - (4000 if item[1]["has_unit"] else 0),
        )
        used.add(best_i)
        return best_d["val"]

    if boxes and dimensions:
        if len(boxes) >= 2:
            has_remote = True
            rx, ry, rw, rh = boxes[0]
            bx, by, bw, bh = boxes[-1]

            # 1. Clear Gap along left or right outer margin between Remote and Adjacent boxes
            gap_mid_y = (ry + rh + by) / 2.0
            left_bound = min(rx, bx) + 45
            right_bound = max(rx + rw, bx + bw) - 45
            g_val = pick_best(
                lambda d: (ry + rh - 25) <= d["y"] <= (by + 25)
                and (d["x"] <= left_bound or d["x"] >= right_bound),
                min(rx, bx),
                gap_mid_y,
            )
            if g_val is not None:
                gap = g_val

            # 2. Adjacent PC Width (centered above or below bottom box)
            aw_val = pick_best(
                lambda d: (bx + 20) <= d["x"] <= (bx + bw - 20)
                and (ry + rh - 10) <= d["y"] <= (by + bh + 120),
                bx + bw / 2.0,
                by,
            )
            if aw_val is not None:
                adj_w = aw_val

            # 3. Adjacent PC Depth (beside left or right edge of bottom box)
            ad_val = pick_best(
                lambda d: (by - 25) <= d["y"] <= (by + bh + 25)
                and (d["x"] <= bx + 40 or d["x"] >= bx + bw - 40),
                bx,
                by + bh / 2.0,
            )
            if ad_val is not None:
                adj_d = ad_val

            # 4. Remote PC Width (generous vertical window above/below top box)
            rw_val = pick_best(
                lambda d: (rx - 40) <= d["x"] <= (rx + rw + 40)
                and (ry - 160) <= d["y"] <= (ry + rh + 60),
                rx + rw / 2.0,
                ry,
            )
            if rw_val is not None:
                rem_w = rw_val

            # 5. Remote PC Depth (beside left or right edge of top box)
            rd_val = pick_best(
                lambda d: (ry - 35) <= d["y"] <= (ry + rh + 35)
                and (d["x"] <= rx + 40 or d["x"] >= rx + rw - 40),
                rx + rw,
                ry + rh / 2.0,
            )
            if rd_val is not None:
                rem_d = rd_val

            # Orientation guard for Adjacent PC if clearly wider than tall on sketch
            if bw > bh * 1.35 and adj_w < adj_d:
                adj_w, adj_d = adj_d, adj_w
        else:
            bx, by, bw, bh = boxes[0]
            aw_val = pick_best(
                lambda d: (bx - 30) <= d["x"] <= (bx + bw + 30)
                and (by - 140) <= d["y"] <= (by + bh + 140),
                bx + bw / 2.0,
                by,
            )
            if aw_val is not None:
                adj_w = aw_val
            ad_val = pick_best(
                lambda d: (by - 30) <= d["y"] <= (by + bh + 30),
                bx,
                by + bh / 2.0,
            )
            if ad_val is not None:
                adj_d = ad_val
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
