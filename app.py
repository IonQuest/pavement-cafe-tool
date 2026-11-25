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
    h, w = img_cv.shape[:2]
    page_area = h * w
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
    )

    k_light = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed_light = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k_light)

    bridge_w = max(25, int(w * 0.14))
    k_horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (bridge_w, 3))
    closed_horiz = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k_horiz)

    raw_boxes = []
    for mask in (closed_light, closed_horiz):
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < (page_area * 0.002) or area > (page_area * 0.45):
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            if bw < int(w * 0.08) or bh < int(h * 0.06):
                continue
            if bw > int(w * 0.85) or by < 15 or (by + bh) > (h - 15):
                continue
            rect_fill = area / float(bw * bh) if (bw * bh) > 0 else 0
            if rect_fill > 0.62:
                has_text_inside = any(
                    (bx + 12) < d["x"] < (bx + bw - 12) and (by + 12) < d["y"] < (by + bh - 12)
                    for d in dimensions
                )
                if not has_text_inside:
                    raw_boxes.append((bx, by, bw, bh))

    raw_boxes.sort(key=box_area, reverse=True)
    unique_boxes = []
    for bx, by, bw, bh in raw_boxes:
        is_dup = False
        for ux, uy, uw, uh in unique_boxes:
            if (
                abs(bx - ux) < 35
                and abs(by - uy) < 35
                and abs(bw - uw) < 50
                and abs(bh - uh) < 40
            ):
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


def extract_dimensions(pil_img):
    enhanced = ImageEnhance.Contrast(pil_img).enhance(1.9)
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.6)
    img_np = np.array(enhanced)

    dimensions = []
    ocr_results, _ = ocr(img_np)
    if ocr_results:
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
            for match in re.finditer(r"(\d+\.?\d*)", clean_text):
                val = float(match.group(1))
                if val > 100:
                    val /= 1000.0
                if val == 1.0 and not has_unit:
                    continue
                if 0.2 <= val <= 50.0:
                    center_x, center_y = np.array(bbox, dtype=float).mean(axis=0)
                    dimensions.append(
                        {
                            "val": round(val, 2),
                            "x": int(center_x),
                            "y": int(center_y),
                            "has_unit": has_unit,
                        }
                    )
    return dimensions


def score_dim(d, tx, ty):
    dist = ((d["x"] - tx) ** 2 + (d["y"] - ty) ** 2) ** 0.5
    return dist - (50 if d.get("has_unit") else 0)


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

    adj_w, adj_d, has_remote, rem_w, rem_d, gap = 4.0, 1.5, False, 4.0, 1.5, 2.0
    used_indices = set()

    def claim_nearest(candidates, tx, ty):
        if not candidates:
            return None
        best_idx, best_dim = min(
            candidates, key=lambda pair: score_dim(pair[1], tx, ty)
        )
        used_indices.add(best_idx)
        return best_dim["val"]

    if boxes and dimensions:
        indexed_dims = list(enumerate(dimensions))
        bx, by, bw, bh = boxes[-1]

        if len(boxes) >= 2:
            has_remote = True
            rx, ry, rw, rh = next(iter(boxes))

            # 1. Clear Gap along outer left/right margin between Remote and Adjacent boxes
            gap_y = (ry + rh + by) / 2
            gap_candidates = [
                (i, d)
                for i, d in indexed_dims
                if i not in used_indices
                and (ry + rh - 15) <= d["y"] <= (by + 15)
                and (d["x"] <= min(bx, rx) + 25 or d["x"] >= max(bx + bw, rx + rw) - 25)
            ]
            val = claim_nearest(gap_candidates, min(bx, rx) - 40, gap_y)
            if val is not None:
                gap = val

            # 2. Remote PC Width (centered above or below Remote box)
            rw_candidates = [
                (i, d)
                for i, d in indexed_dims
                if i not in used_indices
                and (rx - 40) <= d["x"] <= (rx + rw + 40)
                and ((ry - 140) <= d["y"] <= (ry + 25) or (ry + rh - 20) <= d["y"] <= (ry + rh + 75))
            ]
            val = claim_nearest(rw_candidates, rx + rw / 2, ry - 45)
            if val is not None:
                rem_w = val

            # 3. Remote PC Depth (left or right side of Remote box)
            rd_candidates = [
                (i, d)
                for i, d in indexed_dims
                if i not in used_indices
                and (ry - 25) <= d["y"] <= (ry + rh + 25)
                and (d["x"] <= rx + 25 or d["x"] >= rx + rw - 25)
            ]
            val = claim_nearest(rd_candidates, rx + rw + 45, ry + rh / 2)
            if val is not None:
                rem_d = val

        # 4. Adjacent PC Width (centered above in walkway or below frontage)
        aw_candidates = [
            (i, d)
            for i, d in indexed_dims
            if i not in used_indices
            and (bx + 25) <= d["x"] <= (bx + bw - 25)
            and ((by - 110) <= d["y"] <= (by + 25) or (by + bh - 20) <= d["y"] <= (by + bh + 90))
        ]
        val = claim_nearest(aw_candidates, bx + bw / 2, by - 40)
        if val is not None:
            adj_w = val

        # 5. Adjacent PC Depth (left or right edge of Adjacent box)
        ad_candidates = [
            (i, d)
            for i, d in indexed_dims
            if i not in used_indices
            and (by - 25) <= d["y"] <= (by + bh + 25)
            and (d["x"] <= bx + 30 or d["x"] >= bx + bw - 30)
        ]
        val = claim_nearest(ad_candidates, bx - 45, by + bh / 2)
        if val is not None:
            adj_d = val

        # Enforce visual aspect ratio orientation for Adjacent PC
        if bw > bh * 1.25 and adj_w < adj_d:
            adj_w, adj_d = adj_d, adj_w

    with col2:
        st.subheader("1. Verify or Edit Dimensions")
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
                "Clear Gap (m)", value=float(gap), min_value=0.1, step=0.1
            )

        st.subheader("2. Generated JPEG Output")
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

            draw.rectangle([ax0, ry0, rx1, ry1], outline="black", fill="#fef3c7", width=3)
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
