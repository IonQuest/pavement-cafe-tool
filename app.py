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


def normalize_lighting(img_cv):
  """Removes warm phone-photo lighting gradients and shadows."""
  gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
  bg = cv2.GaussianBlur(gray, (51, 51), 0)
  norm = cv2.divide(gray, bg, scale=255)
  return norm


def find_boxes_in_binary(binary_img, page_area, max_w, max_h):
  contours, _ = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
  found = []
  for c in contours:
    area = cv2.contourArea(c)
    if area < (page_area * 0.0015) or area > (page_area * 0.45):
      continue
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.03 * peri, True)
    bx, by, bw, bh = cv2.boundingRect(approx)
    if bw < 35 or bh < 25 or bw > max_w * 0.90 or bh > max_h * 0.85:
      continue
    rect_fill = area / float(bw * bh)
    if 4 <= len(approx) <= 8 and rect_fill > 0.65:
      found.append((bx, by, bw, bh))
  return found


def detect_cafe_zones(img_cv):
  """Multi-pass detector: finds intact boxes AND bridges customer access gaps."""
  h, w = img_cv.shape[:2]
  page_area = h * w
  norm = normalize_lighting(img_cv)
  blurred = cv2.GaussianBlur(norm, (5, 5), 0)
  thresh = cv2.adaptiveThreshold(
      blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 12
  )

  # Pass 1: Light close (catches intact Remote PC squares cleanly)
  k_light = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
  closed_light = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k_light)
  boxes_pass1 = find_boxes_in_binary(closed_light, page_area, w, h)

  # Pass 2: Wide horizontal close (bridges customer entrance gaps in Adjacent PC)
  k_horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (65, 5))
  closed_horiz = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k_horiz)
  k_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 35))
  closed_wide = cv2.morphologyEx(closed_horiz, cv2.MORPH_CLOSE, k_vert)
  boxes_pass2 = find_boxes_in_binary(closed_wide, page_area, w, h)

  all_candidates = boxes_pass1 + boxes_pass2
  all_candidates.sort(key=lambda b: b[2] * b[3], reverse=True)

  # Deduplicate overlapping detections (keep the larger/bridged version of a zone)
  merged = []
  for bx, by, bw, bh in all_candidates:
    is_overlap = False
    for mx, my, mw, mh in merged:
      ix1 = max(bx, mx)
      iy1 = max(by, my)
      ix2 = min(bx + bw, mx + mw)
      iy2 = min(by + bh, my + mh)
      if ix2 > ix1 and iy2 > iy1:
        inter = (ix2 - ix1) * (iy2 - iy1)
        if inter > 0.45 * min(bw * bh, mw * mh):
          is_overlap = True
          break
    if not is_overlap:
      merged.append((bx, by, bw, bh))

  # Remove any giant outer container box that wraps around other detected zones
  final_zones = []
  for i, (bx, by, bw, bh) in enumerate(merged):
    encloses_another = False
    for j, (ox, oy, ow, oh) in enumerate(merged):
      if i != j and bx <= ox + 10 and by <= oy + 10 and (bx + bw) >= (ox + ow - 10) and (by + bh) >= (oy + oh - 10):
        encloses_another = True
        break
    if not encloses_another:
      final_zones.append((bx, by, bw, bh))

  # Sort top-to-bottom by Y coordinate
  final_zones.sort(key=lambda b: b[1])
  return final_zones


def extract_dimensions(pil_img):
  """Extracts numeric dimension labels with positions and text bounding boxes."""
  img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
  norm = normalize_lighting(img_cv)
  norm_rgb = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)
  enhanced = ImageEnhance.Contrast(Image.fromarray(norm_rgb)).enhance(1.6)

  dimensions = []
  ocr_results, _ = ocr(np.array(enhanced))
  if ocr_results:
    for item in ocr_results:
      bbox, text, _ = item
      clean_text = (
          str(text).replace(",", ".").replace("O", "0").replace("o", "0")
      )
      for match in re.finditer(r"(\d+\.?\d*)", clean_text):
        val = float(match.group(1))
        if val > 100:
          val /= 1000.0
        if 0.2 <= val <= 50.0:
          pts = np.array(bbox, dtype=float)
          center_x, center_y = pts.mean(axis=0)
          dimensions.append({
              "val": round(val, 2),
              "x": int(center_x),
              "y": int(center_y),
              "used": False,
          })
  return dimensions


def claim_closest(dimensions, filter_fn, score_fn, fallback):
  candidates = [d for d in dimensions if not d["used"] and filter_fn(d)]
  if not candidates:
    return fallback
  best = min(candidates, key=score_fn)
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
    boxes = detect_cafe_zones(img_cv)
    dimensions = extract_dimensions(pil_img)

  col1, col2 = st.columns(2)
  with col1:
    st.subheader("Uploaded Plan & Detected Features")
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
    bx, by, bw, bh = boxes[-1]  # Bottom box = Adjacent PC
    adj_mid_x = bx + bw / 2.0
    adj_mid_y = by + bh / 2.0

    if len(boxes) >= 2:
      has_remote = True
      rx, ry, rw, rh = boxes[0]  # Top box = Remote PC
      rem_mid_x = rx + rw / 2.0
      rem_mid_y = ry + rh / 2.0
      gap_top = ry + rh
      gap_bot = by
      gap_mid_y = (gap_top + gap_bot) / 2.0

      # 1. Claim Adjacent Width FIRST (centered horizontally over the wide bottom box, inside the walkway strip)
      adj_w = claim_closest(
          dimensions,
          lambda d: abs(d["x"] - adj_mid_x) < max(bw * 0.45, 70) and (gap_top + 10) < d["y"] < (by + bh + 90),
          lambda d: abs(d["x"] - adj_mid_x) * 1.5 + min(abs(d["y"] - by), abs(d["y"] - (by + bh))),
          fallback=round(bw / 100.0, 1),
      )

      # 2. Claim Remote Width (centered horizontally over the top box, above its bottom edge)
      rem_w = claim_closest(
          dimensions,
          lambda d: abs(d["x"] - rem_mid_x) < max(rw * 0.65, 70) and d["y"] < (ry + rh * 0.6),
          lambda d: abs(d["x"] - rem_mid_x) * 1.5 + abs(d["y"] - ry),
          fallback=round(rw / 100.0, 1),
      )

      # 3. Claim Clear Gap (vertically between the two boxes, offset to the left/right side rather than centered)
      gap = claim_closest(
          dimensions,
          lambda d: (ry + rh * 0.3) <= d["y"] <= (by + bh * 0.3),
          lambda d: abs(d["y"] - gap_mid_y) * 2.0 + abs(d["x"] - min(bx, rx)),
          fallback=2.0,
      )

      # 4. Claim Remote Depth (vertically aligned with Remote box midpoint, to the left or right)
      rem_d = claim_closest(
          dimensions,
          lambda d: abs(d["y"] - rem_mid_y) < max(rh * 0.65, 60) and (d["x"] < rx + 10 or d["x"] > rx + rw - 10),
          lambda d: abs(d["y"] - rem_mid_y) * 2.0 + min(abs(d["x"] - rx), abs(d["x"] - (rx + rw))),
          fallback=round(rh / 100.0, 1),
      )

      # 5. Claim Adjacent Depth (vertically aligned with Adjacent box midpoint, to the left or right)
      adj_d = claim_closest(
          dimensions,
          lambda d: abs(d["y"] - adj_mid_y) < max(bh * 0.65, 60) and (d["x"] < bx + 10 or d["x"] > bx + bw - 10),
          lambda d: abs(d["y"] - adj_mid_y) * 2.0 + min(abs(d["x"] - bx), abs(d["x"] - (bx + bw))),
          fallback=round(bh / 100.0, 1),
      )
    else:
      # Single Adjacent PC layout
      adj_w = claim_closest(
          dimensions,
          lambda d: abs(d["x"] - adj_mid_x) < max(bw * 0.55, 80),
          lambda d: abs(d["x"] - adj_mid_x) * 1.5 + min(abs(d["y"] - by), abs(d["y"] - (by + bh))),
          fallback=round(bw / 100.0, 1),
      )
      adj_d = claim_closest(
          dimensions,
          lambda d: abs(d["y"] - adj_mid_y) < max(bh * 0.65, 60),
          lambda d: abs(d["y"] - adj_mid_y) * 2.0 + min(abs(d["x"] - bx), abs(d["x"] - (bx + bw))),
          fallback=round(bh / 100.0, 1),
      )

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
        f"Adjacent PC ({adj_area} m²)",
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
          f"Remote PC ({rem_area} m²)",
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
          (60, 30), f"Total Pavement Cafe Area: {total_area} m²", fill="black"
      )
    else:
      draw.text(
          (60, 30), f"Total Pavement Cafe Area: {adj_area} m²", fill="black"
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
