import io
import re
import fitz  # PyMuPDF for PDF uploads
import numpy as np
from PIL import Image, ImageDraw
from rapidocr_onnxruntime import RapidOCR
import streamlit as st

st.set_page_config(page_title="Pavement Cafe Plan Generator", layout="wide")
st.title("Pavement Cafe Dimensioned Polygon Generator")


@st.cache_resource
def load_ocr():
  return RapidOCR()


ocr = load_ocr()


def draw_dimension_arrow(draw, p1, p2, text, offset=25, color="black"):
  """Draws a clean architectural dimension line with extension ticks and label."""
  x1, y1 = p1
  x2, y2 = p2
  if y1 == y2:  # Horizontal dimension
    yd = y1 + offset
    draw.line([(x1, y1), (x1, yd)], fill="#64748b", width=1)
    draw.line([(x2, y2), (x2, yd)], fill="#64748b", width=1)
    draw.line([(x1, yd), (x2, yd)], fill=color, width=2)
    draw.line([(x1 - 4, yd + 4), (x1 + 4, yd - 4)], fill=color, width=2)
    draw.line([(x2 - 4, yd + 4), (x2 + 4, yd - 4)], fill=color, width=2)
    draw.text(((x1 + x2) // 2 - 18, yd + 5), text, fill=color)
  else:  # Vertical dimension
    xd = x1 + offset
    draw.line([(x1, y1), (xd, y1)], fill="#64748b", width=1)
    draw.line([(x2, y2), (xd, y2)], fill="#64748b", width=1)
    draw.line([(xd, y1), (xd, y2)], fill=color, width=2)
    draw.line([(xd - 4, y1 + 4), (xd + 4, y1 - 4)], fill=color, width=2)
    draw.line([(xd - 4, y2 + 4), (xd + 4, y2 - 4)], fill=color, width=2)
    draw.text((xd + 8, (y1 + y2) // 2 - 6), text, fill=color)


def find_boxes_no_cv2(pil_img):
  """Finds main rectangular boundary boxes using pure NumPy without OpenCV."""
  gray = np.array(pil_img.convert("L"))
  dark = gray < 180
  h, w = dark.shape
  visited = np.zeros_like(dark, dtype=bool)
  boxes = []

  # Downsample by 4x for fast pure-Python connected-component scan
  step = 4
  small = dark[::step, ::step]
  sh, sw = small.shape
  vis = np.zeros_like(small, dtype=bool)

  for y in range(sh):
    for x in range(sw):
      if small[y, x] and not vis[y, x]:
        stack = [(y, x)]
        vis[y, x] = True
        min_x = max_x = x
        min_y = max_y = y
        while stack:
          cy, cx = stack.pop()
          if cx < min_x:
            min_x = cx
          if cx > max_x:
            max_x = cx
          if cy < min_y:
            min_y = cy
          if cy > max_y:
            max_y = cy
          for ny, nx in (
              (cy - 1, cx),
              (cy + 1, cx),
              (cy, cx - 1),
              (cy, cx + 1),
          ):
            if 0 <= ny < sh and 0 <= nx < sw and small[ny, nx] and not vis[ny, nx]:
              vis[ny, nx] = True
              stack.append((ny, nx))
        bw = (max_x - min_x) * step
        bh = (max_y - min_y) * step
        if bw > 80 and bh > 40 and bw < w * 0.95:
          boxes.append((min_x * step, min_y * step, bw, bh))

  boxes.sort(key=lambda b: b[1])  # Sort top-to-bottom
  return boxes


uploaded_file = st.file_uploader(
    "Upload Site Plan (JPG, PNG, WEBP, or PDF)",
    type=["jpg", "jpeg", "png", "webp", "pdf"],
)

if uploaded_file:
  file_bytes = uploaded_file.read()

  # Convert PDF page 1 or standard image into Pillow RGB format
  if uploaded_file.name.lower().endswith(".pdf"):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pix = doc.load_page(0).get_pixmap(dpi=150)
    pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
  else:
    pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")

  # Resize large images so free hosting never runs out of memory
  w, h = pil_img.size
  if w > 1200:
    pil_img = pil_img.resize((1200, int(h * (1200 / w))))

  col1, col2 = st.columns(2)
  with col1:
    st.subheader("Uploaded Plan")
    st.image(pil_img, use_container_width=True)

  # --- AUTOMATIC EXTRACTION (RapidOCR + Pure NumPy) ---
  with st.spinner("Reading dimensions and detecting boundary polygons..."):
    img_np = np.array(pil_img)
    ocr_results, _ = ocr(img_np)
    dimensions = []
    if ocr_results:
      for item in ocr_results:
        bbox, text, _ = item
        match = re.search(r"(\d+\.?\d*)", str(text))
        if match:
          val = float(match.group(1))
          if val > 100:  # Convert mm (e.g. 4500) to meters (4.5)
            val /= 1000.0
          if 0.2 <= val <= 50.0:
            cx = int((bbox[0][0] + bbox[2][0]) / 2)
            cy = int((bbox[0][1] + bbox[2][1]) / 2)
            dimensions.append({"val": val, "x": cx, "y": cy})

    boxes = find_boxes_no_cv2(pil_img)

  # Default fallback values
  adj_w, adj_d, has_remote, rem_w, rem_d, gap = 4.0, 1.5, False, 4.0, 1.5, 2.0

  if boxes and dimensions:
    bx, by, bw, bh = boxes[-1]  # Bottom box assumed to be Adjacent PC
    adj_w = min(
        dimensions,
        key=lambda d: (d["x"] - (bx + bw / 2)) ** 2 + (d["y"] - by) ** 2,
    )["val"]
    adj_d = min(
        dimensions,
        key=lambda d: (d["x"] - bx) ** 2 + (d["y"] - (by + bh / 2)) ** 2,
    )["val"]

    if len(boxes) > 1:
      has_remote = True
      rx, ry, rw, rh = boxes[0]  # Top box assumed to be Remote PC
      rem_w = min(
          dimensions,
          key=lambda d: (d["x"] - (rx + rw / 2)) ** 2 + (d["y"] - ry) ** 2,
      )["val"]
      rem_d = min(
          dimensions,
          key=lambda d: (d["x"] - rx) ** 2 + (d["y"] - (ry + rh / 2)) ** 2,
      )["val"]
      gap_y = (ry + rh + by) / 2
      gap = min(
          dimensions,
          key=lambda d: (d["x"] - (rx + rw / 2)) ** 2 + (d["y"] - gap_y) ** 2,
      )["val"]

  # --- AUTO-FILLED CONTROLS & RENDERER ---
  with col2:
    st.subheader("Dimensioned Polygon Output")
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

    # Auto-calculate scale so the drawing fits neatly on an 800x800 canvas
    total_w_m = max(adj_w, rem_w) if has_remote else adj_w
    total_h_m = (adj_d + gap + rem_d) if has_remote else adj_d
    scale = min(520 / total_w_m, 520 / total_h_m)

    out = Image.new("RGB", (800, 800), "white")
    draw = ImageDraw.Draw(out)

    # Premises / Building Frontage reference line along the bottom
    ax0 = 120
    ay1 = 680
    ay0 = ay1 - int(adj_d * scale)
    ax1 = ax0 + int(adj_w * scale)

    draw.line([(60, ay1), (740, ay1)], fill="black", width=4)
    draw.text((65, ay1 + 12), "PREMISES / BUILDING FRONTAGE", fill="black")

    # Draw Adjacent PC Polygon
    draw.rectangle([ax0, ay0, ax1, ay1], outline="black", fill="#dbeafe", width=3)
    adj_area = round(adj_w * adj_d, 2)
    draw.text(
        (ax0 + 10, (ay0 + ay1) // 2 - 8),
        f"Adjacent PC ({adj_area} m²)",
        fill="black",
    )

    # External dimension lines for Adjacent PC
    draw_dimension_arrow(draw, (ax0, ay1), (ax1, ay1), f"{adj_w}m", offset=35)
    draw_dimension_arrow(draw, (ax1, ay0), (ax1, ay1), f"{adj_d}m", offset=25)

    # Draw Remote PC Polygon & Gap
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

      # External dimension lines for Remote PC
      draw_dimension_arrow(draw, (ax0, ry0), (rx1, ry0), f"{rem_w}m", offset=-25)
      draw_dimension_arrow(draw, (rx1, ry0), (rx1, ry1), f"{rem_d}m", offset=25)

      # Separation gap arrow between Adjacent and Remote zones
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

    # Export to JPEG
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=95)
    st.download_button(
        label="Download Dimensioned JPEG",
        data=buf.getvalue(),
        file_name="pavement_cafe_plan.jpg",
        mime="image/jpeg",
    )
