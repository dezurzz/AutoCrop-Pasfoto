import os
import io
import time
import zipfile
import urllib.request
from pathlib import Path
import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps

# Page Configuration
st.set_page_config(
    page_title="AutoCrop Pasfoto 3x4 (Manual Rotate) - Web App",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# MediaPipe model download config
MODELS = {
    "short": {
        "url": "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
        "file": "blaze_face_short_range.tflite"
    },
    "full": {
        "url": "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/latest/blaze_face_full_range.tflite",
        "file": "blaze_face_full_range.tflite"
    }
}

@st.cache_resource
def download_model(model_type="full"):
    model_info = MODELS.get(model_type, MODELS["full"])
    file_name = model_info["file"]
    url = model_info["url"]
    path = Path(file_name)
    
    if not path.exists():
        with st.spinner(f"Mengunduh model pendeteksi wajah ({file_name}). Harap tunggu..."):
            try:
                urllib.request.urlretrieve(url, file_name)
            except Exception as e:
                st.error(f"Gagal mengunduh model dari Google CDN: {e}")
                return None
    return file_name

# Helper functions
def load_image(uploaded_file):
    """Loads uploaded image, auto-resolving EXIF orientation so it appears upright."""
    try:
        pil_img = Image.open(uploaded_file)
        pil_img = ImageOps.exif_transpose(pil_img)
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        # Convert PIL (RGB) to OpenCV (BGR)
        open_cv_image = np.array(pil_img)
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
        return open_cv_image
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

def hex_to_bgr(hex_str):
    """Converts hex color (#RRGGBB) to BGR tuple."""
    hex_str = hex_str.lstrip('#')
    rgb = tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
    return (rgb[2], rgb[1], rgb[0]) # BGR for OpenCV

def get_largest_detection(detections):
    if not detections:
        return None
    largest_area = -1
    best_det = None
    for detection in detections:
        bbox = detection.bounding_box
        area = bbox.width * bbox.height
        if area > largest_area:
            largest_area = area
            best_det = detection
    return best_det

def rotate_image_around_center(image, angle, padding_color_bgr):
    """Rotates image around its center point by a given angle."""
    img_h, img_w, _ = image.shape
    center_x, center_y = img_w / 2, img_h / 2
    M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (img_w, img_h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=padding_color_bgr
    )
    return rotated

def crop_face_to_pasfoto(image, bbox, target_w, target_h, face_ratio, vertical_pos, padding_color_bgr, custom_cx=None, custom_cy_offset=0):
    """Crops face and resizes, handling padding and custom offsets."""
    img_h, img_w, _ = image.shape
    
    face_x = bbox.origin_x
    face_y = bbox.origin_y
    face_w = bbox.width
    face_h = bbox.height
    
    face_cx = face_x + face_w / 2
    face_cy = face_y + face_h / 2
    
    if custom_cx is not None:
        face_cx = custom_cx
        
    face_cy += custom_cy_offset
    
    crop_h = face_h / face_ratio
    crop_w = crop_h * (target_w / target_h)
    
    y1 = face_cy - vertical_pos * crop_h
    y2 = y1 + crop_h
    
    x1 = face_cx - 0.5 * crop_w
    x2 = x1 + crop_w
    
    ix1 = int(round(x1))
    iy1 = int(round(y1))
    ix2 = ix1 + int(round(crop_w))
    iy2 = iy1 + int(round(crop_h))
    
    actual_crop_w = ix2 - ix1
    actual_crop_h = iy2 - iy1
    
    canvas = np.full((actual_crop_h, actual_crop_w, 3), padding_color_bgr, dtype=np.uint8)
    
    src_x1 = max(0, ix1)
    src_y1 = max(0, iy1)
    src_x2 = min(img_w, ix2)
    src_y2 = min(img_h, iy2)
    
    dest_x1 = src_x1 - ix1
    dest_y1 = src_y1 - iy1
    dest_x2 = src_x2 - ix1
    dest_y2 = src_y2 - iy1
    
    if src_x2 > src_x1 and src_y2 > src_y1:
        canvas[dest_y1:dest_y2, dest_x1:dest_x2] = image[src_y1:src_y2, src_x1:src_x2]
        
    resized = cv2.resize(canvas, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    return resized, (ix1, iy1, ix2, iy2)

# Streamlit App UI
st.title("📸 AutoCrop Pasfoto 3x4 (Manual Rotate)")
st.markdown(
    """
    Aplikasi web pintar berbasis AI untuk mendeteksi wajah secara otomatis dan memotong (crop) foto Anda menjadi pasfoto standar resmi. **Anda bisa memutar kemiringan foto secara manual di panel kiri (rotasi sederhana pada poros tengah gambar).**
    """
)

# Sidebar settings
st.sidebar.header("⚙️ Pengaturan Global")

# Target Aspect Ratio / Dimensions
with st.sidebar.expander("📐 Ukuran & Layout Pasfoto", expanded=True):
    preset = st.selectbox(
        "Pilih Ukuran Preset",
        ["3x4 (1200 x 1600 px) - Kualitas Cetak 300 DPI",
         "4x6 (1200 x 1800 px)",
         "2x3 (800 x 1200 px)",
         "Custom"]
    )
    
    if preset.startswith("3x4"):
        target_w, target_h = 1200, 1600
    elif preset.startswith("4x6"):
        target_w, target_h = 1200, 1800
    elif preset.startswith("2x3"):
        target_w, target_h = 800, 1200
    else:
        target_w = st.number_input("Lebar Target (px)", value=1200, step=50)
        target_h = st.number_input("Tinggi Target (px)", value=1600, step=50)

# Composition Parameters
with st.sidebar.expander("🎯 Parameter Komposisi", expanded=False):
    face_ratio = st.slider(
        "Rasio Tinggi Wajah",
        min_value=0.20, max_value=0.50, value=0.30, step=0.01,
        help="Proporsi tinggi wajah terhadap total tinggi foto. Standar pasfoto adalah 30% s.d. 35%."
    )
    vertical_pos = st.slider(
        "Posisi Vertikal Wajah (Y)",
        min_value=0.20, max_value=0.50, value=0.35, step=0.01,
        help="Posisi titik tengah wajah dari batas atas foto. Standar pasfoto adalah 35%."
    )
    center_weight = st.slider(
        "Bobot Horizontal (Center Weight)",
        min_value=0.0, max_value=1.0, value=1.0, step=0.1,
        help="1.0 = pusatkan tepat di mata. 0.0 = pusatkan di tengah gambar asli (menyeimbangkan bahu)."
    )
    rotation_val = st.slider(
        "Rotasi Manual Global (derajat)",
        min_value=-30.0, max_value=30.0, value=0.0, step=0.5,
        help="Putar foto secara manual pada poros tengah gambar (berlaku untuk Single & Batch). Positif = berlawanan arah jarum jam, Negatif = searah jarum jam."
    )

# Background color picker with official presets
with st.sidebar.expander("🎨 Warna Background (Padding)", expanded=True):
    st.write("Warna latar belakang jika area pemotongan keluar batas:")
    bg_preset = st.radio(
        "Preset Warna Pasfoto Resmi:",
        ["Putih Polos (#FFFFFF)", "Merah Pasfoto - Tahun Ganjil (#DB1514)", "Biru Pasfoto - Tahun Genap (#0F3B8C)", "Custom"],
        index=0
    )
    
    if bg_preset.startswith("Putih"):
        bg_color = "#FFFFFF"
    elif bg_preset.startswith("Merah"):
        bg_color = "#DB1514"
    elif bg_preset.startswith("Biru"):
        bg_color = "#0F3B8C"
    else:
        bg_color = st.color_picker("Pilih Warna Kustom", "#FFFFFF")
        
    padding_color_bgr = hex_to_bgr(bg_color)

# Load MediaPipe Detector
detector_file = download_model("full")
if not detector_file:
    st.stop()

# Initialize MediaPipe Face Detector
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except ImportError:
    st.error("Gagal memuat library MediaPipe. Pastikan requirements sudah terinstall.")
    st.stop()

base_options = python.BaseOptions(model_asset_path=detector_file)
options = vision.FaceDetectorOptions(
    base_options=base_options,
    min_detection_confidence=0.5
)
detector = vision.FaceDetector.create_from_options(options)

# Tabs for single vs batch mode
tab_single, tab_batch = st.tabs(["🖼️ Pemrosesan Satu Foto", "📁 Pemrosesan Massal (Batch)"])

# ----------------- TAB 1: SINGLE PROCESSING -----------------
with tab_single:
    st.subheader("Unggah & Sesuaikan Satu Foto")
    uploaded_file = st.file_uploader(
        "Pilih foto portrait Anda (.jpg, .jpeg, .png, .webp)",
        type=["jpg", "jpeg", "png", "webp"],
        key="single_uploader"
    )
    
    if uploaded_file is not None:
        # Load the image
        img = load_image(uploaded_file)
        if img is not None:
            # Apply manual rotation on the original image (around center) if angle != 0.0
            processed_img = img.copy()
            if rotation_val != 0.0:
                processed_img = rotate_image_around_center(img, rotation_val, padding_color_bgr)
                
            # Run face detection on the (possibly rotated) image
            final_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            mp_final = mp.Image(image_format=mp.ImageFormat.SRGB, data=final_rgb)
            final_results = detector.detect(mp_final)
            
            if not final_results.detections:
                st.warning("⚠️ Wajah tidak terdeteksi. Silakan coba foto lain atau sesuaikan parameter rotasi.")
            else:
                final_det = get_largest_detection(final_results.detections)
                
                st.markdown("### 🛠️ Penyesuaian Halus (Fine-Tuning)")
                st.write("Sesuaikan posisi crop secara interaktif:")
                
                # Fine-tuning sliders
                col_slider1, col_slider2 = st.columns(2)
                with col_slider1:
                    offset_x = st.slider(
                        "Geser Horizontal (Offset X px)",
                        min_value=-500, max_value=500, value=0, step=1,
                        help="Geser kotak crop ke kiri (negatif) atau kanan (positif)."
                    )
                with col_slider2:
                    offset_y = st.slider(
                        "Geser Vertikal (Offset Y px)",
                        min_value=-500, max_value=500, value=0, step=1,
                        help="Geser kotak crop ke atas (negatif) atau bawah (positif)."
                    )
                
                # Let user override face ratio for this photo
                col_check1, col_check2 = st.columns([1, 2])
                with col_check1:
                    override_ratio = st.checkbox("Gunakan Rasio Wajah Khusus", value=False)
                with col_check2:
                    current_face_ratio = face_ratio
                    if override_ratio:
                        current_face_ratio = st.slider(
                            "Rasio Wajah Khusus",
                            min_value=0.15, max_value=0.60, value=face_ratio, step=0.01,
                            key="single_face_ratio"
                        )
                
                # Calculate horizontal crop center
                img_h, img_w, _ = processed_img.shape
                fkp0 = final_det.keypoints[0]
                fkp1 = final_det.keypoints[1]
                fx1 = fkp0.x * img_w
                fx2 = fkp1.x * img_w
                f_eye_cx = (fx1 + fx2) / 2
                
                crop_cx = (center_weight * f_eye_cx) + ((1.0 - center_weight) * (img_w / 2))
                crop_cx += offset_x
                
                # Crop logic
                cropped_img, (cx1, cy1, cx2, cy2) = crop_face_to_pasfoto(
                    image=processed_img,
                    bbox=final_det.bounding_box,
                    target_w=target_w,
                    target_h=target_h,
                    face_ratio=current_face_ratio,
                    vertical_pos=vertical_pos,
                    padding_color_bgr=padding_color_bgr,
                    custom_cx=crop_cx,
                    custom_cy_offset=offset_y
                )
                
                # Create visual confirmation image
                vis_img = processed_img.copy()
                
                # Draw face bounding box (Green)
                fb = final_det.bounding_box
                cv2.rectangle(
                    vis_img, 
                    (int(fb.origin_x), int(fb.origin_y)), 
                    (int(fb.origin_x + fb.width), int(fb.origin_y + fb.height)), 
                    (0, 255, 0), 3
                )
                
                # Draw crop area box (Yellow for crop box)
                cv2.rectangle(vis_img, (cx1, cy1), (cx2, cy2), (0, 255, 255), 4)
                
                # Draw eyes center (Blue dot)
                cv2.circle(vis_img, (int(f_eye_cx), int((fkp0.y * img_h + fkp1.y * img_h)/2)), 8, (255, 0, 0), -1)
                
                # Convert back to PIL for displaying in Streamlit
                vis_img_pil = Image.fromarray(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
                cropped_img_pil = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
                
                # Columns for showing side-by-side preview
                col_preview1, col_preview2 = st.columns(2)
                with col_preview1:
                    st.write("**Deteksi & Area Potong (Crop Box)**")
                    st.image(vis_img_pil, use_container_width=True)
                    st.caption("Legenda: Boks Hijau = Deteksi Wajah | Boks Kuning = Area Potong | Titik Biru = Pusat Mata")
                with col_preview2:
                    st.write("**Hasil Crop Pasfoto Final**")
                    st.image(cropped_img_pil, width=300)
                    st.write(f"Ukuran: {target_w} x {target_h} piksel")
                    
                    # Convert to bytes for download
                    buf = io.BytesIO()
                    cropped_img_pil.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    
                    # Clean filename
                    base_name = Path(uploaded_file.name).stem
                    st.download_button(
                        label="📥 Unduh Hasil Pasfoto (PNG)",
                        data=byte_im,
                        file_name=f"pasfoto_3x4_{base_name}.png",
                        mime="image/png"
                    )

# ----------------- TAB 2: BATCH PROCESSING -----------------
with tab_batch:
    st.subheader("Unggah & Crop Banyak Foto Sekaligus")
    st.write("Semua foto akan dipotong otomatis menggunakan pengaturan global dari panel kiri (sidebar) termasuk sudut rotasi.")
    
    batch_files = st.file_uploader(
        "Pilih beberapa foto (.jpg, .jpeg, .png, .webp)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="batch_uploader"
    )
    
    if batch_files:
        st.write(f"Menemukan **{len(batch_files)}** foto untuk diproses.")
        
        if st.button("🚀 Mulai Potong Massal (Batch Process)"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Zip file in-memory
            zip_buffer = io.BytesIO()
            success_count = 0
            fail_count = 0
            
            cropped_previews = []
            
            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                for i, u_file in enumerate(batch_files):
                    status_text.text(f"Memproses ({i+1}/{len(batch_files)}): {u_file.name}")
                    
                    img = load_image(u_file)
                    if img is not None:
                        # Apply manual rotation on the original image (around center) if angle != 0.0
                        processed_img = img.copy()
                        if rotation_val != 0.0:
                            processed_img = rotate_image_around_center(img, rotation_val, padding_color_bgr)
                                
                        # Run final face detection on processed image
                        final_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                        mp_final = mp.Image(image_format=mp.ImageFormat.SRGB, data=final_rgb)
                        final_results = detector.detect(mp_final)
                        
                        if final_results.detections:
                            final_det = get_largest_detection(final_results.detections)
                            
                            img_h, img_w, _ = processed_img.shape
                            fkp0 = final_det.keypoints[0]
                            fkp1 = final_det.keypoints[1]
                            fx1 = fkp0.x * img_w
                            fx2 = fkp1.x * img_w
                            f_eye_cx = (fx1 + fx2) / 2
                            
                            crop_cx = (center_weight * f_eye_cx) + ((1.0 - center_weight) * (img_w / 2))
                            
                            cropped_img, _ = crop_face_to_pasfoto(
                                image=processed_img,
                                bbox=final_det.bounding_box,
                                target_w=target_w,
                                target_h=target_h,
                                face_ratio=face_ratio,
                                vertical_pos=vertical_pos,
                                padding_color_bgr=padding_color_bgr,
                                custom_cx=crop_cx,
                                custom_cy_offset=0
                            )
                            
                            # Convert to bytes
                            cropped_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                            pil_cropped = Image.fromarray(cropped_rgb)
                            
                            img_byte_arr = io.BytesIO()
                            pil_cropped.save(img_byte_arr, format="PNG")
                            img_byte_arr = img_byte_arr.getvalue()
                            
                            # Add to ZIP
                            base_name = Path(u_file.name).stem
                            zip_file.writestr(f"pasfoto_3x4_{base_name}.png", img_byte_arr)
                            
                            # Store preview info
                            cropped_previews.append((u_file.name, pil_cropped, "Sukses"))
                            success_count += 1
                        else:
                            cropped_previews.append((u_file.name, None, "Wajah Tidak Terdeteksi"))
                            fail_count += 1
                    else:
                        cropped_previews.append((u_file.name, None, "Gagal Membaca File"))
                        fail_count += 1
                        
                    progress_bar.progress((i + 1) / len(batch_files))
            
            status_text.text("Pemrosesan massal selesai!")
            
            # Show summary
            st.success(f"Berhasil memotong {success_count} foto. Gagal: {fail_count} foto.")
            
            # Download zip button
            if success_count > 0:
                zip_buffer.seek(0)
                st.download_button(
                    label="📥 Unduh Semua Hasil (ZIP)",
                    data=zip_buffer,
                    file_name="pasfoto_cropped_batch.zip",
                    mime="application/zip"
                )
                
            # Display preview grid
            st.markdown("### Preview Hasil Batch")
            cols_grid = st.columns(4)
            for idx, (name, preview_img, status) in enumerate(cropped_previews):
                col_grid = cols_grid[idx % 4]
                with col_grid:
                    st.write(f"**{name}**")
                    if status == "Sukses" and preview_img is not None:
                        st.image(preview_img, use_container_width=True)
                        st.caption("✅ Sukses")
                    else:
                        st.error(f"❌ {status}")
