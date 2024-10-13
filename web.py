import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

def apply_denoising(img):
    return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

def apply_sharpening(img):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(img, -1, kernel)

def apply_sobel_edge_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_8U, 0, 1, ksize=3)
    return cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)

def apply_prewitt_edge_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    img_prewittx = cv2.filter2D(gray, -1, kernelx)
    img_prewitty = cv2.filter2D(gray, -1, kernely)
    return cv2.addWeighted(img_prewittx, 0.5, img_prewitty, 0.5, 0)

def apply_canny_edge_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, 50, 150)


st.title("Ứng Dụng Xử Lý Ảnh với Streamlit")


uploaded_files = st.file_uploader("Tải lên ảnh (png, jpg)", accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
   
        img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

    
        denoised_image = apply_denoising(img)
        sharpened_image = apply_sharpening(img)
        sobel_image = apply_sobel_edge_detection(img)
        prewitt_image = apply_prewitt_edge_detection(img)
        canny_image = apply_canny_edge_detection(img)

    
        denoised_image_rgb = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB)
        sharpened_image_rgb = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB)
        sobel_image_rgb = cv2.cvtColor(sobel_image, cv2.COLOR_BGR2RGB)
        prewitt_image_rgb = cv2.cvtColor(prewitt_image, cv2.COLOR_BGR2RGB)
        canny_image_rgb = cv2.cvtColor(canny_image, cv2.COLOR_BGR2RGB)


        st.subheader(f"Ảnh Gốc: {uploaded_file.name}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
        st.image(img_rgb, use_column_width=True)

        col1, col2, col3, col4, col5 = st.columns(5)

        def convert_to_jpg(image):
            """Chuyển đổi ảnh numpy array thành định dạng JPG và trả về bytes."""
            pil_img = Image.fromarray(image)
            img_byte_arr = io.BytesIO()
            pil_img.convert("RGB").save(img_byte_arr, format='JPEG')
            return img_byte_arr.getvalue()

        with col1:
            st.subheader("Denoised")
            st.image(denoised_image_rgb, use_column_width=True)
            st.download_button(
                label="Tải xuống Denoised",
                data=convert_to_jpg(denoised_image_rgb),
                file_name=f"denoised_{uploaded_file.name.split('.')[0]}.jpg",
                mime="image/jpeg"
            )

        with col2:
            st.subheader("Sharpened")
            st.image(sharpened_image_rgb, use_column_width=True)
            st.download_button(
                label="Tải xuống Sharpened",
                data=convert_to_jpg(sharpened_image_rgb),
                file_name=f"sharpened_{uploaded_file.name.split('.')[0]}.jpg",
                mime="image/jpeg"
            )

        with col3:
            st.subheader("Sobel Edge")
            st.image(sobel_image_rgb, use_column_width=True)
            st.download_button(
                label="Tải xuống Sobel",
                data=convert_to_jpg(sobel_image_rgb),
                file_name=f"sobel_{uploaded_file.name.split('.')[0]}.jpg",
                mime="image/jpeg"
            )

        with col4:
            st.subheader("Prewitt Edge")
            st.image(prewitt_image_rgb, use_column_width=True)
            st.download_button(
                label="Tải xuống Prewitt",
                data=convert_to_jpg(prewitt_image_rgb),
                file_name=f"prewitt_{uploaded_file.name.split('.')[0]}.jpg",
                mime="image/jpeg"
            )

        with col5:
            st.subheader("Canny Edge")
            st.image(canny_image_rgb, use_column_width=True)
            st.download_button(
                label="Tải xuống Canny",
                data=convert_to_jpg(canny_image_rgb),
                file_name=f"canny_{uploaded_file.name.split('.')[0]}.jpg",
                mime="image/jpeg"
            )
