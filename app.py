import streamlit as st
import cv2
import numpy as np
from PIL import Image
# import io


def find_defects(master_img, input_img):
    # Convert to grayscale
    master_gray = cv2.cvtColor(master_img, cv2.COLOR_BGR2GRAY)
    input_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and minor variations
    master_gray_blur = cv2.GaussianBlur(master_gray, (5, 5), 0)
    input_gray_blur = cv2.GaussianBlur(input_gray, (5, 5), 0)

    # Image subtraction
    diff_img = cv2.absdiff(master_gray_blur, input_gray_blur)

    # Apply precise thresholding to highlight differences
    _, thresh_img = cv2.threshold(diff_img, 5, 255, cv2.THRESH_BINARY)

    # Use morphological operations to enhance the differences
    kernel = np.ones((3, 3), np.uint8)
    dilated_img = cv2.dilate(thresh_img, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around the defects
    for contour in contours:
        if cv2.contourArea(contour) > 5:  # Adjust the area threshold for smaller defects
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(input_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return input_img


# Streamlit application
st.title("Defect Detection Application")

# Upload master and input images
uploaded_master = st.file_uploader("Choose the master image...", type=["jpg", "jpeg", "png"])
uploaded_input = st.file_uploader("Choose the input image...", type=["jpg", "jpeg", "png"])

if uploaded_master is not None and uploaded_input is not None:
    master_img = Image.open(uploaded_master)
    input_img = Image.open(uploaded_input)

    master_img = np.array(master_img)
    input_img = np.array(input_img)

    if master_img.shape == input_img.shape:
        # Detect defects
        result_img = find_defects(master_img, input_img)

        # Convert result image to display in Streamlit
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        result_pil = Image.fromarray(result_img)

        # Display images
        # st.image(master_img, caption='Master Image', use_column_width=True)
        # st.image(input_img, caption='Input Image', use_column_width=True)
        st.image(result_pil, caption='Output Image with Defects Marked', use_column_width=True)
    else:
        st.error("The master and input images do not have the same dimensions.")
