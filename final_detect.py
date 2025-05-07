import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import pytesseract
import numpy as np
import Levenshtein
import re

pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"
st.set_page_config(page_title="Vehicle Detection and OCR", layout="wide")

st.title("🚘 Vehicle Detection and OCR (YOLOv8 + Tesseract)")

model = YOLO("yolov8n.pt")

tess_config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

vehicle_classes = ["car", "truck", "bus", "motorbike"]
detected_plates = set()
plate_pattern = r"[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}"
video_source = st.radio("Choose video source", ["Webcam", "Upload Video"])


def detect_vehicles_and_ocr(frame):
    results = model(frame, imgsz=640, conf=0.5)

    for result in results:
        boxes = result.boxes
        names = result.names

        for box in boxes:
            cls_id = int(box.cls[0])
            label = names[cls_id]

            if label.lower() in vehicle_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                vehicle_roi = frame[y1:y2, x1:x2]

                # Focus on lower part
                h = y2 - y1
                plate_region = vehicle_roi[int(h * 0.6) :, :]

                # OCR Preprocessing
                gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
                gray = cv2.bilateralFilter(gray, 11, 17, 17)
                _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

                # OCR
                text = pytesseract.image_to_string(thresh, config=tess_config).strip()
                text = re.sub(r"\W+", "", text).upper()

                if re.fullmatch(plate_pattern, text):
                    # Check if it's a near duplicate
                    is_duplicate = any(
                        Levenshtein.distance(text, plate) <= 2
                        for plate in detected_plates
                    )

                    if not is_duplicate:
                        detected_plates.add(text)
                        st.write(f"🔍 Detected Plate: `{text}`")
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            frame,
                            text,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 0, 0),
                            2,
                        )

    return frame


# Webcam Mode
if video_source == "Webcam":
    run = st.checkbox("Start Webcam")
    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.write("❌ Failed to grab frame.")
            break
        processed = detect_vehicles_and_ocr(frame)
        FRAME_WINDOW.image(processed, channels="BGR")
    cap.release()

# Upload Mode
else:
    uploaded_file = st.file_uploader("📂 Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        FRAME_WINDOW = st.image([])

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            processed = detect_vehicles_and_ocr(frame)
            FRAME_WINDOW.image(processed, channels="BGR")
        cap.release()
