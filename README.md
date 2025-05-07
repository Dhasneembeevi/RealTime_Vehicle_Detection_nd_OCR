## 🚘 Real-Time Vehicle Detection and OCR (YOLOv8 + Tesseract)

This Streamlit-based web app detects vehicles in real-time from a webcam or uploaded video and extracts license plate numbers using **YOLOv8** for object detection and **Tesseract OCR** for text recognition.

---

### 📸 Features

* Detects vehicles like **cars, buses, trucks, and motorbikes**
* Extracts and displays license plate numbers using OCR
* Supports **webcam** and **video file uploads**
* Filters out duplicate or near-duplicate plates
* Real-time processing with live UI using **Streamlit**

---

### 🧠 Tech Stack

* **Python**
* **YOLOv8** (via Ultralytics)
* **Tesseract OCR**
* **OpenCV**
* **Streamlit**
* **Levenshtein Distance** (to reduce false OCR duplicates)
* **Regex** for plate pattern validation

---

### 🛠️ Installation

1. **Clone the repository**:

   ```bash
   https://github.com/Dhasneembeevi/RealTime_Vehicle_Detection_nd_OCR.git
   cd RealTime_Vehicle_Detection_nd_OCR
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Install Tesseract OCR**:

   * Download from [https://github.com/tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract)
   * Set the path to your Tesseract executable in the script:

     ```python
     pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"
     ```

---

### ▶️ Run the App

```bash
streamlit run app.py
```

Then open the URL shown in your terminal (usually `http://localhost:8501`) to start using the app.

---

### 📁 Example Output

* 🟩 Bounding boxes around vehicles
* 🔍 License plate numbers extracted and displayed in real-time

---

### 📌 Notes

* This app works best in good lighting and clear plate visibility.
* The OCR result may vary based on image clarity and resolution.
* The YOLO model used is `yolov8n.pt` (YOLOv8 Nano) for speed it can be replaced it with a heavier version for better accuracy.
