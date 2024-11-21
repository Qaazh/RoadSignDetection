# 🚦 Road Sign Detection System

A machine learning-based system for detecting and classifying road signs. Designed to improve traffic management and assist autonomous driving systems. Developed as a Hackaton challenge.

---

## **Features**

- **Single-Object Detection**:  
  - A CNN model built with TensorFlow/Keras capable of classifying 43 different road sign categories.
  - User-friendly **Tkinter GUI** for real-time predictions.

- **Multi-Object Detection (In Development)**:  
  - Leveraging **PyTorch** and pre-trained models (e.g., ResNet50V2) to detect multiple road signs in a single frame.

- **Interactive GUI**:  
  - Easy-to-use interface for image uploads and predictions.  
  - Real-time classification and detailed results display.

---

## **Technologies Used**

- **Programming Languages**: Python
- **Frameworks**: TensorFlow/Keras, PyTorch
- **Image Processing**: OpenCV, PIL
- **GUI**: Tkinter
- **Data Handling**: JSON, Annotation Parsing

---

## **How to Use**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/<your-username>/Road-Sign-Detection.git
   cd Road-Sign-Detection
2. **Install Dependencies**:
   Use pip to install required Python libraries:
   ```bash
   pip install -r requirements.txt
3. **Run the application**:
   For single-object detection
   ```bash
   python gui.py

## **Future improvements**
- Complete the multi-object detection pipeline with pre-trained models.
- Enhance GUI features with drag-and-drop functionality.
- Integrate real-time video feed for live road sign detection.

## **Project Structure**
```plaintext
Road-Sign-Detection/
├── gui.py                  # GUI application for predictions
├── training.py             # Model training script
├── traffic_classifier.keras  # Pre-trained single-object model
├── final_main.py           # Multi-object detection script (in progress)
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation ```


## **Acknowledgments**
The dataset was sourced from the German Traffic Sign Recognition Benchmark (GTSRB).
