# 🅿 ParkIntel – AI Smart Parking Management System

> **YOLOv11 vehicle detection + Random Forest prediction + Streamlit dashboard**  
> Dataset: *pfe Object Detection* by bouakkaz144.lotfi@gmail.com (Roboflow)

---

## 🏗 Project Structure

```
ParkIntel/
├── modules/
│   ├── module1_data_acquisition.py    # Video input & frame extraction
│   ├── module2_vehicle_detection.py   # YOLOv11 vehicle detection
│   ├── module3_occupancy.py           # Occupancy estimation & CSV logging
│   ├── module4_prediction.py          # Random Forest 6-hour forecasting
│   ├── module5_peak_hours.py          # Peak hour & congestion analysis
│   ├── module6_psi.py                 # Parking Stress Index (YOUR INNOVATION)
│   ├── module7_recommendation.py      # Explainable AI recommendations
│   └── module8_simulation.py          # What-if scenario simulation
├── data/
│   └── occupancy_log.csv              # Auto-generated occupancy log
├── models/
│   ├── best.pt                        # YOLOv11 weights (Roboflow)
│   └── rf_model.pkl                   # Trained Random Forest model
├── dashboard.py                       # Streamlit dashboard (Module 9)
├── main_pipeline.py                   # Full pipeline orchestrator
├── download_model.py                  # Roboflow model downloader
└── requirements.txt                   # Python dependencies
```

---

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup (Generate Training Data + Train ML Model)
```bash
python main_pipeline.py --mode setup
```

### 3. Launch Dashboard
```bash
streamlit run dashboard.py
```

### 4. Run Demo Pipeline (no camera needed)
```bash
python main_pipeline.py --mode demo
```

### 5. Use Live Camera or Video File
```bash
# Webcam
python main_pipeline.py --mode live --source 0

# Video file
python main_pipeline.py --mode live --source parking.mp4

# RTSP CCTV stream
python main_pipeline.py --mode live --source rtsp://192.168.1.10/stream
```

### 6. Full Report
```bash
python main_pipeline.py --mode report
```

### 7. All Steps at Once
```bash
python main_pipeline.py --mode all
```

---

## 🤖 Download Your Roboflow YOLOv11 Model

```bash
# Method 1: Automatic download
python download_model.py --key YOUR_API_KEY --workspace YOUR_WORKSPACE

# Method 2: Manual instructions
python download_model.py --manual
```

**Get your API key at:** https://app.roboflow.com/settings/api

If no model is found, the system runs in **demo mode** (simulates detections) — all other modules work fully.

---

## 🧠 System Architecture

```
CCTV Video Input
      ↓
Module 1: Frame Extraction (OpenCV)
      ↓
Module 2: YOLOv11 Vehicle Detection (Roboflow model)
      ↓
Module 3: Occupancy Estimation → CSV Log
      ↓
Module 4: Random Forest Prediction (next 6 hours)
      ↓
Module 5: Peak Hour Analysis
      ↓
Module 6: Parking Stress Index (PSI) ← YOUR INNOVATION
      ↓
Module 7: Explainable AI Recommendation
      ↓
Module 8: What-If Simulation
      ↓
Module 9: Streamlit Dashboard
```

---

## 📊 PSI Formula (Your Innovation)

```
PSI = (Current Occupancy / Total Capacity) × (Predicted Demand / Total Capacity)
```

| PSI Range | Level  | Meaning                          |
|-----------|--------|----------------------------------|
| < 0.30    | 🟢 LOW    | Comfortable – no action needed  |
| 0.30–0.69 | 🟡 MEDIUM | Monitor closely                 |
| ≥ 0.70    | 🔴 HIGH   | Critical – activate overflow    |

---

## 🛠 Tools Used

| Purpose           | Tool                         |
|-------------------|------------------------------|
| Object Detection  | YOLOv11 (Ultralytics)        |
| Training Dataset  | Roboflow – pfe OD            |
| Video Processing  | OpenCV                       |
| ML Prediction     | Scikit-learn (Random Forest) |
| Data Handling     | Pandas, NumPy                |
| Dashboard         | Streamlit                    |
| Visualization     | Plotly                       |
| Model Save/Load   | Joblib                       |

---

## 🌍 Real-World Applications

- 🏪 Shopping malls
- ✈️ Airports
- 🏥 Hospitals
- 🏙 Smart city infrastructure
- 🏢 IT parks / corporate campuses

---

## 📋 CLI Reference

```bash
python main_pipeline.py --help

Options:
  --mode    setup | demo | live | report | all
  --source  None=demo, 0=webcam, path.mp4, rtsp://...
  --slots   Total parking capacity (default: 50)
  --model   Path to YOLOv11 .pt weights
  --days    Historical data generation days (default: 30)
  --iter    Demo loop iterations (default: 8)
  --delay   Demo loop delay in seconds (default: 0.8)
```

---

## 🔬 Module Test (individual)

```bash
cd ParkIntel
python modules/module1_data_acquisition.py
python modules/module2_vehicle_detection.py
python modules/module3_occupancy.py
python modules/module4_prediction.py
python modules/module5_peak_hours.py
python modules/module6_psi.py
python modules/module7_recommendation.py
python modules/module8_simulation.py
python main_pipeline.py --mode all
```

python c:/projects/ParkIntel/test_detection.py
streamlit run c:/projects/ParkIntel/dashboard.py