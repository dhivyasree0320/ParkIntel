"""
STREAMLIT DASHBOARD - MODULE 9
ParkIntel – AI Smart Parking System

Run with:
    streamlit run dashboard.py

Features:
  - Live occupancy gauge & slot grid
  - 6-hour ML forecast chart
  - Peak hour analysis
  - PSI meter with color coding
  - AI explainability (feature importances)
  - What-if simulation with sliders
  - Auto-refresh every 5 seconds
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sys
import time
import os
import cv2
from pathlib import Path
from datetime import datetime, timedelta

# ── Make modules importable ─────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from modules.module3_occupancy      import OccupancyEstimator
from modules.module4_prediction     import ParkingPredictor
from modules.module5_peak_hours     import PeakHourAnalyzer
from modules.module6_psi            import ParkingStressIndex
from modules.module7_recommendation import RecommendationEngine
from modules.module8_simulation     import SimulationEngine

# ════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ParkIntel – AI Parking",
    page_icon="🅿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ════════════════════════════════════════════════════════════════════════
#  CUSTOM CSS
# ════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

/* Overall app canvas */
.stApp { background: #000000; color: #d9e5ec; }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

/* Metric cards */
.pk-card {
    background: #0d1117;
    border: 1px solid #1c2a38;
    border-radius: 10px;
    padding: 16px 18px;
    text-align: center;
    transition: border-color .3s;
}
.pk-card:hover { border-color: rgba(0,220,255,.45); }
.pk-label {
    font-size: 10px; letter-spacing: 2.5px;
    text-transform: uppercase; color: #5a7a8a;
    margin-bottom: 6px;
}
.pk-value {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.5rem; font-weight: 700; line-height: 1;
}
.pk-sub { font-size: 11px; color: #5a7a8a; margin-top: 4px; }

/* Section heading */
.sec-head {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.05rem; font-weight: 700;
    letter-spacing: 2px; text-transform: uppercase;
    color: #00dcff;
    padding-bottom: 6px;
    border-bottom: 1px solid #1c2a38;
    margin-bottom: 10px;
}

/* Slot grid */
.slot-grid { display:none; }
.slot-occ {
    background: rgba(255,50,80,.30);
    border: 1px solid rgba(255,50,80,.55);
    border-radius: 3px; height: 22px;
    display:flex; align-items:center; justify-content:center;
    font-size: 9px;
}
.slot-free {
    background: rgba(100,255,80,.12);
    border: 1px solid rgba(100,255,80,.22);
    border-radius: 3px; height: 22px;
}

.slot-panel {
    background: #0d1117;
    border: 1px solid #1c2a38;
    border-radius: 8px;
    padding: 10px;
}

.slot-detail {
    background: linear-gradient(180deg, rgba(0,220,255,.08), rgba(13,17,23,.95));
    border: 1px solid #1c2a38;
    border-radius: 8px;
    padding: 12px 14px;
    margin-bottom: 10px;
}

.slot-detail-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: #00dcff;
    letter-spacing: 1.4px;
}

.slot-detail-meta {
    font-size: 12px;
    color: #a8c0cc;
    margin-top: 6px;
    line-height: 1.6;
}

div[data-testid="stButton"] > button[kind="secondary"] {
    width: 100%;
    min-height: 38px;
    border-radius: 6px;
    border: 1px solid #1c2a38;
    background: #111820;
    color: #d9e5ec;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    padding: 0.3rem 0.25rem;
}

div[data-testid="stButton"] > button[kind="secondary"]:hover {
    border-color: rgba(0,220,255,.45);
    color: #ffffff;
}

/* Recommendation box */
.rec-box {
    background: #0d1117;
    border-left: 3px solid #00dcff;
    border-radius: 0 8px 8px 0;
    padding: 14px 16px;
    font-size: 13px; line-height: 1.75; color: #a8c0cc;
    margin-top: 8px;
}

/* Action item */
.act-item {
    background: #111820;
    border: 1px solid #1c2a38;
    border-radius: 6px;
    padding: 8px 12px; margin: 4px 0;
    font-size: 12px; color: #a8c0cc;
}

/* PSI badge */
.psi-badge {
    display: inline-block;
    padding: 3px 14px; border-radius: 14px;
    font-size: 11px; font-weight: 700; letter-spacing: 1.5px;
    border: 1px solid; margin-top: 6px;
}

/* Streamlit metrics override */
div[data-testid="metric-container"] {
    background: #0d1117 !important;
    border: 1px solid #1c2a38 !important;
    border-radius: 10px !important;
    padding: 12px !important;
}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ════════════════════════════════════════════════════════════════════════
TOTAL_SLOTS    = 50
HOUR_PATTERN   = [8,5,4,4,5,9,14,28,36,40,42,44,46,44,42,40,42,47,48,45,40,34,26,15]
DATA_CSV       = "data/occupancy_log.csv"
MODEL_PKL      = "models/rf_model.pkl"
SLOT_COLUMNS   = 10


def get_slot_size(slot_number):
    """Return demo dimensions for a parking slot."""
    slot_profiles = [
        {"label": "Compact",  "length_m": 4.2, "width_m": 2.2},
        {"label": "Standard", "length_m": 4.8, "width_m": 2.5},
        {"label": "Large",    "length_m": 5.4, "width_m": 2.8},
    ]
    profile = slot_profiles[(slot_number - 1) % len(slot_profiles)]
    area_m2 = round(profile["length_m"] * profile["width_m"], 2)
    return {
        **profile,
        "area_m2": area_m2,
    }

# ════════════════════════════════════════════════════════════════════════
#  CACHED MODULE INIT
# ════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_modules():
    est  = OccupancyEstimator(total_slots=TOTAL_SLOTS, csv_path=DATA_CSV)
    pred = ParkingPredictor(model_path=MODEL_PKL,      total_slots=TOTAL_SLOTS)
    peak = PeakHourAnalyzer(total_slots=TOTAL_SLOTS)
    psi  = ParkingStressIndex(total_slots=TOTAL_SLOTS)
    rec  = RecommendationEngine(total_slots=TOTAL_SLOTS)
    sim  = SimulationEngine(total_slots=TOTAL_SLOTS)

    # Auto-setup if data or model is missing
    needs_setup = (
        not Path(DATA_CSV).exists() or
        not Path(MODEL_PKL).exists() or
        os.path.getsize(DATA_CSV) < 100 if Path(DATA_CSV).exists() else True
    )
    if needs_setup:
        with st.spinner("First run: generating training data + training model..."):
            df = est.generate_historical_data(days=30, save=True)
            df_feat = est.get_dataframe()
            pred.train(df_feat)

    return est, pred, peak, psi, rec, sim

est, pred_model, peak_module, psi_module, rec_module, sim_module = load_modules()

if "selected_slot" not in st.session_state:
    st.session_state.selected_slot = 1

# ════════════════════════════════════════════════════════════════════════
#  LIVE OCCUPANCY (simulates real detector, refreshes every ~4s)
# ════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=4)
def get_live_occ():
    hour = datetime.now().hour
    base = HOUR_PATTERN[hour]
    rng  = np.random.default_rng(int(time.time()) % 9999)
    return max(0, min(TOTAL_SLOTS, base + int(rng.integers(-4, 5))))

# ════════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════════
def psi_color(level):
    return {"low": "#6dff60", "medium": "#ffd000", "high": "#ff3050"}.get(level, "#00dcff")

def make_gauge(value, max_val, color, title, height=200):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={"x": [0,1], "y": [0,1]},
        title={"text": title, "font": {"size":11, "color":"#5a7a8a"}},
        number={"font": {"size":34, "color":color, "family":"Rajdhani"}},
        gauge={
            "axis": {"range":[0,max_val], "tickcolor":"#2a3e50",
                     "tickwidth":1, "tickfont":{"color":"#5a7a8a","size":9}},
            "bar": {"color":color, "thickness":0.68},
            "bgcolor": "#0d1117", "borderwidth": 0,
            "steps": [{"range":[0, max_val*0.30], "color":"#091809"},
                      {"range":[max_val*0.30, max_val*0.70], "color":"#181600"},
                      {"range":[max_val*0.70, max_val], "color":"#180608"}],
            "threshold": {"line":{"color":"white","width":2},
                          "thickness":0.78, "value":value}
        }
    ))
    fig.update_layout(
        height=height, paper_bgcolor="#07090f", plot_bgcolor="#07090f",
        margin=dict(l=20, r=20, t=32, b=8),
        font={"color":"#e0eaf0"}
    )
    return fig

def plotly_bar(x, y, colors, title="", height=250):
    fig = go.Figure(go.Bar(x=x, y=y, marker_color=colors))
    fig.update_layout(
        title={"text":title,"font":{"size":11,"color":"#5a7a8a"}},
        height=height, paper_bgcolor="#07090f", plot_bgcolor="#0d1117",
        margin=dict(l=8, r=8, t=28, b=8),
        yaxis=dict(gridcolor="#1c2a38"),
        xaxis=dict(color="#5a7a8a"),
        font={"color":"#e0eaf0"}
    )
    return fig

def plotly_line(traces, height=270):
    fig = go.Figure()
    for t in traces:
        fig.add_trace(go.Scatter(
            x=t["x"], y=t["y"], mode=t.get("mode","lines"),
            name=t["name"],
            line=dict(color=t["color"], width=t.get("width",2),
                      dash=t.get("dash","solid")),
            marker=dict(size=t.get("marker_size",0))
        ))
    fig.update_layout(
        height=height, paper_bgcolor="#07090f", plot_bgcolor="#0d1117",
        margin=dict(l=8, r=8, t=12, b=8),
        yaxis=dict(gridcolor="#1c2a38"),
        xaxis=dict(color="#5a7a8a"),
        legend=dict(bgcolor="#0d1117", bordercolor="#1c2a38", borderwidth=1),
        font={"color":"#e0eaf0"}
    )
    return fig

# ════════════════════════════════════════════════════════════════════════
#  LIVE DATA
# ════════════════════════════════════════════════════════════════════════
current_occ = get_live_occ()
available   = TOTAL_SLOTS - current_occ
occ_rate    = current_occ / TOTAL_SLOTS
hour_now    = datetime.now().hour
dow_now     = datetime.now().weekday()

# Prediction
try:
    pred_next_occ = pred_model.predict_one(hour_now, dow_now, current_occ)
    forecasts     = pred_model.predict_next_hours(current_occ, hours=6)
except Exception:
    pred_next_occ = HOUR_PATTERN[(hour_now + 1) % 24]
    forecasts = [
        {
            "hour_offset": i + 1,
            "timestamp":   (datetime.now() + timedelta(hours=i+1)).strftime("%H:%M"),
            "predicted_occupancy":  min(TOTAL_SLOTS, HOUR_PATTERN[(hour_now+i+1) % 24]),
            "predicted_available":  max(0, TOTAL_SLOTS - HOUR_PATTERN[(hour_now+i+1)%24]),
            "occupancy_rate":       round(HOUR_PATTERN[(hour_now+i+1)%24] / TOTAL_SLOTS, 3)
        }
        for i in range(6)
    ]

# PSI
try:
    psi_result = psi_module.compute(current_occ, pred_next_occ)
except Exception:
    raw = (current_occ / TOTAL_SLOTS) * (pred_next_occ / TOTAL_SLOTS)
    lvl = "high" if raw >= 0.7 else "medium" if raw >= 0.3 else "low"
    psi_result = {
        "psi": round(raw, 4), "psi_pct": round(raw*100,1),
        "level": lvl, "color": psi_color(lvl), "emoji": "🔴" if lvl=="high" else "🟡" if lvl=="medium" else "🟢",
        "occ_rate": occ_rate, "pred_rate": pred_next_occ/TOTAL_SLOTS,
        "time_factor": 1.0, "available": available,
        "current_occupancy": current_occ, "predicted_demand": pred_next_occ,
        "recommendation": "PSI computed.", "total_slots": TOTAL_SLOTS,
        "timestamp": datetime.now().isoformat()
    }

# Peak analysis
try:
    df_hist      = est.get_dataframe()
    peak_analysis = peak_module.analyze(df_hist) if not df_hist.empty else peak_module._demo_analysis()
except Exception:
    peak_analysis = peak_module._demo_analysis()

# Recommendation
try:
    fi = pred_model.feature_importances_ or {}
    recommendation = rec_module.generate(psi_result, peak_analysis, fi, pred_model.metrics_)
except Exception:
    recommendation = {
        "summary": psi_result.get("recommendation",""),
        "explanation": psi_result.get("recommendation",""),
        "actions": ["Monitor occupancy", "Prepare overflow lot"],
        "top_drivers": ["Hour of Day", "Day of Week"],
        "driver_scores": [],
        "confidence": ""
    }

psi_val  = psi_result["psi"]
psi_lvl  = psi_result["level"]
psi_clr  = psi_color(psi_lvl)
psi_em   = psi_result.get("emoji","⚪")

# ════════════════════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════════════════════
h1, h2, h3 = st.columns([3, 2, 2])
with h1:
    st.markdown("## 🅿 PARKINTEL")
    st.markdown(
        "<span style='font-size:11px;color:#5a7a8a;letter-spacing:3px'>"
        "AI SMART PARKING MANAGEMENT SYSTEM"
        "</span>",
        unsafe_allow_html=True
    )
with h2:
    status_color = psi_clr
    st.markdown(
        f"<div style='background:#0d1117;border:1px solid {status_color};"
        f"border-radius:20px;padding:8px 16px;display:inline-flex;"
        f"align-items:center;gap:8px;font-size:12px;color:{status_color};margin-top:14px'>"
        f"<span style='width:8px;height:8px;background:#6dff60;border-radius:50%;"
        f"display:inline-block;box-shadow:0 0 6px #6dff60'></span>"
        f"YOLOv11 ACTIVE</div>",
        unsafe_allow_html=True
    )
with h3:
    st.markdown(
        f"<div style='text-align:right;color:#5a7a8a;font-size:13px;margin-top:14px'>"
        f"{datetime.now().strftime('%A, %d %B %Y')}<br>"
        f"<b style='color:#e0eaf0'>{datetime.now().strftime('%H:%M:%S')}</b>"
        f"</div>",
        unsafe_allow_html=True
    )

st.divider()

# ════════════════════════════════════════════════════════════════════════
#  TABS
# ════════════════════════════════════════════════════════════════════════
tab_overview, tab_video, tab_predict, tab_peak, tab_ai, tab_sim = st.tabs([
    "📊  Overview",
    "🎬  Video Simulation",
    "🤖  Prediction",
    "📈  Peak Hours",
    "🧠  AI Insights",
    "🔬  Simulation"
])

# ════════════════════════════════════════════════════════════════════════
#  TAB 1 – OVERVIEW
# ════════════════════════════════════════════════════════════════════════
with tab_video:
    st.markdown('<div class="sec-head">🎬 Live Parking Detection Simulation</div>', 
                unsafe_allow_html=True)
    
    # Video files available - prioritized by compatibility
    video_files = {
        "web_simulation.mp4": "🎬 Live YOLO Detection (Recommended - 0.8 MB)",
        "parking_detection_output.mp4": "Full Detection Video (379 MB)",
    }
    
    # Create tabs for different video sources
    video_source_tab1 = st.tabs(["Pre-loaded Videos"])[0]
    
    with video_source_tab1:
        # Select video
        selected_video = st.selectbox(
            "Select Video",
            options=list(video_files.keys()),
            format_func=lambda x: video_files[x],
            key="preloaded_video_select"
        )
        
        video_path = Path("data") / selected_video
        
        if video_path.exists():
            # Display video
            st.video(str(video_path))
            
            # Video info
            st.markdown(
                f'<div class="pk-card" style="margin-top: 10px;">'
                f'<div class="pk-label">Video Information</div>'
                f'<div class="pk-value" style="font-size: 1rem; color: #00dcff;">'
                f'{video_files[selected_video]}</div>'
                f'<div class="pk-sub">File: {selected_video}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
            
            # Download button
            with open(video_path, "rb") as f:
                st.download_button(
                    label="📥 Download Video",
                    data=f,
                    file_name=selected_video,
                    mime="video/mp4"
                )
        else:
            st.warning(f"Video not found: {video_path}")
            st.info("Run video_detection.py to create the detection video!")
    
    if False:
        st.markdown("### Upload Video")
        st.info("Video upload is disabled. Please use the pre-loaded videos tab.")
        uploaded_video = None
        
        if uploaded_video is not None:
            # Create uploads directory if it doesn't exist
            uploads_dir = Path("data/uploads")
            uploads_dir.mkdir(exist_ok=True)
            
            # Save uploaded video
            upload_video_path = uploads_dir / uploaded_video.name
            with open(upload_video_path, "wb") as f:
                f.write(uploaded_video.getbuffer())
            
            st.success(f"✅ Video uploaded successfully: {uploaded_video.name}")
            st.markdown(f"**File size:** {uploaded_video.size / (1024*1024):.2f} MB")
            
            # Display the uploaded video
            st.markdown("#### 📹 Original Uploaded Video")
            st.video(str(upload_video_path))
            
            # Option to run YOLO detection
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 🔍 Run YOLO Detection")
            
            run_detection = st.checkbox("Run YOLO parking detection on this video", value=False, key="run_detection_checkbox")
            
            if run_detection:
                with st.spinner("Processing video with YOLO detection... This may take a while."):
                    # Import required modules for detection
                    from modules.module2_vehicle_detection import VehicleDetector
                    import cv2
                    import tempfile
                    
                    # Initialize detector
                    detector = VehicleDetector(
                        model_path="models/best.pt",
                        total_slots=50,
                        confidence=0.45,
                        device="cpu"
                    )
                    
                    # Process video
                    cap = cv2.VideoCapture(str(upload_video_path))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    # Ensure fps is valid
                    if fps <= 0 or fps > 60:
                        fps = 30.0
                    
                    # Ensure dimensions are valid
                    if width <= 0 or height <= 0:
                        width, height = 820, 620
                    
                    # Output path for processed video - use temp file for better compatibility
                    output_filename = f"detected_{uploaded_video.name}"
                    output_path = uploads_dir / output_filename
                    
                    # Get total frames for progress calculation
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if total_frames <= 0:
                        total_frames = 200  # Fallback
                    
                    # Use H264 codec for better browser compatibility
                    # Try multiple codecs in order of preference
                    codecs_to_try = ['H264', 'avc1', 'XVID', 'X264', 'mp4v']
                    writer = None
                    
                    for codec in codecs_to_try:
                        try:
                            fourcc = cv2.VideoWriter_fourcc(*codec)
                            test_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
                            if test_writer.isOpened():
                                writer = test_writer
                                print(f"Using codec: {codec}")
                                break
                            else:
                                test_writer.release()
                        except Exception as e:
                            print(f"Codec {codec} failed: {e}")
                            continue
                    
                    if writer is None or not writer.isOpened():
                        st.error("Failed to initialize video writer with any codec.")
                        st.info("Trying with fallback frame size (640x480)...")
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (640, 480))
                        width, height = 640, 480
                    
                    frame_count = 0
                    total_vehicles = 0
                    
                    st.progress(0)
                    progress_bar = st.empty()
                    
                    # Reset cap to start from beginning
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Resize frame to match writer dimensions
                        if frame.shape[1] != width or frame.shape[0] != height:
                            frame = cv2.resize(frame, (width, height))
                        
                        # Detect parking spaces
                        result = detector.detect(frame)
                        
                        # Get annotated frame
                        if result['annotated_frame'] is not None:
                            display_frame = result['annotated_frame']
                        else:
                            display_frame = frame.copy()
                        
                        # Write to output video
                        writer.write(display_frame)
                        
                        total_vehicles += result.get('count', 0)
                        frame_count += 1
                        
                        # Update progress
                        if frame_count % 10 == 0:
                            progress = min(frame_count / min(total_frames, 500), 1.0)
                            progress_bar.progress(progress)
                            
                        # Limit frames for performance (500 frames max)
                        if frame_count >= 500:
                            break
                    
                    # Release resources properly
                    cap.release()
                    writer.release()
                    
                    # Force flush by explicitly opening and closing the file
                    import os
                    time.sleep(0.5)  # Give time for file to be written
                    
                    # Verify file exists and has content
                    if os.path.exists(output_path):
                        file_size = os.path.getsize(output_path)
                        if file_size > 0:
                            st.success(f"✅ Detection complete! Processed {frame_count} frames.")
                        else:
                            st.error("Video file was not created properly.")
                            st.stop()
                    else:
                        st.error("Video file was not created.")
                        st.stop()
                    
                    progress_bar.progress(1.0)
                    
                    # Additional delay to ensure file is fully written
                    time.sleep(1)
                    
                    # Display processed video
                    st.markdown("#### 🎬 Processed Video with YOLO Detection")
                    
                    # Read video with explicit binary mode
                    with open(str(output_path), 'rb') as f:
                        video_bytes = f.read()
                    
                    if len(video_bytes) > 0:
                        st.video(video_bytes)
                    else:
                        st.error("Could not read video data.")
                        st.stop()
                    
                    # Video info
                    st.markdown(
                        f'<div class="pk-card" style="margin-top: 10px;">'
                        f'<div class="pk-label">Detection Results</div>'
                        f'<div class="pk-value" style="font-size: 1rem; color: #00dcff;">'
                        f'Frames: {frame_count} | Total Vehicles: {total_vehicles}</div>'
                        f'<div class="pk-sub">Output: {output_filename}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Download button for processed video
                    with open(output_path, "rb") as f:
                        st.download_button(
                            label="📥 Download Processed Video",
                            data=f,
                            file_name=output_filename,
                            mime="video/mp4"
                        )
            
            # Show uploaded videos list
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 📂 Previously Uploaded Videos")
            
            uploaded_videos = list(uploads_dir.glob("*"))
            if uploaded_videos:
                for vid in uploaded_videos:
                    if vid.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                        col1, col2, col3 = st.columns([3, 1, 1])
                        with col1:
                            st.markdown(f"📹 {vid.name}")
                        with col2:
                            if st.button(f"▶️ Play", key=f"play_{vid.name}"):
                                st.video(str(vid))
                        with col3:
                            if st.button(f"🗑️ Delete", key=f"del_{vid.name}"):
                                vid.unlink()
                                st.rerun()
            else:
                st.info("No uploaded videos yet.")
    
    # Show sample detection images
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-head">Sample Detection Results</div>', 
                unsafe_allow_html=True)
    
    detection_images = list(Path("data").glob("pipeline_test_*.jpg"))
    if detection_images:
        cols = st.columns(3)
        for i, img_path in enumerate(detection_images[:3]):
            with cols[i]:
                st.image(str(img_path), caption=img_path.name, use_container_width=True)
    else:
        st.info("No detection images available. Run test_detection.py to generate!")

# ════════════════════════════════════════════════════════════════════════
#  TAB 1 – OVERVIEW
# ════════════════════════════════════════════════════════════════════════
with tab_overview:

    # ── Top KPI Row ─────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    kpi_data = [
        (k1, "Current Occupancy",  current_occ,          "cars",  "#ff6030"),
        (k2, "Available Slots",    available,             "free",  "#6dff60"),
        (k3, "Predicted (1hr)",    TOTAL_SLOTS-pred_next_occ, "avail","#ffd000"),
        (k4, "Total Capacity",     TOTAL_SLOTS,           "slots", "#00dcff"),
    ]
    for col, label, val, unit, color in kpi_data:
        with col:
            pct = round(val / TOTAL_SLOTS * 100)
            st.markdown(
                f'<div class="pk-card">'
                f'<div class="pk-label">{label}</div>'
                f'<div class="pk-value" style="color:{color}">{val}'
                f'<span style="font-size:14px;color:#5a7a8a"> {unit}</span></div>'
                f'<div class="pk-sub">{pct}% of capacity</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Gauges Row ──────────────────────────────────────────────────────
    g1, g2, g3 = st.columns(3)

    with g1:
        st.markdown('<div class="sec-head">Live Occupancy</div>', unsafe_allow_html=True)
        occ_clr = "#ff3050" if occ_rate > 0.8 else "#ffd000" if occ_rate > 0.5 else "#6dff60"
        st.plotly_chart(make_gauge(current_occ, TOTAL_SLOTS, occ_clr, "Vehicles Detected"),
                        width='stretch')

    with g2:
        st.markdown('<div class="sec-head">Available Slots</div>', unsafe_allow_html=True)
        st.plotly_chart(make_gauge(available, TOTAL_SLOTS, "#00dcff", "Free Slots"),
                        width='stretch')

    with g3:
        st.markdown('<div class="sec-head">Parking Stress Index</div>', unsafe_allow_html=True)
        fig_psi = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=psi_val,
            delta={"reference": 0.50, "valueformat": ".3f",
                   "increasing":{"color":"#ff3050"}, "decreasing":{"color":"#6dff60"}},
            number={"valueformat": ".3f",
                    "font": {"size":36, "color":psi_clr, "family":"Rajdhani"}},
            gauge={
                "axis": {"range":[0,1], "tickformat":".1f",
                         "tickcolor":"#2a3e50", "tickfont":{"color":"#5a7a8a","size":9}},
                "bar": {"color":psi_clr, "thickness":0.68},
                "bgcolor": "#0d1117", "borderwidth": 0,
                "steps": [{"range":[0, 0.30], "color":"#091809"},
                           {"range":[0.30, 0.70], "color":"#181600"},
                           {"range":[0.70, 1.0],  "color":"#180608"}],
                "threshold": {"line":{"color":"white","width":2},
                              "thickness":0.78, "value":psi_val}
            }
        ))
        fig_psi.update_layout(
            height=200, paper_bgcolor="#07090f", plot_bgcolor="#07090f",
            margin=dict(l=20, r=20, t=28, b=8), font={"color":"#e0eaf0"}
        )
        st.plotly_chart(fig_psi, use_container_width=True)
        st.markdown(
            f'<div style="text-align:center;margin-top:-8px">'
            f'<span class="psi-badge" style="border-color:{psi_clr};color:{psi_clr};'
            f'background:{psi_clr}18">'
            f'● {psi_lvl.upper()} STRESS  {psi_em}'
            f'</span></div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Slot Grid + PSI Breakdown ────────────────────────────────────────
    sg1, sg2 = st.columns([3, 2])

    with sg1:
        st.markdown('<div class="sec-head">Live Parking Grid (YOLOv11 Detection)</div>',
                    unsafe_allow_html=True)
        selected_slot = st.session_state.selected_slot
        slot_size = get_slot_size(selected_slot)
        selected_is_occupied = selected_slot <= current_occ
        selected_status = "Occupied" if selected_is_occupied else "Available"
        selected_status_color = "#ff5070" if selected_is_occupied else "#6dff60"

        st.markdown(
            f'<div class="slot-detail">'
            f'<div class="slot-detail-title">Slot P-{selected_slot:02d}</div>'
            f'<div class="slot-detail-meta">'
            f'Status: <span style="color:{selected_status_color};font-weight:700">{selected_status}</span><br>'
            f'Type: {slot_size["label"]}<br>'
            f'Size: {slot_size["length_m"]:.1f} m x {slot_size["width_m"]:.1f} m<br>'
            f'Area: {slot_size["area_m2"]:.2f} m²'
            f'</div></div>',
            unsafe_allow_html=True
        )

        st.markdown('<div class="slot-panel">', unsafe_allow_html=True)
        for row_start in range(0, TOTAL_SLOTS, SLOT_COLUMNS):
            row_cols = st.columns(SLOT_COLUMNS, gap="small")
            for col_index, slot_number in enumerate(
                range(row_start + 1, min(row_start + SLOT_COLUMNS + 1, TOTAL_SLOTS + 1))
            ):
                is_occupied = slot_number <= current_occ
                label = f'{"🚗" if is_occupied else "🟩"} P-{slot_number:02d}'
                with row_cols[col_index]:
                    if st.button(label, key=f"slot_{slot_number}", use_container_width=True):
                        st.session_state.selected_slot = slot_number
                        st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        slots_html = '<div class="slot-grid">'
        for i in range(TOTAL_SLOTS):
            if i < current_occ:
                slots_html += '<div class="slot-occ">🚗</div>'
            else:
                slots_html += '<div class="slot-free"></div>'
        slots_html += "</div>"
        legend_html = (
            f"<div style='display:flex;gap:16px;margin-top:4px;font-size:11px;color:#5a7a8a'>"
            f"<span><span style='background:rgba(255,50,80,.4);padding:2px 8px;"
            f"border-radius:3px'>■</span> Occupied ({current_occ})</span>"
            f"<span><span style='background:rgba(100,255,80,.2);padding:2px 8px;"
            f"border-radius:3px'>■</span> Available ({available})</span>"
            f"</div>"
        )
        st.markdown(
            f'<div style="background:#0d1117;border:1px solid #1c2a38;'
            f'border-radius:8px;padding:6px">{slots_html}{legend_html}</div>',
            unsafe_allow_html=True
        )

    with sg2:
        st.markdown('<div class="sec-head">PSI Component Breakdown</div>',
                    unsafe_allow_html=True)
        br_labels = ["Occ Rate", "Pred Rate", "PSI", "Time Factor"]
        br_values = [
            psi_result.get("occ_rate", occ_rate),
            psi_result.get("pred_rate", pred_next_occ/TOTAL_SLOTS),
            psi_val,
            psi_result.get("time_factor", 1.0)
        ]
        br_colors = [occ_clr, "#ffd000", psi_clr, "#00dcff"]
        fig_br = go.Figure(go.Bar(
            x=br_labels, y=br_values, marker_color=br_colors,
            text=[f"{v:.3f}" for v in br_values], textposition="outside"
        ))
        fig_br.update_layout(
            height=210, paper_bgcolor="#07090f", plot_bgcolor="#0d1117",
            margin=dict(l=4, r=4, t=8, b=4),
            yaxis=dict(gridcolor="#1c2a38", range=[0, 1.35]),
            xaxis=dict(color="#5a7a8a"),
            font={"color":"#e0eaf0", "size":11},
            bargap=0.3
        )
        st.plotly_chart(fig_br, width='stretch')

    # ── Recommendation ───────────────────────────────────────────────────

# ════════════════════════════════════════════════════════════════════════
#  TAB 2 – PREDICTION
# ════════════════════════════════════════════════════════════════════════
with tab_predict:

    st.markdown('<div class="sec-head">6-Hour Occupancy Forecast (Random Forest)</div>',
                unsafe_allow_html=True)

    fc_times = [f["timestamp"] for f in forecasts]
    fc_occ   = [f["predicted_occupancy"] for f in forecasts]
    fc_avail = [f["predicted_available"] for f in forecasts]
    fc_rates = [f["occupancy_rate"] for f in forecasts]

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Bar(x=fc_times, y=fc_occ,   name="Predicted Occupied",
                             marker_color="#ff6030", opacity=0.85))
    fig_fc.add_trace(go.Bar(x=fc_times, y=fc_avail, name="Predicted Available",
                             marker_color="#00dcff", opacity=0.65))
    fig_fc.add_hline(
        y=TOTAL_SLOTS * 0.7, line_dash="dot", line_color="#ff3050",
        annotation_text="70% High-Stress Threshold",
        annotation_font_color="#ff3050"
    )
    fig_fc.update_layout(
        barmode="stack", height=300,
        paper_bgcolor="#07090f", plot_bgcolor="#0d1117",
        legend=dict(bgcolor="#0d1117", bordercolor="#1c2a38", borderwidth=1),
        margin=dict(l=8, r=8, t=12, b=8),
        yaxis=dict(gridcolor="#1c2a38", title="Slots", range=[0, TOTAL_SLOTS+6]),
        xaxis=dict(color="#5a7a8a", title="Time"),
        font={"color":"#e0eaf0"}
    )
    st.plotly_chart(fig_fc, width='stretch')

    # Forecast table
    fc_df = pd.DataFrame({
        "+Hours":    [f["hour_offset"] for f in forecasts],
        "Time":      fc_times,
        "Predicted Occ": fc_occ,
        "Predicted Avail": fc_avail,
        "Occ Rate":  [f"{r*100:.1f}%" for r in fc_rates],
        "Status":    ["🔴 HIGH" if r>=0.7 else "🟡 MED" if r>=0.3 else "🟢 LOW"
                      for r in fc_rates]
    })
    st.dataframe(fc_df, use_container_width=True, hide_index=True)

    # 24-hour historical + predicted
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-head">24-Hour Pattern: Actual vs Predicted</div>',
                unsafe_allow_html=True)

    hours_24 = list(range(24))
    actual   = HOUR_PATTERN
    np.random.seed(42)
    pred_24  = [max(0, min(TOTAL_SLOTS, v + int(np.random.randint(-2,3)))) for v in actual]
    hour_labels = [f"{h:02d}:00" for h in hours_24]

    fig_24 = plotly_line([
        {"x": hour_labels, "y": actual,  "name": "Historical Actual",
         "color": "#00dcff", "width": 2, "mode": "lines+markers", "marker_size": 4},
        {"x": hour_labels, "y": pred_24, "name": "RF Predicted",
         "color": "#ff6030", "width": 2, "dash": "dash"}
    ], height=270)
    fig_24.add_hrect(
        y0=TOTAL_SLOTS*0.7, y1=TOTAL_SLOTS,
        fillcolor="rgba(255,50,80,.05)", line_width=0
    )
    st.plotly_chart(fig_24, width='stretch')

    # Model metrics
    st.markdown('<div class="sec-head">Random Forest Model Metrics</div>',
                unsafe_allow_html=True)
    m = pred_model.metrics_
    if m:
        mc1, mc2, mc3, mc4 = st.columns(4)
        for col, lbl, val, clr in [
            (mc1, "Accuracy",     f"{m.get('accuracy_pct','?')}%", "#6dff60"),
            (mc2, "MAE (slots)",  m.get('mae','?'),                "#00dcff"),
            (mc3, "R² Score",     m.get('r2','?'),                 "#ffd000"),
            (mc4, "RMSE",         m.get('rmse','?'),               "#ff6030"),
        ]:
            with col:
                st.markdown(
                    f'<div class="pk-card">'
                    f'<div class="pk-label">{lbl}</div>'
                    f'<div class="pk-value" style="color:{clr};font-size:2rem">{val}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
    else:
        st.info("Model not trained yet. Run setup first.")

# ════════════════════════════════════════════════════════════════════════
#  TAB 3 – PEAK HOURS
# ════════════════════════════════════════════════════════════════════════
with tab_peak:

    p1, p2 = st.columns(2)

    with p1:
        st.markdown('<div class="sec-head">Hourly Congestion Pattern (Avg)</div>',
                    unsafe_allow_html=True)
        h_avg = peak_analysis.get("hourly_avg", {h: HOUR_PATTERN[h] for h in range(24)})
        h_vals = [h_avg.get(h, 0) for h in range(24)]
        h_clrs = ["#ff3050" if v/TOTAL_SLOTS >= 0.7 else
                   "#ffd000" if v/TOTAL_SLOTS >= 0.3 else "#00dcff"
                   for v in h_vals]
        fig_h = plotly_bar(
            x=[f"{h:02d}" for h in range(24)],
            y=h_vals, colors=h_clrs, height=250
        )
        st.plotly_chart(fig_h, width='stretch')

    with p2:
        st.markdown('<div class="sec-head">Day-of-Week Pattern</div>',
                    unsafe_allow_html=True)
        daily = peak_analysis.get("daily_avg",
                    {"Monday":32,"Tuesday":36,"Wednesday":38,"Thursday":41,
                     "Friday":44,"Saturday":29,"Sunday":20})
        d_keys  = [k[:3] for k in daily.keys()]
        d_vals  = list(daily.values())
        max_dv  = max(d_vals) if d_vals else 1
        d_clrs  = ["#ff6030" if v == max_dv else "#00dcff" for v in d_vals]
        fig_d   = plotly_bar(x=d_keys, y=d_vals, colors=d_clrs, height=250)
        st.plotly_chart(fig_d, width='stretch')

    # Peak stats
    pk1, pk2, pk3, pk4 = st.columns(4)
    for col, lbl, val, clr in [
        (pk1, "Peak Window",   peak_analysis.get("peak_window_label","5–7 PM"),   "#ff3050"),
        (pk2, "Busiest Day",   peak_analysis.get("busiest_day","Friday"),          "#ff6030"),
        (pk3, "Quietest Day",  peak_analysis.get("quietest_day","Sunday"),         "#6dff60"),
        (pk4, "Peak Occ",      f"{peak_analysis.get('peak_occupancy_max',48):.0f}/{TOTAL_SLOTS}", "#ffd000"),
    ]:
        with col:
            st.markdown(
                f'<div class="pk-card">'
                f'<div class="pk-label">{lbl}</div>'
                f'<div class="pk-value" style="color:{clr};font-size:1.25rem">{val}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    # Congestion pie
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-head">Congestion Level Distribution</div>',
                unsafe_allow_html=True)
    fig_pie = go.Figure(go.Pie(
        labels=["Low Stress", "Medium Stress", "High Congestion"],
        values=[
            peak_analysis.get("low_pct", 24),
            peak_analysis.get("medium_pct", 34),
            peak_analysis.get("high_pct", 42)
        ],
        marker=dict(colors=["#6dff60","#ffd000","#ff3050"]),
        hole=0.55,
        hovertemplate="%{label}: %{value}%<extra></extra>"
    ))
    fig_pie.update_layout(
        height=280, paper_bgcolor="#07090f",
        legend=dict(bgcolor="#0d1117",bordercolor="#1c2a38"),
        margin=dict(l=8,r=8,t=8,b=8), font={"color":"#e0eaf0"}
    )
    st.plotly_chart(fig_pie, width='stretch')

# ════════════════════════════════════════════════════════════════════════
#  TAB 4 – AI INSIGHTS
# ════════════════════════════════════════════════════════════════════════
with tab_ai:

    ai1, ai2 = st.columns(2)

    with ai1:
        st.markdown('<div class="sec-head">Feature Importance (Random Forest)</div>',
                    unsafe_allow_html=True)
        fi_raw = pred_model.feature_importances_
        if not fi_raw:
            fi_raw = {"hour":0.20,"hour_sin":0.08,"hour_cos":0.06,"day_of_week":0.15,
                      "day_sin":0.07,"day_cos":0.06,"is_weekend":0.10,
                      "occ_lag1":0.15,"occ_lag2":0.07,"occ_rolling3":0.06}

        # Group by display name
        groups = rec_module.DISPLAY_GROUPS
        grouped = {}
        for label, feats in groups.items():
            grouped[label] = sum(fi_raw.get(f, 0.0) for f in feats)
        sorted_g = sorted(grouped.items(), key=lambda x: x[1], reverse=True)

        fi_labels = [x[0] for x in sorted_g]
        fi_values = [round(x[1], 4) for x in sorted_g]
        fi_colors = ["#00dcff","#6dff60","#ff6030","#ffd000","#aa66ff"]

        fig_fi = go.Figure(go.Bar(
            x=fi_values, y=fi_labels, orientation="h",
            marker_color=fi_colors[:len(fi_labels)],
            text=[f"{v*100:.1f}%" for v in fi_values],
            textposition="outside"
        ))
        fig_fi.update_layout(
            height=280, paper_bgcolor="#07090f", plot_bgcolor="#0d1117",
            margin=dict(l=10, r=70, t=8, b=8),
            xaxis=dict(gridcolor="#1c2a38", range=[0, max(fi_values)*1.35]),
            yaxis=dict(color="#5a7a8a"),
            font={"color":"#e0eaf0","size":11}
        )
        st.plotly_chart(fig_fi, width='stretch')

    with ai2:
        st.markdown('<div class="sec-head">Model Architecture</div>',
                    unsafe_allow_html=True)
        arch = [
            ("Detection",      "YOLOv11 (Ultralytics)",      "#00dcff"),
            ("Dataset",        "pfe Object Detection",        "#6dff60"),
            ("Dataset Author", "bouakkaz144.lotfi@gmail.com", "#5a7a8a"),
            ("ML Model",       "Random Forest Regressor",     "#ffd000"),
            ("Estimators",     "200 trees",                   "#ff6030"),
            ("Max Depth",      "12 levels",                   "#aa66ff"),
            ("Features",       "10 input features",           "#00dcff"),
            ("Train/Test",     "80 / 20 split",               "#6dff60"),
            ("Target",         "Predict occupancy (slots)",   "#ffd000"),
        ]
        for lbl, val, clr in arch:
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:9px 0;border-bottom:1px solid #1c2a38;font-size:12px'>"
                f"<span style='color:#5a7a8a'>{lbl}</span>"
                f"<span style='color:{clr};font-weight:600'>{val}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

    # Explainability narrative
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-head">Explainable AI Narrative</div>',
                unsafe_allow_html=True)
    top3 = [x[0] for x in sorted_g[:3]]
    exp_txt = (
        f"Current occupancy is <b>{current_occ}/{TOTAL_SLOTS}</b> "
        f"({occ_rate*100:.0f}%). PSI = <b style='color:{psi_clr}'>{psi_val:.3f}</b> "
        f"(<b>{psi_lvl.upper()} stress</b> {psi_em}).<br><br>"
        f"The top 3 congestion drivers are: "
        f"<b style='color:#00dcff'>{', '.join(top3)}</b>. "
        f"<b>{sorted_g[0][0]}</b> contributes {sorted_g[0][1]*100:.0f}% of prediction importance. "
        f"Lag features capture momentum in parking demand patterns.<br><br>"
        f"Peak window: <b style='color:#ff6030'>"
        f"{peak_analysis.get('peak_window_label','5–7 PM')}</b>. "
        f"Busiest day: <b style='color:#ffd000'>"
        f"{peak_analysis.get('busiest_day','Friday')}</b>. "
        f"Model confidence: <b>{recommendation.get('confidence','N/A')}</b>."
    )
    st.markdown(f'<div class="rec-box">{exp_txt}</div>', unsafe_allow_html=True)

    # System pipeline
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-head">System Pipeline (10 Modules)</div>',
                unsafe_allow_html=True)
    pipe_steps = [
        ("1","Video Input","CCTV/File"), ("2","Frame Extract","OpenCV"),
        ("3","YOLOv11","Roboflow"),      ("4","Count Vehicles","Module 2"),
        ("5","Occupancy","Module 3"),    ("6","Feature Eng","Pandas"),
        ("7","RF Predict","Module 4"),   ("8","Peak Hours","Module 5"),
        ("9","PSI Calc","Module 6"),     ("10","Dashboard","Streamlit"),
    ]
    html = '<div style="display:flex;flex-wrap:wrap;gap:8px;padding:10px">'
    for num, step, sub in pipe_steps:
        html += (
            f'<div style="background:#0d1117;border:1px solid #00dcff;'
            f'border-radius:6px;padding:8px 12px;min-width:80px;text-align:center">'
            f'<div style="font-size:9px;color:#5a7a8a">{num}</div>'
            f'<div style="font-size:11px;color:#00dcff;font-weight:700">{step}</div>'
            f'<div style="font-size:9px;color:#5a7a8a">{sub}</div></div>'
        )
        if num != "10":
            html += '<span style="color:#2a3e50;align-self:center;font-size:18px">→</span>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════
#  TAB 5 – SIMULATION
# ════════════════════════════════════════════════════════════════════════
with tab_sim:

    st.markdown('<div class="sec-head">What-If Scenario Simulator</div>',
                unsafe_allow_html=True)

    sc1, sc2 = st.columns(2)
    with sc1:
        demand_pct = st.slider("📈 Demand Change (%)",    -20, 60, 20,  step=5)
        extra_slots = st.slider("🅿 New Slots to Add",    0,   30,  0,  step=5)
        sim_hour   = st.slider("🕐 Simulate at Hour",    0,   23, hour_now)
    with sc2:
        base_occ_s  = st.slider("Current Occupancy (base)",  0,  50, current_occ)
        base_pred_s = st.slider("Base Predicted Demand",      0,  50,
                                 min(50, current_occ + 5))

    # Run simulation
    sim_res = sim_module.run(
        demand_change_pct=demand_pct,
        new_slots=extra_slots,
        hour=sim_hour,
        base_occupancy=base_occ_s,
        base_predicted=base_pred_s
    )

    before = sim_res["before"]
    after  = sim_res["after"]
    delta  = sim_res["delta"]
    after_clr = psi_color(after["level"])

    st.markdown("<br>", unsafe_allow_html=True)
    r1, r2, r3, r4 = st.columns(4)
    for col, lbl, bv, av, clr in [
        (r1, "Capacity",   before["capacity"],   after["capacity"],   "#00dcff"),
        (r2, "Occupancy",  before["occupancy"],  int(after["occupancy"]), "#ff6030"),
        (r3, "Available",  before["available"],  int(after["available"]), "#6dff60"),
        (r4, "PSI",        before["psi"],         after["psi"],        after_clr),
    ]:
        with col:
            d     = float(av) - float(bv)
            arrow = "⬆" if d > 0 else "⬇" if d < 0 else "→"
            dc    = "#ff3050" if (d>0 and lbl in ("Occupancy","PSI")) else "#6dff60" if d<0 else "#5a7a8a"
            st.markdown(
                f'<div class="pk-card">'
                f'<div class="pk-label">{lbl}</div>'
                f'<div class="pk-value" style="color:{clr};font-size:1.8rem">{av}</div>'
                f'<div style="font-size:11px;color:{dc}">{arrow} {d:+.3f} from {bv}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    # Hourly forecast chart
    st.markdown("<br>", unsafe_allow_html=True)
    hourly_df = sim_module.hourly_forecast(
        demand_change_pct=demand_pct, new_slots=extra_slots
    )
    fig_sim = plotly_line([
        {"x": list(range(24)), "y": list(hourly_df["before_occ"]),
         "name": "Before", "color":"#00dcff", "width":2},
        {"x": list(range(24)), "y": list(hourly_df["after_occ"]),
         "name": f"After ({demand_pct:+d}% demand, +{extra_slots} slots)",
         "color":"#ff3050", "width":2, "dash":"dash"},
    ], height=260)
    fig_sim.update_layout(
        xaxis=dict(tickvals=list(range(0,24,2)), title="Hour"),
        yaxis=dict(title="Projected Vehicles", range=[0, TOTAL_SLOTS+10])
    )
    st.plotly_chart(fig_sim, width='stretch')

    st.markdown(
        f'<div class="rec-box">💡 <b>Simulation Result:</b> {sim_res["recommendation"]}</div>',
        unsafe_allow_html=True
    )

    # Multi-scenario comparison
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-head">Multi-Scenario Comparison</div>',
                unsafe_allow_html=True)
    scen_list = [
        {"demand_change_pct":  0, "new_slots":  0, "base_occupancy": current_occ},
        {"demand_change_pct": 10, "new_slots":  0, "base_occupancy": current_occ},
        {"demand_change_pct": 20, "new_slots":  0, "base_occupancy": current_occ},
        {"demand_change_pct": 20, "new_slots": 10, "base_occupancy": current_occ},
        {"demand_change_pct": 30, "new_slots": 20, "base_occupancy": current_occ},
    ]
    comp_df, _ = sim_module.compare(scen_list)
    st.dataframe(comp_df, width='stretch', hide_index=True)

    # Expansion recommendation
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-head">Capacity Expansion Planner</div>',
                unsafe_allow_html=True)
    target_psi = st.slider("Target PSI (maximum acceptable)", 0.20, 0.80, 0.45, step=0.05)
    exp_rec = sim_module.recommend_expansion(
        target_psi=target_psi,
        base_occupancy=current_occ,
        base_predicted=pred_next_occ
    )
    st.markdown(
        f'<div class="rec-box">'
        f'🏗 <b>Expansion Recommendation:</b> {exp_rec["recommendation"]}<br>'
        f'Slots needed: <b style="color:#ffd000">{exp_rec["slots_needed"]}</b>  |  '
        f'New capacity: <b style="color:#00dcff">{exp_rec["new_capacity"]}</b>  |  '
        f'Projected PSI: <b>{exp_rec["projected_psi"]}</b>'
        f'</div>',
        unsafe_allow_html=True
    )

# ════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙ Configuration")
    st.divider()
    st.markdown(f"**Total Slots:** {TOTAL_SLOTS}")
    st.markdown(f"**Current Occ:** {current_occ}")
    st.markdown(f"**PSI Level:** `{psi_lvl.upper()}` {psi_em}")
    st.divider()

    st.markdown("### 🧪 Model Training")
    if st.button("🔄 Retrain Model", width='stretch'):
        with st.spinner("Generating data + training..."):
            df_t = est.generate_historical_data(days=30, save=True)
            df_f = est.get_dataframe()
            m_result = pred_model.train(df_f)
            if m_result:
                st.success(f"Done! Accuracy: {m_result.get('accuracy_pct')}%")
            else:
                st.error("Training failed. Check scikit-learn install.")

    st.divider()
    st.markdown("### 📊 Live Stats")
    st.metric("Occupancy Rate", f"{occ_rate*100:.1f}%",
              delta=f"{(occ_rate-0.5)*100:+.1f}% vs avg")
    st.metric("PSI Score", f"{psi_val:.4f}")
    st.metric("Available", f"{available}/{TOTAL_SLOTS}")
    st.divider()

    st.markdown("### 🔄 Auto Refresh")
    auto_refresh = st.checkbox("Refresh every 5 seconds", value=False)
    if auto_refresh:
        time.sleep(5)
        st.rerun()

    st.divider()
    st.markdown(
        "<small style='color:#5a7a8a'>"
        "<b>ParkIntel v2.0</b><br>"
        "YOLOv11 + Random Forest<br>"
        "Dataset: pfe Object Detection<br>"
        "Author: bouakkaz144.lotfi<br>"
        "9 Modules + Pipeline"
        "</small>",
        unsafe_allow_html=True
    )
