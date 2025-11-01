from fastapi import FastAPI
import random
import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Realistic RAN Data Generator API")

# Allow requests from Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Max file size in bytes (1 MB)
MAX_FILE_SIZE = 1 * 1024 * 1024

# Supported file extensions
ALLOWED_EXTENSIONS = {".csv", ".json", ".txt"}

ttt_choices = [160, 320, 480, 640]
slice_types = ["eMBB", "URLLC", "mMTC"]
handover_failure_causes = ["LowRSRP", "TargetBusy", "Other"]

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # Check file extension
    ext = file.filename.lower().rsplit(".", 1)[-1]
    if f".{ext}" not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Supported formats: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Read content
    content = await file.read()
    
    # Check file size
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum allowed size is 1 MB."
        )
    
    # Need to integrate Bedrock agent / RAN optimizer 
    result = {
        "filename": file.filename,
        "status": "processed",
        "data_length": len(content)
    }
    return result

def generate_dataset(num_cells: int = 5):
    data = {"cells": []}

    for i in range(1, num_cells + 1):
        # Original features
        rsrp = round(random.uniform(-115, -70), 1)
        rsrq = round(random.uniform(-20, -5), 1)
        sinr = round(random.uniform(max(-5, (rsrp + 120) / 2), 30), 1)
        load = round(random.uniform(0.3, 0.95), 2)
        hof = min(8, max(0, int((load - 0.6) * 20 + random.gauss(2, 1))))
        rlf = min(5, max(0, int((-rsrp - 90) / 10 + random.gauss(1, 0.5))))
        a3_hyst = random.randint(1, 4)
        ttt = random.choice(ttt_choices)

        # New additional features
        cell_load = round(random.uniform(10, 100), 2)  # in percentage
        active_ues = random.randint(1, 100)
        prb_utilization = round(random.uniform(10, 100), 2)
        avg_throughput_dl = round(random.uniform(1, 500), 2)
        avg_throughput_ul = round(random.uniform(1, 100), 2)
        cqi = random.randint(1, 15)
        handover_success_rate = round(random.uniform(80, 100), 2)
        avg_time_between_handover = round(random.uniform(5, 120), 2)
        handover_failure_cause = random.choice(handover_failure_causes)
        rsrp_delta = round(random.uniform(-5, 5), 2)
        sinr_trend = random.choice(["Increasing", "Decreasing", "Stable"])
        neighbor_count = random.randint(1, 6)
        strong_neighbor_count = random.randint(0, neighbor_count)
        avg_neighbor_rsrp = round(random.uniform(-110, -70), 2)
        slice_type = random.choice(slice_types)
        slice_load = round(random.uniform(0, 100), 2)
        qos_violation_count = random.randint(0, 5)
        timestamp = datetime.datetime.now().isoformat()

        cell = {
            "cell_id": f"C{i:03d}",
            "rsrq": rsrq,
            "rsrp": rsrp,
            "sinr": sinr,
            "hof": hof,
            "rlf": rlf,
            "load": load,
            "a3_hyst": a3_hyst,
            "ttt": ttt,
            "cell_load": cell_load,
            "active_ues": active_ues,
            "prb_utilization": prb_utilization,
            "avg_throughput_dl": avg_throughput_dl,
            "avg_throughput_ul": avg_throughput_ul,
            "cqi": cqi,
            "handover_success_rate": handover_success_rate,
            "avg_time_between_handover": avg_time_between_handover,
            "handover_failure_cause": handover_failure_cause,
            "rsrp_delta": rsrp_delta,
            "sinr_trend": sinr_trend,
            "neighbor_count": neighbor_count,
            "strong_neighbor_count": strong_neighbor_count,
            "avg_neighbor_rsrp": avg_neighbor_rsrp,
            "slice_type": slice_type,
            "slice_load": slice_load,
            "qos_violation_count": qos_violation_count,
            "timestamp": timestamp
        }

        data["cells"].append(cell)

    return data

@app.get("/")
def root():
    return {"message": "API is running!"}

@app.get("/generate_dataset")
def get_dataset(num_cells: int = 5):
    return generate_dataset(num_cells)