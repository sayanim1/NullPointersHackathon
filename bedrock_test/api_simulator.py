import random
from fastapi import FastAPI
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from agent_bedrock import load_kpi_data, analyze_with_bedrock, simulate_optimization

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
ALLOWED_EXTENSIONS = {".json"}


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
    
    temp_file_path = f"data/{file.filename}"
    with open(temp_file_path, "wb") as f:
        f.write(content)

    # Run agent workflow on uploaded file
    kpi_data = load_kpi_data(source="file", file_path=temp_file_path)
    suggestion = analyze_with_bedrock(kpi_data)
    result = simulate_optimization(kpi_data, suggestion)

    return {"filename": file.filename, "simulation_optimization": result, "suggestion": suggestion}
    
    

@app.post("/simulate")
def run_agent_on_simulated_data():
    """
    Simulate real-time KPI data and call Claude model for AI reasoning and optimization
    """
    kpi_json = load_kpi_data(source="api")
    suggestion = analyze_with_bedrock(kpi_json)
    result = simulate_optimization(kpi_json, suggestion)
    return {"simulation_optimization": result, "suggestion": suggestion}

def synth_labeling_rules(latency, packet_loss, throughput, num_users, signal_quality, cell_load):
    
    # target bandwidth
    target_bandwidth = min(100.0, max(10.0, throughput/2.0 +num_users/20.0 + random.gauss(0, 2)))


    # power adjustment
    power_adjustment = float(max(-3.0, min(6.0, (-signal_quality)/3.0 + random.gauss(0, 0.5))))

    # handover_threshold
    handover_threshold = float(max(30.0, min(95.0, 50.0 + latency/4.0 - signal_quality/2.0 + random.gauss(0, 2))))

    # slice_priority(categorical) - higher for highly loaded cells
    if cell_load<50:
        slice_priority = 1
    elif cell_load<80:
        slice_priority = 2
    else:
        slice_priority = 3

    return {
        "target_bandwidth": round(target_bandwidth, 2),
        "power_adjustment": round(power_adjustment, 2),
        "handover_threshold": round(handover_threshold, 2),
        "slice_priority": int(slice_priority)
    }

@app.get("/generate_labeled_dataset")
def generate_labeled_dataset(num_samples: int = 1000, save_csv:  bool = True):
    rows = []
    for i in range(num_samples):
        latency = round(random.uniform(5, 250), 2)
        packet_loss = round(random.uniform(0, 6), 3)
        throughput = round(random.uniform(1, 500), 2)
        num_users = random.randint(1, 500)
        signal_quality = round(random.uniform(-20, 5), 2)
        cell_load = round(random.uniform(5, 100), 2)

@app.get("/")
def root():
    return {"message": "API is running!"}
