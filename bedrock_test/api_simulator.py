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
ALLOWED_EXTENSIONS = {".csv", ".json", ".txt"}


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

    return {"filename": file.filename, "agent_result": result}
    
    

@app.post("/simulate")
def run_agent_on_simulated_data():
    """
    Simulate real-time KPI data and call Claude model for AI reasoning and optimization
    """
    kpi_json = load_kpi_data(source="api")
    suggestion = analyze_with_bedrock(kpi_json)
    result = simulate_optimization(kpi_json, suggestion)
    return {"agent_result": result}



@app.get("/")
def root():
    return {"message": "API is running!"}
