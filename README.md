# Vendor-Agnostic RAN Optimization Using Generative AI 



# Details

* Problem Statement

Modern multi-vendor RANs generate complex, dynamic/inconsistent data that makes manual optimization time-consuming and error-prone. Traditional rule-based solutions can't adapt to changing network conditions or vendor diversity. There's a need for a sophisticated, vendor-agnostic solution that can reason on KPIs, identify anomalies, and autonomously suggest optimizations.  This project proposes a Generative AI–driven reasoning framework that learns from real telemetry to deliver explainable, adaptive RAN performance improvements.

*  What interesting technology did we use?

- **Dataset simulation:** Generate synthetic KPI data using configurable parameters.
- **File upload support:** Accepts `.json` datasets for analysis.
- **Optimization agent:** Runs an AWS Bedrock-powered reasoning model (Anthropic Claude 3.5) via `agent_bedrock.py`.
- **Interactive frontend:** Built with Streamlit for an intuitive workflow.
- **FastAPI backend:** Handles dataset generation, file uploads, and agent orchestration.
- **Flexible input sources:** Users can choose between API simulation or file upload for logs

* File Structure
NullPointersHackathon
|--bedrock_test
  |--agent_bedrock.py
  |--api_simulator.py
|--Frontend
  |--app.py


# Set Up Instructions
1. Clone the repository
2. Move into the repository folder:
`cd NullPointersHackathon`
3. Setup Virtual Environment:  `python -m venv venv`
4. Activate: 
`venv\Scripts\activate`
5. 
`pip install --upgrade pip`
`pip install -r requirements.txt`
6. Start the FastAPI backend

`uvicorn api_simulator:app --host 0.0.0.0 --port 8000`

7. Start the Streamlit frontend
`streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0`

8. Use the application
    * Open the Streamlit UI in your browser
    * Choose Data Source:
      - Generate dataset: Create synthetic KPI data
      - Upload File: Upload .json dataset
    * Click on the Run Optimization

