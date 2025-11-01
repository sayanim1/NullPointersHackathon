import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

# --- Team and intro ---
st.markdown("<h1 style='color:#4CAF50;'>Team: NullPointers</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size:16px;'>I am <b>Cookie</b>, your RAN optimizer! Upload a dataset or use the API endpoint to get optimization suggestions.</p>", unsafe_allow_html=True)

# --- Upload vs API selection ---
option = st.radio("Choose how to test:", ("Upload dataset", "Use API endpoint"))

# Live output area
st.subheader("Live model output")
live_output = st.empty()

# --- Handle upload ---
if option == "Upload dataset":
    uploaded_file = st.file_uploader("Upload CSV/JSON/TXT", type=["csv", "json", "txt"])
    if uploaded_file:
        # Check file size before sending to API
        uploaded_file.seek(0, 2)  
        size = uploaded_file.tell()
        uploaded_file.seek(0)  

        if size > 1 * 1024 * 1024:
            st.error("File too large. Maximum allowed size is 1 MB.")
        else:
            files = {"file": (uploaded_file.name, uploaded_file, "application/octet-stream")}
            try:
                response = requests.post(f"{API_URL}/upload", files=files)
                response.raise_for_status()
                live_output.success("File uploaded and validated successfully!")
                live_output.json(response.json())
            except requests.exceptions.HTTPError:
                error_detail = response.json().get("detail", "Unknown error")
                live_output.error(f"Error: {error_detail}")

# --- Handle API endpoint ---
else:
    try:
        response = requests.get(f"{API_URL}/generate_dataset")
        response.raise_for_status()
        live_output.success("API endpoint reachable!")
        live_output.json(response.json())
    except requests.exceptions.RequestException as e:
        live_output.error(f"API request failed: {str(e)}")
