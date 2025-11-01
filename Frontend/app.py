import streamlit as st
import requests
import json
import pandas as pd

API_URL = "http://127.0.0.1:8000"  # Change this to EC2 public DNS when deployed

st.set_page_config(page_title="NullPointers – RAN Optimizer", layout="wide")

st.title("NullPointers: Vendor-Agnostic RAN Optimization Using Generative AI")
st.write("Upload a dataset or simulate data to get optimization insights powered by **Amazon Bedrock**.")

# Choose mode
option = st.radio("Select Mode:", ["Upload Dataset", "Simulate Data"])

def display_result(data):
    """Display only the 'after' optimization KPI data."""
    
    if "simulation_optimization" in data:
        sim_data = data["simulation_optimization"]
        after_cells = sim_data.get("after", {}).get("cells", [])
        
        if after_cells:
            st.subheader("After Optimization")
            after_df = pd.DataFrame(after_cells)
            st.dataframe(after_df, use_container_width=True)
        else:
            st.info("No 'after' optimization data available.")
    else:
        st.info("Simulation optimization data not found in response.")

# Upload Dataset Mode
if option == "Upload Dataset":
    uploaded_file = st.file_uploader("Upload JSON", type=["json"])

    if uploaded_file:
        uploaded_file.seek(0, 2)
        size = uploaded_file.tell()
        uploaded_file.seek(0)

        if size > 1 * 1024 * 1024:
            st.error("File too large. Maximum allowed size is 1 MB.")
        else:
            if st.button("🚀 Run Optimization"):
                with st.spinner("Running Bedrock optimization..."):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file, "application/octet-stream")}
                        response = requests.post(f"{API_URL}/upload", files=files)
                        response.raise_for_status()
                        data = response.json()
                        st.success("Optimization complete!")
                        display_result(data)
                    except Exception as e:
                        st.error(f"Error: {e}")

# Simulate Data Mode
elif option == "Simulate Data":
    if st.button("Run Simulation"):
        with st.spinner("Generating and optimizing simulated data..."):
            try:
                response = requests.post(f"{API_URL}/simulate")
                response.raise_for_status()
                data = response.json()
                st.success("Simulation complete!")
                display_result(data)
            except Exception as e:
                st.error(f"Error: {e}")
