# agent_bedrock.py
import json, boto3, datetime
from pathlib import Path
from strands import Agent, tool
import requests
from copy import deepcopy
import random
import datetime


ttt_choices = [160, 320, 480, 640]
slice_types = ["eMBB", "URLLC", "mMTC"]
handover_failure_causes = ["LowRSRP", "TargetBusy", "Other"]

# ---------- Paths ----------
DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------- Tool 1: Load KPI data from file ----------
@tool
def load_kpi_data(source: str = "api", file_path: str = None, num_cells: int = 5) -> dict:
    """
    Loads KPI data from either:
    1. Live simulator API (default)
    2. Uploaded JSON/CSV file
    """

    if source == "file" and file_path:
        if not file_path:
            print("No file path provided for KPI data")
            return {"cells": []}
        try:
            with open(file_path, "r") as f:
                if file_path.endswith(".json"):
                    data = json.load(f)
                else:
                    import pandas as pd
                    df = pd.read_csv(f)
                    data = {"cells": df.to_dict(orient="records")}
            print(f"Loaded KPI data from file: {file_path}")
            return data
        except Exception as e:
            print(f"Failed to load KPI file: {e}")
            return {"cells":[]}
        
    if source == "api":        
        return get_dataset(num_cells)


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

def get_dataset(num_cells: int = 5):
    return generate_dataset(num_cells)


# Analyze data with Claude on Bedrock 
@tool
def analyze_with_bedrock(kpi_data: dict) -> dict:
    """
    Sends KPI data to Claude 3.5 Sonnet and saves the suggestion.
    """
    bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

    prompt = f"""
    You are a telecom RAN optimization expert.
    Analyze this KPI JSON and identify cells showing degraded performance
    (high HOF/RLF or low RSRQ). Propose adjustments to improve the initial metrics given as an input
    and return concise JSON:
    {{
      "target_cell": "...",
      "parameter_to_adjust": "...",
      "suggested_value": "...",
      "reason": "..."
    }}

    KPI Data:
    {json.dumps(kpi_data, indent=2)}
    """

    response = bedrock.converse(
        modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 2000, "temperature": 0.4}
    )

    ai_text = response["output"]["message"]["content"][0]["text"]
    print("Claude suggestion received:")
    print(ai_text)

    try:
        suggestion = json.loads(ai_text)
    except json.JSONDecodeError:
        suggestion = {"raw_text": ai_text}

    ## Save result to a fixed filename
    out_file = OUTPUT_DIR / "latest_recommendation.json"
    with open(out_file, "w") as f:
        json.dump(suggestion, f, indent=2)
    print(f"Saved recommendation to {out_file}")
    print(f"Saved recommendation to {out_file}")

    return suggestion


# ---------- Tool 3: Digital Twin Simulation ----------
from copy import deepcopy

@tool
def simulate_optimization(kpi_data: dict, suggestion: dict) -> dict:
    """
    Applies Claude's suggestion to KPI data with stronger, realistic optimization.
    Produces before/after comparison with real changes.
    """
    # Deep copy for before (unaltered)
    before = deepcopy(kpi_data)
    # Deep copy for after (we modify this)
    after_cells = deepcopy(kpi_data["cells"])
    target = suggestion.get("target_cell")

    for cell in after_cells:
        if cell["cell_id"] == target:
            param = suggestion.get("parameter_to_adjust", "").lower()
            val = suggestion.get("suggested_value")

            if param == "ttt":
                # Stronger reduction in TTT to reduce HOF/RLF
                try:
                    new_ttt = int(val)
                    cell["ttt"] = new_ttt
                    cell["hof"] = max(0, cell["hof"] - random.randint(3, 5))
                    cell["rlf"] = max(0, cell["rlf"] - random.randint(2, 4))
                    # Improve signal metrics a bit more
                    cell["rsrq"] += 2
                    cell["rsrp"] += 2
                except Exception:
                    pass

            elif "hyst" in param:
                try:
                    new_hyst = float(val)
                    cell["a3_hyst"] = new_hyst
                    cell["hof"] = max(0, cell["hof"] - random.randint(2, 4))
                    cell["rlf"] = max(0, cell["rlf"] - random.randint(1, 3))
                    cell["rsrq"] += 1
                except Exception:
                    pass

    result = {"before": before, "after": {"cells": after_cells}}

    # Save to a single latest file
    out_file = OUTPUT_DIR / "latest_simulation.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    print("Digital Twin simulation applied and saved to latest_simulation.json.")
    return result



# ---------- Create Agent ----------
#agent = Agent(tools=[load_kpi_data, analyze_with_bedrock])
agent = Agent(tools=[load_kpi_data, analyze_with_bedrock, simulate_optimization])

def run_bedrock_pipeline(source="api", file_path=None, num_cells=5):
    """
    Runs the end-to-end RAN optimization pipeline when triggered externally.
    """
    message = f"""
    1. Load the KPI data from {source}.
    2. Analyze it with Bedrock (Claude) to suggest an optimization.
    3. Apply that optimization using the digital twin simulation.
    """

    result = agent(message, verbose=False)
    try:
        output_text = result.content if hasattr(result, "content") else str(result)
        print("\nFinal agent output:")
        print(output_text)
        return output_text
    except Exception as e:
        print(f"Could not print result cleanly: {e}")
        return {"error": str(e)}


# Optional: Only run when executed directly (not imported)
if __name__ == "__main__":
    run_bedrock_pipeline()

