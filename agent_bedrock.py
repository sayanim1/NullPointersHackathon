# agent_bedrock.py
import json, boto3, datetime
from pathlib import Path
from strands import Agent, tool
import requests
from copy import deepcopy

# ---------- Paths ----------
DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------- Tool 1: Load KPI data from file ----------
@tool
def load_kpi_data() -> dict:
    """
    Loads KPI data from /data/sample_kpi.json
    """
    api_url = "http://127.0.0.1:8000/generate_dataset?mnum_cells=5"
    response  = requests.get(api_url)

    if response.status_code == 200:
        data = response.json()
        print(f"Called KPI data from FastAPI simulator: {len(data['cells'])} cells")
        return data
    else:
        print(f"Failed to fetch KPI data, status code: {response.status_code}")
        return {"cells": []}

    #file_path = DATA_DIR / "sample_kpi.json"
    #with open(file_path, "r") as f:
    #    data = json.load(f)
    #print(f"Loaded KPI data from {file_path}")
    #return data


# ---------- Tool 2: Analyze data with Claude on Bedrock ----------
@tool
def analyze_with_bedrock(kpi_data: dict) -> dict:
    """
    Sends KPI data to Claude 3.5 Sonnet and saves the suggestion.
    """
    bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

    prompt = f"""
    You are a telecom RAN optimization expert.
    Analyze this KPI JSON and identify cells showing degraded performance
    (high HOF/RLF or low RSRQ). Propose ONE adjustment to A3 hysteresis or TTT
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

    # Save result
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_file = OUTPUT_DIR / f"recommendation_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(suggestion, f, indent=2)
    print(f"Saved recommendation to {out_file}")

    return suggestion


# ---------- Tool 3: Digital Twin Simulation ----------
@tool
def simulate_optimization(kpi_data: dict, suggestion: dict) -> dict:
    """
    Simulates applying Claude's suggested parameter change to KPI data.
    Produces a before/after comparison to mimic a digital twin validation.
    """
    cells = deepcopy(kpi_data["cells"])
    target = suggestion.get("target_cell")

    # Define simple rule-based improvements
    for cell in cells:
        if cell["cell_id"] == target:
            param = suggestion.get("parameter_to_adjust", "").lower()
            val = suggestion.get("suggested_value")

            if param == "ttt":
                # Example: shorter TTT reduces handover failures
                try:
                    new_ttt = int(val)
                    cell["ttt"] = new_ttt
                    cell["hof"] = max(0, cell["hof"] - 2)
                    cell["rlf"] = max(0, cell["rlf"] - 1)
                except Exception:
                    pass
            elif "hyst" in param:
                # Example: optimized hysteresis stabilizes RSRQ
                cell["a3_hyst"] = float(val)
                cell["rsrq"] += 1.0  # slight improvement
                cell["hof"] = max(0, cell["hof"] - 1)

    result = {"before": kpi_data, "after": {"cells": cells}}
    print("Digital Twin simulation applied.")
    return result



# ---------- Create Agent ----------
#agent = Agent(tools=[load_kpi_data, analyze_with_bedrock])
agent = Agent(tools=[load_kpi_data, analyze_with_bedrock, simulate_optimization])


# ---------- Agent Task ----------
# message = """
# Load the KPI data and analyze it using Bedrock to suggest a handover optimization.
# """

# result = agent(message, verbose=False)

# print("\n Final agent output:")
# #print(json.dumps(result, indent=2))
# # ---------- Agent Task ----------
# message = """
# Load the KPI data and analyze it using Bedrock to suggest a handover optimization.
# """

message = """
1. Load the KPI data.
2. Analyze it with Bedrock (Claude) to suggest an optimization.
3. Apply that optimization using the digital twin simulation.
"""

result = agent(message, verbose=False)

# Handle AgentResult object safely
try:
    # Strands AgentResult stores model reply under .content
    output_text = result.content if hasattr(result, "content") else str(result)
    print("\nFinal agent output:")
    print(output_text)
except Exception as e:
    print(f"Could not print result cleanly: {e}")

