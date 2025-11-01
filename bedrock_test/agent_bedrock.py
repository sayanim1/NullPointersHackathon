# agent_bedrock.py
import json, boto3
from strands import Agent, tool

# ---------- Tool 1: Simulated KPI data ----------
@tool
def get_kpi_data() -> dict:
    """
    Returns simulated RAN KPI data.
    Replace this later with real or sandbox data.
    """
    data = {
        "cells": [
            {"cell_id": "C001", "rsrq": -9, "rsrp": -87, "hof": 4, "rlf": 2, "load": 0.75, "a3_hyst": 2, "ttt": 320},
            {"cell_id": "C002", "rsrq": -12, "rsrp": -91, "hof": 7, "rlf": 4, "load": 0.82, "a3_hyst": 3, "ttt": 480},
            {"cell_id": "C003", "rsrq": -7, "rsrp": -80, "hof": 2, "rlf": 1, "load": 0.60, "a3_hyst": 2, "ttt": 320}
        ]
    }
    print("📡 Loaded KPI data")
    return data


# ---------- Tool 2: Call Claude 3 Sonnet on Bedrock ----------
@tool
def analyze_with_bedrock(kpi_data: dict) -> str:
    """
    Sends KPI data to Claude 3 Sonnet (Bedrock) for optimization suggestions.
    """
    bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

    prompt = f"""
    You are a telecom RAN optimization expert.
    Here is the current KPI data (JSON):

    {json.dumps(kpi_data, indent=2)}

    Based on the data, identify any cells showing degraded performance
    (high HOF/RLF or poor RSRQ/RSRP) and propose ONE safe handover parameter
    adjustment (A3 hysteresis or TTT) with reasoning. 
    Respond in concise JSON:
    {{
      "target_cell": "...",
      "parameter_to_adjust": "...",
      "suggested_value": "...",
      "reason": "..."
    }}
    """

    response = bedrock.converse(
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 400, "temperature": 0.4}
    )

    ai_reply = response["output"]["message"]["content"][0]["text"]
    print("🧠 Claude suggestion received:")
    print(ai_reply)
    return ai_reply


# ---------- Create the agent ----------
agent = Agent(tools=[get_kpi_data, analyze_with_bedrock])

# ---------- Agent message ----------
message = """
Load the KPI data and analyze it using Bedrock to suggest a handover optimization.
"""

result = agent(message, verbose=False)
print("\n🤖 Final agent output:")
print(result)
