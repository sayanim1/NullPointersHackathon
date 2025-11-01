import boto3, json

# Create a Bedrock runtime client in us-east-1
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

response = bedrock.converse(
    modelId="anthropic.claude-3-sonnet-20240229-v1:0",
    messages=[
        {
            "role": "user",
            "content": [{"text": "Hello Claude! Respond in one short sentence."}]
        }
    ],
    inferenceConfig={"maxTokens": 100, "temperature": 0.5}
)

print(json.dumps(response["output"]["message"]["content"], indent=2))
