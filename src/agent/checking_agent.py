

import os

from anthropic import AnthropicBedrock


def validate_response_with_claude(context, generated_response):
    validation_prompt = f"""
You are a critical and precise evaluator.

Your task is to validate whether the following generated responses are:
1. Factually supported by the information in the given question-context pairs.
2. Free from hallucinated or made-up information.
3. Relevant to each question and its context.

Instructions:
- Carefully compare the response to the matching context.
- If all answers are factually grounded and fully supported by the context, respond with: "Valid".
- If any answer contains unsupported, irrelevant, or made-up content, respond with: "Invalid".
- Do NOT explain or include anything other than the word "Valid" or "Invalid".

Question-Context Pairs:
{context}

Generated Response:
{generated_response}
"""

    # Load AWS credentials and config from environment
    aws_access_key = os.getenv("AWS_ACCESS_KEY")
    aws_secret_key = os.getenv("AWS_SECRET_KEY")
    aws_region = os.getenv("AWS_REGION")
    aws_model = os.getenv("AWS_MODEL")  # e.g. "anthropic.claude-3-sonnet-20240229"

    # Initialize Claude (Anthropic) client
    client = AnthropicBedrock(
        aws_access_key=aws_access_key,
        aws_secret_key=aws_secret_key,
        aws_region=aws_region
    )

    message = client.messages.create(
        model=aws_model,
        max_tokens=200,
        temperature=0.0,
        messages=[{"role": "user", "content": validation_prompt.strip()}]
    )

    return message.content[0].text.strip()
