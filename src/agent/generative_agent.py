import os
from anthropic import AnthropicBedrock


class BedrockClient:

    def __init__(self, aws_access_key: str, aws_secret_key: str, aws_region: str, model: str, max_token: int = 256):

        self.model = model
        self.max_token = max_token
        self.client = AnthropicBedrock(
            aws_access_key=aws_access_key,
            aws_secret_key=aws_secret_key,
            aws_region=aws_region
        )

    def create_message(self, chat_history: list[dict[str, str]]):
        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_token,
            messages=chat_history
        )

        return message.content[0].text


def talk_to_anthropic(prompt, chat_history, context):

    system_prompt = f"""
    You are a helpful and informative assistant. Your task is to answer user questions based on the provided context.
    First, analyze the user's question and identify the key information needed to answer it accurately.
    Next, retrieve relevant information from the following context:
    {context}
    Finally, synthesize the retrieved information into a clear, concise, and well-organized response. Avoid using jargon or technical terms unless necessary.
    
    Instruction:
        - Don't say about the context provided in the prompt
    
    """
    
    aws_access_key = os.getenv("AWS_ACCESS_KEY")
    aws_secret_key = os.getenv("AWS_SECRET_KEY")
    aws_region = os.getenv("AWS_REGION")
    aws_model = os.getenv("AWS_MODEL")
    

    client = AnthropicBedrock(
        aws_access_key=aws_access_key,
        aws_secret_key=aws_secret_key,
        aws_region=aws_region
    )

    full_prompt = f"{system_prompt}\n\n{prompt}"

    message = client.messages.create(
        model=aws_model,
        max_tokens=5000,
        messages=chat_history + [{"role": "user", "content": full_prompt}],
    )

    return message.content[0].text

