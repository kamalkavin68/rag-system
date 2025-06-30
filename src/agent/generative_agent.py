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


# def talk_to_anthropic(context):
#     system_prompt = f"""
# You are a helpful and concise AI assistant. Your task is to answer each user question using only the information in its associated context.

# Instructions:
# - Each question has its own context.
# - Answer each question independently, using only its paired context.
# - For each answer:
#     1. Start with a short, descriptive title (do NOT prefix with "Title:" or similar).
#     2. On the next line, write answer using only the provided context.
#     3. If the content supports it, format the answer using bullet points or numbered lists to improve readability.

# Now process the following question-context pairs:

# {context}
# """

#     # print(system_prompt)
#     aws_access_key = os.getenv("AWS_ACCESS_KEY")
#     aws_secret_key = os.getenv("AWS_SECRET_KEY")
#     aws_region = os.getenv("AWS_REGION")
#     aws_model = os.getenv("AWS_MODEL")

#     client = AnthropicBedrock(
#         aws_access_key=aws_access_key,
#         aws_secret_key=aws_secret_key,
#         aws_region=aws_region
#     )

#     message = client.messages.create(
#         model=aws_model,
#         max_tokens=5000,
#         # messages=chat_history + [{"role": "user", "content": system_prompt}],
#         messages=[{"role": "user", "content": system_prompt}],
#     )

#     return message.content[0].text, system_prompt


def talk_to_anthropic(context):

    system_prompt = f"""
You are a helpful and concise AI assistant. Your task is to generate informative and structured answers based solely on the paired context for each user question.

Instructions:
- Each question is followed by its corresponding context.
- Answer each question independently.
- Do NOT include or repeat the question in your response.
- Begin each answer with a short, descriptive title (do NOT prefix it with "Title:", "Question [No]:").
- On the line below the title, provide the most complete and accurate answer using only the provided context.
- You may synthesize or paraphrase the context to form a complete answer.
- Use logical inference if necessary, but do NOT introduce facts not supported or implied by the context.
- Structure the answer using bullet points or numbered lists if helpful.
- If the context lacks enough information to generate a meaningful response, write:
  Information not available for the given title: The provided context does not contain sufficient details to answer this question.

Now process the following question-context pairs:

{context}
"""

    # print(system_prompt)
    aws_access_key = os.getenv("AWS_ACCESS_KEY")
    aws_secret_key = os.getenv("AWS_SECRET_KEY")
    aws_region = os.getenv("AWS_REGION")
    aws_model = os.getenv("AWS_MODEL")

    client = AnthropicBedrock(
        aws_access_key=aws_access_key,
        aws_secret_key=aws_secret_key,
        aws_region=aws_region
    )

    message = client.messages.create(
        model=aws_model,
        max_tokens=5000,
        temperature=0.4,
        # messages=chat_history + [{"role": "user", "content": system_prompt}],
        messages=[{"role": "user", "content": system_prompt}],
    )

    return message.content[0].text, system_prompt