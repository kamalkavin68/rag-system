import json
import os
import re
from typing import Dict, Optional
from anthropic import AnthropicBedrock


class QueryTransformation:
    
    """
    Query Transformation: Query Rewriting, Query Expansion \n
    Query Decomposition
    
    """
    
    def __init__(self):
        self.client = AnthropicBedrock(
            aws_access_key=self._get_env("AWS_ACCESS_KEY"),
            aws_secret_key=self._get_env("AWS_SECRET_KEY"),
            aws_region=self._get_env("AWS_REGION")
        )
        self.model = self._get_env("AWS_MODEL")

    def _get_env(self, key: str) -> str:
        value = os.getenv(key)
        if not value:
            raise EnvironmentError(f"Missing environment variable: {key}")
        return value

    def _build_prompt(self, query: str) -> str:
        return f"""
You are a helpful assistant.

INSTRUCTIONS:
- If the query is already a complete, grammatical question, do not change it.
- Otherwise, correct grammar, spelling, and slang.
- If the query contains multiple parts or topics, split them into separate questions, even if they are grammatically joined.
- Do NOT assume or add new information.
- Return valid JSON in the format:
{{ "Q1": "question 1", "Q2": "question 2", ... }}

Only return JSON. No explanation or extra lines.

Query: {query}
""".strip()

    def _extract_json(self, response_text: str) -> Optional[Dict[str, str]]:
        match = re.search(r'\{[\s\S]*?\}', response_text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError as e:
                print("JSON decode error:", e)
        else:
            print("No JSON object found in response.")
        return None

    def process_query(self, query: str) -> Optional[Dict[str, str]]:
        prompt = self._build_prompt(query)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=5000,
            messages=[{"role": "user", "content": prompt}]
        )
        return self._extract_json(response.content[0].text)