import os
from openai import OpenAI
from dotenv import load_dotenv
from types import *

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INPUT_TOKEN_COST = 0.0000011
OUTPUT_TOKEN_COST = 0.0000044

class OpenAIService():
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def parse_information(self, system_prompt, user_prompt, response_format) -> any:
        response = self.client.responses.parse(
            model="o4-mini",
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            text_format=response_format,
            # temperature=0.1,
        )
        # print price
        print(f"Total input tokens used: {response.usage.input_tokens}")
        print(f"Total cache tokens used: {response.usage.input_tokens_details.cached_tokens}")
        print(f"Total output tokens used: {response.usage.output_tokens}")
        print(f"Total tokens used: {response.usage.total_tokens}")
        total_cost = (
            response.usage.input_tokens * INPUT_TOKEN_COST +
            response.usage.output_tokens * OUTPUT_TOKEN_COST
        )
        print(f"Total cost: ${total_cost:.4f}")
        return response.output_parsed
    
# main
if __name__ == "__main__":
    service = OpenAIService()
    service.parse_information()
   
        
