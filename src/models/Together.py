import pprint
import os
import dotenv
import time
from .Base import BaseModel
from together import Together as TogetherClient
dotenv.load_dotenv()
class Together(BaseModel):
    def __init__(self, temperature=0):
        self.client = TogetherClient()
    # @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def prompt(self, processed_input):
        for i in range(10):
            try:
                response = self.client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=processed_input
                )
                return response.choices[0].message.content, 0, 0
            except Exception as e:
                print(f"Attempt {i+1} failed with error: {e}")
                sleep_time = 2.0
                time.sleep(sleep_time)
       
        return response.choices[0].message.content, 0, 0