import os
from abc import ABC, abstractmethod
from openai import OpenAI
from dotenv import load_dotenv
class CompletionGenerator(ABC):
    @abstractmethod
    def generate_one_completion(self, prompt):
        pass

class OpenAICompletionGenerator(CompletionGenerator):
    def __init__(self, model):
        # Load environment variables
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def generate_one_completion(self, prompt):
        chat_completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model
        )
        return chat_completion.choices[0].message.content