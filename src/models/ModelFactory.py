from models.Gemini import Gemini
from models.OpenAI import ChatGPT
from models.OpenAI import GPT4
from models.Together import Together


class ModelFactory:
    @staticmethod
    def get_model_class(model_name):
        if model_name == "Gemini":
            return Gemini
        elif model_name == "ChatGPT":
            return ChatGPT
        elif model_name == "GPT4":
            return GPT4
        elif model_name == "Together":
            return Together
        else:
            raise Exception(f"Unknown model name {model_name}")
