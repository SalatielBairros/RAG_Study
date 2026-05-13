from google import genai

class GeminiService:
    def __init__(self):
        pass

    def get_avaliable_chat_models(self):
        client = genai.Client()
        models = client.models.list()
        print("Available chat models:\n")
        for model in models:
            print(f"- {model.name} (Version: {model.version} | Description: {model.description} | Actions: {model.supported_actions})")