import os
from google import genai

class GeminiClient:
    def __init__(self):
        # Le SDK peut lire GEMINI_API_KEY ou GOOGLE_API_KEY automatiquement,
        # mais on garde ce check pour avoir une erreur claire.
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GEMINI_API_KEY (or GOOGLE_API_KEY) in environment.")

        self.model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt: str) -> str:
        resp = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        return resp.text or ""
