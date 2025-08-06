from typing import Any, Dict, List

from openCHA.llms import BaseLLM
from openCHA.utils import get_from_dict_or_env
from pydantic import model_validator


class GeminiLLM(BaseLLM):
    """
    **description:**

        This class implements the Gemini API using Google's Generative AI SDK.
        https://ai.google.dev/gemini-api/docs/get-started/python
    """

    models: Dict = {
        "gemini-2.5-flash-lite": 512
    }

    llm_model: Any = None
    api_key: str = ""
    max_tokens: int = 100

    @model_validator(mode="before")
    def validate_environment(cls, values: Dict) -> Dict:
        """
        Validate that Gemini API key and SDK are available in the environment.
        """
        gemini_api_key = get_from_dict_or_env(
            values, "gemini_api_key", "GEMINI_API_KEY"
        )
        print(" Using API_KEY: {}".format(gemini_api_key))
        values["api_key"] = gemini_api_key
        try:
            import google.generativeai as genai

            genai.configure(api_key=gemini_api_key)
            values["llm_model"] = genai
        except ImportError:
            raise ValueError(
                "Could not import `google-generativeai` package. "
                "Install it with `pip install google-generativeai`."
            )
        return values

    def get_model_names(self) -> List[str]:
        return list(self.models.keys())

    def is_max_token(self, model_name: str, query: str) -> bool:
        model_max_token = self.models.get(model_name, self.max_tokens)
        return len(query.split()) > model_max_token  # simplistic token count

    def _parse_response(self, response: Any) -> str:
        if hasattr(response, "text"):
            return response.text
        elif hasattr(response, "candidates"):
            return response.candidates[0].text if response.candidates else ""
        return str(response)

    def _prepare_prompt(self, prompt: str) -> Any:
        return prompt  # Gemini handles plain strings

    def generate(self, query: str, **kwargs: Any) -> str:
        model_name = kwargs.get("model_name", "gemini-2.5-flash-lite")
        if model_name not in self.get_model_names():
            raise ValueError(
                f"Model '{model_name}' is not supported by GeminiLLM."
            )

        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        prompt = self._prepare_prompt(query)

        model = self.llm_model.GenerativeModel(model_name)
        response = model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": max_tokens,
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 1.0),
            }
        )

        try:
            response = model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature": kwargs.get("temperature", 0.7),
                    "top_p": kwargs.get("top_p", 1.0),
                }
            )
            return self._parse_response(response)
        except Exception as e:
            raise RuntimeError(f"Erro ao gerar conteúdo com Gemini: {str(e)}")
