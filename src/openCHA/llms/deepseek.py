from typing import Any, Dict, List
from openCHA.llms import BaseLLM
from openCHA.utils import get_from_dict_or_env
from pydantic import model_validator
import requests
import json


class DeepSeekLLM(BaseLLM):
    """
    DeepSeek via API oficial
    """

    models: Dict = {
        "deepseek-chat": 64000,
        "deepseek-coder": 64000,
    }

    api_key: str = ""
    base_url: str = "https://api.deepseek.com/v1"
    max_tokens: int = 1000

    @model_validator(mode="before")
    def validate_environment(cls, values: Dict) -> Dict:
        api_key = get_from_dict_or_env(
            values, "deepseek_api_key", "DEEPSEEK_API_KEY"
        )

        print("\n" + "="*60)
        print(" INICIALIZANDO DEEPSEEK VIA API OFICIAL")
        print(f" API Key: {api_key[:15]}...")
        print(" Modelos DeepSeek disponiveis:")
        print("   - deepseek-chat")
        print("   - deepseek-coder")
        print("="*60 + "\n")

        values["api_key"] = api_key
        return values

    def get_model_names(self) -> List[str]:
        return list(self.models.keys())

    def _parse_response(self, response: Any) -> str:
        """Método abstrato obrigatório"""
        if isinstance(response, str):
            return response
        if isinstance(response, dict):
            return response.get("content", str(response))
        return str(response)

    def _prepare_prompt(self, query: str, **kwargs: Any) -> str:
        """Método abstrato obrigatório"""
        return query

    def generate(self, query: str, **kwargs: Any) -> str:
        model_name = kwargs.get("model_name", "deepseek-chat")

        print("\n" + "="*60)
        print(f" MODELO ATIVO: {model_name}")
        print(f" QUERY: {query[:100]}...")
        print("="*60 + "\n")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": query}],
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", 0.7),
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()["choices"][0]["message"]["content"]

            print(f" RESPOSTA GERADA: {result[:100]}...\n")
            return result

        except Exception as e:
            error_msg = f"Erro ao gerar conteudo com DeepSeek: {str(e)}"
            print(f" ERRO: {error_msg}\n")
            raise RuntimeError(error_msg)
