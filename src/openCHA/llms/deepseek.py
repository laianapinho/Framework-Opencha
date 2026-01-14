import logging
from typing import Any, Dict, List, Optional
from openCHA.llms import BaseLLM
from openCHA.utils import get_from_dict_or_env
from pydantic import model_validator, Field
import requests
from requests.exceptions import RequestException, Timeout, HTTPError

logger = logging.getLogger(__name__)


class DeepSeekLLM(BaseLLM):
    """
    Implementação do DeepSeek via API oficial.
    Compatível com a arquitetura do OpenCHA (BaseLLM).

    Exemplos de uso:
        >>> llm = DeepSeekLLM()
        >>> resposta = llm.generate("Explique o que é recursão")
        >>>
        >>> # Com parâmetros customizados
        >>> resposta = llm.generate(
        ...     "Escreva um código Python",
        ...     model_name="deepseek-coder",
        ...     temperature=0.3,
        ...     max_tokens=2000
        ... )
    """

    models: Dict[str, int] = {
        "deepseek-chat": 64000,
        "deepseek-coder": 64000,
    }

    api_key: str = Field(default="", exclude=True)
    base_url: str = "https://api.deepseek.com/v1"
    max_tokens: int = 1000
    temperature: float = 0.7
    timeout: int = 180
    default_model: str = "deepseek-chat"

    @model_validator(mode="before")
    def validate_environment(cls, values: Dict) -> Dict:
        """
        Valida se a API key do DeepSeek está disponível no ambiente.
        Carrega de values dict ou variável de ambiente DEEPSEEK_API_KEY.
        """
        api_key = get_from_dict_or_env(
            values, "deepseek_api_key", "DEEPSEEK_API_KEY"
        )

        if not api_key:
            raise ValueError(
                "API key do DeepSeek não encontrada. "
                "Configure a variável de ambiente DEEPSEEK_API_KEY "
                "ou passe 'deepseek_api_key' no construtor."
            )

        logger.info("DeepSeekLLM inicializado com sucesso")
        logger.debug(f"Modelos disponíveis: {list(values.get('models', {}).keys())}")

        values["api_key"] = api_key
        return values

    def get_model_names(self) -> List[str]:
        """
        Retorna a lista de modelos suportados pelo adaptador.

        Returns:
            List[str]: Lista com nomes dos modelos disponíveis
        """
        return list(self.models.keys())

    def get_max_context_length(self, model_name: str) -> int:
        """
        Retorna o tamanho máximo do contexto para um modelo específico.

        Args:
            model_name: Nome do modelo

        Returns:
            int: Número máximo de tokens de contexto
        """
        return self.models.get(model_name, 64000)

    def _validate_model(self, model_name: str) -> None:
        """
        Valida se o modelo especificado é suportado.

        Args:
            model_name: Nome do modelo a validar

        Raises:
            ValueError: Se o modelo não for suportado
        """
        if model_name not in self.get_model_names():
            raise ValueError(
                f"Modelo '{model_name}' não é suportado. "
                f"Modelos válidos: {', '.join(self.get_model_names())}"
            )

    def _parse_response(self, response: Dict[str, Any]) -> str:
        """
        Extrai o texto da resposta no formato da API oficial do DeepSeek.

        Args:
            response: Objeto de resposta da API do DeepSeek

        Returns:
            str: Texto gerado pelo modelo

        Raises:
            ValueError: Se a estrutura da resposta for inválida
        """
        try:
            if "choices" not in response:
                raise ValueError("Resposta não contém o campo 'choices'")

            if not response["choices"]:
                raise ValueError("Lista 'choices' está vazia")

            content = response["choices"][0]["message"]["content"]

            if not content:
                logger.warning("Resposta gerada está vazia")

            return content

        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Erro ao parsear resposta: {e}")
            logger.debug(f"Resposta recebida: {response}")
            raise ValueError(
                f"Estrutura de resposta inválida da API DeepSeek: {str(e)}"
            )

    def _prepare_prompt(
        self,
        query: str,
        system_message: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        """
        Prepara o prompt no formato esperado pela API do DeepSeek.
        Suporta mensagens do sistema e histórico de conversa.

        Args:
            query: Pergunta ou prompt do usuário
            system_message: Mensagem do sistema (opcional)
            conversation_history: Histórico de mensagens anteriores (opcional)

        Returns:
            List[Dict[str, str]]: Lista de mensagens formatadas
        """
        messages = []

        # Adiciona mensagem do sistema se fornecida
        if system_message:
            messages.append({"role": "system", "content": system_message})

        # Adiciona histórico de conversa se fornecido
        if conversation_history:
            messages.extend(conversation_history)

        # Adiciona a query atual do usuário
        messages.append({"role": "user", "content": query})

        logger.debug(f"Prompt preparado com {len(messages)} mensagens")

        return messages

    def _validate_parameters(self, max_tokens: int, temperature: float) -> None:
        """
        Valida os parâmetros de geração.

        Args:
            max_tokens: Número máximo de tokens
            temperature: Valor de temperatura

        Raises:
            ValueError: Se os parâmetros forem inválidos
        """
        if max_tokens <= 0:
            raise ValueError("max_tokens deve ser maior que 0")

        if not 0 <= temperature <= 2:
            raise ValueError("temperature deve estar entre 0 e 2")

    def generate(self, query: str, **kwargs: Any) -> str:
        """
        Gera uma resposta usando o modelo DeepSeek.

        Args:
            query: Pergunta ou prompt do usuário
            **kwargs: Parâmetros opcionais:
                - model_name (str): Nome do modelo (padrão: "deepseek-chat")
                - max_tokens (int): Limite de tokens de saída (padrão: 1000)
                - temperature (float): Controle de criatividade 0-2 (padrão: 0.7)
                - stop (List[str] ou str): Stop tokens opcionais
                - system_message (str): Mensagem do sistema
                - conversation_history (List[Dict]): Histórico de conversa

        Returns:
            str: Resposta gerada pelo modelo

        Raises:
            ValueError: Se parâmetros forem inválidos
            RuntimeError: Se houver erro na comunicação com a API
        """
        # Extrai e valida parâmetros
        model_name = kwargs.get("model_name", self.default_model)
        self._validate_model(model_name)

        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        self._validate_parameters(max_tokens, temperature)

        stop = kwargs.get("stop", None)
        system_message = kwargs.get("system_message", None)
        conversation_history = kwargs.get("conversation_history", None)

        # Log do modelo e tokens sendo utilizados
        print("here", max_tokens, model_name)

        # Prepara o prompt
        messages = self._prepare_prompt(query, system_message, conversation_history)

        # Monta headers e payload
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if stop:
            payload["stop"] = stop if isinstance(stop, list) else [stop]

        logger.info(f"Gerando resposta com modelo: {model_name}")
        logger.debug(f"Parâmetros: max_tokens={max_tokens}, temperature={temperature}")

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )

            response.raise_for_status()
            data = response.json()

            result = self._parse_response(data)
            logger.info("Resposta gerada com sucesso")

            return result

        except Timeout:
            error_msg = f"Timeout ao conectar com DeepSeek API após {self.timeout}s"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        except HTTPError as e:
            status_code = e.response.status_code
            error_msg = f"Erro HTTP {status_code} da API DeepSeek"

            try:
                error_detail = e.response.json()
                error_msg += f": {error_detail}"
            except:
                error_msg += f": {e.response.text}"

            logger.error(error_msg)
            raise RuntimeError(error_msg)

        except RequestException as e:
            error_msg = f"Erro de conexão com DeepSeek API: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        except Exception as e:
            error_msg = f"Erro inesperado ao gerar conteúdo: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg)
