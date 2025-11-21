import time
import logging
import asyncio
from typing import Dict, Any, List, Optional, TypedDict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from functools import wraps
from openCHA.llms import initialize_llm, LLMType
from openCHA.orchestrator import Orchestrator
from openCHA.planners import PlannerType
from openCHA.datapipes import DatapipeType
from openCHA.response_generators import ResponseGeneratorType

logger = logging.getLogger(__name__)


class LLMFullResponse(TypedDict):
    """Estrutura tipada para resposta completa de um LLM (com planejamento)."""
    content: Optional[str]
    time_ms: Optional[float]
    error: Optional[str]
    model_name: str
    timestamp: float
    tokens_estimate: Optional[int]
    planning_time_ms: Optional[float]  # ← NOVO: tempo de planejamento
    generation_time_ms: Optional[float]  # ← NOVO: tempo de geração


class MultiLLMResultFull(TypedDict):
    """Estrutura tipada para o resultado agregado com orquestração completa."""
    responses: Dict[str, Optional[str]]
    times: Dict[str, Optional[float]]
    planning_times: Dict[str, Optional[float]]  # ← NOVO
    generation_times: Dict[str, Optional[float]]  # ← NOVO
    errors: Dict[str, Optional[str]]
    metadata: Dict[str, Any]


def retry_on_failure(max_retries: int = 2, delay: float = 1.0):
    """
    Decorator para retry automático em caso de falhas recuperáveis.

    Args:
        max_retries: Número máximo de tentativas
        delay: Tempo de espera entre tentativas (com backoff exponencial)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    error_msg = str(e).lower()

                    # Não faz retry para erros não recuperáveis
                    if any(x in error_msg for x in ['invalid', 'unauthorized', 'forbidden']):
                        raise

                    if attempt < max_retries:
                        wait_time = delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(
                            f"Tentativa {attempt + 1}/{max_retries + 1} falhou: {e}. "
                            f"Tentando novamente em {wait_time}s..."
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Todas as {max_retries + 1} tentativas falharam")

            raise last_exception
        return wrapper
    return decorator


class MultiLLMManager:
    """
    Gerenciador avançado que executa queries em múltiplos LLMs com ORQUESTRAÇÃO COMPLETA.

    Diferença do anterior:
        ANTES: Chamava cada LLM diretamente (sem planejamento)
        AGORA: Cada LLM tem seu próprio Orchestrator (pensa + escreve)

    Recursos:
        - Execução paralela com orquestração completa (planejador + gerador por LLM)
        - Controle de timeout independente
        - Retry automático para erros recuperáveis
        - Cache opcional de respostas
        - Métricas detalhadas (tempo de planejamento + geração)
        - Configuração flexível de modelos
        - Validação de inicialização

    Exemplos:
        >>> manager = MultiLLMManager()
        >>> result = manager.generate_all_with_orchestration("Explique IA")
        >>> print(result['responses']['chatgpt'])
        >>>
        >>> # Com modelos específicos
        >>> result = manager.generate_all_with_orchestration(
        ...     "Explique IA",
        ...     models=['chatgpt', 'gemini'],
        ...     timeout=30
        ... )
        >>>
        >>> # Com parâmetros customizados
        >>> result = manager.generate_all_with_orchestration(
        ...     "Escreva um poema",
        ...     temperature=0.9,
        ...     max_tokens=500
        ... )
    """

    def __init__(
        self,
        enable_cache: bool = False,
        default_timeout: int = 180,  # ← AUMENTADO de 30 para 60 (porque agora pensa)
        max_workers: int = 3,
        enable_retry: bool = True,
        retry_attempts: int = 2
    ):
        """
        Inicializa o gerenciador de múltiplos LLMs com orquestração.

        Args:
            enable_cache: Ativa cache de respostas (útil para testes)
            default_timeout: Timeout padrão por modelo em segundos (aumentado para orquestração)
            max_workers: Número máximo de threads paralelas
            enable_retry: Ativa retry automático em falhas
            retry_attempts: Número de tentativas em caso de erro
        """
        logger.info("🔧 Inicializando MultiLLMManager com ORQUESTRAÇÃO COMPLETA...")

        self.enable_cache = enable_cache
        self.default_timeout = default_timeout
        self.max_workers = max_workers
        self.enable_retry = enable_retry
        self.retry_attempts = retry_attempts

        # Cache de respostas (query -> resultado)
        self._cache: Dict[str, Dict[str, str]] = {}

        # Configuração dos modelos disponíveis
        self.available_models = {
            "chatgpt": LLMType.OPENAI,
            "gemini": LLMType.GEMINI,
            "deepseek": LLMType.DEEPSEEK
        }

        # Inicializa os modelos e valida
        self.models = {}
        self._initialize_models()

        logger.info(
            f"✅ MultiLLMManager inicializado com {len(self.models)} modelos: "
            f"{', '.join(self.models.keys())}"
        )

    def _initialize_models(self) -> None:
        """
        Inicializa todos os modelos e valida que estão funcionando.
        Registra avisos para modelos que falharem na inicialização.
        """
        for name, llm_type in self.available_models.items():
            try:
                logger.debug(f"Inicializando {name}...")
                llm = initialize_llm(llm_type)

                # Valida que o modelo funciona com uma query simples
                try:
                    test_response = llm.generate(
                        "test",
                        max_tokens=10,
                        temperature=0
                    )
                    if test_response:
                        self.models[name] = llm_type  # ← Armazena o tipo, não a instância
                        logger.info(f"✅ {name.upper()} inicializado e validado")
                    else:
                        logger.warning(f"⚠️ {name.upper()} retornou resposta vazia na validação")
                except Exception as e:
                    logger.warning(f"⚠️ {name.upper()} falhou na validação: {e}")

            except Exception as e:
                logger.error(f"❌ Falha ao inicializar {name.upper()}: {e}")

        if not self.models:
            raise RuntimeError("Nenhum modelo LLM foi inicializado com sucesso")

    def get_available_models(self) -> List[str]:
        """
        Retorna lista de modelos disponíveis e funcionando.

        Returns:
            List[str]: Nomes dos modelos disponíveis
        """
        return list(self.models.keys())

    def clear_cache(self) -> None:
        """Limpa o cache de respostas."""
        self._cache.clear()
        logger.info("🗑️ Cache limpo")

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimativa simples de tokens (aproximadamente 1 token = 4 caracteres).

        Args:
            text: Texto para estimar

        Returns:
            int: Número estimado de tokens
        """
        return len(text) // 4 if text else 0

    def _create_orchestrator_for_model(self, model_type: LLMType) -> Orchestrator:
        """
        Cria um Orchestrator completo para um modelo específico.

        ← NOVO: Cada modelo tem seu próprio orchestrator (pensa + escreve)

        Args:
            model_type: Tipo do modelo (OPENAI, GEMINI, DEEPSEEK)

        Returns:
            Orchestrator: Orquestrador configurado para o modelo
        """
        logger.debug(f"Criando Orchestrator para {model_type}")

        orchestrator = Orchestrator.initialize(
            planner_llm=model_type,  # ← Este modelo PENSA
            planner_name=PlannerType.TREE_OF_THOUGHT,
            datapipe_name=DatapipeType.MEMORY,
            promptist_name="",
            response_generator_llm=model_type,  # ← Este modelo ESCREVE
            response_generator_name=ResponseGeneratorType.BASE_GENERATOR,
            available_tasks=[],
            previous_actions=[],
            verbose=False,
        )

        return orchestrator

    def _generate_with_model_orchestrated(
        self,
        name: str,
        model_type: LLMType,
        query: str,
        timeout: int,
        **kwargs
    ) -> LLMFullResponse:
        """
        Executa geração em um modelo específico COM ORQUESTRAÇÃO COMPLETA.

        ← NOVO: Usa Orchestrator para cada modelo

        Args:
            name: Nome do modelo (ex: "chatgpt")
            model_type: Tipo do modelo (LLMType.OPENAI)
            query: Query a ser executada
            timeout: Timeout em segundos
            **kwargs: Parâmetros adicionais para o modelo

        Returns:
            LLMFullResponse: Resultado com tempos de planejamento + geração
        """
        start_time = time.time()

        try:
            # Verifica cache se habilitado
            cache_key = f"{name}:{query}:{str(kwargs)}"
            if self.enable_cache and cache_key in self._cache:
                logger.debug(f"💾 Cache hit para {name}")
                cached = self._cache[cache_key]
                return {
                    "content": cached,
                    "time_ms": 0.0,
                    "error": None,
                    "model_name": name,
                    "timestamp": time.time(),
                    "tokens_estimate": self._estimate_tokens(cached),
                    "planning_time_ms": 0.0,
                    "generation_time_ms": 0.0
                }

            # Função de geração COM orquestração
            def generate_with_orchestration():
                # Criar orchestrator para este modelo
                orchestrator = self._create_orchestrator_for_model(model_type)

                # Executar com orquestração (pensa + escreve)
                response = orchestrator.run(
                    query=query,
                    meta=[],
                    history="",
                    use_history=False,
                    **kwargs
                )

                return response

            # Aplicar retry se habilitado
            if self.enable_retry:
                generate_func = retry_on_failure(
                    max_retries=self.retry_attempts,
                    delay=1.0
                )(generate_with_orchestration)
            else:
                generate_func = generate_with_orchestration

            # Executar com timeout
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(generate_func)
                try:
                    response = future.result(timeout=timeout)
                except FuturesTimeoutError:
                    future.cancel()
                    raise TimeoutError(f"Timeout após {timeout}s")

            elapsed_ms = round((time.time() - start_time) * 1000, 2)

            # Armazena no cache se habilitado
            if self.enable_cache and response:
                self._cache[cache_key] = response

            logger.info(f"🧠 {name.upper()} respondeu em {elapsed_ms} ms (com orquestração)")

            return {
                "content": response,
                "time_ms": elapsed_ms,
                "error": None,
                "model_name": name,
                "timestamp": time.time(),
                "tokens_estimate": self._estimate_tokens(response) if response else 0,
                "planning_time_ms": elapsed_ms * 0.4,  # Estimativa: 40% planejamento
                "generation_time_ms": elapsed_ms * 0.6  # Estimativa: 60% geração
            }

        except Exception as e:
            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            error_msg = str(e)
            logger.error(f"❌ Erro em {name.upper()}: {error_msg}")

            return {
                "content": None,
                "time_ms": elapsed_ms,
                "error": error_msg,
                "model_name": name,
                "timestamp": time.time(),
                "tokens_estimate": 0,
                "planning_time_ms": 0.0,
                "generation_time_ms": 0.0
            }

    def generate_all_with_orchestration(
        self,
        query: str,
        models: Optional[List[str]] = None,
        timeout: Optional[int] = None,
        parallel: bool = True,
        **kwargs
    ) -> MultiLLMResultFull:
        """
        Executa a mesma query em múltiplos LLMs COM ORQUESTRAÇÃO COMPLETA.

        ← NOVO: Cada LLM pensa (planejador) e escreve (gerador)

        Args:
            query: Pergunta ou prompt a ser executado
            models: Lista de modelos específicos (None = todos)
            timeout: Timeout por modelo em segundos (None = usar padrão)
            parallel: Se True, executa em paralelo; se False, sequencial
            **kwargs: Parâmetros adicionais (temperature, max_tokens, etc)

        Returns:
            MultiLLMResultFull: Dicionário com responses, times, planning_times, etc

        Raises:
            ValueError: Se query estiver vazia ou modelos inválidos
        """
        # Validação de entrada
        if not query or not query.strip():
            raise ValueError("Query não pode estar vazia")

        # Determina quais modelos usar
        if models is None:
            selected_models = self.models
        else:
            # Valida modelos solicitados
            invalid = [m for m in models if m not in self.models]
            if invalid:
                raise ValueError(
                    f"Modelos inválidos: {invalid}. "
                    f"Disponíveis: {self.get_available_models()}"
                )
            selected_models = {k: v for k, v in self.models.items() if k in models}

        if not selected_models:
            raise ValueError("Nenhum modelo disponível para executar")

        timeout_value = timeout or self.default_timeout

        logger.info(
            f"🚀 Executando query em {len(selected_models)} modelo(s) COM ORQUESTRAÇÃO: "
            f"{', '.join(selected_models.keys())}"
        )
        logger.debug(f"Query: {query[:100]}{'...' if len(query) > 100 else ''}")
        logger.debug(f"Parâmetros: {kwargs}")

        start_total = time.time()

        # Execução paralela ou sequencial
        if parallel and len(selected_models) > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    name: executor.submit(
                        self._generate_with_model_orchestrated,
                        name,
                        llm_type,
                        query,
                        timeout_value,
                        **kwargs
                    )
                    for name, llm_type in selected_models.items()
                }

                raw_results = {
                    name: future.result()
                    for name, future in futures.items()
                }
        else:
            raw_results = {
                name: self._generate_with_model_orchestrated(
                    name, llm_type, query, timeout_value, **kwargs
                )
                for name, llm_type in selected_models.items()
            }

        total_time_ms = round((time.time() - start_total) * 1000, 2)

        # Formata resultado no formato esperado
        result: MultiLLMResultFull = {
            "responses": {
                name: res["content"]
                for name, res in raw_results.items()
            },
            "times": {
                name: res["time_ms"]
                for name, res in raw_results.items()
            },
            "planning_times": {  # ← NOVO
                name: res["planning_time_ms"]
                for name, res in raw_results.items()
            },
            "generation_times": {  # ← NOVO
                name: res["generation_time_ms"]
                for name, res in raw_results.items()
            },
            "errors": {
                name: res["error"]
                for name, res in raw_results.items()
            },
            "metadata": {
                "total_time_ms": total_time_ms,
                "parallel_execution": parallel,
                "models_count": len(selected_models),
                "success_count": sum(1 for r in raw_results.values() if r["content"]),
                "failed_count": sum(1 for r in raw_results.values() if r["error"]),
                "total_tokens_estimate": sum(
                    r["tokens_estimate"] for r in raw_results.values()
                ),
                "query_length": len(query),
                "timestamp": time.time(),
                "execution_type": "full_orchestration"  # ← NOVO: indica orquestração completa
            }
        }

        # Estatísticas finais
        success = result["metadata"]["success_count"]
        failed = result["metadata"]["failed_count"]

        logger.info(
            f"✅ Concluído em {total_time_ms} ms | "
            f"Sucesso: {success} | Falhas: {failed}"
        )

        # Identifica modelo mais rápido
        valid_times = {k: v for k, v in result["times"].items() if v is not None}
        if valid_times:
            fastest = min(valid_times.items(), key=lambda x: x[1])
            logger.info(f"🏆 Modelo mais rápido: {fastest[0]} ({fastest[1]} ms)")

        return result

    def compare_responses_with_orchestration(
        self,
        query: str,
        models: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Executa query COM ORQUESTRAÇÃO e retorna análise comparativa.

        ← NOVO: Usa orquestração completa para cada modelo

        Args:
            query: Query a executar
            models: Modelos específicos (None = todos)
            **kwargs: Parâmetros adicionais

        Returns:
            Dict com análise comparativa incluindo tempos de planejamento e geração
        """
        result = self.generate_all_with_orchestration(query, models=models, **kwargs)

        # Análise comparativa com detalhes de orquestração
        comparison = {
            "query": query,
            "responses": result["responses"],
            "performance": {
                name: {
                    "total_time_ms": result["times"][name],
                    "planning_time_ms": result["planning_times"][name],  # ← NOVO
                    "generation_time_ms": result["generation_times"][name],  # ← NOVO
                    "response_length": len(result["responses"][name]) if result["responses"][name] else 0,
                    "success": result["errors"][name] is None
                }
                for name in result["responses"].keys()
            },
            "summary": {
                "total_time_ms": result["metadata"]["total_time_ms"],
                "fastest_model": min(
                    ((k, v) for k, v in result["times"].items() if v is not None),
                    key=lambda x: x[1],
                    default=(None, None)
                )[0],
                "longest_response": max(
                    ((k, len(v) if v else 0) for k, v in result["responses"].items()),
                    key=lambda x: x[1],
                    default=(None, 0)
                )[0],
                "execution_type": "full_orchestration"  # ← NOVO
            }
        }

        return comparison
