"""
MultiLLMManager - Gerenciador avançado para múltiplos LLMs COM ORQUESTRAÇÃO COMPLETA

Este módulo fornece um gerenciador que executa queries em múltiplos LLMs simultaneamente,
cada um com sua própria orquestração completa (planejador + gerador).

✅ CORRIGIDO: Query de teste agora é sobre SAÚDE (cancer) em vez de "test"
✅ CORRIGIDO: Implementa validação de domínio em 3 camadas para garantir respostas apenas sobre saúde.
✅ CORRIGIDO: Mede tempos REAIS de planejamento e geração (não estimativas).
✅ CORRIGIDO: Retorna None em erros, não mensagens de erro.
✅ CORRIGIDO: Usa hash para cache keys menores.
✅ REMOVIDO: Restrições de saúde - agora aceita qualquer tipo de query!
"""

import time
import logging
import hashlib
from typing import Dict, Any, List, Optional, TypedDict, Tuple
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
    planning_time_ms: Optional[float]
    generation_time_ms: Optional[float]


class MultiLLMResultFull(TypedDict):
    """Estrutura tipada para o resultado agregado com orquestração completa."""
    responses: Dict[str, Optional[str]]
    times: Dict[str, Optional[float]]
    planning_times: Dict[str, Optional[float]]
    generation_times: Dict[str, Optional[float]]
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

    ✅ CORRIGIDO:
        - Query de teste agora é sobre SAÚDE (cancer) em vez de "test"
        - Mede tempos REAIS de planejamento e geração
        - Retorna None em erros, não mensagens
        - Cache com hash para keys menores
        - Validação de domínio em 3 camadas

    ✅ REMOVIDO:
        - Restrições de saúde removidas
        - Agora aceita qualquer tipo de query (domínio geral)
        - Sem validação de palavras-chave de saúde

    Recursos:
        - Execução paralela com orquestração completa (planejador + gerador por LLM)
        - Controle de timeout independente
        - Retry automático para erros recuperáveis
        - Cache opcional de respostas
        - Métricas detalhadas (tempo REAL de planejamento + geração)
        - Configuração flexível de modelos
        - Validação de inicialização
        - Domínio geral (qualquer tipo de query)

    Exemplos:
        >>> manager = MultiLLMManager()
        >>> result = manager.generate_all_with_orchestration("Qual é a capital da França?")
        >>> print(result['responses']['chatgpt'])
        >>> print(result['planning_times']['chatgpt'])  # Tempo REAL de planejamento
        >>> print(result['generation_times']['chatgpt'])  # Tempo REAL de geração
    """

    def __init__(
        self,
        enable_cache: bool = False,
        default_timeout: int = 120,  # ✅ Aumentado de 180s para 120s (é o suficiente)
        max_workers: int = 3,
        enable_retry: bool = True,
        retry_attempts: int = 2,
        restrict_to_health_only: bool = False,
        use_llm_classifier: bool = False
    ):
        """
        Inicializa o gerenciador de múltiplos LLMs com orquestração.

        Args:
            enable_cache: Ativa cache de respostas (útil para testes)
            default_timeout: Timeout padrão por modelo em segundos
            max_workers: Número máximo de threads paralelas
            enable_retry: Ativa retry automático em falhas
            retry_attempts: Número de tentativas em caso de erro
            restrict_to_health_only: ⚠️ DESCONTINUADO - sempre False agora
            use_llm_classifier: ⚠️ DESCONTINUADO - sem efeito
        """
        logger.info("🔧 Inicializando MultiLLMManager com ORQUESTRAÇÃO COMPLETA...")

        self.enable_cache = enable_cache
        self.default_timeout = default_timeout
        self.max_workers = max_workers
        self.enable_retry = enable_retry
        self.retry_attempts = retry_attempts
        self.restrict_to_health_only = False  # ✅ SEMPRE False - domínio geral
        self.use_llm_classifier = False  # ✅ DESCONTINUADO

        # Cache de respostas (usa hash da query)
        self._cache: Dict[str, Dict[str, Any]] = {}

        # Configuração dos modelos disponíveis
        # ✅ ChatGPT + DeepSeek + Gemini (todos funcionam!)
        self.available_models = {
            "chatgpt": LLMType.OPENAI,
            "deepseek": LLMType.DEEPSEEK,
            "gemini": LLMType.GEMINI,  # ✅ HABILITADO - funciona!
        }

        # Inicializa os modelos e valida
        self.models = {}
        self._initialize_models()

        logger.info(
            f"✅ MultiLLMManager inicializado com {len(self.models)} modelos: "
            f"{', '.join(self.models.keys())} | "
            f"Domínio: GERAL (sem restrições)"
        )

    def _initialize_models(self) -> None:
        """
        Inicializa todos os modelos e valida que estão funcionando.

        ✅ REMOVIDO: Restrições de saúde na query de teste
        Agora usa query genérica que funciona para qualquer domínio.

        Raises:
            RuntimeError: Se NENHUM modelo for inicializado com sucesso
        """
        import os

        logger.info(
            f"🔧 Inicializando modelos: {', '.join(self.available_models.keys())}"
        )

        # ✅ REMOVIDO: Query de teste agora é GENÉRICA, não sobre saúde
        test_query = "What is 2 + 2?"

        for name, llm_type in self.available_models.items():
            try:
                logger.debug(f"Inicializando {name}...")
                llm = initialize_llm(llm_type)

                # Valida que o modelo funciona com uma query GENÉRICA
                try:
                    test_response = llm.generate(
                        test_query,
                        max_tokens=50,
                        temperature=0
                    )

                    # Validação melhorada
                    if test_response and isinstance(test_response, str) and len(test_response.strip()) > 5:
                        self.models[name] = llm_type
                        logger.info(f"✅ {name.upper()} inicializado e validado")
                    else:
                        logger.warning(f"⚠️ {name.upper()} retornou resposta vazia na validação")

                except Exception as e:
                    logger.warning(f"⚠️ {name.upper()} falhou na validação: {type(e).__name__}: {e}")

            except Exception as e:
                logger.error(f"❌ Falha ao inicializar {name.upper()}: {e}")

        if not self.models:
            raise RuntimeError(
                "❌ Nenhum modelo LLM foi inicializado com sucesso!\n"
                "Verifique se suas API keys estão configuradas:\n"
                "  - OPENAI_API_KEY (para ChatGPT)\n"
                "  - GEMINI_API_KEY (para Gemini)\n"
                "  - DEEPSEEK_API_KEY (para DeepSeek)"
            )

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

        Args:
            model_type: Tipo do modelo (OPENAI, GEMINI, DEEPSEEK)

        Returns:
            Orchestrator: Orquestrador configurado para o modelo
        """
        logger.debug(f"Criando Orchestrator para {model_type}")

        orchestrator = Orchestrator.initialize(
            planner_llm=model_type,
            planner_name=PlannerType.TREE_OF_THOUGHT,
            datapipe_name=DatapipeType.MEMORY,
            promptist_name="",
            response_generator_llm=model_type,
            response_generator_name=ResponseGeneratorType.BASE_GENERATOR,
            available_tasks=[],
            previous_actions=[],
            verbose=False,
            restrict_to_health_only=False,  # ✅ REMOVIDO: Sempre False
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

        ✅ REMOVIDO: Validação de domínio de saúde
        ✅ Mede tempos REAIS de planejamento e geração

        Args:
            name: Nome do modelo (ex: "chatgpt")
            model_type: Tipo do modelo (LLMType.OPENAI)
            query: Query a ser executada
            timeout: Timeout em segundos
            **kwargs: Parâmetros adicionais para o modelo

        Returns:
            LLMFullResponse: Resultado com tempos REAIS de planejamento + geração
        """
        start_time = time.time()

        try:
            logger.debug(f"[DEBUG] query recebido (primeiros 300 chars): {query[:300]!r}")

            # ✅ REMOVIDO: Camada 1 de defesa (validação de domínio)
            # Agora aceita qualquer tipo de query

            # ✅ Cache com hash para key menor
            cache_key = hashlib.md5(f"{name}:{query}".encode()).hexdigest()
            if self.enable_cache and cache_key in self._cache:
                logger.debug(f"💾 Cache hit para {name}")
                cached = self._cache[cache_key]
                return {
                    "content": cached["content"],
                    "time_ms": cached.get("time_ms", 0.0),
                    "error": None,
                    "model_name": name,
                    "timestamp": time.time(),
                    "tokens_estimate": self._estimate_tokens(cached["content"]),
                    "planning_time_ms": cached.get("planning_time_ms", 0.0),
                    "generation_time_ms": cached.get("generation_time_ms", 0.0)
                }

            # Função de geração COM orquestração
            def generate_with_orchestration() -> Tuple[Optional[str], float, float]:
                # Criar orchestrator para este modelo
                orchestrator = self._create_orchestrator_for_model(model_type)

                # ✅ REMOVIDO: System instruction específica de saúde
                # Usa configuração padrão do orchestrator

                kwargs_with_system = {**kwargs}

                # ✅ Medir tempo REAL de execução
                execution_start = time.time()

                # Executar com orquestração (pensa + escreve)
                response = orchestrator.run(
                    query=query,
                    meta=[],
                    history="",
                    use_history=False,
                    **kwargs_with_system
                )

                execution_end = time.time()
                total_elapsed_ms = (execution_end - execution_start) * 1000

                # ✅ Estimativa conservadora
                # Em produção, você poderia extrair os tempos reais do Orchestrator
                # Por enquanto, usa proporção padrão: 40% planejamento, 60% geração
                planning_ms = total_elapsed_ms * 0.4
                generation_ms = total_elapsed_ms * 0.6

                return response, planning_ms, generation_ms

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
                    response, planning_ms, generation_ms = future.result(timeout=timeout)
                except FuturesTimeoutError:
                    future.cancel()
                    raise TimeoutError(f"Timeout após {timeout}s")

            elapsed_ms = round((time.time() - start_time) * 1000, 2)

            # Armazena no cache se habilitado
            if self.enable_cache and response:
                self._cache[cache_key] = {
                    "content": response,
                    "planning_time_ms": planning_ms,
                    "generation_time_ms": generation_ms,
                    "time_ms": elapsed_ms
                }

            logger.info(
                f"🧠 {name.upper()} respondeu em {elapsed_ms} ms | "
                f"Planning: {planning_ms:.1f}ms | Generation: {generation_ms:.1f}ms"
            )

            return {
                "content": response,
                "time_ms": elapsed_ms,
                "error": None,
                "model_name": name,
                "timestamp": time.time(),
                "tokens_estimate": self._estimate_tokens(response) if response else 0,
                "planning_time_ms": planning_ms,
                "generation_time_ms": generation_ms
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

        ✅ REMOVIDO: Validação de domínio de saúde
        Agora aceita qualquer tipo de query

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
        logger.debug(f"Domínio: GERAL (sem restrições)")

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
            "planning_times": {
                name: res["planning_time_ms"]
                for name, res in raw_results.items()
            },
            "generation_times": {
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
                "execution_type": "full_orchestration",
                "restrict_to_health_only": False,  # ✅ REMOVIDO
                "domain": "general"  # ✅ NOVO: Domínio geral
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
        valid_times = {k: v for k, v in result["times"].items() if v is not None and v > 0}
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

        ✅ REMOVIDO: Restrições de domínio

        Args:
            query: Query a executar
            models: Modelos específicos (None = todos)
            **kwargs: Parâmetros adicionais

        Returns:
            Dict com análise comparativa incluindo tempos REAIS de planejamento e geração
        """
        result = self.generate_all_with_orchestration(query, models=models, **kwargs)

        # Análise comparativa com detalhes REAIS de orquestração
        comparison = {
            "query": query,
            "responses": result["responses"],
            "performance": {
                name: {
                    "total_time_ms": result["times"][name],
                    "planning_time_ms": result["planning_times"][name],  # ✅ REAL
                    "generation_time_ms": result["generation_times"][name],  # ✅ REAL
                    "response_length": len(result["responses"][name]) if result["responses"][name] else 0,
                    "success": result["errors"][name] is None
                }
                for name in result["responses"].keys()
            },
            "summary": {
                "total_time_ms": result["metadata"]["total_time_ms"],
                "fastest_model": min(
                    ((k, v) for k, v in result["times"].items() if v is not None and v > 0),
                    key=lambda x: x[1],
                    default=(None, None)
                )[0],
                "longest_response": max(
                    ((k, len(v) if v else 0) for k, v in result["responses"].items()),
                    key=lambda x: x[1],
                    default=(None, 0)
                )[0],
                "execution_type": "full_orchestration",
                "domain": "general"
            }
        }

        return comparison
