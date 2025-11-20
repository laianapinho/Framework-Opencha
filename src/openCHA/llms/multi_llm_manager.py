import time
import logging
import asyncio
from typing import Dict, Any, List, Optional, TypedDict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from functools import wraps
from openCHA.llms import initialize_llm, LLMType

logger = logging.getLogger(__name__)


class LLMResponse(TypedDict):
    """Estrutura tipada para resposta de um LLM individual."""
    content: Optional[str]
    time_ms: Optional[float]
    error: Optional[str]
    model_name: str
    timestamp: float
    tokens_estimate: Optional[int]


class MultiLLMResult(TypedDict):
    """Estrutura tipada para o resultado agregado."""
    responses: Dict[str, Optional[str]]
    times: Dict[str, Optional[float]]
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
    Gerenciador que executa queries em múltiplos LLMs de forma paralela e eficiente.
    
    Recursos:
        - Execução paralela com controle de timeout
        - Retry automático para erros recuperáveis
        - Cache opcional de respostas
        - Métricas detalhadas de performance
        - Configuração flexível de modelos
        - Validação de inicialização
    
    Exemplos:
        >>> manager = MultiLLMManager()
        >>> result = manager.generate_all("Explique IA")
        >>> print(result['responses']['chatgpt'])
        >>>
        >>> # Com modelos específicos
        >>> result = manager.generate_all(
        ...     "Explique IA",
        ...     models=['chatgpt', 'gemini'],
        ...     timeout=15
        ... )
        >>>
        >>> # Com parâmetros customizados
        >>> result = manager.generate_all(
        ...     "Escreva um poema",
        ...     temperature=0.9,
        ...     max_tokens=500
        ... )
    """
    
    def __init__(
        self,
        enable_cache: bool = False,
        default_timeout: int = 30,
        max_workers: int = 3,
        enable_retry: bool = True,
        retry_attempts: int = 2
    ):
        """
        Inicializa o gerenciador de múltiplos LLMs.
        
        Args:
            enable_cache: Ativa cache de respostas (útil para testes)
            default_timeout: Timeout padrão por modelo em segundos
            max_workers: Número máximo de threads paralelas
            enable_retry: Ativa retry automático em falhas
            retry_attempts: Número de tentativas em caso de erro
        """
        logger.info("🔧 Inicializando MultiLLMManager...")
        
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
                        self.models[name] = llm
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
        return len(text) // 4
    
    def _generate_with_model(
        self,
        name: str,
        llm: Any,
        query: str,
        timeout: int,
        **kwargs
    ) -> LLMResponse:
        """
        Executa geração em um modelo específico com controle de timeout.
        
        Args:
            name: Nome do modelo
            llm: Instância do LLM
            query: Query a ser executada
            timeout: Timeout em segundos
            **kwargs: Parâmetros adicionais para o modelo
            
        Returns:
            LLMResponse: Resultado da geração
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
                    "tokens_estimate": self._estimate_tokens(cached)
                }
            
            # Função de geração com ou sem retry
            def generate():
                return llm.generate(query, **kwargs)
            
            if self.enable_retry:
                generate_func = retry_on_failure(
                    max_retries=self.retry_attempts,
                    delay=1.0
                )(generate)
            else:
                generate_func = generate
            
            # Executa com timeout
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
            
            logger.info(f"🧠 {name.upper()} respondeu em {elapsed_ms} ms")
            
            return {
                "content": response,
                "time_ms": elapsed_ms,
                "error": None,
                "model_name": name,
                "timestamp": time.time(),
                "tokens_estimate": self._estimate_tokens(response) if response else 0
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
                "tokens_estimate": 0
            }
    
    def generate_all(
        self,
        query: str,
        models: Optional[List[str]] = None,
        timeout: Optional[int] = None,
        parallel: bool = True,
        **kwargs
    ) -> MultiLLMResult:
        """
        Executa a mesma query em múltiplos LLMs.
        
        Args:
            query: Pergunta ou prompt a ser executado
            models: Lista de modelos específicos (None = todos)
            timeout: Timeout por modelo em segundos (None = usar padrão)
            parallel: Se True, executa em paralelo; se False, sequencial
            **kwargs: Parâmetros adicionais (temperature, max_tokens, etc)
            
        Returns:
            MultiLLMResult: Dicionário com responses, times, errors e metadata
            
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
            f"🚀 Executando query em {len(selected_models)} modelo(s): "
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
                        self._generate_with_model,
                        name,
                        llm,
                        query,
                        timeout_value,
                        **kwargs
                    )
                    for name, llm in selected_models.items()
                }
                
                raw_results = {
                    name: future.result()
                    for name, future in futures.items()
                }
        else:
            raw_results = {
                name: self._generate_with_model(
                    name, llm, query, timeout_value, **kwargs
                )
                for name, llm in selected_models.items()
            }
        
        total_time_ms = round((time.time() - start_total) * 1000, 2)
        
        # Formata resultado no formato esperado
        result: MultiLLMResult = {
            "responses": {
                name: res["content"]
                for name, res in raw_results.items()
            },
            "times": {
                name: res["time_ms"]
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
                "timestamp": time.time()
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
    
    def compare_responses(
        self,
        query: str,
        models: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Executa query e retorna análise comparativa das respostas.
        
        Args:
            query: Query a executar
            models: Modelos específicos (None = todos)
            **kwargs: Parâmetros adicionais
            
        Returns:
            Dict com análise comparativa incluindo tamanhos, tempos, etc
        """
        result = self.generate_all(query, models=models, **kwargs)
        
        # Análise comparativa
        comparison = {
            "query": query,
            "responses": result["responses"],
            "performance": {
                name: {
                    "time_ms": result["times"][name],
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
                )[0]
            }
        }
        
        return comparison
