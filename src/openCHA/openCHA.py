import os
import logging
from typing import List, Tuple, Dict, Any, Optional

from openCHA.datapipes import DatapipeType
from openCHA.interface import Interface
from openCHA.llms import LLMType
from openCHA.orchestrator import Orchestrator
from openCHA.planners import Action
from openCHA.planners import PlannerType
from openCHA.response_generators import ResponseGeneratorType
from openCHA.tasks import TASK_TO_CLASS
from openCHA.utils import parse_addresses
from pydantic import BaseModel, Field
from openCHA.llms.multi_llm_manager import MultiLLMManager

logger = logging.getLogger(__name__)


class openCHA(BaseModel):
    """
    Classe principal do framework openCHA para agentes conversacionais com IA.
    
    Recursos:
        - Orquestração de tarefas complexas
        - Planejamento com Tree of Thought
        - Geração de respostas contextualizadas
        - Suporte a múltiplos LLMs (ChatGPT, Gemini, DeepSeek)
        - Comparação paralela entre modelos
        - Interface de usuário integrada
        - Upload e processamento de arquivos
    
    Exemplos:
        >>> # Uso básico
        >>> cha = openCHA()
        >>> resposta = cha.run("Explique inteligência artificial")
        >>>
        >>> # Comparar múltiplos modelos
        >>> comparacao = cha.compare_llm_responses(
        ...     "Qual a capital do Brasil?",
        ...     models=['chatgpt', 'gemini', 'deepseek']
        ... )
        >>>
        >>> # Com interface gráfica
        >>> cha.run_with_interface()
    """
    
    name: str = "openCHA"
    previous_actions: List[Action] = Field(default_factory=list)
    orchestrator: Optional[Orchestrator] = None
    planner_llm: str = LLMType.OPENAI
    planner: str = PlannerType.TREE_OF_THOUGHT
    datapipe: str = DatapipeType.MEMORY
    promptist: str = ""
    response_generator_llm: str = LLMType.OPENAI
    response_generator: str = ResponseGeneratorType.BASE_GENERATOR
    meta: List[str] = Field(default_factory=list)
    verbose: bool = False
    
    # Multi-LLM Manager para comparação entre modelos
    multi_llm: Optional[MultiLLMManager] = None
    
    # Configurações do Multi-LLM
    multi_llm_enable_cache: bool = True
    multi_llm_timeout: int = 30
    multi_llm_max_workers: int = 3
    multi_llm_enable_retry: bool = True
    multi_llm_retry_attempts: int = 2

    class Config:
        """Configuração do Pydantic para permitir tipos arbitrários."""
        arbitrary_types_allowed = True

    def _generate_history(
        self, 
        chat_history: Optional[List[Tuple[str, str]]] = None
    ) -> str:
        """
        Gera uma string formatada do histórico de conversação.
        
        Args:
            chat_history: Lista de tuplas (user_message, cha_response)
        
        Returns:
            str: Histórico formatado como string
        """
        if chat_history is None:
            chat_history = []

        history = "".join(
            [
                f"\n------------\nUser: {chat[0]}\nCHA: {chat[1]}\n------------\n"
                for chat in chat_history
            ]
        )
        return history
    
    def get_multi_llm(self) -> MultiLLMManager:
        """
        Retorna uma instância de MultiLLMManager.
        Se ainda não existir, inicializa com as configurações da classe.
        
        Returns:
            MultiLLMManager: Gerenciador de múltiplos LLMs configurado
        """
        if self.multi_llm is None:
            logger.info("Inicializando MultiLLMManager...")
            self.multi_llm = MultiLLMManager(
                enable_cache=self.multi_llm_enable_cache,
                default_timeout=self.multi_llm_timeout,
                max_workers=self.multi_llm_max_workers,
                enable_retry=self.multi_llm_enable_retry,
                retry_attempts=self.multi_llm_retry_attempts,
            )
            logger.info("MultiLLMManager inicializado com sucesso")
        return self.multi_llm

    def compare_llm_responses(
        self,
        query: str,
        models: Optional[List[str]] = None,
        parallel: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compara respostas de múltiplos LLMs para a mesma query.
        
        Este método executa a query em ChatGPT, Gemini e DeepSeek simultaneamente
        e retorna as respostas, tempos de execução e possíveis erros.
        
        Args:
            query: Pergunta ou prompt a ser executado
            models: Lista de modelos específicos ['chatgpt', 'gemini', 'deepseek']
                   Se None, executa em todos os modelos disponíveis
            parallel: Se True, executa em paralelo; se False, sequencial
            **kwargs: Parâmetros adicionais como:
                - temperature (float): Criatividade do modelo (0-2)
                - max_tokens (int): Limite de tokens na resposta
                - top_p (float): Nucleus sampling
        
        Returns:
            Dict contendo:
                - responses: Dicionário com respostas de cada modelo
                - times: Tempos de execução em milissegundos
                - errors: Erros encontrados (None se sucesso)
                - metadata: Informações adicionais (total_time, success_count, etc)
        
        Exemplos:
            >>> cha = openCHA()
            >>> resultado = cha.compare_llm_responses("Explique IA")
            >>> print(resultado['responses']['chatgpt'])
            >>> print(f"Tempo: {resultado['times']['chatgpt']} ms")
            >>>
            >>> # Com parâmetros customizados
            >>> resultado = cha.compare_llm_responses(
            ...     "Escreva um poema",
            ...     models=['chatgpt', 'gemini'],
            ...     temperature=0.9,
            ...     max_tokens=500
            ... )
        """
        if not query or not query.strip():
            raise ValueError("Query não pode estar vazia")
        
        logger.info(f"Comparando respostas de LLMs para query: {query[:100]}...")
        
        manager = self.get_multi_llm()
        result = manager.generate_all(
            query=query,
            models=models,
            parallel=parallel,
            **kwargs
        )
        
        logger.info(
            f"Comparação concluída: {result['metadata']['success_count']} sucessos, "
            f"{result['metadata']['failed_count']} falhas"
        )
        
        return result
    
    def compare_and_analyze(
        self,
        query: str,
        models: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Executa comparação entre LLMs e retorna análise detalhada.
        
        Além das respostas, retorna métricas como modelo mais rápido,
        resposta mais longa, e comparações de performance.
        
        Args:
            query: Query a executar
            models: Modelos específicos ou None para todos
            **kwargs: Parâmetros adicionais
        
        Returns:
            Dict com análise comparativa completa
        """
        manager = self.get_multi_llm()
        return manager.compare_responses(query, models=models, **kwargs)

    def _run(
        self,
        query: str,
        chat_history: Optional[List[Tuple[str, str]]] = None,
        tasks_list: Optional[List[str]] = None,
        use_history: bool = False,
        **kwargs,
    ) -> str:
        """
        Executa a query usando o orchestrator principal.
        
        Args:
            query: Pergunta ou comando do usuário
            chat_history: Histórico de conversação
            tasks_list: Lista de tarefas disponíveis
            use_history: Se True, inclui histórico no contexto
            **kwargs: Parâmetros adicionais
        
        Returns:
            str: Resposta gerada pelo sistema
        """
        if chat_history is None:
            chat_history = []
        if tasks_list is None:
            tasks_list = []

        history = self._generate_history(chat_history=chat_history)

        # Inicializa orchestrator se necessário
        if self.orchestrator is None:
            logger.info("Inicializando Orchestrator...")
            self.orchestrator = Orchestrator.initialize(
                planner_llm=self.planner_llm,
                planner_name=self.planner,
                datapipe_name=self.datapipe,
                promptist_name=self.promptist,
                response_generator_llm=self.response_generator_llm,
                response_generator_name=self.response_generator,
                available_tasks=tasks_list,
                previous_actions=self.previous_actions,
                verbose=self.verbose,
                **kwargs,
            )
            logger.info("Orchestrator inicializado")

        response = self.orchestrator.run(
            query=query,
            meta=self.meta,
            history=history,
            use_history=use_history,
            **kwargs,
        )

        return response

    def respond(
        self,
        message: str,
        openai_api_key_input: str,
        serp_api_key_input: str,
        gemini_api_key_input: str,
        deepseek_api_key_input: str,
        chat_history: List[Tuple[str, str]],
        check_box: bool,
        tasks_list: List[str],
    ) -> Tuple[str, List[Tuple[str, str]]]:
        """
        Método de resposta usado pela interface gráfica.
        
        NOTA: Este método modifica os.environ globalmente, o que não é ideal
        para ambientes multi-usuário. Considere usar gerenciamento de credenciais
        mais seguro em produção.
        
        Args:
            message: Mensagem do usuário
            openai_api_key_input: API key da OpenAI
            serp_api_key_input: API key do SERP
            gemini_api_key_input: API key do Gemini
            deepseek_api_key_input: API key do DeepSeek
            chat_history: Histórico da conversa
            check_box: Flag para usar histórico
            tasks_list: Lista de tarefas disponíveis
        
        Returns:
            Tupla (mensagem_vazia, chat_history_atualizado)
        """
        # Configura API keys (ATENÇÃO: modifica ambiente global)
        os.environ["OPENAI_API_KEY"] = openai_api_key_input
        os.environ["SERP_API_KEY"] = serp_api_key_input  # Corrigido de SEPR
        os.environ["GEMINI_API_KEY"] = gemini_api_key_input
        os.environ["DEEPSEEK_API_KEY"] = deepseek_api_key_input
        
        try:
            response = self._run(
                query=message,
                chat_history=chat_history,
                tasks_list=tasks_list,
                use_history=check_box,
            )

            files = parse_addresses(response)

            if len(files) == 0:
                chat_history.append((message, response))
            else:
                # Processa arquivos na resposta
                for i in range(len(files)):
                    chat_history.append(
                        (
                            message if i == 0 else None,
                            response[: files[i][1]],
                        )
                    )
                    chat_history.append((None, (files[i][0],)))
                    response = response[files[i][2] :]

            return "", chat_history
            
        except Exception as e:
            error_msg = f"Erro ao processar mensagem: {str(e)}"
            logger.error(error_msg, exc_info=True)
            chat_history.append((message, f"❌ {error_msg}"))
            return "", chat_history

    def reset(self) -> None:
        """
        Reseta o estado do sistema, limpando histórico e ações anteriores.
        """
        logger.info("Resetando estado do openCHA...")
        self.previous_actions = []
        self.meta = []
        self.orchestrator = None  # Força reinicialização
        
        # Limpa cache do multi-LLM se existir
        if self.multi_llm is not None:
            self.multi_llm.clear_cache()
        
        logger.info("Estado resetado com sucesso")

    def run_with_interface(self) -> None:
        """
        Inicia a interface gráfica do openCHA.
        
        A interface permite interação via browser com upload de arquivos,
        seleção de tarefas e chat interativo.
        """
        logger.info("Iniciando interface gráfica...")
        available_tasks = [key.value for key in TASK_TO_CLASS.keys()]
        interface = Interface()
        interface.prepare_interface(
            respond=self.respond,
            reset=self.reset,
            upload_meta=self.upload_meta,
            available_tasks=available_tasks,
        )

    def upload_meta(
        self, 
        history: List[Tuple], 
        file: Any
    ) -> List[Tuple]:
        """
        Processa upload de arquivo e adiciona ao histórico.
        
        Args:
            history: Histórico atual
            file: Objeto de arquivo uploaded
        
        Returns:
            Histórico atualizado com o arquivo
        """
        history = history + [((file.name,), None)]
        self.meta.append(file.name)
        logger.info(f"Arquivo uploaded: {file.name}")
        return history

    def run(
        self,
        query: str,
        chat_history: Optional[List[Tuple[str, str]]] = None,
        available_tasks: Optional[List[str]] = None,
        use_history: bool = False,
        use_multi_llm: bool = False,
        compare_models: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """
        Método principal para executar queries no openCHA.
        
        Args:
            query: Pergunta ou comando do usuário
            chat_history: Histórico de conversação anterior
            available_tasks: Lista de tarefas que o sistema pode usar
            use_history: Se True, inclui histórico no contexto
            use_multi_llm: Se True, usa comparação entre múltiplos LLMs
            compare_models: Modelos específicos para comparação
            **kwargs: Parâmetros adicionais
        
        Returns:
            str: Resposta gerada ou comparação formatada
        
        Examples:
            >>> cha = openCHA()
            >>> 
            >>> # Uso normal com orchestrator
            >>> resposta = cha.run("Qual é a capital do Brasil?")
            >>> 
            >>> # Com histórico
            >>> resposta = cha.run(
            ...     "E a população?",
            ...     chat_history=[("Qual é a capital do Brasil?", "Brasília")],
            ...     use_history=True
            ... )
            >>>
            >>> # Comparação entre modelos
            >>> resposta = cha.run(
            ...     "Explique computação quântica",
            ...     use_multi_llm=True,
            ...     compare_models=['chatgpt', 'gemini']
            ... )
        """
        if chat_history is None:
            chat_history = []
        if available_tasks is None:
            available_tasks = []

        try:
            # Modo de comparação multi-LLM
            if use_multi_llm:
                logger.info("Executando em modo multi-LLM comparison")
                results = self.compare_llm_responses(
                    query,
                    models=compare_models,
                    **kwargs
                )
                return self._format_multi_llm_results(results)
            
            # Modo normal com orchestrator
            return self._run(
                query=query,
                chat_history=chat_history,
                tasks_list=available_tasks,
                use_history=use_history,
                **kwargs,
            )
            
        except Exception as e:
            error_msg = f"Erro ao executar query: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return f"❌ {error_msg}"
    
    def _format_multi_llm_results(self, results: Dict[str, Any]) -> str:
        """
        Formata resultados da comparação multi-LLM em string legível.
        
        Args:
            results: Dicionário retornado por compare_llm_responses
        
        Returns:
            str: Resultados formatados
        """
        output_lines = [
            "=" * 80,
            "COMPARAÇÃO ENTRE MÚLTIPLOS LLMs",
            "=" * 80,
            ""
        ]
        
        # Informações gerais
        metadata = results['metadata']
        output_lines.extend([
            f"⏱️  Tempo total: {metadata['total_time_ms']} ms",
            f"✅ Sucessos: {metadata['success_count']} | ❌ Falhas: {metadata['failed_count']}",
            f"🔤 Tokens estimados: {metadata['total_tokens_estimate']}",
            ""
        ])
        
        # Respostas de cada modelo
        for model_name, response in results['responses'].items():
            time_ms = results['times'][model_name]
            error = results['errors'][model_name]
            
            output_lines.extend([
                f"{'=' * 80}",
                f"🤖 {model_name.upper()}",
                f"{'=' * 80}",
            ])
            
            if error:
                output_lines.append(f"❌ Erro: {error}")
            else:
                output_lines.extend([
                    f"⏱️  Tempo: {time_ms} ms",
                    f"📝 Resposta:",
                    f"{response}",
                ])
            
            output_lines.append("")
        
        # Identificar modelo mais rápido
        valid_times = {k: v for k, v in results['times'].items() if v is not None}
        if valid_times:
            fastest = min(valid_times.items(), key=lambda x: x[1])
            output_lines.extend([
                f"{'=' * 80}",
                f"🏆 Modelo mais rápido: {fastest[0].upper()} ({fastest[1]} ms)",
                f"{'=' * 80}",
            ])
        
        return "\n".join(output_lines)
    
    def get_available_models(self) -> List[str]:
        """
        Retorna lista de modelos LLM disponíveis para comparação.
        
        Returns:
            List[str]: Nomes dos modelos disponíveis
        """
        manager = self.get_multi_llm()
        return manager.get_available_models()
    
    def clear_multi_llm_cache(self) -> None:
        """
        Limpa o cache do MultiLLMManager.
        Útil para forçar novas requisições aos modelos.
        """
        if self.multi_llm is not None:
            self.multi_llm.clear_cache()
            logger.info("Cache do MultiLLMManager limpo")
        else:
            logger.warning("MultiLLMManager não foi inicializado ainda")