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
    Classe principal do openCHA - Sistema de IA com Orquestração Completa e Multi-LLM.

    ✅ OTIMIZADO: Usa TreeOfThoughtPlanner com timeout aumentado

    Decide se vai rodar um agente simples (modo normal) ou uma comparação complexa
    entre vários agentes (modo Multi-LLM).

    Recursos:
        - Modo Normal: Um agente com orquestração completa (Tree of Thought)
        - Modo Multi-LLM: Comparação entre múltiplos modelos (ChatGPT, Gemini, DeepSeek)
        - Interface gráfica integrada (Gradio)
        - Suporte a upload de arquivos
        - Histórico de conversação
        - Cache inteligente
        - Retry automático em falhas
        - Timeout aumentado para TreeOfThought

    Exemplos:
        >>> # Modo Normal
        >>> agent = openCHA()
        >>> response = agent.run(query="Explique IA", use_multi_llm=False)
        >>>
        >>> # Modo Multi-LLM
        >>> response = agent.run(
        ...     query="Explique IA",
        ...     use_multi_llm=True,
        ...     compare_models=["chatgpt", "gemini"]
        ... )
        >>>
        >>> # Interface Gráfica
        >>> agent.run_with_interface()
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

    multi_llm: Optional[MultiLLMManager] = None

    multi_llm_enable_cache: bool = True
    multi_llm_timeout: int = 500
    multi_llm_max_workers: int = 3
    multi_llm_enable_retry: bool = True
    multi_llm_retry_attempts: int = 2

    class Config:
        arbitrary_types_allowed = True

    def _generate_history(
        self,
        chat_history: Optional[List[Tuple[str, str]]] = None
    ) -> str:
        """
        Formata a lista de mensagens [('oi', 'olá')] em um texto único para a IA ler.

        Args:
            chat_history: Lista de tuplas (mensagem_usuario, resposta_agente)

        Returns:
            str: Histórico formatado como texto
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
        PADRÃO SINGLETON (Iniciação Preguiçosa):
        Só cria o MultiLLMManager se ele ainda não existir.
        Isso economiza memória se o usuário só quiser usar o modo simples.

        Returns:
            MultiLLMManager: Instância do gerenciador de múltiplos modelos
        """
        if self.multi_llm is None:
            logger.info("Inicializando MultiLLMManager COM ORQUESTRAÇÃO COMPLETA...")
            self.multi_llm = MultiLLMManager(
                enable_cache=self.multi_llm_enable_cache,
                default_timeout=self.multi_llm_timeout,
                max_workers=self.multi_llm_max_workers,
                enable_retry=self.multi_llm_enable_retry,
                retry_attempts=self.multi_llm_retry_attempts,
            )
            logger.info("MultiLLMManager inicializado com sucesso")
        return self.multi_llm

    def compare_llm_responses_full(
        self,
        query: str,
        models: Optional[List[str]] = None,
        parallel: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Método Wrapper: Pega o pedido do usuário e repassa para o MultiLLMManager.
        É aqui que a mágica da comparação acontece.

        Args:
            query: Pergunta ou comando a ser processado
            models: Lista de modelos específicos (None = todos disponíveis)
            parallel: Se True, executa em paralelo; se False, sequencial
            **kwargs: Parâmetros extras (temperature, max_tokens, etc)

        Returns:
            Dict[str, Any]: Dicionário com estrutura MultiLLMResultFull contendo:
                - responses: Dict com respostas de cada modelo
                - times: Dict com tempos de execução
                - planning_times: Dict com tempos de planejamento
                - generation_times: Dict com tempos de geração
                - errors: Dict com erros (se houver)
                - metadata: Estatísticas agregadas

        Raises:
            ValueError: Se a query estiver vazia
        """
        if not query or not query.strip():
            raise ValueError("Query não pode estar vazia")

        logger.info(f"Comparando respostas (COM ORQUESTRAÇÃO TREE OF THOUGHT) para query: {query[:100]}...")

        manager = self.get_multi_llm()

        result = manager.generate_all_with_orchestration(
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

    def compare_and_analyze_full(
        self,
        query: str,
        models: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Versão mais detalhada da comparação, retornando estatísticas de tempo
        e análise comparativa entre modelos.

        Args:
            query: Pergunta ou comando
            models: Lista de modelos específicos (None = todos)
            **kwargs: Parâmetros extras

        Returns:
            Dict[str, Any]: Dicionário com análise comparativa incluindo:
                - query: Query original
                - responses: Respostas dos modelos
                - performance: Métricas detalhadas por modelo
                - summary: Resumo comparativo
        """
        manager = self.get_multi_llm()
        return manager.compare_responses_with_orchestration(query, models=models, **kwargs)

    def compare_llm_responses(
        self,
        query: str,
        models: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Wrapper simplificado para uso na Interface Gráfica (UI).
        Sempre executa em modo paralelo para melhor performance.

        Este método é chamado pelo respond() quando o usuário ativa o Multi-LLM na UI.
        É uma versão "amigável" do compare_llm_responses_full() que sempre usa
        as melhores configurações para interface gráfica.

        Args:
            query: Pergunta a ser processada
            models: Lista de modelos a comparar (None = todos disponíveis)
            **kwargs: Parâmetros extras (temperature, max_tokens, etc)

        Returns:
            Dict[str, Any]: Dicionário com estrutura MultiLLMResultFull

        Raises:
            ValueError: Se query estiver vazia ou modelos inválidos

        Exemplos:
            >>> agent = openCHA()
            >>> results = agent.compare_llm_responses(
            ...     query="Explique Machine Learning",
            ...     models=["chatgpt", "gemini"]
            ... )
            >>> print(results['responses']['chatgpt'])
            >>> print(results['times']['chatgpt'])
        """
        logger.debug(f"compare_llm_responses() chamado para query: {query[:50]}...")

        return self.compare_llm_responses_full(
            query=query,
            models=models,
            parallel=True,
            **kwargs
        )

    def _run(
        self,
        query: str,
        chat_history: Optional[List[Tuple[str, str]]] = None,
        tasks_list: Optional[List[str]] = None,
        use_history: bool = False,
        **kwargs,
    ) -> str:
        """
        MODO CLÁSSICO (Single Agent):
        Executa a lógica original do openCHA para um único agente.

        ✅ OTIMIZADO: Usa TreeOfThoughtPlanner (único disponível)

        Args:
            query: Pergunta ou comando
            chat_history: Histórico da conversa
            tasks_list: Lista de ferramentas/tarefas disponíveis
            use_history: Se True, usa contexto da conversa anterior
            **kwargs: Parâmetros extras

        Returns:
            str: Resposta do agente
        """
        if chat_history is None:
            chat_history = []
        if tasks_list is None:
            tasks_list = []

        history = self._generate_history(chat_history=chat_history)

        if self.orchestrator is None:
            logger.info("Inicializando Orchestrator com TreeOfThoughtPlanner...")
            logger.info("⏱️  AVISO: Tree of Thought pode levar 5-30 segundos para responder")
            logger.info("   Isso é NORMAL! O sistema está pensando em múltiplas estratégias.")

            self.orchestrator = Orchestrator.initialize(
                planner_llm=self.planner_llm,
                planner_name=PlannerType.TREE_OF_THOUGHT,
                datapipe_name=self.datapipe,
                promptist_name=self.promptist,
                response_generator_llm=self.response_generator_llm,
                response_generator_name=self.response_generator,
                available_tasks=tasks_list,
                previous_actions=self.previous_actions,
                verbose=self.verbose,
                **kwargs,
            )
            logger.info("Orchestrator inicializado com sucesso")

        response = self.orchestrator.run(
            query=query,
            meta=self.meta,
            history=history,
            use_history=use_history,
            **kwargs,
        )

        return response

    def run_single_question(self, query: str) -> Tuple[str, float]:
        """
        Roda UMA pergunta com ORQUESTRAÇÃO completa.
        Retorna resposta + tempo em ms.

        Usado pelo benchmark para testar individual cada modelo.

        Args:
            query: A pergunta médica

        Returns:
            Tuple[str, float]: (resposta_completa, tempo_em_ms)
        """
        import time
        start = time.time()

        response = self._run(
            query=query,
            chat_history=[],
            tasks_list=[],
            use_history=False
        )

        elapsed = (time.time() - start) * 1000
        return response, elapsed

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
        use_multi_llm: bool = False,
        compare_models: Optional[List[str]] = None,
    ) -> Tuple[str, List[Tuple[str, str]]]:
        """
        Callback para Interface Gráfica (UI).
        Recebe as chaves de API da tela e configura o ambiente.
        Agora suporta tanto modo normal quanto Multi-LLM!

        Args:
            message: Mensagem do usuário
            openai_api_key_input: API key da OpenAI
            serp_api_key_input: API key do SERP
            gemini_api_key_input: API key do Gemini
            deepseek_api_key_input: API key do DeepSeek
            chat_history: Histórico da conversa
            check_box: Flag para usar histórico
            tasks_list: Lista de tarefas disponíveis
            use_multi_llm: Se True, ativa modo de comparação entre LLMs
            compare_models: Lista de modelos a comparar (ex: ['chatgpt','gemini'])

        Returns:
            Tuple[str, List[Tuple[str, str]]]: Tupla (mensagem_limpa, chat_history_atualizado)
        """
        os.environ["OPENAI_API_KEY"] = openai_api_key_input
        os.environ["SERP_API_KEY"] = serp_api_key_input
        os.environ["GEMINI_API_KEY"] = gemini_api_key_input
        os.environ["DEEPSEEK_API_KEY"] = deepseek_api_key_input

        try:
            if use_multi_llm:
                logger.info("🌐 Respond: modo Multi-LLM ativado")
                logger.info(f"Modelos selecionados: {compare_models}")

                results = self.compare_llm_responses(
                    query=message,
                    models=compare_models if compare_models else None,
                )

                response = self._format_multi_llm_results(results)

            else:
                logger.info("🤖 Respond: modo Normal (single agent) com TreeOfThought")
                response = self._run(
                    query=message,
                    chat_history=chat_history,
                    tasks_list=tasks_list,
                    use_history=check_box,
                )

                # 🔍 DEBUG
                print(f"\n{'='*60}")
                print(f"🔍 DEBUG - Query: {message}")
                print(f"Response (primeiros 200 chars): {response[:200]}")
                print(f"Tem 'Desculpe'?: {'Desculpe' in response}")
                print(f"{'='*60}\n")

            files = parse_addresses(response)

            if len(files) == 0:
                chat_history.append((message, response))
            else:
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
        Limpa tudo para começar do zero.
        Reseta o orchestrator, histórico de ações e cache do Multi-LLM.
        """
        logger.info("Resetando estado do openCHA...")
        self.previous_actions = []
        self.meta = []
        self.orchestrator = None

        if self.multi_llm is not None:
            self.multi_llm.clear_cache()

        logger.info("Estado resetado com sucesso")

    def run_with_interface(self) -> None:
        """
        Lança a interface visual (Gradio).
        Configura todos os callbacks e inicia o servidor web.
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

    def upload_meta(self, history: List[Tuple], file: Any) -> List[Tuple]:
        """
        Lida com upload de arquivos na UI.

        Args:
            history: Histórico atual do chat
            file: Arquivo enviado pelo usuário

        Returns:
            List[Tuple]: Histórico atualizado com o arquivo
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
        O NOVO PONTO DE ENTRADA PRINCIPAL.
        Decide se roda o modo normal ou o modo de comparação (Multi-LLM).

        ✅ OTIMIZADO: Usa TreeOfThoughtPlanner com timeout aumentado

        Args:
            query: Pergunta ou comando
            chat_history: Histórico da conversa
            available_tasks: Lista de tarefas/ferramentas disponíveis
            use_history: Se True, usa contexto anterior
            use_multi_llm: Se True, ativa comparação entre múltiplos modelos
            compare_models: Lista de modelos a comparar
            **kwargs: Parâmetros extras

        Returns:
            str: Resposta formatada (texto simples para modo normal,
                 relatório comparativo para Multi-LLM)

        Examples:
            >>> # Modo Normal
            >>> agent = openCHA()
            >>> response = agent.run(
            ...     query="Explique IA",
            ...     use_multi_llm=False
            ... )
            >>>
            >>> # Modo Multi-LLM
            >>> response = agent.run(
            ...     query="Explique IA",
            ...     use_multi_llm=True,
            ...     compare_models=["chatgpt", "gemini"]
            ... )
        """
        if chat_history is None:
            chat_history = []
        if available_tasks is None:
            available_tasks = []

        try:
            if use_multi_llm:
                logger.info("🌐 Executando em MODO COMPARAÇÃO COM ORQUESTRAÇÃO TREE OF THOUGHT")
                logger.info(f"Modelos: {compare_models if compare_models else 'todos disponíveis'}")

                results = self.compare_llm_responses_full(
                    query,
                    models=compare_models,
                    **kwargs
                )
                return self._format_multi_llm_results(results)

            else:
                logger.info("🤖 Executando em MODO NORMAL (single agent com TreeOfThought)")
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
        Transforma o JSON de resultados em um relatório de texto legível.
        Exibe tempos de planejamento e execução separadamente.

        Args:
            results: Dicionário com estrutura MultiLLMResultFull

        Returns:
            str: Relatório formatado em texto
        """
        output_lines = [
            "=" * 80,
            "COMPARAÇÃO ENTRE MÚLTIPLOS LLMs (COM ORQUESTRAÇÃO TREE OF THOUGHT)",
            "=" * 80,
            ""
        ]

        metadata = results['metadata']
        output_lines.extend([
            f"⏱️  Tempo total: {metadata['total_time_ms']} ms",
            f"✅ Sucessos: {metadata['success_count']} | ❌ Falhas: {metadata['failed_count']}",
            f"🔤 Tokens estimados: {metadata['total_tokens_estimate']}",
            f"🧠 Tipo de execução: {metadata['execution_type']}",
            ""
        ])

        for model_name, response in results['responses'].items():
            time_ms = results['times'][model_name]
            planning_time = results['planning_times'][model_name]
            generation_time = results['generation_times'][model_name]
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
                    f"⏱️  Tempo total: {time_ms} ms",
                    f"  ├─ 🧠 Planejamento: {planning_time:.1f} ms",
                    f"  └─ ✍️  Geração: {generation_time:.1f} ms",
                    f"📝 Resposta:",
                    f"{response}",
                ])

            output_lines.append("")

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
        Helper para saber quais modelos estão disponíveis e funcionando.

        Returns:
            List[str]: Lista de nomes dos modelos disponíveis
        """
        manager = self.get_multi_llm()
        return manager.get_available_models()

    def clear_multi_llm_cache(self) -> None:
        """
        Limpa cache especificamente do MultiLLM.
        Útil quando você quer respostas frescas mesmo para queries repetidas.
        """
        if self.multi_llm is not None:
            self.multi_llm.clear_cache()
            logger.info("Cache do MultiLLMManager limpo")
        else:
            logger.warning("MultiLLMManager não foi inicializado ainda")
