from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from openCHA.CustomDebugFormatter import CustomDebugFormatter
from openCHA.datapipes import DataPipe, DatapipeType, initialize_datapipe
from openCHA.llms import LLMType
from openCHA.orchestrator import Action  # mantém compatibilidade com seu projeto
from openCHA.planners import BasePlanner, initialize_planner
from openCHA.response_generators import (
    BaseResponseGenerator,
    ResponseGeneratorType,
    initialize_response_generator,
)
from openCHA.tasks import BaseTask, initialize_task
from pydantic import BaseModel


class Orchestrator(BaseModel):
    """
    The Orchestrator class is the main execution heart of the CHA.

    ✅ CORRIGIDO (sem exec):
    - NÃO executa código Python retornado pelo planner
    - Trata o retorno do planner como TEXTO ("thinker")
    - Mantém suporte a tasks (ex: google_translate) sem depender de exec()
    - Mantém guardrails REFUSE
    """

    planner: BasePlanner = None
    datapipe: DataPipe = None
    promptist: Any = None
    response_generator: BaseResponseGenerator = None
    available_tasks: Dict[str, BaseTask] = {}

    max_retries: int = 1
    max_task_execute_retries: int = 3
    max_planner_execute_retries: int = 16
    max_final_answer_execute_retries: int = 3

    role: int = 0
    verbose: bool = False

    planner_logger: Optional[logging.Logger] = None
    tasks_logger: Optional[logging.Logger] = None
    orchestrator_logger: Optional[logging.Logger] = None
    final_answer_generator_logger: Optional[logging.Logger] = None
    promptist_logger: Optional[logging.Logger] = None
    error_logger: Optional[logging.Logger] = None

    previous_actions: List[str] = []
    current_actions: List[str] = []
    runtime: Dict[str, bool] = {}

    class Config:
        arbitrary_types_allowed = True

    def print_log(self, log_name: str, message: str):
        if not self.verbose:
            return

        if log_name == "planner" and self.planner_logger:
            self.planner_logger.debug(message)
        elif log_name == "task" and self.tasks_logger:
            self.tasks_logger.debug(message)
        elif log_name == "orchestrator" and self.orchestrator_logger:
            self.orchestrator_logger.debug(message)
        elif log_name == "response_generator" and self.final_answer_generator_logger:
            self.final_answer_generator_logger.debug(message)
        elif log_name == "promptist" and self.promptist_logger:
            self.promptist_logger.debug(message)
        elif log_name == "error" and self.error_logger:
            self.error_logger.debug(message)

    @classmethod
    def initialize(
        self,
        planner_llm: str = LLMType.OPENAI,
        planner_name: str = "",
        datapipe_name: str = DatapipeType.MEMORY,
        promptist_name: str = "",
        response_generator_llm: str = LLMType.OPENAI,
        response_generator_name: str = ResponseGeneratorType.BASE_GENERATOR,
        available_tasks: Optional[List[str]] = None,
        previous_actions: List[Action] = None,
        verbose: bool = False,
        **kwargs,
    ) -> "Orchestrator":
        if available_tasks is None:
            available_tasks = []
        if previous_actions is None:
            previous_actions = []

        planner_logger = tasks_logger = orchestrator_logger = None
        final_answer_generator_logger = promptist_logger = error_logger = None

        if verbose:
            planner_logger = CustomDebugFormatter.create_logger("Planner", "cyan")
            tasks_logger = CustomDebugFormatter.create_logger("Task", "purple")
            orchestrator_logger = CustomDebugFormatter.create_logger("Orchestrator", "green")
            final_answer_generator_logger = CustomDebugFormatter.create_logger("Response Generator", "blue")
            promptist_logger = CustomDebugFormatter.create_logger("Promptist", "blue")
            error_logger = CustomDebugFormatter.create_logger("Error", "red")

        datapipe = initialize_datapipe(datapipe=datapipe_name, **kwargs)
        if verbose and orchestrator_logger:
            orchestrator_logger.debug(f"Datapipe {datapipe_name} initialized.\n")

        tasks: Dict[str, BaseTask] = {}
        for task in available_tasks:
            kwargs["datapipe"] = datapipe
            tasks[task] = initialize_task(task=task, **kwargs)
            if verbose and orchestrator_logger:
                orchestrator_logger.debug(f"Task '{task}' initialized.")

        planner = initialize_planner(
            tasks=list(tasks.values()),
            llm=planner_llm,
            planner=planner_name,
            **kwargs,
        )
        if verbose and orchestrator_logger:
            orchestrator_logger.debug(f"Planner {planner_name} initialized.")

        response_generator = initialize_response_generator(
            response_generator=response_generator_name,
            llm=response_generator_llm,
            **kwargs,
        )
        if verbose and orchestrator_logger:
            orchestrator_logger.debug(f"Response Generator {response_generator_name} initialized.")

        return self(
            planner=planner,
            datapipe=datapipe,
            promptist=None,
            response_generator=response_generator,
            available_tasks=tasks,
            verbose=verbose,
            previous_actions=previous_actions,
            current_actions=[],
            planner_logger=planner_logger,
            tasks_logger=tasks_logger,
            orchestrator_logger=orchestrator_logger,
            final_answer_generator_logger=final_answer_generator_logger,
            promptist_logger=promptist_logger,
            error_logger=error_logger,
        )

    def process_meta(self) -> bool:
        return False

    def _update_runtime(self, action: Action = None):
        if action and getattr(action, "output_type", None):
            self.runtime[action.task_response] = False
        if action:
            for task_input in action.task_inputs:
                if task_input in self.runtime:
                    self.runtime[task_input] = True

    def execute_task(self, task_name: str, task_inputs: List[str]) -> Any:
        """
        Executa uma task diretamente (sem depender de exec()).
        Útil para tasks internas como translate, extract, etc.
        """
        self.print_log(
            "task",
            f"---------------\nExecuting task:\nTask Name: {task_name}\nTask Inputs: {task_inputs}\n",
        )
        try:
            task = self.available_tasks[task_name]
            result = task.execute(task_inputs)

            self.print_log("task", f"Task executed successfully\nResult: {result}\n---------------\n")

            action = Action(
                task_name=task_name,
                task_inputs=task_inputs,
                task_response=result,
                output_type=getattr(task, "output_type", None),
                datapipe=self.datapipe,
            )

            self._update_runtime(action)
            self.previous_actions.append(action)
            self.current_actions.append(action)

            return result

        except Exception as e:
            self.print_log("error", f"Error running task: \n{e}\n---------------\n")
            logging.exception(e)
            raise ValueError(
                f"Error executing task {task_name}: {str(e)}\n\nTry again with different inputs."
            )

    def planner_generate_prompt(self, query: str) -> str:
        return query

    def response_generator_generate_prompt(
        self,
        final_response: str = "",
        history: str = "",
        meta: List[str] = None,
        use_history: bool = False,
    ) -> str:
        if meta is None:
            meta = []

        prompt = "MetaData: {meta}\n\nHistory: \n{history}\n\n"
        if use_history:
            prompt = prompt.replace("{history}", history)
        else:
            prompt = prompt.replace("{history}", "")

        prompt = prompt.replace("{meta}", ", ".join(meta)) + f"\n{final_response}"
        return prompt

    def plan(self, query, history, meta, use_history, **kwargs) -> str:
        """
        ✅ ALTERAÇÃO: O planner retorna TEXTO (thinker), não código python.
        """
        return self.planner.plan(
            query,
            history,
            meta,
            self.previous_actions,
            use_history,
            **kwargs,
        )

    def generate_final_answer(self, query, thinker, **kwargs) -> str:
        """
        ✅ Mantém guardrail de recusa.
        """
        if thinker and ("REFUSE:" in str(thinker)[:200] or "Desculpe, posso responder apenas" in str(thinker)):
            self.print_log("response_generator", f"Planner recusou: {str(thinker)[:120]}...")
            logging.warning(f"Domain restriction detected: {str(thinker)[:200]}")
            return (
                "Desculpe, posso responder apenas a perguntas sobre saúde, medicina, "
                "bem-estar, nutrição, fitness e saúde mental. "
                "Por favor, faça uma pergunta relacionada a esses tópicos!"
            )

        retries = 0
        while retries < self.max_final_answer_execute_retries:
            try:
                prefix = kwargs.get("response_generator_prefix_prompt", "")
                return self.response_generator.generate(
                    query=query,
                    thinker=thinker,
                    prefix=prefix,
                    **kwargs,
                )
            except Exception as e:
                retries += 1
                self.print_log("error", f"Response generator error: {e}")
        return "We currently have a problem processing your question. Please try again after a while."

    def run(
        self,
        query: str,
        meta: List[str] = None,
        history: str = "",
        use_history: bool = False,
        **kwargs: Any,
    ) -> str:
        """
        ✅ ALTERAÇÃO PRINCIPAL: remove exec(actions) e pipeline passa a ser textual.
        """
        if meta is None:
            meta = []

        # 1) Armazena arquivos/meta no datapipe
        meta_infos = ""
        for meta_data in meta:
            key = self.datapipe.store(meta_data)
            meta_infos += (
                f"The file with the name ${meta_data.split('/')[-1]}$ is stored with the key $datapipe:{key}$."
                "Pass this key to the tools when you want to send them over to the tool\n"
            )

        # 2) Prompt inicial
        prompt = self.planner_generate_prompt(query)

        # 3) Tradução opcional via task (sem exec)
        source_language = None
        if "google_translate" in self.available_tasks:
            try:
                translated = self.available_tasks["google_translate"].execute([prompt, "en"])
                source_language = translated[1]
                prompt = translated[0]
            except Exception as e:
                self.print_log("error", f"Translate error (prompt): {e}")

        # 4) Planejamento (agora retorna TEXTO)
        self.print_log("planner", "Planning Started...\n")

        i = 0
        thinker_text = ""
        while True:
            try:
                self.print_log("planner", f"Continuing Planning... Try number {i}\n\n")

                thinker_text = self.plan(
                    query=prompt,
                    history=history,
                    meta=meta_infos,
                    use_history=use_history,
                    **kwargs,
                )

                # ✅ ALTERAÇÃO: valida recusa pelo texto do planner
                if "REFUSE:" in str(thinker_text) or "Desculpe, posso responder apenas" in str(thinker_text):
                    return (
                        "Desculpe, posso responder apenas a perguntas sobre saúde, medicina, "
                        "bem-estar, nutrição, fitness e saúde mental. "
                        "Por favor, faça uma pergunta relacionada a esses tópicos!"
                    )

                break

            except (Exception, SystemExit) as error:
                self.print_log("error", f"Planning Error:\n{error}\n\n")
                i += 1
                if i > self.max_retries:
                    thinker_text = "Problem preparing the answer. Please try again."
                    break

        self.print_log("planner", f"Planner output (thinker_text): {str(thinker_text)[:500]}\nPlanning Ended...\n\n")

        # 5) Monta prompt para response generator
        final_prompt = self.response_generator_generate_prompt(
            final_response=str(thinker_text),
            history=history,
            meta=[meta_infos] if meta_infos else [],
            use_history=use_history,
        )

        self.print_log(
            "response_generator",
            f"Final Answer Generation Started...\nInput Prompt: \n\n{final_prompt}",
        )

        final_response = self.generate_final_answer(query=query, thinker=final_prompt, **kwargs)

        self.print_log(
            "response_generator",
            f"Response: {final_response}\n\nFinal Answer Generation Ended.\n",
        )

        # 6) Tradução de volta (se aplicável)
        if "google_translate" in self.available_tasks and source_language:
            try:
                final_response = self.available_tasks["google_translate"].execute(
                    [final_response, source_language]
                )[0]
            except Exception as e:
                self.print_log("error", f"Translate error (final): {e}")

        return final_response
