"""
Heavily borrowed from langchain: https://github.com/langchain-ai/langchain/
"""
import re
from typing import Any
from typing import List

from openCHA.planners import Action
from openCHA.planners import BasePlanner
from openCHA.planners import PlanFinish


class TreeOfThoughtPlanner(BasePlanner):
    """
    **Description:**

        This class implements Tree of Thought planner, which inherits from the BasePlanner base class.
        Tree of Thought employs parallel chain of thoughts startegies and decides which one is more
        suitable to proceed to get to the final answer.
        `Paper <https://arxiv.org/abs/2305.10601>`_

        This code defines a base class called "BasePlanner" that inherits from the "BaseModel" class of the pydantic library.
        The BasePlanner class serves as a base for implementing specific planners.

        ✅ MODIFICADO: Não executa código Python, apenas descreve estratégias em texto
        ✅ CORRIGIDO: Bug do stop token (index == -1)
        ✅ OTIMIZADO: Planejamento MÍNIMO para respostas rápidas

    """

    summarize_prompt: bool = True
    max_tokens_allowed: int = 10000

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @property
    def _planner_type(self):
        return "zero-shot-react-planner"

    @property
    def _planner_model(self):
        return self.llm_model

    @property
    def _response_generator_model(self):
        return self.llm_model

    @property
    def _stop(self) -> List[str]:
        return ["Wait"]

    @property
    def _shorten_prompt(self):
        return (
            "Summarize the following text. Make sure to keep the main ideas "
            "and objectives in the summary. Keep the links "
            "exactly as they are: "
            "{chunk}"
        )

    @property
    def _planner_prompt(self):
        return [
            """You are a health assistant. Answer the user's health question directly and clearly.
Do NOT generate Python code or function calls.
Provide accurate, helpful information in plain language.

Question: {input}

Answer:""",
            """Based on your answer above, provide a complete, well-structured response to the user's health question.

Question: {input}

Answer:""",
        ]

    def task_descriptions(self):
        return "".join(
            [
                (
                    "\n-----------------------------------\n"
                    f"**{task.name}**: {task.description}"
                    "\nThis tool have the following outputs:\n"
                    + "\n".join(task.outputs)
                    + (
                        "\n- The result of this tool will be stored in the datapipe."
                        if task.output_type
                        else ""
                    )
                    + "\n-----------------------------------\n"
                )
                for task in self.available_tasks
            ]
        )

    def divide_text_into_chunks(
        self,
        input_text: str = "",
        max_tokens: int = 10000,
    ) -> List[str]:
        """
        Generate a response based on the input prefix, query, and thinker (task planner).

        Args:
            input_text (str): the input text (e.g., prompt).
            max_tokens (int): Maximum number of tokens allowed.
        Return:
            chunks(List): List of string variables
        """
        # 1 token ~= 4 chars in English
        chunks = [
            input_text[i : i + max_tokens * 4]
            for i in range(0, len(input_text), max_tokens * 4)
        ]
        return chunks

    def generate_scratch_pad(
        self, previous_actions: List[str] = None, **kwargs: Any
    ):
        if previous_actions is None:
            previous_actions = []

        agent_scratchpad = ""
        if len(previous_actions) > 0:
            agent_scratchpad = "\n".join(
                [f"\n{action}" for action in previous_actions]
            )
        # agent_scratchpad
        if (
            self.summarize_prompt
            and len(agent_scratchpad) / 4 > self.max_tokens_allowed
        ):
            # Shorten agent_scratchpad
            chunks = self.divide_text_into_chunks(
                input_text=agent_scratchpad,
                max_tokens=self.max_tokens_allowed,
            )
            agent_scratchpad = ""
            kwargs["max_tokens"] = min(
                2000, int(self.max_tokens_allowed / len(chunks))
            )
            for chunk in chunks:
                prompt = self._shorten_prompt.replace(
                    "{chunk}", chunk
                )
                chunk_summary = (
                    self._response_generator_model.generate(
                        query=prompt, **kwargs
                    )
                )
                agent_scratchpad += chunk_summary + " "

    def plan(
        self,
        query: str,
        history: str = "",
        meta: str = "",
        previous_actions: List[str] = None,
        use_history: bool = False,
        **kwargs: Any,
    ) -> str:
        """
            Generate a plan using Tree of Thought - OTIMIZADO

        Args:
            query (str): Input query.
            history (str): History information.
            meta (str): meta information.
            previous_actions (List[Action]): List of previous actions.
            use_history (bool): Flag indicating whether to use history.
            **kwargs (Any): Additional keyword arguments.
        Return:
            str: return action in text format.

        ✅ OTIMIZADO: Planejamento MÍNIMO
        ✅ CORRIGIDO: Bug do stop token (index == -1)
        """
        if previous_actions is None:
            previous_actions = []

        previous_actions_prompt = ""
        if len(previous_actions) > 0 and self.use_previous_action:
            previous_actions_prompt = f"Previous Actions:\n{self.generate_scratch_pad(previous_actions, **kwargs)}"

        # ✅ PRIMEIRO PROMPT: PLANEJAMENTO MÍNIMO
        prompt = (
            self._planner_prompt[0]
            .replace("{input}", query)
        )

        print(prompt)
        kwargs["max_tokens"] = 500

        response = self._planner_model.generate(
            query=prompt, **kwargs
        )

        # ✅ SEGUNDO PROMPT: RESPOSTA FINAL
        prompt = (
            self._planner_prompt[1]
            .replace("{input}", query)
        )

        print("prompt2\n\n", prompt)
        kwargs["stop"] = self._stop
        response = self._planner_model.generate(
            query=prompt, **kwargs
        )

        index = min([response.find(text) for text in self._stop])

        # ✅ CORRIGIDO: Verificar se index != -1 antes de cortar
        if index != -1:
            response = response[0:index]

        # ✅ PARSE
        final_response = self.parse(response)

        return final_response

    def parse(
        self,
        query: str,
        **kwargs: Any,
    ) -> str:
        """
            Parse the output query into a clean text response (NOT Python code).

            ✅ MODIFICADO: Extrai resposta em texto, remove qualquer código Python

        Args:
            query (str): The planner output query to process.
            **kwargs (Any): Additional keyword arguments.
        Return:
            str: Clean text response without any Python code.

        """
        response = query.strip()

        # Remove ```python ... ``` se a IA ainda gerar código
        pattern = r"```python\n(.*?)```"
        if re.search(pattern, response, re.DOTALL):
            response = re.sub(r"```python\n", "", response)
            response = re.sub(r"```\n?", "", response)

        # Remove qualquer markdown de código genérico
        response = re.sub(r"```[a-zA-Z]*\n", "", response)
        response = re.sub(r"```\n?", "", response)

        # Remove chamadas de self.execute_task se ainda existirem
        if "self.execute_task" in response:
            response = re.sub(r"self\.execute_task\([^)]*\)", "", response)

        # Remove linhas com apenas "actions" ou "def " que indicam código
        lines = response.split("\n")
        filtered_lines = []
        for line in lines:
            if not line.strip().startswith(("def ", "class ", "import ", "from ")):
                filtered_lines.append(line)

        response = "\n".join(filtered_lines).strip()

        return response
