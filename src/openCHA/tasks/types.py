from typing import Dict
from typing import Type

from openCHA.tasks import AskUser
from openCHA.tasks import BaseTask
from openCHA.tasks import ExtractText
from openCHA.tasks import GoogleSearch
from openCHA.tasks import GoogleTranslate
from openCHA.tasks import RunPythonCode
from openCHA.tasks import SerpAPI
from openCHA.tasks import TaskType
from openCHA.tasks import TestFile

TASK_TO_CLASS: Dict[TaskType, Type[BaseTask]] = {
    TaskType.SERPAPI: SerpAPI,
    TaskType.EXTRACT_TEXT: ExtractText,
    TaskType.GOOGLE_TRANSLATE: GoogleTranslate,
    TaskType.ASK_USER: AskUser,
    TaskType.TEST_FILE: TestFile,
    TaskType.RUN_PYTHON_CODE: RunPythonCode,
    TaskType.GOOGLE_SEARCH: GoogleSearch,
}
