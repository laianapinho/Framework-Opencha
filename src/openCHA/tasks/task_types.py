from enum import Enum

class TaskType(str, Enum):
    SERPAPI = "serpapi"
    EXTRACT_TEXT = "extract_text"
    GOOGLE_TRANSLATE = "google_translate"
    ASK_USER = "ask_user"
    TEST_FILE = "test_file"
    RUN_PYTHON_CODE = "run_python_code"
    GOOGLE_SEARCH = "google_search"
