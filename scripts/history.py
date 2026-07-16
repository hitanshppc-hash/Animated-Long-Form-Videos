import json
from pathlib import Path

from utils import get_logger

logger = get_logger(__name__)

DEFAULT_HISTORY_PATH = "history.json"


def load_titles(history_path: str = DEFAULT_HISTORY_PATH) -> list:
    path = Path(history_path)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text()).get("titles", [])
    except (json.JSONDecodeError, OSError):
        logger.warning(f"Could not read {history_path}, starting fresh")
        return []


def append_title(title: str, history_path: str = DEFAULT_HISTORY_PATH) -> None:
    if not title:
        return
    titles = load_titles(history_path)
    if title in titles:
        return
    titles.append(title)
    Path(history_path).write_text(json.dumps({"titles": titles}, indent=2))
    logger.info(f"Recorded title in {history_path}: {title!r}")
