import os
import sys
from typing import Tuple
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
from configs.logger_config import log_file_path, error_log_file_path
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

error_logger = logging.getLogger("error_logger")
error_logger.setLevel(logging.ERROR)
error_handler = logging.FileHandler(error_log_file_path)
error_logger.addHandler(error_handler)

logger_handler = logging.FileHandler(log_file_path)
logger = logging.getLogger("logger")
logger.addHandler(logger_handler)

SCOPES = ["https://www.googleapis.com/auth/spreadsheets",
          "https://www.googleapis.com/auth/drive.readonly"]

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


