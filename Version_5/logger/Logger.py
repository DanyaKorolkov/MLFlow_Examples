import logging
import sys
import io

class LoggingStream(io.StringIO):
    def __init__(self, logger, level):
        super().__init__()
        self.logger = logger
        self.level = level

    def write(self, message):
        message = message.strip()
        if message:
            self.logger.log(self.level, message)

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
)
logger = logging.getLogger("mlflow_logger")
mlflow_logging_stream = LoggingStream(logger, logging.INFO)
sys.stdout = mlflow_logging_stream
sys.stderr = mlflow_logging_stream