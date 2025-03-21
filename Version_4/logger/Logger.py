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