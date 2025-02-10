import logging
import inspect
import os

class Logger:
    def __init__(self, name="Logger", level=logging.DEBUG, log_file=None, console_level=None, file_level=None):
        """
        Custom Logger class.

        :param name: Logger name
        :param level: Default logging level for the Logger
        :param log_file: File path for logging, if None, no file logging
        :param console_level: Logging level for the console, defaults to Logger's level
        :param file_level: Logging level for the file, defaults to Logger's level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        # self.set_level(level)  # Set the default logging level for the Logger

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s ')

        # Clear existing handlers to avoid duplicate logs
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Add console handler
        if console_level is None:
            console_level = level  # Default to Logger's level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # Add file handler
        if log_file:
            if file_level is None:
                file_level = level  # Default to Logger's level
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(file_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    # def set_level(self, level):
    #     """Dynamically change the logging level of the Logger."""
    #     self.logger.setLevel(level)

    def log(self, level, message):
        """Log a message with additional caller information."""
        caller_frame = inspect.stack()[2]
        filename = os.path.basename(caller_frame.filename)
        line_number = caller_frame.lineno
        self.logger.log(level, f"[{filename}:{line_number}] {message}")

    def debug(self, message):
        self.log(logging.DEBUG, message)

    def info(self, message):
        self.log(logging.INFO, message)

    def warning(self, message):
        self.log(logging.WARNING, message)

    def error(self, message):
        self.log(logging.ERROR, message)

    def critical(self, message):
        self.log(logging.CRITICAL, message)

logger = Logger(
    name="root",  # Logger name
    level=logging.WARNING,  # Default Logger level
    log_file="langgfm.log",  # File path for logging
    console_level=logging.WARNING,  # Console logging level
    file_level=logging.WARNING  # File logging level
)
