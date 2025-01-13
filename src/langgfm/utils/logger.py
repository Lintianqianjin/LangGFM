import logging
import inspect
import os

class Logger:
    def __init__(self, name="Logger", level=logging.DEBUG, log_file=None):
        self.logger = logging.getLogger(name)
        self.set_level(level)  # 统一设置 level
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # 清除已有的 Handler，防止日志重复
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # console logger
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # log file
        if log_file:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def set_level(self, level):
        """ 动态修改日志级别 """
        self.logger.setLevel(level)

    def log(self, level, message):
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


logger = Logger(level=logging.INFO)
