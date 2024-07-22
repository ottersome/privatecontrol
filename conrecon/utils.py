import logging
import os


def create_logger(name: str) -> logging.Logger:
    # Check if .log folder exists if ot crea
    if not os.path.exists(f"logs/"):
        os.makedirs(f"logs/", exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(f"logs/{name}.log", mode="w")
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# Creat decorator with dates and reason saying why a function is deprecated
def deprecated(reason, date):
    def decorator(func):
        def wrapper(*args, **kwargs):
            raise NotImplementedError(
                f"{func.__name__} has deprecated since {date} and will be removed in the future.\n"
                f"Reason: {reason}\n"
            )

        return wrapper

    return decorator
