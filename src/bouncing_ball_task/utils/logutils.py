"""Place for configuring the top level logger."""
import logging
import sys

from loguru import logger
from rich.logging import RichHandler

from bouncing_ball_task import index


class InterceptHandler(logging.Handler):
    """Intercepts the standard logger and sends it to loguru.

    Taken from the
    documentation:
    https://github.com/Delgan/loguru#entirely-compatible-with-standard-logging
    """

    def emit(self, record):
        # Get corresponding Loguru level if it exists.
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message.
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def configure_logger(
    packages=None, verbose=False, trace=False, max_size="10 MB"
):
    logging.basicConfig(handlers=[InterceptHandler()], level=0)

    if packages:
        for pkg in packages:
            try:
                logger_pkg = logging.getLogger(pkg)
                for handler in logger_pkg.handlers.copy():
                    logger_pkg.removeHandler(handler)
                logger_pkg.addHandler(logging.NullHandler())
                logger_pkg.propagate = False
                logger_pkg.addHandler(InterceptHandler())
            except Exception as exc:
                logger.warning(
                    f"Package {pkg} encountered the following exception: {exc},"
                    "skipping."
                )

    log_dir = index.dir_logs / "loguru"
    log_dir.mkdir(parents=True, exist_ok=True)
    level = "INFO"

    handlers = [
        {
            "sink": str(log_dir / "info.log"),
            "level": "INFO",
            "format": "{message}",
            "rotation": max_size,
        },
        {
            "sink": str(log_dir / "warning.log"),
            "level": "WARNING",
            "format": "{message}",
            "rotation": max_size,
        },
    ]

    if verbose:
        handlers.append(
            {
                "sink": str(log_dir / "debug.log"),
                "level": "DEBUG",
                "format": "{message}",
                "rotation": max_size,
            }
        )
        level = "DEBUG"

    if trace:
        handlers.append(
            {
                "sink": str(log_dir / "trace.log"),
                "level": "TRACE",
                "format": "{message}",
                "rotation": max_size,
            }
        )
        level = "TRACE"

    handlers.append(
        {
            "sink": RichHandler(
                log_time_format="[%X]",
                omit_repeated_times=False,
                markup=True,
            ),
            "format": "{message}",
            "level": level,
        }
    )

    logger.configure(handlers=handlers)

    return logger
