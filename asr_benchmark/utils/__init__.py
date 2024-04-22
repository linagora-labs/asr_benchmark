import logging

logging.basicConfig(
    filemode="w",
    filename="debug.log",
    level=logging.INFO,
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(
    __name__,
)