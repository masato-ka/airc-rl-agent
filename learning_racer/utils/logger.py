import sys
from logging import getLogger, StreamHandler, Formatter, DEBUG, INFO, WARNING, ERROR, CRITICAL
from optparse import OptionValueError


def get_logger(name, level=INFO):
    logger = getLogger(name)
    logger.setLevel(level)
    handler = StreamHandler()
    handler.setFormatter(Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.propaget = False
    return logger


# 関心ごとに集中するために、AOP的に例外処理を外出ししたいとロギング処理を集約したい。
def teardown_exception_wrapper(logger=None):
    def _teardown_exception_wrapper(func):
        """
        Decorator for logging
        :param func: function to be decorated
        :param logger: logger object
        :return: decorated function
        """

        def wrapper(*args, **kwargs):
            logger.debug('Starting %s', func.__name__)
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                logger.error(':{}:{}'.format(func.__name__, e))
                sys.exit(-1)
            logger.debug('Finishing %s', func.__name__)
            return result

        return wrapper

    return _teardown_exception_wrapper
