import logging

logger = logging.getLogger(__name__)
# handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s> %(message)s')
for handler in logger.handlers:
    handler.setFormatter(formatter)
# logger.addHandler(handler)
logger.setLevel(logging.INFO)

if __name__ == '__main__':
    logger.info('This is a test.')
