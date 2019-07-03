import logging


class Logger:
    """logging class"""

    def __init__(self, path):
        self.configurateLogger(path)


    def configurateLogger(self, path):
        """ configurates logger and safes terminal logging to file

        :param str path: safe path of logging file
        """
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            fileHandler = logging.FileHandler(path)
            fileHandler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
            logger.addHandler(fileHandler)

            streamHandler = logging.StreamHandler()
            streamHandler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(streamHandler)


    def logg(self, msg):
        logging.info(msg)