import logging


def init_logging(log_filename):
    """
        Init for logging
    """
    logging.basicConfig(
                    level    = logging.INFO,
                    format   = '%(asctime)s: %(message)s',
                    datefmt  = '%m-%d %H:%M:%S',
                    filename = log_filename,
                    filemode = 'w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%m-%d %H:%M:%S')
    # tell the handler to use this format
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)