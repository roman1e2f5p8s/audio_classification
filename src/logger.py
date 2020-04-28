import os
import logging
from datetime import datetime


class Logger:
    '''
    Creates logging
    '''
    def __init__(self, logs_dir, verbose=False):
        '''
        Initialisation
        Arguments:
            - logs_dir -- path to log directory, str
            - verbose -- whether print out to console, bool. Defaults to False
        '''

        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)

        self.logs_dir = logs_dir
        self.verbose = verbose
        self.log_filename = datetime.now().strftime('%d_%b_%Y_%H_%M_%S_%f')[:-3] + '.log'
        self.log_path = os.path.join(logs_dir, self.log_filename)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(
                filename=self.log_path,
                filemode='w',
                format='|%(asctime)s.%(msecs)03d --- %(filename)s [line:%(lineno)4d] %(levelname)s: '
                    '%(message)s',
                datefmt='%d-%b-%Y %H:%M:%S',
                level=logging.DEBUG)

        # Print also some info to console 
        if verbose:
            console = logging.StreamHandler()
            console.setLevel(logging.INFO)
            formatter = logging.Formatter('|%(asctime)s.%(msecs)03d --- %(levelname)s: %(message)s',
                    datefmt='%d-%b-%Y %H:%M:%S')
            console.setFormatter(formatter)
            logging.getLogger('').addHandler(console)

        self.logger = logging
