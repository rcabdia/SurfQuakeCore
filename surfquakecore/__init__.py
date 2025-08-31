import os
import logging

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(os.path.dirname(ROOT_DIR), "surfquakecore", "models", "190703-214543")
TT_DB_PATH = os.path.join(os.path.dirname(ROOT_DIR), "surfquakecore", "real", "tt_db", "mymodel.nd")
POLARITY_NETWORK = os.path.join(os.path.dirname(ROOT_DIR), "surfquakecore", "first_polarity", "Polarcap", "PolarCAP.h5")
FOC_MEC_BASH_PATH = os.path.join(os.path.dirname(ROOT_DIR), "surfquakecore", "first_polarity", "focmec_bash",
                                 "focmec_run")
nll_templates = os.path.join(os.path.dirname(ROOT_DIR), "surfquakecore", "earthquake_location/loc_structure/run")
nll_ak135 = os.path.join(os.path.dirname(ROOT_DIR), "surfquakecore", "earthquake_location/loc_structure/ak135")

def create_logger():

    # create logger.
    logger = logging.getLogger('logger')
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        # create console handler and set level to debug.
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # create file handler.
        file_log = logging.FileHandler(filename="app.log")
        file_log.setLevel(logging.INFO)

        # create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # add formatter to ch
        ch.setFormatter(formatter)
        file_log.setFormatter(formatter)

        # add ch and file_log to logger
        logger.addHandler(ch)
        logger.addHandler(file_log)

    return logger


app_logger = create_logger()