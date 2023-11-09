import logging
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(os.path.dirname(ROOT_DIR), "surfquakecore", "models", "190703-214543")
TT_DB_PATH = os.path.join(os.path.dirname(ROOT_DIR), "surfquakecore", "real", "tt_db", "mymodel.nd")