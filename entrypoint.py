from models.create_model import predict
from conf.conf import logging, settings
import argparse

# Creating argparse to run predictions from the command line
parser = argparse.ArgumentParser(description='Args for models predictions')
parser.add_argument('params', type=str, help='Sample params to get predictions')
args = parser.parse_args()

# Create the list from enetered prediction params
prediction_params = [[int(item) for item in args.params.split(',')]]

pred = predict(prediction_params, model=settings.MODEL.gb)
logging.info(f"Prediction - {pred}")