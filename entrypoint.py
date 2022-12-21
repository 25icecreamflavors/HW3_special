from models.create_model import predict
from conf.conf import logging, settings

pred = predict(settings.PREDICT.test, model=settings.MODEL.gb)
logging.info(f"Prediction - {pred}")