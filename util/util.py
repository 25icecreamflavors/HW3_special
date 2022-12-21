import pickle
from conf.conf import logging, settings

def save_model(model, rf:bool = True) -> None:
    '''
    Function to save models.
    '''
    if rf:
        logging.info('Saving a random forest')
        pickle.dump(model, open(settings.MODEL.rf_path, 'wb'))
        logging.info('Successfully saved RF!')
    else:
        logging.info('saving gradient boosting')
        pickle.dump(model, open(settings.MODEL.gb_path, 'wb'))
        logging.info('Successfully saved GB!')

def load_model(path:str) -> None:
    '''
    Function for loading an existing model.
    '''
    logging.info('Loading model')
    clf = pickle.load(open(settings.MODEL.rf_path, 'rb'))
    logging.info('Your model is loaded!')
    return clf