import json
import os
import logging
import dill
import pandas as pd
from datetime import datetime

path = os.path.expanduser('~/airflow_hw')


def predict():
    last_model = sorted(os.listdir(f'{path}/data/models'))[-1]
    with open(f'{path}/data/models/{last_model}', 'rb') as file:
        model = dill.load(file)
        test_list = os.listdir(f'{path}/data/test')
        preds = pd.DataFrame(columns=['id', 'prediction'])

    for car_id in test_list:
        with open(f'{path}/data/test/{car_id}', 'rb') as f:
            car_list = json.load(f)

        df = pd.DataFrame(car_list, index=[0])
        y = model.predict(df)
        dict_pred = {'id': df['id'].values[0], 'prediction': y[0]}
        df2 = pd.DataFrame([dict_pred])
        preds = pd.concat([df2, preds], ignore_index=True)

    date = datetime.now().strftime("%Y%m%d%H%M")
    preds.to_csv(f'{path}/data/predictions/preds_{date}.csv', index=False)
    logging.info(f'Prediction saved as preds_{date}.csv')


if __name__ == '__main__':
    predict()
