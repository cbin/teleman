import numpy as np
import os
from telemanom._globals import Config
import logging
from datetime import datetime
import sys
import csv
import pandas as pd
import plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode
import cufflinks as cf
import glob

config = Config("config.yaml")

def make_dirs(_id):
    '''Создаем директорию для хранения данных используя ID'''

    if not config.train or not config.predict:
        if not os.path.isdir('data/%s' %config.use_id):
            raise ValueError("Неверный ID %s . Необходим ID.")

    paths = ['data', 'data/%s' %_id, 'data/%s/models' %_id, 'data/%s/smoothed_errors' %_id, 'data/%s/y_hat' %_id]

    for p in paths:
        if not os.path.isdir(p):
            os.mkdir(p)



def setup_logging(config, _id):

    logger =  logging.getLogger('telemanom')
    hdlr = logging.FileHandler('data/%s/params.log' %_id)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)

    stdout = logging.StreamHandler(sys.stdout)
    stdout.setLevel(logging.INFO)
    logger.addHandler(stdout)

    logger.info("Параметры запуска:")
    logger.info("----------------")
    for attr in dir(config):    
        if not "__" in attr and not attr in ['header', 'date_format', 'path_to_config', 'build_group_lookup']:
            logger.info('%s: %s' %(attr, getattr(config, attr)))
    logger.info("----------------\n")

    return logger



def load_data(anom):

    try:
        train = np.load(os.path.join("data", "train", anom['chan_id'] + ".npy"))
        test = np.load(os.path.join("data", "test", anom['chan_id'] + ".npy"))

    except:
        raise ValueError("Исходные данные не найдены необходимо загрузить: <link>")


    X_train, y_train = shape_data(train)
    X_test, y_test = shape_data(test, train=False)

    return X_train, y_train, X_test, y_test


def shape_data(arr, train=True):

    
    # print("LEN ARR: %s" %len(arr))

    data = [] 
    for i in range(len(arr) - config.l_s - config.n_predictions):
        data.append(arr[i:i + config.l_s + config.n_predictions])
    data = np.array(data) 

    assert len(data.shape) == 3

    if train == True:
        np.random.shuffle(data)

    X = data[:,:-config.n_predictions,:]
    y = data[:,-config.n_predictions:,0] #telemetry value is at position 0

    return X, y
    

def final_stats(stats, logger):


    logger.info("Final Totals:")
    logger.info("-----------------")
    logger.info("Верных положительных классификаций: %s " %stats["true_positives"])
    logger.info("Ложных положительных классификаций: %s " %stats["false_positives"])
    logger.info("Ложноотрицательных классификаций: %s\n" %stats["false_negatives"])
    try:
        logger.info("Точность: %s" %(float(stats["true_positives"])/float(stats["true_positives"]+stats["false_positives"])))
        logger.info("Полнота: %s" %(float(stats["true_positives"])/float(stats["true_positives"]+stats["false_negatives"])))
    except:
        logger.info("Точность: NaN")
        logger.info("Полнота: NaN")


def anom_stats(stats, anom, logger):


    logger.info("TP: %s  FP: %s  FN: %s" %(anom["true_positives"], anom["false_positives"], anom["false_negatives"]))
    logger.info('Всего верно положительных классификаций: %s' %stats["true_positives"])
    logger.info('Всего ложных положительных классификаций: %s' %stats["false_positives"])
    logger.info('Всего ложноотрицательных классификаций: %s\n' %stats["false_negatives"])



def view_results(results_fn, plot_errors=True, plot_train=False, rows=None):


    def create_shapes(ranges, range_type, _min, _max):

        if range_type == 'true':
            color = 'red'
        elif range_type == 'predicted':
            color = 'blue'
        
        shapes = []
        if len(ranges) > 0:
        
            for r in ranges:

                shape = {
                    'type': 'rect',
                    'x0': r[0],
                    'y0': _min,
                    'x1': r[1],
                    'y1': _max,
                    'fillcolor': color,
                    'opacity': 0.2,
                    'line': {
                        'width': 0,
                    },
                }
            
                shapes.append(shape)
            
        return shapes



    vals = {}

    with open(results_fn, "r") as f:
        reader = csv.DictReader(f)
        for anom in reader:

            chan = anom["chan_id"]
            vals[chan] = {}
            dirs = ["y_hat", "smoothed_errors"]
            raw_dirs = ["test", "train"]

            for d in dirs:
                if config.predict:
                    vals[chan][d] = list(np.load(os.path.join("../data", anom['run_id'], d, anom["chan_id"]) + ".npy"))
                else:
                    vals[chan][d] = list(np.load(os.path.join("../data", config.use_id, d, anom["chan_id"]) + ".npy"))
            for d in raw_dirs:
                vals[chan][d] = list(np.load(os.path.join("../data", d, anom["chan_id"]) + ".npy"))

            row_start = 0
            row_end = 100000
            if not rows == None:
                try:
                    row_start = rows[0]
                    row_end = rows[1]
                except:
                    raise ValueError("Неверный формат, используйте формат вида (<first row>, <last row>)")

            # Инфо
            # ================================================================================================
            if reader.line_num - 1 >= row_start and reader.line_num -1 <= row_end:
                print("Аппарат: (скрыто)")
                print("Канал: %s" %anom["chan_id"])
                print('Нормализованная ошибка прогнозирования: %.3f' %float(anom['normalized_error']))
                print('Аномальные классификации: %s' %anom['class'])
                print("------------------")
                print('Верных положительных классификаций: %s' %anom['true_positives'])
                print("Ложных положительных классификаций: %s" %anom["false_positives"])
                print("Ложноотрицательные классификации: %s" %anom["false_negatives"])
                print("------------------")
                print('Прогнозируемые оценки аномалий: %s' %anom['scores'])
                print("Число параметров: %s"%len(vals[chan]["test"]))

                # Извлечение значений телеметрии из тестового набора
                # ================================================================================================

                y_test = np.array(vals[chan]['test'])[:,0] 

                # Создаём подсвеченную область (red = true anoms / blue = predicted anoms)
                # ================================================================================================
                y_shapes = create_shapes(eval(anom['anomaly_sequences']), "true", -1, 1)
                y_shapes += create_shapes(eval(anom['tp_sequences']) + eval(anom['fp_sequences']), "predicted", -1, 1)

                e_shapes = create_shapes(eval(anom['anomaly_sequences']), "true", 0, max(vals[chan]['smoothed_errors']))
                e_shapes += create_shapes(eval(anom['tp_sequences']) + eval(anom['fp_sequences']), "predicted", 
                                          0, max(vals[chan]['smoothed_errors']))

                # Построение графиков Plotly
                # ================================================================================================
                train_df = pd.DataFrame({
                    'train': [x[0] for x in vals[chan]['train']]
                })

                y = y_test[config.l_s:-config.n_predictions]
                if not len(y) == len(vals[chan]['y_hat']):
                    modified_l_s = len(y_test) - len(vals[chan]['y_hat']) - 1
                    y = y_test[modified_l_s:-1]
                y_df = pd.DataFrame({
                    'y_hat': vals[chan]['y_hat'],
                    'y': y
                })

                e_df = pd.DataFrame({
                    'e_s': vals[chan]['smoothed_errors']
                })

                y_layout = {
                    'title': "y / y_hat comparison",
                    'shapes': y_shapes,
                } 

                e_layout = {
                    'title': "Smoothed Errors (e_s)",
                    'shapes': e_shapes,
                } 

                if plot_train:
                    train_df.iplot(kind='scatter', color='green')
                
                y_df.iplot(kind='scatter', layout=y_layout)
                
                if plot_errors:
                    e_df.iplot(kind='scatter', layout=e_layout, color='red')



