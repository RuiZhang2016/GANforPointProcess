# Read RData
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import json
from os import listdir, getcwd
from os.path import isfile, join
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
import numpy as np
import re

# Read all files in a directory
mypath = getcwd()+"/data/tweet_youtube"
onlyfiles = [ join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
tweets_time = {}
for f in onlyfiles:
    ro.r['load'](f,ro.globalenv)

shares = ro.globalenv[str('data')].rx2("numShare")
views = ro.globalenv[str('data')].rx2("dailyViewcount")

data = [ (shares[i][:120],views[i][:120])  for i in xrange(len(shares))]

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib import learn
from sklearn.metrics import mean_squared_error
from lstm import generate_data, lstm_model, load_csvdata
import dateutil.parser
import datetime
import matplotlib.dates as mdates

LOG_DIR = './ops_logs/lstm_popularity'
TIMESTEPS = 6
RNN_LAYERS = [{'num_units': 20}]
DENSE_LAYERS = [10,TIMESTEPS]
TRAINING_STEPS = 4000
BATCH_SIZE = 220
PRINT_STEPS = TRAINING_STEPS / 10

def loop_func(index):
    s_i = data[index][0]
    v_i = data[index][1]
    
    if(len(s_i)<120):
        return
        #         continue

    data_i = []
    for i in range(120):
        data_i.append([s_i[i]])
        data_i.append([v_i[i]])
    data_vs = pd.DataFrame(data_i, 
                           columns = ['data'])
    X, y = load_csvdata(data_vs, TIMESTEPS, seperate=False)
    
    regressor = learn.Estimator(model_fn=lstm_model(TIMESTEPS, RNN_LAYERS, DENSE_LAYERS,\
                                               optimizer = "Adagrad"),
                                model_dir=LOG_DIR)
    validation_monitor = learn.monitors.ValidationMonitor(X['val'], y['val'],
                                                          every_n_steps=PRINT_STEPS,
                                                          early_stopping_rounds=100)
    regressor.fit(X['train'], y['train'],
                  monitors=[validation_monitor],
                  batch_size=BATCH_SIZE,
                  steps=TRAINING_STEPS)

    predicted = 0
    pred = []
    test_x = X['test'][0]
    for ii in range(len(X['test'])):
        if(ii == 0):
            pass
        else:
            test_x = np.append(test_x, X['test'][ii][-1])
            test_x = np.reshape(test_x[2:],(-1,1))
        predicted = regressor.predict(np.array([test_x]))
        for p in predicted:
            pred.append(p)
            test_x = np.append(test_x,[p])
    pred_pop = sum(pred)+sum(v_i[:90])
    with open("results3.txt", "a") as myfile:
        myfile.write(str((index,pred_pop))+"\n")

result_f = open("results3.txt","r")
lines = result_f.readlines()
lines = [ l.strip() for l in lines]
existing_index = [ int(re.split("[\(,\) ]+", l)[1]) for l in lines]
rest_index = [ index for index in range(len(shares)) if index not in existing_index]

from joblib import Parallel, delayed
pred = Parallel(n_jobs=15)(delayed(loop_func)(index) \
                                      for index in rest_index)