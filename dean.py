import numpy as np
import main
from flaml import tune
import time
import json
from sklearn.metrics import roc_auc_score
import glob
import os


#load data, and change the shape into (samples, features)
from loaddata import loaddata    
(x_train0, y_train), (x_test0, y_test) = loaddata()
if len(x_train0.shape)>2:
    x_train0=np.reshape(x_train0,(x_train0.shape[0],np.prod(x_train0.shape[1:])))
    x_test0 =np.reshape(x_test0 ,(x_test0.shape[0],np.prod(x_test0.shape[1:])))


def train_one(**hyper):
    y_true,y_score = main.DEAN(**hyper)

    return roc_auc_score(y_true,y_score)

def optimization(config: dict):
    t0 = time.time()
    auc = train_one(**config)
    t1 = time.time()

    return {'score': auc, 'evaluation_cost': t1-t0}

#(index, bag, lr, depth, batch, rounds, pwr)

hyperparameters = {'index': tune.randint(lower=0, upper=9), 
                    'bag': tune.randint(lower =1, upper= x_train0.shape[1]),
                    'lr': tune.uniform(lower = 0.01, upper= 0.05),
                    'depth': tune.randint(lower = 1, upper = 5),
                    'batch': tune.randint(lower= 50, upper = 500),
                    'rounds': tune.randint(lower = 50, upper = 150),
                    'pwr': tune.randint(lower = 0, upper = 2)
                    }



analysis = tune.run(
    optimization,  # the function to evaluate a config
    config= hyperparameters,  # the search space defined
    metric="score",
    mode="max",  # the optimization mode, "min" or "max"
    num_samples= 100,  # the maximal number of configs to try, -1 means infinite
    )


print('best model hyperparameters', analysis.best_config)  # the best config
print('best roc auc score is:  ', analysis.best_trial.last_result['score'])
print('time_s', analysis.best_trial.last_result['evaluation_cost'])  # the best trial's result'


with open('best_hyperparameters.json', 'w', encoding='utf-8') as f:
    json.dump(analysis.best_config, f, ensure_ascii=False, indent=4)

with open('best_auc_score.json', 'w', encoding='utf-8') as f:
    json.dump(analysis.best_trial.last_result, f, ensure_ascii=False, indent=4)
