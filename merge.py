#combines and evaluates all ensemble submodels into one auc score

import os
import numpy as np

import sys

from simplestat import statinf
import json

pwr=0
quiet=False
if len(sys.argv)>1:
    try:
        pwr=float(sys.argv[1])
    except:
        quiet=sys.argv[1]=="quiet"

cmax=100000
if len(sys.argv)>2:
    cmax=int(sys.argv[2])

if not os.path.isdir("results"):exit()

fns=["results/"+zw+"/result.npz" for zw in os.listdir("results")][:cmax+10]
fns=[zw for zw in fns if os.path.isfile(zw)]

from sklearn.metrics import roc_auc_score as auc


y_scores=[]
y_true=None

for fn in fns:
    f=np.load(fn,allow_pickle=True)
    if y_true is None:
        y_true=f["y_true"]

    y_scores.append(f["y_score"])

#fs=[np.load(fn,allow_pickle=True) for fn in fns]


#screw double evaluation
#y_true=fs[0]["y_true"]
#le=int(2*(len(y_true)-sum(y_true)))
#y_true=y_true[:le]
#y_scores=[f["y_score"][:le] for f in fs]

y_scores=y_scores[:cmax]


aucs=[auc(y_true,y_score) for y_score in y_scores]
if not quiet:print(json.dumps([[statinf(aucs)]],indent=2))

wids=[np.std(y_score[np.where(y_true==0)]) for y_score in y_scores]

y_score=np.sqrt(np.mean([(y_score/wid**pwr)**2 for y_score,wid in zip(y_scores,wids)],axis=0))


auc_score=auc(y_true,y_score)

if not quiet:print("----------",auc_score)


np.savez_compressed("auc.npz",auc=auc_score,aucs=aucs,wids=wids)




