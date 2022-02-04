import numpy as np
import json
import os
import importlib

import argparse

import pandas as pd
import numpy as np
from collections import namedtuple
from sklearn.metrics import accuracy_score


class ConfusionMatrix(namedtuple('ConfusionMatrix', 'minority majority label truth')):
    def get_matrix(self):
        
        TP = np.logical_and(self.label == 1, self.truth == 1)
        FP = np.logical_and(self.label == 1, self.truth == 0)
        FN = np.logical_and(self.label == 0, self.truth == 1)
        TN = np.logical_and(self.label == 0, self.truth == 0)

        #maj
        TP_maj = np.logical_and(TP == 1, self.majority == 1)
        FP_maj = np.logical_and(FP == 1, self.majority == 1)
        FN_maj = np.logical_and(FN == 1, self.majority == 1)
        TN_maj = np.logical_and(TN == 1, self.majority == 1)

        nTP_maj = np.sum(TP_maj)
        nFN_maj = np.sum(FN_maj)
        nFP_maj = np.sum(FP_maj)
        nTN_maj = np.sum(TN_maj)

        nPPV_maj = float(nTP_maj) / max((nTP_maj + nFP_maj), 1)
        nTPR_maj = float(nTP_maj) / max((nTP_maj + nFN_maj), 1)

        nFDR_maj = float(nFP_maj) / max((nFP_maj + nTP_maj), 1)
        nFPR_maj = float(nFP_maj) / max((nFP_maj + nTN_maj), 1)

        nFOR_maj = float(nFN_maj) / max((nFN_maj + nTN_maj), 1)
        nFNR_maj = float(nFN_maj) / max((nFN_maj + nTP_maj), 1)

        nNPV_maj = float(nTN_maj) / max((nTN_maj + nFN_maj), 1)
        nTNR_maj = float(nTN_maj) / max((nTN_maj + nFP_maj), 1)

        #min
        TP_min = np.logical_and(TP == 1, self.minority == 1)
        FP_min = np.logical_and(FP == 1, self.minority == 1)
        FN_min = np.logical_and(FN == 1, self.minority == 1)
        TN_min = np.logical_and(TN == 1, self.minority == 1)

        
        nTP_min = np.sum(TP_min)
        nFN_min = np.sum(FN_min)
        nFP_min = np.sum(FP_min)
        nTN_min = np.sum(TN_min)

        nPPV_min = float(nTP_min) / max((nTP_min + nFP_min), 1)
        nTPR_min = float(nTP_min) / max((nTP_min + nFN_min), 1)

        nFDR_min = float(nFP_min) / max((nFP_min + nTP_min), 1)
        nFPR_min = float(nFP_min) / max((nFP_min + nTN_min), 1)

        nFOR_min = float(nFN_min) / max((nFN_min + nTN_min), 1)
        nFNR_min = float(nFN_min) / max((nFN_min + nTP_min), 1)

        nNPV_min = float(nTN_min) / max((nTN_min + nFN_min), 1)
        nTNR_min = float(nTN_min) / max((nTN_min + nFP_min), 1)

        matrix_maj = {
            'TP' : nTP_maj,
            'FP' : nFP_maj,
            'FN' : nFN_maj,
            'TN' : nTN_maj,
            'PPV' : nPPV_maj,
            'TPR' : nTPR_maj,
            'FDR' : nFDR_maj,
            'FPR' : nFPR_maj,
            'FOR' : nFOR_maj,
            'FNR' : nFNR_maj,
            'NPV' : nNPV_maj,
            'TNR' : nTNR_maj}

        matrix_min = {
            'TP' : nTP_min,
            'FP' : nFP_min,
            'FN' : nFN_min,
            'TN' : nTN_min,
            'PPV' : nPPV_min,
            'TPR' : nTPR_min,
            'FDR' : nFDR_min,
            'FPR' : nFPR_min,
            'FOR' : nFOR_min,
            'FNR' : nFNR_min,
            'NPV' : nNPV_min,
            'TNR' : nTNR_min}

        return matrix_min, matrix_maj

class Metric(namedtuple('Metric', 'cm_minority cm_majority')):
    def statistical_parity(self):
        statistical_parity_maj = float(self.cm_majority['TP'] + self.cm_majority['FP']) / max((self.cm_majority['TP'] + self.cm_majority['FP'] + self.cm_majority['FN'] + self.cm_majority['TN']), 1)
        statistical_parity_min = float(self.cm_minority['TP'] + self.cm_minority['FP']) / max((self.cm_minority['TP'] + self.cm_minority['FP'] + self.cm_minority['FN'] + self.cm_minority['TN']), 1)
        return np.fabs(statistical_parity_maj - statistical_parity_min)
    
    def predictive_parity(self):
        return np.fabs(self.cm_majority['PPV'] - self.cm_minority['PPV'])

    def predictive_equality(self):
        return np.fabs(self.cm_majority['FPR'] - self.cm_minority['FPR'])

    def equal_opportunity(self):
        return np.fabs(self.cm_majority['TPR'] - self.cm_minority['TPR'])

    def equalized_odds(self):
        return np.max([np.fabs(self.cm_majority['TPR'] - self.cm_minority['TPR']), np.fabs(self.cm_majority['FPR'] - self.cm_minority['FPR'])])

    def conditional_use_accuracy_equality(self):
        return np.max([np.fabs(self.cm_majority['PPV'] - self.cm_minority['PPV']), np.fabs(self.cm_majority['NPV'] - self.cm_minority['NPV'])])

    def fairness_metric(self, id):

        if id == 1:
            return self.statistical_parity()

        if id == 2:
            return self.predictive_parity()

        if id == 3:
            return self.predictive_equality()

        if id == 4:
            return self.equal_opportunity()

        if id == 5:
            return self.equalized_odds()

        if id == 6:
            return self.conditional_use_accuracy_equality()
       
parser = argparse.ArgumentParser()
parser.add_argument(
    "--strategy", default="uniform", help="Voting strategy, default uniform"
)
parser.add_argument("--container", help="Name of the container")
parser.add_argument("--shards", type=int, default=1, help="Number of shards, default 1")
parser.add_argument(
    "--dataset",
    default="datasets/purchase/datasetfile",
    help="Location of the datasetfile, default datasets/purchase/datasetfile",
)
parser.add_argument(
    "--baseline", type=int, help="Use only the specified shard (lone shard baseline)"
)
parser.add_argument("--data", default="compas",  help='german_credit,adult_income, compas, default_credit, marketing')
parser.add_argument("--per", type=int, default=1, help="Number of requests, default 1")
parser.add_argument("--rseed", default=0, type=int,  help="random seed")
parser.add_argument("--label", default="latest", help="Label, default latest")
args = parser.parse_args()

# Load dataset metadata.
with open(args.dataset) as f:
    datasetfile = json.loads(f.read())
dataloader = importlib.import_module(
    ".".join(args.dataset.split("/")[:-1] + [datasetfile["dataloader"]])
)

# Output files used for the vote.
if args.baseline != None:
    filenames = ["shard-{}:{}.npy".format(args.baseline, args.label)]
else:
    filenames = ["shard-{}:{}.npy".format(i, args.label) for i in range(args.shards)]

# Concatenate output files.
outputs = []
for filename in filenames:
    outputs.append(
        np.load(
            os.path.join("containerss/{}/{}/{}/{}/outputs".format(args.per,args.rseed,args.data, args.container), filename),
            allow_pickle=True,
        )
    )
outputs = np.array(outputs)

# Compute weight vector based on given strategy.
if args.strategy == "uniform":
    weights = (
        1 / outputs.shape[0] * np.ones((outputs.shape[0],))
    )  # pylint: disable=unsubscriptable-object
elif args.strategy.startswith("models:"):
    models = np.array(args.strategy.split(":")[1].split(",")).astype(int)
    weights = np.zeros((outputs.shape[0],))  # pylint: disable=unsubscriptable-object
    weights[models] = 1 / models.shape[0]  # pylint: disable=unsubscriptable-object
elif args.strategy == "proportional":
    split = np.load(
        "containerss/{}/{}/{}/{}/splitfile.npy".format(args.per,args.rseed,args.data,args.container), allow_pickle=True
    )
    weights = np.array([shard.shape[0] for shard in split])

# Tensor contraction of outputs and weights (on the shard dimension).
votes = np.argmax(
    np.tensordot(weights.reshape(1, weights.shape[0]), outputs, axes=1), axis=2
).reshape(
    (outputs.shape[1],)
)  # pylint: disable=unsubscriptable-object

# Load labels.
# X_train, train_labels, maj_train, min_train= dataloader.load(np.arange(datasetfile["nb_train"]), category="unf")
# print(X_train.shape)
X_test, labels, maj_test, min_test= dataloader.load(np.arange(datasetfile["nb_test"]), category="unf")

# Compute and print accuracy.
accuracy = (
    np.where(votes == labels)[0].shape[0] / outputs.shape[1]
)  # pylint: disable=unsubscriptable-object

# print(accuracy)

############# unfairness################
y_test      = labels
y_pred_test = votes 
#fairness constraint
# fairness_metric_name = get_metric(metric)
# fairness_metric = 4
unf_res = []
for metric in [1,3,4,5]:

    cm_test = ConfusionMatrix(min_test, maj_test, y_pred_test, y_test)
    cm_minority_test, cm_majority_test = cm_test.get_matrix()
    fm_test = Metric(cm_minority_test, cm_majority_test)
    acc_test = accuracy_score(y_test, y_pred_test)
    unf_test = fm_test.fairness_metric(metric)
    
    unf_test = np.round(unf_test, 2)
    unf_res.append(unf_test)

accuracy = np.round(accuracy, 2)
statistical_parity, predictive_equality,equal_opportunity, equalized_odds = unf_res
res = [accuracy, statistical_parity, predictive_equality,equal_opportunity, equalized_odds]
print(('{},'*len(res)).format(*res))