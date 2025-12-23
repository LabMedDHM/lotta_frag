import random
import sklearn as sk
import seaborn as sns
import matplotlib.pyplot as plt

random.seed(42)


# fold definition for cross-vlidation
def fold_definition(n_folds, stratified_labels):
    return sk.model_selection.StratifiedKFold(n_folds=n_folds).split(stratified_labels)


# ergebnisse in roc curve darstellen
def cv_fold_run():
    daten_laden()
    verarbeiten()
    ergebnis = lasso_berechnug()
    return ergebnis


def cross_validation():
    ergebnisse = []

    for fold in fold_definition(n_folds=5, stratified_labels=stratified_labels):
        ergebnisse.append(cv_fold_run())

    return ergebnisse


def schoene_graphiken():
    sns.lineplot(data=ergebnisse)
    sns.boxplot(data=ergebnisse)
    ...
