import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt

RANDOM_STATE = 42


def cv_fold_run(X, y, train_index, test_index, pipeline):
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    y_train = y[train_index]
    y_test = y[test_index]
    
    pipeline.fit(X_train, y_train)
    
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    
    return {
        'auc': auc_score,
        'y_test': y_test,
        'y_prob': y_prob,
        'fpr': fpr,
        'tpr': tpr,
    }


def cross_validation(X, y, pipeline, n_folds=5):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    ergebnisse = []
    
    for fold_nr, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(f"Fold {fold_nr + 1}/{n_folds}...")
        
        from sklearn.base import clone
        fold_pipeline = clone(pipeline)
        
        ergebnis = cv_fold_run(X, y, train_index, test_index, fold_pipeline)
        ergebnis['fold'] = fold_nr + 1
        ergebnisse.append(ergebnis)
        
        print(f"  AUC = {ergebnis['auc']:.3f}")
    
    return ergebnisse


def plot_roc_curves(ergebnisse):
    plt.figure(figsize=(10, 8))
    
    for e in ergebnisse:
        plt.plot(e['fpr'], e['tpr'], alpha=0.7,
                 label=f"Fold {e['fold']} (AUC = {e['auc']:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', label='Zufall')
    
    mean_auc = np.mean([e['auc'] for e in ergebnisse])
    std_auc = np.std([e['auc'] for e in ergebnisse])
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - 5-Fold CV\nMean AUC = {mean_auc:.3f} ± {std_auc:.3f}')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_auc_boxplot(ergebnisse):
    auc_values = [e['auc'] for e in ergebnisse]
    
    plt.figure(figsize=(6, 6))
    sns.boxplot(y=auc_values)
    sns.stripplot(y=auc_values, color='red', size=10)
    
    plt.ylabel('AUC')
    plt.title(f'AUC Distribution\nMean = {np.mean(auc_values):.3f} ± {np.std(auc_values):.3f}')
    plt.show()

# # Voraussetzung: X und y sind bereits vorbereitet
# # Voraussetzung: pipeline ist bereits definiert
#
# ergebnisse = cross_validation(X, y, pipeline, n_folds=5)
# plot_roc_curves(ergebnisse)
# plot_auc_boxplot(ergebnisse)
#
