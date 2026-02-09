import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from config import BIN_SIZE, SPECIFIC_GROUP
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    balanced_accuracy_score
)

RANDOM_STATE = 42

'''
- die funktion macht aus den daten 5 folds 
- für jeden fold wird erst die pipeline gecloned dass diese nicht veröndert wird und man mit der geklonten weoterarbeitet
- dann wird die cv_fold_run funktion ausgeführt für diesen fold
- man übergibt dabei X (Die Matrix), y (die labels), train_index (die train indices), test_index (die test indices), pipeline (die pipeline)

'''

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
        
        print(f"  AUC = {ergebnis['auc']:.3f}, Best C = {ergebnis.get('best_C', 'N/A')}")
    
    return ergebnisse

'''
- die funktion holt sich erst mal die Trainings- und Testdaten und dann die Trainings- und Test labels, also wie die samples aufgeteilt werden 
- Beipsiel: train_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] und test_index = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
- dann wird ganz normal die PIpeline trainiert wie wir es in dem lasso modeling skript haben 
- wahrscheinlichkeit für cancer, die auc score  und meine beiden rates werden für den fold zurückgegeben 
- dann wird ein dictionary mit den ergebnissen für den fold zurückgegeben
'''

def cv_fold_run(X, y, train_index, test_index, pipeline):
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    y_train = y[train_index]
    y_test = y[test_index]
    
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    
    # Feature Stability Extraction
    selected_features = {}
    best_C = None
    len_selected_features = 0
    len_all_features = X.shape[1]
    
    # Try to find the model step
    if 'lasso_cv' in pipeline.named_steps:
        model = pipeline.named_steps['lasso_cv']
        if hasattr(model, 'C_'):
             best_C = model.C_[0]
    elif 'model' in pipeline.named_steps:
        model = pipeline.named_steps['model']
        if hasattr(model, 'C'):
            best_C = model.C
    else:
        model = None

    if model and hasattr(model, 'coef_'):
        coefs = model.coef_[0]
        feature_names = X.columns
        for name, coef in zip(feature_names, coefs):
            if coef != 0:
                selected_features[name] = coef
        len_selected_features = len(selected_features)

    relative_feature_selection = len_selected_features / len_all_features
    absolute_feature_selection = len_selected_features

    return {
        'auc': auc_score,
        'best_C': best_C,
        'y_test': y_test,
        'y_prob': y_prob,
        'fpr': fpr,
        'tpr': tpr,
        'selected_features': selected_features,
        'relative_feature_selection': relative_feature_selection,
        'absolute_feature_selection': absolute_feature_selection,
        'accuracy': accuracy_score(y_test, y_pred),
        'sensitivity': recall_score(y_test, y_pred),  
        'specificity': recall_score(y_test, y_pred, pos_label=0),  
        'precision': precision_score(y_test, y_pred),
    }

def print_performance_table(ergebnisse):
    data = []
    for e in ergebnisse:
        data.append({
            'Fold': e['fold'],
            'AUC': e['auc'],
            'Accuracy': e.get('accuracy', np.nan),
            'Sensitivity': e.get('sensitivity', np.nan),
            'Specificity': e.get('specificity', np.nan),
            'Precision': e.get('precision', np.nan),
            'Best_C': e.get('best_C', np.nan),
            'N_Features': e.get('absolute_feature_selection', np.nan)
        })
    
    df = pd.DataFrame(data)
    
    mean_row = {
        'Fold': 'Mean',
        'AUC': df['AUC'].mean(),
        'Accuracy': df['Accuracy'].mean(),
        'Sensitivity': df['Sensitivity'].mean(),
        'Specificity': df['Specificity'].mean(),
        'Precision': df['Precision'].mean(),
        'Best_C': df['Best_C'].mean(),
        'N_Features': df['N_Features'].mean()
    }
    
    std_row = {
        'Fold': 'Std',
        'AUC': df['AUC'].std(),
        'Accuracy': df['Accuracy'].std(),
        'Sensitivity': df['Sensitivity'].std(),
        'Specificity': df['Specificity'].std(),
        'Precision': df['Precision'].std(),
        'Best_C': df['Best_C'].std(),
        'N_Features': df['N_Features'].std()
    }
    
    df = pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    
    return df



def analyze_feature_stability(ergebnisse):
    """
    Analysiert, wie oft jedes Feature über die Folds hinweg ausgewählt wurde.
    Gibt ein DataFrame zurück.
    """
    feature_counts = defaultdict(int)
    feature_coefs_sum = defaultdict(float)
    
    n_folds = len(ergebnisse)
    
    for fold_res in ergebnisse:
        sel_feats = fold_res.get('selected_features', {})
        for feat, coef in sel_feats.items():
            feature_counts[feat] += 1
            feature_coefs_sum[feat] += coef
            
    # Erstelle Tabelle
    data = []
    for feat, count in feature_counts.items():
        data.append({
            'Feature': feat,
            'Frequency': count,
            'Frequency_Percent': (count / n_folds) * 100,
            'Mean_Coef': feature_coefs_sum[feat] / count # Durchschnitt wenn ausgewählt
        })
        
    df_stability = pd.DataFrame(data)
    if not df_stability.empty:
        df_stability = df_stability.sort_values(by=['Frequency', 'Mean_Coef'], ascending=[False, False])
        
    return df_stability



def plot_roc_curves(ergebnisse):
    plt.figure(figsize=(10, 8))
    
    # Individual Folds
    for e in ergebnisse:
        plt.plot(e['fpr'], e['tpr'], alpha=0.5, linestyle=':',
                 label=f"Fold {e['fold']} (AUC = {e['auc']:.3f})")
    
    # Pooled ROC
    all_y_true = np.concatenate([e['y_test'] for e in ergebnisse])
    all_y_prob = np.concatenate([e['y_prob'] for e in ergebnisse])
    fpr_pooled, tpr_pooled, _ = roc_curve(all_y_true, all_y_prob)
    auc_pooled = roc_auc_score(all_y_true, all_y_prob)
    
    plt.plot(fpr_pooled, tpr_pooled, color='blue', linewidth=3, 
             label=f"Pooled ROC (AUC = {auc_pooled:.3f})")
    
    # Mean and Std
    mean_auc = np.mean([e['auc'] for e in ergebnisse])
    std_auc = np.std([e['auc'] for e in ergebnisse])
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - Cross-Validation Mean AUC = {mean_auc:.3f} ± {std_auc:.3f} | Pooled AUC = {auc_pooled:.3f}')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"/labmed/workspace/lotta/finaletoolkit/outputs/plots/roc_curve_per_fold_{BIN_SIZE}_{SPECIFIC_GROUP}.png")
    # In function, we typically don't call show() if we want to save later or combine, 
    # but since user wants immediate plots in notebook, we keep one show at the END of function.
    plt.show()


def plot_auc_boxplot(ergebnisse):
    auc_values = [e['auc'] for e in ergebnisse]
    
    plt.figure(figsize=(6, 6))
    sns.boxplot(y=auc_values)
    sns.stripplot(y=auc_values, color='red', size=10)
    
    plt.ylabel('AUC')
    plt.title(f'AUC Distribution\nMean = {np.mean(auc_values):.3f} ± {np.std(auc_values):.3f}')
    plt.savefig(f"/labmed/workspace/lotta/finaletoolkit/outputs/plots/auc_distribution_{BIN_SIZE}_{SPECIFIC_GROUP}.png")
    plt.show()

# # Voraussetzung: X und y sind bereits vorbereitet
# # Voraussetzung: pipeline ist bereits definiert
#
# ergebnisse = cross_validation(X, y, pipeline, n_folds=5)
# plot_roc_curves(ergebnisse)
# plot_auc_boxplot(ergebnisse)
#
