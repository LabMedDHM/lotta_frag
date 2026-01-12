import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

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
    
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    
    # Feature Stability Extraction
    # Wir nehmen an, dass der letzte Schritt im Pipeline-Objekt das Modell ist 
    # oder spezifisch 'lasso_cv' heißt. Hier versuchen wir es generisch oder fallback auf 'lasso_cv'.
    
    selected_features = {}
    best_C = None
    if 'lasso_cv' in pipeline.named_steps:
        model = pipeline.named_steps['lasso_cv']
        
        # Best C Value extrahieren
        if hasattr(model, 'C_'):
             best_C = model.C_[0]
             
        if hasattr(model, 'coef_'):
            coefs = model.coef_[0]
            feature_names = X.columns
            
            for name, coef in zip(feature_names, coefs):
                if coef != 0:
                    selected_features[name] = coef
    
    return {
        'auc': auc_score,
        'best_C': best_C,
        'y_test': y_test,
        'y_prob': y_prob,
        'fpr': fpr,
        'tpr': tpr,
        'selected_features': selected_features
    }


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
    
    for e in ergebnisse:
        plt.plot(e['fpr'], e['tpr'], alpha=0.7,
                 label=f"Fold {e['fold']} (AUC = {e['auc']:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classification')
    
    # ergebnisse = [
    #     {"y_test": y_test_fold1, "y_prob": y_prob_fold1}, # : wahren Labels im test fold
    #     {"y_test": y_test_fold2, "y_prob": y_prob_fold2}, # : vorhergesagten Wahrscheinlichkeiten für cancer
    #     ...
    # ]

    all_y_true = np.concatenate([e['y_test'] for e in ergebnisse]) # Array von allen wahren labels aus den folds
    all_y_prob = np.concatenate([e['y_prob'] for e in ergebnisse]) # Array von allen vorhergesagten Wahrscheinlichkeiten aus den folds
    
    print(f"DEBUG: Pooled y_true shape: {all_y_true.shape}, y_prob shape: {all_y_prob.shape}")
    if np.isnan(all_y_prob).any():
        print("DEBUG: y_prob contains NaNs!")
    
    fpr_pooled, tpr_pooled, _ = roc_curve(all_y_true, all_y_prob) # ROC Kurve für alle folds
    auc_pooled = roc_auc_score(all_y_true, all_y_prob) # AUC für alle folds
    
    plt.plot(fpr_pooled, tpr_pooled, color='blue', linewidth=2, 
             label=f"Pooled ROC (AUC = {auc_pooled:.3f})")
    
    mean_auc = np.mean([e['auc'] for e in ergebnisse])
    std_auc = np.std([e['auc'] for e in ergebnisse])
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - 5-Fold CV\nMean AUC = {mean_auc:.3f} ± {std_auc:.3f}\nPooled AUC = {auc_pooled:.3f}')
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
