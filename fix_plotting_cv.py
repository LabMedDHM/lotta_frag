import json
import re

def patch_cv_script(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Consolidate plot_roc_curves: One figure, no premature show()
    new_plot_roc = """def plot_roc_curves(ergebnisse):
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
    plt.title(f'ROC Curves - Cross-Validation\nMean AUC = {mean_auc:.3f} ± {std_auc:.3f} | Pooled AUC = {auc_pooled:.3f}')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # In function, we typically don't call show() if we want to save later or combine, 
    # but since user wants immediate plots in notebook, we keep one show at the END of function.
    plt.show()"""

    # Replace the old functions
    pattern = r'def plot_roc_curves\(ergebnisse\):.*?plt\.show\(\)\n\n'
    content = re.sub(r'def plot_roc_curves\(ergebnisse\):.*?plt\.show\(\)\n(?=\n)', new_plot_roc + '\n', content, flags=re.DOTALL)
    
    with open(filepath, 'w') as f:
        f.write(content)

patch_cv_script('/labmed/workspace/lotta/finaletoolkit/ba_analysis_scripts/cv_lasso_single_fold.py')
