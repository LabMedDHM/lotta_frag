import pandas as pd

# Paths for non-stratified
old_train_path = '/labmed/workspace/lotta/finaletoolkit/dataframes_for_ba/filtered_clinical_characteristics.csv'
old_hold_path = '/labmed/workspace/lotta/finaletoolkit/ba_analysis_scripts/holdout_preprocessing/dataframes_holdout/study_matrix_holdout.csv'

# Paths for stratified
new_train_path = '/labmed/workspace/lotta/finaletoolkit/dataframes_for_ba/filtered_clinical_characteristics_stratified.csv'
new_hold_path = '/labmed/workspace/lotta/finaletoolkit/ba_analysis_scripts/holdout_preprocessing/dataframes_holdout/study_matrix_holdout_stratified.csv'

def get_summary(train_path, hold_path, name):
    train_meta = pd.read_csv(train_path, sep=';')
    hold_meta = pd.read_csv(hold_path, sep=';')
    
    train_meta['Set'] = 'Training'
    hold_meta['Set'] = 'Holdout'
    
    cols = ['Study', 'Patient Type', 'Set']
    all_df = pd.concat([train_meta[cols], hold_meta[cols]])
    
    summary = all_df.groupby(['Study', 'Patient Type', 'Set']).size().unstack(fill_value=0)
    
    # Reorder columns to Training, Holdout if present
    if 'Training' not in summary.columns: summary['Training'] = 0
    if 'Holdout' not in summary.columns: summary['Holdout'] = 0
    
    summary = summary[['Training', 'Holdout']]
    summary['Total'] = summary['Training'] + summary['Holdout']
    
    print(f"=== {name} ===")
    print(summary.to_markdown())
    print("\n")

get_summary(old_train_path, old_hold_path, "URSPRÜNGLICHE VERTEILUNG (Ohne Stratifizierung)")
get_summary(new_train_path, new_hold_path, "NEUE VERTEILUNG (Mit Stratifizierung)")
