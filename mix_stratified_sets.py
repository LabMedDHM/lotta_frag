import pandas as pd
from sklearn.model_selection import train_test_split

training_metadata_path  = '/labmed/workspace/lotta/finaletoolkit/dataframes_for_ba/filtered_clinical_characteristics.csv'
holdout_metadata_path   = '/labmed/workspace/lotta/finaletoolkit/ba_analysis_scripts/holdout_preprocessing/dataframes_holdout/study_matrix_holdout.csv'
training_matrix_path    = '/labmed/workspace/lotta/finaletoolkit/dataframes_for_ba/final_feature_matrix_gc_corrected_50000.tsv'
holdout_matrix_path     = '/labmed/workspace/lotta/finaletoolkit/ba_analysis_scripts/holdout_preprocessing/dataframes_holdout/final_feature_matrix_gc_corrected_50000.tsv'

new_training_metadata_path = '/labmed/workspace/lotta/finaletoolkit/dataframes_for_ba/filtered_clinical_characteristics_stratified.csv'
new_holdout_metadata_path  = '/labmed/workspace/lotta/finaletoolkit/ba_analysis_scripts/holdout_preprocessing/dataframes_holdout/study_matrix_holdout_stratified.csv'
new_training_matrix_path   = '/labmed/workspace/lotta/finaletoolkit/dataframes_for_ba/final_feature_matrix_gc_corrected_50000_stratified.tsv'
new_holdout_matrix_path    = '/labmed/workspace/lotta/finaletoolkit/ba_analysis_scripts/holdout_preprocessing/dataframes_holdout/final_feature_matrix_gc_corrected_50000_stratified.tsv'

# ===========================================================
# 1. METADATEN LADEN UND ZUSAMMENFÜHREN
# ===========================================================
print("Lade Metadaten...")
train_meta = pd.read_csv(training_metadata_path, sep=';')
hold_meta  = pd.read_csv(holdout_metadata_path, sep=';')

print(f"  Trainings-Metadaten:  {len(train_meta)} Proben, Spalten: {list(train_meta.columns)}")
print(f"  Holdout-Metadaten:    {len(hold_meta)} Proben,  Spalten: {list(hold_meta.columns)}")

# Nur die 3 gemeinsamen Kernspalten für den Split verwenden
cols_to_keep = ['Extracted_ID', 'Patient Type', 'Study']
all_meta = pd.concat([train_meta[cols_to_keep], hold_meta[cols_to_keep]], ignore_index=True)

# Duplikate entfernen (falls eine ID in beiden Dateien vorkommt)
n_before = len(all_meta)
all_meta = all_meta.drop_duplicates(subset='Extracted_ID').reset_index(drop=True)
n_after  = len(all_meta)
if n_before != n_after:
    print(f"  ⚠️  {n_before - n_after} doppelte Proben-IDs entfernt.")

print(f"\nGesamt kombinierte Proben: {len(all_meta)}")
print("Verteilung nach Studie + Patiententyp:")
print(all_meta.groupby(['Study', 'Patient Type']).size().to_string())

# ===========================================================
# 2. STRATIFIZIERTER SPLIT (80/20)
# ===========================================================
all_meta['strata'] = all_meta['Study'] + "_" + all_meta['Patient Type']

# Strata mit weniger als 2 Proben ausschließen (sonst crasht train_test_split)
strata_counts = all_meta['strata'].value_counts()
rare_strata = strata_counts[strata_counts < 2].index
if len(rare_strata) > 0:
    print(f"\n⚠️  Folgende Strata haben <2 Proben und werden aus dem Split ausgeschlossen:")
    print(list(rare_strata))
    all_meta = all_meta[~all_meta['strata'].isin(rare_strata)]

train_ids_df, hold_ids_df = train_test_split(
    all_meta,
    test_size=0.2,
    stratify=all_meta['strata'],
    random_state=42
)

print(f"\n✅ Split durchgeführt: {len(train_ids_df)} Training, {len(hold_ids_df)} Holdout")
print("\nHoldout-Verteilung (Studie / Patiententyp):")
print(hold_ids_df.groupby(['Study', 'Patient Type']).size().to_string())

# ===========================================================
# 3. METADATEN SPEICHERN
# ===========================================================

# TRAINING: Wir wollen die vollen Spalten aus train_meta (Age, Gender, etc.)
# Proben aus dem alten Training → volle Infos aus train_meta
train_from_old_train = train_meta[train_meta['Extracted_ID'].isin(train_ids_df['Extracted_ID'])]
# Proben aus dem alten Holdout, die jetzt ins Training kommen → nur 3 Spalten verfügbar
train_from_old_hold  = hold_meta[hold_meta['Extracted_ID'].isin(train_ids_df['Extracted_ID'])]

final_train_meta = pd.concat([train_from_old_train, train_from_old_hold], ignore_index=True)
final_train_meta.to_csv(new_training_metadata_path, sep=';', index=False)

# HOLDOUT: Kernspalten (wie das ursprüngliche study_matrix_holdout.csv)
final_hold_meta = all_meta[all_meta['Extracted_ID'].isin(hold_ids_df['Extracted_ID'])][cols_to_keep]
final_hold_meta.to_csv(new_holdout_metadata_path, sep=';', index=False)

print(f"\n✅ Metadaten gespeichert:")
print(f"   Training: {len(final_train_meta)} Proben → {new_training_metadata_path}")
print(f"   Holdout:  {len(final_hold_meta)} Proben  → {new_holdout_metadata_path}")

# ===========================================================
# 4. FEATURE-MATRIZEN LADEN UND ZUSAMMENFÜHREN
# ===========================================================
# Die Holdout-TSV hat 2 Spalten weniger als die Training-TSV:
# Fehlend: 'Extracted_ID' und 'Study'
# Wir fügen diese Spalten hinzu, bevor wir zusammenführen.

print("\nLade Feature-Matrizen")
df_train_feat = pd.read_csv(training_matrix_path, sep='\t')
df_hold_feat  = pd.read_csv(holdout_matrix_path,  sep='\t')

print(f"  Training-Matrix geladen: {df_train_feat['sample'].nunique()} Proben")
print(f"  Holdout-Matrix geladen:  {df_hold_feat['sample'].nunique()} Proben")

# Spalten angleichen: 'Extracted_ID' und 'Study' in Holdout-Matrix ergänzen
# 'Extracted_ID' ist hier dasselbe wie 'sample'
df_hold_feat['Extracted_ID'] = df_hold_feat['sample']

# 'Study' aus den Holdout-Metadaten zuordnen (lookup über sample-ID)
study_lookup = hold_meta.set_index('Extracted_ID')['Study']
df_hold_feat['Study'] = df_hold_feat['sample'].map(study_lookup)

# Jetzt beide zusammenführen (alle Proben in einer Tabelle)
df_all_feat = pd.concat([df_train_feat, df_hold_feat], ignore_index=True)
print(f"\nKombinierte Matrix: {df_all_feat['sample'].nunique()} Proben total")

# ===========================================================
# 5. FEATURE-MATRIZEN FILTERN UND SPEICHERN
# ===========================================================
train_matrix = df_all_feat[df_all_feat['sample'].isin(final_train_meta['Extracted_ID'])]
hold_matrix  = df_all_feat[df_all_feat['sample'].isin(final_hold_meta['Extracted_ID'])]

# Sanity Check: Keine Probe darf in beiden Matrizen sein!
overlap = set(train_matrix['sample'].unique()) & set(hold_matrix['sample'].unique())
if overlap:
    print(f"\n🚨 FEHLER: {len(overlap)} Proben sind in BEIDEN Matrizen! {overlap}")
else:
    print("✅ Sanity Check OK: Keine Überschneidung zwischen Training und Holdout.")

train_matrix.to_csv(new_training_matrix_path, sep='\t', index=False)
hold_matrix.to_csv(new_holdout_matrix_path,   sep='\t', index=False)

print(f"\n✅ Fertig!")
print(f"   Training-Matrix: {train_matrix['sample'].nunique()} Proben → {new_training_matrix_path}")
print(f"   Holdout-Matrix:  {hold_matrix['sample'].nunique()} Proben  → {new_holdout_matrix_path}")
