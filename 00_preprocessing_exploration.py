#!/usr/bin/env python
# coding: utf-8

# # Installing Packages

# In[11]:


import os
import gc
import subprocess
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, levene, ranksums
from sklearn.linear_model import LinearRegression
import numpy as np
import pyBigWig
import math
import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from statsmodels.multivariate.manova import MANOVA
from scipy import stats
import statsmodels.api as sm
from matplotlib import gridspec
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
import glob
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# # Loading Samples (262 without Holdout-Dataset)

# In[12]:


cancer_samples = [
    # bile duct cancer
    "EE87789", "EE87790", "EE87791", "EE87792", "EE87793", "EE87794",
    "EE87795", "EE87796", "EE87797", "EE87798", "EE87799", "EE87800",
    "EE87801", "EE87802", "EE87803", "EE87804", "EE87805", "EE87806",
    "EE87807", "EE87809", "EE87810", "EE88325",

    # colorectal cancer
    # nicht in clinical table
    "EE85727", "EE85730", "EE85731", "EE85732", "EE85733", "EE85734",
    "EE85737", "EE85739", "EE85741", "EE85743", "EE85746", "EE85749",
    "EE85750", "EE85752", "EE85753", 
    
    
    
    "EE86234", "EE86255", "EE86259",
    "EE87865", "EE87866", "EE87867", "EE87868", "EE87869", "EE87870",
    "EE87871", "EE87872", "EE87873", "EE87874", "EE87875", "EE87876",
    "EE87877", "EE87878", "EE87879", "EE87880", "EE87881", "EE87882",
    "EE87883", "EE87884", "EE87885", "EE87886", "EE87887", "EE87888",
    "EE87889", "EE87890", "EE87891",

    # gastric cancer
    "EE87896", "EE87897", "EE87898", "EE87899", "EE87900", "EE87901",
    "EE87902", "EE87903", "EE87904", "EE87905", "EE87906", "EE87907",
    "EE87908", "EE87909", "EE87910", "EE87911", "EE87912", "EE87913",
    "EE87914", "EE87915", "EE87916", "EE87917", "EE87918", "EE87919",

    # pancreatic cancer
    "EE86268", "EE86270", "EE86271", "EE86272", "EE86273",
    "EE88290", "EE88291", "EE88292", "EE88293", "EE88294", "EE88295",
    "EE88296", "EE88297", "EE88298", "EE88299", "EE88300", "EE88301",
    "EE88302", "EE88303", "EE88304", "EE88305", "EE88306", "EE88307",
    "EE88308", "EE88309", "EE88310", "EE88311", "EE88312", "EE88313",
    "EE88314", "EE88315", "EE88316", "EE88317", "EE88318", "EE88319",
    "EE88320", "EE88321", "EE88322", "EE88323", "EE88324"
]
control_samples = [

    # healthy controls
    # nicht in clinical table
    "EE85898", "EE85904", "EE85905", "EE85908", "EE85918", "EE85928",
    "EE85936", "EE85937", "EE85941", "EE85959", "EE85963", "EE85970",
    "EE85971", "EE85980", "EE85985", "EE85987", "EE85988", "EE86275",
    "EE86276", "EE87945", "EE87946",
    

    
    "EE87920", "EE87921", "EE87922", "EE87923", "EE87924",
    "EE87925", "EE87926", "EE87927", "EE87928", "EE87929", "EE87931",
    "EE87932", "EE87933", "EE87934", "EE87935", "EE87936", "EE87937",
    "EE87938", "EE87939", "EE87940", "EE87941", "EE87942", "EE87943",
    "EE87944", "EE87947", "EE87948", "EE87949",
    "EE87950", "EE87951", "EE87952", "EE87953", "EE87954", "EE87955",
    "EE87956", "EE87957", "EE87958", "EE87959", "EE87960", "EE87961",
    "EE87962", "EE87963", "EE87964", "EE87965", "EE87966", "EE87967",
    "EE87968", "EE87969", "EE87970", "EE87971", "EE87972", "EE87973",
    "EE87974", "EE87975", "EE87976", "EE87977", "EE87978", "EE87979",
    "EE87980", "EE87981", "EE87982", "EE87983", "EE87984", "EE87985",
    "EE87986", "EE87987", "EE87988", "EE87989", "EE87990", "EE87991",
    "EE87992", "EE87993", "EE87994", "EE87995", "EE87996", "EE87997",
    "EE87998", "EE87999", "EE88000", "EE88001", "EE88002", "EE88003",
    "EE88004", "EE88005", "EE88006", "EE88007", "EE88008", "EE88009",
    "EE88010", "EE88011", "EE88012", "EE88013", "EE88014", "EE88015",
    "EE88016", "EE88017", "EE88018", "EE88019", "EE88020", "EE88021",
    "EE88022", "EE88023", "EE88024", "EE88025", "EE88026", "EE88027",
    "EE88028", "EE88029", "EE88030", "EE88031", "EE88032"
]

BASE_DIR = "/labmed/workspace/lotta/finaletoolkit/carsten/data_adjust_wps"

def find_sample_folder(sample, base_dir=BASE_DIR):
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.startswith(sample) and f.endswith(".adjust_wps.bw"):
                return root
    return None

def get_bigwig_path(sample):
    folder = find_sample_folder(sample)
    if folder is None:
        raise FileNotFoundError(f"Sample {sample} not found in {BASE_DIR}")
    return os.path.join(folder, f"{sample}.adjust_wps.bw")

def bigwig_summary(bigwig_path, chrom, start, end, n_bins=1):
    bw = pyBigWig.open(bigwig_path)
    bin_size = (end - start) // n_bins
    results = []
    
    for i in range(n_bins):
        b_start = start + i * bin_size
        b_end = start + (i+1) * bin_size if i < n_bins - 1 else end
        
        vals = bw.values(chrom, b_start, b_end)
        vals = [v for v in vals if v is not None and not math.isnan(v)]
        
        results.append(sum(vals)/len(vals) if vals else 0)

    bw.close()
    return results

all_samples = cancer_samples + control_samples
print(f"Configuration loaded for {len(all_samples)} samples:")
print(all_samples)


# # Cancer Typ aus dem Pfad extrahieren

# In[13]:


def get_cancer_type(sample):
    folder = find_sample_folder(sample)  
    if folder is None:
        return "Unknown"
    return os.path.basename(folder) 


# # Creating and Loading of Bedgraph Files 

# In[14]:


bedgraph_dir = os.path.expanduser('/labmed/workspace/lotta/finaletoolkit/carsten/data_adjust_wps')
from config import BIN_SIZE as bin_size

binned_output_path = f"/labmed/workspace/lotta/finaletoolkit/dataframes_for_ba/binned_combined_df_{bin_size}.parquet"

all_binned_dfs = []

if os.path.exists(binned_output_path):
    print(f"Loading existing binned dataframe from {binned_output_path}...")
    binned_combined_df = pd.read_parquet(binned_output_path)
else:
    print(f"Creating new binned dataframe with bin size {bin_size}...")
    
    def find_bedgraphs(sample_id):
        # pattern ist der gesuchte Dateipfad
        pattern = os.path.join(bedgraph_dir, "**", f"{sample_id}.adjust_wps.bedgraph")

        # matches sind alle gefundenen Dateien, die dem Muster entsprechen
        matches = glob.glob(pattern, recursive=True)
        # Gibt die erste gefundene Datei zurück 
        return matches[0] if matches else None

    for sample_id in all_samples:
        file_path = find_bedgraphs(sample_id)
        if file_path:
            try:
                df = pd.read_csv(file_path, sep="\t", header=None, names=["chrom", "start", "end", "wps_value"])
                df['sample'] = sample_id
                group = get_cancer_type(sample_id)
                df['group'] = group
                
                # IMMEDIATE BINNING TO SAVE MEMORY
                df['bin'] = df['start'] // bin_size
                # Calculate mean per bin for this sample immediately
                df_binned = df.groupby(['sample', 'group', 'chrom', 'bin'])['wps_value'].mean().reset_index()
                
                all_binned_dfs.append(df_binned)
                print(f"Loaded and binned {sample_id}. Rows: {len(df)} -> {len(df_binned)}")
                
                del df
                gc.collect()
            except Exception as e:
                print(f"Error processing {sample_id}: {e}")
        else:
            print(f"Bedgraph file for sample {sample_id} not found.")

    if all_binned_dfs:
        binned_combined_df = pd.concat(all_binned_dfs, ignore_index=True)
        print(f"Data successfully loaded and binned. Total rows: {len(binned_combined_df)}")
        
        # Apply median imputation for (chrom, bin) groups
               # Check for NaN values before imputation
        nan_count = binned_combined_df['wps_value'].isna().sum()
        print(f"Number of NaN values before imputation: {nan_count}")

        if nan_count > 0:
            print("Applying median imputation...")
            binned_combined_df['wps_value'] = binned_combined_df.groupby(['chrom', 'bin'])['wps_value'].transform(lambda x: x.fillna(x.median()))
        else:
            print("No NaN values found. Skipping imputation.")
        binned_combined_df.to_parquet(binned_output_path)
        print(f"Saved binned dataframe to {binned_output_path}")
    else:
        print("No data found!")


# Paper Fragment lenght profiles:
# - definiert kurze Fragmente als 100 - 150 und lange Fragmente als 151 - 250
# 
# 
# -   ◦ Kurze ALT-Fragmente: ALT-Fragmente, die kürzer als 150 bp waren
# 
# 
#     ◦ Lange ALT-Fragmente: ALT-Fragmente, die länger als 150 bp waren (nehme ich auch noch rein)
# 

# # Bin-Wide-Analysis, Binning the genome, Bin Size in Config File 
# 

# In[15]:


from config import BIN_SIZE as bin_size

if os.path.exists(f"/labmed/workspace/lotta/finaletoolkit/dataframes_for_ba/binned_combined_df_{bin_size}.parquet"):
    print("Loading existing binned combined dataframe...")
    binned_combined_df = pd.read_parquet(f"/labmed/workspace/lotta/finaletoolkit/dataframes_for_ba/binned_combined_df_{bin_size}.parquet")
else:
    combined_df['bin'] = combined_df['start'] // bin_size
    binned_combined_df = combined_df.groupby(['sample', 'group', 'chrom', 'bin'])['wps_value'].mean()
    binned_combined_df = binned_combined_df.reset_index()
    print(binned_combined_df[binned_combined_df['chrom'] =='chr2'])
    binned_combined_df['wps_value'] = binned_combined_df.groupby(['chrom', 'bin'])['wps_value'].transform(lambda x: x.fillna(x.median()))
    binned_combined_df.to_parquet(f"/labmed/workspace/lotta/finaletoolkit/dataframes_for_ba/binned_combined_df_{bin_size}.parquet")


# # Feature Matrix for LR rows=sample and columns=bins+groups 
# 

# In[16]:


if os.path.exists(f"/labmed/workspace/lotta/finaletoolkit/dataframes_for_ba/final_feature_matrix_{bin_size}.parquet"):
    print("Loading existing final feature matrix...")
    final_feature_matrix = pd.read_parquet(f"/labmed/workspace/lotta/finaletoolkit/dataframes_for_ba/final_feature_matrix_{bin_size}.parquet")
else:
    binned_combined_df['feature_name'] = binned_combined_df['chrom'] + '_bin_' + binned_combined_df['bin'].astype(str)
    feature_matrix = binned_combined_df.pivot(index='sample', columns='feature_name', values='wps_value')
    group_info = binned_combined_df[['sample', 'group']].drop_duplicates().set_index('sample')
    final_feature_matrix = feature_matrix.join(group_info)
    final_feature_matrix = final_feature_matrix.fillna(0)
    final_feature_matrix.to_parquet(f"/labmed/workspace/lotta/finaletoolkit/dataframes_for_ba/final_feature_matrix_{bin_size}.parquet", index=True)
    print(final_feature_matrix.head())


# # Fragment Interval Analysis: Loading Files
# 

# In[17]:


frag_interval_dir = os.path.expanduser('/labmed/workspace/lotta/finaletoolkit/output_workflow/frag_intervals')
frag_intervals_results = []
for sample in all_samples:
    interval_path = os.path.join(frag_interval_dir, '**', f"{sample}.frag_length_intervals.bed")
    files = glob.glob(interval_path, recursive=True)
    if not files:
        print(f"Fragment length Interval file for sample {sample} not found.")
        continue

    df = pd.read_csv(
    files[0],
    sep="\t",
    header=None,
    names=["chrom", "start", "stop", "name", "mean", "median", "stdev", "min", "max"]
    )
    df = df.iloc[1:].reset_index(drop=True)
    group = get_cancer_type(sample)
    df['sample'] = sample
    df['group'] = group
    df["start"] = df["start"].astype(int)
    df["stop"] = df["stop"].astype(int)

    num_cols = ["mean", "median", "stdev", "min", "max"]
    df[num_cols] = df[num_cols].astype(float)
    df['bin'] = df['start'] // bin_size
    frag_intervals_results.append(df)

frag_intervals_df = pd.concat(frag_intervals_results, ignore_index=True)


# In[18]:


print(frag_intervals_df.head())


# # Binning Fragment Interval Files
# 

# In[19]:


binned_df = (
    frag_intervals_df.groupby(['sample', 'group', 'chrom', 'bin'])
      .agg({
          "mean": "mean",
          "median": "mean",
          "stdev": "mean",
          "min": "mean",
          "max": "mean"
      })
      .reset_index()
)

print(binned_df.head())
print(binned_df.shape)


# In[20]:


print(binned_combined_df.head())


# In[21]:


merged_df = pd.merge(
    binned_df,
    binned_combined_df[['sample', 'chrom', 'bin', 'wps_value']],
    how='left',        # left join, falls manche Bins keinen WPS-Wert haben
    on=['sample', 'chrom', 'bin']
)

print(merged_df.head())
merged_df.to_csv(f"/labmed/workspace/lotta/finaletoolkit/dataframes_for_ba/final_feature_matrix_{bin_size}.tsv", sep="\t", index=False)

