import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.impute import SimpleImputer
from config import BIN_SIZE as bin_size



sns.set(style="whitegrid", rc={"figure.figsize":(10,6)})

original_path = f"/labmed/workspace/lotta/finaletoolkit/dataframes_for_ba/final_feature_matrix_{bin_size}.tsv"
blacklist_filtered_path = f"/labmed/workspace/lotta/finaletoolkit/dataframes_for_ba/final_feature_matrix_blacklist_filtered_{bin_size}.tsv"
gc_corrected_path = f"/labmed/workspace/lotta/finaletoolkit/dataframes_for_ba/final_feature_matrix_gc_corrected_{bin_size}.tsv"

num_cols = ["mean", "median", "stdev", "min", "max", "wps_value"]


def recode_groups(df):
    df["group_binary"] = df["group"].apply(lambda x: "healthy" if x.lower() == "healthy" else "cancer")
    return df

def run_analysis(df, title):
    print(f"Columns before pivot: {df.columns.tolist()}")
    
    # 1. Create Bin ID
    if 'bin_id' not in df.columns:
        df["bin_id"] = df["chrom"].astype(str) + "_" + df["start"].astype(str)

    # 2. Extract groups (labels)
    # We assume 'group' is constant for each sample.
    sample_groups = df.groupby("sample")["group"].first()

    # 3. Pivot
    print(f"Pivoting dataframe for {title}...")
    # metrics to pivot
    metrics = [c for c in num_cols if c in df.columns]
    
    pivot_df = df.pivot(index="sample", columns="bin_id", values=metrics)
    
    # Flatten MultiIndex columns if necessary
    # pivot_df.columns will be (metric, bin_id)
    pivot_df.columns = [f"{col[0]}_{col[1]}" for col in pivot_df.columns]
    
    print(f"Pivoted shape: {pivot_df.shape}")
    
    # 4. Filter/Impute
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(pivot_df)
    
    X_scaled = StandardScaler().fit_transform(X)
    
    # 5. Get Labels aligned with X
    # pivot_df index is sample IDs
    y_groups = sample_groups.loc[pivot_df.index]
    
    # Create a plotting dataframe
    plot_df = pd.DataFrame(index=pivot_df.index)
    plot_df["group"] = y_groups
    plot_df = recode_groups(plot_df) # Create group_binary

    # 6. PCA
    print("Running PCA...")
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_scaled)
    plot_df["PC1"] = pcs[:, 0]
    plot_df["PC2"] = pcs[:, 1]
    
    # 7. UMAP
    print("Running UMAP...")
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(X_scaled)
    plot_df["UMAP1"] = embedding[:, 0]
    plot_df["UMAP2"] = embedding[:, 1]
    
    # PCA plot (binary grouping)
    plt.figure()
    sns.scatterplot(x="PC1", y="PC2", hue="group_binary", data=plot_df, palette="tab10")
    plt.title(f"PCA - {title} (Cancer vs Healthy)")
    plt.savefig(f"/labmed/workspace/lotta/finaletoolkit/outputs/plots/PCA_{title}_binary.png", dpi=300)
    plt.show()
    
    # UMAP plot (binary grouping)
    plt.figure()
    sns.scatterplot(x="UMAP1", y="UMAP2", hue="group_binary", data=plot_df, palette="tab10")
    plt.title(f"UMAP - {title} (Cancer vs Healthy)")
    plt.savefig(f"/labmed/workspace/lotta/finaletoolkit/outputs/plots/UMAP_{title}_binary.png", dpi=300)
    plt.show()

    plt.figure()
    sns.scatterplot(x="UMAP1", y="UMAP2", hue="group", data=plot_df, palette="tab10")
    plt.title(f"UMAP - {title} (Cancer Types)")
    plt.savefig(f"/labmed/workspace/lotta/finaletoolkit/outputs/plots/UMAP_{title}_types.png", dpi=300)
    plt.show()

df_orig = pd.read_csv(original_path, sep="\t")
df_gc = pd.read_csv(gc_corrected_path, sep="\t")
df_blacklist_filtered = pd.read_csv(blacklist_filtered_path, sep="\t")

#run_analysis(df_orig, "Original")
#run_analysis(df_blacklist_filtered, "Blacklist-Filtered")
run_analysis(df_gc, "GC-Corrected")


def calc_gc(chrom, start, end):
    seq = genome[chrom][start:end].seq.upper()
    if len(seq) == 0:
        return np.nan
    return (seq.count("G") + seq.count("C")) / len(seq)

def plot_gc_correction_check(sample_id=None):    

    if sample_id is None:
        sample_id = df["sample"].unique()[0]
    
    print(f"Plotting GC correction check for sample: {sample_id}")

    blacklist_filtered_feature_matrix_path = f"/labmed/workspace/lotta/finaletoolkit/dataframes_for_ba/final_feature_matrix_blacklist_filtered_{bin_size}.tsv"
    blacklist_filtered_feature_matrix = pd.read_csv(blacklist_filtered_feature_matrix_path, sep="\t")

    gc_corrected_feature_matrix_path = f"/labmed/workspace/lotta/finaletoolkit/dataframes_for_ba/final_feature_matrix_gc_corrected_{bin_size}.tsv"
    gc_corrected_feature_matrix = pd.read_csv(gc_corrected_feature_matrix_path, sep="\t")



    
    subset_blacklist_filtered = blacklist_filtered_feature_matrix[blacklist_filtered_feature_matrix["sample"] == sample_id].copy()
    subset_gc_corrected = gc_corrected_feature_matrix[gc_corrected_feature_matrix["sample"] == sample_id].copy()
            
    metrics = [ 'mean', 'median', 'stdev', 'min', 'max', 'wps_value']
    for metric in metrics: 
        plt.figure(figsize=(10, 6))
        
        # Plot Original
        sns.regplot(
            data=subset_blacklist_filtered, x="GC", y=metric, 
            scatter_kws={'alpha':0.3, 's':10}, 
            line_kws={'color':'red'},
            lowess=True, 
            label="Before Correction"
        )
    
        # Plot Corrected
        sns.regplot(
            data=subset_gc_corrected, x="GC", y=metric, 
            scatter_kws={'alpha':0.3, 's':10}, 
            line_kws={'color':'blue'},
            lowess=True, 
            label="After Correction"
        )
        
        plt.title(f"GC Bias Correction Check - {metric}\nSample: {sample_id}")
        plt.xlabel("GC Content per Bin")
        plt.ylabel(metric)
        plt.legend()
        
        out_file = os.path.join("/labmed/workspace/lotta/finaletoolkit/outputs/plots", f"GC_Correction_{metric}_{sample_id}.png")
        plt.savefig(out_file, dpi=300)
        print(f"Saved plot to {out_file}")
        plt.show()

# Run the check for one representative sample
plot_gc_correction_check("EE87922")
