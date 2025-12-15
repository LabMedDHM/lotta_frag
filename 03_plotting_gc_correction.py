import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.impute import SimpleImputer

sns.set(style="whitegrid", rc={"figure.figsize":(10,6)})

original_path = "/labmed/workspace/lotta/finaletoolkit/dataframes_notebook/final_feature_matrix.tsv"
blacklist_filtered_path = "/labmed/workspace/lotta/finaletoolkit/dataframes_notebook/final_feature_matrix_blacklist_filtered.tsv"
gc_corrected_path = "/labmed/workspace/lotta/finaletoolkit/dataframes_notebook/final_feature_matrix_gc_corrected.tsv"

num_cols = ["mean", "median", "stdev", "min", "max", "wps_value"]


def recode_groups(df):
    df["group_binary"] = df["group"].apply(lambda x: "healthy" if x.lower() == "healthy" else "cancer")
    return df

def run_analysis(df, title):
    print(df.columns.tolist())
    df = recode_groups(df)

    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(df[num_cols])
    
    X_scaled = StandardScaler().fit_transform(X)
    
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_scaled)
    df["PC1"] = pcs[:, 0]
    df["PC2"] = pcs[:, 1]
    
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(X_scaled)
    df["UMAP1"] = embedding[:, 0]
    df["UMAP2"] = embedding[:, 1]
    
    # PCA plot (binary grouping)
    plt.figure()
    sns.scatterplot(x="PC1", y="PC2", hue="group_binary", data=df, palette="tab10")
    plt.title(f"PCA - {title} (Cancer vs Healthy)")
    plt.savefig(f"/labmed/workspace/lotta/finaletoolkit/outputs/plots/PCA_{title}_binary.png", dpi=300)
    plt.show()
    
    # UMAP plot (binary grouping)
    plt.figure()
    sns.scatterplot(x="UMAP1", y="UMAP2", hue="group_binary", data=df, palette="tab10")
    plt.title(f"UMAP - {title} (Cancer vs Healthy)")
    plt.savefig(f"/labmed/workspace/lotta/finaletoolkit/outputs/plots/UMAP_{title}_binary.png", dpi=300)
    plt.show()

df_orig = pd.read_csv(original_path, sep="\t")
df_gc = pd.read_csv(gc_corrected_path, sep="\t")
df_blacklist_filtered = pd.read_csv(blacklist_filtered_path, sep="\t")

run_analysis(df_orig, "Original")
run_analysis(df_blacklist_filtered, "Blacklist-Filtered")
run_analysis(df_gc, "GC-Corrected")
