import numpy as np
import pandas as pd
import seaborn as sns
import gseapy as gp
import os

import scanpy as sp
import torch

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
matplotlib.rcParams.update({'font.family': 'Arial'})


color_ls = [
    "#3cb44b",
    "#e6194b",
    "#ffe119",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#bcf60c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#9a6324",
    "#fffac8",
    "#800000",
    "#aaffc3",
    "#808000",
    "#ffd8b1",
    "#000075",
    "#808080",
    "#ffd700",
    "#ff4500",
    "#da70d6",
    "#32cd32",
    "#4682b4",
    "#d2691e",
    "#ff1493",
    "#7fff00",
    "#00ced1",
    "#ff00ff",
    "#1e90ff",
    "#b22222",
    "#adff2f",
    "#a020f0",
    "#00ff00",
    "#4682b4",
    "#f0e68c",
    "#b03060",
    "#0000cd",
    "#808000",
    "#8b4513",
    "#ee82ee",
    "#ff8c00",
    "#556b2f",
    "#00bfff",
    "#dc143c",
    "#fa8072",
    "#ff00ff",
    "#32cd32",
    "#ffff00",
    "#daa520",
    "#1e90ff",
    "#2f4f4f",
    "#ff0000",
    "#483d8b",
    "#afeeee",
    "#dda0dd",
    "#8b0000",
    "#9acd32",
    "#8fbc8f",
    "#98fb98",
    "#f4a460",
    "#228b22",
    "#a9a9a9",
    "#ff1493",
    "#ffe4c4",
    "#00008b",
    "#20b2aa",
    "#800080",
    "#00ffff",
    "#7b68ee",
    "#ffb6c1",
]




def preprocess_plot(raw_adata):
    adata_umap=raw_adata.copy()
    sp.pp.filter_cells(adata_umap, min_genes=200)
    sp.pp.filter_genes(adata_umap, min_cells=3)
    # adata_umap_counts=adata_umap.X.copy()

    # Normalize the data to 10,000 reads per cell, log-transform
    sp.pp.normalize_total(adata_umap, target_sum=1e4)
    sp.pp.log1p(adata_umap)

    # Identify highly variable genes
    sp.pp.highly_variable_genes(adata_umap, min_mean=0.0125, max_mean=3, min_disp=0.5)
    # adata = adata[:, adata.var.highly_variable]

    # Regress out effects of total counts per cell and the percentage of mitochondrial genes
    # sc.pp.regress_out(adata, ['n_counts', 'percent_mito'])

    # Scale the data
    sp.pp.scale(adata_umap, max_value=10)

    # Run PCA
    sp.tl.pca(adata_umap, svd_solver='arpack')

    # Compute the neighborhood graph
    sp.pp.neighbors(adata_umap, n_neighbors=10, n_pcs=40)

    # Run UMAP
    sp.tl.umap(adata_umap)
    return adata_umap



def check_celltypes(adata, predicted_cell_indexs):
    res=[]
    for i in predicted_cell_indexs:
        res.append(adata.obs.iloc[i-adata.shape[1]].clusters)
    from collections import Counter
    print("snc in different cell types: ",Counter(res))
    return Counter(res)

def load_snc_info(new_data,file_path,threshold=10):
    adata=new_data.copy()

    sencell_dict,sen_gene_ls,attention_scores,edge_index_selfloop=torch.load(file_path)
    sencell_indexs=list(sencell_dict.keys())

    a=dict(check_celltypes(new_data,sencell_indexs))
    sorted_dict = dict(sorted(a.items(), key=lambda item: item[1], reverse=True))

    # select cell types, for all ct
    selected_ct=[]
    for key, value in sorted_dict.items():
        if value >= threshold:
            selected_ct.append(key)

    sencell_indexs_updated=[]
    for i in sencell_indexs:
        ct=adata.obs.iloc[i-new_data.shape[1]].clusters
        if ct in selected_ct:
            sencell_indexs_updated.append(i)
            
    sencell_indexs=sencell_indexs_updated
    row_indices= np.array(sencell_indexs)-new_data.shape[1]
    new_column = np.array(['normal']*adata.shape[0])
    new_column[row_indices] = 'SnC'
    adata.obs['is_sen'] = new_column

    adata.obs['clusters']=adata.obs['clusters'].astype(str)

    def create_column(row):
        if row['is_sen']=='SnC':
            return 'SnC'
        else:
            return row['clusters']
    adata.obs['new_ct'] = adata.obs.apply(create_column, axis=1)

    print(f"Number of SnC: {adata.obs['is_sen'].value_counts()['SnC']}")

    sub_sencells=adata[adata.obs['is_sen']=='SnC']
    sub_sencells=sub_sencells[sub_sencells.obs["clusters"].isin(selected_ct)]
    
    return adata, sub_sencells



def generate_umap(
    adata,
    column_name="clusters",
    filename=None,
    show_text=False,
    bbox_to_anchor=(1.3, 0.5),
):
    """
    Generates a UMAP plot from the given AnnData object and saves it as an image file.
    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix.
    column_name : str, optional
        The column name in `adata.obs` to use for coloring the UMAP plot. Default is "clusters".
    filename : str, optional
        The name of the file to save the plot. If None, a default name is generated. Default is None.
    show_text : bool, optional
        If True, display the cluster names at the centroid of each cluster. Default is False.
    bbox_to_anchor : tuple, optional
        The bounding box anchor for the legend. Default is (1.3, 0.5).
    Returns:
    --------
    None
    """
    row_numbers_by_category = {}
    for idx, value in enumerate(adata.obs[column_name]):
        row_numbers_by_category.setdefault(value, []).append(idx)

    row_numbers_by_category = dict(
        sorted(row_numbers_by_category.items(), key=lambda item: item[0])
    )

    keys = list(row_numbers_by_category.keys())
    new_row_numbers_by_category = {}
    for i, j in enumerate(keys):
        new_row_numbers_by_category[f"{i}: {j}"] = row_numbers_by_category[j]
    row_numbers_by_category = new_row_numbers_by_category

    umap1 = adata.obsm["X_umap"][:, 0]
    umap2 = adata.obsm["X_umap"][:, 1]

    plt.figure(figsize=(15, 10), dpi=300)

    color_count = 0

    h_ls = []
    for cluster, cell_index in row_numbers_by_category.items():
        plt.scatter(
            umap1[cell_index],
            umap2[cell_index],
            s=4,
            color=color_ls[color_count % len(color_ls)],
            label=cluster,
        )
        h_ls.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                markersize=10,
                color=color_ls[color_count % len(color_ls)],
                linestyle="",
            )
        )
        if show_text:
            # Get the coordinates of the points in this cluster
            x_coords = umap1[cell_index]
            y_coords = umap2[cell_index]
            # Compute the centroid – you can use np.mean or np.median
            x_center = np.mean(x_coords)
            y_center = np.mean(y_coords)
            # Add text at the centroid
            plt.text(
                x_center,
                y_center,
                cluster,
                fontsize=12,
                fontweight="bold",
                ha="center",
                va="center",
                color="black",
                bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"),
            )

        color_count += 1

    _, legend_labels = plt.gca().get_legend_handles_labels()
    plt.legend(
        handles=h_ls,
        labels=legend_labels,
        bbox_to_anchor=bbox_to_anchor,
        loc="right",
        fontsize="large",
    )
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    if filename is None:
        filename = f"high_res_figure_{column_name}.png"
    plt.savefig(filename, dpi=300)
    plt.show()


def generate_umap_snc(
    adata,
    column_name="clusters",
    filename=None,
    show_text=False,
    bbox_to_anchor=(1.3, 0.5),
):
    """
    Generate a UMAP plot for single-cell data with SnC highlighted.
    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix.
    column_name : str, optional
        The column name in `adata.obs` to use for coloring the clusters (default is "clusters").
    filename : str, optional
        The filename to save the plot. If None, a default filename is used (default is None).
    show_text : bool, optional
        Whether to display the cluster names as text on the plot (default is False).
    bbox_to_anchor : tuple, optional
        The bounding box anchor for the legend (default is (1.3, 0.5)).
    Returns:
    --------
    None
    """
    row_numbers_by_category = {}
    for idx, value in enumerate(adata.obs[column_name]):
        row_numbers_by_category.setdefault(value, []).append(idx)

    row_numbers_by_category = dict(
        sorted(row_numbers_by_category.items(), key=lambda item: item[0])
    )

    keys = list(row_numbers_by_category.keys())
    for i, j in enumerate(keys):
        row_numbers_by_category[f"{i}: {j}"] = row_numbers_by_category[j]
        row_numbers_by_category.pop(j)

    umap1 = adata.obsm["X_umap"][:, 0]
    umap2 = adata.obsm["X_umap"][:, 1]

    plt.figure(figsize=(15, 10), dpi=300)

    color_count = 0

    h_ls = []
    for cluster, cell_index in row_numbers_by_category.items():
        if "SnC" in cluster:
            plt.scatter(
                umap1[cell_index], umap2[cell_index], s=5, color="red", label=cluster
            )
            h_ls.append(
                plt.Line2D(
                    [0], [0], marker="o", markersize=10, color="red", linestyle=""
                )
            )
        else:
            plt.scatter(
                umap1[cell_index],
                umap2[cell_index],
                s=4,
                color="#cccccb",
                label=cluster,
            )
            h_ls.append(
                plt.Line2D(
                    [0], [0], marker="o", markersize=10, color="#cccccb", linestyle=""
                )
            )
            color_count += 1

        if show_text:
            # Get the coordinates of the points in this cluster
            x_coords = umap1[cell_index]
            y_coords = umap2[cell_index]
            # Compute the centroid – you can use np.mean or np.median
            x_center = x_coords.mean()
            y_center = y_coords.mean()
            # Add text at the centroid
            plt.text(
                x_center,
                y_center,
                cluster,
                fontsize=12,
                fontweight="bold",
                ha="center",
                va="center",
                color="black",
                bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"),
            )

        color_count += 1

    legend_handles, legend_labels = plt.gca().get_legend_handles_labels()
    plt.legend(
        handles=h_ls,
        labels=legend_labels,
        bbox_to_anchor=bbox_to_anchor,
        loc="right",
        fontsize="large",
    )
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    if filename is None:
        filename = f"high_res_figure_{column_name}.png"
    plt.savefig(filename, dpi=300)
    plt.show()


def generate_heatmap(adata, 
                     gene_order,
                     cluster_order,
                     column_name="clusters", 
                     filename=None):
    
    # If adata.X is sparse, convert it to a dense array.
    if hasattr(adata.X, "toarray"):
        expr_df = pd.DataFrame(adata.X.toarray(), index=adata.obs_names, columns=adata.var.index)
    else:
        expr_df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var.index)

    # Add the cluster information from adata.obs.
    expr_df[column_name] = adata.obs[column_name].values

    # Compute the average expression per cluster.
    # This results in a DataFrame with clusters as rows and genes as columns.
    cluster_avg = expr_df.groupby(column_name).mean()

    # Now, create a heatmap DataFrame:
    # We want genes as rows and clusters as columns.
    # First, subset the columns (genes) to those in gene_order.
    # Then, transpose the DataFrame.
    heatmap_df = cluster_avg.loc[:, gene_order].T

    # Reorder the columns (clusters) as desired.
    heatmap_df = heatmap_df[cluster_order]

    # Plot the heatmap.
    plt.figure(figsize=(12, len(gene_order) * 0.15),dpi=300)
    
    # low color, medium and high color seperately
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#4F99A9','white','#D65A79'])

    heatmap_df_norm = heatmap_df.sub(heatmap_df.mean(axis=1), axis=0).div(heatmap_df.std(axis=1), axis=0)

    sns.heatmap(heatmap_df_norm, cmap=cmap,yticklabels=False)
    # sns.heatmap(heatmap_df, cmap=cmap)


    plt.xlabel("Clusters")
    plt.ylabel("Genes")
    plt.title("Average Expression of Selected Genes by Cluster")
    plt.tight_layout()
    plt.savefig('high_res_figure_heatmap.png', dpi=300)

    plt.show()





def bar_plot_condition(sub_sencells):
    grouped = (
        sub_sencells.obs.groupby(["clusters", "Status"])
        .size()
        .reset_index(name="count")
    )

    # Pivot the table so 'disease_new' values become new columns
    pivot_table = grouped.pivot_table(
        values="count", index="clusters", columns="Status", fill_value=0
    )
    print(pivot_table)
    # Calculate percentages
    percentage_table = pivot_table.div(pivot_table.sum(axis=1), axis=0)

    # Reset index to make 'Rationale_based_annotation' a column again
    percentage_table.reset_index(inplace=True)

    # Rename the columns
    percentage_table.columns.name = None  # remove the name for columns
    percentage_table.rename(columns={"clusters": "cell type"}, inplace=True)

    pivot_table["total"] = pivot_table.sum(axis=1).astype("string")
    pivot_table.reset_index(inplace=True)
    percentage_table["cell type"] = (
        percentage_table["cell type"] + " (" + pivot_table["total"] + ")"
    )

    percentage_table["total"] = pivot_table["total"]

    percentage_table["total"] = percentage_table["total"].astype("int")
    percentage_table = percentage_table.sort_values(by="total", ascending=True)

    # percentage_table.rename(columns={'Mixed': 'Healthy'}, inplace=True)

    new_order = ["cell type", "Healthy", "IPF"]
    percentage_table = percentage_table[new_order]

    percentage_table.plot(
        x="cell type",
        kind="barh",
        stacked=True,
        title="Percentage of SnCs in different condition",
        color=["#1f77b4", "#d62728", "#ff9896"],
        mark_right=True,
    )
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)

    plt.legend(bbox_to_anchor=(1.19, 1), loc="upper right")
    plt.show()


def bar_plot_location(sub_sencells):
    grouped = (
        sub_sencells.obs.groupby(["clusters", "Area"]).size().reset_index(name="count")
    )

    # Pivot the table so 'disease_new' values become new columns
    pivot_table = grouped.pivot_table(
        values="count", index="clusters", columns="Area", fill_value=0
    )
    print(pivot_table)

    # Calculate percentages
    percentage_table = pivot_table.div(pivot_table.sum(axis=1), axis=0)

    # Reset index to make 'Rationale_based_annotation' a column again
    percentage_table.reset_index(inplace=True)

    # Rename the columns
    percentage_table.columns.name = None  # remove the name for columns
    percentage_table.rename(columns={"clusters": "cell type"}, inplace=True)

    pivot_table["total"] = pivot_table.sum(axis=1).astype("string")
    pivot_table.reset_index(inplace=True)
    percentage_table["cell type"] = (
        percentage_table["cell type"] + " (" + pivot_table["total"] + ")"
    )

    # percentage_table.rename(columns={'Mixed': 'Healthy'}, inplace=True)

    percentage_table["total"] = pivot_table["total"]

    percentage_table["total"] = percentage_table["total"].astype("int")
    percentage_table = percentage_table.sort_values(by="total", ascending=True)

    new_order = ["cell type", "Upper Lobe", "Lower Lobe", "Parenchyma"]
    percentage_table = percentage_table[new_order]

    percentage_table.plot(
        x="cell type",
        kind="barh",
        stacked=True,
        title="Percentage of SnCs in different location",
        color=[
            "#d62728",
            "#ff9896",
            "#1f77b4",
        ],
        mark_right=True,
    )
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)

    plt.legend(bbox_to_anchor=(1.25, 1), loc="upper right")
    plt.show()


def bar_plot_age(sub_sencells):
    grouped = (
        sub_sencells.obs.groupby(["clusters", "Age_Status"])
        .size()
        .reset_index(name="count")
    )

    # Pivot the table so 'disease_new' values become new columns
    pivot_table = grouped.pivot_table(
        values="count", index="clusters", columns="Age_Status", fill_value=0
    )
    print(pivot_table)

    # Calculate percentages
    percentage_table = pivot_table.div(pivot_table.sum(axis=1), axis=0)

    # Reset index to make 'Rationale_based_annotation' a column again
    percentage_table.reset_index(inplace=True)

    # Rename the columns
    percentage_table.columns.name = None  # remove the name for columns
    percentage_table.rename(columns={"clusters": "cell type"}, inplace=True)

    pivot_table["total"] = pivot_table.sum(axis=1).astype("string")
    pivot_table.reset_index(inplace=True)
    percentage_table["cell type"] = (
        percentage_table["cell type"] + " (" + pivot_table["total"] + ")"
    )

    # percentage_table.rename(columns={'Mixed': 'Healthy'}, inplace=True)

    percentage_table["total"] = pivot_table["total"]

    percentage_table["total"] = percentage_table["total"].astype("int")
    percentage_table = percentage_table.sort_values(by="total", ascending=True)

    new_order = ["cell type", "Old", "Young"]
    percentage_table = percentage_table[new_order]

    percentage_table.plot(
        x="cell type",
        kind="barh",
        stacked=True,
        title="Percentage of SnCs in different age",
        color=["#7a0000", "#d62728", "#ff9896", "#1f77b4", "#daece5"],
        mark_right=True,
    )
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)

    plt.legend(bbox_to_anchor=(1.25, 1), loc="upper right")
    plt.show()


def create_summary_table(adata):
    """
    Creates a summary table of cell counts for each cluster (CT) 
    across various conditions: IPF vs Healthy, UL vs LL vs Parenchyma,
    and Young vs Old. Also calculates the fraction (percentage) of SnC cells.
    
    Assumes:
    - 'clusters' column has the cell type / cluster names.
    - 'Injury' column has 'IPF' or 'Healthy'.
    - 'area' column has 'Upper Lobe', 'Lower Lobe', or 'Parenchyma'.
    - 'Age_Status' column has 'young' or 'old'.
    - 'is_sen' column has 'SnC' or 'normal' (indicating senescent or not).
    """
    # Work off obs DataFrame
    df = adata.obs.copy()

    # All unique clusters
    clusters = df['clusters'].unique().tolist()

    # Define columns in final summary
    columns = [
        'CT',                 # cluster name
        '# cell',             # total cells in that cluster
        '# IPF non-SnC', '# IPF SnC', 'IPF SnC%',
        '# Healthy non-SnC', '# Healthy SnC', 'Healthy SnC%',
        '# UL non-SnC', '# UL SnC', 'UL SnC%',
        '# LL non-SnC', '# LL SnC', 'LL SnC%',
        '# Parenchyma non-SnC', '# Parenchyma SnC', 'Parenchyma SnC%',
        '# Young non-SnC', '# Young SnC', 'Young SnC%',
        '# Old non-SnC', '# Old SnC', 'Old SnC%'
    ]
    
    # Prepare empty summary DataFrame
    summary_df = pd.DataFrame(columns=columns)

    # Helper function to count SnC vs non-SnC for a subset
    def group_counts(sub_df):
        # Number of non-SnC
        n_nonSnC = (sub_df['is_sen'] == 'normal').sum()
        # Number of SnC
        n_SnC = (sub_df['is_sen'] == 'SnC').sum()
        # Percentage of SnC
        total = n_nonSnC + n_SnC
        pct_SnC = (n_SnC / total * 100) if total > 0 else 0.0
        return n_nonSnC, n_SnC, pct_SnC

    for ct in clusters:
        # Subset the DataFrame to the current cluster
        df_ct = df[df['clusters'] == ct]
        
        # Total cells in this cluster
        total_cells = len(df_ct)

        # Count IPF vs Healthy
        df_ct_ipf = df_ct[df_ct['Status'] == 'IPF']
        ipf_nonSnC, ipf_SnC, ipf_pct = group_counts(df_ct_ipf)

        df_ct_healthy = df_ct[df_ct['Status'] == 'Healthy']
        healthy_nonSnC, healthy_SnC, healthy_pct = group_counts(df_ct_healthy)

        # Count Upper/Lower Lobe vs Parenchyma
        df_ct_ul = df_ct[df_ct['Area'] == 'Upper Lobe']
        ul_nonSnC, ul_SnC, ul_pct = group_counts(df_ct_ul)

        df_ct_ll = df_ct[df_ct['Area'] == 'Lower Lobe']
        ll_nonSnC, ll_SnC, ll_pct = group_counts(df_ct_ll)

        df_ct_parenchyma = df_ct[df_ct['Area'] == 'Parenchyma']
        para_nonSnC, para_SnC, para_pct = group_counts(df_ct_parenchyma)

        # Count Young vs Old
        df_ct_young = df_ct[df_ct['Age_Status'] == 'Young']
        young_nonSnC, young_SnC, young_pct = group_counts(df_ct_young)

        df_ct_old = df_ct[df_ct['Age_Status'] == 'Old']
        old_nonSnC, old_SnC, old_pct = group_counts(df_ct_old)

        # Build a row of results
        row = {
            'CT': ct,
            '# cell': total_cells,
            '# IPF non-SnC': ipf_nonSnC, 
            '# IPF SnC': ipf_SnC, 
            'IPF SnC%': ipf_pct,
            
            '# Healthy non-SnC': healthy_nonSnC, 
            '# Healthy SnC': healthy_SnC, 
            'Healthy SnC%': healthy_pct,
            
            '# UL non-SnC': ul_nonSnC, 
            '# UL SnC': ul_SnC, 
            'UL SnC%': ul_pct,
            
            '# LL non-SnC': ll_nonSnC, 
            '# LL SnC': ll_SnC, 
            'LL SnC%': ll_pct,
            
            '# Parenchyma non-SnC': para_nonSnC, 
            '# Parenchyma SnC': para_SnC, 
            'Parenchyma SnC%': para_pct,
            
            '# Young non-SnC': young_nonSnC, 
            '# Young SnC': young_SnC, 
            'Young SnC%': young_pct,
            
            '# Old non-SnC': old_nonSnC, 
            '# Old SnC': old_SnC, 
            'Old SnC%': old_pct
        }

        # Concatenate this new row to summary_df
        summary_df = pd.concat(
            [summary_df, pd.DataFrame([row])],
            ignore_index=True
        )

    return summary_df



def generate_heatmap_snc(adata,
                         table3_path="SnGs_data2_1/data1_Gene_newTable3_gene_ct_count.csv",
                         z_score=True):
        
    df_table3=pd.read_csv(table3_path)
    
    df_sorted = df_table3.sort_values(
        by=['cell_type','gene'],
        key=lambda col: col.str.lower()
    )
    
    # Define the desired order for genes (rows) and clusters (columns)
    gene_order = df_sorted['gene']
    
    cluster_order = df_sorted['cell_type'].unique().tolist()
    
    # Define the order for cell status; here we assume you want "normal" first, then "SnC"
    status_order = ["SnC","normal"]
    
    
    # Create a DataFrame of expression values (cells x genes).
    # If adata.X is sparse, convert it to a dense array.
    if hasattr(adata.X, "toarray"):
        expr_df = pd.DataFrame(adata.X.toarray(), index=adata.obs_names, columns=adata.var.index)
    else:
        expr_df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var.index)
    
    # Add the cluster and cell status information from adata.obs.
    expr_df["cluster"] = adata.obs["clusters"].values
    expr_df["is_sen"] = adata.obs["is_sen"].values
    
    # Compute the average expression per (cluster, is_sen) combination.
    # The resulting DataFrame will have a MultiIndex: (cluster, is_sen) and columns = genes.
    grouped = expr_df.groupby(["cluster", "is_sen"]).mean()
    
    
    # Transpose so that rows are genes and columns are the MultiIndex (cluster, is_sen)
    heatmap_df = grouped.T
    heatmap_df = heatmap_df.loc[gene_order]
    # Reorder the columns:
    # For each cluster in the specified order, for each cell status in the desired order,
    # add the corresponding column if it exists.
    new_columns = []
    for cl in cluster_order:
        for st in status_order:
            if (cl, st) in heatmap_df.columns:
                new_columns.append((cl, st))
    
    heatmap_df = heatmap_df[new_columns]
    
    # (Optional) Row-normalize the data (e.g., z-score normalization across each gene)
    
    
    if z_score:
        heatmap_df_norm = heatmap_df.sub(heatmap_df.mean(axis=1), axis=0)
        heatmap_df_norm = heatmap_df_norm.div(heatmap_df.std(axis=1), axis=0)
        heatmap_df=heatmap_df_norm
    
    
    # Plot the heatmap.
    plt.figure(figsize=(len(new_columns), len(gene_order)*0.3),dpi=300)
    
    from matplotlib.colors import LinearSegmentedColormap
    # low color, medium and high color seperately
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#4F99A9','white','#D65A79'])
    # cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#54a1c9','#ea599b'])
    sns.heatmap(heatmap_df, cmap=cmap, xticklabels=True, yticklabels=True)
    
    # Flatten the MultiIndex columns for display
    plt.xticks(
        ticks=np.arange(len(new_columns)) + 0.5,
        labels=[f"{cl}\n{st}" for cl, st in new_columns],
        rotation=45,
        ha="right"
    )
    plt.xlabel("Cluster and Cell Status")
    plt.ylabel("Genes")
    plt.title("Row-normalized Mean Expression by Cluster and Cell Status")
    plt.tight_layout()
    
    plt.savefig('high_res_heatmap_snc.png', dpi=300)
    plt.show()


def plot_enrichment(df_table2_path="SnGs_data2_1/data1_Gene_Table2_DEG_ct_SnG_score.csv",
                    outdir = "enrichr_results",
                    cut_off=0.1
                    ):
    df_table2=pd.read_csv(df_table2_path)
    result_df = df_table2.groupby('cell_type')['gene'].agg(
        gene_count='count',                      # Count the number of genes per cell type
        gene_list=lambda x: list(x)              # Aggregate the genes into a list
    ).reset_index()
    
    for index_, row in result_df.iterrows():
        ct_ = row['cell_type']
        gene_list = row['gene_list']
        
        libraries = [
            "KEGG_2021_Human", 
            "GO_Biological_Process_2023", 
            "GO_Cellular_Component_2023", 
            "GO_Molecular_Function_2023"
        ]
        
        os.makedirs(outdir, exist_ok=True)
        
        for lib in libraries:
            print(f"For {ct_}, processing library: {lib}")
            
            # Run enrichment analysis with cutoff=0.05
            try:
                enr = gp.enrichr(
                    gene_list=gene_list,
                    gene_sets=lib,
                    organism="human",
                    outdir=outdir,
                    cutoff=cut_off  # Original cutoff
                )
            except ValueError as e:
                print(f"\tEnrichR error for cutoff = {cut_off}, {ct_} in {lib}: {e}")
            
            
            # Save results to CSV
            csv_path = os.path.join(outdir, f"enrichment_{ct_}_{lib}.csv")
            enr.results.to_csv(csv_path, index=False)
            print(f"\tResults saved to {csv_path}")
            
            
            # Generate a barplot of the enrichment results
            ax = gp.barplot(
                enr.results,
                title=f"Enrichment Barplot - {lib}",
                cutoff=cut_off,
                figsize=(6, 6)
            )
                
            # Save the barplot as a PNG file
            png_path = os.path.join(outdir, f"barplot_{ct_}_{lib}.png")
            plt.savefig(png_path, dpi=300, bbox_inches="tight")
            print(f"\tBarplot saved to {png_path}")
            
            plt.close()



def plot_violin(adata,
                output_dir = "violin_plots",
                table3_path="SnGs_data2_1/data1_Gene_newTable3_gene_ct_count.csv"):
    # generate violin plots for each gene-cell type pair from table 3
    
    os.makedirs(output_dir, exist_ok=True)

    # Define custom order and palette for senescence status
    order = ['SnC', 'normal']
    palette = {'SnC': '#D65A79', 'normal': '#4F99A9'}

    gene_ct_table=pd.read_csv(table3_path)

    # Loop over each gene-cell type pair to generate and save plots.
    for index,row in gene_ct_table.iterrows():
        gene = row['gene']
        cell_type = row['cell_type']
        print(gene,cell_type)
        # Subset adata to the specified cell type
        adata_subset = adata[adata.obs['clusters'] == cell_type, :]

        # Check if gene exists in the dataset
        if gene not in adata_subset.var_names:
            print(f"Gene {gene} not found in adata for cell type {cell_type}. Skipping.")
            continue

        # Extract the expression data for the gene.
        expr_data = adata_subset[:, gene].X
        if hasattr(expr_data, "toarray"):
            expr_data = expr_data.toarray().flatten()
        else:
            expr_data = np.array(expr_data).flatten()

        # Create a DataFrame for plotting.
        df = pd.DataFrame({
            'Expression': expr_data,
            'Senescence': adata_subset.obs['is_sen']
        })

        # Create the violin plot.
        plt.figure(figsize=(8, 6))
        ax = sns.violinplot(x='Senescence', y='Expression', data=df, order=order, 
                            cut=0,
                            palette=palette)

        # Compute and annotate the mean for each group.
        group_means = df.groupby('Senescence')['Expression'].mean().reindex(order)
        for i, group in enumerate(order):
            mean_val = group_means[group]
            # Adjust x position for the annotation (left offset for "snc", right for "normal")
            offset = -0.15 if group == 'snc' else 0.05
            plt.text(i + offset, mean_val, f'Mean: {mean_val:.2f}', color='black', fontweight='bold')

        # Add plot title and labels.
        plt.title(f"{gene} Expression in {cell_type}")
        plt.xlabel("Senescence Status")
        plt.ylabel("Expression Count")

        # Save the plot to a PNG file.
        # Replace spaces in cell type with underscores for filename.
        filename = f"{gene}_{cell_type.replace(' ', '_')}.png"
        filepath = os.path.join(output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()  # Close the figure to free up memory

        print(f"Saved plot for {gene} in {cell_type} to {filepath}")