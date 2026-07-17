import pandas as pd
import numpy as np
from sklearn import metrics
from itertools import combinations

import matplotlib.pyplot as plt
import seaborn as sns

### ANALYSIS FUNCTIONS
def add_barcode_rank_info(df, groupby_cols, count_col='n'):
    df = df.sort_values(count_col, ascending=False)

    # rank by reads
    df['total_reads'] = df.groupby(groupby_cols)[[count_col]].transform('sum')
    df['umi_read_perc'] = df[count_col] / df.total_reads
    df['cum_read_perc'] = df.groupby(groupby_cols)\
                            [['umi_read_perc']].transform('cumsum')

    # rank by unique UMIs
    df['n_umis'] = df.groupby(groupby_cols).transform('size')
    df['umi_rank'] = df.groupby(groupby_cols).transform('cumcount')
    df['umi_perc'] = df.umi_rank / df.n_umis

    # number of unique UMIs that make up 90% of the (cumulative) reads
    n_top_umis = df[df.cum_read_perc <= 0.9]\
                    .groupby(groupby_cols)[['umi_rank']]\
                    .max().reset_index()\
                    .rename(columns={'umi_rank':'n_top_umis'})
    
    df = df.merge(n_top_umis, on=groupby_cols)
    
    return df

def bin_cum_readcounts(df,
                       x_col='umi_perc', 
                       y_col='cum_read_perc',
                       n_steps=100):
    binned_dict = {x_col:[],
                   y_col:[]}
    
    for i in range(0, n_steps):
        query_range = f"{x_col} >= {i/n_steps} & {x_col} < {(i+1)/n_steps}"

        binned_dict[x_col].append(i/n_steps)
        binned_dict[y_col].append(
            df.query(query_range)[y_col].mean()
        )
    
    return pd.DataFrame(binned_dict).set_index(x_col)

def auc_group(df, x_col='umi_perc', y_col='cum_read_perc'):
    if df.shape[0] < 2:
        return np.nan

    y_hat = df[x_col]
    y = df[y_col]
    return pd.Series({'auc':metrics.auc(y_hat, y)})

def create_pairwise_jaccard_df(df,
                               compare_col='tech_rep'):
    jaccard_df_dict = dict({
            'rep1':[],
            'rep2':[],
            'jaccard':[],
            'top_90_jaccard':[],
            'n_intersect':[],
            'n_top_90_intersect':[],
            'n_symm_diff':[],
            'mean_intersect_ct':[],
            'mean_symm_diff_ct':[]
        })

    pairwise_combos = combinations(df[compare_col].unique(), 2)
    for combo in pairwise_combos:
        jaccard_df_dict['rep1'].append(combo[0])
        jaccard_df_dict['rep2'].append(combo[1])

        ## All UMIs
        rep1_umis = set(
                        df[df[compare_col] == combo[0]].umi.unique()
                    )
        rep2_umis = set(
                        df[df[compare_col] == combo[1]].umi.unique()
                    )
        
        intersect = set.intersection(rep1_umis, rep2_umis)
        union = set.union(rep1_umis, rep2_umis)

        ## top 90% UMIs
        top_90_rep1_umis = set(
                        df[(df[compare_col] == combo[0])
                           & (df.cum_read_perc <= 0.9)]\
                        .umi.unique()
                    )
        top_90_rep2_umis = set(
                        df[(df[compare_col] == combo[1])
                           & (df.cum_read_perc <= 0.9)]\
                        .umi.unique()
                    )
        
        top_90_intersect = set.intersection(top_90_rep1_umis, top_90_rep2_umis)
        top_90_union = set.union(top_90_rep1_umis, top_90_rep2_umis)
        
        jaccard_df_dict['jaccard'].append(
            len(intersect) / len(union)
        )
        jaccard_df_dict['top_90_jaccard'].append(
            len(top_90_intersect) / len(top_90_union)
        )
        jaccard_df_dict['n_intersect'].append(
            len(intersect)
        )
        jaccard_df_dict['n_top_90_intersect'].append(
            len(top_90_intersect)
        )
        jaccard_df_dict['n_symm_diff'].append(
            len(union - intersect)
        )
        jaccard_df_dict['mean_intersect_ct'].append(
            df[df.umi.isin(list(intersect))].n.mean()
        )
        jaccard_df_dict['mean_symm_diff_ct'].append(
            df[df.umi.isin(list(union - intersect))].n.mean() 
        )

    return pd.DataFrame(jaccard_df_dict)

### PLOTTING FUNCTIONS
def draw_xy_line(ax, color='#aaa'):
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, alpha=0.75, zorder=2, linestyle='--', color=color);


def plot_overview_by_organ(cts, aucs,
                           organ_list = None,
                           hue_column = 'bio_rep',
                           palette='Set2',
                           replicate_column = 'Sample_Name',
                           #total reads barplot
                           total_reads_column = 'total_reads',
                           #umi representation curve
                           umi_rank_column = 'umi_perc',
                           cumulative_reads_column = 'cum_read_perc',
                           #AUC scatterplot
                           n_umis_column = 'n_top_umis',
                           auc_column='auc',
                           title=None
                           ):
    if organ_list == None:
        organ_list = sorted(cts.organ.unique())

    f, axes = plt.subplots(3, len(organ_list),
                           figsize=(7*len(organ_list), 12),
                          height_ratios=[2,3,1])
    if title:
        f.suptitle(title, fontsize=18)
    f.set_facecolor('white')
    max_total_reads = cts[total_reads_column].max()
    min_aucs = aucs[auc_column].min()
    min_n_umis = aucs[n_umis_column].min()
    if min_n_umis == 0: min_n_umis = 1
    max_n_umis = aucs[n_umis_column].max()

    for i, organ in enumerate(organ_list):
        cts_sub = cts.loc[cts.organ == organ]\
                        .sort_values(replicate_column)
        auc_sub = aucs.loc[aucs.organ == organ]\
                        .sort_values(replicate_column)\
                        .assign(constant='A')
        auc_sub.loc[auc_sub[n_umis_column]==0, n_umis_column] = 1

        #Total Reads Barplot
        plt.sca(axes[0][i])
        plt.title(f"{organ}", fontsize=16)    
        sns.barplot(data=cts_sub.loc[:,
                                    list(set([replicate_column, hue_column, total_reads_column]))]\
                            .drop_duplicates().sort_values(replicate_column),
                    x=replicate_column,
                    y=total_reads_column,
                    hue=hue_column, palette=palette, dodge=False)
        plt.xticks([]); plt.yticks(fontsize=12)
        plt.xlabel('Replicate', fontsize=14); plt.ylabel('Total Counts', fontsize=14)
        plt.yscale('log')
        plt.ylim(1, max_total_reads)
        plt.legend().remove()

        # UMI Representation curve
        plt.sca(axes[1][i])

        #bin readcounts to plot faster
        cts_sub_binned = cts_sub.groupby(['organ', hue_column])\
                            .apply(bin_cum_readcounts)\
                            .reset_index()\
                            .sort_values(hue_column)

        sns.lineplot(data=cts_sub_binned,
                     x=umi_rank_column,
                     y=cumulative_reads_column,
                     hue=hue_column, palette=palette,
                     estimator='mean', err_style='band', errorbar=('sd', 2)
                     )
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        draw_xy_line(axes[1][i])
        plt.axhline(0.9, linestyle='--', color='#aaa')
        plt.xlabel('Umi Rank (% of total)', fontsize=14)
        plt.ylabel('% of Cumulative Reads', fontsize=14)
        plt.xticks(fontsize=12); plt.yticks(fontsize=12)
        plt.legend(loc='lower right')

        # AUC vs. # of UMIs scatterplot
        plt.sca(axes[2][i])
        
        sns.boxplot(
            data=auc_sub,
            hue='constant',
            x=n_umis_column,
            palette=palette,
            fill=False, showfliers=False
        )
        sns.stripplot(
            data=auc_sub,
            x=n_umis_column,
            hue=hue_column,
            palette=palette,
            s=10,
            linewidth=1, edgecolor='darkgrey'
        )
        plt.legend().remove()
        plt.xlim((min_n_umis*0.9), (max_n_umis*1.1))
        if max_n_umis >= 1000: plt.xscale('log')
        plt.xlabel('# of UMIs in top 90% of Cumulative Reads', fontsize=14)
        plt.xticks(fontsize=12); plt.yticks([])
        plt.grid(axis='x')

    plt.tight_layout()
