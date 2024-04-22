import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

from ssak.utils.wer import plot_wer

LABEL_FONTSIZE = 15     

def prepare_data(df, return_format="df", target_wer_column="model"):
    df = df.copy()
    if len(df) == 0:
        print("ERROR: No data to plot")
        return False
    mask = df.map(type) != bool
    d = {True: "TRUE", False: "FALSE"}
    df = df.where(mask, df.replace(d))
    if return_format == "df":
        return df
    elif return_format == "dict":
        wer_scores = {}
        for idx, row in df.iterrows():
            if row["wer_details"] is None or not row["wer_details"]:
                return False
            wer_scores[row[target_wer_column]] = row["wer_details"]
        return wer_scores
    raise Exception("Invalid return_format")

def plot_wer_df(wer_scores, output_folder, x_column="model", title="wer", save_fig=None):
    plt.close()
    plt.figure(figsize=(10, 6))
    if save_fig==True:
        save_fig = f'{title.replace(" ", "_")}'
    plot_wer(wer_scores, interval_type="none", show=os.path.join(output_folder, f'{save_fig}.png') if save_fig else None, title=f"{title.upper()}", sort_best=0, label_rotation=45, scale=1, label_fontdict={'size': LABEL_FONTSIZE}, ymax=100, x_axisname=x_column.upper())
    plot_wer(wer_scores, interval_type="none", show=os.path.join(output_folder, f'{save_fig}_zoomed.png') if save_fig else None, title=f"{title.upper()}", sort_best=0, label_rotation=45, scale=1, label_fontdict={'size': LABEL_FONTSIZE}, ymax=40,x_axisname=x_column.upper())
    plt.close()

def add_plot_detail(output_folder, x_column="VRAM usage", y_column="model", title=None, save_fig=None, limit=None, xlabel=None, ylabel=None):
    plt.xticks(fontsize=LABEL_FONTSIZE)
    plt.yticks(fontsize=LABEL_FONTSIZE)
    if xlabel==True:
        plt.xlabel(x_column.upper(), fontsize=LABEL_FONTSIZE)
    elif xlabel:
        plt.xlabel(xlabel, fontsize=LABEL_FONTSIZE)
    else:
        plt.xlabel("")
    if ylabel==True:
        plt.ylabel(y_column.upper(), fontsize=LABEL_FONTSIZE)
    elif ylabel:
        plt.ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    else:
        plt.ylabel("")
    if limit:
        if isinstance(limit, tuple):
            plt.xlim(limit)
        else:
            plt.xlim(0, limit)
    else:
        plt.xlim(0, None)
    if title:
        plt.title(title, fontsize=LABEL_FONTSIZE)
    else:
        plt.title("")
    plt.tight_layout()
    if save_fig:
        if isinstance(save_fig, str):
            plt.savefig(os.path.join(output_folder, save_fig))
        else:
            plt.savefig(os.path.join(output_folder, f'{y_column}_by_{x_column}.png')) 
        plt.close()

def plot_violin_df(df, output_folder, x_column="rtf", y_column="model", title=None, save_fig=None, limit=None, xlabel=None, ylabel=None):
    df = df.copy()
    rtf_list= []
    for idx, row in df.iterrows():
        rtf_list.append(list(row['process_duration'][i]/row['audio_duration'][i] for i in range(len(row['audio_duration']))))
    df['rtf'] = rtf_list
    if "/" in y_column:
        c = y_column.split("/")
        y_column = y_column.replace("/", "_")
        df[y_column] = df[c[0]] + " " + df[c[1]]
    df = df.explode(x_column)
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x=x_column, y=y_column)
    plt.grid(True)
    add_plot_detail(output_folder=output_folder, x_column=x_column, y_column=y_column, title=title, save_fig=save_fig, limit=limit, xlabel=xlabel, ylabel=ylabel)
    
def plot_bar_df(df, output_folder, x_column="VRAM usage", y_column="model", title=None, save_fig=None, limit=None, xlabel=None, ylabel=None, hue=None):
    rtf_list= []
    import numpy as np
    for idx, row in df.iterrows():
        rtf_list.append(round(np.mean(list(row['process_duration'][i]/row['audio_duration'][i] for i in range(len(row['audio_duration'])))), 3))
    df['rtf'] = rtf_list
    df['rtfx'] = [round(1/i, 1) for i in rtf_list]
    # df = df.explode()
    if "/" in y_column:
        c = y_column.split("/")
        y_column = y_column.replace("/", "_")
        df[y_column] = df[c[0]] + " " + df[c[1]]
    if save_fig==True:
        save_fig = f"{y_column}_by_{x_column}"  
    fig = plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df, x=x_column, y=y_column, hue=hue)
    ax.bar_label(ax.containers[0])
    add_plot_detail(output_folder=output_folder, x_column=x_column, y_column=y_column, title=title, save_fig=save_fig, limit=limit, xlabel=xlabel, ylabel=ylabel)