import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

def df_to_wer_table_format(df):
    
    df_filtered = df[['model', 'dataset', 'wer']]
    df_filtered['wer_mean'] = df_filtered['wer'].apply(lambda x: np.mean(x) if isinstance(x, list) and x else None)
    df_filtered['wer_std'] = df_filtered['wer'].apply(lambda x: np.std(x) if isinstance(x, list) and x else None)
    df_filtered['wer_ci95'] = df_filtered['wer'].apply(
        lambda x: 1.96 * np.std(x, ddof=1) / np.sqrt(len(x)) if isinstance(x, list) and len(x) > 1 else None
    )

    # Pivot so each model is a column, each dataset is a row, values are WERs
    wer_pivot = df_filtered.pivot(index='model', columns='dataset', values='wer_mean')
    wer_std_pivot = df_filtered.pivot(index='model', columns='dataset', values='wer_std')
    wer_conf_pivot = df_filtered.pivot(index='model', columns='dataset', values='wer_ci95')
    
    return wer_pivot, wer_std_pivot, wer_conf_pivot

def reorder_wer_pivot(wer_pivot):
    # Copy to avoid modifying the original
    wer_pivot = wer_pivot.copy()

    # Extract model names
    models = wer_pivot.index.to_list()

    linagora_models = [m for m in models if m.lower().startswith('linagora')]
    openai_whisper_models = [m for m in models if 'whisper' in m.lower() and m.lower().startswith('openai')]
    other_whisper_models = [m for m in models if 'whisper' in m.lower() and m not in linagora_models + openai_whisper_models]
    other_models = [m for m in models if m not in linagora_models + openai_whisper_models + other_whisper_models]

    ordered_models = linagora_models + openai_whisper_models + other_whisper_models + other_models

    # Reorder the DataFrame
    return wer_pivot.loc[ordered_models]

def prepare_model_name(name):
    if 'whisper' in name.lower() and '/' not in name:
        name = 'OpenAI/' + name
    if 'finetuned' in name.lower() and '/' not in name:
        name = 'LINAGORA/' + name
        name = name.replace("-finetuned", "")
    name = name.replace("stt-fr-", "")
    if '/' in name:
        parts = name.split('/')
        parts[0] = parts[0].upper()  # Capitalize the first part of the model name
        return '\n'.join(parts)
    return name.capitalize()

def plot_wer_table(wer_means, wer_ci95, output_filename='wer_table.png', show=True):
    wer_means = wer_means.copy()
    wer_ci95 = wer_ci95.copy()
    n_rows, n_cols = wer_means.shape

    # Colormap : vert → jaune → rouge → violet
    color_map = mcolors.LinearSegmentedColormap.from_list('green_red_purple', ['green', 'yellow', 'red', 'purple'])
    normalizer = mcolors.Normalize(vmin=0, vmax=50)

    fig, axis = plt.subplots(figsize=(1.8 * n_cols, 1.2 * n_rows))

    # Trouver les indices des minimas par colonne
    min_indices = wer_means.idxmin()

    for i in range(n_rows):
        for j in range(n_cols):
            val = wer_means.iat[i, j]
            ci_val = wer_ci95.iat[i, j]

            if not pd.isna(val):
                color = color_map(normalizer(val))
                rect = plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor='gray')
                axis.add_patch(rect)

                fontweight = 'bold' if wer_means.index[i] == min_indices[j] else 'normal'

                # Afficher la valeur WER
                axis.text(j + 0.5, i + 0.5, f"{val:.2f}", ha='center', va='center',
                          fontsize=12, weight=fontweight, color='black')

                # Afficher le CI95
                if not pd.isna(ci_val):
                    axis.text(j + 0.5, i + 0.7, f"± {ci_val:.2f}", ha='center', va='center',
                              fontsize=10, color='black')

    # Ajuster les axes
    axis.set_xlim(0, n_cols)
    axis.set_ylim(0, n_rows)
    axis.set_xticks(np.arange(n_cols) + 0.5)
    axis.set_yticks(np.arange(n_rows) + 0.5)

    axis.set_yticklabels(wer_means.index, fontsize=12, color='black')
    axis.set_xticklabels(wer_means.columns, rotation=0, ha='center', fontsize=12, color='black')

    axis.invert_yaxis()
    axis.xaxis.tick_top()
    axis.tick_params(length=0)

    for spine in axis.spines.values():
        spine.set_visible(False)

    sm = plt.cm.ScalarMappable(cmap=color_map, norm=normalizer)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axis, orientation='vertical', shrink=0.6, pad=0.01)
    cbar.set_label("WER (%)", fontsize=12)

    plt.tight_layout()
    if output_filename:
        plt.savefig(output_filename)
    if show:
        plt.show()

def generate_markdown_table_with_std(df_mean, df_std, highlight_best=True):
    # Format numbers nicely: "mean ± std"
    def format_cell(mean, std, is_best=False):
        if pd.isna(mean):
            return ""
        formatted = f"{mean:.2f} ± {std:.2f}" if not pd.isna(std) else f"{mean:.2f}"
        return f"**{formatted}**" if is_best else formatted

    # Identify best (lowest) mean WER per column
    best_per_column = df_mean.min() if highlight_best else pd.Series([None]*len(df_mean.columns), index=df_mean.columns)

    # Header
    headers = ["Model"] + list(df_mean.columns)
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "|:--" + "|:--:" * (len(headers) - 1) + "|"

    # Body rows
    body_rows = []
    max_length = max(len(str(idx)) for idx in df_mean.index)

    for idx in df_mean.index:
        formatted_cells = []
        for col in df_mean.columns:
            mean = df_mean.loc[idx, col]
            std = df_std.loc[idx, col]
            is_best = highlight_best and mean == best_per_column[col]
            formatted_cells.append(format_cell(mean, std, is_best))
        row_str = "| " + f"{idx}".ljust(max_length) + " | " + " | ".join(cell.ljust(12) for cell in formatted_cells) + " |"
        body_rows.append(row_str)

    # Combine all
    md_table = "\n".join([header_row, separator_row] + body_rows)
    return md_table