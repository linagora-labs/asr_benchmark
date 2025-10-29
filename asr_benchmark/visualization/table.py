import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

def df_to_wer_table_format(df):
    """deprecated"""
    def compute_weighted_wer(wer_details):
        if not isinstance(wer_details, list):
            return None
        total_del = total_ins = total_sub = total_ref = 0

        for entry in wer_details:
            count = entry.get('count')
            total_del += entry.get('del') * count
            total_ins += entry.get('ins') * count
            total_sub += entry.get('sub') * count
            total_ref += entry.get('count')

        if total_ref == 0:
            return None

        return (total_del + total_ins + total_sub) / total_ref

    df['wer_computed'] = df['wer_details'].apply(compute_weighted_wer)

    wer_pivot = df.pivot(index='model', columns='dataset', values='wer_computed')
    
    return wer_pivot

def reorder_wer_pivot(wer_pivot):
    # Copy to avoid modifying the original
    wer_pivot = wer_pivot.copy()

    # Extract model names
    models = wer_pivot.index.to_list()

    linagora_models = [m for m in models if m.lower().startswith('linagora')]
    linto_models = [m for m in models if m.lower().startswith('linto')]
    openai_whisper_models = [m for m in models if 'whisper' in m.lower() and m.lower().startswith('openai')]
    other_whisper_models = [m for m in models if 'whisper' in m.lower() and m not in linagora_models + openai_whisper_models]
    nvidia = [m for m in models if m.lower().startswith('nvidia')]
    other_models = [m for m in models if m not in linagora_models + linto_models + openai_whisper_models + other_whisper_models + nvidia]

    ordered_models = linagora_models + openai_whisper_models + other_whisper_models + nvidia + other_models + linto_models

    # Reorder the DataFrame
    return wer_pivot.loc[ordered_models]

def prepare_model_name(name):
    if 'whisper' in name.lower() and '/' not in name:
        name = 'OpenAI/' + name
    elif 'linto' in name.lower() and '/' not in name:
        name = name.replace("linto-", "LinTO/")
    elif 'linto' in name.lower():
        name = name.replace("linto-", "LinTO/")
        name = name.replace("linagora/", "")
    elif 'finetuned' in name.lower() and '/' not in name:
        name = 'LINAGORA/' + name
        name = name.replace("-finetuned", "")
    elif 'linagora' in name.lower() and '/' not in name:
        name = name.replace("linagora_", "LINAGORA/")
    name = name.replace("stt-fr-", "")
    if '/' in name:
        parts = name.split('/')
        parts[0] = parts[0].upper()  # Capitalize the first part of the model name
        return '\n'.join(parts)
    return name.capitalize()

def plot_wer_table(wer_means, wer_stds=None, output_filename='wer_table.png', show=True, y_label="WER (%)", best="lowest", color_lims=(0,50)):
    wer_means = wer_means.copy()
    n_rows, n_cols = wer_means.shape

    if wer_stds is not None:
        wer_stds = pd.DataFrame(wer_stds, index=wer_means.index, columns=wer_means.columns)

    fig, axis = plt.subplots(figsize=(1.8 * n_cols if n_cols>4 else 3*n_cols, 1.2 * n_rows))
    if best=="highest":
        min_indices = wer_means.idxmax()
        color_map = mcolors.LinearSegmentedColormap.from_list('purple_red_yellow_green', ['purple', 'red', 'yellow', 'green'])        
    else:
        min_indices = wer_means.idxmin()
        color_map = mcolors.LinearSegmentedColormap.from_list('green_red_purple', ['green', 'yellow', 'red', 'purple'])
    normalizer = mcolors.Normalize(vmin=color_lims[0], vmax=color_lims[1])

    for i in range(n_rows):
        for j in range(n_cols):
            val = wer_means.iat[i, j]

            if not pd.isna(val):
                color = color_map(normalizer(val))
                rect = plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor='gray')
                axis.add_patch(rect)

                fontweight = 'bold' if wer_means.index[i] == min_indices[j] else 'normal'
                axis.text(j + 0.5, i + 0.4 if wer_stds is not None else i + 0.5, f"{val:.2f}", ha='center', va='center',
                          fontsize=12, weight=fontweight, color='black')

                if wer_stds is not None:
                    std_val = wer_stds.iat[i, j]
                    if not pd.isna(std_val):
                        axis.text(j + 0.5, i + 0.7, f"±{std_val:.2f}", ha='center', va='center',
                                  fontsize=9, style='italic', color='black')

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
    cbar.set_label(y_label, fontsize=12)

    plt.tight_layout()
    if output_filename:
        plt.savefig(output_filename)
    if show:
        plt.show()

def generate_markdown_table_with_std(df_mean, highlight_best=True):
    def format_cell(value, is_best=False):
        if pd.isna(value):
            return ""
        formatted = f"{value:.2f}"
        return f"**{formatted}**" if is_best else formatted

    best_per_column = df_mean.min() if highlight_best else pd.Series([None]*len(df_mean.columns), index=df_mean.columns)

    headers = ["Model"] + list(df_mean.columns)
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "|:--" + "|:--:" * (len(headers) - 1) + "|"

    body_rows = []
    max_length = max(len(str(idx)) for idx in df_mean.index)

    for idx in df_mean.index:
        formatted_cells = []
        for col in df_mean.columns:
            mean = df_mean.loc[idx, col]
            is_best = highlight_best and mean == best_per_column[col]
            formatted_cells.append(format_cell(mean, is_best))
        row_str = "| " + f"{idx}".ljust(max_length) + " | " + " | ".join(cell.ljust(12) for cell in formatted_cells) + " |"
        body_rows.append(row_str)

    md_table = "\n".join([header_row, separator_row] + body_rows)
    return md_table