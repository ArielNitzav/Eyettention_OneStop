# %%
from EZ.word import word_DVs, word_means
from EZ.sentence import Sentence
from EZ.fixation import Fixation

import os
import pandas as pd
from tqdm import tqdm

# %%
word_info_df = pd.read_csv('/data/home/shared/onestop/OneStop_v1_20250126/lacclab_processed/ia_Paragraph.csv', engine="pyarrow")
word_info_df = word_info_df[word_info_df["article_id"] != 0] # filter out practice article
word_info_df = word_info_df[word_info_df["repeated_reading_trial"] == 0] # use only non repeated reading
word_info_df = word_info_df[word_info_df["question_preview"] == 0] # use only non question preview cases
word_info_df['IA_LABEL'] = word_info_df.IA_LABEL.replace('\t(.*)', '', regex=True)

# %%
paragraphs = word_info_df.groupby(['unique_paragraph_id', 'text_spacing_version'])['word_length'].apply(list).reset_index()
paragraphs

# %%
# Define the folder path
folder_path = "results/Eyettention"

# Get a list of all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Dictionary to store DataFrames
eyettention_outputs = {}

# Loop through each CSV file and load it into a DataFrame
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    eyettention_outputs[file] = df  # Store DataFrame in dictionary with filename as key

# %%
paragraphs_dict = {}
for i, row in paragraphs.iterrows():
    paragraphs_dict[(row["unique_paragraph_id"], row["text_spacing_version"])] = Sentence(i, row["word_length"])

# %%
paragraphs_dict[list(paragraphs_dict.keys())[0]].subj_number

# %%
def split_df_on_zero(df, column_name='fix_id'):
    # Find indices where the column equals 0
    df = df.reset_index()
    split_indices = df.index[df[column_name] == 0].tolist()
    # Add end index
    split_indices.append(len(df))
    # Create a list of DataFrames
    dfs = [df.iloc[split_indices[i]:split_indices[i+1]].reset_index(drop=True) for i in range(len(split_indices)-1)]
    return dfs

# %%
fold_dict = {}
for fold_output in sorted(eyettention_outputs.keys()):
    processed_fold = {}
    output_df = eyettention_outputs[fold_output]
    grouped_dfs = {key: group for key, group in output_df.groupby(['unique_paragraph_id', 'text_spacing_version'])}
    grouped_dfs = {key: split_df_on_zero(group) for key, group in grouped_dfs.items()}
    for key, group in grouped_dfs.items():
        processed_group = []
        for df in group:
            processed_df = []
            for i, row in df.iterrows():
                processed_df.append(Fixation(row["sp_fix_dur"], row["fix_id"], row["sp_fix_pos"]))
            processed_group.append(processed_df)
        processed_fold[key] = processed_group
    fold_dict[fold_output] = processed_fold


# %%
fold_dict_copy = fold_dict.copy()

# %%
final_fold_dict = {}
for file_name, combo_dict in fold_dict_copy.items():
    print(f"Started {file_name}")
    final_combo_dict = {}
    for combo, fix_lists in tqdm(combo_dict.items()):
        paragraph = paragraphs_dict[combo]
        adjusted_paragraph = paragraph
        for fix_list in fix_lists:
            adjusted_paragraph.subj_number += 1
            adjusted_paragraph = word_DVs(adjusted_paragraph, fix_list)
        word_means(adjusted_paragraph)
        final_combo_dict[combo] = adjusted_paragraph
    final_fold_dict['file_name'] = final_combo_dict

# %%
import pickle

with open("final_fold_ia_dict.pkl", 'wb') as file:
    pickle.dump(final_fold_dict, file)


