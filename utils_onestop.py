#coding=utf-8
import numpy as np
import pandas as pd
import os
import seaborn as sns
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
from torchaudio.functional import edit_distance
from sklearn.metrics import r2_score
from transformers import BertTokenizerFast, BertTokenizer, BertForTokenClassification
import torch
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from collections import Counter
import torch.nn as nn
from typing import List, Tuple


def get_fold_indices(i: int) -> Tuple[List[int], List[int], List[int]]:
	"""
	Given a fold index i within the range [0, 9], return the indices for the test,
	validation, and training sets according to the specified folding strategy.

	Parameters:
	i (int): The fold index (should be between 0 and 9).

	Returns:
	tuple: A tuple containing the test indices (as a list), validation index (as an integer),
	and training indices (as a list).
	"""
	num_folds = 10
	if i < 0 or i > num_folds - 1:
		raise ValueError("Fold index must be within the range [0, 9].")

	validation_indices = [i]

	# modulo num_folds for the wraparound
	test_indices = [(i + 1) % num_folds]
	'''
	if self.use_double_test_size:
		test_indices.append((i + 2) % self.num_folds)
	'''

	# The rest are training indices
	train_indices = [
		x
		for x in range(num_folds)
		if x not in test_indices and x not in validation_indices
	]

	test_indices = validation_indices + test_indices
	validation_indices = [train_indices[-1]]
	train_indices = train_indices[:-1]

	print(
	f"Test folds: {test_indices}, \
	Validation fold: {validation_indices}, \
	Train folds: {train_indices}"
	)

	return test_indices, validation_indices, train_indices

def load_fold_subjects():
    fold_dict = {}
    fold_sub_dict = {}
    for i in range(10):
        # Account for folds
        fold_dict[i] = [f'{j+1}_{i+1}' for j in range(3)]
        # Account for fold -> subjects
        df = pd.read_csv(f"CSD/folds_RereadStratified/subjects/fold_{i}.csv")
        sub_list = df.iloc[:, 0].values.tolist()
        fold_sub_dict[i] = sub_list

    return fold_dict, fold_sub_dict

def process_dataset(fold_indx, saf_df, dataset="train"):
	sub_saf_df = saf_df[(saf_df["fold"] == fold_indx) & (saf_df["eval_type"] == dataset)]
	ba_to_participant_id = sub_saf_df.groupby("batch_id_article_id")["participant_id"].apply(set).apply(list).to_dict()

	return ba_to_participant_id

def load_corpus(corpus, include_repeated_reading=False, has_preview=False):
	if corpus == 'celer':
		eyemovement_df = pd.read_csv('./Data/celer/data_v2.0/sent_fix.tsv', delimiter='\t')
		eyemovement_df['CURRENT_FIX_INTEREST_AREA_LABEL'] = eyemovement_df.CURRENT_FIX_INTEREST_AREA_LABEL.replace('\t(.*)', '', regex=True)
		word_info_df = pd.read_csv('./Data/celer/data_v2.0/sent_ia.tsv', delimiter='\t')
		word_info_df['IA_LABEL'] = word_info_df.IA_LABEL.replace('\t(.*)', '', regex=True)
		return word_info_df, None, eyemovement_df
	elif corpus == 'onestop':
		eyemovement_df = pd.read_csv('/data/home/shared/onestop/OneStop_v1_20250126/lacclab_processed/fixations_Paragraph.csv', engine="pyarrow")
		eyemovement_df = eyemovement_df[eyemovement_df["article_id"] != 0] # filter out practice article
		if not include_repeated_reading:
			eyemovement_df = eyemovement_df[eyemovement_df["repeated_reading_trial"] == 0] # use only non repeated reading
		if not has_preview:
			eyemovement_df = eyemovement_df[eyemovement_df["question_preview"] == 0] # use only non question preview cases

		filter_df = pd.read_csv("reread_participant_article_combinations.csv")
		eyemovement_df["batch_id_article_id"] = eyemovement_df["article_batch"].astype(str) + '_' + eyemovement_df["article_id"].astype(str)
		eyemovement_df = eyemovement_df[~eyemovement_df.set_index(['participant_id', 'batch_id_article_id']).index.isin(filter_df.set_index(['participant_id', 'batch_id_article_id']).index)]

		#eyemovement_df['participant_id'] = eyemovement_df.participant_id.apply(lambda x: int(x.split("_")[1])) # separate participant_id from session
		word_info_df = pd.read_csv('/data/home/shared/onestop/OneStop_v1_20250126/lacclab_processed/ia_Paragraph.csv', engine="pyarrow")
		word_info_df = word_info_df[word_info_df["article_id"] != 0] # filter out practice article
		if not include_repeated_reading:
			word_info_df = word_info_df[word_info_df["repeated_reading_trial"] == 0] # use only non repeated reading
		if not has_preview:
			word_info_df = word_info_df[word_info_df["question_preview"] == 0] # use only non question preview cases
		word_info_df['IA_LABEL'] = word_info_df.IA_LABEL.replace('\t(.*)', '', regex=True)
		#word_info_df['participant_id'] = word_info_df.participant_id.apply(lambda x: int(x.split("_")[1])) # separate participant_id from session

		return word_info_df, None, eyemovement_df

def pad_seq(seqs, max_len, pad_value, dtype=np.compat.long):
	padded = np.full((len(seqs), max_len), fill_value=pad_value, dtype=dtype)
	for i, seq in enumerate(seqs):
		padded[i, 0] = 0
		padded[i, 1:(len(seq)+1)] = seq
		if pad_value !=0:
			padded[i, len(seq)+1] = pad_value -1

	return padded

def pad_seq_with_nan(seqs, max_len, dtype=np.compat.long):
	padded = np.full((len(seqs), max_len), fill_value=np.nan, dtype=dtype)
	for i, seq in enumerate(seqs):
		padded[i, 1:(len(seq)+1)] = seq
	return padded


def calculate_mean_std(dataloader, feat_key, padding_value=0, scale=1):
	#calculate mean
	total_sum = 0
	total_num = 0
	for batchh in dataloader:
		batchh.keys()
		feat = batchh[feat_key]/scale
		feat = torch.nan_to_num(feat)
		total_num += len(feat.view(-1).nonzero())
		total_sum += feat.sum()
	feat_mean = total_sum / total_num
	#calculate std
	sum_of_squared_error = 0
	for batchh in dataloader:
		batchh.keys()
		feat = batchh[feat_key]/scale
		feat = torch.nan_to_num(feat)
		mask = ~torch.eq(feat, padding_value)
		sum_of_squared_error += (((feat - feat_mean).pow(2))*mask).sum()
	feat_std = torch.sqrt(sum_of_squared_error / total_num)
	return feat_mean, feat_std

def load_label(sp_pos, cf, labelencoder, device):
	#prepare label and mask
	pad_mask = torch.eq(sp_pos[:, 1:], cf["max_sn_len"])
	end_mask = torch.eq(sp_pos[:, 1:], cf["max_sn_len"]-1)
	mask = pad_mask + end_mask
	sac_amp = sp_pos[:, 1:] - sp_pos[:, :-1]
	label = sp_pos[:, 1:]*mask + sac_amp*~mask
	label = torch.where(label>cf["max_sn_len"]-1, cf["max_sn_len"]-1, label).to('cpu').detach().numpy()
	label = labelencoder.transform(label.reshape(-1)).reshape(label.shape[0], label.shape[1])
	if device == 'cpu':
		pad_mask = pad_mask.to('cpu').detach().numpy()
	else:
		label = torch.from_numpy(label).to(device)
	return pad_mask, label

def likelihood(pred, label, mask):
	#test
	#res = F.nll_loss(torch.tensor(pred), torch.tensor(label))
	label = one_hot_encode(label, pred.shape[1])
	res = np.sum(np.multiply(pred, label), axis=1)
	res = np.sum(res * ~mask)/np.sum(~mask)
	return res

def eval_log_llh(dnn_out, label, pad_mask):
	res = []
	dnn_out = np.log2(dnn_out + 1e-10)
	#For each scanpath calculate the likelihood and then find the average
	for sp_indx in range(dnn_out.shape[0]):
		out = likelihood(dnn_out[sp_indx, :, :], label[sp_indx, :], pad_mask[sp_indx, :])
		res.append(out)

	return res

def NLD(ref, pred):
    return [edit_distance(r, p)/max((len(r), len(p))) for r, p in zip(ref, pred)]

def NLD_benchmark(sp_list):
	res = []
	for i in range(len(sp_list)):
		for j in range(i):
			if i != j:
				l1 = sp_list[i]
				l2 = sp_list[j]
				res.append(edit_distance(l1, l2)/max((len(l1), len(l2))))

	return res

def R2_benchmark(sp_list):
	res = []
	for i in range(len(sp_list)):
		for j in range(len(sp_list)):
			if i != j:
				l1 = sp_list[i]
				l2 = sp_list[j]
				limit = min(len(l1), len(l2))
				res.append(r2_score(l1[:limit], l2[:limit]))

	return res

def prepare_scanpath(sp_dnn, reg_sp_dnn, sn_len, sp_human, cf, ):
	max_sp_len = sp_dnn.shape[1]
	sp_human_cut = []

	#stop_indx = [np.where(sp_dnn[i,:]==(sn_len[i]+1))[0][0] for i in range(sp_dnn.shape[0])]
	#Find the number "sn_len+1" -> the end point
	stop_indx = []
	for i in range(sp_dnn.shape[0]):
		stop = np.where(sp_dnn[i, :]==(sn_len[i]+1))[0] # first place where sn_len actually stops
		if len(stop) == 0: # no end point can be find -> exceeds the maximum length of the generated scanpath
			stop_indx.append(max_sp_len - 1)
		else:
			stop_indx.append(stop[0])

	#Truncating data after the end point
	sp_dnn_cut = [sp_dnn[i][:stop_indx[i]+1] for i in range(sp_dnn.shape[0])]
	#replace the last teminal number to cf["max_sn_len"]-1, keep the same as the human scanpath label
	for i in range(len(sp_dnn_cut)):
		sp_dnn_cut[i][-1] = cf["max_sn_len"]-1

	if sp_human:
		sp_human = sp_human.detach().to('cpu').numpy() 
		#process the human scanpath data, truncating data after the end point
		stop_indx = [np.where(sp_human[i,:]==cf["max_sn_len"]-1)[0][0] for i in range(sp_human.shape[0])]
		sp_human_cut = [sp_human[i][:stop_indx[i]+1] for i in range(sp_human.shape[0])]
	
	#truncate fixation durations
	reg_sp_dnn = [reg_sp[:len(dnn_sp)] for reg_sp, dnn_sp in zip(reg_sp_dnn, sp_dnn_cut)]
	
	return sp_dnn_cut, reg_sp_dnn, sp_human_cut

def ez_reader_formatter(unique_paragraph_ids, text_spacing_version, cls_dnn_sp, reg_dnn_sp, repeat):
	'''
	df_dict = {
		"fix_id": [],
		"unique_paragraph_id": [], 
		"text_spacing_version": [], 
		"sp_fix_pos": [], 
		"sp_fix_dur": np.array([]),
		"repeat": []
	}
	'''

	df_dict = {
        "CURRENT_FIX_INDEX": [],
        "CURRENT_FIX_DURATION": np.array([]),
        "position": [],
        "CURRENT_FIX_INTEREST_AREA_INDEX": [],
        "unique_paragraph_id": [], 
		"text_spacing_version": [], 
        "repeats": []
	}
	
	text_spacing_version = text_spacing_version.numpy().tolist()

	for i, (upi, tsv, cds, rds) in enumerate(zip(unique_paragraph_ids, text_spacing_version, cls_dnn_sp, reg_dnn_sp)):
		# print(i)
		assert(len(cds) == len(rds))
		sp_len = len(cds)-1
		df_dict["CURRENT_FIX_INDEX"] += list(range(sp_len))
		df_dict["unique_paragraph_id"] += [upi]*sp_len
		df_dict["text_spacing_version"] += [tsv]*sp_len
		df_dict["CURRENT_FIX_INTEREST_AREA_INDEX"] += cds[:-1].tolist()
		df_dict["CURRENT_FIX_DURATION"] = np.concatenate([df_dict["CURRENT_FIX_DURATION"], (rds*1000)[:-1]])
	
	df_dict["repeats"] = [repeat]*len(df_dict["CURRENT_FIX_INDEX"])
	df_dict["position"] = [np.nan] * len(df_dict["CURRENT_FIX_INDEX"])

	output_df = pd.DataFrame.from_dict(df_dict)
	output_df.CURRENT_FIX_DURATION = output_df.CURRENT_FIX_DURATION.round(3)
	return output_df

def onestop_load_native_speaker():
	sub_metadata_path = '../OneStop-Eye-Movements/data/session_summary.csv'
	sub_infor = pd.read_csv(sub_metadata_path)
	sub_infor = sub_infor[sub_infor.question_preview == False].participant_id.values.tolist()
	return sub_infor

def compute_word_length_onestop(arr):
	#length of a punctuation is 0, plus an epsilon to avoid division output inf
	arr = arr.astype('float64')
	arr[arr==0] = 1/(0+0.5)
	arr[arr!=0] = 1/(arr[arr!=0])
	return arr

def _process_onestop(sn_list, ba_to_participant_id, word_info_df, eyemovement_df, tokenizer, cf, check_baseline, keep_ids, inference):
	"""
	SN_token_embedding   <CLS>, bla, bla, <SEP>
	SP_token_embedding       <CLS>, bla, bla, <SEP>
	SP_ordinal_pos 0, bla, bla, max_sp_len
	SP_fix_dur     0, bla, bla, 0
	"""

	SN_input_ids, SN_attention_mask, SN_WORD_len, WORD_ids_sn = [], [], [], []
	SP_input_ids, SP_attention_mask, WORD_ids_sp = [], [], []
	SP_ordinal_pos, SP_landing_pos, SP_fix_dur = [], [], []
	sub_id_list =  []
	sn_id_list =  []
	dataset_NLD_baseline = []
	dataset_R2_baseline = []
	
	if keep_ids:
		SN_unique_paragraph_id, SP_text_spacing_version = [], []
	
	if inference:
		for unique_paragraph_id, text_spacing_version in tqdm(word_info_df[['unique_paragraph_id', 'text_spacing_version']].drop_duplicates().values.tolist()):
			full_sn_df = word_info_df[(word_info_df.unique_paragraph_id==unique_paragraph_id) & (word_info_df.text_spacing_version == text_spacing_version)]
			first_sub = full_sn_df.drop_duplicates(["unique_paragraph_id", "text_spacing_version"]).participant_id.values[0]
			sn_sub = full_sn_df[full_sn_df.participant_id == first_sub]
			
			''' START INSERTED SEGMENT'''
			#compute word length for each word
			sn_word_len = compute_word_length_onestop(sn_sub.word_length.values)
			sn_word_list = list(sn_sub.IA_LABEL.values)
			
			#tokenization and padding
			tokenizer.padding_side = 'right'
			sn_word_list = ['[CLS]'] + sn_word_list + ['[SEP]']
			#pre-tokenized input
			tokens = tokenizer.encode_plus(sn_word_list,
											add_special_tokens = False,
											truncation=False,
											max_length = cf['max_sn_token'],
											padding = 'max_length',
											return_attention_mask=True,
											is_split_into_words=True)
			encoded_sn = tokens['input_ids']
			mask_sn = tokens['attention_mask']
			#use offset mapping to determine if two tokens are in the same word.
			#index start from 0, CLS -> 0 and SEP -> last index
			word_ids_sn = tokens.word_ids()
			word_ids_sn = [val if val is not None else np.nan for val in word_ids_sn]

			#sentence information
			SN_input_ids.append(encoded_sn)
			SN_attention_mask.append(mask_sn)
			SN_WORD_len.append(sn_word_len)
			WORD_ids_sn.append(word_ids_sn)
			sub_id_list.append(first_sub)
			sn_id_list.append(unique_paragraph_id)
			
			if keep_ids:
				SN_unique_paragraph_id.append(unique_paragraph_id)
				SP_text_spacing_version.append(sn_sub.text_spacing_version.values)

	else:	
		for sn_id_truncated in tqdm(ba_to_participant_id.keys()): # 1_10
			sn_id_extended_list = {value for value in sn_list if "_".join(value.split("_")[:2]) == sn_id_truncated} # from 1_10 to list of 1_10_..., 1_10_...
			for sn_id in sn_id_extended_list: # sn_id = 1_10_Adv_1...
				#process sentence sequence
				sn_eye_df = eyemovement_df[eyemovement_df.unique_paragraph_id==sn_id]
				#notice: Each sentence is recorded multiple times in file |word_info_df|.
				sn = word_info_df[word_info_df.unique_paragraph_id == sn_id]
				
				#process fixation sequence
				sn_fix_list = []
				sn_dur_list = []

				for sub_id in ba_to_participant_id[sn_id_truncated]:
					sub_df = sn_eye_df[sn_eye_df.participant_id==sub_id]
					# remove fixations on non-words
					sub_df = sub_df.loc[sub_df.CURRENT_FIX_INTEREST_AREA_LABEL != '.']
					
					if len(sub_df) == 0: #no scanpath data found for the subject
						continue
					
					''' START INSERTED SEGMENT'''
					sn_sub = sn[sn['participant_id']==sub_id]
					#compute word length for each word
					sn_word_len = compute_word_length_onestop(sn_sub.word_length.values)
					sn_word_list = list(sn_sub.IA_LABEL.values)
					
					#tokenization and padding
					tokenizer.padding_side = 'right'
					sn_word_list = ['[CLS]'] + sn_word_list + ['[SEP]']
					#pre-tokenized input
					tokens = tokenizer.encode_plus(sn_word_list,
													add_special_tokens = False,
													truncation=False,
													max_length = cf['max_sn_token'],
													padding = 'max_length',
													return_attention_mask=True,
													is_split_into_words=True)
					encoded_sn = tokens['input_ids']
					mask_sn = tokens['attention_mask']
					#use offset mapping to determine if two tokens are in the same word.
					#index start from 0, CLS -> 0 and SEP -> last index
					word_ids_sn = tokens.word_ids()
					word_ids_sn = [val if val is not None else np.nan for val in word_ids_sn]

					''' END INSERTED SEGMENT'''

					#prepare decoder input and output
					sp_word_pos, sp_fix_loc, sp_fix_dur = sub_df.CURRENT_FIX_INTEREST_AREA_ID.values, sub_df.CURRENT_FIX_NEAREST_INTEREST_AREA_DISTANCE.values, sub_df.CURRENT_FIX_DURATION.values

					if keep_ids:
						sp_text_spacing_version = sub_df.text_spacing_version.values

					'''
					# scanpath too long, remove outliers, speed up the inference
					if len(sp_word_pos) > cf["max_sp_len"] - 10:
						outlier_drop +=1
						continue
					
					# 3)scanpath too short for a normal length sentence
					if len(sp_word_pos) <= 1 and sn_len > 10:
						outlier_drop +=1
						continue
					'''
					
					#dataset is noisy -> sanity check
					# 1) check if recorded fixation duration are within reasonable limits
					#Less than 50ms attempt to merge with neighbouring fixation if fixate is on the same word, otherwise delete
					outlier_indx = np.where(sp_fix_dur<50)[0]
					if outlier_indx.size>0:
						for out_idx in range(len(outlier_indx)):
							outlier_i = outlier_indx[out_idx]
							merge_flag = False

							#outliers are commonly found in the fixation of the last record and the first record, and are removed directly
							if outlier_i == len(sp_fix_dur)-1 or outlier_i == 0:
								merge_flag = True

							else:
								if outlier_i-1 >= 0 and merge_flag == False:
									#try to merge with the left fixation
									if sub_df.iloc[outlier_i].CURRENT_FIX_INTEREST_AREA_LABEL == sub_df.iloc[outlier_i-1].CURRENT_FIX_INTEREST_AREA_LABEL:
										sp_fix_dur[outlier_i-1] = sp_fix_dur[outlier_i-1] + sp_fix_dur[outlier_i]
										merge_flag = True

								if outlier_i+1 < len(sp_fix_dur) and merge_flag == False:
									#try to merge with the right fixation
									if sub_df.iloc[outlier_i].CURRENT_FIX_INTEREST_AREA_LABEL == sub_df.iloc[outlier_i+1].CURRENT_FIX_INTEREST_AREA_LABEL:
										sp_fix_dur[outlier_i+1] = sp_fix_dur[outlier_i+1] + sp_fix_dur[outlier_i]
										merge_flag = True

							sp_word_pos = np.delete(sp_word_pos, outlier_i)
							sp_fix_loc = np.delete(sp_fix_loc, outlier_i)
							sp_fix_dur = np.delete(sp_fix_dur, outlier_i)
							sub_df.drop(sub_df.index[outlier_i], axis=0, inplace=True)
							outlier_indx = outlier_indx-1

					# 4) check landing position feature
					#assign missing value to 'nan'
					#sp_fix_loc=np.where(sp_fix_loc=='.', np.nan, sp_fix_loc) ####################
					#convert string of number of float type
					sp_fix_loc = [float(i) for i in sp_fix_loc]

					#Outliers in calculated landing positions due to lack of valid AOI data, assign to 'nan'
					if np.nanmax(sp_fix_loc) > 35:
						missing_idx = np.where(np.array(sp_fix_loc) > 5)[0]
						for miss in missing_idx:
							if sub_df.iloc[miss].CURRENT_FIX_INTEREST_AREA_LEFT in ['NONE', 'BEFORE', 'AFTER', 'BOTH']:
								sp_fix_loc[miss] = np.nan
							else:
								print('Landing position calculation error. Unknown cause, needs to be checked')

					sp_ordinal_pos = sp_word_pos.astype(int)
					SP_ordinal_pos.append(sp_ordinal_pos)
					SP_fix_dur.append(sp_fix_dur)
					SP_landing_pos.append(sp_fix_loc)

					if check_baseline:
						sn_fix_list.append(sp_ordinal_pos)
						sn_dur_list.append(sp_fix_dur)

					sp_token_list = ['[CLS]'] + [sn_word_list[i] for i in sp_ordinal_pos] + ['[SEP]'] ########################## sn_str.split() is shorter than sp_ordinal_pos
					#tokenization and padding for scanpath, i.e. fixated word sequence
					sp_tokens = tokenizer.encode_plus(sp_token_list,
													add_special_tokens = False,
													truncation=False,
													max_length = cf['max_sp_token'],
													padding = 'max_length',
													return_attention_mask=True,
													is_split_into_words=True)
					encoded_sp = sp_tokens['input_ids']
					mask_sp = sp_tokens['attention_mask']
					#index start from 0, CLS -> 0 and SEP -> last index
					word_ids_sp = sp_tokens.word_ids()
					word_ids_sp = [val if val is not None else np.nan for val in word_ids_sp]

					'''
					# tokened scanpath too long, remove outliers, speed up the inference
					if len([entry for entry in encoded_sp if entry != 0]) > cf["max_sp_len"] - 10:
						outlier_drop +=1
						continue
					'''
					
					SP_input_ids.append(encoded_sp)
					SP_attention_mask.append(mask_sp)
					WORD_ids_sp.append(word_ids_sp)

					#sentence information
					SN_input_ids.append(encoded_sn)
					SN_attention_mask.append(mask_sn)
					SN_WORD_len.append(sn_word_len)
					WORD_ids_sn.append(word_ids_sn)
					sub_id_list.append(sub_id)#int(sub_id.split("_")[1])) # from "l39_125" to 125
					sn_id_list.append(sn_id)

					if keep_ids:
						SN_unique_paragraph_id.append(sn_id)
						SP_text_spacing_version.append(sp_text_spacing_version)
		
				if check_baseline:
					dataset_NLD_baseline += NLD_benchmark(sn_fix_list)
					dataset_R2_baseline += R2_benchmark(sn_dur_list)

	if not inference:
		#padding for batch computation
		SP_ordinal_pos = pad_seq(SP_ordinal_pos, max_len=(cf["max_sp_len"]), pad_value=cf["max_sn_len"])
		SP_fix_dur = pad_seq(SP_fix_dur, max_len=(cf["max_sp_len"]), pad_value=0)
		SP_landing_pos = pad_seq(SP_landing_pos, cf["max_sp_len"], pad_value=0, dtype=np.float32)
		SP_input_ids = np.asarray(SP_input_ids, dtype=np.int64)
		SP_attention_mask = np.asarray(SP_attention_mask, dtype=np.float32)

	#padding for batch computation
	SN_WORD_len = pad_seq_with_nan(SN_WORD_len, cf["max_sn_len"], dtype=np.float32)

	#assign type
	SN_input_ids = np.asarray(SN_input_ids, dtype=np.int64)
	SN_attention_mask = np.asarray(SN_attention_mask, dtype=np.float32)
	sub_id_list = np.asarray(sub_id_list)
	sn_id_list = np.asarray(sn_id_list)
	WORD_ids_sn = np.asarray(WORD_ids_sn)
	WORD_ids_sp = np.asarray(WORD_ids_sp)
	
	if check_baseline:
		print(f"\nHuman baseline NLD: {sum(dataset_NLD_baseline)/len(dataset_NLD_baseline)}")
		print(f"Human baseline R2: {sum(dataset_R2_baseline)/len(dataset_R2_baseline)}\n")

	data = {"SN_input_ids": SN_input_ids, "SN_attention_mask": SN_attention_mask, "SN_WORD_len": SN_WORD_len, "WORD_ids_sn": WORD_ids_sn,
	 		"SP_input_ids": SP_input_ids, "SP_attention_mask": SP_attention_mask, "WORD_ids_sp": WORD_ids_sp,
			"SP_ordinal_pos": np.array(SP_ordinal_pos), "SP_landing_pos": np.array(SP_landing_pos), "SP_fix_dur": np.array(SP_fix_dur),
			"sub_id": sub_id_list, "sn_id": sn_id_list
			}
	
	if keep_ids:
				data["SN_unique_paragraph_id"] = np.asarray(SN_unique_paragraph_id)
				data["SP_text_spacing_version"] = SP_text_spacing_version

	
	return data

class onestop_dataset(Dataset):
	"""Return celer dataset."""

	def __init__(
		self,
		word_info_df, eyemovement_df, cf, ba_to_participant_id, sn_list, tokenizer, check_baseline=False, keep_ids=False, inference=False
	):
		self.data = _process_onestop(sn_list, ba_to_participant_id, word_info_df, eyemovement_df, tokenizer, cf, check_baseline, keep_ids, inference)
		self.keep_ids = keep_ids
		self.inference = inference

	def __len__(self):
		return len(self.data["SN_input_ids"])
	
	def __add__(self, other):
		if isinstance(other, onestop_dataset):
			return onestop_dataset(self.value + other.value)
		else:
			raise TypeError("Unsupported operand type for +: 'onestop_dataset' and '{}'".format(type(other)))

	def __getitem__(self, idx):
		sample = {}
		sample["sn_input_ids"] = self.data["SN_input_ids"][idx,:]
		sample["sn_attention_mask"] = self.data["SN_attention_mask"][idx,:]
		sample["sn_word_len"] = self.data['SN_WORD_len'][idx,:]
		sample['word_ids_sn'] =  self.data['WORD_ids_sn'][idx,:]
		sample["sub_id"] = self.data["sub_id"][idx]

		if not self.inference:
			sample["sp_input_ids"] = self.data["SP_input_ids"][idx,:]
			sample["sp_attention_mask"] = self.data["SP_attention_mask"][idx,:]
			sample['word_ids_sp'] =  self.data['WORD_ids_sp'][idx,:]

			sample["sp_pos"] = self.data["SP_ordinal_pos"][idx,:]
			sample["sp_fix_dur"] = self.data["SP_fix_dur"][idx,:]
			sample["sp_landing_pos"] = self.data["SP_landing_pos"][idx,:]
		
		if self.keep_ids:
			sample["sn_unique_paragraph_id"] = self.data["SN_unique_paragraph_id"][idx]
			sample["sp_text_spacing_version"] = self.data["SP_text_spacing_version"][idx][0]

		return sample


def one_hot_encode(arr, dim):
	# one hot encode
	onehot_encoded = np.zeros((arr.shape[0], dim))
	for idx, value in enumerate(arr):
		onehot_encoded[idx, value] = 1

	return onehot_encoded


def gradient_clipping(dnn_model, clip = 10):
	torch.nn.utils.clip_grad_norm_(dnn_model.parameters(), clip)
