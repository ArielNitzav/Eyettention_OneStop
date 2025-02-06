import numpy as np
import pandas as pd
import os
from utils_onestop import *
from sklearn.model_selection import StratifiedKFold, KFold
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, RMSprop
from transformers import BertTokenizerFast
from model import Eyettention
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch.nn.functional import cross_entropy, softmax
from collections import deque
import pickle
import json
import matplotlib.pyplot as plt
import argparse
from time import time

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='runs Eyettention on OneStop dataset')
	parser.add_argument(
		'--test_mode',
		help='New Sentence Split: text, New Reader Split: subject',
		type=str,
		default='text'
	)
	parser.add_argument(
		'--atten_type',
		help='attention type: global, local, local-g',
		type=str,
		default='local-g'
	)
	parser.add_argument(
		'--save_data_folder',
		help='folder path for saving results',
		type=str,
		default='./results/OneStop/'
	)
	parser.add_argument(
		'--scanpath_gen_flag',
		help='whether to generate scanpath',
		type=int,
		default=1
	)
	parser.add_argument(
		'--max_pred_len',
		help='if scanpath_gen_flag is True, you can determine the longest scanpath that you want to generate, which should depend on the sentence length',
		type=int,
		default=250
	)
	parser.add_argument(
		'--gpu',
		help='gpu index',
		type=int,
		default=1
	)
	args = parser.parse_args()
	gpu = args.gpu

	#use FastTokenizer lead to warning -> The current process just got forked
	os.environ["TOKENIZERS_PARALLELISM"] = "false"
	torch.set_default_tensor_type('torch.FloatTensor')
	availbl = torch.cuda.is_available()
	print(torch.cuda.is_available())
	if availbl:
		device = f'cuda:{gpu}'
	else:
		device = 'cpu'
	print(device)
	torch.cuda.set_device(gpu)
	onestop = {"model_pretrained": "bert-base-cased",
			"lr": 1e-3,
			"max_grad_norm": 10,
			"cls_loss_coef": 1.0,
			"reg_loss_coef": 100.0,
			"n_epochs": 1000,
			"n_folds": 10,
			"dataset": 'onestop',
			"atten_type": args.atten_type,
			"train_batch_size": 64,
			"val_batch_size": 64,
			"test_batch_size": 64, # large number, to make sure everything fits in one batch
			"max_sn_len": 170, # max number of words in a sentence, include start token and end token,
			"max_sn_token": 250, # maximum number of tokens a sentence includes. include start token and end token,
			"max_sp_len": 700,#250, # max number of words in a scanpath, include start token and end token
			"max_sp_token": 900,#350, # maximum number of tokens a scanpath includes. include start token and end token
			"norm_type": 'z-score',
			"earlystop_patience": 10, # previously 20
			"max_pred_len":args.max_pred_len
			}
	
	inference_mode = False
	keep_ids = True

	#Encode the label into interger categories, setting the exclusive category 'onestop["max_sn_len"]-1' as the end sign
	le = LabelEncoder()
	le.fit(np.append(np.arange(-onestop["max_sn_len"]+3, onestop["max_sn_len"]-1), onestop["max_sn_len"]-1))

	# initialize tokenizer
	tokenizer = BertTokenizerFast.from_pretrained(onestop['model_pretrained'])

	# load corpus
	word_info_df, _, eyemovement_df = load_corpus(onestop["dataset"])

	# load subject - article - fold df
	saf_df = pd.read_csv("CSD/folds_RereadStratified/all_folds_subjects_items_RereadStratified.csv")
	saf_df["batch_id_article_id"] = saf_df["article_batch"].astype(str) + '_' + saf_df["article_id"].astype(str)

	# Make list with sentence index
	sn_list = np.unique(word_info_df.unique_paragraph_id.values).tolist()

	n_folds = onestop["n_folds"]

	#for scanpath generation
	sp_dnn_list = []
	sp_human_list = []

	for fold_indx in range(n_folds):
		print('*'*50)
		print('*'*50)
		print(f"\nLoading the train dataset for fold {fold_indx}:\n")
		ba_to_participant_id_train = process_dataset(fold_indx, saf_df, 'train')

		#Preparing batch data
		dataset_train = onestop_dataset(word_info_df, eyemovement_df, onestop, ba_to_participant_id_train , sn_list, tokenizer, keep_ids=keep_ids)
		train_dataloaderr = DataLoader(dataset_train, batch_size = onestop["train_batch_size"], shuffle = False, drop_last=False)

		print(f"\nLoading the entire dataset for fold {fold_indx}:\n")
		dataset_all = onestop_dataset(word_info_df, eyemovement_df, onestop, _, sn_list, tokenizer, keep_ids=keep_ids, inference=True)
		all_dataloaderr = DataLoader(dataset_all, batch_size = onestop["val_batch_size"], shuffle = False, drop_last=False)

		print("\nData points detected:")
		print(f"Train: {len(dataset_train)}, All: {len(dataset_all)}\n")

		#z-score normalization for gaze features
		sn_word_len_mean, sn_word_len_std = calculate_mean_std(dataloader=train_dataloaderr, feat_key="sn_word_len")

		# load model
		dnn = Eyettention(onestop)
		
		# Test evaluation
		print("\n- Inferencing -\n")
		dnn.eval()
		
		repeats = 1000
		df_report_list = []
		
		dnn.load_state_dict(torch.load(os.path.join(args.save_data_folder,f'CSD_CELoss_MSELoss_OneStop_{args.test_mode}_eyettention_{args.atten_type}_newloss_fold{fold_indx}.pth'), map_location='cpu', weights_only=True))
		dnn.to(device)

		for repeat in range(repeats):
			print(f"Inferencing fold {fold_indx}, repeat {repeat + 1} out of {repeats}")
			for batchh in tqdm(all_dataloaderr):
				with torch.no_grad():
					sn_input_ids_all = batchh["sn_input_ids"].to(device)
					sn_attention_mask_all = batchh["sn_attention_mask"].to(device)
					word_ids_sn_all = batchh["word_ids_sn"].to(device)
					sn_word_len_all = batchh["sn_word_len"].to(device)
					sn_word_len_all = (sn_word_len_all - sn_word_len_mean)/sn_word_len_std
					sn_word_len_all = torch.nan_to_num(sn_word_len_all)

					if bool(args.scanpath_gen_flag) == True:
						sn_len = (torch.max(torch.nan_to_num(word_ids_sn_all), dim=1)[0]+1-2).detach().to('cpu').numpy()
						#compute the scan path generated from the model when the first few fixed points are given
						cls_sp_dnn, reg_sp_dnn = dnn.scanpath_generation(sn_emd=sn_input_ids_all,
														sn_mask=sn_attention_mask_all,
														word_ids_sn=word_ids_sn_all,
														sn_word_len = sn_word_len_all,
														le=le,
														max_pred_len=onestop['max_pred_len'])

						cls_sp_dnn, reg_sp_dnn, _ = prepare_scanpath(cls_sp_dnn.detach().to('cpu').numpy(), reg_sp_dnn.detach().to('cpu').numpy(), sn_len, None, onestop)
						
						if keep_ids:
							unique_paragraph_ids = batchh["sn_unique_paragraph_id"]
							text_spacing_version = batchh["sp_text_spacing_version"]
							report_output = ez_reader_formatter(unique_paragraph_ids, text_spacing_version, cls_sp_dnn, reg_sp_dnn, repeat)
							df_report_list.append(report_output) #### repeats

		all_df_report = pd.concat(df_report_list, ignore_index=True)
		all_df_report.to_csv(f"results/Eyettention/full_eyettention_output_fold_{fold_indx}.csv", index=False)