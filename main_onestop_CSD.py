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

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='run Eyettention on OneStop dataset')
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
			"val_batch_size": 16,
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
	keep_test_ids = True

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
		print(f"\nStarted fold {fold_indx}\n")
		loss_dict = {'val_loss':[], 
			   		'train_total_loss':[],
					'train_cls_loss':[], 
					'train_reg_loss':[], 
					'test_ll':[], 
					'test_NLD': [],
					'test_AUC':[]
					}
		
		ba_to_participant_id_train = process_dataset(fold_indx, saf_df, 'train') #############################################
		ba_to_participant_id_val = process_dataset(fold_indx, saf_df, 'val')
		ba_to_participant_id_test = process_dataset(fold_indx, saf_df, 'test')

		#Preparing batch data
		dataset_train = onestop_dataset(word_info_df, eyemovement_df, onestop, ba_to_participant_id_train , sn_list, tokenizer)
		train_dataloaderr = DataLoader(dataset_train, batch_size = onestop["train_batch_size"], shuffle = True, drop_last=True)

		dataset_val = onestop_dataset(word_info_df, eyemovement_df, onestop, ba_to_participant_id_val, sn_list, tokenizer)
		val_dataloaderr = DataLoader(dataset_val, batch_size = onestop["val_batch_size"], shuffle = False, drop_last=True)

		dataset_test = onestop_dataset(word_info_df, eyemovement_df, onestop, ba_to_participant_id_test, sn_list, tokenizer, check_baseline=False, keep_ids=keep_test_ids)
		test_dataloaderr = DataLoader(dataset_test, batch_size = onestop["test_batch_size"], shuffle = False, drop_last=False)

		print("\nData points detected:")
		print(f"Train: {len(dataset_train)}, Test: {len(dataset_val)}, Val: {len(dataset_test)}, Total: {len(dataset_train)+len(dataset_val)+len(dataset_test)}\n")

		#z-score normalization for gaze features
		fix_dur_mean, fix_dur_std = calculate_mean_std(dataloader=train_dataloaderr, feat_key="sp_fix_dur", padding_value=0, scale=1000)
		landing_pos_mean, landing_pos_std = calculate_mean_std(dataloader=train_dataloaderr, feat_key="sp_landing_pos", padding_value=0)
		sn_word_len_mean, sn_word_len_std = calculate_mean_std(dataloader=train_dataloaderr, feat_key="sn_word_len")

		# load model
		dnn = Eyettention(onestop)
		
		# define loss functions
		cls_loss = nn.CrossEntropyLoss(reduction="none")
		reg_loss = nn.MSELoss(reduction="none")

		#training
		epoch = 0
		optimizer = Adam(dnn.parameters(), lr=onestop["lr"])
		dnn.train()
		dnn.to(device)
		av_score = deque(maxlen=100)
		cls_av_score = deque(maxlen=100)
		reg_av_score = deque(maxlen=100)
		
		old_score = 1e10
		save_echo_counter = 0
		
		for epoch_i in range(epoch, 10): #onestop["n_epochs"]+1):
			if inference_mode:
				break

			print('\n- Started Training -\n')
			dnn.train()
			print('Epoch:', epoch_i+1)
			counter = 0
			for batchh in train_dataloaderr:
				counter += 1
				batchh.keys()
				sn_input_ids = batchh["sn_input_ids"].to(device)
				sn_attention_mask = batchh["sn_attention_mask"].to(device)
				word_ids_sn = batchh["word_ids_sn"].to(device)
				sn_word_len = batchh["sn_word_len"].to(device)

				sp_input_ids = batchh["sp_input_ids"].to(device)
				sp_attention_mask = batchh["sp_attention_mask"].to(device)
				word_ids_sp = batchh["word_ids_sp"].to(device)

				sp_pos = batchh["sp_pos"].to(device)
				sp_landing_pos = batchh["sp_landing_pos"].to(device)
				sp_fix_dur = (batchh["sp_fix_dur"]/1000).to(device)

				target = sp_fix_dur[:, 1:].to(device)

				#normalize gaze features
				mask = ~torch.eq(sp_fix_dur, 0)
				sp_fix_dur = (sp_fix_dur-fix_dur_mean)/fix_dur_std * mask
				sp_landing_pos = (sp_landing_pos - landing_pos_mean)/landing_pos_std * mask
				sp_fix_dur = torch.nan_to_num(sp_fix_dur)
				sp_landing_pos = torch.nan_to_num(sp_landing_pos)
				sn_word_len = (sn_word_len - sn_word_len_mean)/sn_word_len_std
				sn_word_len = torch.nan_to_num(sn_word_len)

				# zero old gradients
				optimizer.zero_grad()
				# predict output with DNN
				cls_dnn_out, reg_dnn_out, atten_weights = dnn(sn_emd=sn_input_ids,
											sn_mask=sn_attention_mask,
											sp_emd=sp_input_ids,
											sp_pos=sp_pos,
											word_ids_sn=word_ids_sn,
											word_ids_sp=word_ids_sp,
											sp_fix_dur=sp_fix_dur,
											sp_landing_pos=sp_landing_pos,
											sn_word_len = sn_word_len)

				cls_dnn_out = cls_dnn_out.permute(0,2,1) # [batch, dec_o_dim, step]
				reg_dnn_out = reg_dnn_out.permute(0,2,1).squeeze(1) # [batch, step]

				#prepare label and mask
				pad_mask, label = load_label(sp_pos, onestop, le, device)
				
				cls_batch_error = torch.mean(torch.masked_select(cls_loss(cls_dnn_out, label), ~pad_mask))
				reg_batch_error = torch.mean(torch.masked_select(reg_loss(reg_dnn_out, target), ~pad_mask))

				batch_error = onestop["cls_loss_coef"]*cls_batch_error + onestop["reg_loss_coef"]*reg_batch_error
				
				# backpropagate loss
				batch_error.backward()
				# clip gradients
				gradient_clipping(dnn, onestop["max_grad_norm"])

				#learn
				optimizer.step()
				av_score.append(batch_error.to('cpu').detach().numpy())
				cls_av_score.append(cls_batch_error.to('cpu').detach().numpy())
				reg_av_score.append(reg_batch_error.to('cpu').detach().numpy())

				print('Batch {}\tAverage CLS Error: {:.10f}\tAverage REG Error: {:.10f}'.format(counter, np.mean(cls_av_score), np.mean(reg_av_score)))
			
			loss_dict['train_total_loss'].append(np.mean(av_score))
			loss_dict['train_cls_loss'].append(np.mean(cls_av_score))
			loss_dict['train_reg_loss'].append(np.mean(reg_av_score))

			print("\n- Validating -\n")
			val_loss = []
			dnn.eval()
			for batchh in val_dataloaderr:
				with torch.no_grad():
					sn_input_ids_val = batchh["sn_input_ids"].to(device)
					sn_attention_mask_val = batchh["sn_attention_mask"].to(device)
					word_ids_sn_val = batchh["word_ids_sn"].to(device)
					sn_word_len_val = batchh["sn_word_len"].to(device)

					sp_input_ids_val = batchh["sp_input_ids"].to(device)
					sp_attention_mask_val = batchh["sp_attention_mask"].to(device)
					word_ids_sp_val = batchh["word_ids_sp"].to(device)

					sp_pos_val = batchh["sp_pos"].to(device)
					sp_landing_pos_val = batchh["sp_landing_pos"].to(device)
					sp_fix_dur_val = (batchh["sp_fix_dur"]/1000).to(device)

					target_val = sp_fix_dur_val[:, 1:].to(device)

					#normalize gaze features
					mask_val = ~torch.eq(sp_fix_dur_val, 0)
					sp_fix_dur_val = (sp_fix_dur_val-fix_dur_mean)/fix_dur_std * mask_val
					sp_landing_pos_val = (sp_landing_pos_val - landing_pos_mean)/landing_pos_std * mask_val
					sp_fix_dur_val = torch.nan_to_num(sp_fix_dur_val)
					sp_landing_pos_val = torch.nan_to_num(sp_landing_pos_val)
					sn_word_len_val = (sn_word_len_val - sn_word_len_mean)/sn_word_len_std
					sn_word_len_val = torch.nan_to_num(sn_word_len_val)

					cls_dnn_out_val, reg_dnn_out_val, atten_weights_val = dnn(sn_emd=sn_input_ids_val,
														sn_mask=sn_attention_mask_val,
														sp_emd=sp_input_ids_val,
														sp_pos=sp_pos_val,
														word_ids_sn=word_ids_sn_val,
														word_ids_sp=word_ids_sp_val,
														sp_fix_dur=sp_fix_dur_val,
														sp_landing_pos=sp_landing_pos_val,
														sn_word_len = sn_word_len_val)

					cls_dnn_out_val = cls_dnn_out_val.permute(0,2,1) # [batch, dec_o_dim, step]
					reg_dnn_out_val = reg_dnn_out_val.permute(0,2,1).squeeze(1) # [batch, step]

					#prepare label and mask
					pad_mask_val, label_val = load_label(sp_pos_val, onestop, le, device)

					cls_batch_error_val = torch.mean(torch.masked_select(cls_loss(cls_dnn_out_val, label_val), ~pad_mask_val))
					reg_batch_error_val = torch.mean(torch.masked_select(reg_loss(reg_dnn_out_val, target_val), ~pad_mask_val))
					
					batch_error_val = onestop["cls_loss_coef"]*cls_batch_error_val + onestop["reg_loss_coef"]*reg_batch_error_val
					val_loss.append(batch_error_val.detach().to('cpu').numpy())
			
			print('\nvalidation loss is {} \n'.format(np.mean(val_loss)))
			loss_dict['val_loss'].append(np.mean(val_loss))

			if np.mean(val_loss) < old_score:
				# save model if val loss is smallest
				torch.save(dnn.state_dict(), '{}/CSD_CELoss_MSELoss_OneStop_{}_eyettention_{}_newloss_fold{}.pth'.format(args.save_data_folder, args.test_mode, args.atten_type, fold_indx))
				old_score= np.mean(val_loss)
				print('\nsaved model state dict\n')
				save_echo_counter = epoch_i
			else:
				#early stopping
				if epoch_i - save_echo_counter >= onestop["earlystop_patience"]:
					break

		# Test evaluation
		print("\n- Testing -\n")
		dnn.eval()
		res_llh = []
		res_NLD = []
		res_MSE = []
		df_report_list = []
		
		dnn.load_state_dict(torch.load(os.path.join(args.save_data_folder,f'CSD_CELoss_MSELoss_OneStop_{args.test_mode}_eyettention_{args.atten_type}_newloss_fold{fold_indx}.pth'), map_location='cpu', weights_only=True))
		dnn.to(device)

		for batchh in test_dataloaderr:
			with torch.no_grad():
				sn_input_ids_test = batchh["sn_input_ids"].to(device)
				sn_attention_mask_test = batchh["sn_attention_mask"].to(device)
				word_ids_sn_test = batchh["word_ids_sn"].to(device)
				sn_word_len_test = batchh["sn_word_len"].to(device)

				sp_input_ids_test = batchh["sp_input_ids"].to(device)
				sp_attention_mask_test = batchh["sp_attention_mask"].to(device)
				word_ids_sp_test = batchh["word_ids_sp"].to(device)

				sp_pos_test = batchh["sp_pos"].to(device)
				sp_landing_pos_test = batchh["sp_landing_pos"].to(device)
				sp_fix_dur_test = (batchh["sp_fix_dur"]/1000).to(device)

				target_test = sp_fix_dur_test[:, 1:].to(device)

				#normalize gaze features
				mask_test = ~torch.eq(sp_fix_dur_test, 0)
				sp_fix_dur_test = (sp_fix_dur_test-fix_dur_mean)/fix_dur_std * mask_test
				sp_landing_pos_test = (sp_landing_pos_test - landing_pos_mean)/landing_pos_std * mask_test
				sp_fix_dur_test = torch.nan_to_num(sp_fix_dur_test)
				sp_landing_pos_test = torch.nan_to_num(sp_landing_pos_test)
				sn_word_len_test = (sn_word_len_test - sn_word_len_mean)/sn_word_len_std
				sn_word_len_test = torch.nan_to_num(sn_word_len_test)

				cls_dnn_out_test, reg_dnn_out_test, atten_weights_test = dnn(sn_emd=sn_input_ids_test,
														sn_mask=sn_attention_mask_test,
														sp_emd=sp_input_ids_test,
														sp_pos=sp_pos_test,
														word_ids_sn=word_ids_sn_test,
														word_ids_sp=word_ids_sp_test,
														sp_fix_dur=sp_fix_dur_test,
														sp_landing_pos=sp_landing_pos_test,
														sn_word_len = sn_word_len_test)

				#We do not use nn.CrossEntropyLoss here to calculate the likelihood because it combines nn.LogSoftmax and nn.NLL,
				#while nn.LogSoftmax returns a log value based on e, we want 2 instead
				#m = nn.LogSoftmax(dim=2) -- base e, we want base 2
				m = nn.Softmax(dim=2)
				cls_dnn_out_test = m(cls_dnn_out_test).detach().to('cpu').numpy()

				#prepare label and mask
				pad_mask_test, label_test = load_label(sp_pos_test, onestop, le, 'cpu')
				pred = cls_dnn_out_test.argmax(axis=2)
				
				#compute log likelihood for the batch samples
				res_batch = eval_log_llh(cls_dnn_out_test, label_test, pad_mask_test)
				res_llh.append(np.array(res_batch))

				# Regression
				reg_dnn_out_test = reg_dnn_out_test.permute(0,2,1).squeeze(1)
				pad_mask_test = torch.asarray(pad_mask_test, device=device)
				reg_batch_error_test = torch.mean(torch.masked_select(reg_loss(reg_dnn_out_test, target_test), ~pad_mask_test)).to('cpu').numpy()
				res_MSE.append(reg_batch_error_test)
				
				if bool(args.scanpath_gen_flag) == True:
					sn_len = (torch.max(torch.nan_to_num(word_ids_sn_test), dim=1)[0]+1-2).detach().to('cpu').numpy()
					#compute the scan path generated from the model when the first few fixed points are given
					cls_sp_dnn, reg_sp_dnn = dnn.scanpath_generation(sn_emd=sn_input_ids_test,
													 sn_mask=sn_attention_mask_test,
													 word_ids_sn=word_ids_sn_test,
													 sn_word_len = sn_word_len_test,
													 le=le,
													 max_pred_len=onestop['max_pred_len'])

					cls_sp_dnn, reg_sp_dnn, sp_human = prepare_scanpath(cls_sp_dnn.detach().to('cpu').numpy(), reg_sp_dnn.detach().to('cpu').numpy(), sn_len, sp_pos_test, onestop)
					
					if keep_test_ids:
						unique_paragraph_ids = batchh["sn_unique_paragraph_id"]
						text_spacing_version = batchh["sp_text_spacing_version"]
						report_output = ez_reader_formatter(unique_paragraph_ids, text_spacing_version, cls_sp_dnn, reg_sp_dnn)
						df_report_list.append(report_output)

					sp_dnn_list.extend(cls_sp_dnn)
					sp_human_list.extend(sp_human)
					res_NLD.append(NLD(sp_human, cls_sp_dnn))

		res_llh = np.concatenate(res_llh).ravel()
		res_NLD = np.concatenate(res_NLD).ravel()
		loss_dict['test_ll'].append(res_llh)
		loss_dict['test_NLD'].append(res_NLD)
		loss_dict['fix_dur_mean'] = fix_dur_mean
		loss_dict['fix_dur_std'] = fix_dur_std
		loss_dict['landing_pos_mean'] = landing_pos_mean
		loss_dict['landing_pos_std'] = landing_pos_std
		loss_dict['sn_word_len_mean'] = sn_word_len_mean
		loss_dict['sn_word_len_std'] = sn_word_len_std
		
		print('\nTest likelihood is {}, NLD is {}\n'.format(np.mean(res_llh), np.mean(res_NLD)))
		test_df_report = pd.concat(df_report_list, ignore_index=True)
		test_df_report["sp_fix_dur"] = test_df_report["sp_fix_dur"].round(3)
		test_df_report.to_csv(f"results/Eyettention/test_eyettention_output_fold_{fold_indx}.csv", index=False)
		
		#save results
		with open('{}/res_OneStop_{}_eyettention_{}_Fold{}.pickle'.format(args.save_data_folder, args.test_mode, args.atten_type, fold_indx), 'wb') as handle:
			pickle.dump(loss_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

	if bool(args.scanpath_gen_flag) == True:
		#save results
		dic = {"sp_dnn": sp_dnn_list, "sp_human": sp_human_list}
		with open(os.path.join(args.save_data_folder, f'OneStop_scanpath_generation_eyettention_{args.test_mode}_{args.atten_type}.pickle'), 'wb') as handle:
			pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
