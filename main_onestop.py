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
from torcheval.metrics.functional import r2_score
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
			"n_folds": 5,
			"dataset": 'onestop',
			"atten_type": args.atten_type,
			"batch_size": 128, #
			"max_sn_len": 170, # max number of words in a sentence, include start token and end token,
			"max_sn_token": 225, # maximum number of tokens a sentence includes. include start token and end token,
			"max_sp_len": 250, # max number of words in a scanpath, include start token and end token
			"max_sp_token": 350, # maximum number of tokens a scanpath includes. include start token and end token
			"norm_type": 'z-score',
			"earlystop_patience": 20,
			"max_pred_len":args.max_pred_len
			}


	#Encode the label into interger categories, setting the exclusive category 'onestop["max_sn_len"]-1' as the end sign
	le = LabelEncoder()
	le.fit(np.append(np.arange(-onestop["max_sn_len"]+3, onestop["max_sn_len"]-1), onestop["max_sn_len"]-1)) # todo: maybe too much labels?

	#load corpus
	word_info_df, _, eyemovement_df = load_corpus(onestop["dataset"])

	#only use native speaker; make list with reader index
	reader_list = onestop_load_native_speaker()

	# Make list with sentence index
	sn_list = np.unique(word_info_df.unique_paragraph_id.values).tolist()

	#Split training&test sets by text or reader, depending on configuration
	if args.test_mode == 'text':
		print('Start evaluating on new sentences.')
		split_list = sn_list
	elif args.test_mode == 'subject':
		print('Start evaluating on new readers.')
		split_list = reader_list

	n_folds = onestop["n_folds"]
	kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
	fold_indx = 0
	#for scanpath generation
	sp_dnn_list = []
	sp_human_list = []
	for train_idx, test_idx in kf.split(split_list):
		loss_dict = {'val_loss':[], 
			   		'train_total_loss':[],
					'train_cls_loss':[], 
					'train_reg_loss':[], 
					'test_ll':[], 
					'test_NLD': [],
					'test_R2': [],
					'test_AUC':[]
					}
		list_train = [split_list[i] for i in train_idx]
		list_test = [split_list[i] for i in test_idx]

		# create train validation split for training the models:
		kf_val = KFold(n_splits=n_folds, shuffle=True, random_state=0)
		for train_index, val_index in kf_val.split(list_train):
			# we only evaluate a single fold
			break
		list_train_net = [list_train[i] for i in train_index]
		list_val_net = [list_train[i] for i in val_index]

		if args.test_mode == 'text':
			sn_list_train = list_train_net
			sn_list_val = list_val_net
			sn_list_test = list_test
			reader_list_train, reader_list_val, reader_list_test = reader_list, reader_list, reader_list

		elif args.test_mode == 'subject':
			reader_list_train = list_train_net
			reader_list_val = list_val_net
			reader_list_test = list_test
			sn_list_train, sn_list_val, sn_list_test = sn_list, sn_list, sn_list

		#initialize tokenizer
		tokenizer = BertTokenizerFast.from_pretrained(onestop['model_pretrained'])
		#Preparing batch data
		dataset_train = onestop_dataset(word_info_df, eyemovement_df, onestop, reader_list_train, sn_list_train, tokenizer)
		train_dataloaderr = DataLoader(dataset_train, batch_size = onestop["batch_size"], shuffle = True, drop_last=True)

		dataset_val = onestop_dataset(word_info_df, eyemovement_df, onestop, reader_list_val, sn_list_val, tokenizer)
		val_dataloaderr = DataLoader(dataset_val, batch_size = onestop["batch_size"], shuffle = False, drop_last=True)

		dataset_test = onestop_dataset(word_info_df, eyemovement_df, onestop, reader_list_test, sn_list_test, tokenizer, check_baseline=False)
		test_dataloaderr = DataLoader(dataset_test, batch_size = onestop["batch_size"], shuffle = False, drop_last=False)

		#z-score normalization for gaze features
		fix_dur_mean, fix_dur_std = calculate_mean_std(dataloader=train_dataloaderr, feat_key="sp_fix_dur", padding_value=0, scale=1000)
		landing_pos_mean, landing_pos_std = calculate_mean_std(dataloader=train_dataloaderr, feat_key="sp_landing_pos", padding_value=0)
		sn_word_len_mean, sn_word_len_std = calculate_mean_std(dataloader=train_dataloaderr, feat_key="sn_word_len")

		# load model
		dnn = Eyettention(onestop)

		#training
		episode = 0
		optimizer = Adam(dnn.parameters(), lr=onestop["lr"])
		dnn.train()
		dnn.to(device)
		av_score = deque(maxlen=100)
		cls_av_score = deque(maxlen=100)
		reg_av_score = deque(maxlen=100)
		
		old_score = 1e10
		save_ep_couter = 0

		print('\n- Started Training -')
		for episode_i in range(episode, onestop["n_epochs"]+1):
			dnn.train()
			print('episode:', episode_i)
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
				
				cls_loss = nn.CrossEntropyLoss(reduction="none")
				reg_loss = nn.MSELoss(reduction="none")
				
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

			print("\n- Validating -")
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
				torch.save(dnn.state_dict(), '{}/CELoss_MSELoss_OneStop_{}_eyettention_{}_newloss_fold{}.pth'.format(args.save_data_folder, args.test_mode, args.atten_type, fold_indx))
				old_score= np.mean(val_loss)
				print('\nsaved model state dict\n')
				save_ep_couter = episode_i
			else:
				#early stopping
				if episode_i - save_ep_couter >= onestop["earlystop_patience"]:
					break

		# Test evaluation
		print("\n- Testing -")
		dnn.eval()
		res_llh = []
		res_NLD = []
		res_MSE = []
		res_R2 = []
		dnn.load_state_dict(torch.load(os.path.join(args.save_data_folder,f'CELoss_MSELoss_OneStop_{args.test_mode}_eyettention_{args.atten_type}_newloss_fold{fold_indx}.pth'), map_location='cpu', weights_only=True))
		dnn.to(device)
		batch_indx = 0
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

				cls_dnn_out_test, reg_dnn_out_test, atten_weights_val = dnn(sn_emd=sn_input_ids_test,
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

					cls_sp_dnn, sp_human = prepare_scanpath(cls_sp_dnn.detach().to('cpu').numpy(), sn_len, sp_pos_test, onestop)
					sp_dnn_list.extend(cls_sp_dnn)
					sp_human_list.extend(sp_human)
					res_NLD.append(NLD(sp_human, cls_sp_dnn))
					res_R2.append(r2_score(reg_sp_dnn.T.cpu(), target_test.T.cpu(), multioutput="raw_values").numpy())

				batch_indx +=1

		res_llh = np.concatenate(res_llh).ravel()
		res_NLD = np.concatenate(res_NLD).ravel()
		res_R2 = np.concatenate(res_R2).ravel()
		loss_dict['test_ll'].append(res_llh)
		loss_dict['test_NLD'].append(res_NLD)
		loss_dict['test_R2'].append(res_R2)
		loss_dict['fix_dur_mean'] = fix_dur_mean
		loss_dict['fix_dur_std'] = fix_dur_std
		loss_dict['landing_pos_mean'] = landing_pos_mean
		loss_dict['landing_pos_std'] = landing_pos_std
		loss_dict['sn_word_len_mean'] = sn_word_len_mean
		loss_dict['sn_word_len_std'] = sn_word_len_std
		
		print('\nTest likelihood is {}, NLD is {}, R2 is {}\n'.format(np.mean(res_llh), np.mean(res_NLD), np.mean(res_R2)))
		
		#save results
		with open('{}/res_OneStop_{}_eyettention_{}_Fold{}.pickle'.format(args.save_data_folder, args.test_mode, args.atten_type, fold_indx), 'wb') as handle:
			pickle.dump(loss_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
		fold_indx += 1

	if bool(args.scanpath_gen_flag) == True:
		#save results
		dic = {"sp_dnn": sp_dnn_list, "sp_human": sp_human_list}
		with open(os.path.join(args.save_data_folder, f'OneStop_scanpath_generation_eyettention_{args.test_mode}_{args.atten_type}.pickle'), 'wb') as handle:
			pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
