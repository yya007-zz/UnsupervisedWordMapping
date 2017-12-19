import matplotlib.pyplot as plt
import os
import time
import json
import argparse
from collections import OrderedDict
import numpy as np
import torch
from copy import deepcopy




#no lambda
def load_plot_info(name):
	path='./dumped/'
	return json.load(open(path+name+'/plot_info.test'))

e1=load_plot_info('30cxvylr8w')
#no organ no lambda
e2=load_plot_info('pxivch863r')
#no organ
e3=load_plot_info('6zy648e44q')
#epoch 500000
e4=load_plot_info('d2xpd9ugsa')
#epoch 2000000
e5=load_plot_info('0q6fiq7a5x')
#epoch 4000000
e6=load_plot_info('fes9pecbxx')
#lambda 1
e7=load_plot_info('xs1i7k2zeb')
#lambda 5
e8=load_plot_info('3vgrmpfnk3')
#lambda 10
e9=load_plot_info('wdu1gj1v5d')
#lambda 20
e10=load_plot_info('cy7cfw2e91')
#lambda 100
e11=load_plot_info('l1ezjmep0d')

features =["precision_at_1-nn", "precision_at_5-nn","precision_at_10-nn","precision_at_1-csls_knn_10","precision_at_5-csls_knn_10","precision_at_10-csls_knn_10"]
labels=["Top 1 nn", "Top 5 nn","Top 10 nn","Top 1 csls_knn_10","Top 5 csls_knn_10","Top 10 csls_knn_10"]

losses=['DIS_A_COSTS','DIS_B_COSTS','GAN_A_COSTS','GAN_B_COSTS','CYC_A_COSTS','CYC_B_COSTS']
def regular_plot(plot_info, name):
	fig = plt.figure()
	for i in range(len(features)):
		record=features[i]
		label=labels[i]
		plt.plot(plot_info['epoch_train'],plot_info[record+"_t_train"],'-',label=label+' normal direction')
		plt.plot(plot_info['epoch_train'],plot_info[record+"_f_train"],'-',label=label+' reverse direction')
	plt.ylabel('Accuracy')
	plt.xlabel('# Iteration')
	plt.legend()
	fig.savefig('./fig/acc_'+name+'.png')   # save the figure to file
	plt.close(fig)

	fig = plt.figure()
	for loss in losses:
		plt.plot(plot_info['iter_train'],plot_info[loss],'-',label=loss)
	plt.ylabel('Loss')
	plt.xlabel('# Epoch')
	plt.legend()
	fig.savefig('./fig/loss_'+name+'.png')   # save the figure to file
	plt.close(fig)

	fig = plt.figure()
	for i in range(len(features)):
		record=features[i]
		label=labels[i]
		plt.plot(plot_info['iter_refine'],plot_info[record+"_t_refine"],'-',label=label+' reverse direction')
		plt.plot(plot_info['iter_refine'],plot_info[record+"_f_refine"],'-',label=label+' reverse direction')
	plt.ylabel('Accuracy')
	plt.xlabel('# Iteration')
	plt.legend()
	fig.savefig('./fig/refine_'+name+'.png')   # save the figure to file
	plt.close(fig)

# regular_plot(e1, 'nolambda40')
# regular_plot(e2, 'noorg')
# regular_plot(e3, 'noorg&lam')
# regular_plot(e4, 'e500000')
# regular_plot(e9, 'e1000000')
# regular_plot(e5, 'e2000000')
# regular_plot(e6, 'e4000000')
# regular_plot(e7, 'lam1')
# regular_plot(e8, 'lam5')
# regular_plot(e9, 'lam10')
# regular_plot(e10, 'lam20')
# regular_plot(e11, 'lam100')

fig = plt.figure()
for i in range(3):
	plt.plot(e1['epoch_train'],e1[features[i]+"_t_train"],'-',label=labels[i]+' normal direction')
	plt.plot(e1['epoch_train'],e1[features[i]+"_f_train"],'-',label=labels[i]+' reverse direction')
plt.ylabel('Accuracy')
plt.xlabel('# Iteration')
plt.legend()
fig.savefig('./fig/e1_acc.png')  
plt.close(fig)

fig = plt.figure()
for loss in losses:
	plt.plot(e1['iter_train'],e1[loss],'-',label=loss)
plt.ylabel('Loss')
plt.xlabel('# Epoch')
plt.legend()
fig.savefig('./fig/e1_loss.png')   # save the figure to file
plt.close(fig)

fig = plt.figure()
plt.plot(e1['epoch_train'][:10],e1[features[0]+"_t_train"][:10],'-',label='lambda=0')
plt.plot(e7['epoch_train'],e7[features[0]+"_t_train"],'-',label='lambda=1')
plt.plot(e8['epoch_train'],e8[features[0]+"_t_train"],'-',label='lambda=5')
plt.plot(e9['epoch_train'],e9[features[0]+"_t_train"],'-',label='lambda=10')
plt.plot(e10['epoch_train'],e10[features[0]+"_t_train"],'-',label='lambda=20')
plt.plot(e11['epoch_train'],e11[features[0]+"_t_train"],'-',label='lambda=100')
plt.ylabel('Accuracy')
plt.xlabel('# Iteration')
plt.legend()
fig.savefig('./fig/e2_lam_t.png')  
plt.close(fig)

fig = plt.figure()
plt.plot(e1['epoch_train'][:10],e1[features[0]+"_f_train"][:10],'-',label='lambda=0')
plt.plot(e7['epoch_train'],e7[features[0]+"_f_train"],'-',label='lambda=1')
plt.plot(e8['epoch_train'],e8[features[0]+"_f_train"],'-',label='lambda=5')
plt.plot(e9['epoch_train'],e9[features[0]+"_f_train"],'-',label='lambda=10')
plt.plot(e10['epoch_train'],e10[features[0]+"_f_train"],'-',label='lambda=20')
plt.plot(e11['epoch_train'],e11[features[0]+"_f_train"],'-',label='lambda=100')
plt.ylabel('Accuracy')
plt.xlabel('# Iteration')
plt.legend()
fig.savefig('./fig/e2_lam_f.png')  
plt.close(fig)

fig = plt.figure()
plt.plot(e4['epoch_train'],e4[features[0]+"_t_train"],'-',label='epoch size = 500000')
plt.plot(e9['epoch_train'],e9[features[0]+"_t_train"],'-',label='epoch size = 1000000')
plt.plot(e5['epoch_train'],e5[features[0]+"_t_train"],'-',label='epoch size = 2000000')
plt.plot(e6['epoch_train'],e6[features[0]+"_t_train"],'-',label='epoch size = 4000000')
plt.ylabel('Accuracy')
plt.xlabel('# Iteration')
plt.legend()
fig.savefig('./fig/e3_ep_t.png')  
plt.close(fig)

fig = plt.figure()
plt.plot(e4['epoch_train'],e4[features[0]+"_f_train"],'-',label='epoch size = 500000')
plt.plot(e9['epoch_train'],e9[features[0]+"_f_train"],'-',label='epoch size = 1000000')
plt.plot(e5['epoch_train'],e5[features[0]+"_f_train"],'-',label='epoch size = 2000000')
plt.plot(e6['epoch_train'],e6[features[0]+"_f_train"],'-',label='epoch size = 4000000')
plt.ylabel('Accuracy')
plt.xlabel('# Iteration')
plt.legend()
fig.savefig('./fig/e3_ep_f.png')  
plt.close(fig)
