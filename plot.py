from __future__ import print_function
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

# e1=load_plot_info('30cxvylr8w')
# #no organ no lambda
# e2=load_plot_info('pxivch863r')
# #no organ
# e3=load_plot_info('6zy648e44q')
# #epoch 500000
# e4=load_plot_info('d2xpd9ugsa')
# #epoch 2000000
# e5=load_plot_info('0q6fiq7a5x')
# #epoch 4000000
# e6=load_plot_info('fes9pecbxx')
# #lambda 1
# e7=load_plot_info('xs1i7k2zeb')
# #lambda 5
# e8=load_plot_info('3vgrmpfnk3')
# #lambda 10
# e9=load_plot_info('wdu1gj1v5d')
# #lambda 20
# e10=load_plot_info('cy7cfw2e91')
# #lambda 100
# e11=load_plot_info('l1ezjmep0d')


e1=load_plot_info('30cxvylr8w')
#no organ no lambda
e2=load_plot_info('a40bmf4hp6')
#no organ
e3=load_plot_info('cpehfm1ibe')
#epoch 500000
e4=load_plot_info('3obu19xswj')
#epoch 2000000
e5=load_plot_info('q1zxs8kdng')
#epoch 4000000
# 'rvp2ut1j7l'
e6=load_plot_info('fes9pecbxx')
#lambda 1
e7=load_plot_info('yh5vn4k4wt')
#lambda 5
e8=load_plot_info('lqw7ca7ub6')
#lambda 10
e9=load_plot_info('uhekkegkyo')
#lambda 20
e10=load_plot_info('m88xib2npm')
#lambda 100
e11=load_plot_info('mhv05dsbi8')

colors=['b','g','r','c','m','y','k','w']

features =["precision_at_1-nn", "precision_at_5-nn","precision_at_10-nn","precision_at_1-csls_knn_10","precision_at_5-csls_knn_10","precision_at_10-csls_knn_10"]
labels=["Top 1 nn", "Top 5 nn","Top 10 nn","Top 1 csls","Top 5 csls","Top 10 csls"]

losses=['DIS_A_COSTS','DIS_B_COSTS','GAN_A_COSTS','GAN_B_COSTS','CYC_A_COSTS','CYC_B_COSTS']

EVA_IND=0
MAX_EPOCH=20


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
	plt.plot(e1['epoch_train'],e1[features[i]+"_t_train"],colors[i]+'--',label=labels[i]+' normal direction')
	plt.plot(e1['epoch_train'],e1[features[i]+"_f_train"],colors[i]+'-',label=labels[i]+' reverse direction')
plt.ylabel('Accuracy')
plt.xlabel('# Epoch')
plt.legend()
fig.savefig('./fig/e1_acc.png')  
plt.close(fig)

fig = plt.figure()
for loss in losses:
	plt.plot(e1['iter_train'],e1[loss],'-',label=loss)
plt.ylabel('Loss')
plt.xlabel('# Iteration')
plt.legend()
fig.savefig('./fig/e1_loss.png')   # save the figure to file
plt.close(fig)

fig = plt.figure()
plt.plot(e1['epoch_train'][:MAX_EPOCH],e1[features[EVA_IND]+"_t_train"][:MAX_EPOCH],'-',label='lambda=0')
plt.plot(e7['epoch_train'],e7[features[EVA_IND]+"_t_train"],'-',label='lambda=1')
plt.plot(e8['epoch_train'],e8[features[EVA_IND]+"_t_train"],'-',label='lambda=5')
plt.plot(e9['epoch_train'],e9[features[EVA_IND]+"_t_train"],'-',label='lambda=10')
plt.plot(e10['epoch_train'],e10[features[EVA_IND]+"_t_train"],'-',label='lambda=20')
plt.plot(e11['epoch_train'],e11[features[EVA_IND]+"_t_train"],'-',label='lambda=100')
plt.ylabel('Accuracy')
plt.xlabel('# Epoch')
plt.legend()
fig.savefig('./fig/e2_lam_t.png')  
plt.close(fig)

fig = plt.figure()
plt.plot(e1['epoch_train'][:MAX_EPOCH],e1[features[EVA_IND]+"_f_train"][:MAX_EPOCH],'-',label='lambda=0')
plt.plot(e7['epoch_train'],e7[features[EVA_IND]+"_f_train"],'-',label='lambda=1')
plt.plot(e8['epoch_train'],e8[features[EVA_IND]+"_f_train"],'-',label='lambda=5')
plt.plot(e9['epoch_train'],e9[features[EVA_IND]+"_f_train"],'-',label='lambda=10')
plt.plot(e10['epoch_train'],e10[features[EVA_IND]+"_f_train"],'-',label='lambda=20')
plt.plot(e11['epoch_train'],e11[features[EVA_IND]+"_f_train"],'-',label='lambda=100')
plt.ylabel('Accuracy')
plt.xlabel('# Epoch')
plt.legend()
fig.savefig('./fig/e2_lam_f.png')  
plt.close(fig)

for i in range(len(losses)):
	loss=losses[i]
	fig = plt.figure()
	plt.plot(e1['iter_train'][:len(e7['iter_train'])],e1[loss][:len(e7[loss])],'-',label='lambda=0')
	plt.plot(e7['iter_train'],e7[loss],'-',label='lambda=1')
	plt.plot(e8['iter_train'],e8[loss],'-',label='lambda=5')
	plt.plot(e9['iter_train'],e9[loss],'-',label='lambda=10')
	plt.plot(e10['iter_train'],e10[loss],'-',label='lambda=20')
	plt.plot(e11['iter_train'],e11[loss],'-',label='lambda=100')
	plt.ylabel('Loss')
	plt.xlabel('# Iteration')
	plt.legend()
	plt.title(loss)
	fig.savefig('./fig/e2_loss_'+str(i)+'.png')  
	plt.close(fig)

fig = plt.figure()
plt.plot(e4['epoch_train'],e4[features[EVA_IND]+"_t_train"],'-',label='epoch size = 500000')
plt.plot(e9['epoch_train'],e9[features[EVA_IND]+"_t_train"],'-',label='epoch size = 1000000')
plt.plot(e5['epoch_train'],e5[features[EVA_IND]+"_t_train"],'-',label='epoch size = 2000000')
# plt.plot(e6['epoch_train'],e6[features[EVA_IND]+"_t_train"],'-',label='epoch size = 4000000')
plt.ylabel('Accuracy')
plt.xlabel('# Epoch')
plt.legend()
fig.savefig('./fig/e3_ep_t.png')  
plt.close(fig)

fig = plt.figure()
plt.plot(e4['epoch_train'],e4[features[EVA_IND]+"_f_train"],'-',label='epoch size = 500000')
plt.plot(e9['epoch_train'],e9[features[EVA_IND]+"_f_train"],'-',label='epoch size = 1000000')
plt.plot(e5['epoch_train'],e5[features[EVA_IND]+"_f_train"],'-',label='epoch size = 2000000')
# plt.plot(e6['epoch_train'],e6[features[EVA_IND]+"_f_train"],'-',label='epoch size = 4000000')
plt.ylabel('Accuracy')
plt.xlabel('# Epoch')
plt.legend()
fig.savefig('./fig/e3_ep_f.png')  
plt.close(fig)

fig = plt.figure()
plt.plot(e1['epoch_train'][:MAX_EPOCH],e1[features[EVA_IND]+"_t_train"][:MAX_EPOCH],'-',label='lambda=0 beta=0.001')
plt.plot(e3['epoch_train'],e3[features[EVA_IND]+"_t_train"],'-',label='lambda=0 beta=0')
plt.plot(e2['epoch_train'],e2[features[EVA_IND]+"_t_train"],'-',label='lambda=10 beta=0')
plt.plot(e9['epoch_train'],e9[features[EVA_IND]+"_t_train"],'-',label='lambda=10 beta=0.001')
plt.ylabel('Accuracy')
plt.xlabel('# Epoch')
plt.legend()
fig.savefig('./fig/e4_organ_t.png')  
plt.close(fig)

fig = plt.figure()
plt.plot(e1['epoch_train'][:MAX_EPOCH],e1[features[EVA_IND]+"_f_train"][:MAX_EPOCH],'-',label='lambda=0 beta=0.001')
plt.plot(e3['epoch_train'],e3[features[EVA_IND]+"_f_train"],'-',label='lambda=0 beta=0')
plt.plot(e2['epoch_train'],e2[features[EVA_IND]+"_f_train"],'-',label='lambda=10 beta=0')
plt.plot(e9['epoch_train'],e9[features[EVA_IND]+"_f_train"],'-',label='lambda=10 beta=0.001')
plt.ylabel('Accuracy')
plt.xlabel('# Epoch')
plt.legend()
fig.savefig('./fig/e4_organ_f.png')  
plt.close(fig)


fig = plt.figure()
for i in range(3):
	plt.plot(e9['epoch_train'],e9[features[i]+"_t_train"],colors[i]+'-',label=labels[i])
	plt.plot(e9['epoch_train'],e9[features[i+3]+"_t_train"],colors[i]+'--',label=labels[i+3])
plt.ylabel('Accuracy')
plt.xlabel('# Epoch')
plt.legend()
fig.savefig('./fig/e5_csls_t.png')  
plt.close(fig)

fig = plt.figure()
for i in range(3):
	plt.plot(e9['epoch_train'],e9[features[i]+"_f_train"],colors[i]+'-',label=labels[i])
	plt.plot(e9['epoch_train'],e9[features[i+3]+"_f_train"],colors[i]+'--',label=labels[i+3])
plt.ylabel('Accuracy')
plt.xlabel('# Epoch')
plt.legend()
fig.savefig('./fig/e5_csls_f.png')  
plt.close(fig)

fig = plt.figure()
for i in [0,3]:
	plt.plot(e10['epoch_train'],e10[features[i]+"_t_train"],colors[i]+'--',label=labels[i]+' with cycle consistency')
	plt.plot(e1['epoch_train'][:MAX_EPOCH],e1[features[i]+"_t_train"][:MAX_EPOCH],colors[i]+'-',label=labels[i]+' without cycle consistency')
plt.ylabel('Accuracy')
plt.xlabel('# Epoch')
plt.legend()
fig.savefig('./fig/e6_t.png')  
plt.close(fig)

fig = plt.figure()
for i in [0,3]:
	plt.plot(e9['epoch_train'],e9[features[i]+"_f_train"],colors[i]+'--',label=labels[i]+' with cycle consistency')
	plt.plot(e1['epoch_train'][:MAX_EPOCH],e1[features[i]+"_f_train"][:MAX_EPOCH],colors[i]+'-',label=labels[i]+' without cycle consistency')
plt.ylabel('Accuracy')
plt.xlabel('# Epoch')
plt.legend()
fig.savefig('./fig/e6_f.png')  
plt.close(fig)


fig = plt.figure()

for i in range(6):
	print(labels[i],' & ', "%.2f" % e1[features[i]+"_t_train_best"][0],' & ', "%.2f" % e1[features[i]+"_t_refine_best"][0],'\\\\')

print("------")

for i in range(6):
	print(labels[i]+' w/o refinement',' & ', "%.2f" % e1[features[i]+"_t_train_best"][0],' & ',  "%.2f" % e10[features[i]+"_t_train_best"][0],'\\\\')
	print(labels[i]+' w/ refinement',' & ', "%.2f" % e1[features[i]+"_t_refine_best"][0],' & ', "%.2f" % e10[features[i]+"_t_refine_best"][0],'\\\\')

for i in range(6):
	print(labels[i]+' w/o refinement',' & ', "%.2f" % e1[features[i]+"_f_train_best"][0],' & ',  "%.2f" % e9[features[i]+"_t_train_best"][0],'\\\\')
	print(labels[i]+' w refinement',' & ', "%.2f" % e1[features[i]+"_f_refine_best"][0],' & ', "%.2f" % e9[features[i]+"_t_refine_best"][0],'\\\\')

print("------")

for i in range(3):
	print("%.2f" % e1[features[i]+"_t_train_best"][0],' & ', end='')
print("\\\\")

for i in range(3):
	print("%.2f" % e1[features[i+3]+"_t_train_best"][0],' &', end='')
print("\\\\")

for i in range(3):
	print("%.2f" % e10[features[i+3]+"_t_train_best"][0],' & ', end='')
print("\\\\")

for i in range(3):
	print("%.2f" % e1[features[i+3]+"_t_refine_best"][0],' & ', end='')
print("\\\\")



for i in range(3):
	print("%.2f" % e1[features[i]+"_f_train_best"][0],' & ', end='')
print("\\\\")

for i in range(3):
	print("%.2f" % e1[features[i+3]+"_f_train_best"][0],' & ', end='')
print("\\\\")

for i in range(3):
	print("%.2f" % e9[features[i+3]+"_f_train_best"][0],' & ', end='')
print("\\\\")

for i in range(3):
	print("%.2f" % e1[features[i+3]+"_f_refine_best"][0],' & ', end='')
print("\\\\")
