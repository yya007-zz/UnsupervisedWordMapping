from fasttext import FastVector
import numpy as np
from cyclegan.main import CycleGAN


def word2vector(wordlist,dictionary):
	reslist=[]
	for word in wordlist:
		if word in dictionary:
			reslist.append(dictionary[word])
		else:
			reslist.append(np.zeros([300]))
	return np.array(reslist)

def getTheFrequency():
	address="../UnsupervisedWordMappingDataset/EN.200K.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt"
	f = open(address,'r')
	res=[]
	for line in f.readlines():
		word=line.split()[0]
		res.append(word)
	f.close()
	np.save("./data/en_200K.npy",np.array(res))

	address="../UnsupervisedWordMappingDataset/IT.200K.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt"
	f = open(address,'r')
	res=[]
	for line in f.readlines():
		word=line.split()[0]
		res.append(word)
	f.close()
	np.save("./data/it_200K.npy",np.array(res))

def createTrainAndTest():
	address="../UnsupervisedWordMappingDataset/OPUS_en_it_europarl_train_5K.txt"
	f = open(address,'r')
	res=[]
	for line in f.readlines():
		word=line.split()
		res.append(word)
	f.close()

	np.save("./data/en_it_train.npy",np.array(res))

	address="../UnsupervisedWordMappingDataset/OPUS_en_it_europarl_test.txt"
	f = open(address,'r')
	res=[]
	for line in f.readlines():
		word=line.split()
		res.append(word)
	f.close()

	np.save("./data/en_it_test.npy",np.array(res))
	
def encodingFiles():
	en_dictionary = FastVector(vector_file="../UnsupervisedWordMappingDataset/wiki.en/wiki.en.vec")
	it_dictionary = FastVector(vector_file="../UnsupervisedWordMappingDataset/wiki.it/wiki.it.vec")
	
	# file="./data/en_200K"
	# array=np.load(file+".npy")
	# vec=word2vector(array,en_dictionary)
	# np.save(file+'_vec.npy',vec)
	# print vec.shape

	# file="./data/it_200K"
	# array=np.load(file+".npy")
	# vec=word2vector(array,it_dictionary)
	# np.save(file+'_vec.npy',vec)
	# print vec.shape

	file="./data/en_it_train"
	array=np.load(file+".npy")
	vec=word2vector(array[:,0],en_dictionary)
	print vec.shape
	np.save(file+'_en_vec.npy',vec)
	vec=word2vector(array[:,1],it_dictionary)
	np.save(file+'_it_vec.npy',vec)
	print vec.shape

	file="./data/en_it_test"
	array=np.load(file+".npy")
	vec=word2vector(array[:,0],en_dictionary)
	print vec.shape
	np.save(file+'_en_vec.npy',vec)
	vec=word2vector(array[:,1],it_dictionary)
	np.save(file+'_it_vec.npy',vec)
	print vec.shape

	del en_dictionary
	
	

def main():
	# preparation
	# getTheFrequency()
	# createTrainAndTest()	
	encodingFiles();


	# en=["hello","hello","hello","hello","hello"]
	# it=["Ciao","Ciao","Ciao","Ciao","Ciao"]
	# 
	# it_dictionary = FastVector(vector_file="../UnsupervisedWordMappingDataset/wiki.en/wiki.it.vec")
	# np_it=word2vector(it,it_dictionary)
	# del it_dictionary
	# print np_en.shape
	# print np_it.shape
	# np.save("en.npy",np_en)
	# np.save("it.npy",np_it)




main()

