from fasttext import FastVector
import numpy as np

def word2vector_mo(wordlist,dictionary):
	reslist=[]
	for word in wordlist:
		if word in dictionary:
			reslist.append(dictionary[word])
		else:
			pass
			# reslist.append(np.zeros([300]))
	return np.array(reslist)

def word2vector_bi(wl1,wl2,dict1,dict2):
	reslist=[]
	assert wl1.shape[0]==wl2.shape[0]
	for i in range(wl1.shape[0]):
		w1=wl1[i]
		w2=wl2[i]
		if w1 in dict1 and w2 in dict2:
			reslist.append([dict1[w1],dict2[w2]])
		else:
			pass
			# reslist.append(np.zeros([300]))
	return np.array(reslist)

def getTheFrequency():
	address="./data/embeddings/original/en.emb.txt"
	f = open(address,'r')
	res=[]
	for line in f.readlines():
		word=line.split()[0]
		res.append(word)
	f.close()
	np.save("./data/en.npy",np.array(res))

	address="./data/embeddings/original/it.emb.txt"
	f = open(address,'r')
	res=[]
	for line in f.readlines():
		word=line.split()[0]
		res.append(word)
	f.close()
	np.save("./data/it.npy",np.array(res))

def createTrainAndTest():
	address="./data/dictionaries/en-it.train.txt"
	f = open(address,'r')
	res=[]
	for line in f.readlines():
		word=line.split()
		res.append(word)
	f.close()

	np.save("./data/en_it_train.npy",np.array(res))

	address="./data/dictionaries/en-it.test.txt"
	f = open(address,'r')
	res=[]
	for line in f.readlines():
		word=line.split()
		res.append(word)
	f.close()
	np.save("./data/en_it_test.npy",np.array(res))
	
def encodingFiles_mo():
	en_dictionary = FastVector(vector_file="./data/pretrained/en.vec")
	it_dictionary = FastVector(vector_file="./data/pretrained/it.vec")
	
	file="./data/en"
	array=np.load(file+".npy")
	vec=word2vector_mo(array,en_dictionary)
	np.save(file+'_vec.npy',vec)
	print vec.shape

	file="./data/it"
	array=np.load(file+".npy")
	vec=word2vector_mo(array,it_dictionary)
	np.save(file+'_vec.npy',vec)
	print vec.shape

def encodingFiles_bi():
	en_dictionary = FastVector(vector_file="./data/pretrained/en.vec")
	it_dictionary = FastVector(vector_file="./data/pretrained/it.vec")

	file="./data/en_it_train"
	array=np.load(file+".npy")
	vec=word2vector_bi(array[:,0],array[:,1],en_dictionary,it_dictionary)
	np.save(file+'_vec.npy',vec)
	print vec.shape

	file="./data/en_it_test"
	array=np.load(file+".npy")
	vec=word2vector_bi(array[:,0],array[:,1],en_dictionary,it_dictionary)
	np.save(file+'_vec.npy',vec)
	print vec.shape

	del en_dictionary
	del it_dictionary

def main():
	# getTheFrequency()
	# createTrainAndTest()	
	# encodingFiles_mo()
	encodingFiles_bi()


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

