from fasttext import FastVector
import numpy as np
from cyclegan.main import CycleGAN


def word2vector(wordlist,dictionary):
	reslist=[]
	for word in wordlist:
		reslist.append(dictionary[word])
	return np.array(reslist)



def main():
	en=["hello","hello","hello","hello","hello"]
	it=["Ciao","Ciao","Ciao","Ciao","Ciao"]
	en_dictionary = FastVector(vector_file="../UnsupervisedWordMappingDataset/wiki.en/wiki.en.vec")
	np_en=word2vector(en,en_dictionary)
	del en_dictionary
	it_dictionary = FastVector(vector_file="../UnsupervisedWordMappingDataset/wiki.en/wiki.it.vec")
	np_it=word2vector(it,it_dictionary)
	del it_dictionary
	print np_en.shape
	print np_it.shape
	np.save("en.npy",np_en)
	np.save("it.npy",np_it)




main()

