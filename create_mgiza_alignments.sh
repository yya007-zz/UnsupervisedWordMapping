#!/bin/bash

# In this script we assume that the target language is always english, and the source languages those in the "for" cycle

./tokenizer.perl -l en < raw_corp.en > corp.tok.en

tr '[:upper:]' '[:lower:]' < corp.tok.en > corp.tok.low.en

mkcls -n10 -pcorp.tok.low.en -Vcorp.tok.low.en.vcb.classes

for l in "it" "es" "de" "fr" "nl"
do
	echo "Pre-processing: tokenizing and lowering..."

	./tokenizer.perl -l ${l} < raw_corp.${l} > corp.tok.${l}

	tr '[:upper:]' '[:lower:]' < corp.tok.${l} > corp.tok.low.${l}

	echo "Finished pre-processing, starting creation of vocabulary, cooccurrence and classes..."

	mkcls -n10 -pcorp.tok.low.${l} -Vcorp.tok.low.${l}.vcb.classes

	plain2snt corp.tok.low.${l} corp.tok.low.en

	snt2cooc corp.tok.low.${l}_corp.tok.low.en.cooc corp.tok.low.${l}.vcb corp.tok.low.en.vcb corp.tok.low.${l}_corp.tok.low.en.snt

	echo "Finished creation! Now we start, really :)"

	echo "Starting alignment: ${l} -> en" > ${l}.timelog
	date >> ${l}.timelog

	mgiza ${l}_en.dict.gizacfg

	echo "Finished alignment, starting merge of parts" >> ${l}.timelog

	date >> ${l}.timelog

	for i in 0 1 2 3 4 5 6 7
	do
		cat ${l}_en.dict.A3.final.part${i} >> corpus_word_aligned_${l}_en
	done

	rm ${l}_en.dict.A3.final.part*

	date >> ${l}.timelog
	echo "End of process." >> ${l}.timelog
done