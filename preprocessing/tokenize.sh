#../preprocessing/tokenize.sh sentimental.tsv sentimental_tok.tsv

FILENAME=$1
TARGET_FILENAME=$2
MODEL_FILENAME=$3
N_SYMBOLS=50000

cut -f1 ${FILENAME} > ${FILENAME}.text
cut -f2 ${FILENAME} > ${FILENAME}.label

cat ${FILENAME}.text | mecab -O wakati > ${FILENAME}.text.tok


subword-nmt learn-bpe -s ${N_SYMBOLS} < ${FILENAME}.text.tok > ${MODEL_FILENAME}
subword-nmt apply-bpe -c ${MODEL_FILENAME} < ${FILENAME}.text.tok > ${FILENAME}.text.tok.bpe

#./learn_bpe.py -s 30000 < ${FILENAME}.text.tok > ../emoji_classification/models/bpe_model
#./apply_bpe.py -c ../../emoji_classification/models/bpe_model < ${FILENAME}.text.tok > ${TARGET_FILENAME}.text.tok


paste ${FILENAME}.label ${FILENAME}.text.tok.bpe > ${TARGET_FILENAME}
