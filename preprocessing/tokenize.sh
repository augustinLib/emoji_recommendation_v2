#../preprocessing/tokenize.sh sentimental.tsv sentimental_tok.tsv

FILENAME=$1
TARGET_FILENAME=$2

cut -f2 ${FILENAME} > ${FILENAME}.label
cut -f1 ${FILENAME} | mecab -O wakati > ${FILENAME}.text.tok

paste ${FILENAME}.text.tok ${FILENAME}.label > ${TARGET_FILENAME}
