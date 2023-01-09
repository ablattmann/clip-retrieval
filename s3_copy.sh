#!/bin/bash
for i in {0..4}
do
  WORKDIR="/fsx/Andreas/projects/clip-retrieval/data/all-mpnet-base-v2-laion-a-part${i}/text_emb"
  echo "working dir is ${WORKDIR}"
  for filename in $(ls "${WORKDIR}" | grep ".npy")
  do
    tmux new-session -d -s aws"${i}"-"${j}"
    tmux send-keys "aws s3 cp /fsx/Andreas/projects/clip-retrieval/data/all-mpnet-base-v2-laion-a-part${i}/text_emb/${filename} s3://stability-aws/laion-a-native/part-${i}/all-mpnet-base-v2-embeddings-1/text_emb/${filename}"
    tmux detach -s aws$i
#    aws s3 cp /fsx/Andreas/projects/clip-retrieval/data/all-mpnet-base-v2-laion-a-part${i}/text_emb/${filename} s3://stability-aws/laion-a-native/part-${i}/all-mpnet-base-v2-embeddings-1/text_emb/${filename}
  done
done
