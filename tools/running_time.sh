set -x

DATASET=karate

LOG_FILE=../log/${DATASET}.log
EDGE_FILE=../tests/${DATASET}.edgelist
OUTPUTFILE=../deneme.embedding
WALKLEN=5
ALPHA=1
WEIGHTDISTR=cauchy
NUMTHREADS=32
CYCLIC=0
WEIGHTBLOCKSIZE=10

(/usr/bin/time -p ../build/nodesig --edgefile ${EDGE_FILE} \
                          --embfile ${OUTPUTFILE} \
                          --walklen ${WALKLEN} \
                          --alpha ${ALPHA} \
                          --weightdistr ${WEIGHTDISTR} \
                          --numthreads ${NUMTHREADS} \
                          --cyclic ${CYCLIC} \
                          --blocksize ${WEIGHTBLOCKSIZE})  2>&1 | tee ${LOG_FILE}