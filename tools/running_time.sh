set -x

EDGE_FILE=../tests/karate.edgelist
OUTPUTFILE=../deneme.embedding
WALKLEN=5
ALPHA=1
WEIGHTDISTR=cauchy
NUMTHREADS=32
CYCLIC=0

(/usr/bin/time -p ../build/nodesig --edgefile ${EDGE_FILE} \
                          --embfile ${OUTPUTFILE} \
                          --walklen ${WALKLEN} \
                          --alpha ${ALPHA} \
                          --weightdistr ${WEIGHTDISTR} \
                          --numthreads ${NUMTHREADS} \
                          --cyclic ${CYCLIC})  2>&1 | tee ../deneme.log