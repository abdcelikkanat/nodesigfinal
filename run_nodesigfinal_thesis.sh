
#datasets=(blogcatalog cora dblp Homo_sapiens)
basefolder=/media/abdulkadir/storage/Research/2020/thesis/get_auc/
dataset=amazon302


walklen_list=( 1 2 3 4 5 10 15 )
alpha_list=( 1.0 2.0 )

for walklen in ${walklen_list[@]}
do
        for alpha in ${alpha_list[@]}
	
	do
	
		EDGE_FILE=${basefolder}/${dataset}/${dataset}_gcc_train.edgelist
		EMB_FILE=${basefolder}/${dataset}/embeddings/${dataset}_gcc_train_nodesig_alpha=${alpha}_order=${walklen}.binary
		
                echo "---------------"
		echo $dataset, $L
                echo "---------------"

                /home/abdulkadir/Desktop/nodesigfinal_thesis/build/nodesig --edgefile ${EDGE_FILE} --embfile ${EMB_FILE} --walklen ${walklen} --alpha ${alpha}

		
	done
done

