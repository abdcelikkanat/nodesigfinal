

datasetbase="./gcc05_datasets"
embbase="./gcc05_embeddings"
outputbase="./gcc05_results"

datasets=(blogcatalog cora dblp Homo_sapiens)

alpha=8

walk_lens=(15)


for L in ${walk_lens[@]}
do
        for dataset in ${datasets[@]}
	
	do
		emb_file_path=${embbase}/${dataset}_newborn_gcc_train_L=${L}_alpha=${alpha}_cauchy.embedding
		output_path=${outputbase}/${dataset}_newborn_gcc_test_L=${L}_alpha=${alpha}_gcc_precision_recall_cauchy.npy
		
                echo "---------------"
		echo $dataset, $L
                echo "---------------"

	        python3 auroc_new.py predict ${datasetbase}/${dataset}_newborn/ ${dataset}_newborn ${emb_file_path} binary svm-chisquare ${output_path}
	done
done

