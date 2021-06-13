
edgelistbase="./datasets"
outputbase="./gcc_datasets"


#datasets=(blogcatalog cora dblp Homo_sapiens wiki)
datasets=( flickr_small )

for dataset in ${datasets[@]}
do
	edge_file_path=${edgelistbase}/${dataset}_newborn.edgelist
	output_folder=${outputbase}/${dataset}_newborn/
        mkdir -p $output_folder
		
        echo "---------------"
	echo $dataset
        echo "---------------"

  
	python auroc.py split ${edgelistbase}/${dataset}_newborn.gml ${output_folder}

done

