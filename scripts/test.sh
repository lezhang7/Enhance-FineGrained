for atr_weight in 0.2 0.4 0.6 0.8 1.0 
do
    for tec_weight in 0.2 0.4 0.6 0.8 1.0 
    do
       
        output_name=rank_coco-dis_text_mean-hn--5e-06-weightd$tec_weight-weightr$atr_weight-ub5-w_special
        echo "$output_name" >> /home/mila/l/le.zhang/scratch/vision-language-models-are-bows/experiments/run_all_hypertuning_names_coco
    done
done