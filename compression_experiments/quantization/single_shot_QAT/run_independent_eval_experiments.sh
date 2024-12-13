#!/bin/bash

folder="eval_only_configs"
experiment_logs="eval_only_logs"

for file in $folder/qat_YuNET_adaround_seed2*; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        filename="${filename%.*}"
        echo "Starting experiment for $filename"
        /usr/bin/python3 -m eval_only_ind --experiment_config "$file" > "$experiment_logs/$filename.txt"
        echo "Experiment for $filename finished"
    fi
done


# for file in $folder/YuNET*.yml; do

#     if [ -f "$file" ]; then
#         filename=$(basename "$file")
#         filename="${filename%.*}"
#         echo "Starting QAT Experiment for $filename"

#         /usr/bin/python3 -m qat_seed1 --experiment_config "$file" > "$experiment_logs/qat_$filename.txt"
#         echo "QAT Experiment for $filename finished"
#     fi
# done
# for file in $folder/*; do
#     if [ -f "$file" ]; then
#         filename=$(basename "$file")
#         filename="${filename%.*}"
#         python3 -m qat --experiment_config "$file" > "$experiment_logs/qat_$filename.txt"
#         echo "QAT Experiment for $filename finished"
#     fi
# done