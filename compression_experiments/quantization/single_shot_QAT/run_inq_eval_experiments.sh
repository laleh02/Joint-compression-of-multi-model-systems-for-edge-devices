#!/bin/bash

folder="inq_configs"
experiment_logs="inq_logs"

for file in $folder/YuNET*2*; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        filename="${filename%.*}"
        echo "Starting experiment for $filename"
        /usr/bin/python3 -m inq_ptq --experiment_config "$file" > "$experiment_logs/$filename.txt"
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