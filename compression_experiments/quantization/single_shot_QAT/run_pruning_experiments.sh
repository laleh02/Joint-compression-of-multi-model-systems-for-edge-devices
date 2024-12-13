#!/bin/bash

folder="experiment_config/pruning/"
experiment_logs="experiment_logs"

# for file in $folder/*; do
#     if [ -f "$file" ]; then
#         filename=$(basename "$file")
#         filename="${filename%.*}"
#         python3 -m ptq --experiment_config "$file" > "$experiment_logs/$filename.txt"
#         echo "Experiment for $filename finished"
#     fi
# done

for file in $folder/**25*; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        filename="${filename%.*}"
        echo "Pruning Experiment for $filename started"
        python3 -m pruning --experiment_config "$file" > "$experiment_logs/pruning_$filename.txt"
        echo "Pruning Experiment for $filename finished"
    fi
done