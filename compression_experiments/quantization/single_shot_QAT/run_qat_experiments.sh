folder="experiment_config_seed1"
experiment_logs="experiment_logs_seed1"

# for file in $folder/*; do
#     if [ -f "$file" ]; then
#         filename=$(basename "$file")
#         filename="${filename%.*}"
#         python3 -m ptq --experiment_config "$file" > "$experiment_logs/$filename.txt"
#         echo "Experiment for $filename finished"
#     fi
# done
for file in $folder/YuNET.yml; do

    if [ -f "$file" ]; then
        filename=$(basename "$file")
        filename="${filename%.*}"
        echo "Starting QAT Experiment for $filename"

        /usr/bin/python3 -m qat_save_only --experiment_config "$file" > "$experiment_logs/qat_$filename.txt"
        echo "QAT Experiment for $filename finished"
    fi
done






