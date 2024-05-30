set -x

gpu=$1
dataset=$2
size=$3

CUDA_VISIBLE_DEVICES=${gpu}

input_folder="/fs/nexus-projects/audio-visual_dereverberation/utkarsh_diff/diff/data/low_res/${size}/similarity/"

python process.py \
    --input "${input_folder}${dataset}_train.tsv" \
    --output "${input_folder}${dataset}_train_processed.txt" \
    --type 'sim'

# python process.py \
#     --input "${input_folder}${dataset}_dev.tsv" \
#     --output "${input_folder}${dataset}_dev_processed.txt" \
#     --type 'tsv'

rm -rf "${input_folder}${dataset}_train_processed.amr"
# rm -rf "${input_folder}${dataset}_dev_processed.amr"

cd amr_parser_spring
bash predict_amr.sh "${input_folder}${dataset}_train_processed.txt"
# bash predict_amr.sh "${input_folder}${dataset}_dev_processed.txt"

rm -rf "${input_folder}${dataset}_train_processed.source"
# rm -rf "${input_folder}${dataset}_dev_processed.source"
rm -rf "${input_folder}${dataset}_train_processed.target"
# rm -rf "${input_folder}${dataset}_dev_processed.target"

cd ../data_utils/preprocess
bash prepare_data.sh "${input_folder}${dataset}_train_processed.amr"
# bash prepare_data.sh "${input_folder}${dataset}_dev_processed.amr"

cd ../../

python filter_amr.py \
    --input "${input_folder}${dataset}_train_processed.source" \
    --output "${input_folder}${dataset}_train_processed_filtered.source" \
    --output_mixner "${input_folder}${dataset}_train_processed_filtered_mixner.source" \
    --target "${input_folder}${dataset}_train_processed.target" \
    --method "append"

# python filter_amr.py \
#     --input "${input_folder}${dataset}_dev_processed.source" \
#     --output "${input_folder}${dataset}_dev_processed_filtered.source" \
#     --output_mixner "${input_folder}${dataset}_dev_processed_filtered_mixner.source" \
#     --target "${input_folder}${dataset}_dev_processed.target" \
#     --method "append"

cd plms_graph2text

rm -rf "${input_folder}${dataset}_train_processed_filtered_generated.source"
rm -rf "${input_folder}${dataset}_train_processed_filtered_mixner_generated.source"
# rm -rf "${input_folder}${dataset}_dev_processed_filtered_generated.source"
rm -rf /fs/nexus-projects/audio-visual_dereverberation/utkarsh_diff/amr-data-augmentation/plms_graph2text/amr/outputs/${dataset}_train_processed_filtered_generated.source
rm -rf /fs/nexus-projects/audio-visual_dereverberation/utkarsh_diff/amr-data-augmentation/plms_graph2text/amr/outputs/${dataset}_train_processed_filtered_mixner_generated.source
# rm -rf /fs/nexus-projects/audio-visual_dereverberation/utkarsh_diff/amr-data-augmentation/plms_graph2text/amr/outputs/${dataset}_dev_processed_filtered_generated.source

bash decode_AMR.sh t5-large /fs/nexus-projects/audio-visual_dereverberation/utkarsh_diff/amr-data-augmentation/amr_parser_spring/amr-models/amr-t5-large.ckpt 0 "${input_folder}${dataset}_train_processed_filtered.source" ${dataset}_train_processed_filtered_generated.source
mv /fs/nexus-projects/audio-visual_dereverberation/utkarsh_diff/amr-data-augmentation/plms_graph2text/amr/outputs/${dataset}_train_processed_filtered_generated.source "${input_folder}"
bash decode_AMR.sh t5-large /fs/nexus-projects/audio-visual_dereverberation/utkarsh_diff/amr-data-augmentation/amr_parser_spring/amr-models/amr-t5-large.ckpt 0 "${input_folder}${dataset}_train_processed_filtered_mixner.source" ${dataset}_train_processed_filtered_mixner_generated.source
mv /fs/nexus-projects/audio-visual_dereverberation/utkarsh_diff/amr-data-augmentation/plms_graph2text/amr/outputs/${dataset}_train_processed_filtered_mixner_generated.source "${input_folder}"
# bash decode_AMR.sh t5-large /fs/nexus-projects/audio-visual_dereverberation/utkarsh_diff/amr-data-augmentation/amr_parser_spring/amr-models/amr-t5-large.ckpt 0 "${input_folder}${dataset}_dev_processed_filtered.source" ${dataset}_dev_processed_filtered_generated.source
# mv /fs/nexus-projects/audio-visual_dereverberation/utkarsh_diff/amr-data-augmentation/plms_graph2text/amr/outputs/${dataset}_dev_processed_filtered_generated.source "${input_folder}"

# cd ../

# python append.py \
#     --input "${input_folder}${dataset}_train_processed_filtered_generated.source" \
#     --sep_path "${input_folder}${dataset}_train.tsv" \
#     --output "${input_folder}${dataset}_train_processed_filtered_generated_sep.source"

# python append.py \
#     --input "${input_folder}${dataset}_train_processed_filtered_mixner_generated.source" \
#     --sep_path "${input_folder}${dataset}_train.tsv" \
#     --output "${input_folder}${dataset}_train_processed_filtered_mixner_generated_sep.source"

# python append.py \
#     --input "${input_folder}${dataset}_dev_processed_filtered_generated.source" \
#     --sep_path "${input_folder}${dataset}_dev.tsv" \
#     --output "${input_folder}${dataset}_dev_processed_filtered_generated_sep.source"