#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 4
#SBATCH --time=02:00:00
#SBATCH --mem=60GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=llama_fakeddit_finetuned_6way
#SBATCH --output=/scratch/qz2086/Fakeddit/eval/outputs/llama_finetuned_eval_6way_%j.out
#SBATCH --error=/scratch/qz2086/Fakeddit/eval/outputs/llama_finetuned_eval_6way_%j.err

source /scratch/qz2086/Fakeddit/eval/.env

module purge

singularity exec --nv \
    --overlay /scratch/${NETID}/sing-mllm/overlay-25GB-500K.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "

source /ext3/env.sh
conda activate py10

export HF_HOME=/scratch/${NETID}/hf_cache
export TRANSFORMERS_CACHE=/scratch/${NETID}/hf_cache
export HF_TOKEN=${HF_TOKEN}

python /scratch/${NETID}/Fakeddit/eval/llama_fakeddit_eval.py \
    --model_path  /scratch/${NETID}/Fakeddit/finetune/checkpoints/llama_fakeddit \
    --test_tsv    /scratch/${NETID}/Fakeddit/dataset/multimodal_only_samples/multimodal_test_public.tsv \
    --image_dir   /scratch/${NETID}/Fakeddit/public_image_set \
    --output_path /scratch/${NETID}/Fakeddit/eval/outputs/llama_finetuned_results_6way_100.csv \
    --plot_dir    /scratch/${NETID}/Fakeddit/eval/outputs/plots_finetuned_6way \
    --label_type  6_way \
    --max_samples 100
"