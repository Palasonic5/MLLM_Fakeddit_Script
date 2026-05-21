#!/bin/bash

#SBATCH --account=torch_pr_520_tandon_priority
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 4
#SBATCH --time=06:00:00
#SBATCH --mem=60GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=llama_fakeddit_eval_6way
#SBATCH --output=/scratch/qz2086/Fakeddit/eval/outputs/llama_eval_6way_%j.out
#SBATCH --error=/scratch/qz2086/Fakeddit/eval/outputs/llama_eval_6way_%j.err

# Load .env (NETID, HF_TOKEN, SLURM_ACCOUNT defined here)
source /scratch/qz2086/Fakeddit/eval/.env

module purge

singularity exec --nv \
    --overlay /scratch/qz2086/sing-mllm/overlay-25GB-500K.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "

source /ext3/env.sh
conda activate py10

export HF_HOME=/scratch/qz2086/hf_cache
export TRANSFORMERS_CACHE=/scratch/qz2086/hf_cache
export HF_TOKEN=${HF_TOKEN}

python /scratch/qz2086/Fakeddit/eval/llama_fakeddit_eval.py \
    --model_path  /scratch/qz2086/mllm/models/llama/llama-3.2-11B-Vision-Instruct \
    --test_tsv    /scratch/qz2086/Fakeddit/dataset/multimodal_only_samples/multimodal_test_public.tsv \
    --image_dir   /scratch/qz2086/Fakeddit/public_image_set \
    --output_path /scratch/qz2086/Fakeddit/eval/outputs/llama_results_6way.csv \
    --plot_dir    /scratch/qz2086/Fakeddit/eval/outputs/plots_6way \
    --label_type  6_way \
    --max_samples 100
"
