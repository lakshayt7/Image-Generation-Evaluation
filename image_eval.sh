#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --output=logs/exp_jobid_%j.out
#SBATCH --error=logs/image-eval%j.err
#SBATCH --gres=gpu:a100:1

module load cuda90/toolkit/9.1.176
echo -e "GPUS = $CUDA_VISIBLE_DEVICES\n"
nvidia-smi

source /gpfs/scratch/lt2504/image-eval/imagen/bin/activate

python -m pytorch_fid $1 $2
  
python image-eval/Image-Generation-Evaluation/src/main.py --gen_image_path $1 --gt_images $2 --text_prompt "photo of class <sks> 
