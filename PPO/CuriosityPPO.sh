#!/bin/bash
#SBATCH --job-name=CurPPO
#SBATCH --output=CurPPO.log
#SBATCH --error=CurPPO_err.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8      # Using more cores for biggpu            
#SBATCH --partition=bigbatch
#SBATCH --time=3-00:00:00      # 3 days max runtime 
START_TIME=$(date)
echo "Started at: $START_TIME"
echo "LETS COOOK IT UPPP"


python CuriosityPPO.py --use_icm --total_timesteps 2000000 --num_envs 8 --eval_freq 50000 



END_TIME=$(date)
echo "WEEEE DONEEEE"
echo "Done at: $END_TIME"

