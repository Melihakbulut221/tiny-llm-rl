#!/bin/bash
# Full experiment grid: 8 fine-tuning configs x 3 seeds, 4 runs in parallel.
# Base models: python3 tinyrl.py pretrain --seed S --steps 2500 --eval_every 500 --p_hard 0.02 --out runs/base_sS
cd "$(dirname "$0")"
C="--steps 1500 --eval_every 150"
{
for s in 0 1 2; do
B="finetune --ckpt runs/base_s$s.pt --seed $s $C"
echo "$B --lr 1e-4 --kl 0.1 --reward sparse --out runs/rl_s$s"
echo "$B --lr 1e-4 --method sft --out runs/sft_s$s"
echo "$B --lr 1e-4 --kl 0.1 --reward sparse --baseline none --out runs/rl_nobase_s$s"
echo "$B --lr 1e-4 --kl 0.1 --reward dense --out runs/rl_dense_s$s"
echo "$B --lr 1e-4 --kl 0 --reward sparse --out runs/rl_nokl_s$s"
echo "$B --lr 3e-4 --kl 0 --reward sparse --out runs/rl_hilr_nokl_s$s"
echo "$B --lr 3e-4 --kl 1.0 --reward sparse --out runs/rl_hilr_kl1_s$s"
echo "$B --lr 1e-4 --kl 0.1 --reward sparse --tasks sort --out runs/rl_sortonly_s$s"
done
} | xargs -P 4 -I{} sh -c 'python3 tinyrl.py {} > /dev/null 2>&1'
echo ALLDONE
