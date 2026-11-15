# tiny-llm-rl

A 344K-parameter character-level transformer, pretrained with maximum likelihood and then
fine-tuned with REINFORCE (reward only) on hard instances of addition, reversal and sorting.

    pip install torch numpy matplotlib
    for s in 0 1 2; do python3 tinyrl.py pretrain --seed $s --steps 2500 --eval_every 500 --p_hard 0.02 --out runs/base_s$s; done
    ./run_all.sh          # 24 fine-tuning runs, 4 in parallel (~1 hour on 4 cores)
    python3 analyze.py    # writes paper/table_main.tex and figures
    cd paper && pdflatex paper.tex && pdflatex paper.tex

Live dashboard (Turkish UI): `python3 live.py --port 8000`, then open http://localhost:8000 .
Turkish paper: `paper/makale.pdf`.

`runs/trials/` holds the exploratory runs used to pick hyperparameters. Paper: `paper/paper.pdf`.
