"""Tiny character-level transformer LM + supervised pretraining + REINFORCE fine-tuning.

Everything (tasks, model, training loops, evaluation) lives in this one file.
"""
import argparse, copy, json, math, os, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# Vocabulary
# ----------------------------------------------------------------------------
CHARS = list("0123456789abcdefghij+=:ARS") + ["<eos>", "<pad>"]
STOI = {c: i for i, c in enumerate(CHARS)}
EOS, PAD = STOI["<eos>"], STOI["<pad>"]
V = len(CHARS)
MAX_LEN = 28          # context length (prompt + answer + eos)
MAX_ANS = 10          # max generated tokens (incl. eos)
TASKS = ["add", "rev", "sort"]


def enc(s):
    return [STOI[c] for c in s]


def dec(ids):
    return "".join(CHARS[i] for i in ids)


# ----------------------------------------------------------------------------
# Tasks.  Each sample is (prompt string, answer string).
# `hard=False`: the "easy" regime, `hard=True`: the "hard" regime.
# ----------------------------------------------------------------------------
EASY = {"add": (1, 2), "rev": (2, 5), "sort": (2, 5)}   # digits per operand / string length
HARD = {"add": (3, 4), "rev": (6, 9), "sort": (6, 9)}


def sample_task(task, rng, hard):
    lo, hi = (HARD if hard else EASY)[task]
    n = rng.randint(lo, hi)
    if task == "add":
        m = rng.randint(lo, hi)
        a = rng.randint(10 ** (n - 1) if n > 1 else 0, 10 ** n - 1)
        b = rng.randint(10 ** (m - 1) if m > 1 else 0, 10 ** m - 1)
        return f"A:{a}+{b}=", str(a + b)
    if task == "rev":
        s = "".join(rng.choice("abcdefghij") for _ in range(n))
        return f"R:{s}=", s[::-1]
    if task == "sort":
        s = "".join(rng.choice("0123456789") for _ in range(n))
        return f"S:{s}=", "".join(sorted(s))
    raise ValueError(task)


def make_batch(rng, bs, tasks, p_hard):
    """Supervised batch: full sequence prompt+answer+eos, loss on answer part only."""
    x = torch.full((bs, MAX_LEN), PAD, dtype=torch.long)
    mask = torch.zeros((bs, MAX_LEN), dtype=torch.bool)
    for i in range(bs):
        p, a = sample_task(rng.choice(tasks), rng, rng.random() < p_hard)
        ids = enc(p) + enc(a) + [EOS]
        x[i, : len(ids)] = torch.tensor(ids)
        mask[i, len(p): len(ids)] = True
    return x, mask


def make_prompts(rng, bs, tasks, hard):
    ps, ans = [], []
    for _ in range(bs):
        p, a = sample_task(rng.choice(tasks), rng, hard)
        ps.append(p)
        ans.append(a)
    return ps, ans


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, m):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=m, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class TinyLM(nn.Module):
    def __init__(self, d=96, h=4, L=3):
        super().__init__()
        self.tok = nn.Embedding(V, d)
        self.pos = nn.Embedding(MAX_LEN, d)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(L)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, V, bias=False)

    def forward(self, x):
        T = x.shape[1]
        m = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        h = self.tok(x) + self.pos(torch.arange(T))
        for b in self.blocks:
            h = b(h, m)
        return self.head(self.ln(h))


# ----------------------------------------------------------------------------
# Generation: all prompts in a batch are left-aligned; we track a per-row cursor.
# ----------------------------------------------------------------------------
@torch.no_grad()
def generate(model, prompts, greedy):
    bs = len(prompts)
    x = torch.full((bs, MAX_LEN), PAD, dtype=torch.long)
    cur = torch.zeros(bs, dtype=torch.long)
    for i, p in enumerate(prompts):
        ids = enc(p)
        x[i, : len(ids)] = torch.tensor(ids)
        cur[i] = len(ids)
    start = cur.clone()
    done = torch.zeros(bs, dtype=torch.bool)
    rows = torch.arange(bs)
    for _ in range(MAX_ANS):
        T = int(cur.max())
        logits = model(x[:, :T])[rows, cur - 1]
        logits[:, PAD] = -1e9
        nxt = logits.argmax(-1) if greedy else torch.multinomial(F.softmax(logits, -1), 1).squeeze(1)
        act = ~done & (cur < MAX_LEN)
        x[rows[act], cur[act]] = nxt[act]
        cur = cur + act.long()
        done = done | (nxt == EOS) | (cur >= MAX_LEN)
        if done.all():
            break
    outs = []
    for i in range(bs):
        ids = x[i, start[i]: cur[i]].tolist()
        outs.append(dec([t for t in ids if t != EOS]))
    return x, start, cur, outs


def reward_fn(out, ans, kind):
    if kind == "sparse":
        return float(out == ans)
    # dense: exact match gives 1; otherwise partial credit for position-wise matches
    if out == ans:
        return 1.0
    L = max(len(out), len(ans))
    hit = sum(a == b for a, b in zip(out, ans))
    return 0.5 * hit / L


@torch.no_grad()
def evaluate(model, seed=12345, n=500):
    """Greedy exact-match accuracy per task and regime on a fixed held-out set."""
    model.eval()
    res = {}
    for hard in (False, True):
        for t in TASKS:
            rng = random.Random(seed + TASKS.index(t) * 7 + hard)
            ps, ans = make_prompts(rng, n, [t], hard)
            outs = generate(model, ps, greedy=True)[3]
            res[f"{t}_{'hard' if hard else 'easy'}"] = sum(o == a for o, a in zip(outs, ans)) / n
    model.train()
    return res


# ----------------------------------------------------------------------------
# Stage 1: supervised pretraining (maximum likelihood)
# ----------------------------------------------------------------------------
def pretrain(args):
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    model = TinyLM()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    log = []
    for step in range(1, args.steps + 1):
        x, mask = make_batch(rng, args.bs, TASKS, args.p_hard)
        logits = model(x[:, :-1])
        loss = F.cross_entropy(logits[mask[:, 1:]], x[:, 1:][mask[:, 1:]])
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            ev = evaluate(model)
            log.append({"step": step, "loss": loss.item(), **ev})
            print(step, f"{loss.item():.3f}", {k: round(v, 2) for k, v in ev.items()}, flush=True)
    return model, log


# ----------------------------------------------------------------------------
# Stage 2: fine-tuning on the hard regime
#   method = "rl"  : REINFORCE, only a scalar reward from a verifier, no labels
#   method = "sft" : supervised fine-tuning with gold answers (reference upper bound)
# ----------------------------------------------------------------------------
def finetune(model, args):
    torch.manual_seed(args.seed + 1)
    rng = random.Random(args.seed + 1)
    ref = copy.deepcopy(model).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    tasks = args.tasks.split(",")
    base = {t: 0.0 for t in tasks}          # per-task moving-average baseline
    log = [{"step": 0, "reward": float("nan"), **evaluate(model)}]
    rows = torch.arange(args.bs)
    for step in range(1, args.steps + 1):
        if args.method == "sft":
            x, mask = make_batch(rng, args.bs, tasks, 1.0)
            logits = model(x[:, :-1])
            loss = F.cross_entropy(logits[mask[:, 1:]], x[:, 1:][mask[:, 1:]])
            mean_r = float("nan")
        else:
            ps, ans = [], []
            ttag = []
            for _ in range(args.bs):
                t = rng.choice(tasks)
                p, a = sample_task(t, rng, True)
                ps.append(p); ans.append(a); ttag.append(t)
            model.eval()
            x, start, cur, outs = generate(model, ps, greedy=False)
            r = torch.tensor([reward_fn(o, a, args.reward) for o, a in zip(outs, ans)])
            if args.baseline == "greedy":      # self-critical: reward of the greedy decode
                g = generate(model, ps, greedy=True)[3]
                b = torch.tensor([reward_fn(o, a, args.reward) for o, a in zip(g, ans)])
            elif args.baseline == "ema":
                b = torch.tensor([base[t] for t in ttag])
                for t in tasks:
                    sel = [i for i, tt in enumerate(ttag) if tt == t]
                    if sel:
                        base[t] = 0.9 * base[t] + 0.1 * r[sel].mean().item()
            else:
                b = torch.zeros_like(r)
            model.train()
            T = int(cur.max())
            logits = model(x[:, : T - 1])                      # predicts tokens 1..T-1
            logp = F.log_softmax(logits, -1)
            tgt = x[:, 1:T]
            pos = torch.arange(1, T)[None, :]
            amask = (pos >= start[:, None]) & (pos < cur[:, None])   # generated positions
            lp_tok = logp.gather(-1, tgt.clamp(max=V - 1)[..., None]).squeeze(-1)
            lp_seq = (lp_tok * amask).sum(1)
            adv = r - b
            loss = -(adv * lp_seq).mean()
            if args.ent > 0:
                ent = -(logp.exp() * logp).sum(-1)
                loss = loss - args.ent * (ent * amask).sum() / amask.sum()
            if args.kl > 0:                                    # stay close to the pretrained LM
                with torch.no_grad():
                    rlogp = F.log_softmax(ref(x[:, : T - 1]), -1)
                kl = (logp.exp() * (logp - rlogp)).sum(-1)
                loss = loss + args.kl * (kl * amask).sum() / amask.sum()
            mean_r = r.mean().item()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            ev = evaluate(model)
            log.append({"step": step, "reward": mean_r, **ev})
            print(step, f"r={mean_r:.3f}", {k: round(v, 2) for k, v in ev.items()}, flush=True)
    return model, log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pretrain", "finetune"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--p_hard", type=float, default=0.1)
    ap.add_argument("--method", default="rl")
    ap.add_argument("--tasks", default="add,rev,sort")
    ap.add_argument("--reward", default="dense")
    ap.add_argument("--baseline", default="ema")
    ap.add_argument("--ent", type=float, default=0.0)
    ap.add_argument("--kl", type=float, default=0.0)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()
    if args.mode == "pretrain":
        model, log = pretrain(args)
    else:
        model = TinyLM()
        model.load_state_dict(torch.load(args.ckpt))
        model, log = finetune(model, args)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(model.state_dict(), args.out + ".pt")
    json.dump({"args": vars(args), "log": log, "time": time.time() - t0,
               "params": sum(p.numel() for p in model.parameters())}, open(args.out + ".json", "w"))
