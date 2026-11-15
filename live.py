"""Live dashboard: runs RL fine-tuning in a background thread and streams progress to browsers.

    python3 live.py [--port 8000] [--ckpt runs/base_s0.pt]

Open http://localhost:8000 . Every viewer sees the same run; late joiners get the full history.
Uses only the Python standard library for serving (Server-Sent Events).
"""
import argparse, copy, json, os, random, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import torch
import torch.nn.functional as F
from tinyrl import TinyLM, TASKS, V, generate, reward_fn, sample_task, make_prompts, evaluate

HERE = os.path.dirname(os.path.abspath(__file__))
MODES = {  # name -> (learning rate, KL weight)
    "stable": (1e-4, 0.1),
    "collapse": (3e-4, 0.0),
    "anchor": (3e-4, 1.0),
}
STEPS, BS, EVAL_EVERY, REWARD_EVERY, EVAL_N, PAUSE = 1500, 128, 25, 5, 100, 45


class Hub:
    """Shared event history. Clients follow (run_id, index)."""

    def __init__(self):
        self.cv = threading.Condition()
        self.run_id, self.events = 0, []
        self.want = "stable"        # mode for the next run
        self.restart = False

    def new_run(self, ev):
        with self.cv:
            self.run_id += 1
            self.events = [ev]
            self.cv.notify_all()

    def push(self, ev):
        with self.cv:
            self.events.append(ev)
            self.cv.notify_all()


HUB = Hub()


def fixed_samples():
    rng = random.Random(7)
    ps, ans = [], []
    for t in TASKS:
        p, a = make_prompts(rng, 2, [t], True)
        ps += p; ans += a
    p, a = make_prompts(rng, 2, ["add"], False)
    return ps + p, ans + a


def snapshot(model, step, ps, ans):
    ev = evaluate(model, n=EVAL_N)
    outs = generate(model.eval(), ps, greedy=True)[3]
    model.train()
    return {"t": "eval", "step": step, "acc": {k: round(100 * v, 1) for k, v in ev.items()},
            "samples": [{"p": p, "a": a, "o": o, "ok": o == a} for p, a, o in zip(ps, ans, outs)]}


def train_forever(ckpt):
    ps, ans = fixed_samples()
    while True:
        mode = HUB.want
        HUB.restart = False
        lr, kl_w = MODES[mode]
        seed = random.randrange(10 ** 6)
        torch.manual_seed(seed)
        rng = random.Random(seed)
        model = TinyLM()
        model.load_state_dict(torch.load(ckpt))
        ref = copy.deepcopy(model).eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
        base = {t: 0.0 for t in TASKS}
        HUB.new_run({"t": "start", "mode": mode, "lr": lr, "kl": kl_w, "steps": STEPS, "bs": BS})
        HUB.push(snapshot(model, 0, ps, ans))
        acc_r = {t: [0.0, 0] for t in TASKS}
        for step in range(1, STEPS + 1):
            if HUB.restart:
                break
            prompts, gold, tag = [], [], []
            for _ in range(BS):
                t = rng.choice(TASKS)
                p, a = sample_task(t, rng, True)
                prompts.append(p); gold.append(a); tag.append(t)
            model.eval()
            x, start, cur, outs = generate(model, prompts, greedy=False)
            r = torch.tensor([reward_fn(o, a, "sparse") for o, a in zip(outs, gold)])
            b = torch.tensor([base[t] for t in tag])
            for t in TASKS:
                sel = [i for i, tt in enumerate(tag) if tt == t]
                if sel:
                    m = r[sel].mean().item()
                    base[t] = 0.9 * base[t] + 0.1 * m
                    acc_r[t][0] += r[sel].sum().item(); acc_r[t][1] += len(sel)
            model.train()
            T = int(cur.max())
            logp = F.log_softmax(model(x[:, : T - 1]), -1)
            tgt = x[:, 1:T]
            pos = torch.arange(1, T)[None, :]
            amask = (pos >= start[:, None]) & (pos < cur[:, None])
            lp = (logp.gather(-1, tgt.clamp(max=V - 1)[..., None]).squeeze(-1) * amask).sum(1)
            loss = -((r - b) * lp).mean()
            if kl_w > 0:
                with torch.no_grad():
                    rlogp = F.log_softmax(ref(x[:, : T - 1]), -1)
                loss = loss + kl_w * ((logp.exp() * (logp - rlogp)).sum(-1) * amask).sum() / amask.sum()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if step % REWARD_EVERY == 0:
                HUB.push({"t": "reward", "step": step,
                          "r": {t: round(100 * s / max(n, 1), 1) for t, (s, n) in acc_r.items()}})
                acc_r = {t: [0.0, 0] for t in TASKS}
            if step % EVAL_EVERY == 0:
                HUB.push(snapshot(model, step, ps, ans))
        else:
            HUB.push({"t": "done", "pause": PAUSE})
            t0 = time.time()
            while time.time() - t0 < PAUSE and not HUB.restart:
                time.sleep(0.5)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            body = open(os.path.join(HERE, "live.html"), "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        elif u.path == "/start":
            mode = parse_qs(u.query).get("mode", ["stable"])[0]
            if mode in MODES:
                HUB.want = mode
                HUB.restart = True
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif u.path == "/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Connection", "close")
            self.end_headers()
            rid, idx = -1, 0
            try:
                while True:
                    with HUB.cv:
                        if rid == HUB.run_id and idx >= len(HUB.events):
                            HUB.cv.wait(timeout=15)
                        if rid != HUB.run_id:
                            rid, idx = HUB.run_id, 0
                        batch = HUB.events[idx:]
                        idx += len(batch)
                    out = "".join(f"data: {json.dumps(e)}\n\n" for e in batch) or ": keepalive\n\n"
                    self.wfile.write(out.encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--ckpt", default=os.path.join(HERE, "runs", "base_s0.pt"))
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    threading.Thread(target=train_forever, args=(a.ckpt,), daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), Handler)
    srv.daemon_threads = True
    print(f"listening on http://0.0.0.0:{a.port}", flush=True)
    srv.serve_forever()
