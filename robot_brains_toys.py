# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.16",
#     "matplotlib>=3.8",
#     "numpy>=1.26",
#     "torch>=2.3",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="Robot Brains Toybox")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _():
    import copy
    import math
    import time

    import matplotlib
    import numpy as np
    import torch
    import torch.nn as nn
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle

    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    return Circle, Figure, copy, math, matplotlib, nn, np, time, torch


@app.cell(hide_code=True)
def _(matplotlib):
    # One palette for every figure: indigo = System 2, amber = System 1 VLA, teal = world models.
    PAL = {
        "s2": "#4a46c8",
        "s1": "#c27100",
        "wm": "#0b8a7e",
        "bad": "#d1453b",
        "good": "#2f9e5a",
        "ink": "#22262e",
        "grey": "#9aa1ab",
        "faint": "#e3e6ea",
    }
    matplotlib.rcParams.update(
        {
            "figure.dpi": 110,
            "font.size": 9.5,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.edgecolor": "#8a9099",
            "legend.frameon": False,
            "figure.constrained_layout.use": True,
        }
    )
    return (PAL,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Robot brains toybox

    A runnable companion to the *Robot Brains Field Guide*. Each section rebuilds one mechanism from the reading list at toy scale: a 2-D world, 16×16-pixel "video", MLPs and a two-layer transformer. Everything trains on CPU. The first full run takes one to three minutes on a laptop; after that most sliders respond instantly, and the cells that retrain say so.

    Run it with `uvx marimo edit --sandbox robot_brains_toys.py`. The header pins numpy, matplotlib, torch and marimo. The models are tiny on purpose, so a GPU would not help: at these sizes every step is launch overhead.

    | § | Mechanism | Toy | Sources |
    |---|---|---|---|
    | 1 | Two clocks: why System 2 and System 1 split | a shoved point mass with a slow planner | GR00T N1, π0.7, DreamZero, ASPIRE, Argon |
    | 2 | Generative action heads | flow matching vs MSE around an obstacle | π0.7, GR00T N1.7, DreamZero |
    | 3 | Chunks under latency | asynchronous execution, real-time chunking | π0.7, DreamZero, Dream-1 |
    | 4 | Discrete targets for the backbone | a FAST-style DCT tokenizer | π0.7 |
    | 5 | Steering with context | quality and speed metadata, then CFG | π0.7 |
    | 6 | World action models | video model × inverse dynamics vs VLA vs latent alignment | DreamZero, GR00T 2, Dream-1, GEAR FLARE |
    | 7 | Physical prompting | in-context imitation and few-step fine-tuning | GEN-1.5 |
    | 8 | Fleet learning | FedAvg and friends across mismatched robots | NVIDIA FLARE |
    | 9 | A System 2 that learns | code-as-policy agent with a skill library | ASPIRE, Gemini 4 Argon |

    Sections 2–5 build up a System 1 VLA. Sections 6–7 are the two big bets on what System 1 should become. Sections 8–9 are the two learning loops: weights across a fleet, and text skills inside System 2. Every section ends with what its toy leaves out.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · Two clocks

    Every system in the reading list splits work by time scale. GR00T N1 put a VLM System 2 at 10 Hz in front of a flow-matching DiT System 1 at 120 Hz. π0.7 runs a language policy and a subgoal-image generator in separate threads while the VLA serves robots at 50 Hz (20 Hz for the UR5e). ASPIRE and Gemini 4 Argon sit at the slow end: frontier LLMs that write and repair programs, one LLM call at a time. GEN-0 takes the opposite view and interleaves sensing and acting tokens in one stream with no split at all.

    Why split? A feedback loop is only as good as the age of the state it acts on. The toy is a force-controlled point mass that must visit four waypoints while getting shoved twice.

    - **Monolith:** one slow model reads the state, thinks for the latency, and outputs a force that is held until its next answer.
    - **Two systems:** the same slow model only decides which waypoint comes next. A fast PD loop recomputes the force from fresh state at the System 1 rate.

    Set System 2 to 20 Hz with zero latency and the monolith copes. Then add 100 ms. Delay in a feedback loop is phase lag; past some point the controller pushes hard in a direction that was right a while ago.
    """)
    return


@app.cell
def _(math, np):
    CLK_WAYPOINTS = np.array([[0.6, 0.0], [0.6, 0.6], [-0.4, 0.6], [-0.4, -0.3]])

    def clk_simulate(s2_hz, s2_latency, s1_hz, two_systems, T=7.0, dt=0.001, kp=40.0):
        """Force-controlled point mass. Returns times, positions, distance to the active waypoint, waypoints reached."""
        kd = 2 * math.sqrt(kp) * 0.9
        x, v, u = np.array([-0.6, -0.4]), np.zeros(2), np.zeros(2)
        goal_idx, pending = 0, []
        next_s2 = next_s1 = 0.0
        shoves = {int(2.0 / dt): np.array([0.0, -2.5]), int(4.2 / dt): np.array([2.5, 0.0])}
        ts, xs, err = [], [], []
        for i in range(int(T / dt)):
            t = i * dt
            if t >= next_s2 - 1e-12:  # the slow model starts thinking about what it sees now...
                gi = goal_idx
                if np.linalg.norm(x - CLK_WAYPOINTS[gi]) < 0.1 and gi < len(CLK_WAYPOINTS) - 1:
                    gi += 1
                force = kp * (CLK_WAYPOINTS[gi] - x) - kd * v
                pending.append((t + s2_latency, gi, None if two_systems else force))
                next_s2 += 1.0 / s2_hz
            while pending and pending[0][0] <= t + 1e-12:  # ...and its answer lands `latency` later
                _ready, goal_idx, force = pending.pop(0)
                if force is not None:
                    u = force  # monolith: this force is held until the next answer
            if two_systems and t >= next_s1 - 1e-12:  # fast inner loop on fresh state
                u = kp * (CLK_WAYPOINTS[goal_idx] - x) - kd * v
                next_s1 += 1.0 / s1_hz
            if i in shoves:
                v = v + shoves[i]
            v = v + dt * np.clip(u, -30, 30)
            x = np.clip(x + dt * v, -1.5, 1.5)
            if i % 10 == 0:
                ts.append(t)
                xs.append(x.copy())
                err.append(np.linalg.norm(x - CLK_WAYPOINTS[goal_idx]))
        reached = goal_idx + int(np.linalg.norm(x - CLK_WAYPOINTS[goal_idx]) < 0.1)
        return np.array(ts), np.array(xs), np.array(err), reached
    return CLK_WAYPOINTS, clk_simulate


@app.cell
def _(mo):
    clk_s2_hz = mo.ui.slider(0.5, 20, step=0.5, value=2, label="System 2 rate (Hz)", show_value=True)
    clk_lat = mo.ui.slider(0, 600, step=25, value=250, label="System 2 latency (ms)", show_value=True)
    clk_s1_hz = mo.ui.slider(5, 200, step=5, value=100, label="System 1 rate (Hz)", show_value=True)
    mo.hstack([clk_s2_hz, clk_lat, clk_s1_hz], justify="start", gap=1.5, wrap=True)
    return clk_lat, clk_s1_hz, clk_s2_hz


@app.cell
def _(CLK_WAYPOINTS, Figure, PAL, clk_lat, clk_s1_hz, clk_s2_hz, clk_simulate, mo):
    _runs = {
        "Monolith": clk_simulate(clk_s2_hz.value, clk_lat.value / 1000, clk_s1_hz.value, two_systems=False),
        "Two systems": clk_simulate(clk_s2_hz.value, clk_lat.value / 1000, clk_s1_hz.value, two_systems=True),
    }
    _fig = Figure(figsize=(9.6, 3.7))
    _ax_path, _ax_err = _fig.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.4]})
    for _name, _color in [("Monolith", PAL["bad"]), ("Two systems", PAL["s1"])]:
        _t, _x, _e, _n = _runs[_name]
        _ax_path.plot(_x[:, 0], _x[:, 1], color=_color, lw=1.6, label=_name)
        _ax_err.plot(_t, _e, color=_color, lw=1.6, label=_name)
    _ax_path.plot(*CLK_WAYPOINTS.T, "o--", color=PAL["s2"], ms=7, lw=1, label="plan (System 2)")
    _ax_path.plot(-0.6, -0.4, "s", color=PAL["ink"], ms=5)
    _ax_path.set(xlim=(-1.55, 1.55), ylim=(-1.55, 1.55), aspect="equal", title="Paths, shoved at 2.0 s and 4.2 s")
    _ax_path.legend(loc="lower right", fontsize=8)
    for _ts in (2.0, 4.2):
        _ax_err.axvline(_ts, color=PAL["grey"], ls=":", lw=1)
    _ax_err.set(xlabel="time (s)", ylabel="distance to current waypoint", title="Tracking error", ylim=(0, None))
    _ax_err.legend(fontsize=8)
    _summary = " · ".join(
        f"**{_k}:** {_r[3]}/4 waypoints, mean error {_r[2].mean():.2f}" for _k, _r in _runs.items()
    )
    mo.vstack([_fig, mo.md(_summary)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **The real clocks**, as the sources report them:

    | System | Slow loop | Fast loop |
    |---|---|---|
    | GR00T N1 | VLM at 10 Hz on an L40 | DiT action head at 120 Hz |
    | GR00T N1.7 | inside one 3B model | ≈36 Hz on H100, 10.1 Hz on DGX Spark (NVIDIA's TensorRT throughput figure) |
    | π0.7 | language policy + subgoal-image generator in separate threads; new subgoal per subtask or every 4 s at most | 5B VLA, 50-step chunks from 5 denoising steps, trained to tolerate up to 240 ms of delay; robots at 50 Hz |
    | DreamZero | none | 14B WAM, ≈150 ms per 48-step chunk on 2×GB200; robot at 30 Hz |
    | Dream-1 | none | one API call returns 1.6 s of actions |
    | GEN-1.5 | none (GEN-0's interleaved stream) | 100 Hz action trajectories |
    | ASPIRE, Argon | frontier LLM writing programs; latency not stated, one or more LLM calls per repair | scripted primitives |

    **What this toy leaves out.** Real System 1 outputs are setpoints for still faster joint controllers, PD or impedance loops at hundreds of Hz or more; π0.7 applies its actions through a simple PD controller. Real System 2 latency is also variable, one more reason to keep it out of the inner loop.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Why the action head is generative

    Demonstrations are multimodal. Around an obstacle, some operators pass above and some below. Regressing actions with MSE learns $\mathbb{E}[a \mid s]$, and the average of "above" and "below" is "straight through". Every System 1 in the reading list that discloses its head uses a generative one instead: π0.7's 860M flow-matching action expert, GR00T N1.7's 32-layer flow-matching DiT, DreamZero's joint video-and-action denoiser.

    You know the picture from Welch Labs' diffusion explainer: a learned vector field carries noise to data. Flow matching uses straight paths. With $x_0 \sim \mathcal{N}(0, I)$ and a data sample $x_1$, set $x_t = (1-t)\,x_0 + t\,x_1$, regress $v_\theta(x_t, t, s)$ onto $x_1 - x_0$, then integrate $\dot x = v_\theta$ from $t=0$ to $1$. Here $x_1$ is a whole **action chunk**: the next 16 waypoints, 32 numbers, conditioned on the robot's position $s$.

    The next two cells define the world and train both heads on 300 demonstrations (10–30 s on CPU). The flow head is also trained with random *committed prefixes*; ignore that until §3.
    """)
    return


@app.cell
def _(math, nn, np, torch):
    NAV_OBS_C = np.array([0.0, 0.0])
    NAV_OBS_R = 0.30
    NAV_GOAL = np.array([1.0, 0.0])
    NAV_GOAL_R = 0.12
    NAV_H = 16  # waypoints per action chunk
    NAV_SCALE = 4.0  # chunk offsets are small numbers; scale them up for flow training

    def nav_bezier(p0, p1, p2, p3, n):
        t = np.linspace(0, 1, n)[:, None]
        return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3

    def nav_resample(path, step):
        seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        q = np.linspace(0, s[-1], max(2, int(s[-1] / step) + 1))
        return np.stack([np.interp(q, s, path[:, i]) for i in range(2)], 1)

    def nav_demo(rng, side=None, step=0.06, wobble=0.0, detour=0.0):
        """One demonstration from the left edge to the goal, passing above (+1) or below (-1) the obstacle.
        step = waypoint spacing (operator speed); wobble = unsteady hand; detour = a wide swing (a mistake)."""
        side = int(rng.choice([-1, 1])) if side is None else side
        p0 = np.array([-1.0, rng.uniform(-0.25, 0.25)])
        via = np.array([0.0, side * rng.uniform(0.5, 0.62)])
        a = nav_bezier(p0, p0 + [0.30, side * 0.22], via + [-0.45, 0], via, 60)
        b = nav_bezier(via, via + [0.45, 0], NAV_GOAL + [-0.35, 0], NAV_GOAL + rng.normal(0, 0.02, 2), 60)
        path = np.concatenate([a, b[1:]])
        if detour > 0:
            k = rng.integers(25, 95)
            bump = np.exp(-0.5 * ((np.arange(len(path)) - k) / 9.0) ** 2)[:, None]
            path = path + detour * bump * np.array([-0.3, side])
        path = nav_resample(path, step)
        if wobble > 0:
            n = len(path)
            w = np.cumsum(rng.normal(0, wobble, (n, 2)), 0)
            w -= np.linspace(0, 1, n)[:, None] * w[-1]
            path = path + w * np.sin(np.linspace(0, np.pi, n))[:, None]
        return path, side

    def nav_chunks(path):
        """(state, chunk) training pairs: a chunk is the next NAV_H waypoints relative to the current one."""
        S, C = [], []
        for t in range(len(path)):
            fut = path[t + 1 : t + 1 + NAV_H]
            if len(fut) < NAV_H:
                pad = np.repeat(path[-1:], NAV_H - len(fut), 0)
                fut = np.concatenate([fut, pad]) if len(fut) else pad
            S.append(path[t])
            C.append(fut - path[t])
        return np.array(S), np.array(C)

    def nav_hits(a, b):
        """Does the straight segment a -> b pass through the obstacle?"""
        for u in np.linspace(0, 1, 6):
            if np.linalg.norm(a * (1 - u) + b * u - NAV_OBS_C) < NAV_OBS_R:
                return True
        return False

    def tfeat(t):
        f = torch.arange(1, 5, dtype=torch.float32) * math.pi
        return torch.cat([torch.sin(f * t), torch.cos(f * t)], -1)

    class MLP(nn.Module):
        def __init__(self, din, dout, hidden=256, depth=3):
            super().__init__()
            layers, d = [], din
            for _layer in range(depth):
                layers += [nn.Linear(d, hidden), nn.SiLU()]
                d = hidden
            layers.append(nn.Linear(d, dout))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    def train_mse(S, C, steps=1500, bs=512, lr=2e-3, seed=0):
        torch.manual_seed(seed)
        model = MLP(2, 2 * NAV_H, 128, 3)
        opt = torch.optim.Adam(model.parameters(), lr)
        S_ = torch.tensor(S, dtype=torch.float32)
        Y = torch.tensor(C.reshape(len(C), -1) * NAV_SCALE, dtype=torch.float32)
        for _step in range(steps):
            idx = torch.randint(0, len(S_), (bs,))
            loss = ((model(S_[idx]) - Y[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        return model

    @torch.no_grad()
    def predict_mse(model, s):
        out = model(torch.tensor(np.asarray(s), dtype=torch.float32)) / NAV_SCALE
        return out.numpy().reshape(len(s), NAV_H, 2)

    def train_flow(S, C, steps=2000, bs=512, lr=2e-3, cond=None, p_drop=0.0, prefix_max=0, seed=0, hidden=256):
        """Conditional flow matching on whole action chunks.
        cond: optional (N, k) metadata, zeroed with probability p_drop; zeros act as the unconditional model for CFG.
        prefix_max > 0: train with a random clean prefix of 1..prefix_max steps on 60% of samples (training-time RTC)."""
        torch.manual_seed(seed)
        H2 = 2 * NAV_H
        kc = 0 if cond is None else cond.shape[1]
        model = MLP(H2 + 2 + 8 + kc + (NAV_H if prefix_max else 0), H2, hidden, 3)
        opt = torch.optim.Adam(model.parameters(), lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        S_ = torch.tensor(S, dtype=torch.float32)
        X1 = torch.tensor(C.reshape(len(C), -1) * NAV_SCALE, dtype=torch.float32)
        K_ = None if cond is None else torch.tensor(cond, dtype=torch.float32)
        for _step in range(steps):
            idx = torch.randint(0, len(S_), (bs,))
            x1, s = X1[idx], S_[idx]
            x0 = torch.randn_like(x1)
            t = torch.rand(bs, 1)
            xt = (1 - t) * x0 + t * x1
            target = x1 - x0
            weight = torch.ones_like(x1)
            extra = []
            if K_ is not None:
                keep = (torch.rand(bs, 1) >= p_drop).float()
                extra.append(K_[idx] * keep)
            if prefix_max:
                d = torch.randint(1, prefix_max + 1, (bs,)) * (torch.rand(bs) < 0.6)
                m = (torch.arange(NAV_H)[None] < d[:, None]).float()
                m2 = m.repeat_interleave(2, 1)
                xt = m2 * x1 + (1 - m2) * xt  # the committed prefix enters clean...
                extra.append(m)
                weight = 1 - m2  # ...and is not part of the loss
            pred = model(torch.cat([xt, s, tfeat(t)] + extra, -1))
            loss = (((pred - target) ** 2) * weight).sum() / weight.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        model.kc, model.prefix = kc, bool(prefix_max)
        return model

    @torch.no_grad()
    def sample_flow(model, s, n_steps=10, cond=None, beta=0.0, prefix=None, gen=None):
        """Euler-integrate the learned field from noise (t=0) to an action chunk (t=1).
        prefix = (values (B,H,2) relative to s, mask (B,H)) pins the committed steps (real-time chunking).
        beta = classifier-free guidance on the metadata: v = v_c + beta (v_c - v_u)."""
        B, H2 = len(s), 2 * NAV_H
        S_ = torch.tensor(np.asarray(s), dtype=torch.float32)
        x = torch.randn(B, H2, generator=gen)
        m, pv = torch.zeros(B, NAV_H), torch.zeros(B, H2)
        if prefix is not None:
            pv = torch.tensor(np.asarray(prefix[0]).reshape(B, H2) * NAV_SCALE, dtype=torch.float32)
            m = torch.tensor(np.asarray(prefix[1]), dtype=torch.float32)
        m2 = m.repeat_interleave(2, 1)
        K = None
        if model.kc:
            K = torch.zeros(B, model.kc) if cond is None else torch.tensor(np.asarray(cond), dtype=torch.float32)
        for i in range(n_steps):
            x = m2 * pv + (1 - m2) * x
            base = [x, S_, tfeat(torch.full((B, 1), i / n_steps))]
            tail = [m] if model.prefix else []
            if model.kc:
                v = model(torch.cat(base + [K] + tail, -1))
                if beta:
                    v_u = model(torch.cat(base + [torch.zeros_like(K)] + tail, -1))
                    v = v + beta * (v - v_u)
            else:
                v = model(torch.cat(base + tail, -1))
            x = x + v / n_steps
        x = m2 * pv + (1 - m2) * x
        return (x / NAV_SCALE).numpy().reshape(B, NAV_H, 2)

    def nav_rollouts(policy, starts, exec_h=8, max_steps=90, push_at=None, push_vec=None):
        """Batched closed loop: query a chunk, execute exec_h waypoints, re-plan. Optional shove at step push_at."""
        B = len(starts)
        pos = np.array(starts, dtype=float)
        paths = [[p.copy()] for p in pos]
        done, hit = np.zeros(B, bool), np.zeros(B, bool)
        k = 0
        while k < max_steps and not done.all():
            act = np.where(~done)[0]
            start = pos[act].copy()
            chunks = policy(start)
            for i in range(exec_h):
                k += 1
                for j, b in enumerate(act):
                    if done[b]:
                        continue
                    q = start[j] + chunks[j, i]
                    if push_at is not None and k == push_at:
                        q = q + push_vec[b]
                    if nav_hits(pos[b], q):
                        hit[b] = done[b] = True
                    pos[b] = q
                    paths[b].append(q.copy())
                    if np.linalg.norm(q - NAV_GOAL) < NAV_GOAL_R:
                        done[b] = True
                if push_at is not None and k == push_at:
                    break  # re-plan right after the shove
        ok = ~hit & (np.linalg.norm(pos - NAV_GOAL, axis=1) < NAV_GOAL_R)
        return [np.array(p) for p in paths], ok, hit
    return (
        MLP,
        NAV_GOAL,
        NAV_GOAL_R,
        NAV_H,
        NAV_OBS_C,
        NAV_OBS_R,
        nav_chunks,
        nav_demo,
        nav_hits,
        nav_rollouts,
        predict_mse,
        sample_flow,
        train_flow,
        train_mse,
    )


@app.cell
def _(nav_chunks, nav_demo, np, time, train_flow, train_mse):
    _rng = np.random.default_rng(0)
    bimodal_demos = [nav_demo(_rng)[0] for _i in range(300)]
    _S, _C = map(np.concatenate, zip(*[nav_chunks(_d) for _d in bimodal_demos]))
    _t0 = time.time()
    mse_policy = train_mse(_S, _C)
    flow_policy = train_flow(_S, _C, steps=3000, prefix_max=6, seed=1)
    nav_fit_seconds = time.time() - _t0
    return bimodal_demos, flow_policy, mse_policy, nav_fit_seconds


@app.cell
def _(mo):
    fm_steps = mo.ui.slider(1, 20, step=1, value=10, label="Flow integration steps", show_value=True)
    fm_steps
    return (fm_steps,)


@app.cell
def _(
    Circle,
    Figure,
    NAV_GOAL,
    NAV_GOAL_R,
    NAV_OBS_C,
    NAV_OBS_R,
    PAL,
    bimodal_demos,
    flow_policy,
    fm_steps,
    mo,
    mse_policy,
    nav_fit_seconds,
    nav_rollouts,
    np,
    predict_mse,
    sample_flow,
    torch,
):
    _starts = np.stack([np.full(32, -1.0), np.linspace(-0.2, 0.2, 32)], 1)
    _gen = torch.Generator().manual_seed(3)
    _mse = nav_rollouts(lambda s: predict_mse(mse_policy, s), _starts)
    _fm = nav_rollouts(lambda s: sample_flow(flow_policy, s, fm_steps.value, gen=_gen), _starts)
    _s0 = np.repeat([[-1.0, 0.0]], 400, 0)
    _ends = sample_flow(flow_policy, _s0, fm_steps.value, gen=_gen)[:, -1] + _s0
    _mse_end = predict_mse(mse_policy, _s0[:1])[0, -1] + _s0[0]

    def _world(ax, title):
        ax.add_patch(Circle(NAV_OBS_C, NAV_OBS_R, facecolor=PAL["faint"], edgecolor=PAL["grey"], lw=1, zorder=0))
        ax.add_patch(Circle(NAV_GOAL, NAV_GOAL_R, fill=False, edgecolor=PAL["good"], lw=1.5))
        ax.set(xlim=(-1.15, 1.2), ylim=(-0.9, 0.9), aspect="equal", title=title, xticks=[], yticks=[])

    _fig = Figure(figsize=(10, 6.0))
    _axs = _fig.subplots(2, 2)
    _world(_axs[0, 0], "300 demonstrations (40 shown)")
    for _d in bimodal_demos[:40]:
        _axs[0, 0].plot(*_d.T, color=PAL["grey"], lw=0.8, alpha=0.7)
    for _ax, (_paths, _ok, _hit), _title in [
        (_axs[0, 1], _mse, "MSE head"),
        (_axs[1, 0], _fm, f"Flow head, {fm_steps.value} steps"),
    ]:
        _world(_ax, f"{_title}: {_hit.sum()}/32 crash")
        for _p, _h in zip(_paths, _hit):
            _ax.plot(*_p.T, color=PAL["bad"] if _h else PAL["s1"], lw=1, alpha=0.8)
    _world(_axs[1, 1], "Where the first chunk ends, 400 samples")
    _axs[1, 1].scatter(*_ends.T, s=6, color=PAL["s1"], alpha=0.5, label="flow samples")
    _axs[1, 1].scatter(*_mse_end, s=90, marker="X", color=PAL["bad"], label="MSE prediction", zorder=3)
    _axs[1, 1].plot(-1.0, 0.0, "s", color=PAL["ink"], ms=5)
    _axs[1, 1].legend(loc="lower right", fontsize=8)
    mo.vstack(
        [
            _fig,
            mo.md(
                f"Closed-loop rollouts from 32 start positions; each executes 8 waypoints of a chunk, then re-plans. Red paths hit the obstacle. Both heads trained in {nav_fit_seconds:.0f} s."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(r"""
    **One Euler step returns the mean.** At $t=0$, $x_t = x_0$ is pure noise and independent of $x_1$, so the optimal field there is $v^*(x_0, 0) = \mathbb{E}[x_1 \mid s] - x_0$. A single step of size 1 lands exactly on $\mathbb{E}[x_1 \mid s]$, which is the MSE answer. The noise only starts choosing a mode from the second step on. Drag the slider to 1 and watch the bottom-right panel: the two clusters collapse toward the MSE point and crashes climb. A learned field is not exactly optimal, so a one-step head behaves like a regression head with a little noise added, and that noise still breaks some ties. Two or three steps already recover almost all of the gap here.

    That is why few-step inference needs training tricks. π0.7 samples with 5 steps and GR00T N1 with 4 (NVIDIA doesn't publish N1.7's count). DreamZero-Flash reaches one step by training with video timesteps skewed toward noise, so the action head learns to commit while the imagined video is still blurry: on table bussing, 74% task progress at one step vs 52% for naive one-step sampling, and 83% at four steps.
    """),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What this toy leaves out.** Real action experts condition on a VLM's hidden states and proprioception, not on a 2-D position, and they output joint targets or end-effector deltas plus gripper commands. Chunks are longer (50 steps in π0.7, 48 in DreamZero) and the head is a transformer with adaptive-norm timestep conditioning rather than a bare MLP; GR00T still wraps it in small per-embodiment MLP encoders and decoders.

    ## 3 · Chunks under latency: real-time chunking

    A chunk runs open-loop for a while and is then replaced. Deployed systems overlap the two: the robot keeps executing the current chunk while the next one is computed. DreamZero runs 48-step chunks at 30 Hz, so each chunk covers 1.6 s and inference has to finish well inside that window; it targets under about 200 ms.

    The catch is that inference takes `d` control steps. When chunk *n+1* lands, the robot has executed `d` more actions from chunk *n*, actions that chunk *n+1* never saw. If the new sample picked the other mode, the commanded position jumps. Three ways to handle the switch:

    - **naive:** jump to the new chunk.
    - **average:** blend old and new where both cover the same step, ACT-style temporal ensembling. Smooth, but averaging two modes is exactly what §2 showed you can't do.
    - **rtc:** generate the new chunk *conditioned on* the `d` actions that will execute during inference. π0.7 uses the training-time version (Black et al., Dec 2025): during training, feed a random-length clean prefix in and denoise only the rest, simulating delays of 0–12 steps, up to 240 ms at 50 Hz. The flow head from §2 was trained this way, so RTC costs nothing extra at inference.

    The speculative-decoding analogy: the committed prefix is like accepted tokens that the next draft must continue, except that motor commands already sent can never be rejected.
    """)
    return


@app.cell
def _(NAV_GOAL, NAV_GOAL_R, NAV_H, nav_hits, np, sample_flow, torch):
    def async_rollouts(model, starts, delay, exec_h, method, n_steps=10, max_steps=80, seed=0):
        """A query issued at step k sees the state at k and lands at k + delay; the old chunk keeps running meanwhile.
        method: 'naive' | 'average' | 'rtc'. Returns paths, success, crash, and each path's worst jump."""
        gen = torch.Generator().manual_seed(seed)
        B = len(starts)
        pos = np.array(starts, dtype=float)
        paths = [[p.copy()] for p in pos]
        hit, done = np.zeros(B, bool), np.zeros(B, bool)
        plan, plan_q = pos[:, None] + sample_flow(model, pos, n_steps, gen=gen), 0  # absolute targets for steps plan_q+1...
        prev_plan, prev_q, pending, next_q, k = None, None, None, exec_h, 0
        while k < max_steps and not done.all():
            if pending is not None and k == pending[0]:  # a chunk lands: switch to it
                prev_plan, prev_q = plan, plan_q
                plan, plan_q = pending[1], pending[2]
                pending = None
            if k == next_q:  # issue the next query
                q_state = pos.copy()
                prefix = None
                if method == "rtc" and delay > 0:  # the actions that will run while we wait
                    vals, mask = np.zeros((B, NAV_H, 2)), np.zeros((B, NAV_H))
                    vals[:, :delay] = plan[:, k - plan_q : k - plan_q + delay] - q_state[:, None]
                    mask[:, :delay] = 1
                    prefix = (vals, mask)
                rel = sample_flow(model, q_state, n_steps, prefix=prefix, gen=gen)
                pending, next_q = (k + delay, q_state[:, None] + rel, k), k + exec_h
                if delay == 0:
                    prev_plan, prev_q = plan, plan_q
                    plan, plan_q = pending[1], pending[2]
                    pending = None
            target = plan[:, k - plan_q]
            if method == "average" and prev_plan is not None and k - prev_q < NAV_H:
                target = 0.5 * (target + prev_plan[:, k - prev_q])
            k += 1
            for b in range(B):
                if done[b]:
                    continue
                if nav_hits(pos[b], target[b]):
                    hit[b] = done[b] = True
                pos[b] = target[b]
                paths[b].append(target[b].copy())
                if np.linalg.norm(pos[b] - NAV_GOAL) < NAV_GOAL_R:
                    done[b] = True
        ok = ~hit & (np.linalg.norm(pos - NAV_GOAL, axis=1) < NAV_GOAL_R)
        paths = [np.array(p) for p in paths]
        jumps = np.array([np.linalg.norm(np.diff(p, 2, axis=0), axis=1).max() if len(p) > 3 else 0.0 for p in paths])
        return paths, ok, hit, jumps
    return (async_rollouts,)


@app.cell
def _(mo):
    rtc_delay = mo.ui.slider(0, 6, step=1, value=4, label="Inference delay d (control steps)", show_value=True)
    rtc_exec = mo.ui.slider(6, 10, step=1, value=6, label="Steps executed per chunk", show_value=True)
    mo.hstack([rtc_delay, rtc_exec], justify="start", gap=1.5, wrap=True)
    return rtc_delay, rtc_exec


@app.cell
def _(
    Circle,
    Figure,
    NAV_GOAL,
    NAV_GOAL_R,
    NAV_OBS_C,
    NAV_OBS_R,
    PAL,
    async_rollouts,
    flow_policy,
    mo,
    np,
    rtc_delay,
    rtc_exec,
):
    _starts = np.stack([np.full(32, -1.0), np.linspace(-0.2, 0.2, 32)], 1)
    _res = {
        _m: async_rollouts(flow_policy, _starts, rtc_delay.value, rtc_exec.value, _m) for _m in ("naive", "average", "rtc")
    }
    _fig = Figure(figsize=(10.5, 3.3))
    for _ax, (_m, (_paths, _ok, _hit, _jumps)) in zip(_fig.subplots(1, 3), _res.items()):
        _ax.add_patch(Circle(NAV_OBS_C, NAV_OBS_R, facecolor=PAL["faint"], edgecolor=PAL["grey"], lw=1, zorder=0))
        _ax.add_patch(Circle(NAV_GOAL, NAV_GOAL_R, fill=False, edgecolor=PAL["good"], lw=1.5))
        for _p, _h in zip(_paths, _hit):
            _ax.plot(*_p.T, color=PAL["bad"] if _h else PAL["s1"], lw=0.9, alpha=0.8)
        _ax.set(xlim=(-1.15, 1.2), ylim=(-0.9, 0.9), aspect="equal", xticks=[], yticks=[], title=f"{_m}: {_hit.sum()}/32 crash")
    _rows = "\n".join(
        f"| {_m} | {_r[1].mean():.0%} | {_r[2].mean():.0%} | {_r[3].mean():.3f} | {_r[3].max():.3f} |" for _m, _r in _res.items()
    )
    _table = "| switch rule | reach goal | crash | worst jump, mean | worst jump, max |\n|---|---|---|---|---|\n" + _rows
    mo.vstack(
        [
            _fig,
            mo.md(_table),
            mo.md(
                r"*Worst jump* is the largest $\lVert p_{t+1} - 2p_t + p_{t-1} \rVert$ along a rollout. The demonstrations' own curvature puts it near 0.04, so anything well above that is a visible lurch at a chunk boundary."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    What to look for: as `d` grows, naive switching lurches harder at every boundary, averaging stays smooth but crashes more often, and RTC keeps the worst jump flat at the demonstrations' own level.

    **What this toy leaves out.** The first RTC paper (June 2025) worked at test time, inpainting the prefix with guidance during denoising; the training-time version used here and in π0.7 removes that inference cost. DreamZero smooths differently: it upsamples each chunk 2×, applies a Savitzky–Golay filter and downsamples again. Real inference delay also varies from call to call.

    ## 4 · Tokens for the backbone: FAST and knowledge insulation

    π0.7 trains two heads on one model. The VLM backbone predicts discrete **FAST** tokens with cross-entropy; the 860M action expert generates continuous chunks with flow matching, reads the backbone's activations, and sends no gradient back into it. The paper calls this knowledge insulation: the backbone keeps a stable, LLM-style objective and its web knowledge, and the action expert stays fast and continuous. In π0.7 the FAST tokens supervise the backbone during training; at runtime the actions come from the flow-matching expert.

    Why not bin each action value into 256 levels? A 50-step × 7-DoF chunk becomes 350 tokens, and consecutive tokens are so correlated that extrapolating the previous two in a straight line usually lands within a couple of bins. Most next-token predictions then carry little signal. FAST compresses first: a DCT along time for each action dimension, scale by γ, round, flatten low frequencies first, then BPE. Smooth motion puts almost all of its energy into a few low-frequency coefficients, the same reason JPEG works. If you have used FSQ, this is its hand-designed cousin: a fixed transform followed by rounding.
    """)
    return


@app.cell
def _(np):
    def fast_dct_matrix(N):
        k, n = np.arange(N)[:, None], np.arange(N)[None, :]
        D = np.cos(np.pi / N * (n + 0.5) * k) * np.sqrt(2.0 / N)
        D[0] /= np.sqrt(2.0)
        return D  # orthonormal DCT-II; the inverse is D.T

    def fast_make_chunks(n=256, T=50, dims=7, seed=0):
        """Smooth synthetic action chunks (sums of slow sinusoids plus drift), normalised to [-1, 1] by quantiles like FAST."""
        rng = np.random.default_rng(seed)
        t = np.linspace(0, 1, T)[None, :, None, None]
        f = rng.uniform(0.2, 2.0, (n, 1, dims, 3))
        ph = rng.uniform(0, 2 * np.pi, (n, 1, dims, 3))
        amp = rng.uniform(0.1, 0.5, (n, 1, dims, 3))
        x = (amp * np.sin(2 * np.pi * f * t + ph)).sum(-1) + rng.uniform(-0.5, 0.5, (n, 1, dims)) * t[..., 0]
        x = x + rng.normal(0, 0.004, x.shape)
        lo, hi = np.quantile(x, 0.01), np.quantile(x, 0.99)
        return np.clip(2 * (x - lo) / (hi - lo) - 1, -1, 1)

    FAST_D = fast_dct_matrix(50)
    fast_chunks = fast_make_chunks()
    return FAST_D, fast_chunks


@app.cell
def _(mo):
    fast_scale = mo.ui.slider(1, 60, step=1, value=10, label="Quantization scale γ", show_value=True)
    fast_scale
    return (fast_scale,)


@app.cell
def _(FAST_D, Figure, PAL, fast_chunks, fast_scale, mo, np):
    _X = fast_chunks
    _C = np.einsum("kt,ntd->nkd", FAST_D, _X)  # DCT along time, one transform per action dimension
    _q = np.round(_C * fast_scale.value)  # scale and round: the only lossy step
    _rec = np.einsum("kt,nkd->ntd", FAST_D, _q / fast_scale.value)
    _nz = (_q != 0).sum((1, 2)).mean()
    _rmse = np.sqrt(((_rec - _X) ** 2).mean())
    _bins = np.clip(((_X + 1) / 2 * 256).astype(int), 0, 255)  # naive: one 256-way token per step and dimension
    _guess = 2 * _bins[:, 1:-1] - _bins[:, :-2]  # continue the previous two tokens in a straight line
    _near = (np.abs(_bins[:, 2:] - _guess) <= 2).mean()
    _curve = []
    for _g in [1, 2, 3, 5, 8, 12, 20, 30, 45, 60]:
        _qq = np.round(_C * _g)
        _curve.append(
            ((_qq != 0).sum((1, 2)).mean(), np.sqrt(((np.einsum("kt,nkd->ntd", FAST_D, _qq / _g) - _X) ** 2).mean()))
        )
    _curve = np.array(_curve)

    _fig = Figure(figsize=(10.5, 3.2))
    _a, _b, _c = _fig.subplots(1, 3)
    _a.plot(_X[0, :, 0], color=PAL["ink"], lw=1.6, label="original")
    _a.plot(_rec[0, :, 0], color=PAL["s1"], lw=1.4, ls="--", label=f"decoded, γ = {fast_scale.value}")
    _a.set(title="One action dimension of one chunk", xlabel="time step")
    _a.legend(fontsize=8)
    _b.imshow(np.log10(np.abs(_C).mean(0).T + 1e-4), aspect="auto", cmap="magma")
    _b.set(title="log mean |DCT coefficient|", xlabel="frequency", ylabel="action dimension")
    _b.grid(False)
    _c.plot(_curve[:, 0], _curve[:, 1], "o-", color=PAL["s2"], ms=4)
    _c.scatter([_nz], [_rmse], s=80, color=PAL["s1"], zorder=3)
    _c.axvline(350, color=PAL["grey"], ls=":")
    _c.text(330, _curve[:, 1].max(), "naive: 350 tokens ", fontsize=8, va="top", ha="right")
    _c.set(xscale="log", yscale="log", title="Error vs nonzero coefficients", xlabel="nonzero coefficients per chunk", ylabel="RMSE")
    _c.set_xticks([30, 50, 100, 200, 350], labels=["30", "50", "100", "200", "350"])
    _c.minorticks_off()
    mo.vstack(
        [
            _fig,
            mo.md(
                f"At γ = {fast_scale.value}: **{_nz:.0f}** nonzero coefficients per 50×7 chunk against 350 naive tokens, at RMSE **{_rmse:.4f}** on the [−1, 1] scale. With naive 256-bin tokens, a straight-line guess from the previous two tokens lands within ±2 bins of the next one **{_near:.0%}** of the time; that redundancy is what makes next-token prediction on raw bins a weak training signal."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What this toy leaves out.** Real FAST runs BPE over the integer sequence afterwards, so long runs of zeros cost almost nothing; here we only count nonzero coefficients. Synthetic sinusoids are also smoother than teleoperation data.

    ## 5 · Steering with context: metadata and CFG

    π0.7's central move is to stop filtering data and start labelling it. Each training episode carries metadata (speed as the episode length binned into 500-step buckets, quality from 1 to 5, and mistake flags) plus optional subtask text and subgoal images. At test time you ask for what you want: quality 5, no mistakes, speed at the 15th-percentile episode length for that task. Each part of the prompt is dropped during training (all metadata 15% of the time), which also yields an unconditional model for classifier-free guidance:

    $$v = v_c + \beta\,(v_c - v_u), \qquad \beta \in \{1.3,\ 1.7,\ 2.2\} \text{ in the paper.}$$

    The paper's scaling result: on laundry, training on the top 30%, 50%, 80% and 100% of the data ranked by quality made the model *without* metadata worse as data was added, while the model *with* metadata kept improving.

    Toy: 360 demonstrations from operators of quality 1–5. Lower quality means a shakier hand, more wide swings, slower runs. We train on the best 30%, 60% and 100% of the data, with and without metadata (six flow heads, roughly 20–60 s on CPU), then prompt for quality 5 at the fastest speed bin. Each evaluation run includes a shove at step 14.
    """)
    return


@app.cell
def _(nav_chunks, nav_demo, np):
    META_SPEEDS = np.array([0.035, 0.05, 0.065])  # waypoint spacing of slow / medium / fast operators

    def meta_vec(quality, speed_bin):
        v = np.zeros(9, np.float32)
        v[quality - 1] = 1
        v[5 + speed_bin] = 1
        v[8] = 1  # "metadata present"; all zeros is the dropped-out, unconditional prompt
        return v

    def make_mixed_demos(n=360, seed=3):
        rng = np.random.default_rng(seed)
        out = []
        for _demo in range(n):
            q = int(rng.integers(1, 6))
            p_fast = 0.05 + 0.15 * (q - 1)
            p_slow = max(0.0, 0.55 - 0.13 * (q - 1))
            sp = rng.choice(3, p=[p_slow, 1 - p_slow - p_fast, p_fast])  # better operators tend to be faster
            detour = 0.0
            if q <= 2 and rng.uniform() < 0.5:
                detour = rng.uniform(0.15, 0.3)
            elif q == 3 and rng.uniform() < 0.15:
                detour = rng.uniform(0.1, 0.2)
            path, _side = nav_demo(rng, step=META_SPEEDS[sp], wobble=0.006 * (5 - q), detour=detour)
            out.append({"path": path, "q": q, "n": len(path)})
        edges = np.quantile([d["n"] for d in out], [1 / 3, 2 / 3])
        for d in out:
            d["speed_bin"] = int(np.digitize(d["n"], edges))  # 0 = the fastest third
        return out

    def meta_dataset(demos, idx, with_meta):
        S, C, K = [], [], []
        for i in idx:
            s_, c_ = nav_chunks(demos[i]["path"])
            S.append(s_)
            C.append(c_)
            K.append(np.repeat(meta_vec(demos[i]["q"], demos[i]["speed_bin"])[None], len(s_), 0))
        return np.concatenate(S), np.concatenate(C), (np.concatenate(K) if with_meta else None)

    mixed_demos = make_mixed_demos()
    return meta_dataset, meta_vec, mixed_demos


@app.cell
def _(meta_dataset, mixed_demos, time, train_flow):
    _order = sorted(range(len(mixed_demos)), key=lambda i: (-mixed_demos[i]["q"], mixed_demos[i]["n"]))
    meta_models = {}
    _t0 = time.time()
    for _frac in (0.3, 0.6, 1.0):
        _idx = _order[: int(_frac * len(mixed_demos))]
        for _with in (False, True):
            _S, _C, _K = meta_dataset(mixed_demos, _idx, _with)
            meta_models[(_frac, _with)] = train_flow(_S, _C, steps=1500, cond=_K, p_drop=0.15, seed=2)
    meta_fit_seconds = time.time() - _t0
    return meta_fit_seconds, meta_models


@app.cell
def _(meta_models, meta_vec, nav_rollouts, np, sample_flow, torch):
    def meta_eval(model, use_meta, beta=0.0, shove=True, n=40, seed=11):
        """Prompt for quality 5 at the fastest speed bin; closed-loop rollouts with an optional shove at step 14."""
        rng = np.random.default_rng(seed)
        starts = np.stack([np.full(n, -1.0), rng.uniform(-0.2, 0.2, n)], 1)
        ang = rng.uniform(0, 2 * np.pi, n)
        push = 0.18 * np.stack([np.cos(ang), np.sin(ang)], 1)
        prompt = np.repeat(meta_vec(5, 0)[None], n, 0)
        gen = torch.Generator().manual_seed(5)

        def policy(s):
            return sample_flow(model, s, 10, cond=prompt[: len(s)] if use_meta else None, beta=beta, gen=gen)

        paths, ok, hit = nav_rollouts(policy, starts, exec_h=8, max_steps=140, push_at=14 if shove else None, push_vec=push)
        steps = np.array([len(p) - 1 for p in paths])
        return {"paths": paths, "ok": ok, "hit": hit, "success": ok.mean(), "steps": steps[ok].mean() if ok.any() else float("nan")}

    meta_scores = {key: meta_eval(model, key[1]) for key, model in meta_models.items()}
    return meta_eval, meta_scores


@app.cell
def _(Figure, PAL, meta_fit_seconds, meta_scores, mo):
    _fracs = [0.3, 0.6, 1.0]
    _fig = Figure(figsize=(9.5, 3.1))
    _a, _b = _fig.subplots(1, 2)
    for _with, _color, _label in [(False, PAL["grey"], "no metadata"), (True, PAL["s1"], "metadata; prompt: quality 5, fast")]:
        _a.plot([f * 100 for f in _fracs], [meta_scores[(f, _with)]["steps"] for f in _fracs], "o-", color=_color, label=_label)
        _b.plot([f * 100 for f in _fracs], [100 * meta_scores[(f, _with)]["success"] for f in _fracs], "o-", color=_color, label=_label)
    _a.set(xlabel="training data (% of demos, best first)", ylabel="steps to reach the goal", title="Speed (lower is better)")
    _b.set(xlabel="training data (% of demos, best first)", ylabel="success with a shove (%)", title="Success", ylim=(0, 105))
    _a.legend(fontsize=8)
    _lines = "\n".join(
        f"| {int(f * 100)}% | {meta_scores[(f, False)]['steps']:.1f} | {meta_scores[(f, True)]['steps']:.1f} | {meta_scores[(f, False)]['success']:.0%} | {meta_scores[(f, True)]['success']:.0%} |"
        for f in _fracs
    )
    mo.vstack(
        [
            _fig,
            mo.md(
                "| training data | steps, no metadata | steps, metadata | success, no metadata | success, metadata |\n|---|---|---|---|---|\n"
                + _lines
                + f"\n\nSix flow heads trained in {meta_fit_seconds:.0f} s."
            ),
        ]
    )
    return


@app.cell
def _(mo):
    meta_beta = mo.ui.slider(0.0, 3.0, step=0.1, value=0.0, label="CFG weight β on the metadata", show_value=True)
    meta_beta
    return (meta_beta,)


@app.cell
def _(
    Circle,
    Figure,
    NAV_GOAL,
    NAV_GOAL_R,
    NAV_OBS_C,
    NAV_OBS_R,
    PAL,
    meta_beta,
    meta_eval,
    meta_models,
    mo,
    np,
):
    _r = meta_eval(meta_models[(1.0, True)], True, beta=meta_beta.value, shove=False)
    _turn = np.mean(
        [
            np.linalg.norm(np.diff(p, 2, axis=0), axis=1).mean() / max(1e-9, np.linalg.norm(np.diff(p, axis=0), axis=1).mean())
            for p in _r["paths"]
        ]
    )
    _fig = Figure(figsize=(5.4, 3.0))
    _ax = _fig.subplots()
    _ax.add_patch(Circle(NAV_OBS_C, NAV_OBS_R, facecolor=PAL["faint"], edgecolor=PAL["grey"], lw=1, zorder=0))
    _ax.add_patch(Circle(NAV_GOAL, NAV_GOAL_R, fill=False, edgecolor=PAL["good"], lw=1.5))
    for _p, _h in zip(_r["paths"], _r["hit"]):
        _ax.plot(*_p.T, ".-", ms=2.5, lw=0.8, color=PAL["bad"] if _h else PAL["s1"], alpha=0.75)
    _ax.set(xlim=(-1.15, 1.2), ylim=(-0.9, 0.9), aspect="equal", xticks=[], yticks=[], title=f"All data + metadata, β = {meta_beta.value:.1f}")
    mo.hstack(
        [
            _fig,
            mo.md(
                f"""
    | β = {meta_beta.value:.1f} | |
    |---|---|
    | success | {_r['success']:.0%} |
    | steps to goal | {_r['steps']:.1f} |
    | turning per unit step | {_turn:.3f} |

    The dots are waypoints, so their spacing is speed.
    """
            ),
        ],
        justify="start",
        gap=2,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    What to look for: without metadata, every extra slice of worse data makes the robot slower, because it imitates the average operator. With metadata, the same data does no harm: prompting for quality 5 recovers the best operators from the mixture. CFG then pushes past the data. At the paper's smallest weight, β = 1.3, the robot needs about 28 steps where the fastest quality-5 demonstration took 34; raise β further and the turning cost climbs while the speed gain flattens.

    **What this toy leaves out.** π0.7 reports the extra data actively *helping*, through more varied states and better robustness; this 2-D world is too easy to show that part. Real metadata also marks mistakes per segment, and it comes from human annotation, which does not scale for free. Argon's progress curves and hindsight labels are one bid to automate it.

    ## 6 · World action models: learning from video you can't act on

    DreamZero's claim, which Dream-1 inherits and which NVIDIA says its previewed GR00T 2 builds on, is that System 1 should predict the future as well as the action. Its own decomposition makes the argument:

    $$\underbrace{\pi(o_{l:l+H}, a_{l:l+H} \mid o_{0:l}, c, q_l)}_{\text{DreamZero}} = \underbrace{\pi(o_{l:l+H} \mid o_{0:l}, c, q_l)}_{\text{video prediction}}\;\underbrace{\pi(a_{l:l+H} \mid o_{0:l+H}, q_l)}_{\text{inverse dynamics}}$$

    The video term never needs actions, so human and other-robot videos train it directly. The inverse-dynamics term never needs task labels: it only maps "this frame, then that frame" to a motor command, which any robot motion teaches, undirected play included. A VLA has to learn the product of the two from task-labelled action data. DreamZero trains both terms as one 14B network (Wan2.1-I2V-14B, flow matching with a shared timestep for video and action); here they are two small MLPs on 16×16 frames.

    The world: a robot (a blob) and eight landmarks. Instruction *k* means "go to landmark *k*". Robots have action-labelled demonstrations for landmarks 0, 2, 4, 6. For 1, 3, 5, 7 there is only action-free "human video", plus a few robot demos that you choose below. Four System 1s:

    - **VLA:** (frame, instruction) → command, trained on robot demos.
    - **VLA + latent alignment:** GEAR's FLARE in miniature, the objective GR00T N1.5 adopted. The same VLA gets a second head that predicts a frozen embedding of the next frame; that loss also runs on the human videos. It is a JEPA-style auxiliary loss on a policy.
    - **WAM, inverse dynamics from task demos:** a video model (frame, instruction) → next frame trained on all videos, and an inverse-dynamics model (frame, next frame) → command trained on the robot demos.
    - **WAM, inverse dynamics from task demos + play:** the same, with 2,000 transitions of undirected robot play added to the inverse-dynamics data. DreamZero's ablation found diverse data beat repetitive data at equal hours (50% vs 33% task progress) and blamed the inverse-dynamics part.

    Training all four takes about 10–30 s and reruns when you press the button.
    """)
    return


@app.cell
def _(np, torch):
    W_G = 16
    _axis = np.linspace(-1 + 1 / W_G, 1 - 1 / W_G, W_G)
    _gx, _gy = np.meshgrid(_axis, _axis)
    W_PIX = np.stack([_gx.ravel(), _gy.ravel()], 1)  # pixel centres, (256, 2)
    W_LM = 0.72 * np.stack([np.cos(np.arange(8) * np.pi / 4), np.sin(np.arange(8) * np.pi / 4)], 1)
    W_SEEN, W_UNSEEN = [0, 2, 4, 6], [1, 3, 5, 7]

    def w_render(p):
        """Positions (N, 2) -> 16x16 frames (N, 256): a Gaussian blob where the robot is."""
        d2 = ((p[:, None, :] - W_PIX[None]) ** 2).sum(-1)
        return np.exp(-d2 / (2 * 0.13**2)).astype(np.float32)

    def w_onehot(k):
        z = np.zeros((len(k), 8), np.float32)
        z[np.arange(len(k)), k] = 1
        return z

    def w_rot(deg):
        a = np.deg2rad(deg)
        return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

    def w_step(p, a, M):
        """Robot dynamics: command a moves the robot by M a. M is the identity for the robot we train on."""
        return np.clip(p + a @ M.T, -0.95, 0.95)

    def w_demos(ks, n_per, rng, M=np.eye(2), T=12):
        """Expert episodes 'go to landmark k'. Returns a list of (k, positions (T+1, 2), commands (T, 2))."""
        Minv = np.linalg.inv(M)
        out = []
        for k in ks:
            for _ep in range(n_per):
                r, th = 0.35 * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
                p = np.array([r * np.cos(th), r * np.sin(th)])
                P, A = [p], []
                for _t in range(T):
                    d = W_LM[k] - p
                    n = np.linalg.norm(d)
                    disp = 0.09 * d / max(n, 1e-6) * min(1.0, n / 0.09) + rng.normal(0, 0.012, 2)
                    a = Minv @ disp
                    p = w_step(p[None], a[None], M)[0]
                    P.append(p)
                    A.append(a)
                out.append((k, np.array(P), np.array(A)))
        return out

    def w_play(n, rng, M=np.eye(2)):
        """Undirected play: random commands from random positions, with no task label."""
        p0 = rng.uniform(-0.8, 0.8, (n, 2))
        th, sp = rng.uniform(0, 2 * np.pi, n), rng.uniform(0.0, 0.11, n)
        a = np.stack([np.cos(th), np.sin(th)], 1) * sp[:, None]
        return p0, a, w_step(p0, a, M)

    def w_fit(model, X, Y, steps, bs=128, lr=2e-3, seed=0):
        torch.manual_seed(seed)
        X, Y = torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(Y, dtype=torch.float32)
        opt = torch.optim.Adam(model.parameters(), lr)
        for _step in range(steps):
            idx = torch.randint(0, len(X), (min(bs, len(X)),))
            loss = ((model(X[idx]) - Y[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        return model

    def w_rollouts(policy, k, n=24, M=np.eye(2), T=16, seed=123):
        """Closed loop from n random starts towards landmark k. Returns trajectories (n, T+1, 2) and success flags."""
        rng = np.random.default_rng(seed + k)
        r, th = 0.35 * np.sqrt(rng.uniform(size=n)), rng.uniform(0, 2 * np.pi, n)
        p = np.stack([r * np.cos(th), r * np.sin(th)], 1)
        traj, done = [p.copy()], np.zeros(n, bool)
        for _t in range(T):
            a = policy(p, np.full(n, k))
            p = np.where(done[:, None], p, w_step(p, a, M))
            done |= np.linalg.norm(p - W_LM[k], axis=1) < 0.12
            traj.append(p.copy())
        return np.stack(traj, 1), done

    def w_success(policy, ks, M=np.eye(2)):
        return float(np.mean([w_rollouts(policy, k, M=M)[1].mean() for k in ks]))
    return (
        W_LM,
        W_SEEN,
        W_UNSEEN,
        w_demos,
        w_fit,
        w_onehot,
        w_play,
        w_render,
        w_rollouts,
        w_rot,
        w_success,
    )


@app.cell
def _(mo):
    wam_form = mo.ui.dropdown(
        options=["0", "1", "3"], value="1", label="Action-labelled robot demos per video-only instruction"
    ).form(submit_button_label="Retrain the four models")
    wam_form
    return (wam_form,)


@app.cell
def _(MLP, W_SEEN, W_UNSEEN, nn, np, time, torch, w_demos, w_fit, w_onehot, w_play, w_render, wam_form):
    _n_few = int(wam_form.value) if wam_form.value is not None else 1
    _rng = np.random.default_rng(0)
    _t0 = time.time()
    _robot = w_demos(W_SEEN, 50, _rng) + (w_demos(W_UNSEEN, _n_few, _rng) if _n_few else [])
    _video = w_demos(W_UNSEEN, 50, _rng)  # "human video": we keep the frames and throw the commands away
    _Xa = np.concatenate([np.concatenate([w_render(P[:-1]), w_onehot(np.full(len(A), k))], 1) for k, P, A in _robot])
    _Ya = np.concatenate([A for k, P, A in _robot])
    _Xv = np.concatenate(
        [np.concatenate([w_render(P[:-1]), w_onehot(np.full(len(A), k))], 1) for k, P, A in _robot + _video]
    )
    _Yv = np.concatenate([w_render(P[1:]) for k, P, A in _robot + _video])
    _Xi = np.concatenate([np.concatenate([w_render(P[:-1]), w_render(P[1:])], 1) for k, P, A in _robot])
    _p0, _ap, _p1 = w_play(2000, _rng)
    _Xp = np.concatenate([w_render(_p0), w_render(_p1)], 1)

    _vla = w_fit(MLP(264, 2, 128, 2), _Xa, _Ya, 800)
    _wm = w_fit(MLP(264, 256, 256, 2), _Xv, _Yv, 1200)
    _idm_task = w_fit(MLP(512, 2, 128, 2), _Xi, _Ya, 800)
    _idm_play = w_fit(MLP(512, 2, 128, 2), np.concatenate([_Xi, _Xp]), np.concatenate([_Ya, _ap]), 800)

    class FlareStylePolicy(nn.Module):
        """A VLA with a second head that predicts a frozen embedding of the next frame."""

        def __init__(self):
            super().__init__()
            self.trunk = nn.Sequential(nn.Linear(264, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU())
            self.act = nn.Linear(256, 2)
            self.future = nn.Linear(256, 32)

        def forward(self, x):
            h = self.trunk(x)
            return self.act(h), self.future(h)

    torch.manual_seed(0)
    _encoder = torch.randn(256, 32, generator=torch.Generator().manual_seed(1)) / 16  # frozen stand-in for a pretrained image encoder
    _flare = FlareStylePolicy()
    _opt = torch.optim.Adam(_flare.parameters(), 2e-3)
    _XA, _YA = torch.tensor(_Xa), torch.tensor(_Ya, dtype=torch.float32)
    _XV, _ZV = torch.tensor(_Xv), torch.tensor(_Yv) @ _encoder
    for _step in range(1200):
        _ia, _iv = torch.randint(0, len(_XA), (128,)), torch.randint(0, len(_XV), (128,))
        _loss = ((_flare(_XA[_ia])[0] - _YA[_ia]) ** 2).mean() + ((_flare(_XV[_iv])[1] - _ZV[_iv]) ** 2).mean()
        _opt.zero_grad()
        _loss.backward()
        _opt.step()

    wam_models = {
        "vla": _vla,
        "wm": _wm,
        "idm_task": _idm_task,
        "idm_play": _idm_play,
        "flare": _flare,
        "n_few": _n_few,
        "seconds": time.time() - _t0,
    }
    return (wam_models,)


@app.cell
def _(np, torch, w_onehot, w_render):
    def wam_policy_set(models):
        """Closed-loop policies (positions, instructions) -> commands. All of them only ever see rendered frames."""

        def frames(p, k):
            return torch.tensor(np.concatenate([w_render(p), w_onehot(k)], 1))

        @torch.no_grad()
        def vla(p, k):
            return models["vla"](frames(p, k)).numpy()

        @torch.no_grad()
        def flare(p, k):
            return models["flare"](frames(p, k))[0].numpy()

        def wam(idm_key):
            @torch.no_grad()
            def policy(p, k):
                now = torch.tensor(w_render(p))
                imagined = models["wm"](frames(p, k))  # imagine the next frame...
                return models[idm_key](torch.cat([now, imagined], 1)).numpy()  # ...and read the command off it

            return policy

        return {
            "VLA": vla,
            "VLA + latent alignment": flare,
            "WAM, IDM from task demos": wam("idm_task"),
            "WAM, IDM from task demos + play": wam("idm_play"),
        }
    return (wam_policy_set,)


@app.cell
def _(Figure, PAL, W_LM, W_SEEN, W_UNSEEN, mo, np, torch, w_onehot, w_render, w_rollouts, w_success, wam_models, wam_policy_set):
    _pols = wam_policy_set(wam_models)
    _scores = {_name: (w_success(_pol, W_SEEN), w_success(_pol, W_UNSEEN)) for _name, _pol in _pols.items()}
    _fig = Figure(figsize=(11, 3.1))
    for _ax, (_name, _pol) in zip(_fig.subplots(1, 4), _pols.items()):
        for _k in W_UNSEEN:
            _traj, _done = w_rollouts(_pol, _k, n=6)
            for _tr, _d in zip(_traj, _done):
                _ax.plot(*_tr.T, color=PAL["wm"] if _d else PAL["bad"], lw=0.9, alpha=0.85)
        _ax.scatter(*W_LM[W_SEEN].T, s=36, color=PAL["ink"], zorder=3)
        _ax.scatter(*W_LM[W_UNSEEN].T, s=36, facecolor="white", edgecolor=PAL["ink"], zorder=3)
        _ax.set(xlim=(-1, 1), ylim=(-1, 1), aspect="equal", xticks=[], yticks=[])
        _ax.set_title(_name, fontsize=8.5)

    # the world model's imagination for one video-only instruction: open loop vs the real closed-loop frames
    _k = W_UNSEEN[1]
    _real_traj, _ = w_rollouts(_pols["WAM, IDM from task demos + play"], _k, n=1, T=16, seed=999)
    _o = torch.tensor(w_render(_real_traj[0, :1]))
    _dream = [_o.numpy()[0]]
    with torch.no_grad():
        for _t in range(16):
            _o = wam_models["wm"](torch.cat([_o, torch.tensor(w_onehot(np.array([_k])))], 1))
            _dream.append(_o.numpy()[0])
    _strip = Figure(figsize=(9.5, 2.7))
    _sa = _strip.subplots(2, 9)
    for _j, _step in enumerate(range(0, 17, 2)):
        _sa[0, _j].imshow(w_render(_real_traj[0, _step : _step + 1])[0].reshape(16, 16), origin="lower", cmap="Greys", vmin=0, vmax=1)
        _sa[1, _j].imshow(_dream[_step].reshape(16, 16), origin="lower", cmap="Greys", vmin=0, vmax=1)
        _sa[0, _j].set_title(f"t = {_step}", fontsize=8, fontweight="normal")
        for _r in (0, 1):
            _sa[_r, _j].axis("off")
    _sa[0, 0].text(-0.15, 0.5, "real", transform=_sa[0, 0].transAxes, rotation=90, va="center", ha="right", fontsize=8)
    _sa[1, 0].text(-0.15, 0.5, "imagined", transform=_sa[1, 0].transAxes, rotation=90, va="center", ha="right", fontsize=8)

    _table = "| System 1 | landmarks with robot demos | video-only landmarks |\n|---|---|---|\n" + "\n".join(
        f"| {_name} | {_s:.0%} | {_u:.0%} |" for _name, (_s, _u) in _scores.items()
    )
    mo.vstack(
        [
            mo.md(
                f"Trained with **{wam_models['n_few']}** robot demo(s) per video-only landmark, in {wam_models['seconds']:.0f} s. Paths below head for the video-only landmarks (hollow); teal arrives, red doesn't."
            ),
            _fig,
            mo.md(_table),
            mo.md(
                f"**The world model's imagination**, landmark {_k}. Top row: frames the robot actually saw while the WAM drove it. Bottom row: the video model fed its own predictions, open loop, for 16 steps. Ghosts appear within a few steps and, a few steps later, the frame dissolves: each prediction is a slightly off-distribution input to the next. That is why DreamZero swaps real camera frames back into its KV cache after every chunk."
            ),
            _strip,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **A new body.** Now move the trained System 1 to a robot whose actuation is rotated 40° and 30% weaker: the same command moves it somewhere else. The video of a blob moving toward a landmark looks the same for either robot, so only the inverse-dynamics part needs to change. Two ways to adapt:

    - **Refit inverse dynamics on play data.** DreamZero adapted to a new robot, YAM, from 30 minutes of play: 55 trajectories with no task structure. Here the play is random commands on the new robot.
    - **Predict end-effector motion and let the robot's IK do the rest.** Dream-1's route: it outputs gripper poses, and every robot's own inverse-kinematics solver turns them into joint commands. With a correct solver nothing needs retraining. Dream-1 notes that performance then depends on the quality of each robot's IK solver, which the error slider simulates.
    """)
    return


@app.cell
def _(mo):
    adapt_n = mo.ui.slider(0, 400, step=20, value=100, label="Play transitions on the new robot", show_value=True)
    adapt_ik = mo.ui.slider(0, 80, step=5, value=0, label="IK error (degrees)", show_value=True)
    mo.hstack([adapt_n, adapt_ik], justify="start", gap=1.5, wrap=True)
    return adapt_ik, adapt_n


@app.cell
def _(W_SEEN, W_UNSEEN, adapt_ik, adapt_n, copy, mo, np, w_fit, w_play, w_render, w_rot, w_success, wam_models, wam_policy_set):
    _M2 = 0.7 * w_rot(40)  # the new robot
    _all = W_SEEN + W_UNSEEN
    _pols = wam_policy_set(wam_models)
    _wam_name = "WAM, IDM from task demos + play"
    _vla0 = w_success(_pols["VLA"], _all, M=_M2)
    _wam0 = w_success(_pols[_wam_name], _all, M=_M2)
    if adapt_n.value > 0:
        _p0, _a, _p1 = w_play(adapt_n.value, np.random.default_rng(7), _M2)
        _idm = w_fit(copy.deepcopy(wam_models["idm_play"]), np.concatenate([w_render(_p0), w_render(_p1)], 1), _a, steps=300, bs=64, lr=1e-3)
        _refit = w_success(wam_policy_set({**wam_models, "idm_play": _idm})[_wam_name], _all, M=_M2)
    else:
        _refit = _wam0
    _ik = np.linalg.inv(_M2) @ w_rot(adapt_ik.value)  # the robot's IK, wrong by adapt_ik degrees
    _eef_source = _pols[_wam_name]  # on the training robot, its output is the end-effector displacement
    _eef = w_success(lambda p, k: _eef_source(p, k) @ _ik.T, _all, M=_M2)
    mo.md(
        f"""
    | on the new robot | success, all 8 landmarks |
    |---|---|
    | VLA, no adaptation | {_vla0:.0%} |
    | WAM, no adaptation | {_wam0:.0%} |
    | WAM, inverse dynamics refit on {adapt_n.value} play transitions | {_refit:.0%} |
    | WAM output as end-effector motion + robot IK ({adapt_ik.value}° error) | {_eef:.0%} |
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    What to look for: with zero robot demos for a landmark the VLA scores nothing there, since it has never seen that instruction with an action attached. The world action model gets there from video alone, provided its inverse-dynamics model saw varied motion; trained only on task demos it covers a few directions. Latent alignment helps a little once a demo or two exist, which matches the FLARE paper's setting, but on its own it never routes the action through the predicted future. That gap is the question the Fast-WAM paper (arXiv 2603.16666) asks: does a world action model need to imagine at test time, or only during training?

    **What this toy leaves out.** Predicting a blob is trivial; predicting a kitchen is the whole problem, and DreamZero reports that most of its failures are video errors rather than action errors. The toy's "human video" also shows the same body as the robot, while real human-to-robot transfer must bridge an appearance gap (DreamZero: 38% → 54% task progress on unseen tasks from 12 minutes of human video). The frozen random projection stands in for FLARE's frozen pretrained encoder; replacing it with an EMA target encoder turns the auxiliary loss into the JEPA setup you have been building.

    ## 7 · Physical prompting: in-context imitation

    GEN-1.5's headline: put 3–12 seconds of a demonstration (sensor data plus actions, a "physical prompt") into the model's 30-second context and the pretrained model does the task. That gave 59% average success over 10 short tasks with zero gradient steps, while 10 gradient steps on 5 minutes of data gave 83%. Generalist says it did not train for this: no meta-learning objective and no special packing. Its hypotheses come from the LLM in-context-learning literature, such as bursty, Zipf-like data.

    Toy: a two-layer transformer trained on an endless family of tasks. Each task is a hidden goal plus a hidden curvature of approach. The context holds one 8-step demonstration of (state, action) pairs and the query is the current state. Since every training sequence comes from a fresh random task, the only way to lower the loss is to read the task off the demonstration. That is closer to explicit meta-learning than GEN-1.5's setup, so this shows the mechanism rather than the emergence. Training takes about 10–30 s.
    """)
    return


@app.cell
def _(math, nn, torch):
    ICL_T = 8  # demonstration steps in the context window
    ICL_SPD = 0.11  # expert step size

    def icl_expert(p, g, c):
        """Expert for task (goal g, curvature c): head for g, rotated by up to 60 degrees while still far away."""
        d = g - p
        n = torch.linalg.norm(d, dim=-1, keepdim=True).clamp_min(1e-6)
        u = d / n
        ang = (c * math.pi / 3)[..., None] * torch.clamp(n / 0.6, max=1.0)
        ca, sa = torch.cos(ang), torch.sin(ang)
        rot = torch.cat([ca * u[..., :1] - sa * u[..., 1:], sa * u[..., :1] + ca * u[..., 1:]], -1)
        return ICL_SPD * rot * torch.clamp(n / ICL_SPD, max=1.0)

    def icl_tasks(B, gen=None):
        r = 0.75 * torch.sqrt(torch.rand(B, generator=gen))
        th = 2 * math.pi * torch.rand(B, generator=gen)
        return torch.stack([r * torch.cos(th), r * torch.sin(th)], -1), 2 * torch.rand(B, generator=gen) - 1

    def icl_starts(B, gen=None):
        r = 0.8 * torch.sqrt(torch.rand(B, generator=gen))
        th = 2 * math.pi * torch.rand(B, generator=gen)
        return torch.stack([r * torch.cos(th), r * torch.sin(th)], -1)

    def icl_demo(g, c, gen=None):
        p, P, A = icl_starts(len(g), gen), [], []
        for _t in range(ICL_T):
            a = icl_expert(p, g, c)
            P.append(p)
            A.append(a)
            p = p + a
        return torch.stack(P, 1), torch.stack(A, 1)

    class ICLPolicy(nn.Module):
        """Tokens: ICL_T demonstration steps (state, action) plus one query state. Output: the action at the query."""

        def __init__(self, d=48, layers=2):
            super().__init__()
            self.inp = nn.Linear(6, d)
            self.pos = nn.Parameter(torch.randn(ICL_T + 1, d) * 0.02)
            layer = nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.out = nn.Linear(d, 2)

        def forward(self, demoP, demoA, q, has_ctx):
            B = q.shape[0]
            h = has_ctx[:, None, None]
            demo = torch.cat([demoP * h, demoA * h / ICL_SPD, h.expand(B, ICL_T, 1), torch.zeros(B, ICL_T, 1)], -1)
            query = torch.cat([q, torch.zeros(B, 3), torch.ones(B, 1)], -1)[:, None]
            x = self.inp(torch.cat([demo, query], 1)) + self.pos
            return self.out(self.enc(x)[:, -1]) * ICL_SPD

    def icl_batch(B, p_empty=0.2, gen=None):
        g, c = icl_tasks(B, gen)
        dP, dA = icl_demo(g, c, gen)
        P2, _A2 = icl_demo(g, c, gen)  # a second run of the same task supplies the query states
        q = P2[torch.arange(B), torch.randint(0, ICL_T, (B,), generator=gen)] + 0.05 * torch.randn(B, 2, generator=gen)
        has = (torch.rand(B, generator=gen) > p_empty).float()
        return dP, dA, q, has, icl_expert(q, g, c)

    @torch.no_grad()
    def icl_rollout(m, g, c, dP, dA, has, n=16, T=16, gen=None):
        p = icl_starts(n, gen)
        traj, done = [p], torch.zeros(n, dtype=torch.bool)
        for _t in range(T):
            a = m(dP.expand(n, -1, -1), dA.expand(n, -1, -1), p, torch.full((n,), float(has)))
            p = torch.where(done[:, None], p, p + a)
            done |= torch.linalg.norm(p - g, dim=-1) < 0.1
            traj.append(p)
        return torch.stack(traj, 1).numpy(), done.float().mean().item()
    return ICLPolicy, ICL_SPD, ICL_T, icl_batch, icl_demo, icl_rollout, icl_tasks


@app.cell
def _(ICLPolicy, ICL_SPD, icl_batch, time, torch):
    torch.manual_seed(0)
    _t0 = time.time()
    icl_model = ICLPolicy()
    _opt = torch.optim.AdamW(icl_model.parameters(), 1e-3)
    _sched = torch.optim.lr_scheduler.OneCycleLR(_opt, 1e-3, total_steps=1200, pct_start=0.1)
    for _step in range(1200):
        _dP, _dA, _q, _has, _y = icl_batch(128)
        _loss = ((icl_model(_dP, _dA, _q, _has) - _y) ** 2).mean() / ICL_SPD**2
        _opt.zero_grad()
        _loss.backward()
        _opt.step()
        _sched.step()
    icl_seconds = time.time() - _t0
    return icl_model, icl_seconds


@app.cell
def _(mo):
    icl_task = mo.ui.slider(0, 11, step=1, value=4, label="Held-out task to plot", show_value=True)
    icl_task
    return (icl_task,)


@app.cell
def _(Figure, PAL, icl_demo, icl_model, icl_rollout, icl_seconds, icl_task, icl_tasks, mo, np, torch):
    _gen = torch.Generator().manual_seed(7)
    _tasks = [icl_tasks(1, _gen) for _i in range(12)]
    _demos = [icl_demo(_g, _c, _gen) for _g, _c in _tasks]
    _res = {"the task's own demo": [], "another task's demo": [], "no demo": []}
    for _i, ((_g, _c), (_dP, _dA)) in enumerate(zip(_tasks, _demos)):
        _wP, _wA = _demos[(_i + 1) % 12]
        _res["the task's own demo"].append(icl_rollout(icl_model, _g, _c, _dP, _dA, 1, gen=_gen)[1])
        _res["another task's demo"].append(icl_rollout(icl_model, _g, _c, _wP, _wA, 1, gen=_gen)[1])
        _res["no demo"].append(icl_rollout(icl_model, _g, _c, _dP, _dA, 0, gen=_gen)[1])
    (_g, _c), (_dP, _dA) = _tasks[icl_task.value], _demos[icl_task.value]
    _with, _ = icl_rollout(icl_model, _g, _c, _dP, _dA, 1, n=10, gen=torch.Generator().manual_seed(1))
    _without, _ = icl_rollout(icl_model, _g, _c, _dP, _dA, 0, n=10, gen=torch.Generator().manual_seed(1))
    _fig = Figure(figsize=(8.4, 3.6))
    _axs = _fig.subplots(1, 2)
    for _ax, _traj, _title, _col in [(_axs[0], _with, "Demo in context", PAL["s1"]), (_axs[1], _without, "No demo", PAL["grey"])]:
        for _tr in _traj:
            _ax.plot(*_tr.T, color=_col, lw=1, alpha=0.85)
        _ax.scatter(*_g[0].numpy(), marker="*", s=200, color=PAL["good"], zorder=2, label="hidden goal")
        _ax.plot(*_dP[0].numpy().T, "--", color=PAL["s2"], lw=2, zorder=3, label="the demonstration")
        _ax.scatter(*_dP[0, 0].numpy(), s=30, color=PAL["s2"], zorder=3)
        _ax.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), aspect="equal", xticks=[], yticks=[], title=_title)
    _axs[0].legend(loc="lower left", fontsize=8)
    _table = "| in the context window | success on 12 held-out tasks |\n|---|---|\n" + "\n".join(
        f"| {_k} | {np.mean(_v):.0%} |" for _k, _v in _res.items()
    )
    mo.vstack([_fig, mo.md(_table + f"\n\nTrained in {icl_seconds:.0f} s on freshly drawn tasks; these 12 were never seen.")])
    return


@app.cell
def _(Figure, ICLPolicy, ICL_SPD, ICL_T, PAL, copy, icl_demo, icl_model, icl_rollout, icl_tasks, mo, np, torch):
    def icl_finetune(model, g, c, k, n_demos=5, lr=1e-3, gen=None):
        """k Adam steps on n_demos demonstrations of a new task, with the context window left empty."""
        m = copy.deepcopy(model)
        opt = torch.optim.Adam(m.parameters(), lr)
        P, A = icl_demo(g.expand(n_demos, 2), c.expand(n_demos), gen)
        q, y = P.reshape(-1, 2), A.reshape(-1, 2)
        empty = torch.zeros(len(q), ICL_T, 2)
        for _step in range(k):
            loss = ((m(empty, empty, q, torch.zeros(len(q))) - y) ** 2).mean() / ICL_SPD**2
            opt.zero_grad()
            loss.backward()
            opt.step()
        w0 = torch.cat([p.flatten() for p in model.parameters()])
        w1 = torch.cat([p.flatten() for p in m.parameters()])
        return m, ((w1 - w0).norm() / w0.norm()).item()

    _gen = torch.Generator().manual_seed(21)
    torch.manual_seed(21)
    _fresh = ICLPolicy()
    _rows = []
    for _k in (1, 3, 10, 30):
        _pre, _rnd, _dw = [], [], []
        for _trial in range(10):
            _g, _c = icl_tasks(1, _gen)
            _dP, _dA = icl_demo(_g, _c, _gen)
            _m_pre, _rel = icl_finetune(icl_model, _g, _c, _k, gen=_gen)
            _m_rnd, _rel_rnd = icl_finetune(_fresh, _g, _c, _k, gen=_gen)
            _pre.append(icl_rollout(_m_pre, _g, _c, _dP, _dA, 0, gen=_gen)[1])
            _rnd.append(icl_rollout(_m_rnd, _g, _c, _dP, _dA, 0, gen=_gen)[1])
            _dw.append(_rel)
        _rows.append((_k, np.mean(_pre), np.mean(_rnd), np.mean(_dw)))
    _rows = np.array(_rows)
    _fig = Figure(figsize=(5.2, 3.0))
    _ax = _fig.subplots()
    _ax.plot(_rows[:, 0], 100 * _rows[:, 1], "o-", color=PAL["s1"], label="pretrained")
    _ax.plot(_rows[:, 0], 100 * _rows[:, 2], "o-", color=PAL["grey"], label="random init")
    _ax.set(xscale="log", xticks=[1, 3, 10, 30], xticklabels=["1", "3", "10", "30"], ylim=(0, 105), xlabel="gradient steps", ylabel="success (%)", title="Few-step fine-tuning, empty context")
    _ax.legend(fontsize=8)
    _table = "| steps | pretrained | random init | ‖Δθ‖ / ‖θ‖, pretrained |\n|---|---|---|---|\n" + "\n".join(
        f"| {int(_r[0])} | {_r[1]:.0%} | {_r[2]:.0%} | {100 * _r[3]:.2f}% |" for _r in _rows
    )
    mo.hstack([_fig, mo.md(_table)], justify="start", gap=2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    What to look for: with the right demonstration in context the policy reaches tasks it never saw; another task's demonstration sends it to the wrong place, which proves it is reading the context and not guessing. Fine-tuning the pretrained model without a demo needs more steps than in-context learning, and pretraining makes those steps count. GEN-1.5 saw the same ordering on some tasks: in-context learning beat 1–5 gradient steps on the same data.

    The weight change is percents here and under 0.15% in GEN-1.5. Model size is not the reason. Adam moves each weight by at most about the learning rate per step, so after $k$ steps $\|\Delta\theta\| / \|\theta\|$ is bounded by roughly $k \cdot \mathrm{lr}$ over the typical weight magnitude, and the parameter count cancels. The toy fine-tunes at lr = 1e-3; large-model fine-tuning usually runs one to two orders of magnitude lower. The lesson carries over anyway: a small move in the right subspace is enough when pretraining already did the work.

    **What this toy leaves out.** Our model was trained on demo-then-query sequences, which is meta-learning by construction; GEN-1.5's point is that the capability *emerged* from plain sensorimotor pretraining. To probe that, train on continuous streams where tasks repeat with varying burstiness and see when in-context learning appears, following Chan et al. (2022), which Generalist cites.

    ## 8 · FLARE as a learning schema: federated learning across a fleet

    NVIDIA FLARE (Federated Learning Application Runtime Environment) is an SDK for training without moving data: clients train locally, ship model updates, and a server aggregates them. FedAvg, FedOpt and FedProx come built in, along with privacy filters meant to stop the server from reconstructing client data from updates.

    Mapped onto robots, every deployed robot or site is a client, and its homes, objects and body are its private, non-IID data. Dream Labs pitches exactly this flywheel, in which useful robots generate data that makes the next model better; federated learning is the version where the video of someone's kitchen never leaves the kitchen.

    The toy fleet has eight robots. Each has its own actuation (rotated up to ±60°, scaled ±40%) and its own slice of goal directions, so the data is non-IID in two ways. The task is goal reaching. Three ways to share one policy:

    - **One shared model, raw commands.** The same (state, goal) needs a different command on each robot, so the clients' labels conflict.
    - **Shared trunk, per-robot heads.** Only the trunk is averaged and each robot keeps its own output layer, much like GR00T's embodiment-specific action decoders.
    - **Shared end-effector space plus robot IK.** Every robot converts its labels into displacements and executes predicted displacements through its own IK. This is the GR00T N1.7 and Dream-1 move.

    For you: FedAvg with many local steps and a momentum outer step is DiLoCo. Each run takes a few seconds and reruns as you move the controls.
    """)
    return


@app.cell
def _(copy, nn, np, torch):
    FL_SPD = 0.1

    def fl_rot(t):
        return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])

    def fl_goals(robot, n, rng, width=None):
        w = robot["width"] if width is None else width
        phi = robot["center"] + rng.uniform(-w / 2, w / 2, n)
        r = rng.uniform(0.45, 0.8, n)
        return np.stack([r * np.cos(phi), r * np.sin(phi)], 1)

    def fl_desired(s, g):
        d = g - s
        n = np.linalg.norm(d, axis=1, keepdims=True)
        return FL_SPD * d / np.maximum(n, 1e-6) * np.minimum(1, n / FL_SPD)

    def fl_make_fleet(K=8, width_deg=60, n=256, seed=0):
        """K robots, each with its own actuation M and its own slice of goal directions."""
        rng = np.random.default_rng(seed)
        fleet = []
        for k in range(K):
            robot = {
                "M": (1 + rng.uniform(-0.4, 0.4)) * fl_rot(np.deg2rad(rng.uniform(-60, 60))),
                "center": 2 * np.pi * k / K,
                "width": np.deg2rad(width_deg),
            }
            drng = np.random.default_rng(seed * 100 + k)
            g = fl_goals(robot, n, drng)
            s = drng.uniform(-0.8, 0.8, (n, 2))
            d = fl_desired(s, g)
            robot["x"] = torch.tensor(np.concatenate([s, g], 1), dtype=torch.float32)
            robot["eef"] = torch.tensor(d / FL_SPD, dtype=torch.float32)  # shared end-effector space
            robot["raw"] = torch.tensor(d @ np.linalg.inv(robot["M"]).T / FL_SPD, dtype=torch.float32)  # this robot's commands
            fleet.append(robot)
        return fleet

    class FLPolicy(nn.Module):
        def __init__(self, h=64):
            super().__init__()
            self.trunk = nn.Sequential(nn.Linear(4, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU())
            self.head = nn.Linear(h, 2)

        def forward(self, x):
            return self.head(self.trunk(x))

    def fl_run(fleet, mode="fedavg", space="raw", local_heads=False, rounds=60, E=10, lr=0.05, mu=0.0,
               server_momentum=0.0, dp_sigma=0.0, clip=0.5, bs=32, seed=0):
        """mode: 'fedavg' | 'local' (no sharing) | 'central' (pooled data). Returns one policy per robot."""
        torch.manual_seed(seed)
        gen = torch.Generator().manual_seed(seed)
        key = "eef" if space == "eef" else "raw"
        glob = FLPolicy()
        if mode == "central":
            X, Y = torch.cat([r["x"] for r in fleet]), torch.cat([r[key] for r in fleet])
            opt = torch.optim.SGD(glob.parameters(), lr, momentum=0.9)
            for _step in range(rounds * E):
                idx = torch.randint(0, len(X), (2 * bs,), generator=gen)
                loss = ((glob(X[idx]) - Y[idx]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            return {"models": [glob] * len(fleet), "space": space}
        if mode == "local":
            models = []
            for r in fleet:
                m = copy.deepcopy(glob)
                opt = torch.optim.SGD(m.parameters(), lr, momentum=0.9)
                for _step in range(rounds * E):
                    idx = torch.randint(0, len(r["x"]), (bs,), generator=gen)
                    loss = ((m(r["x"][idx]) - r[key][idx]) ** 2).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                models.append(m)
            return {"models": models, "space": space}
        heads = [copy.deepcopy(glob.head) for _r in fleet]

        def shared(m):
            return m.trunk.parameters() if local_heads else m.parameters()

        mom = None
        for _round in range(rounds):
            g0 = torch.nn.utils.parameters_to_vector(shared(glob)).detach().clone()
            deltas = []
            for i, r in enumerate(fleet):
                m = copy.deepcopy(glob)
                if local_heads:
                    m.head = heads[i]
                opt = torch.optim.SGD(m.parameters(), lr, momentum=0.9)
                for _step in range(E):
                    idx = torch.randint(0, len(r["x"]), (bs,), generator=gen)
                    loss = ((m(r["x"][idx]) - r[key][idx]) ** 2).mean()
                    if mu > 0:  # FedProx: a leash back to this round's global model
                        loss = loss + 0.5 * mu * ((torch.nn.utils.parameters_to_vector(shared(m)) - g0) ** 2).sum()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                if local_heads:
                    heads[i] = m.head  # never leaves the robot
                delta = torch.nn.utils.parameters_to_vector(shared(m)).detach() - g0
                if dp_sigma > 0:  # bound each robot's influence before noise is added
                    delta = delta * min(1.0, clip / (delta.norm().item() + 1e-12))
                deltas.append(delta)
            agg = torch.stack(deltas).mean(0)
            if dp_sigma > 0:
                agg = agg + torch.randn(agg.shape, generator=gen) * dp_sigma * clip / len(fleet)
            if server_momentum > 0:  # FedAvgM, or DiLoCo's outer momentum
                mom = agg if mom is None else server_momentum * mom + agg
                agg = mom
            torch.nn.utils.vector_to_parameters(g0 + agg, shared(glob))
        models = []
        for i in range(len(fleet)):
            m = copy.deepcopy(glob)
            if local_heads:
                m.head = heads[i]
            models.append(m)
        return {"models": models, "space": space}

    @torch.no_grad()
    def fl_evaluate(fleet, result, own=True, n=64, T=25, seed=1):
        """Closed-loop success per robot, on its own goal slice (own=True) or on goals in every direction."""
        rng = np.random.default_rng(seed)
        out = []
        for i, r in enumerate(fleet):
            g = fl_goals(r, n, rng, None if own else 2 * np.pi)
            s = rng.uniform(-0.8, 0.8, (n, 2))
            done = np.zeros(n, bool)
            for _t in range(T):
                y = result["models"][i](torch.tensor(np.concatenate([s, g], 1), dtype=torch.float32)).numpy() * FL_SPD
                cmd = y @ np.linalg.inv(r["M"]).T if result["space"] == "eef" else y  # eef: the robot's own IK
                s = np.where(done[:, None], s, np.clip(s + cmd @ r["M"].T, -1, 1))
                done |= np.linalg.norm(s - g, axis=1) < 0.08
            out.append(done.mean())
        return np.array(out)
    return fl_evaluate, fl_make_fleet, fl_run


@app.cell
def _(mo):
    fl_space = mo.ui.dropdown(
        options={
            "One shared model, raw commands": "raw",
            "Shared trunk, per-robot heads": "heads",
            "Shared end-effector space + robot IK": "eef",
        },
        value="One shared model, raw commands",
        label="Action interface",
    )
    fl_local_steps = mo.ui.slider(1, 50, step=1, value=10, label="Local steps per round", show_value=True)
    fl_width = mo.ui.slider(30, 360, step=30, value=60, label="Goal directions per robot (°)", show_value=True)
    fl_mu = mo.ui.slider(0.0, 1.0, step=0.05, value=0.0, label="FedProx μ", show_value=True)
    fl_mom = mo.ui.slider(0.0, 0.9, step=0.1, value=0.0, label="Server momentum", show_value=True)
    fl_dp = mo.ui.slider(0.0, 2.0, step=0.1, value=0.0, label="DP noise σ", show_value=True)
    mo.vstack(
        [
            mo.hstack([fl_space, fl_local_steps, fl_width], justify="start", gap=1.5, wrap=True),
            mo.hstack([fl_mu, fl_mom, fl_dp], justify="start", gap=1.5, wrap=True),
        ]
    )
    return fl_dp, fl_local_steps, fl_mom, fl_mu, fl_space, fl_width


@app.cell
def _(Figure, PAL, fl_dp, fl_evaluate, fl_local_steps, fl_make_fleet, fl_mom, fl_mu, fl_run, fl_space, fl_width, mo, np, time):
    _fleet = fl_make_fleet(width_deg=fl_width.value)
    _t0 = time.time()
    _res = fl_run(
        _fleet,
        space="eef" if fl_space.value == "eef" else "raw",
        local_heads=fl_space.value == "heads",
        E=fl_local_steps.value,
        mu=fl_mu.value,
        server_momentum=fl_mom.value,
        dp_sigma=fl_dp.value,
    )
    _own, _all = fl_evaluate(_fleet, _res, own=True), fl_evaluate(_fleet, _res, own=False)
    _fig = Figure(figsize=(7.5, 2.9))
    _ax = _fig.subplots()
    _x = np.arange(len(_fleet))
    _ax.bar(_x - 0.2, 100 * _own, width=0.4, color=PAL["s1"], label="its own goal slice")
    _ax.bar(_x + 0.2, 100 * _all, width=0.4, color=PAL["s2"], label="goals in every direction")
    _ax.set(xticks=_x, xticklabels=[f"robot {i + 1}" for i in _x], ylim=(0, 105), ylabel="success (%)", title="Per-robot success after 60 rounds")
    _ax.legend(fontsize=8, loc="upper right")
    mo.vstack(
        [
            _fig,
            mo.md(
                f"Fleet average: **{_own.mean():.0%}** on each robot's own goals, **{_all.mean():.0%}** on goals in every direction. Run time {time.time() - _t0:.1f} s."
            ),
        ]
    )
    return


@app.cell
def _(fl_evaluate, fl_make_fleet, fl_run, mo):
    _fleet = fl_make_fleet()
    _configs = [
        ("Each robot alone, no sharing", {"mode": "local", "space": "raw"}),
        ("Pooled data in EEF space (privacy ignored)", {"mode": "central", "space": "eef"}),
        ("FedAvg, one model, raw commands", {"space": "raw"}),
        ("FedAvg, per-robot heads", {"space": "raw", "local_heads": True}),
        ("FedAvg, shared EEF space + IK", {"space": "eef"}),
        ("… plus server momentum 0.6", {"space": "eef", "server_momentum": 0.6}),
        ("… plus DP noise σ = 0.3", {"space": "eef", "dp_sigma": 0.3}),
    ]
    _rows = []
    for _name, _cfg in _configs:
        _res = fl_run(_fleet, **_cfg)
        _rows.append(f"| {_name} | {fl_evaluate(_fleet, _res, own=True).mean():.0%} | {fl_evaluate(_fleet, _res, own=False).mean():.0%} |")
    mo.md(
        "**Reference runs** at the default settings (60° slices, 10 local steps, 60 rounds):\n\n| setup | own goals | every direction |\n|---|---|---|\n"
        + "\n".join(_rows)
    )
    return


@app.cell
def _(mo, np):
    _rng = np.random.default_rng(0)
    _updates = _rng.normal(0, 1, (3, 5)).round(2)  # what three robots would like to send
    _masked = _updates.copy()
    for _i in range(3):
        for _j in range(_i + 1, 3):
            _mask = _rng.normal(0, 10, 5)  # robots i and j agree on this privately
            _masked[_i] += _mask
            _masked[_j] -= _mask

    def _fmt(v):
        return " ".join(f"{x:+7.2f}" for x in v)

    mo.md(
        f"""
    **Secure aggregation in five numbers.** Each pair of robots agrees on a random mask; one adds it, the other subtracts it. The server sees only masked vectors, yet their sum is exact.

    | | update (private) | what the server receives |
    |---|---|---|
    | robot A | `{_fmt(_updates[0])}` | `{_fmt(_masked[0])}` |
    | robot B | `{_fmt(_updates[1])}` | `{_fmt(_masked[1])}` |
    | robot C | `{_fmt(_updates[2])}` | `{_fmt(_masked[2])}` |
    | sum | `{_fmt(_updates.sum(0))}` | `{_fmt(_masked.sum(0))}` |
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    What to look for: one shared model with raw commands fails because the robots disagree about the right command. Per-robot heads fix each robot's own tasks but leave each robot competent only in its own slice of directions. The shared end-effector space works almost as well as pooling the data, and every robot inherits the others' directions. Server momentum speeds things up. DP noise is expensive with eight clients, because it has to hide any single client's contribution: σ = 0.3 drops the end-effector fleet from about 91% to 78%, and σ = 1 to about 30%. DP in federated learning needs many clients per round. Widen each robot's goal slice to 360° and the per-robot heads stop favouring their own slice, so that gap came from non-IID task coverage; the shared end-effector space still comes out ahead.

    **What this toy leaves out.** Communication: this policy is about 4.5k floats, whereas one full update of a 14B world action model is roughly 28 GB in bf16, and DreamZero reports that low-rank adapters underperformed full fine-tuning. GEN-1.5's tiny ten-step deltas hint that real updates may compress well; measuring their rank and sparsity is an open question. Also missing are clients dropping out mid-round, poisoned updates, privacy accounting (ε, δ), and FLARE's own client API, which wraps an existing PyTorch loop around receiving the global model, training and sending the update.

    ## 9 · A System 2 that learns: skills, not weights

    ASPIRE's System 2 is a frontier coding agent (Claude Opus 4.6 in the June paper; the project page adds Claude Opus 5.5 and GPT-6 Astra) that writes code-as-policy programs against a fixed primitive API. It reads multimodal execution traces, repairs the program, re-runs the fix on a small debug split and saves validated fixes to a text skill library; held-out seeds stay locked until the final evaluation. Tasks still below a success threshold go to evolutionary search, which evolves a population of program variants from the best earlier ones, because trace-guided repair alone can get stuck in local loops. Gemini 4 Argon shows the same pattern from Google: code-as-policies primitives, practice runs in MuJoCo, and verifier sub-agents that its demo prompts call in before the robot lets go of something. In both, the learning happens in text and no weights change.

    Toy: five object types with hidden requirements. A mug must be grasped at the handle, a bottle needs enough grip force but not too much, a marble needs a closer look first, and a plate must be slid to the table edge. The "agent" is a twenty-line rule table that reads the failing primitive's diagnostic, standing in for the LLM. A 24-task curriculum tracks tokens and repair rounds. At task 16 the agent moves to a different real robot whose gripper is weaker and whose wrist is mounted the other way round, so two stored skills become wrong. ASPIRE lists stale skills as an open problem.
    """)
    return


@app.cell
def _(np):
    AG_OBJECTS = {
        "cube": {"tol": 1.5, "noise": 0.6},
        "mug": {"tol": 1.5, "noise": 0.6, "yaw": 90},
        "bottle": {"tol": 1.5, "noise": 0.6, "fmin": 0.7, "fmax": 0.95},
        "marble": {"tol": 0.8, "noise": 1.6, "noise_close": 0.3},
        "plate": {"tol": 2.0, "noise": 0.6, "slide": True},
    }
    AG_DEFAULT = {"look_closer": False, "yaw": 0, "force": 0.4, "slide_first": False}

    def ag_execute(obj, prog, rng, robot="sim"):
        """The execution engine: detect -> (slide) -> grasp -> lift -> place, logging a trace of each call."""
        spec = dict(AG_OBJECTS[obj])
        gain = 1.0
        if robot == "real":  # a different robot: weaker gripper, wrist mounted the other way round
            gain = 0.75
            if "yaw" in spec:
                spec["yaw"] = -90
        trace = []
        sig = spec.get("noise_close", spec["noise"]) if prog["look_closer"] else spec["noise"]
        err = abs(rng.normal(0, sig)) + abs(rng.normal(0, sig))
        trace.append(("detect", f"pose estimate, error {err:.1f} cm"))
        if spec.get("slide") and not prog["slide_first"]:
            return False, trace + [("grasp", "FAIL: fingers hit the table under a flat object")]
        if spec.get("slide"):
            trace.append(("slide_to_edge", "ok"))
        if err > spec["tol"]:
            return False, trace + [("grasp", f"FAIL: missed, position error {err:.1f} cm > {spec['tol']} cm")]
        if "yaw" in spec and abs((prog["yaw"] - spec["yaw"] + 180) % 360 - 180) > 20:
            return False, trace + [("grasp", "FAIL: finger collided with the handle")]
        force = gain * prog["force"]
        if "fmin" in spec and force < spec["fmin"]:
            return False, trace + [("lift", f"FAIL: object slipped at commanded force {prog['force']:.2f}")]
        if "fmax" in spec and force > spec["fmax"]:
            return False, trace + [("lift", f"FAIL: object deformed at commanded force {prog['force']:.2f}")]
        return True, trace + [("place", "ok")]

    def ag_propose(prog, trace, rng, n):
        """Stand-in for the coding LLM: read the failing call's diagnostic, propose n edited programs."""
        msg, out = trace[-1][1], []
        for _cand in range(n):
            p = dict(prog)
            if "missed" in msg:
                p["look_closer"] = True if rng.uniform() < 0.8 else p["look_closer"]
            elif "handle" in msg:
                p["yaw"] = int(rng.choice([45, 90, 135, 180, -90, -45]))  # ambiguous fix: parallel candidates help
            elif "slipped" in msg:
                p["force"] = round(min(1.0, p["force"] + rng.choice([0.1, 0.2, 0.3])), 2)
            elif "deformed" in msg:
                p["force"] = round(max(0.0, p["force"] - 0.1), 2)
            elif "table" in msg:
                p["slide_first"] = True
            out.append(p)
        return out

    def ag_solve(objs, library, rng, use_library, pop, robot, max_rounds=12, val_seeds=5):
        """Write, run, diagnose, repair, validate. Returns (tokens, rounds, solved)."""
        tokens = rounds = 0
        solved_all = True
        for obj in objs:
            prog = dict(AG_DEFAULT)
            if use_library and obj in library:
                prog.update(library[obj])
                tokens += 300  # retrieving a text skill into context is cheap
            trace, solved = None, False
            for r in range(max_rounds):
                rounds += 1
                cands = [prog] if r == 0 else ag_propose(prog, trace, rng, pop)
                tokens += 0 if r == 0 else 2500 * len(cands)  # writing repaired programs
                results = []
                for c in cands:
                    ok, tr = ag_execute(obj, c, rng, robot)
                    tokens += 600  # reading the trace
                    results.append((ok, tr, c))
                win = next((x for x in results if x[0]), None)
                if win is not None:
                    val = np.mean([ag_execute(obj, win[2], rng, robot)[0] for _seed in range(val_seeds)])
                    tokens += 200 * val_seeds
                    prog = win[2]
                    if val >= 0.8:  # validated on held-out seeds: keep it
                        solved = True
                        if use_library:
                            library[obj] = {k: v for k, v in prog.items() if v != AG_DEFAULT[k]}
                        break
                    fails = [x for x in results if not x[0]]
                    trace = fails[0][1] if fails else [("grasp", "FAIL: missed, position error (intermittent)")]
                else:
                    trace, prog = results[0][1], results[0][2]
            solved_all = solved_all and solved
        return tokens, rounds, solved_all

    def ag_curriculum(use_library, pop, shift_at, n_tasks=24, seed=0):
        rng = np.random.default_rng(seed)
        library, rows = {}, []
        names = list(AG_OBJECTS)
        for i in range(n_tasks):
            objs = [str(o) for o in rng.choice(names, size=int(rng.integers(1, 3)), replace=False)]
            robot = "real" if shift_at is not None and i >= shift_at else "sim"
            tok, rnd, ok = ag_solve(objs, library, rng, use_library, pop, robot)
            rows.append({"task": i, "objects": objs, "tokens": tok, "rounds": rnd, "ok": ok, "robot": robot})
        return rows, library
    return (ag_curriculum,)


@app.cell
def _(mo):
    ag_library = mo.ui.switch(value=True, label="Keep a skill library")
    ag_pop = mo.ui.slider(1, 6, step=1, value=1, label="Candidate programs per repair round", show_value=True)
    ag_shift = mo.ui.switch(value=True, label="Move to a different robot at task 16")
    mo.hstack([ag_library, ag_pop, ag_shift], justify="start", gap=1.5, wrap=True)
    return ag_library, ag_pop, ag_shift


@app.cell
def _(Figure, PAL, ag_curriculum, ag_library, ag_pop, ag_shift, mo, np):
    _shift = 16 if ag_shift.value else None
    _rows, _lib = ag_curriculum(ag_library.value, ag_pop.value, _shift)
    _base, _base_lib = ag_curriculum(False, 1, _shift)  # reference: no library, one candidate per round
    _fig = Figure(figsize=(10, 3.1))
    _a, _b = _fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.6, 1]})
    _x = np.arange(len(_rows))
    _a.bar(_x, [r["tokens"] / 1000 for r in _rows], width=0.75, color=[PAL["s2"] if r["ok"] else PAL["bad"] for r in _rows])
    if _shift is not None:
        _a.axvline(_shift - 0.5, color=PAL["ink"], ls="--", lw=1)
        _a.text(_shift - 0.8, _a.get_ylim()[1] * 0.92, "new robot →", fontsize=8, ha="right")
    _a.set(xlabel="task in the curriculum", ylabel="thousand tokens", title="Cost per task (red: unsolved)")
    _b.plot(np.cumsum([r["tokens"] for r in _base]) / 1000, color=PAL["grey"], label="no library, 1 candidate")
    _b.plot(np.cumsum([r["tokens"] for r in _rows]) / 1000, color=PAL["s2"], label="these settings")
    _b.set(xlabel="task", ylabel="cumulative thousand tokens", title="Running total")
    _b.legend(fontsize=8)
    _lines = []
    for _obj in sorted(_lib):
        _fix = ", ".join(f"{_p} = {_v}" for _p, _v in _lib[_obj].items())
        _lines.append(f"| {_obj} | {_fix if _fix else 'the default program works'} |")
    _skills = "\n".join(_lines) if _lines else "| (library off) | |"
    _solved = sum(r["ok"] for r in _rows)
    _rounds = sum(r["rounds"] for r in _rows)
    _tok = sum(r["tokens"] for r in _rows) / 1000
    _tok_base = sum(r["tokens"] for r in _base) / 1000
    mo.vstack(
        [
            _fig,
            mo.md(
                f"Solved {_solved}/24 tasks with {_rounds} repair rounds and {_tok:.0f}k tokens; the reference spent {_tok_base:.0f}k.\n\n| skill library entry | stored fix |\n|---|---|\n"
                + _skills
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    What to look for: the library roughly halves the token bill, because later tasks start from validated fixes. More candidates per round cut the number of rounds, which is wall-clock time, at a higher token cost, the evolutionary-search trade-off. After the robot change, the stored mug and bottle skills are wrong for the new robot, the agent pays to re-learn them, and the library entries are overwritten.

    **What this toy leaves out.** The intelligence. ASPIRE's results come from frontier models reading code and images. In the project's October PDF, weaker models still run the loop but score lower: on 10 LIBERO-Pro Goal tasks, success runs from 15% with GPT-6 Luna to 99% with Claude Opus 5.5. Real primitives are also messier than one diagnostic string, and deciding *whether* a task succeeded is itself unsolved in the real world: ASPIRE says real-world deployment still needs robust success detection, safe resets, safety monitoring and calibration, and Argon's demo prompts call in verifier sub-agents for the same reason.

    ## Where to take it next

    1. **§2:** replace the MLP flow head with a small DiT over chunk tokens, with adaptive-norm timestep conditioning as in GR00T N1.7 and π0.7, and measure how the step count needed falls.
    2. **§3:** implement test-time RTC, inpainting the prefix with guidance, and compare it with the training-time version for cost and smoothness.
    3. **§6:** swap the frozen random projection for an EMA target encoder, which is your JEPA setup, and see whether latent alignment closes the gap to the world action model when no demos exist.
    4. **§7:** pretrain on continuous streams with tunable task burstiness and find where in-context learning appears.
    5. **§8:** federate the §6 video model across robots with different cameras, log the update sizes, and try low-rank deltas.
    6. **Real models on the DGX Spark:** NVIDIA reports GR00T N1.7 at about 10 Hz there with TensorRT. Profiling its 32-layer flow-matching action head at batch 1, and asking how much of each denoising step is launch overhead, is exactly your kind of problem.
    """)
    return


if __name__ == "__main__":
    app.run()
