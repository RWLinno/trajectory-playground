"""A minimal conditional diffusion experiment in a one-coefficient path family.
All paths are x(t)=-.9+1.8t, y(t)=slope*x(t)+a*sin(pi*t).
This is a pedagogical model, not an unconstrained motion planner.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
STEPS = 64


def paths(amplitudes, condition, points=65):
    x = np.linspace(-.9, .9, points)
    y = np.asarray(amplitudes)[:, None] * np.sin(np.pi * (x + .9) / 1.8)
    y += condition[2] * x
    return np.stack([np.broadcast_to(x, y.shape), y], axis=-1)


def valid_paths(trajectories, condition):
    """Continuous segment-circle collision checks, including endpoints."""
    radius, cx, slope = condition
    center = np.array([cx, slope * cx])
    a, b = trajectories[:, :-1], trajectories[:, 1:]
    delta = b - a
    frac = np.sum((center - a) * delta, axis=-1) / np.maximum(np.sum(delta**2, axis=-1), 1e-15)
    nearest = a + np.clip(frac, 0, 1)[..., None] * delta
    clearance = np.linalg.norm(nearest - center, axis=-1).min(axis=1) - radius
    return (clearance > 0) & (np.abs(trajectories).max(axis=(1, 2)) <= 1), clearance


def metrics(trajectories, condition):
    valid, clearance = valid_paths(trajectories, condition)
    lengths = np.linalg.norm(np.diff(trajectories, axis=1), axis=-1).sum(axis=1)
    mode = trajectories[:, trajectories.shape[1] // 2, 1] > 0
    valid_modes = len(set(mode[valid].tolist()))
    return {'valid_rate': float(valid.mean()), 'collision_rate': float((clearance <= 0).mean()),
            'valid_modes': valid_modes, 'mean_length': float(lengths.mean()),
            'min_clearance': float(clearance.min()), 'samples': len(trajectories)}


def conditions(rng, n):
    return np.column_stack([rng.uniform(.16, .32, n), rng.uniform(-.18, .18, n), rng.uniform(-.2, .2, n)])


def reference_amplitudes(rng, c):
    # Symmetric paired demonstrations make the conditional mean exactly zero.
    clearance = (c[:, 0] + .16) / np.sin(np.pi * (c[:, 1] + .9) / 1.8)
    return (clearance + rng.uniform(-.025, .025, len(c))) * rng.choice([-1, 1], len(c))


class Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))
    def forward(self, c):
        return self.net(c)


class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(12, 96), nn.SiLU(), nn.Linear(96, 96), nn.SiLU(), nn.Linear(96, 1))
    def forward(self, x, t, c):
        phase = t[:, None].float() / STEPS * torch.tensor([1., 2., 4., 8.], device=x.device) * np.pi
        return self.net(torch.cat([x, c, phase.sin(), phase.cos()], dim=1))


def schedule():
    f = torch.cos((torch.linspace(0, 1, STEPS + 1) + .008) / 1.008 * torch.pi / 2) ** 2
    beta = (1 - f[1:] / f[:-1]).clamp(.0001, .999)
    alpha = 1 - beta
    return beta, alpha, alpha.cumprod(0)


@torch.no_grad()
def sample(model, c, generator):
    beta, alpha, abar = schedule()
    x = torch.randn((len(c), 1), generator=generator)
    for step in reversed(range(STEPS)):
        t = torch.full((len(c),), step, dtype=torch.long)
        predicted = model(x, t, c)
        mean = (x - beta[step] / (1 - abar[step]).sqrt() * predicted) / alpha[step].sqrt()
        if step:
            posterior = beta[step] * (1 - abar[step-1]) / (1 - abar[step])
            x = mean + posterior.sqrt() * torch.randn(x.shape, generator=generator)
        else:
            x = mean
    return x[:, 0].numpy()


def run(args):
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    start = time.time()
    # Train/held-out conditions are drawn independently with distinct RNG seeds.
    c = conditions(rng, 4096)
    a = np.abs(reference_amplitudes(rng, c))
    train_c = torch.tensor(np.repeat(c, 2, axis=0), dtype=torch.float32)
    train_a = torch.tensor(np.stack([a, -a], axis=1).reshape(-1, 1), dtype=torch.float32)
    mlp, diffusion = Regressor(), Denoiser()
    optim_m = torch.optim.Adam(mlp.parameters(), lr=.002)
    optim_d = torch.optim.Adam(diffusion.parameters(), lr=.001)
    _, _, abar = schedule()
    trace = []
    for step in range(args.steps):
        # Both members of each pair are included: no accidental mode imbalance.
        pair = torch.randint(4096, (128,))
        idx = (pair[:, None] * 2 + torch.tensor([0, 1])).flatten()
        bc, ba = train_c[idx], train_a[idx]
        loss_m = ((mlp(bc) - ba) ** 2).mean()
        optim_m.zero_grad(); loss_m.backward(); optim_m.step()
        t = torch.randint(STEPS, (len(idx),))
        noise = torch.randn_like(ba)
        noisy = abar[t, None].sqrt() * ba + (1 - abar[t, None]).sqrt() * noise
        loss_d = ((diffusion(noisy, t, bc) - noise) ** 2).mean()
        optim_d.zero_grad(); loss_d.backward(); optim_d.step()
        if step % 500 == 0 or step == args.steps-1:
            row = {'step': step, 'mlp_mse': loss_m.item(), 'diffusion_noise_mse': loss_d.item()}
            trace.append(row); print(json.dumps(row), flush=True)
    mlp.eval(); diffusion.eval()
    out = ROOT / 'artifacts'; out.mkdir(exist_ok=True)
    torch.save({'mlp': mlp.state_dict(), 'diffusion': diffusion.state_dict(), 'seed': args.seed, 'steps': args.steps}, out / 'models.pt')
    eval_rng = np.random.default_rng(20261014)
    heldout = conditions(eval_rng, 128)
    generator = torch.Generator().manual_seed(54321)
    all_rows = {'mlp': [], 'diffusion': [], 'reference': []}
    scenes = []
    # Nine published presets plus a separate 128-condition evaluation.
    presets = [(r, cx, s) for r in [.18, .25, .32] for cx, s in [(-.18, -.2), (0., 0.), (.18, .2)]]
    for i, cond in enumerate(list(presets) + list(heldout)):
        cc = torch.tensor(np.tile(cond, (64, 1)), dtype=torch.float32)
        with torch.no_grad(): ma = mlp(cc)[:, 0].numpy()
        da = sample(diffusion, cc, generator)
        ra = reference_amplitudes(eval_rng, np.tile(cond, (64, 1)))
        bundles = {}
        for name, amplitude in [('mlp', ma), ('diffusion', da), ('reference', ra)]:
            traj = paths(amplitude, cond)
            m = metrics(traj, cond)
            if i >= len(presets): all_rows[name].append(m)
            bundles[name] = {'amplitudes': np.round(amplitude, 6).tolist(), 'metrics': m}
        if i < len(presets):
            scenes.append({'id': i, 'condition': list(cond), 'models': bundles})
    summary = {}
    for name, rows in all_rows.items():
        summary[name] = {key: float(np.mean([r[key] for r in rows])) for key in ['valid_rate','collision_rate','valid_modes','mean_length']}
    metadata = {'seed': args.seed, 'training_steps': args.steps, 'training_conditions': 4096, 'training_examples':8192,
                'evaluation_seed': 54321, 'test_conditions':128, 'samples_per_condition':64,
                'diffusion_steps':STEPS, 'torch':torch.__version__, 'numpy':np.__version__,
                'seconds':round(time.time()-start, 2), 'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'weights_sha256':hashlib.sha256((out/'models.pt').read_bytes()).hexdigest()}
    report = {'metadata':metadata, 'summary':summary, 'per_condition':all_rows, 'training_trace':trace}
    (out/'evaluation.json').write_text(json.dumps(report, indent=2)+'\n')
    payload = {'metadata':metadata, 'summary':summary, 'scenes':scenes}
    (ROOT/'docs/assets/samples.json').write_text(json.dumps(payload, indent=2)+'\n')
    print(json.dumps({'metadata':metadata,'summary':summary}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=6000)
    parser.add_argument('--seed', type=int, default=42)
    run(parser.parse_args())
