"""Paired zero-start / 100-item prelude control, using the original model.

Run with the project's Python environment. No original src file is modified.
The vectorized recurrence is checked against TCMEncoder including gradients.
"""
from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from _paths import REPO_ROOT
from src.model import TCMEncoder, AttentionDecoder
from src.config import D_ITEM, D_CONTEXT, EPOCHS, BATCH_SIZE, LR, SEQ_MIN, SEQ_MAX

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
RHO, SIGMA = .95, .05
POSITIONS = [.2, .4, .6, .8]
MIRROR_INDICES = [20, 40, 59, 79]
LEGACY_INDICES = [20, 40, 60, 80]


def generator(seed):
    return torch.Generator(device='cpu').manual_seed(int(seed))


def encode_fast(encoder, items):
    """Algebraically identical finite-start EMA; safe for rho=.95, L<=580."""
    f = F.normalize(encoder.proj(items), dim=-1)
    powers = encoder.rho ** torch.arange(items.shape[1], dtype=f.dtype, device=f.device)
    c = (1 - encoder.rho) * powers[None, :, None] * torch.cumsum(
        f / powers[None, :, None], dim=1)
    return f, c


def encode_window(encoder, items, prefix=None):
    if prefix is None:
        return encode_fast(encoder, items)
    f, c = encode_fast(encoder, torch.cat([prefix, items], dim=1))
    return f[:, prefix.shape[1]:], c[:, prefix.shape[1]:]


def validate_encoder():
    """Check actual recurrence and parameter/input gradients, not just shapes."""
    torch.set_num_threads(1)
    records = []
    for length in [60, 180, 300, 580]:
        torch.manual_seed(391 + length)
        reference = TCMEncoder(RHO).double()
        candidate = TCMEncoder(RHO).double()
        candidate.load_state_dict(reference.state_dict())
        x1 = torch.randn(3, length, D_ITEM, dtype=torch.float64, requires_grad=True)
        x2 = x1.detach().clone().requires_grad_(True)
        f1, c1 = reference(x1)
        f2, c2 = encode_fast(candidate, x2)
        weights = torch.randn_like(c1)
        (c1 * weights).sum().backward()
        (c2 * weights).sum().backward()
        torch.testing.assert_close(c1, c2, rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(x1.grad, x2.grad, rtol=1e-9, atol=1e-11)
        torch.testing.assert_close(reference.proj.weight.grad, candidate.proj.weight.grad,
                                   rtol=1e-9, atol=1e-11)
        enc32 = reference.float()
        with torch.no_grad():
            orig32 = enc32(x1.detach().float())[1]
            fast32 = encode_fast(enc32, x1.detach().float())[1]
        torch.testing.assert_close(orig32, fast32, rtol=2e-5, atol=1e-7)
        records.append({'length': length, 'float64_max_error': (c1-c2).abs().max().item(),
                        'float32_max_error': (orig32-fast32).abs().max().item()})
    # Warm-up states cannot enter the decoder or shift its position template.
    e = TCMEncoder(RHO)
    seq, prefix = torch.randn(2, 100, 64), torch.randn(2, 200, 64)
    _, all_c = e(torch.cat([prefix, seq], 1))
    _, window = encode_window(e, seq, prefix)
    assert window.shape == (2, 100, 64)
    torch.testing.assert_close(window, all_c[:, 200:], rtol=2e-5, atol=1e-7)
    assert all(MIRROR_INDICES[i] + MIRROR_INDICES[3-i] == 99 for i in range(4))
    OUT.mkdir(exist_ok=True)
    (OUT / 'validation.json').write_text(json.dumps(records, indent=2))
    print('Encoder outputs, input/parameter gradients, and window indexing verified.', flush=True)


def evaluate(encoder, decoder, seed, burnin, n_test=2000, batch=50):
    """Same held-out windows/noise at every condition and burn-in length.

    The recent 200 items of the 400-item prefix are shared with 200 burn-in.
    A single set of sequences is probed at all positions. This permits paired
    comparisons; independent sampling units for inference are trained seeds.
    """
    items_rng, prefix_rng, noise_rng = [generator(seed + off) for off in (500000, 600000, 700000)]
    idx = sorted(set(MIRROR_INDICES + LEGACY_INDICES))
    errors = {q: [] for q in idx}
    norm_sq, attention = [], {q: [] for q in MIRROR_INDICES}
    fw_gaps = []
    encoder.eval(); decoder.eval()
    with torch.no_grad():
        for start in range(0, n_test, batch):
            b = min(batch, n_test-start)
            items = torch.randn(b, 100, D_ITEM, generator=items_rng)
            pre = torch.randn(b, 400, D_ITEM, generator=prefix_rng)
            _, c = encode_window(encoder, items, pre[:, -burnin:] if burnin else None)
            norm_sq.append(c.square().sum(-1).numpy())
            # Mean raw matched-lag dot product: both neighbors exist at each q.
            gaps = []
            for q in MIRROR_INDICES:
                gaps.append(torch.stack([
                    (c[:, q]*(c[:, q+k]-c[:, q-k])).sum(-1) for k in range(1, 11)
                ], -1).numpy())
            fw_gaps.append(np.stack(gaps, axis=1))
            for q in idx:
                noise = torch.randn(b, D_CONTEXT, generator=noise_rng)
                query = F.normalize(c[:, q] + SIGMA * noise, dim=-1)
                pred, weights = decoder(query, c, return_weights=True)
                errors[q].append(((pred - q/99)*100).numpy())
                if q in attention:
                    attention[q].append(weights.numpy())
    result = {f'p{int(p*100)}': np.concatenate(errors[q])
              for p, q in zip(POSITIONS, MIRROR_INDICES)}
    result.update({f'legacy_p{int(p*100)}': np.concatenate(errors[q])
                   for p, q in zip(POSITIONS, LEGACY_INDICES)})
    result['context_norm_squared'] = np.concatenate(norm_sq).mean(0)
    result['matched_lag_dot_gap'] = np.concatenate(fw_gaps).mean(0)
    result['attention_mean'] = np.stack([np.concatenate(attention[q]).mean(0) for q in MIRROR_INDICES])
    return result


def train_one(seed, burnin, epochs=EPOCHS, n_test=2000, benchmark=False):
    if burnin not in (0, 100):
        raise ValueError('The active experiment supports only zero or 100 burn-in items.')
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    condition = 'zero' if burnin == 0 else f'burnin{burnin}'
    directory = OUT / condition / f'seed{seed}'
    if not benchmark and (directory/'complete.json').exists():
        previous=json.loads((directory/'complete.json').read_text())
        if previous['epochs']!=epochs or previous['test_per_position']!=n_test or previous['burnin']!=burnin:
            raise ValueError('Completed run settings differ; use a separate output directory.')
        return {'seed': seed, 'condition': condition, 'status': 'cached'}
    torch.manual_seed(seed)
    encoder, decoder = TCMEncoder(RHO), AttentionDecoder()
    optimizer = torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr=LR)
    params = list(encoder.parameters())+list(decoder.parameters())
    lengths = np.random.RandomState(seed)
    item_rng, query_rng, noise_rng, prefix_rng = [generator(seed+off) for off in (100000,200000,300000,400000)]
    losses, checksum = [], hashlib.sha256()
    started = time.monotonic()
    for epoch in range(epochs):
        length = lengths.randint(SEQ_MIN, SEQ_MAX+1)
        items = torch.randn(BATCH_SIZE, length, D_ITEM, generator=item_rng)
        q = torch.randint(0, length, (BATCH_SIZE,), generator=query_rng)
        noise = torch.randn(BATCH_SIZE, D_CONTEXT, generator=noise_rng)
        if epoch in [0, epochs-1]:
            for a in [items, q, noise]:
                checksum.update(a.numpy().tobytes())
        prefix = torch.randn(BATCH_SIZE, burnin, D_ITEM, generator=prefix_rng) if burnin else None
        _, c = encode_window(encoder, items, prefix)
        query = F.normalize(c[torch.arange(BATCH_SIZE), q]+SIGMA*noise, dim=-1)
        pred = decoder(query, c)
        loss = F.mse_loss(pred, q.float()/(length-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.)
        optimizer.step()
        losses.append(loss.item())
        if (epoch+1) % 1000 == 0:
            print(f'{condition} seed={seed} step={epoch+1}/{epochs} mse={np.mean(losses[-100:]):.6f}', flush=True)
    train_seconds = time.monotonic()-started
    if benchmark:
        return {'condition': condition,'steps': epochs,'seconds': train_seconds,
                'seconds_per_step': train_seconds/epochs}
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory/'errors.npz', **evaluate(encoder,decoder,seed,burnin,n_test))
    np.save(directory/'training_loss.npy', np.array(losses))
    torch.save({'encoder':encoder.state_dict(),'decoder':decoder.state_dict(),
                'seed':seed,'burnin':burnin},directory/'checkpoint.pt')
    info={'seed':seed,'condition':condition,'burnin':burnin,'epochs':epochs,
          'train_seconds':train_seconds,'training_input_checksum':checksum.hexdigest(),
          'final_train_mse':losses[-1],'last100_train_mse':float(np.mean(losses[-100:])),
          'temperature':decoder.temperature.item(),'test_per_position':n_test}
    (directory/'complete.json').write_text(json.dumps(info,indent=2))
    print('COMPLETE '+json.dumps(info),flush=True)
    return info


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--seeds',type=int,default=20)
    parser.add_argument('--epochs',type=int,default=EPOCHS)
    parser.add_argument('--n-test',type=int,default=2000)
    parser.add_argument('--benchmark',action='store_true')
    args=parser.parse_args()
    validate_encoder()
    if args.benchmark:
        for burnin in (0,100):
            print(json.dumps(train_one(42,burnin,epochs=60,benchmark=True)),flush=True)
        return
    config={'rho':RHO,'sigma':SIGMA,'dimensions':D_CONTEXT,'batch_size':BATCH_SIZE,
            'lr':LR,'epochs':args.epochs,'train_lengths':[SEQ_MIN,SEQ_MAX],
            'eval_length':100,'eval_indices':MIRROR_INDICES,'legacy_indices':LEGACY_INDICES,
            'seeds':list(range(42,42+args.seeds)),'n_test':args.n_test,
            'workers':args.workers,'torch_version':torch.__version__,
            'numpy_version':np.__version__,'device':'cpu','threads_per_worker':1,
            'source_sha256':{str(p.relative_to(REPO_ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (REPO_ROOT/'src').glob('*.py') if not p.name.startswith('._')}}
    if (OUT/'config.json').exists():
        previous=json.loads((OUT/'config.json').read_text())
        previous['source_sha256']={k:v for k,v in previous['source_sha256'].items()
                                   if not Path(k).name.startswith('._')}
        for key in ['rho','sigma','dimensions','batch_size','lr','epochs','train_lengths',
                    'eval_length','eval_indices','legacy_indices','seeds','n_test','source_sha256']:
            if previous[key]!=config[key]:
                raise ValueError(f'Existing output differs on {key}; use a separate output directory.')
    (OUT/'config.json').write_text(json.dumps(config,indent=2))
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(train_one,s,b,args.epochs,args.n_test)
                 for s in config['seeds'] for b in (0,100)]
        for f in concurrent.futures.as_completed(futures):
            f.result()
    for seed in config['seeds']:
        a=json.loads((OUT/'zero'/f'seed{seed}'/'complete.json').read_text())
        b=json.loads((OUT/'burnin100'/f'seed{seed}'/'complete.json').read_text())
        assert a['training_input_checksum']==b['training_input_checksum'], 'Pairing mismatch'
    analysis_config={**config,'train_burnin':100,'evaluation_burnins':[100]}
    (OUT/'config_burnin100.json').write_text(json.dumps(analysis_config,indent=2))
    print('ALL DONE; paired zero/100 training inputs verified.',flush=True)


if __name__=='__main__':
    main()
