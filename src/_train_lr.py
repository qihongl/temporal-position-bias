"""Train one model for LR x epochs grid. Usage: python _train_lr.py <lr> <epochs> <seed> <outfile>"""
import torch, numpy as np, sys, os
D = 64; RHO = 0.95; SIGMA = 0.05
lr = float(sys.argv[1]); epochs = int(sys.argv[2]); seed = int(sys.argv[3]); outfile = sys.argv[4]

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)
from src.model import TCMEncoder, AttentionDecoder
from src.train import make_items
from src.utils import apply_noise
from src.config import TEST_POSITIONS

torch.manual_seed(seed); np.random.seed(seed)
enc = TCMEncoder(RHO, D, D, use_proj=True)
dec = AttentionDecoder(D, use_position_template=True)
opt = torch.optim.Adam(list(enc.parameters())+list(dec.parameters()), lr=lr)
crit = torch.nn.MSELoss()

for _ in range(epochs):
    L = np.random.randint(60, 181)
    items = make_items(32, L, D)
    _, c = enc(items)
    q = torch.randint(0, L, (32,))
    c_q = apply_noise(c[torch.arange(32), q, :], SIGMA)
    pred = dec(c_q, c)
    loss = crit(pred, q.float()/L)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(list(enc.parameters())+list(dec.parameters()), 1.0)
    opt.step()

with torch.no_grad():
    eval_items = make_items(500, 100, D)
    _, eval_c = enc(eval_items)
    means = []
    for pf in TEST_POSITIONS:
        ti = int(pf*100)
        cq_n = apply_noise(eval_c[:50, ti, :], SIGMA)
        p = dec(cq_n, eval_c[:50]).cpu().numpy()
        means.append(float(np.mean((p-pf)*100)))
    asm = (abs(means[0])+abs(means[1])-abs(means[2])-abs(means[3]))/2
    mae = float(np.mean([abs(m) for m in means]))
    train_loss = float(loss.item())

os.makedirs(os.path.dirname(outfile), exist_ok=True)
torch.save({'means': means, 'asm': asm, 'mae': mae, 'train_loss': train_loss,
            'lr': lr, 'epochs': epochs, 'seed': seed}, outfile)
print(f'done: lr={lr:.0e} ep={epochs} seed={seed} HL reported later')
