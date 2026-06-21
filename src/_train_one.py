"""Train one model and save pos_fn f(t) + evaluation. Called per (rho, sigma)."""
import numpy as np, torch, sys, os, json

D = 64; EPOCHS = 4000; SEED = 42; N_T = 201

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

from src.model import TCMEncoder, AttentionDecoder
from src.train import make_items
from src.utils import apply_noise
from src.config import TEST_POSITIONS
from src.analysis import compute_human_loss

rho = float(sys.argv[1])
sigma_m = float(sys.argv[2])
out_file = sys.argv[3]

torch.manual_seed(SEED); np.random.seed(SEED)
encoder = TCMEncoder(rho, D, D)
decoder = AttentionDecoder(D, use_position_template=False)
opt = torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr=5e-4)
crit = torch.nn.MSELoss()

for _ in range(EPOCHS):
    L = np.random.randint(60, 181)
    items = make_items(32, L, D)
    fv, c = encoder(items)
    q = torch.randint(0, L, (32,))
    c_q = apply_noise(c[torch.arange(32), q, :], sigma_m)
    pred = decoder(c_q, c)
    loss = crit(pred, q.float()/L)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(list(encoder.parameters())+list(decoder.parameters()), 1.0)
    opt.step()

with torch.no_grad():
    t = torch.linspace(0, 1, N_T).unsqueeze(-1)
    f_vals = decoder.pos_fn(t).squeeze(-1).cpu().numpy()
    t_np = t.squeeze().cpu().numpy()

    eval_items = make_items(500, 100, D)
    _, eval_c = encoder(eval_items)
    errors = {}
    for pf in TEST_POSITIONS:
        ti = int(pf*100)
        cq_n = apply_noise(eval_c[:50, ti, :], sigma_m)
        p = decoder(cq_n, eval_c[:50]).cpu().numpy()
        errors[pf] = (p-pf)*100
    means = [float(np.mean(errors[pf])) for pf in TEST_POSITIONS]
    asm = (abs(means[0])+abs(means[1])-abs(means[2])-abs(means[3]))/2
    hl = float(compute_human_loss(means))
    mse = float(loss.item())

    lin_fit = np.polyfit(t_np, f_vals, 1)
    lin_pred = np.polyval(lin_fit, t_np)
    r2 = float(1 - np.sum((f_vals-lin_pred)**2) / np.sum((f_vals-np.mean(f_vals))**2))

os.makedirs(os.path.dirname(out_file), exist_ok=True)
np.savez(out_file, t=t_np, f_vals=f_vals, means=np.array(means),
         asm=asm, hl=hl, mse=mse, r2=r2)
print(f"done: ρ={rho:.2f} σ={sigma_m:.2f} asm={asm:+.1f} R²={r2:.3f}")
