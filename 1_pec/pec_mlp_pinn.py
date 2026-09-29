import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
import time

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
torch.manual_seed(0)
np.random.seed(0)

k      = 2.0 * np.pi   # wave number  (wavelength = 1)
a      = 0.05           # scatterer radius  (ka ~ 0.314, Rayleigh regime)
cx, cy = 0.5, 0.5       # scatterer centre

# Collocation sizes
N_INT   = 3000   # interior PDE residual points
N_SCAT  = 600    # points on scatterer boundary dD
N_OUT   = 600    # points on outer boundary dOmega (150 per side)

# Training schedule
ADAM_EPOCHS  = 3000
LBFGS_STEPS  = 1000
LR_ADAM      = 8e-4

# Loss weights
W_PDE   = 1.0
W_SCAT  = 80.0
W_OUT   = 20.0

# Network architecture — wider/deeper than Fourier version to compensate
# for the lack of spectral embedding
HIDDEN    = 3     # hidden layers  (was 3 with Fourier features)
NEURONS   = 96    # neurons per layer  (was 96)
IN_DIM    = 2     # raw (x, y) input — no embedding

DEVICE = torch.device("cpu")
ka = k * a
print(f"Device     : {DEVICE}")
print(f"k={k:.4f},  a={a:.4f},  ka={ka:.4f}  (Rayleigh regime: ka < 1)")

# ─────────────────────────────────────────────────────────────────────────────
# Network definition
# ─────────────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Linear(IN_DIM, NEURONS), nn.Tanh()]
        for _ in range(HIDDEN - 1):
            layers += [nn.Linear(NEURONS, NEURONS), nn.Tanh()]
        layers += [nn.Linear(NEURONS, 1)]
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, xy):
        return self.net(xy)   # raw (x, y) coordinates — no embedding


class PINN(nn.Module):
    """Two MLPs predicting Re(u_scat) and Im(u_scat) separately."""
    def __init__(self):
        super().__init__()
        self.net_re = MLP()
        self.net_im = MLP()

    def forward(self, xy):
        return self.net_re(xy), self.net_im(xy)


model = PINN().to(DEVICE)
n_p   = sum(p.numel() for p in model.parameters())
print(f"Parameters : {n_p:,}")

# ─────────────────────────────────────────────────────────────────────────────
# Incident wave
# ─────────────────────────────────────────────────────────────────────────────
def u_inc_re(xy): return torch.cos(k * xy[:, 0:1])
def u_inc_im(xy): return torch.sin(k * xy[:, 0:1])

# ─────────────────────────────────────────────────────────────────────────────
# Collocation point samplers
# ─────────────────────────────────────────────────────────────────────────────
def sample_interior(n):
    """Uniform random points in [0,1]^2 outside the scatterer."""
    collected = []
    while sum(len(p) for p in collected) < n:
        xy = np.random.rand(n * 4, 2)
        r2 = (xy[:, 0] - cx)**2 + (xy[:, 1] - cy)**2
        xy = xy[r2 > (a + 0.008)**2]
        collected.append(xy)
    xy = np.vstack(collected)[:n].astype(np.float32)
    return torch.tensor(xy, device=DEVICE, requires_grad=True)


def sample_scatterer(n):
    """Points uniformly on the circle dD with outward normals."""
    theta = np.random.rand(n) * 2 * np.pi
    x  = cx + a * np.cos(theta);  y  = cy + a * np.sin(theta)
    nx = np.cos(theta);            ny = np.sin(theta)
    xy  = torch.tensor(np.stack([x, y], 1).astype(np.float32),  device=DEVICE, requires_grad=True)
    nxy = torch.tensor(np.stack([nx, ny], 1).astype(np.float32), device=DEVICE)
    return xy, nxy


def sample_outer(n_per_side):
    """Points on all 4 sides of [0,1]^2 with outward normals."""
    t   = np.random.rand(n_per_side)
    b   = np.stack([t, np.zeros(n_per_side)], 1);  nb = np.tile([0., -1.], (n_per_side, 1))
    tp  = np.stack([t, np.ones(n_per_side)],  1);  nt = np.tile([0.,  1.], (n_per_side, 1))
    l   = np.stack([np.zeros(n_per_side), t], 1);  nl = np.tile([-1., 0.], (n_per_side, 1))
    r   = np.stack([np.ones(n_per_side),  t], 1);  nr = np.tile([ 1., 0.], (n_per_side, 1))
    xy  = torch.tensor(np.vstack([b, tp, l, r]).astype(np.float32),   device=DEVICE, requires_grad=True)
    nxy = torch.tensor(np.vstack([nb, nt, nl, nr]).astype(np.float32), device=DEVICE)
    return xy, nxy

# ─────────────────────────────────────────────────────────────────────────────
# Autograd helpers
# ─────────────────────────────────────────────────────────────────────────────
def grad1(u, xy):
    g = torch.autograd.grad(u.sum(), xy, create_graph=True)[0]
    return g[:, 0:1], g[:, 1:2]

def laplacian(u, xy):
    ux, uy = grad1(u, xy)
    uxx = torch.autograd.grad(ux.sum(), xy, create_graph=True)[0][:, 0:1]
    uyy = torch.autograd.grad(uy.sum(), xy, create_graph=True)[0][:, 1:2]
    return uxx + uyy

def normal_deriv(u, xy, n):
    ux, uy = grad1(u, xy)
    return ux * n[:, 0:1] + uy * n[:, 1:2]

# ─────────────────────────────────────────────────────────────────────────────
# Loss function
# ─────────────────────────────────────────────────────────────────────────────
def loss_fn(xy_int, xy_sc, nsc, xy_out, nout):
    # PDE residual: (nabla^2 + k^2) u_scat = 0
    uR, uI = model(xy_int)
    lapR   = laplacian(uR, xy_int)
    lapI   = laplacian(uI, xy_int)
    l_pde  = ((lapR + k**2 * uR)**2 + (lapI + k**2 * uI)**2).mean()

    # Scatterer BC: u_scat = -u_inc on dD
    uRs, uIs = model(xy_sc)
    l_scat   = ((uRs + u_inc_re(xy_sc))**2 + (uIs + u_inc_im(xy_sc))**2).mean()

    # Absorbing BC: du/dn - ik u = 0  =>  duR/dn - k uI = 0,  duI/dn + k uR = 0
    uRo, uIo = model(xy_out)
    dnR      = normal_deriv(uRo, xy_out, nout)
    dnI      = normal_deriv(uIo, xy_out, nout)
    l_out    = ((dnR + k * uIo)**2 + (dnI - k * uRo)**2).mean()

    total = W_PDE * l_pde + W_SCAT * l_scat + W_OUT * l_out
    return total, l_pde, l_scat, l_out

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Adam optimiser
# ─────────────────────────────────────────────────────────────────────────────
optimizer = torch.optim.Adam(model.parameters(), lr=LR_ADAM)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=ADAM_EPOCHS, eta_min=1e-5)

loss_history = []
t0 = time.time()
print(f"\n--- Phase 1: Adam ({ADAM_EPOCHS} epochs) ---")
print(f"{'Epoch':>6} | {'Total':>9} | {'PDE':>9} | {'Scat':>9} | {'Outer':>9} | {'s':>5}")
print("-" * 58)

for ep in range(1, ADAM_EPOCHS + 1):
    # Resample collocation points every 250 epochs for better domain coverage
    if ep == 1 or ep % 250 == 0:
        xy_int       = sample_interior(N_INT)
        xy_sc, nsc   = sample_scatterer(N_SCAT)
        xy_out, nout = sample_outer(N_OUT // 4)

    optimizer.zero_grad()
    loss, lp, ls, lo = loss_fn(xy_int, xy_sc, nsc, xy_out, nout)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    loss_history.append(loss.item())

    if ep % 250 == 0 or ep == 1:
        print(f"{ep:>6} | {loss.item():>9.3e} | {lp.item():>9.3e} | "
              f"{ls.item():>9.3e} | {lo.item():>9.3e} | {time.time()-t0:>5.1f}")

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: L-BFGS fine-tuning
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n--- Phase 2: L-BFGS ({LBFGS_STEPS} steps) ---")
# Fix a rich, dense collocation set for L-BFGS (deterministic)
xy_int_lb        = sample_interior(N_INT)
xy_sc_lb, nsc_lb = sample_scatterer(N_SCAT)
xy_out_lb, no_lb = sample_outer(N_OUT // 4)

lbfgs = torch.optim.LBFGS(model.parameters(), lr=0.1,
                            max_iter=20, history_size=50,
                            line_search_fn='strong_wolfe')
step_count = [0]

def closure():
    lbfgs.zero_grad()
    loss, lp, ls, lo = loss_fn(xy_int_lb, xy_sc_lb, nsc_lb, xy_out_lb, no_lb)
    loss.backward()
    step_count[0] += 1
    loss_history.append(loss.item())
    if step_count[0] % 50 == 0:
        print(f"  step {step_count[0]:>3} | loss {loss.item():.4e} | "
              f"PDE {lp.item():.3e} | Scat {ls.item():.3e} | "
              f"Outer {lo.item():.3e} | {time.time()-t0:.1f}s")
    return loss

for _ in range(LBFGS_STEPS):
    lbfgs.step(closure)
    if step_count[0] >= LBFGS_STEPS:
        break

print(f"\nTotal training time: {time.time()-t0:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# Evaluate on a dense grid
# ─────────────────────────────────────────────────────────────────────────────
N_grid = 200
xg = np.linspace(0, 1, N_grid)
yg = np.linspace(0, 1, N_grid)
Xg, Yg = np.meshgrid(xg, yg)
R2    = (Xg - cx)**2 + (Yg - cy)**2
in_sc = R2 <= a**2

xy_ev = torch.tensor(
    np.stack([Xg.ravel(), Yg.ravel()], axis=1).astype(np.float32), device=DEVICE)

model.eval()
with torch.no_grad():
    uR_g, uI_g = model(xy_ev)

U_scat        = (uR_g.numpy().reshape(N_grid, N_grid) +
                 1j * uI_g.numpy().reshape(N_grid, N_grid))
U_scat[in_sc] = 0.0

U_inc         = np.exp(1j * k * Xg)
U_inc[in_sc]  = 0.0
U_total       = U_inc + U_scat

sigma_rayleigh = (np.pi**2 / 2.0) * ka**4 * a
print(f"\nPhysical parameters:")
print(f"  Wave number k       = {k:.4f} rad/m")
print(f"  Wavelength lambda   = {2*np.pi/k:.4f} m")
print(f"  Scatterer radius a  = {a:.4f} m")
print(f"  ka (Rayleigh param) = {ka:.4f}  (Rayleigh regime: ka << 1)")
print(f"  Rayleigh sigma      = {sigma_rayleigh:.4e} m")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: Field panels (same layout as FD script)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle(
    f"PINN (Plain MLP) — Rayleigh Scattering — Unit Square\n"
    f"k={k:.2f}, a={a:.3f}, ka={ka:.3f}  |  "
    f"Adam {ADAM_EPOCHS} ep + L-BFGS {LBFGS_STEPS} steps  |  "
    f"Plain MLP ({HIDDEN}x{NEURONS}, no embedding)",
    fontsize=12, fontweight='bold')

fields = [
    (U_inc.real,      "Incident field  Re(u_inc)",   "RdBu_r"),
    (U_scat.real,     "Scattered field Re(u_scat)",  "RdBu_r"),
    (U_total.real,    "Total field     Re(u_total)", "RdBu_r"),
    (np.abs(U_inc),   "Incident |u_inc|",            "viridis"),
    (np.abs(U_scat),  "Scattered |u_scat|",          "hot"),
    (np.abs(U_total), "Total field     |u_total|",   "inferno"),
]

for ax, (data, title, cmap) in zip(axes.flat, fields):
    lim  = np.percentile(np.abs(data[~in_sc]), 98)
    vmin = -lim if "Re" in title else 0.0
    im   = ax.imshow(data, origin='lower', extent=[0, 1, 0, 1],
                     cmap=cmap, vmin=vmin, vmax=lim, interpolation='bilinear')
    ax.add_patch(plt.Circle((cx, cy), a, color='cyan', lw=1.5,
                             fill=False, zorder=5, ls='--'))
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, shrink=0.85)

plt.tight_layout()
plt.savefig("figures/pec/fields/pec_mlp_fields.png", dpi=150, bbox_inches='tight')
print("Saved: figures/pec/fields/pec_mlp_fields.png")
# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: Training loss history
# ─────────────────────────────────────────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(9, 4))
ax3.semilogy(loss_history, color='steelblue', lw=0.8)
ax3.axvline(ADAM_EPOCHS, color='orange', ls='--', lw=1.5,
            label=f'Adam -> L-BFGS switch (step {ADAM_EPOCHS})')
ax3.set_xlabel("Optimiser step")
ax3.set_ylabel("Total loss (log scale)")
ax3.set_title("Plain MLP PINN Training Loss — Rayleigh Scattering")
ax3.legend(); ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/pec/training/pec_mlp_loss.png", dpi=150, bbox_inches='tight')
print("Saved: figures/pec/training/pec_mlp_loss.png")

plt.show()
print("\nDone.")