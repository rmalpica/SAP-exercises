import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from pathlib import Path
HERE = Path(__file__).resolve().parent
CHAPTER_DIR = HERE.parent
DATA = CHAPTER_DIR / "data"
OUTPUTS = CHAPTER_DIR / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

def get_b777_engine():
    this_dir = os.path.split(__file__)[0]

    nt = 12 * 11 * 8
    xt = np.loadtxt(os.path.join(this_dir, "b777_engine_inputs.dat")).reshape((nt, 3))
    yt = np.loadtxt(os.path.join(this_dir, "b777_engine_outputs.dat")).reshape((nt, 2))
    dyt_dxt = np.loadtxt(os.path.join(this_dir, "b777_engine_derivs.dat")).reshape(
        (nt, 2, 3)
    )

    xlimits = np.array([[0, 0.9], [0, 15], [0, 1.0]])

    return xt, yt, dyt_dxt, xlimits


def plot_b777_engine(xt, yt, model, scaler_X, scaler_Y, device='cpu'):
    """
    xt: (N,3) raw inputs [M, h, throttle]
    yt: (N,2) raw outputs [thrust, SFC]
    model: trained PyTorch model that expects scaled inputs and outputs scaled predictions
    scaler_X: StandardScaler used on inputs
    scaler_Y: StandardScaler used on outputs
    device: 'cpu' or 'cuda'
    """
    # grid definitions
    val_M = np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.85,0.9])
    val_h = np.array([0.0,0.6096,1.524,3.048,4.572,6.096,7.62,9.144,10.668,11.8872,13.1064])
    val_t = np.array([0.05,0.2,0.3,0.4,0.6,0.8,0.9,1.0])

    def get_pts(xt, yt, iy, ind_M=None, ind_h=None, ind_t=None):
        eps = 1e-5
        mask = np.ones(len(xt), bool)
        if ind_M is not None:
            mask &= np.abs(xt[:,0] - val_M[ind_M]) < eps
        if ind_h is not None:
            mask &= np.abs(xt[:,1] - val_h[ind_h]) < eps
        if ind_t is not None:
            mask &= np.abs(xt[:,2] - val_t[ind_t]) < eps
        xvar = xt[mask]
        yvar = yt[mask, iy]
        # scale for plotting
        if iy == 0:
            yvar = yvar / 1e6
        else:
            yvar = yvar / 1e-4
        if ind_M is None:
            return xvar[:,0], yvar
        if ind_h is None:
            return xvar[:,1], yvar
        return xvar[:,2], yvar

    num = 100
    lins_M = np.linspace(0.0, 0.9, num)
    lins_h = np.linspace(0.0, 13.1064, num)
    lins_t = np.linspace(0.05, 1.0, num)

    def make_grid(ind_M=None, ind_h=None, ind_t=None):
        Xg = np.zeros((num,3), dtype=np.float32)
        Xg[:,0] = lins_M if ind_M is None else val_M[ind_M]
        Xg[:,1] = lins_h if ind_h is None else val_h[ind_h]
        Xg[:,2] = lins_t if ind_t is None else val_t[ind_t]
        return Xg

    # prepare model
    model.eval()
    model.to(device)
    dtype = next(model.parameters()).dtype

    fig, axs = plt.subplots(6, 2, figsize=(15,25), gridspec_kw={'hspace':0.5})
    # set titles and labels
    axs[0,0].set(title=f"M={val_M[-2]}", xlabel="throttle", ylabel="thrust (×1e6 N)")
    axs[0,1].set(title=f"M={val_M[-2]}", xlabel="throttle", ylabel="SFC (×1e-3 N/N/s)")
    axs[1,0].set(title=f"M={val_M[-5]}", xlabel="throttle", ylabel="thrust (×1e6 N)")
    axs[1,1].set(title=f"M={val_M[-5]}", xlabel="throttle", ylabel="SFC (×1e-3 N/N/s)")
    axs[2,0].set(title=f"throttle={val_t[1]}", xlabel="altitude (km)", ylabel="thrust (×1e6 N)")
    axs[2,1].set(title=f"throttle={val_t[1]}", xlabel="altitude (km)", ylabel="SFC (×1e-3 N/N/s)")
    axs[3,0].set(title=f"throttle={val_t[-1]}", xlabel="altitude (km)", ylabel="thrust (×1e6 N)")
    axs[3,1].set(title=f"throttle={val_t[-1]}", xlabel="altitude (km)", ylabel="SFC (×1e-3 N/N/s)")
    axs[4,0].set(title=f"throttle={val_t[1]}", xlabel="Mach number", ylabel="thrust (×1e6 N)")
    axs[4,1].set(title=f"throttle={val_t[1]}", xlabel="Mach number", ylabel="SFC (×1e-3 N/N/s)")
    axs[5,0].set(title=f"throttle={val_t[-1]}", xlabel="Mach number", ylabel="thrust (×1e6 N)")
    axs[5,1].set(title=f"throttle={val_t[-1]}", xlabel="Mach number", ylabel="SFC (×1e-3 N/N/s)")

    ind_h_list = [4,7,10]
    ind_M_list = [3,6,11]
    colors = ['b','r','g','c','m']

    # plotting sections
    # throttle slices
    for k, ind_h in enumerate(ind_h_list):
        for row, ind_M in enumerate([-2,-5]):
            ax_th, ax_sfc = axs[row,0], axs[row,1]
            # raw data
            xtp, ytp = get_pts(xt, yt, 0, ind_M=ind_M, ind_h=ind_h)
            ax_th.plot(xtp, ytp, 'o'+colors[k])
            xtp, ytp = get_pts(xt, yt, 1, ind_M=ind_M, ind_h=ind_h)
            ax_sfc.plot(xtp, ytp, 'o'+colors[k])
            # grid inputs -> scaled -> model -> inverse scale
            Xg = make_grid(ind_M=ind_M, ind_h=ind_h)
            Xg_s = scaler_X.transform(Xg)
            with torch.no_grad():
                inp = torch.from_numpy(Xg_s).to(device, dtype=dtype)
                out_s = model(inp).cpu().numpy()
            out = scaler_Y.inverse_transform(out_s)
            ax_th.plot(lins_t, out[:,0]/1e6, colors[k])
            ax_sfc.plot(lins_t, out[:,1]/1e-4, colors[k])

    # altitude slices
    for k, ind_M in enumerate(ind_M_list):
        for idx, ind_t in enumerate([1,-1]):
            row = 2+idx
            ax_th, ax_sfc = axs[row,0], axs[row,1]
            xtp, ytp = get_pts(xt, yt, 0, ind_M=ind_M, ind_t=ind_t)
            ax_th.plot(xtp, ytp, 'o'+colors[k])
            xtp, ytp = get_pts(xt, yt, 1, ind_M=ind_M, ind_t=ind_t)
            ax_sfc.plot(xtp, ytp, 'o'+colors[k])
            Xg = make_grid(ind_M=ind_M, ind_t=ind_t)
            Xg_s = scaler_X.transform(Xg)
            with torch.no_grad():
                inp = torch.from_numpy(Xg_s).to(device, dtype=dtype)
                out_s = model(inp).cpu().numpy()
            out = scaler_Y.inverse_transform(out_s)
            ax_th.plot(lins_h, out[:,0]/1e6, colors[k])
            ax_sfc.plot(lins_h, out[:,1]/1e-4, colors[k])

    # Mach slices
    for k, ind_h in enumerate(ind_h_list):
        for idx, ind_t in enumerate([1,-1]):
            row = 4+idx
            ax_th, ax_sfc = axs[row,0], axs[row,1]
            xtp, ytp = get_pts(xt, yt, 0, ind_h=ind_h, ind_t=ind_t)
            ax_th.plot(xtp, ytp, 'o'+colors[k])
            xtp, ytp = get_pts(xt, yt, 1, ind_h=ind_h, ind_t=ind_t)
            ax_sfc.plot(xtp, ytp, 'o'+colors[k])
            Xg = make_grid(ind_h=ind_h, ind_t=ind_t)
            Xg_s = scaler_X.transform(Xg)
            with torch.no_grad():
                inp = torch.from_numpy(Xg_s).to(device, dtype=dtype)
                out_s = model(inp).cpu().numpy()
            out = scaler_Y.inverse_transform(out_s)
            ax_th.plot(lins_M, out[:,0]/1e6, colors[k])
            ax_sfc.plot(lins_M, out[:,1]/1e-4, colors[k])

    for k in range(2):
        legend_entries = []
        for ind_h in ind_h_list:
            legend_entries.append("h={}".format(val_h[ind_h]))
            legend_entries.append("")

        axs[k, 0].legend(legend_entries)
        axs[k, 1].legend(legend_entries)

        axs[k + 4, 0].legend(legend_entries)
        axs[k + 4, 1].legend(legend_entries)

        legend_entries = []
        for ind_M in ind_M_list:
            legend_entries.append("M={}".format(val_M[ind_M]))
            legend_entries.append("")

        axs[k + 2, 0].legend(legend_entries)
        axs[k + 2, 1].legend(legend_entries)

    #plt.tight_layout()
    plt.savefig(OUTPUTS / "plot-engine-perfo.png", dpi=800, transparent=False)
    plt.show()
