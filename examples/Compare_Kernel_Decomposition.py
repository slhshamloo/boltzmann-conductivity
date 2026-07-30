import elecboltz
import os
import scipy
import numpy as np
from scipy.constants import m_e, hbar, electron_volt
from skimage.measure import find_contours
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# matplotlib settings
# reset defaults
mpl.rcdefaults()
# font
mpl.rcParams['font.size'] = 16
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['xtick.major.width'] = 1.0
mpl.rcParams['ytick.major.width'] = 1.0
mpl.rcParams['pdf.fonttype'] = 3
# plotting
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['lines.markersize'] = 10
mpl.rcParams['axes.formatter.useoffset'] = False


params = {
    'a': 3.75,
    'b': 3.75,
    'c': 13.2,
    'energy_scale': 160,
    'band_params': {'mu': -0.82439881, 't': 1, 'tp': -0.13642799,
                    'tpp': 0.06816836, 'tz': 0.06512192},
    'resolution': 41,
    'periodic': 2,
    'domain_size': [1.0, 1.0, 2.0],
}


def calc_kernel_values(kernel, low_res=31, n_interp=201):
    band = elecboltz.BandStructure(**elecboltz.easy_params(params))
    band.discretize()
    params_low = params.copy()
    params_low['resolution'] = low_res
    band_low = elecboltz.BandStructure(**elecboltz.easy_params(params_low))
    band_low.discretize()
    kernel.decompose(band)
    if isinstance(kernel, elecboltz.kernel.SumKernel):
        kernel = kernel.custom_kernel

    phi = np.linspace(-np.pi, np.pi, n_interp)
    kx, ky = _get_interpolated_kx_ky(
        band.kpoints[:, 0], band.kpoints[:, 1], phi)
    kz = np.zeros_like(kx)
    kernel_values = kernel.kernel_func(
        kx[:, None], ky[:, None], kz[:, None],
        kx[None, :], ky[None, :], kz[None, :])

    kernel_decomposed = \
        kernel.eigenvectors @ kernel.coeffs @ kernel.eigenvectors.T
    # get points roughly around the kz=0 plane
    kz_low = band_low.kpoints[:, 2]
    idx = np.where(np.abs(kz_low) < 1e-3)[0]
    kx_low, ky_low = band_low.kpoints[idx, 0], band_low.kpoints[idx, 1]
    kernel_decomposed = kernel_decomposed[np.ix_(idx, idx)]
    kernel_decomposed = _interpolate_to_phi(
        kernel_decomposed, kx_low, ky_low, phi)
    
    return kernel_values, kernel_decomposed


def calc_kernel_values_gaussian(
        C_f0=5.5, C_f1=0.0, sigma_f0=0.1, sigma_f1=0.0, m=1, delta=0.0,
        rank=20, low_res=31, n_interp=201):
    kernel = elecboltz.kernel.CustomKernel(
        {'kernel_func': elecboltz.kernel.AnisotropicGaussianScattering(
            C_f0, C_f1, sigma_f0, sigma_f1, m, delta),
         'rank': rank, 'low_res': low_res})
    return calc_kernel_values(kernel, low_res=low_res, n_interp=n_interp)


def calc_kernel_values_spinfluctuation(
        C_s=1, xi=1, rank=20, low_res=31, n_interp=201):
    params['scattering_kernel_names'] = ['spin_fluctuation']
    params['scattering_kernel_params'] = [
        {'C_s': C_s, 'xi': xi, 'Qratio': [np.pi, np.pi, 0]},
        {'rank': rank, 'low_res': low_res}]
    kernel = elecboltz.easy_params(params)['scattering_kernel']
    return calc_kernel_values(kernel, low_res=low_res, n_interp=n_interp)


def calc_kernel_values_azimuthal(
        C_0=5.5, C_1=0.0, m=1, nu=1.0, rank=20, low_res=31, n_interp=201):
    kernels = [elecboltz.kernel.IsotropicKernelFunction(C_0),
               elecboltz.kernel.AzimuthalKernelFunction(C_1, m, nu)]
    kernel = elecboltz.kernel.CustomKernel(
        {'kernel_func': elecboltz.kernel.SumKernelFunction(kernels),
         'rank': rank, 'low_res': low_res})
    return calc_kernel_values(kernel, low_res=low_res, n_interp=n_interp)


def plot_comparison(kernel_values, kernel_decomposed,
                    title, figsize=(10.0, 4.5)):
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    im0 = axs[0].imshow(
        kernel_values, extent=(-180, 180, -180, 180),
        origin='lower', cmap='plasma')
    axs[0].set_title("Original Kernel")
    axs[0].set_xlabel("$\\phi$ ($^\\circ$)")
    axs[0].set_ylabel("$\\phi'$ ($^\\circ$)")
    divider0 = make_axes_locatable(axs[0])
    cax0 = divider0.append_axes("right", size="5%", pad=0.08)
    fig.colorbar(im0, cax=cax0)

    im1 = axs[1].imshow(
        kernel_decomposed, extent=(-180, 180, -180, 180),
        origin='lower', cmap='plasma')
    axs[1].set_title("Decomposed Kernel")
    axs[1].set_xlabel("$\\phi$ ($^\\circ$)")
    axs[1].set_ylabel("$\\phi'$ ($^\\circ$)")
    divider1 = make_axes_locatable(axs[1])
    cax1 = divider1.append_axes("right", size="5%", pad=0.08)
    fig.colorbar(im1, cax=cax1)

    fig.suptitle(title)
    axs[0].set_xticks([-180, -90, 0, 90, 180])
    axs[1].set_xticks([-180, -90, 0, 90, 180])
    axs[0].set_yticks([-180, -90, 0, 90, 180])
    axs[1].set_yticks([-180, -90, 0, 90, 180])

    fig.tight_layout(pad=0.3)
    return fig, axs


def compare_decomposition_azimuthal(C_0=5.5, C_1=50, m=2, nu=10, rank=20):
    kernel_values, kernel_decomposed = calc_kernel_values_azimuthal(
        C_0=C_0, C_1=C_1, m=m, nu=nu, rank=rank)
    fig, _ = plot_comparison(
        kernel_values, kernel_decomposed,
        title="$C(\\mathbf{k}, \\mathbf{k}') = C_0 + \\left|\\mathrm{cos}"
              f"(\\phi+\\phi')\\right|^\\nu$, $C_0 = {C_0}$, "
              f"$C_1 = {C_1}$, $\\nu = {nu}$, $\\mathrm{{rank}} = {rank}$",
        figsize=(10.0, 4.5))
    fig.savefig(os.path.dirname(os.path.relpath(__file__))
                + "/Kernel/Decomposition Comparison/"
                + f"NdLSCO_C0={C_0}_C1={C_1}_nu={nu}_rank={rank}.pdf",
                bbox_inches='tight')
    plt.show()


def compare_decomposition_gaussian(sigma_f0=0.5, rank=50):
    kernel_values, kernel_decomposed = calc_kernel_values_gaussian(
        sigma_f0=sigma_f0, rank=rank)
    fig, _ = plot_comparison(
        kernel_values, kernel_decomposed,
        title="$C(\\mathbf{k}, \\mathbf{k}') = C_f \\mathrm{exp}\\left("
              "-\\frac{|\\phi-\\phi'|^2}{2\\sigma_f^2}\\right)$, $C_f=5.5$, "
              f"$\\sigma_f = {sigma_f0}$, $\\mathrm{{rank}} = {rank}$",
        figsize=(10.0, 4.5))
    fig.savefig(os.path.dirname(os.path.relpath(__file__))
                + "/Kernel/Decomposition Comparison/"
                + f"NdLSCO_sigmaf={sigma_f0}_rank={rank}.pdf",
                bbox_inches='tight')
    plt.show()


def compare_decomposition_anisotropic(
        C_f0=5.5, C_f1=2.0, sigma_f0=0.5, sigma_f1=0.3, rank=20):
    kernel_values, kernel_decomposed = calc_kernel_values_gaussian(
        C_f0=C_f0, C_f1=C_f1, sigma_f0=sigma_f0, sigma_f1=sigma_f1,
        m=4, rank=rank)
    fig, _ = plot_comparison(
        kernel_values, kernel_decomposed,
        title="$C(\\mathbf{k}, \\mathbf{k}') = C_f \\mathrm{exp}"
              "\\left(-\\frac{|\\phi-\\phi'|^2}{2\\sigma_f^2}\\right)$, "
              f"$C_f = {C_f0} + {C_f1}\\mathrm{{cos}}"
              "\\left(\\frac{\\phi+\\phi'}{2}\\right)$,\n"
              f"$\\sigma_f = {sigma_f0} + {sigma_f1}\\mathrm{{cos}}"
              f"\\left(\\frac{{\\phi+\\phi'}}{{2}}\\right)$, "
              f"$\\mathrm{{rank}} = {rank}$",
        figsize=(10.0, 6.5))
    fig.savefig(os.path.dirname(os.path.relpath(__file__))
                + "/Kernel/Decomposition Comparison/"
                + f"Cf0={C_f0}_Cf1={C_f1}_sigmaf0={sigma_f0}"
                + f"_sigmaf1={sigma_f1}_rank={rank}.pdf",
                bbox_inches='tight')
    plt.show()


def compare_decomposition_spinfluctuation(
        C_s=1, xi=1, low_res=31, rank=20, n_interp=201):
    kernel_values, kernel_decomposed = calc_kernel_values_spinfluctuation(
        C_s=C_s, xi=xi, low_res=low_res, rank=rank, n_interp=n_interp)
    fig, _ = plot_comparison(
        kernel_values, kernel_decomposed,
        title="$C(\\mathbf{k}, \\mathbf{k}') = "
              "\\frac{C_s}{1 + \\xi^2 |\\mathbf{k}-\\mathbf{k}'-\\mathbf{Q}|^2}$, "
              f"$C_s = {C_s}$, $\\xi = {xi}$, $\\mathrm{{rank}} = {rank}$",
        figsize=(10.0, 4.5))
    fig.savefig(os.path.dirname(os.path.relpath(__file__))
                + "/Kernel/Decomposition Comparison/"
                + f"NdLSCO_Cs={C_s}_xi={xi}_rank={rank}.pdf",
                bbox_inches='tight')
    plt.show()


def _get_interpolated_kx_ky(kx, ky, phi_interp):
    phi = np.arctan2(kx, ky)
    idx = np.argsort(phi)
    phi_fs = phi[idx]
    kx_fs, ky_fs = kx[idx], ky[idx]

    # avoid floating point precision issues
    phi_fs = np.round(phi_fs, decimals=8)
    phi_fs, idx = np.unique(phi_fs, return_index=True)
    kx_fs, ky_fs = kx_fs[idx], ky_fs[idx]

    r = np.sqrt(kx_fs**2 + ky_fs**2)
    phi_periodic = np.concatenate([phi_fs - 2*np.pi, phi_fs, phi_fs + 2*np.pi])
    r_periodic = np.concatenate([r, r, r])

    r_interp = np.interp(phi_interp, phi_periodic, r_periodic)
    return r_interp * np.cos(phi_interp), r_interp * np.sin(phi_interp)


def _interpolate_to_phi(values, kx, ky, phi_interp):
    phi = np.arctan2(ky, kx)
    # avoid floating point precision issues
    phi = np.round(phi, decimals=8)
    phi, idx = np.unique(phi, return_index=True)
    values = values[np.ix_(idx, idx)]

    idx = np.argsort(phi)
    phi = phi[idx]
    values = values[np.ix_(idx, idx)]
    interp = scipy.interpolate.RectBivariateSpline(phi, phi, values)
    return interp(phi_interp, phi_interp)


if __name__ == "__main__":
    compare_decomposition_spinfluctuation(C_s=1, xi=10, rank=70, n_interp=201)