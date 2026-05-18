import elecboltz
import os, re
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.measure import find_contours


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
mpl.rcParams['figure.figsize'] = (8.0, 5.2)
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['lines.markersize'] = 20
mpl.rcParams['axes.formatter.useoffset'] = False

params = {
    'band_name': "Nd-LSCO",
    'a': 3.75,
    'b': 3.75,
    'c': 13.2,
    'energy_scale': 160,
    'band_params': {'mu': -0.82439881, 't': 1, 'tp': -0.13642799,
                    'tpp': 0.06816836, 'tz': 0.06512192},
    'resolution': 41,
    'periodic': 2,
    'domain_size': [1.0, 1.0, 2.0],
    'Bamp': 45,
    'scattering_kernel_name': 'cylindrical',
    'scattering_kernel_params': {
        'constant': 5.5, 'cos4': 1.0, 'cos4cos4': 3.0
    },
    'march_square': True
}

label_to_latex = {
    "tp": "t'",
    "tpp": "t''",
    "tppp": "t'''",
    "tz": "t_z",
    "tzp": "t_z'",
    "tzpp": "t_z''",
    "tzppp": "t_z'''",
    "gamma_0": "\\Gamma_0",
    "gamma_k": "\\Gamma_k",
    "power": "\\nu",
    "mu": "\\mu"
}

kernel_names = {
    'cylindrical': "Cylindrical\nHarmonics",
    'spherical': "Spherical\nHarmonics",
    'legendre': "Legendre\nPolynomials",
}


def print_band_params(ax, renderer, params, name=None, xshift=0.22,
                      electron_doped=False):
    band = elecboltz.BandStructure(**elecboltz.easy_params(params))
    ax.axis('off')
    ypos = 1.0

    if electron_doped:
        header = f'$n = {band.calculate_filling_fraction():.3g}$'
    else:
        header = f'$p = {1-band.calculate_filling_fraction():.3g}$'
    if name is not None:
        header = f'{name}\n{header}'
    t = ax.text(xshift, ypos, header, color="mediumblue",
                fontsize='large', ha='left', va='top', transform=ax.transAxes)
    bbox = t.get_window_extent(renderer=renderer)
    ypos -= ax.transAxes.inverted().transform_bbox(bbox).height + 0.05

    t = ax.text(xshift, ypos, "Lattice Dimensions", color="darkturquoise",
                fontsize='small', ha='left', va='top', transform=ax.transAxes)
    bbox = t.get_window_extent(renderer=renderer)
    ypos -= ax.transAxes.inverted().transform_bbox(bbox).height + 0.02
    
    params_text = '\n'.join(fr"${label}={value}$ Å" for label, value
                            in zip(band.axis_names, band.unit_cell))
    t = ax.text(xshift, ypos, params_text, fontsize='x-small',
            ha='left', va='top', transform=ax.transAxes)
    bbox = t.get_window_extent(renderer=renderer)
    ypos -= ax.transAxes.inverted().transform_bbox(bbox).height + 0.02

    t = ax.text(xshift, ypos, "Band Parameters", color="darkturquoise",
                fontsize='small', ha='left', va='top', transform=ax.transAxes)
    bbox = t.get_window_extent(renderer=renderer)
    ypos -= ax.transAxes.inverted().transform_bbox(bbox).height + 0.02

    params_text = f"$t={params['energy_scale']:.4g}$ meV"
    for (label, value) in params['band_params'].items():
        if label != 't':
            value /= params['band_params']['t']
            label = label_to_latex.get(label, label)
            if value != 0:
                params_text += f"\n${label}={value:.4g}t$"
            else:
                params_text += f"\n${label}=0$"
    ax.text(xshift, ypos, params_text, color="black", fontsize='x-small',
            ha='left', va='top', transform=ax.transAxes)


def print_kernel_params(ax, renderer, params, xshift=0.0):
    ax.axis('off')
    ypos = 1.0

    t = ax.text(
        xshift, ypos, kernel_names.get(
            params['scattering_kernel_name'],
            params['scattering_kernel_name']),
        color="mediumblue", fontsize='large', ha='left', va='top',
        transform=ax.transAxes)
    bbox = t.get_window_extent(renderer=renderer)
    ypos -= ax.transAxes.inverted().transform_bbox(bbox).height + 0.05

    t = ax.text(
        xshift, ypos, "Coefficients (Å$^2$ THz)", color="darkturquoise",
        fontsize='small', ha='left', va='top', transform=ax.transAxes)
    bbox = t.get_window_extent(renderer=renderer)
    ypos -= ax.transAxes.inverted().transform_bbox(bbox).height + 0.005

    params_text = ''
    for (label, value) in params['scattering_kernel_params'].items():
        if value != 0:
            coeff_label = _get_scattering_kernel_coeff_label(
                params['scattering_kernel_name'], label)
            params_text += f"{coeff_label}: {value:.3g}"
            params_text += '\n'
    if params_text.endswith('\n'):
        params_text = params_text[:-1]
    ax.text(xshift, ypos, params_text, color="black", fontsize='x-small',
            ha='left', va='top', transform=ax.transAxes)


def _get_scattering_kernel_coeff_label(kernel_name, label):
    if kernel_name == 'cylindrical':
        if isinstance(label, str):
            label = elecboltz.kernel._get_cylindrical_indices(label)
        return _get_cylindrical_coeff_label(label)
    elif kernel_name == 'spherical':
        return f'$Y^{{{label[0][1]}}}_{{{label[0][0]}}}' \
               f'Y^{{{label[1][1]}}}_{{{label[1][0]}}}$'
    elif kernel_name == 'legendre':
        return f'$P_{{{label[0]}}}P_{{{label[1]}}}$'
    else:
        return label


def _get_cylindrical_coeff_label(label):
    m, m_prime = label
    if m == 0 and m_prime == 0:
        return 'const.'
    elif m_prime == 0:
        trig = 'cos' if m > 0 else 'sin'
        if abs(m) == 1:
            return f'${trig}(\\varphi)$'
        else:
            return f'${trig}({abs(m)}\\varphi)$'
    elif m == 0:
        trig = 'cos' if m_prime > 0 else 'sin'
        if abs(m_prime) == 1:
            return f'${trig}(\\varphi)$'
        else:
            return f'${trig}({abs(m_prime)}\\varphi)$'
    else:
        trig1 = 'cos' if m > 0 else 'sin'
        trig2 = 'cos' if m_prime > 0 else 'sin'
        if abs(m) == 1:
            trig1_str = f"${trig1}(\\varphi)$"
        else:
            trig1_str = f"${trig1}({abs(m)}\\varphi)$"
        if abs(m_prime) == 1:
            trig2_str = f"${trig2}(\\varphi')$"
        else:
            trig2_str = f"${trig2}({abs(m_prime)}\\varphi')$"
        return f'{trig1_str}{trig2_str}'


def plot_full_fermi_surface_slices(ax, renderer, params, n_interp=300, res=101,
                                   **kwargs):
    band = elecboltz.BandStructure(**elecboltz.easy_params(params))
    ax.set_xlim(-band.domain_size[0] * np.pi / band.unit_cell[0],
                band.domain_size[0] * np.pi / band.unit_cell[0])
    ax.set_ylim(-band.domain_size[1] * np.pi / band.unit_cell[1],
                band.domain_size[1] * np.pi / band.unit_cell[1])
    ax.set_aspect('equal')

    divider = make_axes_locatable(ax)
    ax_text = divider.append_axes("right", size="5%", pad=0.1)
    ax_text.axis('off')

    colors = ['lime', 'red', 'blue']
    labels = [r'$0$', r'$\frac{3\pi}{2c}$', r'$\frac{3\pi}{c}$']
    kzs = [0.0, 1.5 * np.pi / band.unit_cell[2],
           3.0 * np.pi / band.unit_cell[2]]
    ypos = 0.0
    for kz, color, label in zip(kzs, colors, labels):
        plot_fermi_surface_slice(ax, band, kz=kz, n_interp=n_interp,
                                 res=res, color=color, **kwargs)
        t = ax_text.text(0.0, ypos, label, color=color, fontsize='small',
                    ha='left', va='bottom', transform=ax_text.transAxes)
        bbox = t.get_window_extent(renderer=renderer)
        ypos += ax_text.transAxes.inverted().transform_bbox(bbox).height + 0.03
    ax_text.text(0.0, ypos, '$k_z$', color='black', fontsize='small',
            ha='left', va='bottom', transform=ax_text.transAxes)
    ax.tick_params(axis='both', labelsize='x-small')
    ax.set_xlabel("$k_x$ (Å$^{-1}$)", fontsize='x-small', labelpad=2)
    ax.set_ylabel("$k_y$ (Å$^{-1}$)", fontsize='x-small', labelpad=2)


def plot_fermi_surface_slice(ax, band, kz=0.0, n_interp=300, res=101,
                             **kwargs):
    gvec = band.domain_size * np.pi / band.unit_cell
    res = res or band.resolution
    if isinstance(res, (int, float)):
        res = [res, res]

    kgrid = np.mgrid[-gvec[0]:gvec[0]:1j*res[0], -gvec[1]:gvec[1]:1j*res[1]]
    contours = find_contours(
        band.energy_func(kgrid[0], kgrid[1], kz),
        band.chemical_potential)
    kxs, kys = _get_interpolated_contours_in_order(contours, res, n_interp)
    for i in range(len(kxs)):
        kxs[i] = kxs[i] * 2 * gvec[0] / res[0] - gvec[0]
        kys[i] = kys[i] * 2 * gvec[1] / res[1] - gvec[1]
    for kx, ky in zip(kxs, kys):
        ax.plot(kx, ky, **kwargs)

def plot_scattering_heatmap(fig, ax, params, res=101, **kwargs):
    thetas = np.linspace(-np.pi, np.pi, res)
    kx, ky = np.cos(thetas), np.sin(thetas)
    kz = np.zeros_like(kx)
    kernel_values = np.zeros((len(thetas), len(thetas)))
    kernel = elecboltz.easy_params(params)['scattering_kernel']
    for i in range(kernel.coeffs.shape[0]):
        basis_vec_i = kernel.eval_basis(i, kx, ky, kz)
        for j in range(kernel.coeffs.shape[0]):
            basis_vec_j = kernel.eval_basis(j, kx, ky, kz)
            kernel_values += kernel.coeffs[i, j] * np.outer(
                basis_vec_i, basis_vec_j)

    heatmap = ax.imshow(kernel_values, extent=(-180, 180, -180, 180),
                        origin='lower', aspect='equal', **kwargs)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(heatmap, cax=cax,
                        label=r"$C(\varphi, \varphi')$ (Å$^2$ THz)")
    cbar.ax.tick_params(labelsize='xx-small')
    cbar.ax.set_ylabel(cbar.ax.get_ylabel(), fontsize='xx-small')
    ax.tick_params(axis='both', labelsize='x-small')
    ax.set_xlabel(r"$\varphi$ ($^\circ$)", fontsize='x-small', labelpad=2)
    ax.set_ylabel(r"$\varphi'$ ($^\circ$)", fontsize='x-small', labelpad=2)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_yticks([-180, -90, 0, 90, 180])

def _get_interpolated_contours_in_order(contours, res, n_interp):
    kxs = []
    kys = []
    contour_idx = 0
    while contours:
        contour = contours.pop(contour_idx)
        contour = _interpolate_contour(contour, n_interp)
        kxs.append(contour[:, 0])
        kys.append(contour[:, 1])
        # Find closest contour
        min_distance_squared = np.inf
        for (i, neighboring_contour) in enumerate(contours):
            delta_k = np.abs(contour[-1,:] - neighboring_contour[0,:])
            # periodic boundary conditions, in array units
            delta_k[0] %= res[0]
            delta_k[1] %= res[1]
            distance_squared = delta_k[0]**2 + delta_k[1]**2
            if distance_squared < min_distance_squared:
                min_distance_squared = distance_squared
                contour_idx = i
    return kxs, kys


def _interpolate_contour(contour, n_interp):
    dk = np.diff(contour, axis=0)
    # line segment lengths
    ds = np.linalg.norm(dk, axis=1)
    # parametrization of the curve, which is the length along the curve
    s = np.concatenate(([0], np.cumsum(ds)))
    interpolated_s = np.linspace(0, s[-1], n_interp + 1)
    interpolated_contour = np.column_stack((
        np.interp(interpolated_s, s, contour[:, 0]),
        np.interp(interpolated_s, s, contour[:, 1])))
    return interpolated_contour


def plot_info(fig, name=None, electron_doped=False, n_interp=300, res=101):
    prev_font = mpl.rcParams['font.size']
    mpl.rcParams['font.size'] = 16  # smaller font for the plots
    gs = fig.add_gridspec(2, 3)
    ax_fermi = fig.add_subplot(gs[0, 0])
    ax_scat_heat = fig.add_subplot(gs[1, 0])
    ax_band_params = fig.add_subplot(gs[:, 1])
    ax_scat_params = fig.add_subplot(gs[:, 2])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    plot_full_fermi_surface_slices(ax_fermi, renderer, params,
                                   n_interp=n_interp, res=res, lw=1)
    plot_scattering_heatmap(fig, ax_scat_heat, params, res=res, cmap='plasma')
    print_band_params(ax_band_params, renderer, params,
                      name=name, electron_doped=electron_doped)
    print_kernel_params(ax_scat_params, renderer, params)

    mpl.rcParams['font.size'] = prev_font  # restore original font size


def calc_and_plot_admr(ax, params):
    params = elecboltz.easy_params(params)
    band = elecboltz.BandStructure(**params)
    band.discretize()
    cond = elecboltz.Conductivity(band, **params)
    Bthetas = np.linspace(-30, 150, 100)
    Bphis = [0, 15, 30, 45]
    rho_zz = np.zeros((len(Bphis), len(Bthetas)))
    for i, Bphi in enumerate(Bphis):
        cond.Bphi = Bphi
        for j, Btheta in enumerate(Bthetas):
            cond.Btheta = Btheta
            rho_zz[i, j] = np.linalg.inv(cond.calculate())[2, 2]
    
    cond.Bphi = 0
    cond.Btheta = 0
    rho_zz_0 = np.linalg.inv(cond.calculate())[2, 2]

    for i, phi in enumerate(Bphis):
        ax.plot(Bthetas, rho_zz[i] / rho_zz_0, label=f'${phi}^\\circ$',
                color=mpl.cm.Blues((i+1) / len(Bphis)))
    ax.set_xlabel(r'$\theta$ ($^\circ$)')
    ax.set_ylabel(r'$\rho_{zz}(\theta)/\rho_{zz}(0)$')
    ax.set_title('Nd-LSCO')
    ax.legend(frameon=False, title=r"$\phi$")
    ax.set_xticks([-30, 0, 30, 60, 90, 120, 150])


def plot_kernel_admr(params, show=False):
    fig1, ax = plt.subplots()
    calc_and_plot_admr(ax, params)
    fig1.tight_layout()

    fig2 = plt.figure(layout='constrained')
    fig2.get_layout_engine().set(rect=(0, 0.02, 1.0, 0.94))
    plot_info(fig2, name="Nd-LSCO", electron_doped=False)
    
    coeffs_text = ''
    for label, value in sorted(params['scattering_kernel_params'].items()):
        if value == np.floor(value):
            value = int(value)
        if label != 'constant' and label != '1':
            if value != 1:
                coeffs_text += str(value)
            coeffs_text += label
        else:
            coeffs_text += str(value)
        coeffs_text += '+'
    # remove extra + at the end
    coeffs_text = coeffs_text[:-1]

    with PdfPages(os.path.dirname(os.path.relpath(__file__))
                  + f'/Kernel/ADMR_NdLSCO_coeffs={coeffs_text}.pdf') as pdf:
        pdf.savefig(fig1)
        pdf.savefig(fig2)
    if show:
        plt.show()
    else:
        plt.close(fig1)
        plt.close(fig2)


def main():
    plot_kernel_admr(params)


if __name__ == "__main__":
    main()
