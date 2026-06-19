import elecboltz
import os
import scipy, sympy
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.measure import find_contours
from copy import copy


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
    'scattering_kernel_names': ['isotropic', 'forward_anisotropic'],
    'scattering_kernel_params': [
        {'C_0': 2.5},
        {'m': 2, 'nu_fc': 20.0, 'C_f1': 300.0, 'sigma_f0': 0.3},
        {'rank': 20, 'low_res': 21}],
}

label_to_latex = {
    "tp": "t'",
    "tpp": "t''",
    "tppp": "t'''",
    "tz": "t_z",
    "tzp": "t_z'",
    "tzpp": "t_z''",
    "tzppp": "t_z'''",
    "sigma_f": "\\sigma_f",
    "sigma_b": "\\sigma_b",
    "C_f0": "C_{f0}",
    "C_f1": "C_{f1}",
    "sigma_f0": "\\sigma_{f0}",
    "sigma_f1": "\\sigma_{f1}",
    "C_b0": "C_{b0}",
    "C_b1": "C_{b1}",
    "sigma_b0": "\\sigma_{b0}",
    "sigma_b1": "\\sigma_{b1}",
    "phi_fc": "\\varphi_{fc}",
    "phi_fs": "\\varphi_{fs}",
    "phi_bc": "\\varphi_{bc}",
    "phi_bs": "\\varphi_{bs}",
    "nu_fc": "\\nu_{fc}",
    "nu_fs": "\\nu_{fs}",
    "nu_bc": "\\nu_{bc}",
    "nu_bs": "\\nu_{bs}"
}


def get_scattering_latex(name, params):
    if name == 'isotropic':
        return "C_0"
    elif name == 'forward':
        return "C_f \\mathrm{exp}\\left(" \
               "\\frac{-|\\mathbf{k}-\\mathbf{k'}|^2}{2\\sigma_f^2}\\right)"
    elif name == 'backward':
        return "C_b \\mathrm{exp}\\left(" \
               "\\frac{-|\\mathbf{k}+\\mathbf{k'}|^2}{2\\sigma_b^2}\\right)"
    elif name == 'forward_phi':
        return "C_f \\mathrm{exp}\\left(-\\frac{|\\phi-\\phi'|^2}" \
               "{2\\sigma_f^2}\\right)"
    elif name == 'backward_phi':
        return "C_b \\mathrm{exp}\\left(-\\frac{(|\\phi-\\phi'|-\\pi)^2}" \
               "{2\\sigma_b^2}\\right)"
    elif name == 'forward_anisotropic':
        if params.get('C_f0', 0) != 0:
            text = "C_{f1} \\left|\\mathrm{cos}\\left(" \
                f"{params['m']}\\left[\\frac{{\\varphi+\\varphi'}}{{2}}"
            if params.get('phi_fc', 0) != 0:
                text += "+\\varphi_{fc}"
            text += "\\right]\\right)\\right|^{\\nu_{fc}}"
        else:
            text = "\\left(C_{f0} + C_{f1} \\left|\\mathrm{cos}\\left(" \
                f"{params['m']}\\left[\\frac{{\\varphi+\\varphi'}}{{2}}"
            if params.get('phi_fc', 0) != 0:
                text += "+\\varphi_{fc}"
            text += "\\right]\\right)\\right|^{\\nu_{fc}}\\right)"
        text += "\\mathrm{exp}\\left(-\\frac{|\\varphi-\\varphi'|^2}"
        if params.get('sigma_b1', 0) != 0:
            return (text +
                "{2\\left(\\sigma_{f0}+\\sigma_{f1}\\left|\\mathrm{cos}" \
                f"\\left({params['m']}\\left[\\frac" \
                "{\\varphi+\\varphi'}{2}-\\varphi_{fs}\\right]\\right)"\
                "\\right|^{\\nu_{fs}}\\right)}\\right)")
        else:
            return text + "{2\\sigma_{f0}^2}\\right)"
    elif name == 'backward_anisotropic':
        if params.get('C_b0', 0) != 0:
            text = "C_{b1} \\left|\\mathrm{cos}\\left(" \
                f"{params['m']}\\left[\\frac{{\\varphi+\\varphi'}}{{2}}"
            if params.get('phi_bc', 0) != 0:
                text += "+\\varphi_{bc}"
            text += "\\right]\\right)\\right|^{\\nu_{bc}}"
        else:
            text = "\\left(C_{b0} + C_{b1} \\left|\\mathrm{cos}\\left(" \
                f"{params['m']}\\left[\\frac{{\\varphi+\\varphi'}}{{2}}"
            if params.get('phi_bc', 0) != 0:
                text += "+\\varphi_{bc}"
            text += "\\right]\\right)\\right|^{\\nu_{bc}}\\right)"
        text += "\\mathrm{exp}\\left(-\\frac{|\\varphi-\\varphi'|^2}"
        if params.get('sigma_b1', 0) != 0:
            return (text +
                "{2\\left(\\sigma_{b0}+\\sigma_{b1}\\left|\\mathrm{cos}" \
                f"\\left({params['m']}\\left[\\frac" \
                "{\\varphi+\\varphi'}{2}-\\varphi_{bs}\\right]\\right)"\
                "\\right|^{\\nu_{bs}}\\right)}\\right)")
        else:
            return text + "{2\\sigma_{b0}^2}\\right)"
    else:
        raise ValueError(f"Unknown scattering kernel name: {name}")


def print_params_left(ax, renderer, params, name=None, xshift=0.22,
                      electron_doped=False):
    band = elecboltz.BandStructure(**elecboltz.easy_params(params))
    band.discretize()
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

    params_y = ypos
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
    
    return params_y


def print_params_right(ax, renderer, params, params_y=None, xshift=0.05):
    band = elecboltz.BandStructure(**elecboltz.easy_params(params))
    ax.axis('off')
    ypos = 0.98

    t = ax.text(xshift, ypos, "Lattice Dimensions", color="darkturquoise",
                fontsize='small', ha='left', va='top', transform=ax.transAxes)
    bbox = t.get_window_extent(renderer=renderer)
    ypos -= ax.transAxes.inverted().transform_bbox(bbox).height + 0.02
    
    params_text = ", ".join(fr"${label}={value}$ Å" for label, value
                            in zip(band.axis_names, band.unit_cell))
    t = ax.text(xshift, ypos, params_text, fontsize='x-small',
            ha='left', va='top', transform=ax.transAxes)
    bbox = t.get_window_extent(renderer=renderer)
    ypos -= ax.transAxes.inverted().transform_bbox(bbox).height + 0.05

    ypos = params_y or ypos
    t = ax.text(xshift, ypos, "Scattering Parameters", color="darkturquoise",
                fontsize='small', ha='left', va='top', transform=ax.transAxes)
    bbox = t.get_window_extent(renderer=renderer)
    ypos -= ax.transAxes.inverted().transform_bbox(bbox).height + 0.005

    params_text = ''
    for i, kernel in enumerate(params['scattering_kernel_names']):
        for (label, value) in params['scattering_kernel_params'][i].items():
            label_latex = label_to_latex.get(label, label)
            params_text += f"${label_latex}={value:.3g}$"
            if label_latex.startswith('C_'):
                params_text += " Å$^2$ THz"
            elif label.startswith('sigma'):
                if kernel in ['forward_phi', 'backward_phi',
                              'forward_anisotropic', 'backward_anisotropic']:
                    params_text += " rad"
                else:
                    params_text += " Å$^{-1}$"
            elif label.startswith('phi'):
                params_text += "$^\\circ$"
            params_text += '\n'
    ax.text(xshift, ypos, params_text, color="black", fontsize='x-small',
            ha='left', va='top', transform=ax.transAxes)


def print_equations(ax, renderer, params, xshift=0.1):
    band = elecboltz.BandStructure(**elecboltz.easy_params(params))
    ax.axis('off')
    ypos = 1.0

    t = ax.text(xshift, ypos, "Dispersion Relation", color="darkturquoise",
                fontsize='small', ha='left', va='top', transform=ax.transAxes)
    bbox = t.get_window_extent(renderer=renderer)
    ypos -= ax.transAxes.inverted().transform_bbox(bbox).height + 0.03
    latex_dispersion = sympy.latex(band._energy_sympy)
    for direction in ['x', 'y', 'z']:
        latex_dispersion = latex_dispersion.replace(
            f"k{direction}", f"k_{direction}")
    yshift = _print_wrapped_equation(
        ax, renderer, xshift, ypos, "\\varepsilon_{\\mathbf{k}}",
        latex_dispersion, fontsize='xx-small')
    ypos -= yshift + 0.05

    t = ax.text(xshift, ypos, "Scattering Kernel", color="darkturquoise",
                fontsize='small', ha='left', va='top', transform=ax.transAxes)
    bbox = t.get_window_extent(renderer=renderer)
    ypos -= ax.transAxes.inverted().transform_bbox(bbox).height + 0.03
    latex_scattering = get_scattering_latex(
        params['scattering_kernel_names'][0],
        params['scattering_kernel_params'][0])
    for i, kernel in enumerate(params['scattering_kernel_names'][1:], start=1):
        part = get_scattering_latex(
            kernel, params['scattering_kernel_params'][i])
        if part.startswith('+') or part.startswith('-'):
            latex_scattering += part
        else:
            latex_scattering += " + " + part
    _print_wrapped_equation(ax, renderer, xshift, ypos,
                            "C(\\mathbf{k}, \\mathbf{k'})", latex_scattering)


def _print_wrapped_equation(ax, renderer, xpos, ypos, lhs, equation,
                            fontsize='x-small'):
    cursor = 0
    split_points = [0]
    parenthesis_tracker = 0
    for cursor in range(len(equation)):
        if equation[cursor] == '(':
            parenthesis_tracker += 1
        elif equation[cursor] == ')':
            parenthesis_tracker -= 1
        if equation[cursor] == '+' or equation[cursor] == '-':
            if parenthesis_tracker == 0:
                split_points.append(cursor)
    split_points.append(len(equation))
    dispersion_split = [equation[split_points[i]:split_points[i+1]]
                        for i in range(len(split_points)-1)]
    dispersion_split[0] = lhs + " = " + dispersion_split[0]

    wrapped = "$" + dispersion_split.pop(0)
    while dispersion_split:
        part = dispersion_split.pop(0)
        t = ax.text(xpos, ypos, wrapped + part + "$", fontsize=fontsize,
                    ha='left', va='top', transform=ax.transAxes)
        bbox = t.get_window_extent(renderer=renderer)
        width = ax.transAxes.inverted().transform_bbox(bbox).width
        t.remove()
        if xpos + width > 1.0:
            wrapped += "$\n$" + lhs + " = " + part
        else:
            wrapped += part
    wrapped += "$"

    t = ax.text(xpos, ypos, wrapped.split('\n')[0], fontsize=fontsize,
                ha='left', va='top', transform=ax.transAxes)
    bbox = t.get_window_extent(renderer=renderer)
    yshift = ax.transAxes.inverted().transform_bbox(bbox).height

    t = ax.text(xpos, ypos, "$" + lhs + " =$", fontsize=fontsize,
                ha='left', va='top', transform=ax.transAxes)
    bbox = t.get_window_extent(renderer=renderer)
    xpos += ax.transAxes.inverted().transform_bbox(bbox).width
    t.remove()

    for line in wrapped.split('\n')[1:]:
        line = "$" + line.split("=")[1]
        t = ax.text(xpos, ypos - yshift, line, fontsize=fontsize,
                     ha='left', va='top', transform=ax.transAxes)
        bbox = t.get_window_extent(renderer=renderer)
        yshift += ax.transAxes.inverted().transform_bbox(bbox).height

    return yshift


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
    kxs, kys = _get_plane_kx_ky(band, kz, n_interp, res)
    for kx, ky in zip(kxs, kys):
        ax.plot(kx, ky, **kwargs)


def plot_scattering_heatmap(fig, ax, params, res=101, **kwargs):
    kernel_values = _calc_heatmap(params, res)
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


def _calc_heatmap(params, res=101):
    phis = np.linspace(-np.pi, np.pi, res)
    kx, ky = np.cos(phis), np.sin(phis)
    kz = np.zeros_like(kx)
    kernel = elecboltz.easy_params(params)['scattering_kernel']
    if kernel.is_explicit:
        kernel_values = np.zeros((len(phis), len(phis)))
        for i in range(kernel.coeffs.shape[0]):
            basis_vec_i = kernel.eval_basis(i, kx, ky, kz)
            for j in range(kernel.coeffs.shape[0]):
                basis_vec_j = kernel.eval_basis(j, kx, ky, kz)
                kernel_values += kernel.coeffs[i, j] * np.outer(
                    basis_vec_i, basis_vec_j)
        return kernel_values
    else:
        band = elecboltz.BandStructure(**elecboltz.easy_params(params))
        band_low = copy(band)
        band_low.resolution = kernel.params.get('low_res', 21)
        band.discretize()
        band_low.discretize()

        kernel.decompose(band)
        kernel_values = \
            kernel.eigenvectors @ kernel.coeffs @ kernel.eigenvectors.T
        return _interpolate_to_phi(
            kernel_values, band_low.kpoints[:, 0],
            band_low.kpoints[:, 1], phis)


def _interpolate_to_phi(values, kx, ky, phi_interp):
    phi = np.arctan2(ky, kx)
    phi_grid, phi_prime_grid = np.meshgrid(phi, phi)
    points = np.column_stack((phi_grid.flatten(), phi_prime_grid.flatten()))
    values_flat = values.flatten()

    phi_interp = (phi_interp + np.pi) % (2*np.pi) - np.pi
    phi_interp_grid, phi_prime_interp_grid = np.meshgrid(
        phi_interp, phi_interp)
    interp_points = np.column_stack((phi_interp_grid.flatten(),
                                     phi_prime_interp_grid.flatten()))

    interp = scipy.interpolate.NearestNDInterpolator(points, values_flat)
    return interp(interp_points).reshape(len(phi_interp), len(phi_interp))


def _get_plane_kx_ky(band, kz, n_interp=300, res=None):
    gvec = band.domain_size * np.pi / band.unit_cell
    res = res or band.resolution
    if isinstance(res, (int, float)):
        res = [res, res]
    kgrid = np.mgrid[-gvec[0]:gvec[0]:1j*res[0], -gvec[1]:gvec[1]:1j*res[1]]
    contours = find_contours(
        band.energy_func(kgrid[0], kgrid[1], kz),
        band.chemical_potential)
    kx, ky = _get_interpolated_contours_in_order(contours, res, n_interp)
    for i in range(len(kx)):
        kx[i] = kx[i] * 2 * gvec[0] / res[0] - gvec[0]
        ky[i] = ky[i] * 2 * gvec[1] / res[1] - gvec[1]
    return kx, ky


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
    ds = np.linalg.norm(dk, axis=1)
    s = np.concatenate(([0], np.cumsum(ds)))
    interpolated_s = np.linspace(0, s[-1], n_interp)
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
    ax_equations = fig.add_subplot(gs[1, 1:])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    plot_full_fermi_surface_slices(ax_fermi, renderer, params,
                                   n_interp=n_interp, res=res, lw=1)
    plot_scattering_heatmap(fig, ax_scat_heat, params, res=res, cmap='plasma')
    params_y = print_params_left(ax_band_params, renderer, params,
                                 name=name, electron_doped=electron_doped)
    print_params_right(ax_scat_params, renderer, params, params_y=params_y)
    print_equations(ax_equations, renderer, params)

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
    ax.legend(frameon=False, title=r"$\varphi$")
    ax.set_xticks([-30, 0, 30, 60, 90, 120, 150])


def plot_kernel_admr(params, show=False):
    fig1, ax = plt.subplots()
    calc_and_plot_admr(ax, params)
    fig1.tight_layout()

    fig2 = plt.figure(layout='constrained')
    fig2.get_layout_engine().set(rect=(0, 0.02, 1.0, 0.94))
    plot_info(fig2, name="Nd-LSCO", electron_doped=False)
    
    filename = (os.path.dirname(os.path.relpath(__file__))
                + "/Kernel/ADMR_NdLSCO")
    for p in params['scattering_kernel_params'][
            :len(params['scattering_kernel_names'])]:
        for key, value in p.items():
            label = ''.join(key.split('_'))
            filename += f"_{label}={value}"
    filename += ".pdf"

    with PdfPages(filename) as pdf:
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
