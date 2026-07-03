import elecboltz
import os, pathlib
import sympy, scipy
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
mpl.rcParams['figure.figsize'] = (10.0, 5.2)
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['lines.markersize'] = 20
mpl.rcParams['axes.formatter.useoffset'] = False


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
    "sigma_h": "\\sigma_h",
    "dphi_h": "\\Delta\\varphi_h",
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
    "phi_h": "\\varphi_h",
    "nu_fc": "\\nu_{fc}",
    "nu_fs": "\\nu_{fs}",
    "nu_bc": "\\nu_{bc}",
    "nu_bs": "\\nu_{bs}"
}


def get_params():
    return {
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
    'scattering_kernel_names': ['isotropic', 'hotspot_phi'],
    'scattering_kernel_params': [
        {'C_0': 2.9},
        {'phi_h': [0, 90, 180, 270], 'dphi_h': 90,
         'C_h': 85, 'sigma_h': 0.13},
        {'rank': 20, 'low_res': 21}],
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
    elif name == 'hotspot_phi':
        return "\\sum_{i}C_h\\mathrm{exp}\\left(-\\frac" \
            "{(\\varphi-\\varphi_{h,i})^2}{2\\sigma_h^2}\\right)" \
            "\\mathrm{exp}\\left(-\\frac{(\\varphi'-\\varphi'_{h,i})^2}" \
            "{2\\sigma_h^2}\\right)"
    else:
        return ""


def print_params_left(ax, renderer, params, name=None, xshift=0.05,
                      electron_doped=False):
    band = elecboltz.BandStructure(**params)
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


def print_params_right(ax, renderer, params, params_y=None, xshift=0.0):
    band = elecboltz.BandStructure(**params)
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
            if isinstance(value, (list, np.ndarray)):
                value = [f"{v:.3g}" for v in value]
                value = ", ".join(value)
            else:
                value = f"{value:.3g}"
            params_text += f"${label_latex}={value}$"
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


def print_equations(ax, renderer, params, xshift=0.02):
    band = elecboltz.BandStructure(**params)
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
    for label in label_to_latex:
        latex_dispersion = latex_dispersion.replace(
            label, label_to_latex[label])
    yshift = _print_wrapped_equation(
        ax, renderer, xshift, ypos, "\\varepsilon_{\\mathbf{k}}",
        latex_dispersion, fontsize='x-small')
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
    band = elecboltz.BandStructure(**params)
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
    kernel = elecboltz.kernel.build_kernel(
        params['scattering_kernel_names'], params['scattering_kernel_params'])
    if hasattr(kernel, 'eval_basis'):   # explicit basis evaluation
        kernel_values = np.zeros((len(phis), len(phis)))
        for i in range(kernel.coeffs.shape[0]):
            basis_vec_i = kernel.eval_basis(i, kx, ky, kz)
            for j in range(kernel.coeffs.shape[0]):
                basis_vec_j = kernel.eval_basis(j, kx, ky, kz)
                kernel_values += kernel.coeffs[i, j] * np.outer(
                    basis_vec_i, basis_vec_j)
        return kernel_values
    else:
        band = elecboltz.BandStructure(**params)
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


def _calculate_fit(params, x_data, n_interp, theta_norm,
                   save_rho, load_rho, fit_path=None):
    phis = np.unique(x_data['Bphi'])
    theta_min, theta_max = np.min(x_data["Btheta"]), np.max(x_data["Btheta"])
    theta_range = np.linspace(theta_min, theta_max, n_interp)

    if save_rho and fit_path is not None:
        path = pathlib.Path(fit_path)
        save_directory = str(path.parent / path.stem)
    else:
        save_directory = None
    if load_rho and fit_path is not None:
        path = pathlib.Path(fit_path)
        load_directory = str(path.parent / path.stem)
        rho_zz = np.empty((len(phis), n_interp))
        for i, phi in enumerate(phis):
            theta_range, rho_zz[i, :] = np.loadtxt(
                pathlib.Path(load_directory) / f"phi={phi}.csv",
                delimiter=',', skiprows=1, unpack=True)
    else:
        rho_zz = calculate_rho(
            params, phis, theta_range, field=x_data['Bamp'][0],
            theta_norm=theta_norm, save_directory=save_directory)
    return theta_range, rho_zz


def _plot_data_and_fit_curves(axs, theta_range, rho_zz, x_data, y_data,
                              units_fit, **kwargs):
    phis = np.unique(x_data['Bphi'])
    palette = mpl.colormaps.get_cmap(kwargs.get('cmap', 'Blues'))
    kwargs.pop('cmap', None)

    for i, phi in enumerate(phis):
        color = palette((i+1) / len(phis))
        axs[0].plot(x_data['Btheta'][x_data['Bphi'] == phi],
                    y_data['rho_zz'][x_data['Bphi'] == phi],
                    color=color, **kwargs)
        axs[1].plot(theta_range, units_fit * rho_zz[i, :], color=color,
                    label = f"${phi}^\\circ$", **kwargs)

    axs[1].legend(title=r"$\phi$", fontsize='x-small',
                  title_fontsize='small', frameon=False)


def _set_fit_plot_labels(fig, axs, theta_norm, exp_info, units_label, ticks):
    if exp_info is not None:
        axs[0].text(0.05, 0.03, exp_info, fontsize='small',
                    ha='left', va='bottom', transform=axs[0].transAxes)

    ylabel = "$\\rho_{zz}(\\theta)$"
    if theta_norm is not None:
        ylabel = ylabel[:-1] + f"/\\rho_{{zz}}({theta_norm}$°$)$"
    if units_label is not None:
        ylabel += f" ({units_label})"
    axs[0].set_ylabel(ylabel, fontsize='large')

    axs[0].set_title("Data")
    axs[1].set_title("Fit")
    fig.supxlabel(r"$\theta$ (°)")
    if ticks is not None:
        axs[0].set_xticks(ticks)
        axs[1].set_xticks(ticks)


def calculate_rho(params, phis, thetas, field,
                  theta_norm=None, save_directory=None):
    band = elecboltz.BandStructure(**params)
    band.discretize()
    cond = elecboltz.Conductivity(band, **params)
    cond.Bamp = field

    rho_zz = np.empty((len(phis), len(thetas)))
    for i, phi in enumerate(phis):
        cond.Bphi = phi
        for j, theta in enumerate(thetas):
            cond.Btheta = theta
            cond.calculate()
            rho = np.linalg.inv(cond.sigma)
            rho_zz[i, j] = rho[2, 2]

    if theta_norm is not None:
        cond.Bphi = 0
        cond.Btheta = theta_norm
        cond.calculate()
        rho = np.linalg.inv(cond.sigma)
        rho_zz_0 = rho[2, 2]
        rho_zz /= rho_zz_0
    if save_directory is not None:
        for i, phi in enumerate(phis):
            file_path = pathlib.Path(save_directory) / f"phi={phi}.csv"
            file_path.parent.mkdir(exist_ok=True, parents=True)
            np.savetxt(file_path, np.column_stack((thetas, rho_zz[i, :])),
                       delimiter=',', header=f"theta,rho_zz")
    return rho_zz


def plot_fit(fig, axs, params, x_data, y_data, fit_path=None, exp_info=None,
             units_label=None, units_fit=1, theta_norm=None, ticks=None,
             save_rho=False, load_rho=False, n_interp=100, **kwargs):
    theta_range, rho_zz = _calculate_fit(
        params, x_data, n_interp, theta_norm,
        save_rho, load_rho, fit_path=fit_path)
    _plot_data_and_fit_curves(
        axs, theta_range, rho_zz, x_data, y_data, units_fit, **kwargs)
    _set_fit_plot_labels(fig, axs, theta_norm, exp_info, units_label, ticks)
    # fit_params = json.load(open(fit_path))['fit_params']
    # _print_fit_params_on_plot(axs[1], fit_params)
    fig.tight_layout(pad=0.3)


def plot_info(fig, params, name=None, electron_doped=False,
              n_interp=300, res=101):
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


def plot_admr(
        x_data, y_data, save_path, fit_path=None, name=None, exp_info=None,
        n_interp_fit=50, n_interp_info=300, res=101, electron_doped=False,
        save_fig=True, show_fig=False, show_info=False, **kwargs):
    if fit_path is None:
        params = elecboltz.easy_params(get_params())
    else:
        params = elecboltz.easy_params({'load_fit': fit_path})

    fig1, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    plot_fit(fig1, axs, params, x_data, y_data, fit_path=fit_path,
             exp_info=exp_info, n_interp=n_interp_fit, **kwargs)

    fig2 = plt.figure(layout='constrained')
    fig2.get_layout_engine().set(rect=(0, 0.02, 1.0, 0.94))
    plot_info(fig2, params, name=name, res=res, n_interp=n_interp_info,
              electron_doped=electron_doped)
    
    if save_fig:
        with PdfPages(save_path) as pdf:
            pdf.savefig(fig1)
            pdf.savefig(fig2)
    if not show_info:
        plt.close(fig2)
    if show_fig:
        plt.show()
    else:
        plt.close(fig1)


def single():
    phis = [0, 15, 30, 45]
    temperature = 25
    field = 45
    n_interp = 35

    loader = elecboltz.Loader(
        x_vary_label='theta', y_label='rho_zz',
        x_search={'phi': phis.copy(), 'H': [field] * 4},
        save_new_labels=False)
    loader.load(
        os.path.dirname(os.path.relpath(__file__)) + "/../data/ADMR_NdLSCO",
        f"NdLSCO_0p25_rho_c-vs-theta_{temperature}K_",
        x_columns=[0], y_columns=[1])
    loader.interpolate(n_interp, x_normalize=0)

    folder_name = \
        f"ADMR_NdLSCO_T{temperature}_relative_band=t+tp+tpp+tz" \
        f"_scat=hotspot_free=C0+Ch+sigmah"
    filedir = os.path.dirname(os.path.relpath(__file__))
    exp_info = f"Nd-LSCO, $p=0.24$\n$T={temperature}$ K, $B={field}$ T"

    save_path = (filedir + "/../fits/ADMR_NdLSCO/" + folder_name
                 + "/" + folder_name)
    fit_path = save_path + ".json"

    # forward calculation
    # save_path = filedir + "/test"
    # fit_path = None

    plot_admr(
        loader.x_data, loader.y_data, save_path+".pdf", fit_path=fit_path,
        name="Nd-LSCO", exp_info=exp_info, save_rho=True, theta_norm=0,
        cmap='Blues', n_interp_fit=n_interp, electron_doped=False,
        ticks=[0, 30, 60, 90, 120])


if __name__ == "__main__":
    single()
