import elecboltz
import numpy as np
import sympy
from skimage.measure import find_contours
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pathlib
import os


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
mpl.rcParams['figure.figsize'] = (7.48, 5.2)
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['lines.markersize'] = 20


label_to_latex = {
    "tp": "t'",
    "tpp": "t''",
    "tz": "t_z",
    "tzp": "t_z'",
    "gamma_0": "\\Gamma_0",
    "gamma_k": "\\Gamma_k",
    "power": "\\nu",
    "mu": "\\mu"
}

def get_scattering_latex(model, params):
    if model == 'isotropic':
        return "\\Gamma_0"
    elif any(model.startswith(trig) for trig in ['cos', 'sin', 'tan', 'cot']):
        trig = model[:3]
        if len(model) > 3:
            sym = int(model[3:-3])
        else:
            sym = params.get('sym', 1)
        return f"\\Gamma_k |{trig}({sym}\\phi)|^{{\\nu}}"
    else:
        raise ValueError(f"Unknown scattering model: {model}")


def get_init_params():
    return {
        'a': 2.94,
        'b': 2.94,
        'c': 18.06,
        'energy_scale': -789,
        'band_params': {'mu': -0.22, 't': 1, 'tp': -0.23,
                        'tz': 0.03, 'tzp': 0.0},
        'periodic': True,
        'resolution': [41, 41, 41],
        'domain_size': [1.0, 1.0, 3.0],
        'bz_ratio': 0.8660254037844386,
        'sort_axis': 2,
        'dispersion': (
            "mu + t*(2*cos(sqrt(3)/2*a*kx)*cos(1/2*b*ky) + cos(b*ky))"
            " + tp*(2*cos(sqrt(3)/2*a*kx)*cos(3/2*b*ky) + cos(sqrt(3)*a*kx))"
            " + tz*(cos(sqrt(3)*a*kx/3+c*kz/3) + cos(-sqrt(3)*a*kx/6+a*ky/2+c*kz/3) + cos(-sqrt(3)*a*kx/6-a*ky/2+c*kz/3))"
            " + tzp*(cos(-2*sqrt(3)*a*kx/3+c*kz/3) + cos(sqrt(3)*a*kx/3 - a*ky + c*kz/3) + cos(sqrt(3)*a*kx/3+a*ky + c*kz/3))"),
        'scattering_models': [
            'isotropic',
            'tan3phi'
        ],
        'scattering_params': {
            'gamma_0': 2.5,
            'gamma_k': 20.0,
            'power': 10.0
        }
    }


def print_band_params(ax, renderer, params, name=None, xshift=0.1):
    ax.axis('off')
    ypos = 1.0

    if name is not None:
        t = ax.text(xshift, ypos, name, color="mediumblue", fontsize='large',
                    ha='left', va='top', transform=ax.transAxes)
        bbox = t.get_window_extent(renderer=renderer)
        ypos -= ax.transAxes.inverted().transform_bbox(bbox).height + 0.05

    t = ax.text(xshift, ypos, "Band Parameters", color="darkturquoise",
                fontsize='small', ha='left', va='top', transform=ax.transAxes)
    bbox = t.get_window_extent(renderer=renderer)
    ypos -= ax.transAxes.inverted().transform_bbox(bbox).height + 0.03

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


def print_scattering_params(ax, renderer, band, params,
                            electron_doped=False, xshift=0.1):
    ax.axis('off')
    ypos = 1.0

    if electron_doped:
        filling_text = f'$n = {band.calculate_filling_fraction():.3g}$'
    else:
        filling_text = f'$p = {1-band.calculate_filling_fraction():.3g}$'
    t = ax.text(xshift, ypos, filling_text, color="mediumblue", fontsize='large',
                ha='left', va='top', transform=ax.transAxes)
    bbox = t.get_window_extent(renderer=renderer)
    ypos -= ax.transAxes.inverted().transform_bbox(bbox).height + 0.05

    t = ax.text(xshift, ypos, "Scattering Parameters", color="darkturquoise",
                fontsize='small', ha='left', va='top', transform=ax.transAxes)
    bbox = t.get_window_extent(renderer=renderer)
    ypos -= ax.transAxes.inverted().transform_bbox(bbox).height + 0.03

    params_text = ''
    for (label, value) in params['scattering_params'].items():
        label_latex = label_to_latex.get(label, label)
        params_text += f"${label_latex}={value:.4g}$"
        if label.startswith('gamma'):
            params_text += " THz"
        params_text += '\n'
    ax.text(xshift, ypos, params_text, color="black", fontsize='x-small',
            ha='left', va='top', transform=ax.transAxes)


def print_equations(ax, renderer, band, params, xshift=0.05):
    ax.axis('off')
    ypos = 1.0

    t = ax.text(xshift, ypos, "Band Dimensions", color="darkturquoise",
                fontsize='small', ha='left', va='top', transform=ax.transAxes)
    bbox = t.get_window_extent(renderer=renderer)
    ypos -= ax.transAxes.inverted().transform_bbox(bbox).height
    params_text = ', '.join(
        fr"${label}={value}$ Å" for label, value
        in zip(('a', 'b', 'c'), (params['a'], params['b'], params['c'])))
    t = ax.text(xshift, ypos, params_text, fontsize='x-small',
            ha='left', va='top', transform=ax.transAxes)
    bbox = t.get_window_extent(renderer=renderer)
    ypos -= ax.transAxes.inverted().transform_bbox(bbox).height + 0.03

    t = ax.text(xshift, ypos, "Dispersion Relation", color="darkturquoise",
                fontsize='small', ha='left', va='top', transform=ax.transAxes)
    bbox = t.get_window_extent(renderer=renderer)
    ypos -= ax.transAxes.inverted().transform_bbox(bbox).height
    latex_dispersion = sympy.latex(band._energy_sympy)
    for direction in ['x', 'y', 'z']:
        latex_dispersion = latex_dispersion.replace(
            f"k{direction}", f"k_{direction}")
    yshift = _print_wrapped_equation(
        ax, renderer, xshift, ypos, "\\varepsilon_{\\mathbf{k}}",
        latex_dispersion, fontsize='xx-small')
    ypos -= yshift + 0.03

    t = ax.text(xshift, ypos, "Scattering Rate", color="darkturquoise",
                fontsize='small', ha='left', va='top', transform=ax.transAxes)
    bbox = t.get_window_extent(renderer=renderer)
    ypos -= ax.transAxes.inverted().transform_bbox(bbox).height
    latex_scattering = get_scattering_latex(
        params['scattering_models'][0], params['scattering_params'])
    for model in params['scattering_models'][1:]:
        part = get_scattering_latex(model, params['scattering_params'])
        if part.startswith('+') or part.startswith('-'):
            latex_scattering += part
        else:
            latex_scattering += " + " + part
    _print_wrapped_equation(ax, renderer, xshift, ypos,
                            "\\Gamma_{\\mathrm{tot}}", latex_scattering)


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


def plot_full_fermi_surface_slices(ax, renderer, band,
                                   n_interp=300, res=101, **kwargs):
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


def plot_fermi_surface_slice(ax, band, kz=0.0, n_interp=300, res=101,
                             **kwargs):
    kxs, kys = _get_fermi_surface_slice(band, kz=kz, n_interp=n_interp, res=res)
    for kx, ky in zip(kxs, kys):
        ax.plot(kx, ky, **kwargs)


def plot_scattering_heatmap(ax, band, cond, kz=0.0, n_interp=300, res=101,
                            **kwargs):
    # Code based on the example at:
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html
    kwargs['capstyle'] = 'butt'

    kxs, kys = _get_fermi_surface_slice(
        band, kz=kz, n_interp=n_interp, res=res)
    for kx, ky in zip(kxs, kys):
        kx_mid = np.hstack((kx[0], 0.5 * (kx[1:] + kx[:-1]), kx[-1]))
        ky_mid = np.hstack((ky[0], 0.5 * (ky[1:] + ky[:-1]), ky[-1]))
        coord_start = np.column_stack((kx_mid[:-1], ky_mid[:-1])
                                      )[:, np.newaxis, :]
        coord_mid = np.column_stack((kx, ky))[:, np.newaxis, :]
        coord_end = np.column_stack((kx_mid[1:], ky_mid[1:]))[:, np.newaxis, :]
        segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

        lc = mpl.collections.LineCollection(segments, **kwargs)
        if isinstance(cond.scattering_rate, (int, float)):
            lc.set_array(np.full_like(kx, cond.scattering_rate))
        else:
            scat = cond.scattering_rate(kx, ky, kz)
            if isinstance(scat, (int, float)):
                scat = np.full_like(kx, scat)
            lc.set_array(scat)
        ax.add_collection(lc)
    return ax.collections[-1]  # return the last collection added for colorbar

def plot_full_scattering_heatmap(fig, ax, band, cond, kz=0.0,
                                 n_interp=300, res=101, **kwargs):
    ax.set_xlim(-band.domain_size[0] * np.pi / band.unit_cell[0],
                band.domain_size[0] * np.pi / band.unit_cell[0])
    ax.set_ylim(-band.domain_size[1] * np.pi / band.unit_cell[1],
                band.domain_size[1] * np.pi / band.unit_cell[1])
    ax.set_aspect('equal')
    lines = plot_scattering_heatmap(ax, band, cond, kz=kz,
                                    n_interp=n_interp, res=res, **kwargs)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(lines, cax=cax,
                        label=r"$\Gamma_{\mathrm{tot}}$ (THz)")
    cbar.ax.tick_params(labelsize='xx-small')
    cbar.ax.set_ylabel(cbar.ax.get_ylabel(), fontsize='xx-small')
    ax.tick_params(axis='both', labelsize='x-small')


def _get_fermi_surface_slice(band, kz=0.0, n_interp=300, res=101):
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
    return kxs, kys


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


def plot_info(fig, fit_path, name=None, electron_doped=False,
              n_interp=300, res=101):
    params = elecboltz.easy_params({'load_fit': fit_path})
    band = elecboltz.BandStructure(**params)
    cond = elecboltz.Conductivity(band, **params)

    prev_font = mpl.rcParams['font.size']
    mpl.rcParams['font.size'] = 16  # smaller font for the plots
    gs = fig.add_gridspec(2, 3)
    ax_fermi = fig.add_subplot(gs[0, 0])
    ax_scat_heat = fig.add_subplot(gs[1, 0])
    ax_band_params = fig.add_subplot(gs[0, 1])
    ax_scat_params = fig.add_subplot(gs[0, 2])
    ax_equations = fig.add_subplot(gs[1, 1:])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    plot_full_fermi_surface_slices(ax_fermi, renderer, band,
                                   n_interp=n_interp, res=res, linewidth=1)
    plot_full_scattering_heatmap(fig, ax_scat_heat, band, cond,
                                 n_interp=n_interp, res=res, linewidth=7)
    print_band_params(ax_band_params, renderer, params, name=name)
    print_scattering_params(ax_scat_params, renderer, band, params,
                            electron_doped=electron_doped)
    print_equations(ax_equations, renderer, band, params)

    mpl.rcParams['font.size'] = prev_font  # restore original font size


def calculate_fit(params, phis, thetas, field,
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


def plot_fit(fig, axs, fit_path, data_path, temperature, n_interp=100,
             name=None, units_data=1, units_fit=1, units_label=None,
             save_rho=False, load_rho=False, theta_norm=None, ticks=None,
             **kwargs):
    params, loader, rho_zz = _load_and_calculate_fit(
        fit_path, data_path, temperature, n_interp,
        theta_norm, save_rho, load_rho)
    _plot_data_and_fit_curves(axs, rho_zz, loader, n_interp,
                              units_data, units_fit, **kwargs)
    _set_fit_plot_labels(fig, axs, name, temperature, loader,
                         theta_norm, units_label, ticks)
    _set_fit_plot_legend(axs, params)
    fig.tight_layout(pad=0.3)


def _load_and_calculate_fit(fit_path, data_path, temperature, n_interp,
                            theta_norm, save_rho, load_rho):
    params = elecboltz.easy_params({'load_fit': fit_path})
    loader = elecboltz.Loader(
        x_vary_label='theta', y_label='rho_zz',
        x_search={'T': [temperature] * 3}, save_new_labels=True)
    loader.load(data_path, x_columns=[0], y_columns=[1])
    loader.interpolate(n_interp, x_normalize=theta_norm)

    if save_rho:
        path = pathlib.Path(fit_path)
        save_directory = str(path.parent / path.stem)
    else:
        save_directory = None
    if load_rho:
        path = pathlib.Path(fit_path)
        load_directory = str(path.parent / path.stem)
        rho_zz = np.empty((len(loader.x_search['phi']), n_interp))
        for i, phi in enumerate(loader.x_search['phi']):
            theta_range, rho_zz[i, :] = np.loadtxt(
                pathlib.Path(load_directory) / f"phi={phi}.csv",
                delimiter=',', skiprows=1, unpack=True)
    else:
        rho_zz = calculate_fit(
            params, loader.x_search['phi'], theta_range,
            field=loader.x_data['Bamp'][0], theta_norm=theta_norm,
            save_directory=save_directory)
    return params, loader, rho_zz


def _plot_data_and_fit_curves(axs, rho_zz, loader, n_interp,
                              units_data, units_fit, **kwargs):
    theta_min = min(np.min(x) for x in loader.x_data_raw['theta'])
    theta_max = max(np.max(x) for x in loader.x_data_raw['theta'])
    theta_range = np.linspace(theta_min, theta_max, n_interp)

    palette = mpl.colormaps.get_cmap(kwargs.get('cmap', 'Blues'))

    for i, phi in enumerate(loader.x_search['phi']):
        color = palette((i+1) / len(loader.x_search['phi']))
        axs[0].plot(loader.x_data_interpolated['theta'][i],
                    units_data * loader.y_data_interpolated['rho_zz'][i],
                    label=f"${phi}$°", color=color, **kwargs)
        axs[1].plot(theta_range, units_fit * rho_zz[i, :],
                    label=f"${phi}$°", color=color, **kwargs)


def _set_fit_plot_legend(axs, params):
    axs[0].legend(frameon=False, handlelength=1.5, handletextpad=0.5,
                  fontsize='medium', title=r"$\phi$")
    axs[1].text(
        0.95, 0.94,
        fr"$\Gamma_0={params['scattering_params']['gamma_0']:.3g}$"
        + " THz\n"
        + fr"$\Gamma_k={params['scattering_params']['gamma_k']:.3g}$"
        + " THz\n"
        + fr"$\nu={params['scattering_params']['power']:.3g}$"
        + "\n$t_z'="
        + f"{params['band_params']['tzp']/params['energy_scale']:.3g}t$",
        transform=axs[1].transAxes, va='top', ha='right', fontsize='medium')


def _set_fit_plot_labels(fig, axs, name, temperature, loader,
                         theta_norm, units_label, ticks):
    title = ''
    if name is not None:
        title += name + ", "
    title += fr"$T={temperature}$ K, $B={loader.x_data['Bamp'][0]:.1f}$ T"
    fig.suptitle(title)

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


def plot_fit_and_info(
        fit_path, data_path, save_path, temperature, name=None, show=False,
        n_interp_fit=100, n_interp_info=300, res=101, **kwargs):
    fig1, axs = plt.subplots(1, 2, figsize=(7.48, 4.0),
                            sharex=True, sharey=True)
    plot_fit(fig1, axs, fit_path, data_path, temperature,
             n_interp=n_interp_fit, **kwargs)

    fig2 = plt.figure(figsize=(7.48, 4.0), layout='constrained')
    plot_info(fig2, fit_path, name=name, electron_doped=True,
              n_interp=n_interp_info, res=res)
    
    with PdfPages(save_path) as pdf:
        pdf.savefig(fig1, bbox_inches='tight')
        pdf.savefig(fig2, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig1)
        plt.close(fig2)

def main():
    folder_name = "ADMR_PdCoO2_normalized_band=t+tp+tz+tzp_scat" \
                  "=iso+tan3phi_free=g0+gk+nu+tzp"
    filedir = os.path.dirname(os.path.relpath(__file__))
    Ts = [15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
    ns = [250, 250, 250, 250, 235, 235, 225, 170, 135, 110, 95, 85, 65]
    cmaps = ['Reds', 'Oranges', 'Greens', 'Blues', 'Purples', 'Greys']
    for i, (T, n) in enumerate(zip(Ts, ns)):
        save_path = (filedir + "/../fits/" + folder_name + "/"
                     + folder_name + f"_T{T}")
        plot_fit_and_info(
            save_path + ".json", filedir + "/../data/ADMR_PdCoO2",
            save_path + ".pdf", T, name="PdCoO$_2$", n_interp_fit=n,
            save_rho=True, cmap=cmaps[i % len(cmaps)], theta_norm=180,
            ticks=[60, 90, 120, 150, 180, 210])

if __name__ == "__main__":
    main()
