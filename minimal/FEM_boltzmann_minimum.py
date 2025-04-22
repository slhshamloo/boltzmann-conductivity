import os, timeit
import numpy as np
import scipy as sp
from scipy.constants import e, hbar, m_e
from matplotlib import pyplot as plt
# set matplotlib defaults
import mpl_defaults

def energy_func(kx, ky):
    """
    Calculate the energy based on the wave vectors kx and ky.
    """
    return 0.5 * hbar**2 / m_e * (kx**2 + ky**2) # Free electrons placeholder


def velocity_func(kx, ky):
    """
    Calculate the velocity based on the wave vectors kx and ky.
    Returns vx, vy.
    """
    return hbar * kx / m_e, hbar * ky / m_e # Free electrons placeholder


def calculate_lengths(kx, ky):
    """
    Calculate the lengths of line segments in the kx-ky space.
    """
    # Calculate the differences between consecutive points
    dx = np.roll(kx, -1) - kx
    dy = np.roll(ky, -1) - ky
    
    # Calculate the lengths of the segments
    lengths = np.sqrt(dx**2 + dy**2)

    return lengths


def generate_fem_matrices(lengths, inverse_scattering_length):
    """
    Generate the overlap matrix $M_{ij}$ in sparse form and the covariant
    matrices $D_{ij}$ (derivative) and $Gamma_{ij}$ (out-scattering)
    in diagonal ordered form.
    """
    # l_{i-1}
    lengths_shifted_backward = np.roll(lengths, 1)
    # M matrix
    overlap_matrix = generate_overlap(lengths, lengths_shifted_backward)
    # D matrix
    derivative_matrix = np.vstack([np.full_like(lengths, -0.5),
                                   np.zeros_like(lengths),
                                   np.full_like(lengths, 0.5)])
    # Gamma matrix
    out_scattering_matrix = generate_out_scattering(
        lengths, lengths_shifted_backward, inverse_scattering_length)

    return overlap_matrix, derivative_matrix, out_scattering_matrix


def generate_overlap(lengths, lengths_shifted_backward):
    """
    Generate the overlap matrix $M_{ij}$ in sparse form.
    """
    # indices (i, i+1)
    indices = np.arange(len(lengths))
    indices_shifted_forward = np.roll(indices, -1)

    overlap_main = sp.sparse.csr_matrix(
        ((lengths+lengths_shifted_backward) / 3, (indices, indices)))
    overlap_upper = sp.sparse.csr_matrix(
        (lengths / 6, (indices_shifted_forward, indices)))
    overlap_lower = sp.sparse.csr_matrix(
        (lengths / 6, (indices, indices_shifted_forward)))
    return overlap_main + overlap_upper + overlap_lower


def generate_out_scattering(lengths, lengths_shifted_backward,
                            inverse_scattering_length):
    """
    Generate the out-scattering matrix $Gamma_{ij}$ in diagonal ordered form.
    """
    # gamma_{i+1}
    scattering_shifted_forward = np.roll(inverse_scattering_length, -1)
    # gamma_{i-1}
    scattering_shifted_backward = np.roll(inverse_scattering_length, 1)
    # Gamma matrix
    minor_diagonal_term = lengths / 12 * (
        inverse_scattering_length + scattering_shifted_forward)
    major_diagonal_term = (
        (lengths_shifted_backward+lengths) * inverse_scattering_length / 4
        + lengths * scattering_shifted_forward / 12
        + lengths_shifted_backward * scattering_shifted_backward / 12)
    return np.vstack(
        [minor_diagonal_term,major_diagonal_term, minor_diagonal_term])



def solve_conductivity(differential_operator, overlap_matrix,
                       a_velocity_ratio, b_velocity_ratio):
    """
    Calculate the conductivity by solving the linear system.

    # Parameters:
    differential_operator: the covariant differential operator matrix in
        diagonal ordered form.
    overlap_matrix: the overlap matrix in sparse form.
    a_velocity_ratio: the ratio of the velocity in the a
        direction to the full magnitude of the velocity.
    b_velocity_ratio: the ratio of the velocity in the b
        direction to the full magnitude of the velocity.

    # Returns:
    The conductivity tensor component sigma_{ab}
    """
    a_velocity_ratio_covariant = overlap_matrix @ a_velocity_ratio
    b_velocity_ratio_covariant = overlap_matrix @ b_velocity_ratio

    linear_solution = solve_cyclic_tridiagonal_system(
        differential_operator, b_velocity_ratio_covariant)
    # e^2/\hbar (\hat{v}_a)_i (A^{-1})^{ij} (\hat{v}_b)_j
    return e**2 / (2 * np.pi**2 * hbar) * np.dot(
        a_velocity_ratio_covariant, linear_solution)


def solve_conductivity_column(differential_operator, overlap_matrix,
                              a_velocity_ratio, b_velocity_ratio):
    """
    Calculate the conductivities $\sigma_{aa}$ and $\sigma_{ba}
    by solving the linear system.

    # Parameters:
    differential_operator: the covariant differential operator matrix in
        diagonal ordered form.
    overlap_matrix: the overlap matrix in sparse form.
    a_velocity_ratio: the ratio of the velocity in the a
        direction to the full magnitude of the velocity.
    b_velocity_ratio: the ratio of the velocity in the b
        direction to the full magnitude of the velocity.

    # Returns:
    The conductivity tensor components sigma_{aa} and sigma_{ba}
    """
    a_velocity_ratio_covariant = overlap_matrix @ a_velocity_ratio
    b_velocity_ratio_covariant = overlap_matrix @ b_velocity_ratio

    linear_solution = solve_cyclic_tridiagonal_system(
        differential_operator, a_velocity_ratio_covariant)
    # e^2/\hbar (\hat{v}_a)_i (A^{-1})^{ij} (\hat{v}_a)_j
    sigma_aa = e**2 / (2 * np.pi**2 * hbar) * np.dot(
        a_velocity_ratio_covariant, linear_solution)
    # e^2/\hbar (\hat{v}_b)_i (A^{-1})^{ij} (\hat{v}_a)_j
    sigma_ba = e**2 / (2 * np.pi**2 * hbar) * np.dot(
        b_velocity_ratio_covariant, linear_solution)
    return sigma_aa, sigma_ba


def solve_cyclic_tridiagonal_system(A, b):
    """
    Solve the cyclic tridiagonal system of equations Ax = b.

    # Parameters:
    A: a cyclic tridiagonal matrix in diagonal ordered form.
    b: the right-hand side vector.
    # Returns:
    x: the solution vector.
    """
    # A = B + uv^T where B is banded
    # building perturbation matrix uv^T
    u = np.zeros_like(b)
    free_factor = -A[1, 0] # arbitrary, but avoid division by zero
    A[1, 0] -= free_factor
    A[1, -1] -= A[0, 0] * A[2, -1] / free_factor
    u[0] = 1
    u[-1] = A[0, 0] / free_factor
    # For v, only the dot product is calculated, so only the first
    # and the last elements (the only nonzero elements) are kept
    v_1 = free_factor
    v_n = A[2, -1]

    # Tridiagonal matrix inversions
    # B^{-1}b
    banded_solution = sp.linalg.solve_banded((1, 1), A, b)
    # B^{-1}u
    rank_1_solution = sp.linalg.solve_banded((1, 1), A, u)

    # Dot products
    v_dot_banded_solution = v_1*banded_solution[0] + v_n*banded_solution[-1]
    v_dot_rank_1_solution = v_1*rank_1_solution[0] + v_n*rank_1_solution[-1]

    # A^{-1}b = B^{-1}b - (B^{-1}u) (v^T B^{-1}b) / (1 + v^T B^{-1}u)
    full_solution = (banded_solution - rank_1_solution
                     * (v_dot_banded_solution
                        / (1 + v_dot_rank_1_solution)))

    # Reset the matrix
    A[1, 0] += free_factor
    A[1, -1] += A[0, 0] * A[2, -1] / free_factor

    return full_solution


def calculate_boltzmann(kx, ky, scattering_rate, magnetic_fields):
    """
    Calculate Boltzmann transport conductivity for a range of magnetic fields
    using a finite element method.

    # Parameters:
    kx: x-component of the wavevector on the Fermi surface
    ky: y-component of the  wavevector on the Fermi surface
    scattering_rate: scattering rate of each point on the Fermi surface
        in kx-ky space
    magnetic_fields: magnetic field strengths

    # Returns:
    sigma_xx, sigma_xy: conductivity tensor components
    """
    # Calculate velocities
    vx, vy = velocity_func(kx, ky)
    v = np.sqrt(vx**2 + vy**2)
    x_velocity_ratio, y_velocity_ratio = vx / v, vy / v
    # Calculate lengths in kx-ky space
    lengths = calculate_lengths(kx, ky)
    # Generate the comprising matrices
    overlap, derivative, out_scattering = generate_fem_matrices(
        lengths, scattering_rate / v)

    sigma_xx = np.zeros_like(magnetic_fields)
    sigma_yx = np.zeros_like(magnetic_fields)
    for i, field in enumerate(magnetic_fields):
        # A = \Gamma - eB/\hbar D - S
        differential_operator = out_scattering - e*field/hbar * derivative
            # - in_scattering_matrix which is taken to be zero
            # for now for the free electron placeholder
        sigma_xx[i], sigma_yx[i] = solve_conductivity_column(
            differential_operator, overlap, x_velocity_ratio, y_velocity_ratio)
    return sigma_xx, -sigma_yx


def calculate_drude(kf, scattering_rate, magnetic_field):
    """
    Calculate the Drude model conductivity.

    # Parameters:
    kf: Fermi wavevector
    scattering_rate: scattering rate
    magnetic_field: magnetic field strength

    # Returns:
    sigma_xx, sigma_xy: conductivity tensor components
    """
    omega_c = e * magnetic_field / m_e
    sigma_xx = (e**2 * kf**2 / (2*np.pi*m_e) * scattering_rate
                / (omega_c**2 + scattering_rate**2))
    sigma_xy = -(e**2 * kf**2 / (2*np.pi*m_e) * omega_c
                 / (omega_c**2 + scattering_rate**2))

    return sigma_xx, sigma_xy


def free_electron_test():
    """
    Test Boltzmann transport for the free electron system vs the Drude model.
    """
    magnetic_fields = np.linspace(0, 30, 1000)
    kf = 1e10
    theta = np.linspace(0, 2*np.pi, 100, endpoint=False)
    kx = kf * np.cos(theta)
    ky = kf * np.sin(theta)
    scattering_rate_constant = 1e12
    scattering_rate = np.full_like(kx, scattering_rate_constant)

    sigma_xx_drude, sigma_xy_drude = calculate_drude(
        kf, scattering_rate_constant, magnetic_fields)
    sigma_xx_boltzmann, sigma_xy_boltzmann = calculate_boltzmann(
        kx, ky, scattering_rate, magnetic_fields)

    plt.plot(magnetic_fields, sigma_xx_drude, label=fr"$\sigma_{{xx}}$ Drude",
             color='blue')
    plt.plot(magnetic_fields, sigma_xy_drude, label=fr"$\sigma_{{xy}}$ Drude",
             color='black')
    plt.plot(magnetic_fields, sigma_xx_boltzmann, linestyle=(0, (3, 3)),
             color='red', label=fr"$\sigma_{{xx}}$ Boltzmann")
    plt.plot(magnetic_fields, sigma_xy_boltzmann, linestyle=(0, (3, 3)),
             color='lime', label=fr"$\sigma_{{xy}}$ Boltzmann")
    make_plot("Free Electron Test", "Magnetic Field (T)",
              "Conductivity (S/m)", "free_electron_test.pdf")


def compare_error():
    magnetic_fields = np.linspace(0, 30, 1000)
    kf = 1e10
    scattering_rate_constant = 1e12
    
    sigma_xx_drude, sigma_xy_drude = calculate_drude(
        kf, scattering_rate_constant, magnetic_fields)
    ns = np.linspace(10, 200, 50, dtype=int)
    errors = np.zeros_like(ns, dtype=float)
    for (i, n) in enumerate(ns):
        theta = np.linspace(0, 2*np.pi, n, endpoint=False)
        kx = kf * np.cos(theta)
        ky = kf * np.sin(theta)
        scattering_rate = np.full_like(kx, scattering_rate_constant)
        sigma_xx_boltzmann, sigma_xy_boltzmann = calculate_boltzmann(
            kx, ky, scattering_rate, magnetic_fields)
        errors[i] = max(np.max(np.abs(sigma_xx_boltzmann - sigma_xx_drude)),
                        np.max(np.abs(sigma_xy_boltzmann - sigma_xy_drude)))
    
    plt.loglog(ns, errors, color='mediumblue')
    # plt.xticks([10] + list(np.linspace(0, ns[-1], 5, dtype=int)[1:]))
    make_plot("Boltzmann FEM Model Error", "Number of Elements",
              "Maximum Error", "free_electron_error.pdf", legend=False)


def compare_performance():
    magnetic_fields = np.linspace(0, 1, 10)
    kf = 1e10
    scattering_rate_constant = 1e12

    ns = np.linspace(10, 1000, 50, dtype=int)
    times = np.zeros_like(ns, dtype=float)
    for (i, n) in enumerate(ns):
        theta = np.linspace(0, 2*np.pi, n, endpoint=False)
        kx = kf * np.cos(theta)
        ky = kf * np.sin(theta)
        scattering_rate = np.full_like(kx, scattering_rate_constant)
        timer = timeit.Timer(lambda: calculate_boltzmann(
            kx, ky, scattering_rate, magnetic_fields))
        nloops = 100
        # single loop for compilations and other warmup
        calculate_boltzmann(kx, ky, scattering_rate, magnetic_fields)
        time_list = timer.repeat(5, nloops)
        times[i] = min(time_list) / nloops * 1e3 # ms

    plt.plot(ns, times, color='mediumblue')
    plt.xticks([10] + list(np.linspace(0, ns[-1], 5, dtype=int)[1:]))
    make_plot("Boltzmann FEM Model Performance", "Number of Elements",
              "Execution Time (ms)", "free_electron_performance.pdf",
              legend=False)


def make_plot(title, xlabel, ylabel, save_path, legend=True):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tick_params(axis='x', which='major', pad=8)
    plt.tick_params(axis='y', which='major', pad=10)
    if legend:
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.dirname(os.path.relpath(__file__))
                + '/' + save_path, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # free_electron_test()
    # compare_error()
    compare_performance()
