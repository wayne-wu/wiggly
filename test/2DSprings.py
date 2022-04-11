'''
Observations:

1. Very finicky :(
2. Even if I use a little bit of the physically-based energy term, it throws everything off.
3. Maybe that's because the lambdas and deltas aren't big enough, but if I make them big then matrix becomes
   super unconditioned and inverses aren't calculated properly.

Very important note: What you're seeing here is NOT a set of wiggly splines. It is the final vector-valued displacement
                     function which is being plotted component-wise.


Next steps:
1. Make it work for velocity constraints.
2. Make it work for arbitrary partial keyframes.
3. Make it work for 2D mesh... yikes

'''
import numpy.linalg
import scipy.integrate
from scipy.optimize import minimize
from scipy import integrate
from scipy.linalg import eigh
import numpy as np
import tkinter as tk
from functools import partial
import random
import time
from sympy import *

# Window size
n = 10
window_w = int(2 ** n)
window_h = int(2 ** n)
np.set_printoptions(suppress=True)

# Tkinter Setup
root = tk.Tk()
root.title("2D Wiggly Spline, Partially Keyframed + Found via Energy Minimization")
root.attributes("-topmost", True)
root.geometry(str(window_w) + "x" + str(window_h))  # window size hardcoded
w = tk.Canvas(root, width=window_w, height=window_h)
w.configure(background='black')
w.pack()

# Epsilon
epsilon = 1e-12


# Coordinate Shift
def A(x, y):
    return np.array([x + window_w / 2, -y + window_h / 2])


# Basis functions
def b(t, delta, lamb, i):
    global epsilon
    """
        t: the input value
        delta, lamb: real numbers, parameters
        i: integer in [0,3] denoting which basis fn out of the four to use

        NOTE: Here is the sign convention:
        i  |  sign1, sign2
        0  |  +, +
        1  |  +, -
        2  |  -, +
        3  |  -, -
    """
    assert np.power(delta, 2) != lamb or (lamb == 0 and delta == 0)
    assert 0 <= i <= 3

    sign1 = -1 if i <= 1 else 1
    sign2 = 1 if i % 2 == 0 else -1

    # Special cases (TODO: include more!)
    if abs(delta) < epsilon and abs(lamb) < epsilon:
        return np.power(t, i)

    if abs(delta) > epsilon > abs(lamb):
        if i == 0:
            return 1
        if i == 1:
            return t
        if i == 2:
            return np.exp(-2 * delta * t) / (4 * np.power(delta, 2))
        if i == 3:
            return np.exp(2 * delta * t) / (4 * np.power(delta, 2))

    # Normal cases
    if np.power(delta, 2) > lamb:
        return np.exp(t * ((sign1 * delta) + (sign2 * np.sqrt(np.power(delta, 2) - lamb))))
    else:
        if sign2 == 1:
            return np.exp(sign1 * t * delta) * np.cos(t * np.sqrt(-np.power(delta, 2) + lamb))

        return np.exp(sign1 * t * delta) * np.sin(t * np.sqrt(-np.power(delta, 2) + lamb))


# First derivatives of basis functions
def b_dot(t, delta, lamb, i):
    assert np.power(delta, 2, ) != lamb or (lamb == 0 and delta == 0)
    assert 0 <= i <= 3

    sign1 = -1 if i <= 1 else 1
    sign2 = 1 if i % 2 == 0 else -1

    # Special cases (TODO: include more!)
    if abs(delta) < epsilon and abs(lamb) < epsilon:
        return 0 if i == 0 else i * np.power(t, i - 1)

    if abs(delta) > epsilon > abs(lamb):
        if i == 0:
            return 0
        if i == 1:
            return 1
        if i == 2:
            return np.exp(-2 * delta * t) / (-2 * delta)
        if i == 3:
            return np.exp(2 * delta * t) / (2 * delta)

    if np.power(delta, 2) > lamb:
        expr = (sign1 * delta) + (sign2 * np.sqrt(np.power(delta, 2) - lamb))
        return np.exp(t * expr) * expr
    else:
        expr = np.sqrt(-np.power(delta, 2) + lamb)

        if sign2 == 1:
            return np.exp(sign1 * t * delta) * ((sign1 * delta * np.cos(t * expr)) - (expr * np.sin(t * expr)))

        return -np.exp(sign1 * t * delta) * (-(expr * np.cos(t * expr)) - (sign1 * delta * np.sin(t * expr)))


# Second derivatives of basis functions
def b_ddot(t, delta, lamb, i):
    assert np.power(delta, 2, ) != lamb
    assert 0 <= i <= 3

    sign1 = -1 if i <= 1 else 1
    sign2 = 1 if i % 2 == 0 else -1

    # Special cases (TODO: include more!)
    if abs(delta) < epsilon and abs(lamb) < epsilon:
        return 0 if i <= 1 else i * (i - 1) * np.power(t, i - 2)

    if abs(delta) > epsilon > abs(lamb):
        if i == 0 or i == 1:
            return 0
        if i == 2:
            return np.exp(-2 * delta * t)
        if i == 3:
            return np.exp(2 * delta * t)

    if np.power(delta, 2) > lamb:
        expr = (sign1 * delta) + (sign2 * np.sqrt(np.power(delta, 2) - lamb))
        return np.exp(t * expr) * np.power(expr, 2)
    else:
        expr = np.sqrt(-np.power(delta, 2) + lamb)
        dsqExpr = (2 * np.power(delta, 2)) - lamb

        if sign2 == 1:
            return np.exp(t * sign1) * ((dsqExpr * np.cos(t * expr)) + (2 * delta * expr * (-sign1) * np.sin(t * expr)))

        return -(-sign1 * np.exp(sign1 * t * delta) * (
                    (2 * delta * expr * np.cos(t * expr)) + (sign1 * dsqExpr * np.sin(t * expr))))


# Piecewise Integrand for 1D Energy Function
def integrand(t, delta, lamb, g, keyframes, coeffs):
    w_ddot = wiggly_ddot(t, delta, lamb, g, keyframes, coeffs)
    w_dot = wiggly_dot(t, delta, lamb, g, keyframes, coeffs)
    w = wiggly(t, delta, lamb, g, keyframes, coeffs)
    return np.power(w_ddot + (2 * delta * w_dot) + (lamb * w) + g, 2.) * 0.5


# 1D Integral Energy Function
def integral_energy_1d(delta, lamb, g, keyframes, coeffs):
    a, b = keyframes[0][0], keyframes[-1][0]
    simple_integrand = partial(integrand, delta=delta, lamb=lamb, g=g, keyframes=keyframes, coeffs=coeffs)
    return scipy.integrate.quad(simple_integrand, a, b)[0]


# Main Integral Energy Funtion to minimize
def integral_energy(deltas, lambs, g, keyframes, coeffs):
    total = 0
    for delta, lamb in zip(deltas, lambs):
        total += integral_energy_1d(delta, lamb, g, keyframes, coeffs)

    return total


# 1D Keyframe Energy Function to minimize
# TODO: You didn't fix this!
def keyframe_energy(deltas, lambs, g, keyframes, coeffs, eigvecs, cA=1., cB=1.):
    m, d = len(keyframes)-1, len(keyframes[0][1])-1
    total = 0.
    for i, frame in enumerate(keyframes):
        t, positions, velocities = frame[0], frame[1], frame[2]
        # Compute u(tk), udot(tk)
        u, udot = np.zeros(len(eigvecs[0])), np.zeros(len(eigvecs[0]))
        for j in range(d):
            coeffs_j = coeffs[4*m*j:4*m*(j+1)]
            omega = wiggly(t, deltas[j], lambs[j], g, keyframes, coeffs_j)
            omega_dot = wiggly_dot(t, deltas[j], lambs[j], g, keyframes, coeffs_j)
            u += omega * eigvecs[j]
            udot += omega_dot * eigvecs[j]

        # Build Ak, Bk, ak, bk matrices
        Ak, Bk, ak, bk = [], [], [], []
        for j in range(len(u)):
            p, v = positions[j], velocities[j]
            if p is not None:
                Ak.append([1 if k==j else 0 for k in range(len(u))])
                ak.append(p)
            if v is not None:
                Bk.append([1 if k == j else 0 for k in range(len(u))])
                bk.append(v)

        Ak, Bk, ak, bk = np.array(Ak), np.array(Bk), np.array(ak), np.array(bk)
        total += cA * np.dot(np.dot(Ak, u) - ak, np.dot(Ak, u) - ak)
        if len(Bk) != 0:
            total += cB * np.dot(np.dot(Bk, udot) - bk, np.dot(Bk, udot) - bk)

    return total

# Construct M and K for any number of degrees of freedom & any connectivity info
def get_MK_mats(mass, stiff, edges, n_verts, dof=2):
    '''
        mass, stiff: floats, m and k (all springs and masses will have same value)
        n_verts: int, number of total vertices
        edges: list of tuples (vi, vj) for i != j describing connection from vertex i to vertex j
        dof: int, degrees of freedom per vertex (in this file, we're in 2D so dof=2)

        NOTE: The vertices are numbered in the edge tuples as they appear in the u(t) vector.
    '''

    M, K = np.diag(np.ones(dof * n_verts) * mass), np.zeros((dof * n_verts, dof * n_verts))

    # Build equations one by one (or rather, in groups of size dof)
    row = 0
    for i in range(n_verts):
        for e in edges:
            if i in e:
                j = e[0] if i!=e[0] else e[1]
                K[row][row] -= stiff
                K[row][dof*j] += stiff

        for k in range(1, dof):
            for col in range(n_verts*dof):
                K[row+k][col] = K[row+k-1][(col-1)%(n_verts*dof)]

        row += dof

    return M, K


# Construct mass and stiffness matrices (for 2 & 3 spring cases, in 2D)
def get_mass_stiffness_mats(m_vals, k_vals):
    '''
        m_vals = [m1, m2] or [m1, m2, m3]
        k_vals = [k] or [k1, k2]
    '''

    M, K = None, None
    if len(k_vals) == 1:
        k = k_vals[0]
        K = np.array([[-k, 0, k, 0], [0, -k, 0, k], [k, 0, -k, 0], [0, k, 0, -k]])
        m1, m2 = m_vals[0], m_vals[1]
        M = np.diag([m1, m1, m2, m2])

    # For 3 masses in a line in 2D (TODO: FIX, IT MIGHT BE WRONG! INSPECT IT AGAIN)
    if len(k_vals) == 2:
        k_1, k_2 = k_vals[0], k_vals[1]
        K = np.array([
            [-k_1 - k_2, 0, k_1, 0, k_2, 0],
            [0, -k_1 - k_2, 0, k_1, 0, k_2],
            [k_1, 0, -k_1 - k_2, 0, k_2, 0],
            [0, k_1 - k_2, 0, -k_1, 0, k_2],
            [k_2, 0, k_1, 0, -k_1 - k_2, 0],
            [0, k_2, 0, k_1, 0, -k_1 - k_2]
        ])
        m1, m2, m3 = m_vals[0], m_vals[1], m_vals[2]
        M = np.diag([m1, m1, m2, m2, m3, m3])

    # For 3-mass Triangle in 2D
    if len(k_vals) == 3:
        k_1, k_2, k_3 = k_vals[0], k_vals[1], k_vals[2]
        K = np.array([
            [-k_1 - k_3, 0, k_1, 0, k_3, 0],
            [0, -k_1 - k_3, 0, k_1, 0, k_3],
            [k_1, 0, -k_1 - k_2, 0, k_2, 0],
            [0, k_1, 0, -k_1 - k_2, 0, k_2],
            [k_3, 0, k_2, 0, -k_2 - k_3, 0],
            [0, k_3, 0, k_2, 0, -k_2 - k_3]
        ])
        m1, m2, m3 = m_vals[0], m_vals[1], m_vals[2]
        M = np.diag([m1, m1, m2, m2, m3, m3])

    return M, K


# Compute A submatrix and B subvector for single given wiggly spline
def compute_portion_of_AB(keyframes, delta, lamb, g, spline_num):
    '''
        Build A submatrix and B subvector based on continuity conditions (determined by keyframes) ONLY.
        NO BOUNDARY CONDITIONS.

        Additional parameters:
        spline_num: For debugging purposes
    '''
    n_coeffs = 4 * (len(keyframes) - 1)  # for this particular spline
    n_constraints = 0  # already filled boundary constraints before this function is called!
    for i in range(1, len(keyframes)-1):
        frame = keyframes[i]
        n_constraints += 3 if frame[2].count(None) == len(frame[2]) else 2   # plus 3 continuity constraints per middle keyframe

    # Full matrix vector equation is Aw = B
    A = np.zeros((n_constraints, n_coeffs))
    B = np.zeros(n_constraints)
    c = g / abs(lamb) if lamb != 0 else 0  # extra constant, we include this into the eqns by modifying B

    # Part i) Iterate over keyframes, filling in the rows of A and B, one by one.
    row = 0
    for i in range(1, len(keyframes)-1):
        # Get keyframe data for this time stamp
        frame = keyframes[i]
        t, p_vals, v_vals = frame[0], frame[1], frame[2]

        # Correct continuity conditions must be satisfied at middle keyframes.
        must_be_C2 = v_vals.count(None) == len(v_vals)

        # Set B
        B[row] = 0
        B[row+1] = 0
        if must_be_C2:
            B[row+2] = 0

        # Set A
        for col in range(n_coeffs):
            pos = b(t, delta, lamb, col % 4)
            deriv = b_dot(t, delta, lamb, col % 4)
            secondDeriv = b_ddot(t, delta, lamb, col % 4)
            if 4 * (i-1) <= col < 4 * i:
                A[row][col] = pos
                A[row+1][col] = deriv
                if must_be_C2:
                    A[row+2][col] = secondDeriv
            elif 4 * i <= col < 4 * (i+1):
                A[row][col] = -pos
                A[row+1][col] = -deriv
                if must_be_C2:
                    A[row+2][col] = -secondDeriv
            else:
                A[row][col] = 0
                A[row+1][col] = 0
                if must_be_C2:
                    A[row+2][col] = 0

        row += 3 if must_be_C2 else 2

    return A, B

def compute_eigenstuff(M, K):
    # Solve generalized eigenvalue problem
    eigvals, eigvecs = eigh(K, M, eigvals_only=False)  # Note: Eigenvectors are returned ROW-WISE!
    eigvecs = np.transpose(eigvecs)
    eigvecs *= -1; eigvals *= -1
    # # Discard final eigenvalue of 0 (and corresponding eigenvector)
    # eigvecs = eigvecs[:-1]
    # eigvals = eigvals[:-1]
    return eigvals, eigvecs

def compute_lambs_deltas(eigvals, alpha, beta):
    lambs = np.array(eigvals)
    deltas = ((lambs * beta) + alpha) * 0.5
    return lambs, deltas


# Coefficient computation (Using boundary full keyframes, appropriate continuity conditions, and energy minimization.)
def compute_coefficients(keyframes, M, K, eigvecs, eigvals, lambs, deltas, g):
    '''
        Params:
            keyframes: array with entries of the form (time, (pos_mass1, pos_mass2, pos_mass3), ((vel_mass1, vel_mass2, vel_mass3))),
                       where pos_mass and vel_mass will be of type None if no partial keyframe data for particular time.
            TODO
            g: float, should be 0 for this exanple, but in general it is represents external force like gravity
            alpha, beta: The alpha and beta as part of the Rayleigh damping term.

        Note:

        Example — Here is what each row in a 3md+d x 4md = (3*2*2)+2 x (4*2*2) = 14 x 16 matrix represents
                  (for 2 wiggly splines (two masses, one spring), 2 segments, 3 keyframes). Suppose the 1 middle
                  keyframe has only a position specified, and no tangent (meaning splines satisfy all 3 continuity
                  conditions there).

        1. Segment 1 of spline 1 must satisfy start positional boundary condition (in modal coordinates)
        2. Segment 2 of spline 1 must satisfy end positional boundary condition (in modal coordinates)
        3. Segment 1 of spline 2 must satisfy start positional boundary condition (in modal coordinates)
        4. Segment 2 of spline 2 must satisfy end positional boundary condition (in modal coordinates)

        5. Segment 1 of spline 1 must satisfy start velocity boundary condition (in modal coordinates)
        6. Segment 2 of spline 1 must satisfy end velocity boundary condition (in modal coordinates)
        7. Segment 1 of spline 2 must satisfy start velocity boundary condition (in modal coordinates)
        8. Segment 2 of spline 2 must satisfy end velocity boundary condition (in modal coordinates)

        9. Spline 1 must be C0 continuous at middle timestamp.
        10. Spline 1 must be C1 continuous at middle timestamp.
        11. Spline 1 must be C2 continuous at middle timestamp.
        12. Spline 2 must be C0 continuous at middle timestamp.
        13. Spline 2 must be C1 continuous at middle timestamp.
        14. Spline 2 must be C2 continuous at middle timestamp.


        Remark: If, at any knot (keyframe), ANY of the masses have velocities prescribed (in addition to the positions),
                then ALL wiggly splines have to be only C0, C1 continuous (not C2).

    '''
    assert len(keyframes) >= 2
    # assert len(m_vals) - 1 == len(k_vals) >= 1
    m = len(keyframes) - 1
    d = len(eigvals)

    # Construct the giant A matrix and giant B vector, portion by portion, per wiggly spline
    # Part a) Initialize them, and compute d total values for constant c, lambdas, and deltas
    n_rows, n_cols = ((3 * m * d) + d), 4 * m * d    # assuming C2 everywhere. maybe that's why velocity ones don't work yet
    A = np.zeros((n_rows, n_cols))
    B = np.zeros(n_rows)
    c = np.array([g / abs(l) if abs(l) != 0 else 0 for l in eigvals])
    # Part b) Add the 4d positional boundary conditions to A and B
    # Part bi) Add the 2d positional ones (first d at the starting keyframe, then d at the ending)
    B[:d] = np.dot(np.dot(eigvecs, M), keyframes[0][1]) + c
    print('Initial Positions in Modal Coords:', np.dot(np.dot(eigvecs, M), keyframes[0][1]) + c)
    for row in range(d):
        for col_grp in range(0, n_cols - (4 * m) + 1, 4 * m):
            A[row][col_grp:col_grp + (4 * m)] = np.array([b(keyframes[0][0], deltas[row], lambs[row], i % 4) if i<=3 else 0 for i in range(4 * m)]) if row*4*m == col_grp else np.zeros(4*m)

    B[d:2*d] = np.dot(np.dot(eigvecs, M), keyframes[-1][1]) + c
    print('Final Positions in Modal Coords:', np.dot(np.dot(eigvecs, M), keyframes[-1][1]))
    for row in range(d, 2*d):
        for col_grp in range(0, n_cols - (4 * m) + 1, 4 * m):
            A[row][col_grp:col_grp + (4 * m)] = np.array([b(keyframes[-1][0], deltas[row-d], lambs[row-d], i % 4) if i>=(4*m)-4 else 0 for i in range(4 * m)]) if (row-d)*4*m == col_grp else np.zeros(4*m)

    # Part bii) Add the 2d velocity ones (d at the starting keyframe, d at the ending)
    B[2*d:3*d] = np.dot(np.dot(eigvecs, M), keyframes[0][2])
    print('Initial Tangents in Modal Coords:', np.dot(np.dot(eigvecs, M), keyframes[0][2]) + c)
    for row in range(2*d, 3*d):
        for col_grp in range(0, n_cols - (4 * m) + 1, 4 * m):
            A[row][col_grp:col_grp + (4 * m)] = np.array([b_dot(keyframes[0][0], deltas[row-(2*d)], lambs[row-(2*d)], i % 4) if i<=3 else 0 for i in range(4 * m)]) if (row - (2*d))*4*m == col_grp else np.zeros(4*m)
    B[3*d:4*d] = np.dot(np.dot(eigvecs, M), keyframes[-1][2])
    print('Final Tangents in Modal Coords:', np.dot(np.dot(eigvecs, M), keyframes[-1][2]))
    for row in range(3*d, 4*d):
        for col_grp in range(0, n_cols - (4 * m) + 1, 4 * m):
            A[row][col_grp:col_grp + (4 * m)] = np.array([b_dot(keyframes[-1][0], deltas[row-(3*d)], lambs[row-(3*d)], i % 4) if i>=(4*m)-4 else 0 for i in range(4 * m)]) if (row - (3*d))*4*m == col_grp else np.zeros(4*m)

    # Part c) For each eigenmode, get the individual A submatrix and B subvector
    for i in range(d):
        # Part ci) Get the raw A submatrix and B subvector
        A_sub, B_sub = compute_portion_of_AB(keyframes, deltas[i], lambs[i], g, spline_num=i)
        # Part ciii) Stick in into the main matrix and vector!
        A[(4*d)+(3*i) : (4*d)+(3*i)+len(A_sub), 4*m*i : (4*m)*(i+1)] = A_sub
        B[(4*d)+(3*i) : (4*d)+(3*i)+len(B_sub)] = B_sub

    #  Use SVD of matrix A to find reduced search space (unless special case of m=1)
    if m!=1:
        subspace_dim = n_cols - n_rows
        _, _, VT = numpy.linalg.svd(A)
        U = np.transpose(VT)[:, (n_cols - subspace_dim):]
        w_star = np.transpose([np.linalg.pinv(A).dot(B)])

        # All solutions are of the form: w = w_star + Uz, for some z of size subspace_dim.
        # Here is the integral energy function to minimize.
        energy_fn1 = lambda z: integral_energy(deltas, lambs, g, keyframes, np.transpose(w_star)[0] + U.dot(z))
        energy_fn2 = lambda z: keyframe_energy(deltas, lambs, g, keyframes, np.transpose(w_star)[0] + U.dot(z), eigvecs, cA=1., cB=1.)
        energy = lambda z: (energy_fn1(z) * 0.001) + energy_fn2(z)  # 0.00001

        # Now we search through the reduced space and return the corresponding coefficients!
        z = minimize(energy, np.zeros((subspace_dim, 1)), options={'disp': True}, method='BFGS')
        return np.transpose(w_star)[0] + U.dot(z.x)

    return np.linalg.inv(A).dot(B)


# Wiggly spline computation
def wiggly(t, delta, lamb, g, keyframes, coeffs):
    # Find correct segment of spline
    idx = 0
    for i in range(len(keyframes) - 1):
        if keyframes[i][0] <= t <= keyframes[i + 1][0]:
            idx = i
            break

    # Evaluate spline
    c = g / abs(lamb) if lamb != 0 else 0
    total = 0
    for i in range(idx, idx + 4):
        total += (coeffs[(4 * idx) + (i - idx)] * b(t, delta, lamb, i - idx))

    return total - c


# Wiggly spline dot
def wiggly_dot(t, delta, lamb, g, keyframes, coeffs):
    # Find correct segment of spline
    idx = 0
    for i in range(len(keyframes) - 1):
        if keyframes[i][0] <= t <= keyframes[i + 1][0]:
            idx = i
            break

    # Evaluate spline derivative
    total = 0
    for i in range(idx, idx + 4):
        total += (coeffs[(4 * idx) + (i - idx)] * b_dot(t, delta, lamb, i - idx))

    return total


# Wiggly spline double dot
def wiggly_ddot(t, delta, lamb, g, keyframes, coeffs):
    # Find correct segment of spline
    idx = 0
    for i in range(len(keyframes) - 1):
        if keyframes[i][0] <= t <= keyframes[i + 1][0]:
            idx = i
            break

    # Evaluate spline double derivative
    total = 0
    for i in range(idx, idx + 4):
        total += (coeffs[(4 * idx) + (i - idx)] * b_ddot(t, delta, lamb, i - idx))

    return total

# Main wiggly spline params
t_start, t_end = 0, 1
pos_absmax = 2  # positions randomly generated, for now (tangents all 0, for now)
alpha, beta = 0.01, 0.01

# # Set keyframes (2 mass 1 spring case, 0 partial keyframe case)
# key1 = (0, [0, 0, 3, 3], [0, 0, 0, 0])   # ordered like: x1, y1, x2, y2
# key3 = (1, [-1, 3, 5, 5], [0, 0, 0, 0])
# keyframes = [key1, key3]

# Set keyframes (2 mass 1 spring case, 1 partial keyframe case)
# key1 = (0, [0, 0, 3, 3], [0, 0, 0, 0])   # ordered like: x1, y1, x2, y2
# partial_key2 = (0.5, [None, None, 7, 8], [None, None, None, None])   # Try: t=0.5: [5, None, None] & [None, None, 5], t=0.2: [None, None, -5], & t=0.5: [None, -2, 5] doesn't properly do it unless change energy
# key3 = (1, [-1, 3, 5, 5], [0, 0, 0, 0])
# keyframes = [key1, partial_key2, key3]

# Set keyframes (3 mass 2 spring case, 1 partial keyframe case)
# key1 = (0, [0, 0, 3, 3, 1, 1], [0, 0, 0, 0, 0, 0])   # ordered like: x1, y1, x2, y2, x3, y3
# partial_key2 = (0.5, [None, None, 6, 8, None, None], [None, None, None, None, None, None])
# key3 = (1, [-1, 3, 5, 5, -2, -5], [0, 0, 0, 0, 0, 0])
# keyframes = [key1, partial_key2, key3]

# Set keyframes (4 mass 6 spring case, 0 partial keyframe case)
key1 = (0, [0, 0,
            0, 0,
            0, 0,
            0, 0], [0, 0, 0, 0, 0, 0, 0, 0])

partial_key2 = (0.5, [0, -10,
                      0, -13,
                      None, None,
                      None, None], [None, None, None, None, None, None, None, None])

key3 = (1, [0, 0,
            0, 0,
            0, 0,
            0, 0], [0, 0, 0, 0, 0, 0, 0, 0])

keyframes = [key1, partial_key2, key3]

mass, stiff = 1, 50
# m_vals = [mass, mass, mass]
# k_vals = [stiff, stiff]
g = 0  # assume non-existent gravity (for this example)

# Compute splines
# M, K = get_mass_stiffness_mats(m_vals, k_vals)
edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (3, 1)]
M, K = get_MK_mats(mass, stiff, edges, n_verts=4)
eigvals, eigvecs = compute_eigenstuff(M, K)
lambs, deltas = compute_lambs_deltas(eigvals, alpha, beta)
print('Eigenvals:', eigvals)
print('Eigenvecs:', eigvecs)
print('Lambdas:', lambs)
print('Deltas:', deltas)
coeffs = compute_coefficients(keyframes, M, K, eigvecs, eigvals, lambs, deltas, g)
m, d = len(keyframes)-1, len(eigvals)

# Other params
first_run = True
scale_x, scale_y = 500, 50
pt_radius = 10
tg_linelen = 30
sample_size = 0.01
colors = ['red', 'blue', 'green', 'cyan', 'purple', 'orange', 'yellow', 'magenta']

# Mass spring graphics params
# p0 = [0, -10, 5, -10]  # initial positions
# p0 = [0, -10, 5, -10, 10, -10]  # initial positions
p0 = [0, 0,
      10, 0,
      10, 10,
      0, 10]  # initial positions
cube_apothem = 20
sim_scale = cube_apothem * 1.5
t_curr = 0
dt = 0.005


# Full function u(t) computation
def u(t, deltas, lambs, g, eigvals, eigvecs, keyframes, coeffs):
    global m, d
    total = np.zeros(d)
    for i in range(d):
        delta, lamb = deltas[i], lambs[i]
        coeffs_i = coeffs[4 * m * i:4 * m * (i + 1)]
        total += wiggly(t, delta, lamb, g, keyframes, coeffs_i) * eigvecs[i]

    return total

# Key bind
def key_pressed(event):
    global t_curr

    if event.char == 't':
        t_curr = 0


# Key binding
w.bind("<KeyPress>", key_pressed)
w.bind("<1>", lambda event: w.focus_set())
w.pack()


# Main Loop
def run_step():
    # Get references to global vars / params
    global first_run, keyframes, coeffs, eigvecs, eigvals, lambs, deltas, scale_x, scale_y,\
           pt_radius, tg_linelen, sample_size, t_start, t_end, m, d, p0, cube_apothem, dt, t_curr, sim_scale, edges

    # Stupid Precomputations
    radiusBy2 = pt_radius / 2.
    scale_xBy2 = scale_x / 2.

    # # Plot keyframe points
    # for i, frame in enumerate(keyframes):
    #     t, y_vals = frame[0], frame[1]
    #     for j, y_val in enumerate(y_vals):
    #         if y_val is not None:
    #             x, y = t * scale_x, y_val * scale_y
    #             if first_run:
    #                 w.create_oval(*A(x - radiusBy2 - scale_xBy2, y - radiusBy2), *A(x + radiusBy2 - scale_xBy2, y + radiusBy2),
    #                               fill=colors[j], outline='white', tag='pt' + str(i)+str(j))
    #             else:
    #                 oval = w.find_withtag('pt' + str(i)+str(j))
    #                 w.coords(oval, *A(x - radiusBy2 - scale_xBy2, y - radiusBy2), *A(x + radiusBy2 - scale_xBy2, y + radiusBy2))

    # # Compute trajectory curves
    # curves = [[] for _ in range(d)]
    # for t in np.arange(t_start, t_end + sample_size, sample_size):
    #     val = u(t, deltas, lambs, g, eigvals, eigvecs, keyframes, coeffs)
    #     for i, ui in enumerate(val):
    #         x, y = t * scale_x, ui * scale_y
    #         curves[i].extend(A(x, y) - np.array([scale_x / 2, 0]))

    # # Plot them
    # for i, curve in enumerate(curves):
    #     if first_run:
    #         w.create_line(curve, fill=colors[i], tag='line'+str(i))
    #     else:
    #         w.coords(w.find_withtag('line'+str(i)), curve)

    # Compute cube positions + plot springs
    pos = p0 + u(t_curr, deltas, lambs, g, eigvals, eigvecs, keyframes, coeffs)
    pos = np.reshape(pos, (int(len(pos) / 2), 2))
    for i, e in enumerate(edges):
        p1, p2 = pos[e[0]], pos[e[1]]
        x1, y1, x2, y2 = p1[0] * sim_scale, p1[1] * sim_scale, p2[0] * sim_scale, p2[1] * sim_scale
        if first_run:
            w.create_line(*A(x1, y1), *A(x2, y2), fill='white', tags='spring'+str(i))
        else:
            w.coords(w.find_withtag('spring'+str(i)), *A(x1, y1), *A(x2, y2))

    # Plot cubes
    for i, val in enumerate(pos):
        x = val[0] * sim_scale  # scaling factor
        y = val[1] * sim_scale
        if first_run:
            w.create_rectangle(*A(x - cube_apothem, y - cube_apothem),
                               *A(x + cube_apothem, y + cube_apothem), fill=colors[i], outline='white',
                               tags='cube' + str(i))
        else:
            w.coords(w.find_withtag('cube' + str(i)), *A(x - cube_apothem, y - cube_apothem),
                     *A(x + cube_apothem, y + cube_apothem))

    # Plot keyframe bars
    for i, frame in enumerate(keyframes):
        posData = np.reshape(frame[1], (int(len(frame[1])/2), 2))
        for j, disp in enumerate(posData):
            disp_x, disp_y = disp[0], disp[1]
            if disp_x is not None:   # EITHER BOTH ARE NONE OR BOTH ARE GIVEN (think about it)
                col = colors[j]
                if i == 0:
                    col = 'white'
                elif i == len(keyframes) - 1:
                    col = 'yellow'

                x = (p0[2*j] + disp_x) * sim_scale  # scale factor
                y = (p0[(2*j)+1] + disp_y) * sim_scale  # scale factor
                lenBy2 = cube_apothem * 2
                if first_run:
                    w.create_line(*A(x, y + lenBy2), *A(x, y - lenBy2), fill=col, tags='bar' + str(i) + str(j), width=2)
                    w.create_line(*A(x + lenBy2, y), *A(x - lenBy2, y), fill=col, tags='bar' + str(i) + str(j), width=2)

    # Plot text
    if first_run:
        w.create_text(*A(0, window_h / 2 - 100), font="AvenirNext 30", text='t='+str(t_curr)[:5], tag='text')
    else:
        w.itemconfig(w.find_withtag('text'), text='t='+str(t_curr)[:5])

    # End run
    first_run = False
    # time.sleep(0.001)
    t_curr = min(t_curr + dt, 1)
    w.update()


if __name__ == '__main__':
    while True:
        run_step()


tk.mainloop()