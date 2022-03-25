'''
Observations:

1. The springier I make it (i.e. the more I increase lambda), the lesser of an impact the physically-based energy term
   needs to have, otherwise the middle points won't be interpolated exactly.

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

# Window size
n = 10
window_w = int(2 ** n)
window_h = int(2 ** n)
np.set_printoptions(suppress=True)

# Tkinter Setup
root = tk.Tk()
root.title("1D Wiggly Spline, Partially Keyframed + Found via Energy Minimization")
root.attributes("-topmost", True)
root.geometry(str(window_w) + "x" + str(window_h))  # window size hardcoded
w = tk.Canvas(root, width=window_w, height=window_h)
w.configure(background='black')
w.pack()


# Coordinate Shift
def A(x, y):
    return np.array([x + window_w / 2, -y + window_h / 2])


# Basis functions
def b(t, delta, lamb, i):
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

    sign1 = 1 if i <= 1 else -1
    sign2 = 1 if i % 2 == 0 else -1

    # Special cases (TODO: include more!)
    if delta == 0 and lamb == 0:
        return np.power(t, i)

    # Normal cases
    if np.power(delta, 2) > lamb:
        return np.exp(t * ((sign1 * delta) + (sign2 * np.sqrt(np.power(delta, 2) - lamb))))
    else:
        if sign2 == 1:
            return np.exp(sign1 * t * delta) * np.cos(t * np.sqrt(-np.power(delta, 2) + lamb))

        return -np.exp(sign1 * t * delta) * np.sin(t * np.sqrt(-np.power(delta, 2) + lamb))


# First derivatives of basis functions
def b_dot(t, delta, lamb, i):
    assert np.power(delta, 2, ) != lamb or (lamb == 0 and delta == 0)
    assert 0 <= i <= 3

    sign1 = 1 if i <= 1 else -1
    sign2 = 1 if i % 2 == 0 else -1

    # Special cases (TODO: include more!)
    if delta == 0 and lamb == 0:
        return 0 if i == 0 else i * np.power(t, i - 1)

    if np.power(delta, 2) > lamb:
        expr = (sign1 * delta) + (sign2 * np.sqrt(np.power(delta, 2) - lamb))
        return np.exp(t * expr) * expr
    else:
        expr = np.sqrt(-np.power(delta, 2) + lamb)

        if sign2 == 1:
            return np.exp(sign1 * t * delta) * ((sign1 * delta * np.cos(t * expr)) - (expr * np.sin(t * expr)))

        return np.exp(sign1 * t * delta) * (-(expr * np.cos(t * expr)) - (sign1 * delta * np.sin(t * expr)))


# Second derivatives of basis functions
def b_ddot(t, delta, lamb, i):
    assert np.power(delta, 2, ) != lamb
    assert 0 <= i <= 3

    sign1 = 1 if i <= 1 else -1
    sign2 = 1 if i % 2 == 0 else -1

    # Special cases (TODO: include more!)
    if delta == 0 and lamb == 0:
        return 0 if i <= 1 else i * (i - 1) * np.power(t, i - 2)

    if np.power(delta, 2) > lamb:
        expr = (sign1 * delta) + (sign2 * np.sqrt(np.power(delta, 2) - lamb))
        return np.exp(t * expr) * np.power(expr, 2)
    else:
        expr = np.sqrt(-np.power(delta, 2) + lamb)
        dsqExpr = (2 * np.power(delta, 2)) - lamb

        if sign2 == 1:
            return np.exp(t * sign1) * ((dsqExpr * np.cos(t * expr)) + (2 * delta * expr * (-sign1) * np.sin(t * expr)))

        return -sign1 * np.exp(sign1 * t * delta) * (
                    (2 * delta * expr * np.cos(t * expr)) + (sign1 * dsqExpr * np.sin(t * expr)))


# Piecewise Integrand for 1D Energy Function
def integrand(t, delta, lamb, g, keyframes, coeffs):
    w_ddot = wiggly_ddot(t, delta, lamb, g, keyframes, coeffs)
    w_dot = wiggly_dot(t, delta, lamb, g, keyframes, coeffs)
    w = wiggly(t, delta, lamb, g, keyframes, coeffs)
    return np.power(w_ddot + (2 * delta * w_dot) + (lamb * w) + g, 2.) * 0.5


# 1D Integral Energy Function to minimize
def integral_energy(delta, lamb, g, keyframes, coeffs):
    a, b = keyframes[0][0], keyframes[-1][0]
    simple_integrand = partial(integrand, delta=delta, lamb=lamb, g=g, keyframes=keyframes, coeffs=coeffs)
    return scipy.integrate.quad(simple_integrand, a, b)[0]


# 1D Keyframe Energy Function to minimize
def keyframe_energy(delta, lamb, g, keyframes, coeffs, cA=1., cB=1.):
    total = 0.
    for i in range(1, len(keyframes)-1):
        frame = keyframes[i]
        t, p, v = frame[0], frame[1], frame[2]
        w = wiggly(t, delta, lamb, g, keyframes, coeffs)
        w_dot = wiggly_dot(t, delta, lamb, g, keyframes, coeffs)
        if p is not None:
            total += np.power(w - p, 2.) * cA

        if v is not None:
            total += np.power(w_dot - v, 2.) * cB

    return 0.5 * total


# Coefficient computation (Using boundary full keyframes, appropriate continuity conditions, and energy minimization.)
def compute_coefficients(keyframes, delta, lamb, g):
    '''
        Note:

        Example — Here is what each row in a 7 x 8 matrix represents (2 segments, 3 keyframes).
        Suppose the middle keyframe only has a specified position (no specified tangent).

        1. Segment 1 must satisfy positional boundary condition
        2. Segment 1 must satisfy velocity (tangent) boundary condition
        3. Spline must be C0 continuous at middle timestamp.
        4. Spline must be C1 continuous at middle timestamp.
        5. Spline must be C2 continuous at middle timestamp.
        6. Segment 2 must satisfy positional boundary condition
        7. Segment 2 must satisfy velocity (tangent) boundary condition

        Remark: Constraints 3-5 (three of them) are given by the middle keyframe. Any additional keyframe
                in the middle should add three similar such constraints too.

        Remark 2: Remember that the middle keyframe data is NOT part of this initial matrix!! Only place that is used
                  is in the second term in the energy function.

        Remark 3: In this particular example, since cols - rows = 1, the dimension of the reduced space will be 1.
                  This means the minimization algorithm only needs to find a SCALAR alpha such that w = w_star + U*alpha
    '''
    assert len(keyframes) >= 2
    n_coeffs = 4 * (len(keyframes) - 1)
    n_constraints = 4
    for i in range(1, len(keyframes)-1):
        frame = keyframes[i]
        n_constraints += 3 if frame[2] is None else 2

    # Full matrix vector equation is Aw = B
    A = np.zeros((n_constraints, n_coeffs))
    B = np.zeros(n_constraints)
    c = g / abs(lamb) if lamb != 0 else 0  # extra constant, we include this into the eqns by modifying B

    # Part i) Iterate over keyframes, filling in the rows of A and B, one by one.
    row = 0
    for i, frame in enumerate(keyframes):
        t, p, v = frame[0], frame[1], frame[2]

        # 1, 2. Left boundary conditions must be satisfied.
        if i == 0:
            B[row] = p + c
            B[row+1] = v
            for col in range(n_coeffs):
                if col <= 3:
                    A[row][col] = b(t, delta, lamb, col)
                    A[row+1][col] = b_dot(t, delta, lamb, col)
                else:
                    A[row][col] = 0
                    A[row+1][col] = 0

            row += 2

        # 3, 4, 5. Correct continuity conditions must be satisfied at middle keyframes.
        if i != 0 and i != len(keyframes) - 1:
            must_be_C2 = v is None

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

        # 6, 7. Right boundary conditions must be satisfied.
        if i == len(keyframes)-1:
            B[row] = p + c
            B[row+1] = v
            for col in range(n_coeffs):
                if n_coeffs-4 <= col:
                    A[row][col] = b(t, delta, lamb, col % 4)
                    A[row+1][col] = b_dot(t, delta, lamb, col % 4)
                else:
                    A[row][col] = 0
                    A[row+1][col] = 0

    # Part ii) Use SVD of matrix A to find reduced search space.
    _, _, VT = numpy.linalg.svd(A)
    subspace_dim = n_coeffs - n_constraints
    U = np.transpose(VT)[:, (n_coeffs - subspace_dim):]
    w_star = np.transpose([np.linalg.pinv(A).dot(B)])

    # All solutions are of the form: w = w_star + Uz, for some z of size subspace_dim.
    # Here is the integral energy function to minimize.
    energy_fn1 = lambda z: integral_energy(delta, lamb, g, keyframes, np.transpose(w_star)[0] + U.dot(z))
    energy_fn2 = lambda z: keyframe_energy(delta, lamb, g, keyframes, np.transpose(w_star)[0] + U.dot(z), cA=1., cB=1.)
    energy = lambda z: energy_fn1(z) + np.power(lamb, 2.1) * energy_fn2(z)

    # Now we search through the reduced space and return the corresponding coefficients!
    z = minimize(energy, np.zeros((subspace_dim, 1)), options={'disp': True})
    return np.transpose(w_star)[0] + U.dot(z.x)


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


# Main wiggly spline params (full keyframes case)
t_start, t_end = 0, 1
n_frames = 3  # must be >=2
pos_absmax = 2  # positions randomly generated, for now (tangents all 0, for now)
delta, lamb, g = 2, 200, 1
keyframes = [(i / (n_frames - 1), 0 if i % 2 == 0 else 1, None if i != 0 and i != n_frames - 1 else 0) for i in
             range(n_frames)]  # No tangent data specified  # pos_absmax * ((random.random() * 2) - 1)

# Compute spline
coeffs = compute_coefficients(keyframes, delta, lamb, g)

# Other params
first_run = True
scale_x, scale_y = 500, 100
pt_radius = 10
tg_linelen = 30
sample_size = 0.01


# Key bind
def key_pressed(event):
    global delta, lamb, g, coeffs
    dlamb, ddamp, dg = 10, 10, 5

    if event.char == 'l':
        lamb += dlamb

    if event.char == 'j':
        lamb -= dlamb

    if event.char == 'd':
        delta += ddamp

    if event.char == 'a':
        delta -= ddamp

    if event.char == 'h':
        g += dg

    if event.char == 'g':
        g -= dg

    delta = max(delta, 0)
    coeffs = compute_coefficients(keyframes, delta, lamb, g)


# Key binding
w.bind("<KeyPress>", key_pressed)
w.bind("<1>", lambda event: w.focus_set())
w.pack()


# Main Loop
def run_step():
    # Get references to global vars / params
    global first_run, keyframes, coeffs, scale_x, scale_y, pt_radius, tg_linelen, sample_size, t_start, t_end, delta, lamb, g

    # Plot keyframe points
    for i, frame in enumerate(keyframes):
        x, y = frame[0] * scale_x, frame[1] * scale_y
        radiusBy2 = pt_radius / 2.
        scale_xBy2 = scale_x / 2.
        if first_run:
            w.create_oval(*A(x - radiusBy2 - scale_xBy2, y - radiusBy2), *A(x + radiusBy2 - scale_xBy2, y + radiusBy2),
                          fill='red', outline='white', tag='pt' + str(i))
        else:
            oval = w.find_withtag('pt' + str(i))
            w.coords(oval, *A(x - radiusBy2 - scale_xBy2, y - radiusBy2), *A(x + radiusBy2 - scale_xBy2, y + radiusBy2))

    # Plot spline
    curve = []
    for t in np.arange(t_start, t_end + sample_size, sample_size):
        x, y = t * scale_x, wiggly(t, delta, lamb, g, keyframes, coeffs) * scale_y
        curve.extend(A(x, y) - np.array([scale_x / 2, 0]))

    if first_run:
        w.create_line(curve, fill='white', tag='line')
    else:
        w.coords(w.find_withtag('line'), curve)

    # Text
    if first_run:
        w.create_text(*A(0, -window_h / 2 + 100), font="AvenirNext 30",
                      text='λ:' + str(lamb) + ', δ:' + str(delta) + ', g:' + str(g), tag='text')
    else:
        w.itemconfig(w.find_withtag('text'), text='λ:' + str(lamb) + ', δ:' + str(delta) + ', g:' + str(g))

    # End run
    first_run = False
    w.update()


if __name__ == '__main__':


    eigvals, eigvecs = eigh(A, B, eigvals_only=False, subset_by_index=[0, 1, 2])

    # while True:
    #     run_step()

tk.mainloop()