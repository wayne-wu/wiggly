'''
Observations:

1. As the lambda parameter changes, the fit changes quite chaotically.

'''

from scipy.optimize import minimize
import numpy as np
import tkinter as tk
import random

# Window size
n = 10
window_w = int(2**n)
window_h = int(2**n)
np.set_printoptions(suppress=True)

# Epsilon
epsilon = 1e-12

# Tkinter Setup
root = tk.Tk()
root.title("1D Wiggly Spline, Only Positions Specified at Non-Boundaries (C2)")
root.attributes("-topmost", True)
root.geometry(str(window_w) + "x" + str(window_h))  # window size hardcoded
w = tk.Canvas(root, width=window_w, height=window_h)
w.configure(background='black')
w.pack()


# Coordinate Shift
def A(x, y):
    return np.array([x + window_w/2, -y + window_h/2])


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


# Coefficient computation (Using x = (A^-1)*b.)
def compute_coefficients(keyframes, delta, lamb, g):
    '''
        Note:

        Example — Here is what each row in a 8 x 8 matrix represents (2 segments, 3 keyframes).
        Suppose the middle keyframe only has a specified position (no specified tangent)

        1. Segment 1 must satisfy positional boundary condition
        2. Segment 1 must satisfy velocity (tangent) boundary condition
        3. Segment 1 must satisfy middle positional keyframe constraint
        4. Segment 2 must satisfy middle positional keyframe constraint
        5. Spline must be C1 continuous at middle timestamp.
        6. Spline must be C2 continuous at middle timestamp.
        7. Segment 2 must satisfy positional boundary condition
        8. Segment 2 must satisfy velocity (tangent) boundary condition

        Remark: Constraints 3-6 (four of them) are given by the middle keyframe. Any additional keyframe
                in the middle should add four similar such constraints too.
    '''
    assert len(keyframes) >= 2
    n_coeffs = 4 * (len(keyframes) - 1)

    # Matrix vector equation to solve is Aw = B
    A = np.zeros((n_coeffs, n_coeffs))
    B = np.zeros(n_coeffs)
    c = g / abs(lamb) if lamb != 0 else 0  # extra constant, we include this into the eqns by modifying B

    # Each segment of spline has 4 coefficients.
    # In this case, each coefficient has a corresponding equation.
    row = 0
    for i, frame in enumerate(keyframes):
        t, p, v = frame[0], frame[1], frame[2]

        # 1. Segment left of point must satisfy given position.
        if i != 0:
            B[row] = p + c
            for col in range(n_coeffs):
                if 4 * (i-1) <= col < 4 * i:
                    A[row][col] = b(t, delta, lamb, col % 4)
                else:
                    A[row][col] = 0
            
            row += 1

        # 2. Segment right of points must satisfy given position.
        if i != len(keyframes)-1:
            B[row] = p + c
            for col in range(n_coeffs):
                if 4 * i <= col < 4 * (i+1):
                    A[row][col] = b(t, delta, lamb, col % 4)
                else:
                    A[row][col] = 0

            row += 1

        # 3. Given tangent at boundaries must be satisfied by respective edge segments.
        if i == 0:
            assert v is not None
            B[row] = v
            A[row] = np.array([b_dot(t, delta, lamb, j) if j <= 3 else 0 for j in range(n_coeffs)])
            row += 1

        if i == len(keyframes)-1:
            assert v is not None
            B[row] = v
            A[row] = np.array([b_dot(t, delta, lamb, j % 4) if j >= n_coeffs-4 else 0 for j in range(n_coeffs)])
            row += 1

        # 4. Left and right segments must match in first & second derivatives.
        if i != 0 and i != len(keyframes)-1:
            B[row] = 0
            B[row+1] = 0
            for col in range(n_coeffs):
                deriv = b_dot(t, delta, lamb, col % 4)
                secondDeriv = b_ddot(t, delta, lamb, col % 4)
                if 4 * (i-1) <= col < 4 * i:
                    A[row][col] = deriv
                    A[row+1][col] = secondDeriv
                elif 4 * i <= col < 4 * (i+1):
                    A[row][col] = -deriv
                    A[row+1][col] = -secondDeriv
                else:
                    A[row][col] = 0
                    A[row+1][col] = 0

            row += 2

    return np.linalg.inv(A).dot(B)


# Wiggly spline computation
def wiggly(t, delta, lamb, g, keyframes, coeffs):
    # Find correct segment of spline
    idx = 0
    for i in range(len(keyframes)-1):
        if keyframes[i][0] <= t <= keyframes[i+1][0]:
            idx = i
            break

    # Evaluate spline
    c = g / abs(lamb) if lamb != 0 else 0
    total = 0
    for i in range(idx, idx+4):
        total += (coeffs[(4 * idx) + (i - idx)] * b(t, delta, lamb, i - idx))

    return total - c


# Main wiggly spline params (full keyframes case)
t_start, t_end = 0, 1
n_frames = 4    # must be >=2
pos_absmax = 2  # positions randomly generated, for now (tangents all 0, for now)
delta, lamb, g = 20, 0, 0
keyframes = [(i / (n_frames-1), 0 if i % 2 == 0 else 1, None if i != 0 and i != n_frames-1 else 0) for i in range(n_frames)]  # No tangent data specified  # pos_absmax * ((random.random() * 2) - 1)

# Compute spline
coeffs = compute_coefficients(keyframes, delta, lamb, g)

# Other params
first_run = True
scale_x, scale_y = 500, 100
pt_radius = 10
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
    global first_run, keyframes, coeffs, scale_x, scale_y, pt_radius, sample_size, t_start, t_end, delta, lamb, g

    # Plot keyframe points
    for i, frame in enumerate(keyframes):
        x, y = frame[0] * scale_x, frame[1] * scale_y
        radiusBy2 = pt_radius / 2.
        scale_xBy2 = scale_x / 2.
        if first_run:
            w.create_oval(*A(x - radiusBy2 - scale_xBy2, y - radiusBy2), *A(x + radiusBy2 - scale_xBy2, y + radiusBy2), fill='red', outline='white', tag='pt' + str(i))
        else:
            oval = w.find_withtag('pt'+str(i))
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
        w.create_text(*A(0, -window_h/2 + 100), font="AvenirNext 30", text='λ:'+str(lamb)+', δ:'+str(delta)+', g:'+str(g), tag='text')
    else:
        w.itemconfig(w.find_withtag('text'), text='λ:'+str(lamb)+', δ:'+str(delta)+', g:'+str(g))

    # End run
    first_run = False
    w.update()


if __name__ == '__main__':
    # while True:
    #     run_step()
    lamb, delta, t = 15, 3, 0.3
    print([b(t, delta, lamb, i) for i in range(4)])
    print([b_dot(t, delta, lamb, i) for i in range(4)])
    print([b_ddot(t, delta, lamb, i) for i in range(4)])
    print()

    lamb, delta, t = 15, 0, 0.3
    print([b(t, delta, lamb, i) for i in range(4)])
    print([b_dot(t, delta, lamb, i) for i in range(4)])
    print([b_ddot(t, delta, lamb, i) for i in range(4)])
    print()
    
    lamb, delta, t = 0, 15, 0.3
    print([b(t, delta, lamb, i) for i in range(4)])
    print([b_dot(t, delta, lamb, i) for i in range(4)])
    print([b_ddot(t, delta, lamb, i) for i in range(4)])

tk.mainloop()