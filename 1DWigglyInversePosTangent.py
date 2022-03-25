'''
Observations:

1. Delta parameter (damping) is WAY more powerful than the lambda one.

'''

# TODO: Display tangent lines at points

import numpy as np
import tkinter as tk
import random

# Window size
n = 10
window_w = int(2**n)
window_h = int(2**n)
np.set_printoptions(suppress=True)

# Tkinter Setup
root = tk.Tk()
root.title("1D Wiggly Spline, Positions & Tangents Specified Everywhere (C1)")
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
    assert 0<=i<=3

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
    assert np.power(delta, 2,) != lamb or (lamb == 0 and delta == 0)
    assert 0 <= i <= 3

    sign1 = 1 if i <= 1 else -1
    sign2 = 1 if i % 2 == 0 else -1

    # Special cases (TODO: include more!)
    if delta == 0 and lamb == 0:
        return 0 if i == 0 else i * np.power(t, i-1)

    if np.power(delta, 2) > lamb:
        expr = (sign1 * delta) + (sign2 * np.sqrt(np.power(delta, 2) - lamb))
        return np.exp(t * expr) * expr
    else:
        expr = np.sqrt(-np.power(delta, 2) + lamb)

        if sign2 == 1:
            return np.exp(sign1 * t * delta) * ((sign1 * delta * np.cos(t * expr)) - (expr * np.sin(t * expr)))

        return np.exp(sign1 * t * delta) * (-(expr * np.cos(t * expr)) - (sign1 * delta * np.sin(t * expr)))


# Coefficient computation (Using x = (A^-1)*b.)
def compute_coefficients(keyframes, delta, lamb, g):
    '''
        Note:

        Example — Here is what each row in a 8 x 8 matrix represents (2 segments, 3 keyframes).

        1. Segment 1 must satisfy positional boundary condition
        2. Segment 1 must satisfy velocity (tangent) boundary condition
        3. Segment 1 must satisfy middle positional keyframe constraint
        4. Segment 1 must satisfy middle velocity keyframe constraint
        5. Segment 2 must satisfy middle positional keyframe constraint
        6. Segment 2 must satisfy middle velocity keyframe constraint
        7. Segment 2 must satisfy positional boundary condition
        8. Segment 2 must satisfy velocity (tangent) boundary condition
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
    for seg in range(len(keyframes) - 1):
        frame0, frame1 = keyframes[seg], keyframes[seg+1]

        # Each segment has 2 points whose positions + velocities we've specified (4 rows on matrix per seg)
        t0, p0, v0 = frame0[0], frame0[1], frame0[2]
        t1, p1, v1 = frame1[0], frame1[1], frame1[2]

        for col in range(n_coeffs):
            B[row], B[row+1], B[row+2], B[row+3] = p0 + c, v0, p1 + c, v1

            if col < 4 * seg or col >= 4 * (seg + 1):
                A[row][col] = 0     # pos, point 0 coeff = 0
                A[row+1][col] = 0   # vel, point 0 coeff = 0
                A[row+2][col] = 0   # pos, point 1 coeff = 0
                A[row+3][col] = 0   # vel, point 1 coeff = 0
            else:
                j = col % 4
                A[row][col] = b(t0, delta, lamb, j)        # pos, point 0 coeff = b(t0)
                A[row+1][col] = b_dot(t0, delta, lamb, j)  # vel, point 0 coeff = b(t0)
                A[row+2][col] = b(t1, delta, lamb, j)      # pos, point 1 coeff = b(t1)
                A[row+3][col] = b_dot(t1, delta, lamb, j)  # vel, point 1 coeff = b(t1)

        row += 4

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
n_frames = 6  # must be >=2
pos_absmax = 2  # positions randomly generated, for now (tangents all 0, for now)
delta, lamb, g = 2, 10, 0
keyframes = [(i / (n_frames-1), pos_absmax * ((random.random() * 2) - 1), 0) for i in range(n_frames)]

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
    while True:
        run_step()

tk.mainloop()