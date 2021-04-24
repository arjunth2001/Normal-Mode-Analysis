# import automatic differentiator to compute gradient module
from autograd import grad
import autograd.numpy as np
import autograd

# We need these functions as autograd uses a specific numpy.. normal numpy wont work...


def sep(p1, p2):
    return p2 - p1


def pbc_sep(p1, p2):

    arrow = sep(p1, p2)
    rem = np.mod(arrow, 18)  # The image in the first cube
    mic_separation_vector = np.mod(rem+18/2, 18)-18/2

    return np.array(mic_separation_vector)


def get_energy(geom):
    eps = 0.238
    sigma = 3.4
    te = 0
    tgeom = np.array(geom)
    pairs = [(a, b) for idx, a in enumerate(tgeom)
             for b in tgeom[idx + 1:]]
    for pair in pairs:
        rij = np.linalg.norm(pbc_sep(pair[0], pair[1]))
        if rij == 0:
            continue
        te += 4*eps*((sigma/rij)**12-(sigma/rij)**6)

    return te


def gradient_descent(alpha, max_its, w):
    # compute gradient module using autograd
    gradient = grad(get_energy)

    # run the gradient descent loop
    weight_history = [w]           # container for weight history
    # container for corresponding cost function history
    cost_history = [get_energy(w)]
    for k in range(max_its):
        # evaluate the gradient, store current weights and cost function value
        #print(k, w)
        print("It:", k)
        grad_eval = gradient(w)

        # take gradient descent step
        w = w - alpha*grad_eval
        print(get_energy(w))
        # record weight and cost
        weight_history.append(w)
        cost_history.append(get_energy(w))
    return weight_history, cost_history
