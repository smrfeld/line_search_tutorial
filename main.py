import logging
from pathlib import Path
from newton import Newton
import plot
import matplotlib.pyplot as plt
import numpy as np

def obj_func(x : np.array) -> float:
    # Six hump function
    # http://scipy-lectures.org/intro/scipy/auto_examples/plot_2d_minimization.html
    return ((4 - 2.1*x[0]**2 + x[0]**4 / 3.) * x[0]**2 + x[0] * x[1] + (-4 + 4*x[1]**2) * x[1] **2)

def gradient(x : np.array) -> np.array:
    return np.array([
        8*x[0] - 4 * 2.1 * x[0]**3 + 2 * x[0]**5 + x[1],
        x[0] - 8 * x[1] + 16 * x[1]**3
        ])

def hessian(x : np.array) -> np.array:
    return np.array([
        [
            2 * (4 - 6 * 2.1 * x[0]**2 + 5 * x[0]**4),
            1
        ],
        [
            1,
            -8 + 48 * x[1]**2
        ]])

def is_pos_def(x : np.array) -> bool:
    return np.all(np.linalg.eigvals(x) > 0)

def reg_inv_hessian(x : np.array) -> np.array:
    # Check pos def
    hes = hessian(x)
    if is_pos_def(hes):
        return np.linalg.inv(hes)
    else:
        # Regularize
        identity = np.eye(len(x))
        eps = 1e-8
        hes_reg = hes + eps * identity
        eps_max = 100.0
        while not is_pos_def(hes_reg) and eps <= eps_max:
            eps *= 10.0
            hes_reg = hes + eps * identity
        
        if eps > eps_max:
            print(hes_reg)
            print(is_pos_def(hes_reg))
            raise ValueError("Failed to regularize Hessian!")
        
        return np.linalg.inv(hes_reg)

def get_random_uniform_in_range(x_rng : np.array, y_rng : np.array) -> np.array:
    p = np.random.rand(2)
        
    p[0] *= x_rng[1] - x_rng[0]
    p[0] += x_rng[0]

    p[1] *= y_rng[1] - y_rng[0]
    p[1] += y_rng[0]

    return p

if __name__ == "__main__":

    opt = Newton(
        obj_func=obj_func,
        gradient_func=gradient,
        reg_inv_hessian=reg_inv_hessian
    )
    opt.log.setLevel(logging.INFO)

    x_rng = [-2,2]
    y_rng = [-1,1]
    
    fig_dir = "figures"
    Path(fig_dir).mkdir(parents=True, exist_ok=True)

    # params_init = get_random_uniform_in_range(x_rng, y_rng)
    # params_init = np.array([-1.55994695, -0.31833122])
    params_init = np.array([-1.1,-0.5])
    
    converged, no_opt_steps, final_update, traj, line_search_factors = opt.run(
        no_steps=10,
        params_init=params_init,
        tol=1e-8,
        store_traj=True
    )

    print("Converged: %s" % converged)
    print(traj)
    print(line_search_factors)

    trajs = {}
    trajs[0] = traj

    endpoints, counts = plot.get_endpoints_and_counts(trajs)
    for i in range(0,len(endpoints)):
        print(endpoints[i], " : ", counts[i])
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # plot.plot_3d_endpoint_lines(ax, trajs)
    plot.plot_3d(ax, x_rng,y_rng,obj_func)
    plt.savefig(fig_dir+"/3d.png", dpi=200)

    plt.figure()
    plot.plot_obj_func(x_rng, y_rng, obj_func)
    plot.plot_trajs(x_rng, y_rng, trajs, [0])
    plt.title("Trajs")
    plt.savefig(fig_dir+"/trajs.png", dpi=200)

    plt.figure()
    plot.plot_obj_func(x_rng, y_rng, obj_func)
    # plot.plot_quiver(x_rng, y_rng, trajs)
    plot.plot_endpoint_counts(trajs)
    plt.title("Endpoints")
    plt.savefig(fig_dir+"/endpoints.png", dpi=200)

    plt.figure()
    plot.plot_histogram(x_rng,y_rng,trajs,50)
    plot.plot_endpoint_counts(trajs)
    plt.title("Endpoints")
    plt.savefig(fig_dir+"/histogram.png", dpi=200)

    # plt.show()

    plt.close('all')

    # Line search
    plt.figure()
    plot.plot_line_search(line_search_factors, traj, obj_func)
    plt.title("Line search")
    plt.savefig(fig_dir+"/line_search.png", dpi=200)