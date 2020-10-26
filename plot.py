from main import obj_func
import numpy as np

import matplotlib.pyplot as plt

from scipy.interpolate import griddata

from typing import Dict, Any, Tuple, List

from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage.filters import gaussian_filter

def make_points_and_vectors(trajs : Dict[int, Dict[int, np.array]]) -> Tuple[np.array,np.array]:
    
    points = []
    vecs = []
    for traj in trajs.values():
        vals = list(traj.values())
        xi = vals[0][0]
        xf = vals[-1][0]
        yi = vals[0][1]
        yf = vals[-1][1]

        points.append(np.array([xi,yi]))
        v = np.array([xf-xi,yf-yi])
        v /= np.sqrt(v[0]**2 + v[1]**2)
        vecs.append(v)

    return (np.array(points),np.array(vecs))

def get_endpoints_and_counts(trajs : Dict[int, Dict[int, np.array]]) -> Tuple[np.array, np.array]:
    endpoints = []
    counts = []

    for traj in trajs.values():
        vals = list(traj.values())
        x = vals[-1][0]
        y = vals[-1][1]

        matched = False
        dist_where_points_match = 0.1
        # Get closest
        dists = np.array([np.sqrt((x-pt[0])**2 + (y-pt[1])**2) for pt in endpoints])
        if len(dists) == 0:
            endpoints.append([x,y])
            counts.append(1)
        else:
            dist = np.min(dists)
            if dist < dist_where_points_match:
                idx_min = np.argmin(dists)
                counts[idx_min] += 1
            else:
                endpoints.append([x,y])
                counts.append(1)
        
    return (np.array(endpoints), np.array(counts))

def plot_obj_func(x_rng : np.array, y_rng : np.array, obj_func : Any):

    # Plot objective func
    x_lspace = np.linspace(x_rng[0], x_rng[1])
    y_lspace = np.linspace(y_rng[0], y_rng[1])
    xg, yg = np.meshgrid(x_lspace, y_lspace)
    plt.imshow(obj_func([xg, yg]), extent=[x_rng[0], x_rng[1], y_rng[0], y_rng[1]], origin="lower")
    plt.colorbar()

def plot_trajs(x_rng : np.array, y_rng : np.array, trajs : Dict[int,Dict[int,np.array]], trials : List[int]):

    for trial in trials:
        traj = trajs[trial]
        x = [t[0] for t in traj.values()]
        y = [t[1] for t in traj.values()]
        plt.plot(x,y, color='white')
        plt.plot([x[-1]], [y[-1]], marker='o', markersize=3, color="red")
        plt.plot([x[0]], [y[0]], marker='o', markersize=3, color="cyan")
    plt.xlim(x_rng)
    plt.ylim(y_rng)

def plot_quiver(x_rng : np.array, y_rng : np.array, trajs : Dict[int,Dict[int,np.array]]):
    
    points, vecs = make_points_and_vectors(trajs)

    x_lspace = np.linspace(x_rng[0], x_rng[1])
    y_lspace = np.linspace(y_rng[0], y_rng[1])

    x_msh,y_msh = np.meshgrid(x_lspace,y_lspace)
    grid = griddata(points, vecs, (x_msh,y_msh), method='nearest')

    u = grid[:,:,0]
    v = grid[:,:,1]
    plt.quiver(x_msh,y_msh,u,v)

def plot_endpoint_counts(trajs : Dict[int,Dict[int,np.array]]):

    endpoints, counts = get_endpoints_and_counts(trajs)
    for i in range(0, len(endpoints)):
        # plt.plot([x[-1]], [y[-1]], marker='o', markersize=3, color="red")
        plt.text(endpoints[i,0], endpoints[i,1], str(counts[i]), color='red')

def plot_3d_endpoint_lines(ax : Any, trajs : Dict[int,Dict[int,np.array]]):

    endpoints, counts = get_endpoints_and_counts(trajs)
    for i in range(0, len(endpoints)):
        ax.plot([endpoints[i,0],endpoints[i,0]],[endpoints[i,1],endpoints[i,1]],[-1,6],color='red')

def plot_3d(ax : Any, x_rng : np.array, y_rng : np.array, obj_func : Any):

    x_lspace = np.linspace(x_rng[0], x_rng[1])
    y_lspace = np.linspace(y_rng[0], y_rng[1])
    xg, yg = np.meshgrid(x_lspace, y_lspace)

    # surf = ax.plot_wireframe(xg, yg, obj_func([xg, yg]), rstride=5, cstride=5)
    surf = ax.plot_surface(xg, yg, obj_func([xg, yg]), rstride=1, cstride=1, cmap=plt.cm.jet, linewidth=0, antialiased=False)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    ax.set_title('Six-hump Camelback function')

def plot_histogram(x_rng : np.array, y_rng : np.array, trajs : Dict[int,Dict[int,np.array]], s : float):
    
    endpoints, counts = get_endpoints_and_counts(trajs)

    x = endpoints[:,0]
    y = endpoints[:,1]
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=1000, range=[x_rng,y_rng])
    heatmap = gaussian_filter(heatmap, sigma=s)
    extent = [x_rng[0], x_rng[1], y_rng[0], y_rng[1]]
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower')

def plot_line_search(line_search_factors : Dict[int,np.array], traj: Dict[int,np.array], obj_func : Any):

    # Line search
    opt_step_use = None
    alpha_use = None
    for opt_step, alpha in line_search_factors.items():
        if alpha < 1:
            opt_step_use = opt_step
            alpha_use = alpha

    if opt_step_use != None:
        params_start = traj[opt_step_use-1]
        params_end = traj[opt_step_use]
        direction = params_end - params_start
        direction /= alpha_use
        params_end = params_start + direction

        pts = [ params_start ]
        for i in range(0,100):
            pt = pts[-1] + direction / 100.0
            pts.append(pt)

        xvals = [ i/100.0 for i in range(0,101) ]
        yvals = [ obj_func(x) for x in pts ]

        plt.plot(xvals,yvals)
        f = 0.95
        plt.arrow(xvals[0],yvals[0],f*(xvals[-1]-xvals[0]),f*(yvals[-1]-yvals[0]),head_width=0.05,color='r')
        f = 0.89
        plt.arrow(xvals[0],yvals[0],f*(xvals[50]-xvals[0]),f*(yvals[50]-yvals[0]),head_width=0.05,color='g')
