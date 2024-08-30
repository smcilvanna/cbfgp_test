import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize
import scipy.stats
from scipy.special import erf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def simulate(xy):       # define the function that will be approximated by gpr
    x, y = xy           # unpack input variables
    sigma = 0.2         # distribution sigma
    alpha = -0.1        # skews the distribution  
    m = 0.05*(y-5)**2   # relationship between dist mean and y(obstacle size)
    phi = np.exp(-0.5 * ((x - m) ** 2) / (sigma ** 2))
    Phi = 0.5 * (1 + erf(alpha * (x - m) / (sigma * np.sqrt(2))))
    z = 30 * phi * Phi - 5
    return z

def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01): # Define the Expected Improvement acquisition function for 
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample = gpr.predict(X_sample)
    sigma = sigma.reshape(-1, 1)

    with np.errstate(divide='warn'):
        imp = mu - np.max(mu_sample) - xi
        Z = imp / sigma
        ei = imp * scipy.stats.norm.cdf(Z) + sigma * scipy.stats.norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
    
    return ei

def propose_location(acquisition, X_sample, Y_sample, gpr, bounds, n_restarts=25): # Optimize the acquisition function to find the next test point
    dim = X_sample.shape[1]
    min_val = 1
    min_x = None

    def min_obj(X):
        return -acquisition(X.reshape(-1, dim), X_sample, Y_sample, gpr).flatten()

    for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
        res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')
        if res.fun < min_val:
            min_val = res.fun
            min_x = res.x

    return min_x.reshape(-1, dim)

def plot_results(print=False):     # plot 2d and 3d of true and predicted functions   
    x = np.linspace(0, 5, 50)   # define plot area (x/y)
    y = np.linspace(0, 5, 50)
    X, Y = np.meshgrid(x, y)
    xy_grid = np.vstack([X.ravel(), Y.ravel()]).T
    z_sim = np.array([simulate(xy) for xy in xy_grid])  # define true z values
    z_gpr, sigma = gp.predict(xy_grid, return_std=True) # define predicted z values

    Z_sim = z_sim.reshape(X.shape)      # Reshape results back into grid form for plotting
    Z_gpr = z_gpr.reshape(X.shape)

    plt.figure(figsize=(14, 6))         # 2d plot
    plt.suptitle('Bayesian Optimization with Gaussian Process Regression\n' + str(n_iter) + ' Iterations')
    # Plot the original simulation function surface
    plt.subplot(1, 2, 1)
    plt.contourf(X, Y, Z_sim, levels=20, cmap='viridis')
    plt.colorbar()
    plt.scatter(X_init[:, 0], X_init[:, 1], c='red', marker='x')
    plt.title('Original Simulation Function')
    plt.xlabel('x')
    plt.ylabel('y')

    # Plot the GPR model predictions surface
    plt.subplot(1, 2, 2)
    plt.contourf(X, Y, Z_gpr, levels=20, cmap='viridis')
    plt.colorbar()
    plt.scatter(X_init[:, 0], X_init[:, 1], c='red', marker='x')
    plt.title('GPR Model Predictions')

    plt.xlabel('x')
    plt.ylabel('y')

    plt.tight_layout()
    
    if print:
        plt.savefig(pltname2d, format='pdf')
    else:
        plt.show()


    fig = plt.figure(figsize=(14, 6))       # 3d plot
    plt.suptitle('Bayesian Optimization with Gaussian Process Regression\n' + str(n_iter) + ' Iterations')
    
    ax1 = fig.add_subplot(121, projection='3d') # true plot
    ax1.plot_surface(X, Y, Z_sim, cmap='viridis', alpha=0.8)
    ax1.scatter(X_init[:, 0], X_init[:, 1], z_init, c='red', marker='x')  # initial sampled points
    ax1.set_title('Original Simulation Function')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')

    ax2 = fig.add_subplot(122, projection='3d') # trained gpr predicted plot
    ax2.plot_surface(X, Y, Z_gpr, cmap='viridis', alpha=0.8)
    ax2.scatter(X_init[:, 0], X_init[:, 1], z_init, c='red', marker='x')  # initial sampled points
    ax2.set_title('GPR Model Predictions')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')

    plt.tight_layout()
    if print:
        plt.savefig(pltname3d, format='pdf')
    else:
        plt.show()


if __name__ == '__main__':
    n_iter = 200    # <<<<<<<<<<< Set number of iterations for Bayesian Optimization ##########################################
    # Setup a set of random sample points for x and y to start with
    np.random.seed(42)                                  # For reproducibility
    X_init = np.random.uniform(0, 5, (5, 2))            # 5 initial samples in 2 dimensions (x and y)
    z_init = np.array([simulate(xy) for xy in X_init])  # Corresponding performance values (z)
    # Train initial GP model with starting samples
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))                    # Define the kernel for the GP model                       
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)   # Define the GP model   
    gp.fit(X_init, z_init)                                                  # Train the GP model with initial samples
    # Iteratively update the model with new data
    bounds = np.array([[0, 5], [0, 5]])         # Bounds for x and y variables
    
    for i in range(n_iter):
        X_next = propose_location(expected_improvement, X_init, z_init, gp, bounds)     # Propose the next sampling point
        Z_next = simulate(X_next[0])            # Get the simulation result for the new point
        X_init = np.vstack((X_init, X_next))    # Add the new sample points to the dataset
        z_init = np.append(z_init, Z_next)      # Add the new sample performance result to the dataset
        gp.fit(X_init, z_init)                  # Update the model with new samples
        print(f"Iteration {i+1}: X_next = {X_next}, Z_next = {Z_next}") # Print the new point and its corresponding output

    print("Optimization finished!")
    # print("Final sampled points (x, y):")
    # print(X_init)
    # print("Final performance values (z):")
    # print(z_init)

    pltname2d = 'gpr_comparison_2d_' + str(n_iter) + '.pdf'
    pltname3d = 'gpr_comparison_3d_' + str(n_iter) + '.pdf'
    plot_results(print=True)  # plot the results

