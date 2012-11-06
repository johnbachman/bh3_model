"""
Source code for fitting a simple model derived from Newmeyer to
Sarosiek/Letai BH3 profiling data.
"""

import sys
import pickle
import bayessb
from bayessb.plot import surf
import numpy
import matplotlib.pyplot as plt
from pysb.util.report import Report
import collections

# Get the data
import bh3_titration_data as data
(data_mean, data_sd) = data.normalize_to_relative_max(11, 20)
COL_TO_FIT = None # This will be initialized as a command-line arg

T_OFFSET_INDEX = 3 

# Instantiate report
rep = Report()

# Create the "dummy" model class
# ==============================
Parameter = collections.namedtuple('Parameter', 'name value')
Observable = collections.namedtuple('Observable', 'name species coefficients')
Initial = collections.namedtuple('Initial', 'param_index species_index')

class Model(object):
    """A model for the BH3 profiling data based on the
    "Catalyst Activation" model in the Newmeyer PLoS Biology paper.
    """

    def __init__(self):
        self.num_params = 4
        self.parameters = [None] * self.num_params
        self.observables = [None] * 1
        self.initial_conditions = []
        self.sim_param_values = numpy.empty(self.num_params)
        self.parameters[0] = Parameter('k1', 1)
        self.parameters[1] = Parameter('k2', 0.043158)
        self.parameters[2] = Parameter('n', 4)
        self.parameters[3] = Parameter('t_offset', 1)
        self.observables[0] = Observable('jc1', [0], [1])

    def simulate(self, tspan, param_values=None):
        if param_values is not None:
            # accept vector of parameter values as an argument
            if len(param_values) != len(self.parameters):
                raise Exception("param_values must have length %d" %
                                len(self.parameters))
            self.sim_param_values[:] = param_values
        else:
            # create parameter vector from the values in the model
            self.sim_param_values[:] = [p.value for p in self.parameters]

        # Aliases to the parameters
        k1 = self.sim_param_values[0]
        k2 = self.sim_param_values[1]
        n = self.sim_param_values[2]
        t_offset = self.sim_param_values[3]
        #t_offset = 0
        n = 4

        sim = numpy.exp(-k1*((1 - numpy.exp(-k2 *(tspan+t_offset)))**n))

        return sim

model = Model()

# Do the fit!
# ===========
def do_fit(nsteps, output_filename):
    # Set the random number generator seed
    seed = 2
    random = numpy.random.RandomState(seed)

    # Initialize the MCMC arguments
    opts = bayessb.MCMCOpts()
    opts.model = model
    opts.estimate_params = model.parameters
    opts.initial_values = [p.value for p in opts.estimate_params]
    opts.nsteps = nsteps
    opts.likelihood_fn = likelihood
    opts.prior_fn = prior
    opts.step_fn = step
    opts.use_hessian = True
    opts.hessian_period = opts.nsteps / 10 # Calculate the Hessian 10 times
    opts.seed = seed
    mcmc = bayessb.MCMC(opts)

    # Plot before curves
    mcmc.initialize()
    plot_position(mcmc, mcmc.position, title="Initial Values", report=rep)
    plt.close()

    # Run it!
    mcmc.estimate()

    """
    # Plot best fit position
    #mcmc.position = mcmc.positions[numpy.argmin(mcmc.likelihoods)]
    mcmc.position = mcmc.positions[numpy.argmin(mcmc.posteriors)]
    plot_position(mcmc, mcmc.position, title="Best Fit", report=rep)
    plt.close()
    p_names = [p.name for p in opts.estimate_params]
    p_name_vals = zip(p_names, mcmc.cur_params(mcmc.position))
    s = "best fit position: " + str(p_name_vals)
    print s
    rep.addText(s)

    # Get the accepted positions
    mixed_start = opts.nsteps / 2
    mixed_positions = mcmc.positions[mixed_start:,:]
    mixed_accepts = mcmc.accepts[mixed_start:]
    mixed_accept_positions = mixed_positions[mixed_accepts]

    # Output some statistics
    marginal_geo_mean_pos = numpy.mean(mixed_accept_positions, 0)
    geo_mean_name_vals = zip(p_names, mcmc.cur_params(marginal_geo_mean_pos))
    s = "marginal (geometric) mean position: " + str(geo_mean_name_vals) \
            + "\n\n"

    marginal_ar_mean_pos = numpy.mean(10**mixed_accept_positions, 0)
    ar_mean_name_vals = zip(p_names, marginal_ar_mean_pos)
    s += "marginal (arithmetic) mean position: " + str(ar_mean_name_vals) \
            + "\n\n"

    marginal_median_pos = numpy.median(mixed_accept_positions, 0)
    median_name_vals = zip(p_names, mcmc.cur_params(marginal_median_pos))
    s += "marginal median position: " + str(median_name_vals) + "\n\n"
    print s
   
    marginal_ar_std = numpy.std(10**mixed_accept_positions, 0)
    ar_std_name_vals = zip(p_names, marginal_ar_std)
    s += "marginal (arithmetic) std: " + str(ar_std_name_vals) + "\n\n"
    print s

    marginal_geo_std = numpy.std(mixed_accept_positions, 0)
    geo_std_name_vals = zip(p_names, marginal_geo_std)
    s += "marginal (geometric) std: " + str(geo_std_name_vals) + "\n\n"
    print s
    rep.addText(s)

    # Show histograms for all parameters
    for (index, p) in enumerate(model.parameters):
        plt.figure()
        param_logs = mixed_accept_positions[:,index]
        param_vals = 10**param_logs
        param_mean = numpy.mean(param_vals)
        plt.hist(param_vals, bins=50)
        plt.title('%s, mean %f' % (p.name, param_mean))
        rep.addCurrentFigure()
        plt.close()

    # Plot a sampling of trajectories
    plot_curve_distribution(mcmc, mixed_accept_positions, 400,
                            title="Sampling of Fits", report=rep)
    plt.close()

    # Plot k1/k2 surface
    bayessb.plot.surf(mcmc, 0, 1, mask=mixed_start)
    plt.title("k1/k2 surface")
    rep.addCurrentFigure()
    plt.close()

    # Add source code to report 
    rep.addPythonCode('bh3_model_scaled.py')

    # Write report
    rep.writeReport(output_filename)
    """

    # Pickle the positions
    f = open(output_filename + str('.pck'), 'w')
    pickle.dump(mcmc.positions, f)
    return mcmc

# Define the likelihood, prior and step functions
# ===============================================

def likelihood(mcmc, position):

    # Get the current value for the time offset
    cur_t_offset = mcmc.cur_params(position=position)[T_OFFSET_INDEX]
    offset_tspan = data.exp_tspan + cur_t_offset

    # Calculate objective function
    sim = model.simulate(offset_tspan, mcmc.cur_params(position))
    #sim = model.simulate(data.exp_tspan, mcmc.cur_params(position))
    return numpy.sum((data_mean[:,COL_TO_FIT] - sim) ** 2 /
                     (2 * data_sd[:, COL_TO_FIT] ** 2))

def step(mcmc):
    """The function to call at every iteration. Currently just prints
    out a few progress indicators.
    """
    if mcmc.iter % 1000 == 0:
        print 'iter=%-5d  sigma=%-.3f  T=%-.3f  acc=%-.3f, lkl=%g  prior=%g  post=%g' % \
            (mcmc.iter, mcmc.sig_value, mcmc.T, mcmc.acceptance/(mcmc.iter+1),
             mcmc.accept_likelihood, mcmc.accept_prior, mcmc.accept_posterior)

def prior(mcmc, position):
    """Create uniform priors by giving an infinite penalty
    if bounds are exceeded.
    """
    param_vals = mcmc.cur_params(position)

    if param_vals[0] < 0: # k1
        return 1e15
    if param_vals[1] < 0: # k2
        return 1e15
    if param_vals[2] < 1: # n
        return 1e15
    if param_vals[T_OFFSET_INDEX] < 0: # t_offset
        return 1e15

    return ((param_vals[T_OFFSET_INDEX] - 1)**2) / (2*(0.01**2)) # t_offset
    
# Plotting Functions
# ==================
def plot_curve_distribution(mcmc, mixed_positions, num_samples,
                            title=None, report=None):
    plt.figure()

    # Plot data
    plt.errorbar(data.exp_tspan, data_mean[:,COL_TO_FIT],
            yerr=data_sd[:,COL_TO_FIT])
    
    for i in range(0, num_samples):
        num_mixed_positions = len(mixed_positions)
        rand_position_index = numpy.random.randint(num_mixed_positions)
        rand_position = mixed_positions[rand_position_index]

        # Get the current value for the time offset
        cur_t_offset = mcmc.cur_params(rand_position)[T_OFFSET_INDEX]
        offset_tspan = data.exp_tspan + cur_t_offset

        # Get the simulation data
        #sim = mcmc.options.model.simulate(data.exp_tspan,
        sim = mcmc.options.model.simulate(offset_tspan,
                                          mcmc.cur_params(rand_position))
        # Plot simulations
        plt.plot(data.exp_tspan, sim, 'r', alpha=0.01)

    plt.xlim(-10, 190)
    if title is not None:
        plt.title(title)
    #plt.show()

    # Add figure to report 
    if report is not None:
        rep.addCurrentFigure()

def plot_position(mcmc, position, title=None, report=None):
    """Note that position is expected to be in log-transform format."""
    #plt.ion()
    plt.figure()

    # Get the current value for the time offset
    cur_t_offset = mcmc.cur_params(position)[T_OFFSET_INDEX]
    offset_tspan = data.exp_tspan + cur_t_offset

    # Get the simulation data
    #sim = mcmc.options.model.simulate(data.exp_tspan,
    sim = mcmc.options.model.simulate(offset_tspan,
                                      mcmc.cur_params(position))
   
    # Plot data
    plt.errorbar(offset_tspan, data_mean[:,COL_TO_FIT],
            yerr=data_sd[:,COL_TO_FIT])
    #plt.errorbar(data.exp_tspan, data_mean[:,COL_TO_FIT],

    # Plot simulations
    plt.plot(offset_tspan, sim)
    #plt.plot(data.exp_tspan, sim)

    #plt.ylim(0, 1.2)
    plt.xlim(-10, 190)
    if title is not None:
        plt.title(title)
    #plt.show()

    # Add figure to report 
    if report is not None:
        rep.addCurrentFigure()

# Main function
# =============
if __name__ == '__main__':
    if len(sys.argv) <= 2:
        raise Exception("You must specify the column index, number of steps, and the output filename.")
    COL_TO_FIT = int(sys.argv[1])
    nsteps = int(sys.argv[2])
    output_filename = sys.argv[3]
    do_fit(nsteps, output_filename)


