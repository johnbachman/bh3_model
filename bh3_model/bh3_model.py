"""
Source code for fitting a simple model derived from Newmeyer to
Sarosiek/Letai BH3 profiling data.
"""

import bayessb
from bayessb.plot import surf
import numpy
import matplotlib.pyplot as plt
from tbidbaxlipo.util.report import Report
import collections

#from bh3_fig1_data import *
from bh3_titration_data import *

T_OFFSET_INDEX = 2 
exp_dmso_max = max(dmso_avgs)
exp_fccp_min = min(fccp_avgs)

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

    def __init__(self, f0):
        self.f0 = f0

        self.num_params = 9 
        self.parameters = [None] * self.num_params
        self.observables = [None] * 1
        self.initial_conditions = []
        self.sim_param_values = numpy.empty(self.num_params)
    
        self.parameters[0] = Parameter('fmax', 1.1629)
        self.parameters[1] = Parameter('k_agg', 0.023158)
        self.parameters[2] = Parameter('t_offset', 2.184)
        self.parameters[3] = Parameter('f0', 0.16441)
        self.parameters[4] = Parameter('k1_bim', 0.0135)
        self.parameters[5] = Parameter('k1_bid', 0.09)
        self.parameters[6] = Parameter('k2_bim', 5)
        self.parameters[7] = Parameter('k2_bid', 5)
        self.parameters[8] = Parameter('k_sharp', 8)
        #self.parameters[7] = Parameter('k2_bid', 0.004541)
        #self.parameters[8] = Parameter('n', 8.059)
        #self.parameters[9] = Parameter('f0', self.f0)
        #self.parameters[9] = Parameter('f0', 16.441)

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
        fmax = self.sim_param_values[0]
        k_agg = self.sim_param_values[1]
        t_offset = self.sim_param_values[2]
        f0 = self.sim_param_values[3]
        k1_bim = self.sim_param_values[4]
        k1_bid = self.sim_param_values[5]
        k2_bim = self.sim_param_values[6]
        k2_bid = self.sim_param_values[7]
        k_sharp = self.sim_param_values[8]

        #dmso = (f0 +
        #       (fmax*
        #       (1 - numpy.exp(-k_agg*(tspan + t_offset)))) *
        #       numpy.exp(-k_bg*(tspan + t_offset)))

        # Catalysis Activation Model
        # ==========================
        bim = (f0 +
               (fmax*
                    (1 - numpy.exp(-k_agg*(tspan + t_offset)))) *
               numpy.exp(-(k_bg + (k1_bim * ((1 - numpy.exp(-k2*(tspan+t_offset)))**n) *
                   (tspan + t_offset)))))
        bid = (f0 +
               (fmax*
                    (1 - numpy.exp(-k_agg*(tspan + t_offset)))) *
               numpy.exp(-(k_bg + (k1_bid * ((1 - numpy.exp(-k2*(tspan+t_offset)))**n) *
                   (tspan + t_offset)))))

        return (dmso, bim, bid) 
        #return (bim, bid) 

# Instantiate the model
# =====================
model = Model(exp_fccp_min)

# Do the fit!
# ===========
def do_fit():
    # Set the random number generator seed
    seed = 2
    random = numpy.random.RandomState(seed)

    # Initialize the MCMC arguments
    opts = bayessb.MCMCOpts()
    opts.model = model
    #opts.tspan = sim_tspan
    opts.estimate_params = model.parameters
    opts.initial_values = [p.value for p in opts.estimate_params]
    opts.nsteps = 100000 
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

    # Plot best fit position
    mcmc.position = mcmc.positions[numpy.argmin(mcmc.likelihoods)]
    plot_position(mcmc, mcmc.position, title="Best Fit", report=rep)
    plt.close()
    p_names = [p.name for p in opts.estimate_params]
    p_name_vals = zip(p_names, mcmc.cur_params(mcmc.position))
    s = "best fit position: " + str(p_name_vals)
    print s
    rep.addText(s)

    # Output some statistics
    mixed_start = opts.nsteps / 2
    mixed_positions = mcmc.positions[mixed_start:,:]
    mixed_accepts = mcmc.accepts[mixed_start:]
    mixed_accept_positions = mixed_positions[mixed_accepts]

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
    rep.addText(s)
   
    marginal_ar_std = numpy.std(10**mixed_accept_positions, 0)
    ar_std_name_vals = zip(p_names, marginal_ar_std)
    s += "marginal (arithmetic) std: " + str(ar_std_name_vals) + "\n\n"
    print s
    rep.addText(s)

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

    # Plot histogram of k1_bid/k1_bim ratio
    plt.figure()
    k1_diffs = mixed_accept_positions[:,5] - mixed_accept_positions[:,4]
    plt.hist(k1_diffs, bins=20)
    plt.title("Log ratio of k1_bid/k1_bim")
    rep.addCurrentFigure()
    plt.close()

    s = "Geometric mean of k1_bid/k1_bim ratio: %f\n\n" % \
            10**numpy.mean(k1_diffs)
    s += "Arithmetic mean of k1_bid/k1_bim ratio: %f\n\n" % \
            numpy.mean(10**k1_diffs)
    s += "Median of k1_bid/k1_bim ratio: %f" % numpy.median(10**k1_diffs)
    rep.addText(s)

    """
    # Plot histogram of k2_bid/k2_bim ratio
    plt.figure()
    k2_diffs = mixed_accept_positions[:,7] - mixed_accept_positions[:,6]
    plt.hist(k2_diffs, bins=20)
    plt.title("Log ratio of k2_bid/k2_bim")
    rep.addCurrentFigure()
    plt.close()

    s = "Geometric mean of k2_bid/k2_bim ratio: %f\n\n" % \
            10**numpy.mean(k2_diffs)
    s += "Arithmetic mean of k2_bid/k2_bim ratio: %f\n\n" % \
            numpy.mean(10**k2_diffs)
    s += "Median of k2_bid/k2_bim ratio: %f" % numpy.median(10**k2_diffs)
    rep.addText(s)
    """

    # Plot k1_bim/k2 surface
    bayessb.plot.surf(mcmc, 4, 6, mask=mixed_start)
    plt.title("k1_bim/k2 surface")
    rep.addCurrentFigure()
    plt.close()

    # Plot k1_bid/k2 surface
    bayessb.plot.surf(mcmc, 5, 6, mask=mixed_start)
    plt.title("k1_bid/k2_bid surface")
    rep.addCurrentFigure()
    plt.close()

    # Plot histogram of k1_bid, k2, and sum
    plt.figure()
    bid_sum = mixed_accept_positions[:,5] + mixed_accept_positions[:,6]
    plt.hist(mixed_accept_positions[:,5], bins=20, alpha=0.5, label="k1_bid")
    plt.hist(mixed_accept_positions[:,6], bins=20, alpha=0.5, label="k2_bid")
    plt.hist(bid_sum, bins=20, alpha=0.5, label="k1_bid*k2_bid") 
    plt.xlabel("Log(k)")
    plt.ylabel("Count")
    plt.title("k1_bid, k2, and k1_bid*k2")
    plt.legend(loc="center right")
    rep.addCurrentFigure()
    plt.close()

    # Plot histogram of k1_bim, k2, and sum
    plt.figure()
    bim_sum = mixed_accept_positions[:,4] + mixed_accept_positions[:,6]
    plt.hist(mixed_accept_positions[:,4], bins=20, alpha=0.5)
    plt.hist(mixed_accept_positions[:,6], bins=20, alpha=0.5)
    plt.hist(bim_sum, bins=20, alpha=0.5) 
    plt.xlabel("Log(k)")
    plt.ylabel("Count")
    plt.title("k1_bim, k2, and k1_bim*k2")
    plt.legend(loc="center right")
    rep.addCurrentFigure()
    plt.close()

    # Plot histogram of bid_sum - bim_sum 
    plt.figure()
    ratio = bid_sum - bim_sum
    plt.hist(ratio, bins=20)
    plt.xlabel("Log(k)")
    plt.ylabel("Count")
    plt.title("Log((k1_bid*k2)/(k1_bim*k2))")
    rep.addCurrentFigure()
    plt.close()

    # Add source code to report 
    rep.addPythonCode('bh3_model.py')

    # Write report
    rep.writeReport('bh3_model_mcmc_fit')

    return mcmc

# Define the likelihood, prior, and step functions
# ================================================

def likelihood(mcmc, position):
    """The likelihood function. Calculates the error between model and data
    to give a measure of the likelihood of observing the data given the
    current parameter values.
    """

    # Get the current value for the time offset
    cur_t_offset = mcmc.cur_params(position=position)[T_OFFSET_INDEX]
    offset_tspan = exp_tspan + cur_t_offset

    # Calculate objective function values
    # ===================================
    (dmso_sim, bim_sim, bid_sim) = model.simulate(offset_tspan,
                                                mcmc.cur_params(position))
    #(bim_sim, bid_sim) = model.simulate(offset_tspan,
    #                                            mcmc.cur_params(position))

    #dmso_obj = numpy.sum((dmso_avgs - dmso_sim) ** 2 /
    #                     (2 * dmso_stds ** 2))
    bim_obj = numpy.sum((bim_avgs - bim_sim) ** 2 /
                         (2 * bim_stds ** 2))
    bid_obj = numpy.sum((bid_avgs - bid_sim) ** 2 /
                         (2 * bid_stds ** 2))

    #return (dmso_obj**2) + bim_obj + bid_obj
    return bim_obj + bid_obj
    #return dmso_obj

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

    if param_vals[4] < 0 or param_vals[4] > 10: # k1_bim
        return 1e15
    if param_vals[5] < 0 or param_vals[5] > 10: # k1_bid
        return 1e15
    if param_vals[6] < 0 or param_vals[6] > 10: # k2
        return 1e15
    if param_vals[7] < 1 or param_vals[7] > 10: # k2
        return 1e15
    if param_vals[8] < 0 or param_vals[8] > 10: # k_bg
        return 1e15
    if param_vals[3] < 0:
        return 1e15

    prior = 0
    prior += ((param_vals[0] - 1.85)**2) / (2 * 0.1**2) # fmax
    prior += ((param_vals[1] - 0.022)**2) / (2 *0.03**2) # k_agg
    prior += ((param_vals[2] - 2.5)**2) / (2*0.5**2) # t_offset
    prior += ((param_vals[3] - 0.0)**2) / (2 * 0.1**2) # f0
    
    return prior
    """
    if param_vals[0] < 0.50 or param_vals[0] > 2.0: # fmax
        return 1e15
    if param_vals[1] < 0 or param_vals[1] > 10: # k_agg
        return 1e15
    if param_vals[2] < 0 or param_vals[2] > 20: # t_offset
        return 1e15
    if param_vals[3] < 0 or param_vals[3] > 0.10: # f0
        return 1e15
    return 0
    """
    
# Plotting Functions
# ==================
def plot_curve_distribution(mcmc, mixed_positions, num_samples,
                            title=None, report=None):
    plt.figure()
    
    # Plot data
    plt.errorbar(exp_tspan, dmso_avgs, yerr=dmso_stds, label='DMSO data')
    plt.errorbar(exp_tspan, bim_avgs, yerr=bim_stds, label='Bim data')
    plt.errorbar(exp_tspan, bid_avgs, yerr=bid_stds, label='Bid data')

    for i in range(0, num_samples):
        num_mixed_positions = len(mixed_positions)
        rand_position_index = numpy.random.randint(num_mixed_positions)
        rand_position = mixed_positions[rand_position_index]

        # Get the current value for the time offset
        cur_t_offset = mcmc.cur_params(rand_position)[T_OFFSET_INDEX]
        offset_tspan = exp_tspan + cur_t_offset

        # Get the simulation data
        (dmso_sim, bim_sim, bid_sim) = mcmc.options.model.simulate(offset_tspan,
                                                mcmc.cur_params(rand_position))
   
        # Plot simulations
        plt.plot(exp_tspan, dmso_sim, 'r', alpha=0.01)
        plt.plot(exp_tspan, bim_sim, 'g', alpha=0.01)
        plt.plot(exp_tspan, bid_sim, 'b', alpha=0.01)

    #plt.ylim(0, 100.5)
    if title is not None:
        plt.title(title)
    plt.show()

    # Add figure to report 
    if report is not None:
        rep.addCurrentFigure()

def plot_position(mcmc, position, title=None, report=None):
    """Note that position is expected to be in log-transform format."""
    plt.ion()
    plt.figure()

    # Get the current value for the time offset
    cur_t_offset = mcmc.cur_params(position)[T_OFFSET_INDEX]
    offset_tspan = exp_tspan + cur_t_offset

    # Get the simulation data
    (dmso_sim, bim_sim, bid_sim) = mcmc.options.model.simulate(offset_tspan,
                                                    mcmc.cur_params(position))
   
    # Plot data
    plt.errorbar(offset_tspan, dmso_avgs, yerr=dmso_stds, label='DMSO data')
    plt.errorbar(offset_tspan, bim_avgs, yerr=bim_stds, label='Bim data')
    plt.errorbar(offset_tspan, bid_avgs, yerr=bid_stds, label='Bid data')

    # Plot simulations
    plt.plot(offset_tspan, dmso_sim, label='DMSO')
    plt.plot(offset_tspan, bim_sim, label='Bim')
    plt.plot(offset_tspan, bid_sim, label='Bid')

    #plt.ylim(0, 100.5)
    if title is not None:
        plt.title(title)
    plt.show()

    # Add figure to report 
    if report is not None:
        rep.addCurrentFigure()

# Mathematical helpers
# ====================
def gaussian_pdf(x, mu, sigma):
    return (1./sigma) * phi((x - mu)/sigma)

def phi(x):
    return (1 / numpy.sqrt(2*numpy.pi)) * numpy.exp((-1./2.) * x**2.)

# Main function
# =============
if __name__ == '__main__':
    do_fit()


