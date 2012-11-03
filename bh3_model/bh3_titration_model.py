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

T_OFFSET_INDEX = 3
#exp_dmso_max = max(dmso_avgs)
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
        self.num_params = 21 
        self.f0 = f0

        self.parameters = [None] * self.num_params
        self.observables = [None] * 1
        self.initial_conditions = []
        self.sim_param_values = numpy.empty(self.num_params)
    
        self.parameters[0] = Parameter('fmax', 1.1629)
        self.parameters[1] = Parameter('k_agg', 0.020)
        self.parameters[2] = Parameter('k_bg', 0.002261)
        self.parameters[3] = Parameter('t_offset', 4.0)
        self.parameters[4] = Parameter('f0', 0)

        k1_initval = 0.0135
        self.parameters[5] = Parameter('k1_bim100', 0.032)
        self.parameters[6] = Parameter('k1_bim30', 0.023)
        self.parameters[7] = Parameter('k1_bim10', 0.0147)
        self.parameters[8] = Parameter('k1_bim3', 0.0097)
        self.parameters[9] = Parameter('k1_bim1', 0.0095)
        #self.parameters[10] = Parameter('k1_bim03', 0.0185)
        #self.parameters[11] = Parameter('k1_bim01', 0.009)
        #self.parameters[12] = Parameter('k1_bim003', 0.006)
        #self.parameters[13] = Parameter('k1_bim001', 0.001)

        self.parameters[10] = Parameter('k1_bid100', 0.065)
        self.parameters[11] = Parameter('k1_bid30', 0.054)
        self.parameters[12] = Parameter('k1_bid10', 0.0428)
        self.parameters[13] = Parameter('k1_bid3', 0.0300)
        self.parameters[14] = Parameter('k1_bid1', 0.025)
        #self.parameters[19] = Parameter('k1_bid03', 0.022)
        #self.parameters[20] = Parameter('k1_bid01', 0.01)
        #self.parameters[21] = Parameter('k1_bid003', 0.01)
        #self.parameters[22] = Parameter('k1_bid001', 0.01)

        self.parameters[15] = Parameter('k2_100', 1)
        self.parameters[16] = Parameter('k2_30', 2)
        self.parameters[17] = Parameter('k2_10', 3)
        self.parameters[18] = Parameter('k2_3', 4)
        self.parameters[19] = Parameter('k2_1', 5)
        #self.parameters[28] = Parameter('k2_03', 0.0037)
        #self.parameters[29] = Parameter('k2_01', 0.001)
        #self.parameters[30] = Parameter('k2_003', 0.001)
        #self.parameters[31] = Parameter('k2_001', 0.0001)

        self.parameters[20] = Parameter('n', 0.2)

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
        k_bg = self.sim_param_values[2]
        t_offset = self.sim_param_values[3]
        f0 = self.sim_param_values[4]
        n = self.sim_param_values[20]

        #dmso = (f0 +
        #       (fmax*
        #       (1 - numpy.exp(-k_agg*(tspan + t_offset)))) *
        #       numpy.exp(-k_bg*(tspan + t_offset)))

        # Catalyst Activation Model
        # ==========================
        bim = numpy.zeros((MAX_TIME_INDEX, len(BIM_RANGE)))
        bid = numpy.zeros((MAX_TIME_INDEX, len(BID_RANGE)))

        #### A version with the k2 linearly dependent on concentration
        #def catalyst_activation(k1, conc):
        #    return (f0 + (fmax * (1 - numpy.exp(-k_agg*(tspan + t_offset)))) *
        #            numpy.exp(-k1 * ((1 - numpy.exp(-((k2_slope*conc)+k2_int)*(tspan+t_offset)))**n) * (tspan + t_offset)))

        # A version with the k2 shared for each concentration
        def catalyst_activation(k1, k2):
            return (f0 + (fmax * (1 - numpy.exp(-k_agg*(tspan + t_offset)))) *
                    numpy.exp(-k1 * ((1 - numpy.exp(-k2*(tspan+t_offset)))**n) * (tspan + t_offset)))

        def logistic_activation(k1, k2):
            return (f0 + (fmax * (1 - numpy.exp(-k_agg*(tspan + t_offset)))) *
                    numpy.exp(-k1 * (1/(1 + numpy.exp(-n*(tspan + k2)))) * (tspan + t_offset)))

        for i in range(len(BIM_RANGE)):
            bim[:, i] = catalyst_activation(self.sim_param_values[5 + i], 
                                            self.sim_param_values[15 + i])
            bid[:, i] = catalyst_activation(self.sim_param_values[10 + i],
                                            self.sim_param_values[15 + i])

        #return (dmso, bim, bid) 
        return (bim, bid) 

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
    opts.nsteps = 1000000
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
    #plt.close()

    # Run it!
    mcmc.estimate()

    # Plot best fit position
    mcmc.position = mcmc.positions[numpy.argmin(mcmc.likelihoods)]
    plot_position(mcmc, mcmc.position, title="Best Fit", report=rep)
    #plt.close()
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

    """
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

    # Plot k1_bim/k2_bim surface
    bayessb.plot.surf(mcmc, 4, 6, mask=mixed_start)
    plt.title("k1_bim/k2_bim surface")
    rep.addCurrentFigure()
    plt.close()

    # Plot k1_bid/k2_bid surface
    bayessb.plot.surf(mcmc, 5, 7, mask=mixed_start)
    plt.title("k1_bid/k2_bid surface")
    rep.addCurrentFigure()
    plt.close()

    # Plot histogram of k1_bid, k2_bid, and sum
    plt.figure()
    bid_sum = mixed_accept_positions[:,5] + mixed_accept_positions[:,7]
    plt.hist(mixed_accept_positions[:,5], bins=20, alpha=0.5, label="k1_bid")
    plt.hist(mixed_accept_positions[:,7], bins=20, alpha=0.5, label="k2_bid")
    plt.hist(bid_sum, bins=20, alpha=0.5, label="k1_bid*k2_bid") 
    plt.xlabel("Log(k)")
    plt.ylabel("Count")
    plt.title("k1_bid, k2_bid, and k1_bid*k2_bid")
    plt.legend(loc="center right")
    rep.addCurrentFigure()
    plt.close()

    # Plot histogram of k1_bim, k2_bim, and sum
    plt.figure()
    bim_sum = mixed_accept_positions[:,4] + mixed_accept_positions[:,6]
    plt.hist(mixed_accept_positions[:,4], bins=20, alpha=0.5)
    plt.hist(mixed_accept_positions[:,6], bins=20, alpha=0.5)
    plt.hist(bim_sum, bins=20, alpha=0.5) 
    plt.xlabel("Log(k)")
    plt.ylabel("Count")
    plt.title("k1_bim, k2_bim, and k1_bim*k2_bim")
    plt.legend(loc="center right")
    rep.addCurrentFigure()
    plt.close()

    # Plot histogram of bid_sum - bim_sum 
    plt.figure()
    ratio = bid_sum - bim_sum
    plt.hist(ratio, bins=20)
    plt.xlabel("Log(k)")
    plt.ylabel("Count")
    plt.title("Log((k1_bid*k2_bid)/(k1_bim*k2_bim))")
    rep.addCurrentFigure()
    plt.close()
    """

    # Add source code to report 
    rep.addPythonCode('bh3_titration_model.py')

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
    (bim_sim, bid_sim) = model.simulate(offset_tspan,
                                                mcmc.cur_params(position))

    #dmso_obj = numpy.sum((dmso_avgs - dmso_sim) ** 2 /
    #                     (2 * dmso_stds ** 2))

    bim_obj = 0
    for i in range(len(BIM_RANGE)):
        bim_obj += numpy.sum((bim_avgs[:, i] - bim_sim[:, i]) ** 2 /
                (2 * bim_stds[:, i] ** 2))

    bid_obj = 0
    for i in range(len(BID_RANGE)):
        bid_obj += numpy.sum((bid_avgs[:, i] - bid_sim[:, i]) ** 2 /
                (2 * bid_stds[:, i] ** 2))

    # return (dmso_obj**2) + bim_obj
    #return (dmso_obj**2) + bim_obj + bid_obj
    return bim_obj + bid_obj
 
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

    if param_vals[0] < 0.1 or param_vals[0] > 10: # fmax
        return 1e15
    if param_vals[1] < 0 or param_vals[1] > 10: # k_agg
        return 1e15
    if param_vals[2] < 0 or param_vals[2] > 10: # k_bg
        return 1e15
    if param_vals[3] < 0 or param_vals[3] > 20: # t_offset
        return 1e15
    if param_vals[4] < 0 or param_vals[4] > 0.4: # f0
        return 1e15
    for i in range(5, 20):                       # All rate parameters
        if param_vals[i] < 0 or param_vals[i] > 50:
            return 1e15
    if param_vals[20] < 0 or param_vals[20] > 10:  # n
        return 1e15

    return 0
    
# Plotting Functions
# ==================
def plot_curve_distribution(mcmc, mixed_positions, num_samples,
                            title=None, report=None):
    plt.figure()
    
    # Plot data
    #plt.plot(exp_tspan, dmso_avgs, 'rx', label='DMSO data')
    plt.plot(exp_tspan, bim_avgs, 'gx', label='Bim data')
    plt.plot(exp_tspan, bid_avgs, 'bx', label='Bid data')

    for i in range(0, num_samples):
        num_mixed_positions = len(mixed_positions)
        rand_position_index = numpy.random.randint(num_mixed_positions)
        rand_position = mixed_positions[rand_position_index]

        # Get the current value for the time offset
        cur_t_offset = mcmc.cur_params(rand_position)[T_OFFSET_INDEX]
        offset_tspan = exp_tspan + cur_t_offset

        # Get the simulation data
        (bim_sim, bid_sim) = mcmc.options.model.simulate(offset_tspan,
                                                mcmc.cur_params(rand_position))
   
        # Plot simulations
        #plt.plot(exp_tspan, dmso_sim, 'r', alpha=0.01)
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

    # Get the current value for the time offset
    cur_t_offset = mcmc.cur_params(position)[T_OFFSET_INDEX]
    offset_tspan = exp_tspan + cur_t_offset

    # Get the simulation data
    (bim_sim, bid_sim) = mcmc.options.model.simulate(offset_tspan,
                                                    mcmc.cur_params(position))
   
    # Plot Bim data
    plt.figure()
    for i in range(len(BIM_RANGE)):
        plt.plot(offset_tspan, bim_avgs[:,i], 'x')
    # Plot Bim simulations
    for i in range(len(BIM_RANGE)):
        plt.plot(offset_tspan, bim_sim[:, i])

    if title is not None:
        plt.title(title + ", Bim")
    plt.show()

    # Add figure to report 
    if report is not None:
        rep.addCurrentFigure()

    # Plot Bid data
    plt.figure()
    for i in range(len(BID_RANGE)):
        plt.plot(offset_tspan, bid_avgs[:,i], 'x')
    # Plot Bid simulations
    for i in range(len(BID_RANGE)):
        plt.plot(offset_tspan, bid_sim[:, i])

    if title is not None:
        plt.title(title + ", Bid")
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


