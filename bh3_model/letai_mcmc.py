"""Functions for fitting the tBid/Bax/Lipo models to the NBD-Bax mutant
fluorescence data using MCMC.
"""

import bayessb
from pysb.integrate import odesolve
import numpy
import matplotlib.pyplot as plt
from letai import model
import pickle
from tbidbaxlipo.util.report import Report
from scipy.interpolate import interp1d

TIME_COL = 0
DMSO_MEAN = 1
DMSO_STD = 2 
FCCP_MEAN = 4
FCCP_STD =5 
BIM_MEAN = 7
BIM_STD = 8
BID_MEAN = 10
BID_STD = 11

# Prepare the data
# ================
wt = numpy.loadtxt('wt_mef.csv', skiprows=1, delimiter=',')
exp_tspan = wt[:, TIME_COL]
sim_tspan = numpy.linspace(0,240,241)
fccp_avgs = wt[:, FCCP_MEAN]
fccp_stds = wt[:, FCCP_STD]
dmso_avgs = wt[:, DMSO_MEAN]
dmso_stds = wt[:, DMSO_STD]
bim_avgs = wt[:, BIM_MEAN]
bim_stds = wt[:, BIM_STD]
bid_avgs = wt[:, BID_MEAN]
bid_stds = wt[:, BID_STD]


# Get the index of the t_offset parameter
# =======================================
T_OFFSET_INDEX = model.parameters.index(model.parameters['t_offset'])
exp_dmso_max = max(dmso_avgs)
exp_fccp_min = min(fccp_avgs)
 

# TODO: Need to make it so the global data array contains all of the
# data, appropriately normalized
#ydata_norm = dmso_avgs

# MCMC Functions
# ==============

def get_norm_sims(mcmc, position):
    # Run simulations for all conditions
    # ==================================
    # Run simulation for FCCP
    #import pdb; pdb.set_trace()
    model.parameters['Bim_0'].value = 0
    model.parameters['Bid_0'].value = 0
    model.parameters['FCCP_0'].value = 1
    model.parameters['M_0'].value = 0
    yobs = mcmc.simulate(position, observables=True)
    fccp_sim = numpy.array(yobs['jc1'])

    # Run simulation for DMSO
    model.parameters['Bim_0'].value = 0
    model.parameters['Bid_0'].value = 0
    model.parameters['FCCP_0'].value = 0
    model.parameters['M_0'].value = 0
    yobs = mcmc.simulate(position, observables=True)
    dmso_sim = numpy.array(yobs['jc1'])

    # Run simulation for Bim
    if False:
        model.parameters['Bim_0'].value = 10
        model.parameters['M_0'].value = 0
        model.parameters['Bid_0'].value = 0
        model.parameters['FCCP_0'].value = 0
        yobs = mcmc.simulate(position, observables=True)
        bim_sim = numpy.array(yobs['jc1'])

        # Run simulation for Bid
        model.parameters['Bim_0'].value = 0
        model.parameters['M_0'].value = 0
        model.parameters['Bid_0'].value = 3
        model.parameters['FCCP_0'].value = 0
        yobs = mcmc.simulate(position, observables=True)
        bid_sim = numpy.array(yobs['jc1'])

        bim_norm = (bim_sim - sim_fccp_min) / (sim_dmso_max - sim_fccp_min)
        bid_norm = (bid_sim - sim_fccp_min) / (sim_dmso_max - sim_fccp_min)

        bim_norm = (bim_norm * exp_range) + exp_fccp_min
        bid_norm = (bid_norm * exp_range) + exp_fccp_min

    # Run simulation for M
    model.parameters['Bim_0'].value = 0
    model.parameters['Bid_0'].value = 0
    model.parameters['FCCP_0'].value = 0
    model.parameters['M_0'].value = 10
    yobs = mcmc.simulate(position, observables=True)
    m_sim = numpy.array(yobs['jc1'])

    # Normalize each curve to within-sim range: 
    # =========================================
    # Note: all signals are normalized to the max achieved by simulated
    # DMSO trajectory. This way the DMSO trajectory always reaches 100%
    # and the other conditions reach some fraction of that.
    sim_fccp_min = min(fccp_sim)
    sim_dmso_max = max(dmso_sim)

    fccp_norm = (fccp_sim - sim_fccp_min) / (sim_dmso_max - sim_fccp_min)
    dmso_norm = (dmso_sim - sim_fccp_min) / (sim_dmso_max - sim_fccp_min)
    m_norm = (m_sim - sim_fccp_min) / (sim_dmso_max - sim_fccp_min)

    # Rescale to use scale of experimental data:
    # ==========================================
    exp_range = exp_dmso_max - exp_fccp_min
    fccp_norm = (fccp_norm * exp_range) + exp_fccp_min
    dmso_norm = (dmso_norm * exp_range) + exp_fccp_min
    m_norm = (m_norm * exp_range) + exp_fccp_min

    #return (fccp_norm, dmso_norm, bim_norm, bid_norm, m_norm)
    return (fccp_norm, dmso_norm, None, None, m_norm)

def plot_position(mcmc, position):
    plt.ion()
    plt.figure()

    # Get the simulation data
    (fccp_norm, dmso_norm, bim_norm, bid_norm, m_norm) = get_norm_sims(mcmc, position)

    # Get the current value for the time offset
    cur_t_offset = mcmc.cur_params(position=position)[T_OFFSET_INDEX]
    offset_tspan = exp_tspan + cur_t_offset
   
    # Plot data
    plt.plot(offset_tspan, fccp_avgs, 'x', label='FCCP data') # Plot FCCP data
    plt.plot(offset_tspan, dmso_avgs, 'x', label='DMSO data') # Plot DMSO data
    plt.plot(offset_tspan, bim_avgs, 'x', label='Bim data')  # Plot Bim data
    plt.plot(offset_tspan, bid_avgs, 'x', label='Bid data')  # Plot Bid data

    # Plot simulations
    plt.plot(sim_tspan, fccp_norm, label='FCCP') # Plot FCCP data
    plt.plot(sim_tspan, dmso_norm, label='DMSO') # Plot DMSO data
    #plt.plot(sim_tspan, bim_norm, label='Bim')  # Plot Bim data
    #plt.plot(sim_tspan, bid_norm, label='Bid')  # Plot Bid data
    plt.plot(sim_tspan, m_norm, label='M')  # Plot M data

    plt.ylim(0, 100.5)
    plt.legend(loc='center right')
    plt.show()

    # Add figure to report 
    rep = Report()
    rep.addCurrentFigure()
    rep.writeReport('letai_mcmc_fit')

def do_fit():
    """Runs MCMC on the globally defined model."""

    # Define the likelihood function
    def likelihood(mcmc, position):
        """The likelihood function. Calculates the error between model and data
        to give a measure of the likelihood of observing the data given the
        current parameter values.
        """

        # Get the simulation data for this position in parameter space
        (fccp_norm, dmso_norm, bim_norm, bid_norm, m_norm) = get_norm_sims(mcmc, position)

        # Get the current value for the time offset
        cur_t_offset = mcmc.cur_params(position=position)[T_OFFSET_INDEX]
        offset_tspan = exp_tspan + cur_t_offset

        # Calculate objective function values
        # ===================================
        # FCCP
        # ----
        # Calculate differences at the offset timepoints
        # - scipy.interpolate.interp1d takes lists of x and y values and
        #   returns and interpolation function
        # - numpy.vectorize takes a function f(a) -> b and turns into in a
        #   function that operates on numpy vectors/arrays:
        #   f([a1, a2, a3 ...]) -> [b1, b2, b3...]
        fccp_f = numpy.vectorize(interp1d(sim_tspan, fccp_norm, bounds_error=False,
            fill_value=numpy.inf))
        fccp_interp = fccp_f(offset_tspan)
        fccp_obj = numpy.sum((fccp_avgs - fccp_interp) ** 2 /
                             (2 * fccp_stds ** 2))

        # DMSO
        # ----
        dmso_f = numpy.vectorize(interp1d(sim_tspan, dmso_norm, bounds_error=False,
            fill_value=numpy.inf))
        dmso_interp = dmso_f(offset_tspan)
        dmso_obj = numpy.sum((dmso_avgs - dmso_interp) ** 2 /
                             (2 * dmso_stds ** 2))

        # Bim
        # ---
        # Fit the Bim curve with the M function! # FIXME
        bim_f = numpy.vectorize(interp1d(sim_tspan, m_norm, bounds_error=False,
            fill_value=numpy.inf))
        #bim_f = numpy.vectorize(interp1d(sim_tspan, bim_norm, bounds_error=False,
        #    fill_value=numpy.inf))
        bim_interp = bim_f(offset_tspan)
        bim_obj = numpy.sum((bim_avgs - bim_interp) ** 2 /
                             (2 * bim_stds ** 2))

        # Bid
        # ---
        """
        bid_f = numpy.vectorize(interp1d(sim_tspan, bid_norm, bounds_error=False,
            fill_value=numpy.inf))
        bid_interp = bid_f(offset_tspan)
        bid_obj = numpy.sum((bid_avgs - bid_interp) ** 2 /
                             (2 * bid_stds ** 2))
        """

        # Add up all terms in the objective function
        #return fccp_obj + dmso_obj + bim_obj + bid_obj
        return fccp_obj + dmso_obj + bim_obj

    # Set the random number generator seed
    seed = 2
    random = numpy.random.RandomState(seed)

    # Initialize the MCMC arguments
    opts = bayessb.MCMCOpts()
    opts.model = model
    opts.tspan = sim_tspan

    # Estimate rates only (not initial conditions) from wild guesses
    #opts.estimate_params = [p for p in model.parameters
    #                          if not p.name.endswith('_0')]

    #opts.estimate_params = [model.parameters['dye_uptake_kf'],
    #                        model.parameters['spontaneous_release_kf'],
    #                        model.parameters['t_offset']]

    # Parameters to fit for simple catalysis model
    #opts.estimate_params = [model.parameters['Bim_kf'],
    #                        model.parameters['Bid_kf']]

    # Parameters to fit for catalyst activation model
    #opts.estimate_params = [model.parameters['Bim_activation_kf'],
    #                        model.parameters['Bid_activation_kf'],
    #                        model.parameters['BimA_kf'],
    #                        model.parameters['BidA_kf'],
    #                        model.parameters['dye_uptake_kf']]

    # Parameters to fit for catalyst assembly model
    opts.estimate_params = [model.parameters['M_activation_kf'],
                            model.parameters['pore_kf'],
                            model.parameters['pore_kr'],
                            model.parameters['pore_mito_kf'],
                            model.parameters['spontaneous_release_kf'],
                            model.parameters['dye_uptake_kf'],
                            model.parameters['t_offset']]

    opts.initial_values = [p.value for p in opts.estimate_params]
    opts.nsteps = 5000 
    opts.likelihood_fn = likelihood
    opts.step_fn = step
    opts.use_hessian = True
    opts.hessian_period = opts.nsteps / 10 # Calculate the Hessian 10 times
    opts.seed = seed
    mcmc = bayessb.MCMC(opts)

    # Plot before curves
    #plot_position(mcmc, mcmc.position)

    # Run it!
    mcmc.run()


    # Set to best fit position
    mcmc.position = mcmc.positions[numpy.argmin(mcmc.likelihoods)]
    p_names = [p.name for p in opts.estimate_params]
    p_name_vals = zip(p_names, mcmc.position)
    print ("position: " + str(p_name_vals))

    # Plot "After" curves
    plot_position(mcmc, mcmc.position)

    # Pickle it!
    #mcmc.solver = None # FIXME This is a hack to make the MCMC pickleable
    #mcmc.options.likelihood_fn = None
    #output_file = open('letai_simple_5k.pck', 'w')
    #pickle.dump(mcmc, output_file)
    #output_file.close()

    return mcmc

def phi(x):
    return (1 / numpy.sqrt(2*numpy.pi)) * numpy.exp((-1./2.) * x**2.)

def prior(mcmc, position):
    # TODO Need to put some decent priors on the on and off rates
    mean = math.log10([1e-2, 1e7])
    var = [100, 100]
    return numpy.sum((position - means) ** 2 / ( 2 * var))

def step(mcmc):
    """The function to call at every iteration. Currently just prints
    out a few progress indicators.
    """
    if mcmc.iter % 20 == 0:
        print 'iter=%-5d  sigma=%-.3f  T=%-.3f  acc=%-.3f, lkl=%g  prior=%g  post=%g' % \
            (mcmc.iter, mcmc.sig_value, mcmc.T, mcmc.acceptance/(mcmc.iter+1),
             mcmc.accept_likelihood, mcmc.accept_prior, mcmc.accept_posterior)


# Plotting
# ========

def plot_from_params(params_dict):
    """Plot the model output using the given parameter value."""

    # The specific model may need to be changed here
    m1c = tBid_Bax_1c(params_dict=params_dict)
    m1c.build_model2()
    model = m1c.model

    plt.ion()

    plt.plot(tspan, ydata_norm)
    output = odesolve(model, tspan)
    #output_array = output.view().reshape(len(tspan), len(output.dtype))
    iBax = 2*output['Bax2'] + 4*output['Bax4']
    iBax_norm = iBax / max(iBax)
    plt.plot(tspan, iBax_norm[:])

    plt.show()


# Main function
# =============

if __name__ == '__main__':
    do_fit()



"""
Notes

Basic version:

We fit the kinetic parameters in the model and make assumptions about the observables
and their relation to the NBD signals (e.g., we can first do Bax2 for the Bax62C data,
and compare it to a case where Bax62C is tBid/Bax driven).

So--need to load the data (the three curves, and then normalize it to 0->1)
Then run the model, fitting only the kinetic parameters (not the initial conditions),
evaluate the objective function over the timecourse. Use a figure for the error based on the
variance in the data in a relatively straight-line area, and/or over a short distance.

Build out the ODE/core model with Bax2 binding and Bax4. So will have many parameters...

Advanced version

Ideally, would have a way of pre-equilibrating the system for the just-Bax condition,
and then perturb it with the addition of tBid.

Could develop more comprehensive/enumerated set of hypotheses where the Bax binding was due
to other states
"""
