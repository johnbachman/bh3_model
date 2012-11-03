from letai import model

from pysb.integrate import odesolve
import numpy as np
import matplotlib.pyplot as plt

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

wt = np.loadtxt('wt_mef.csv', skiprows=1, delimiter=',')
exp_tspan = wt[:, TIME_COL]
sim_tspan = np.linspace(0,240,241)
fccp_avgs = wt[:, FCCP_MEAN]
fccp_stds = wt[:, FCCP_STD]
dmso_avgs = wt[:, DMSO_MEAN]
dmso_stds = wt[:, DMSO_STD]
bim_avgs = wt[:, BIM_MEAN]
bim_stds = wt[:, BIM_STD]
bid_avgs = wt[:, BID_MEAN]
bid_stds = wt[:, BID_STD]

def plot_nominal():

    plt.ion()
    plt.figure()

    # Some values we'll need:
    # =======================
    cur_t_offset = model.parameters['t_offset'].value
    offset_tspan = exp_tspan + cur_t_offset
    exp_dmso_max = max(dmso_avgs)
    exp_fccp_min = min(fccp_avgs)

    # Run simulations for all conditions
    # ==================================
    # Run simulation for FCCP
    model.parameters['Bim_0'].value = 0
    model.parameters['Bid_0'].value = 0
    model.parameters['FCCP_0'].value = 1
    model.parameters['M_0'].value = 0
    fccp_sim = odesolve(model, sim_tspan)

    # Run simulation for DMSO
    model.parameters['Bim_0'].value = 0
    model.parameters['Bid_0'].value = 0
    model.parameters['FCCP_0'].value = 0
    model.parameters['M_0'].value = 0
    dmso_sim = odesolve(model, sim_tspan)

    # Run simulation for Bim
    model.parameters['Bim_0'].value = 10
    model.parameters['M_0'].value = 0
    model.parameters['Bid_0'].value = 0
    model.parameters['FCCP_0'].value = 0
    bim_sim = odesolve(model, sim_tspan)

    # Run simulation for Bid
    model.parameters['Bim_0'].value = 0
    model.parameters['M_0'].value = 0
    model.parameters['Bid_0'].value = 3
    model.parameters['FCCP_0'].value = 0
    bid_sim = odesolve(model, sim_tspan)

    # Run simulation for M
    model.parameters['Bim_0'].value = 0
    model.parameters['Bid_0'].value = 0
    model.parameters['FCCP_0'].value = 0
    model.parameters['M_0'].value = 10
    m_sim = odesolve(model, sim_tspan)

    # Normalize each curve to within-sim range: 
    sim_dmso_max = max(dmso_sim['jc1'])
    sim_fccp_min = min(fccp_sim['jc1'])

    fccp_norm = (fccp_sim['jc1'] - sim_fccp_min) / (sim_dmso_max - sim_fccp_min)
    dmso_norm = (dmso_sim['jc1'] - sim_fccp_min) / (sim_dmso_max - sim_fccp_min)
    bim_norm = (bim_sim['jc1'] - sim_fccp_min) / (sim_dmso_max - sim_fccp_min)
    bid_norm = (bid_sim['jc1'] - sim_fccp_min) / (sim_dmso_max - sim_fccp_min)
    m_norm = (m_sim['jc1'] - sim_fccp_min) / (sim_dmso_max - sim_fccp_min)

    # Rescale to use scale of experimental data:
    # ==========================================
    exp_range = exp_dmso_max - exp_fccp_min
    dmso_norm = (dmso_norm * exp_range) + exp_fccp_min
    fccp_norm = (fccp_norm * exp_range) + exp_fccp_min
    bim_norm = (bim_norm * exp_range) + exp_fccp_min
    bid_norm = (bid_norm * exp_range) + exp_fccp_min
    m_norm = (m_norm * exp_range) + exp_fccp_min

    # Plot data
    # =========
    plt.plot(offset_tspan, fccp_avgs, 'x', label='FCCP data') # Plot FCCP data
    plt.plot(offset_tspan, dmso_avgs, 'x', label='DMSO data') # Plot DMSO data
    plt.plot(offset_tspan, bim_avgs, 'x', label='Bim data')  # Plot Bim data
    plt.plot(offset_tspan, bid_avgs, 'x', label='Bid data')  # Plot Bim data

    # Plot simulations
    # ================
    plt.plot(sim_tspan, fccp_norm, label='FCCP') # Plot FCCP data
    plt.plot(sim_tspan, dmso_norm, label='DMSO') # Plot DMSO data
    plt.plot(sim_tspan, bim_norm, label='Bim')  # Plot Bim data
    plt.plot(sim_tspan, bid_norm, label='Bid')  # Plot Bid data
    plt.plot(sim_tspan, m_norm, label='M')  # Plot M data

    plt.legend(loc='center right')
    plt.ylim(-0.5, 100.5)

    plt.show()

if __name__ == "__main__":
    plot_nominal()
