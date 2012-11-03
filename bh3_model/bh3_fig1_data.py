import numpy

# Prepare the data
# ================
#wt = numpy.loadtxt('wt_mef.csv', skiprows=1, delimiter=',')
data = numpy.loadtxt('bax_ko_mef.csv', skiprows=1, delimiter=',')
#wt = numpy.loadtxt('bak_ko_mef.csv', skiprows=1, delimiter=',')

TIME_COL = 0
DMSO_MEAN = 1
DMSO_STD = 2 
FCCP_MEAN = 4
FCCP_STD =5 
BIM_MEAN = 7
BIM_STD = 8
BID_MEAN = 10
BID_STD = 11

exp_tspan = data[:, TIME_COL]
sim_tspan = numpy.linspace(0,240,241)

fccp_avgs = data[:, FCCP_MEAN]
fccp_stds = data[:, FCCP_STD]
dmso_avgs = data[:, DMSO_MEAN]
dmso_stds = data[:, DMSO_STD]
bim_avgs = data[:, BIM_MEAN]
bim_stds = data[:, BIM_STD]
bid_avgs = data[:, BID_MEAN]
bid_stds = data[:, BID_STD]
