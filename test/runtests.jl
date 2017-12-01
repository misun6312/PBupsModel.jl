# compute models in parallel: multiprocess
# addprocs(Sys.CPU_CORES) # add a worker process per core
# print_with_color(:white, "Setup:\n")
# println("  > Using $(nprocs()-1) worker processes")

# @everywhere using PBupsModel
# @everywhere using Base.Test
# @everywhere using MAT

dt = 0.02

# read test data file
ratdata = matread("data/testdata.mat")

# using TrialData load trial data
RightClickTimes, LeftClickTimes, maxT, rat_choice = TrialData(ratdata["rawdata"], 1)

Nsteps = Int(ceil(maxT/dt))

# sigma_a = rand()*4.; sigma_s = rand()*4.; sigma_i = rand()*30.;
# lam = randn()*0.4; B = rand()*12.+5.; bias = randn();
# phi = rand()*1.39+0.01; tau_phi = 0.695*rand()+0.005; lapse = rand();

# known parameter set
sigma_a = 1; sigma_s = 0.1; sigma_i = 0.2; 
lam = -0.5; B = 6.1; bias = 0.1; 
phi = 0.3; tau_phi = 0.1; lapse = 0.05*2;

params = [sigma_a, sigma_s, sigma_i, lam, B, bias, phi, tau_phi, lapse]

LL = LogLikelihood(params, RightClickTimes, LeftClickTimes, Nsteps, rat_choice)

# write your own tests here
@test (LL - -0.9972) < 0.0001
