# compute models in parallel: multiprocess

addprocs(Sys.CPU_CORES) # add a worker process per core
print_with_color(:white, "Setup:\n")
println("  > Using $(nprocs()-1) worker processes")
n_core = nprocs()-1#2

if nworkers() < n_core
    addprocs(n_core-nworkers(); exeflags="--check-bounds=yes")
end
@assert nprocs() > n_core
@assert nworkers() >= n_core

println(workers())

@everywhere using PBupsModel
@everywhere using Base.Test
@everywhere using MAT

# read test data file
mpath = "data"
fname = "testdata.mat"
ratdata, ntrials = LoadData(mpath, fname)

dt = 0.02

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

params = [lam, sigma_a, sigma_s, sigma_i, B, phi, tau_phi, bias, lapse] 

LL = LogLikelihood(params, RightClickTimes, LeftClickTimes, Nsteps, rat_choice)

# write your own tests here
@test (LL - -0.9972) < 0.0001

###### test2 : Compute Logliklihood of 40 trials
ntrials = 40
LLs = SharedArray(Float64, ntrials)
LL2 = ComputeLL(LLs, params, ratdata["rawdata"], ntrials)

@test (LL2 - 9.0591) < 0.0001

# ###### test3 : Compute Gradients of 40 trials
# LL, LLgrad = ComputeGrad(params, ratdata["rawdata"], ntrials)
# print(LLgrad)

# ###### test4 : Compute Hessian Matrix of 40 trials
# LL, LLgrad, LLhess = ComputeHess(params, ratdata["rawdata"], ntrials)
# print(LLhess)

###### test5 : Model Optimization
init_params = InitParams()
result = ModelFitting(init_params, ratdata, ntrials)
FitSummary(mpath, fname, D)

