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
sigma_a = 1.; sigma_s = 0.1; sigma_i = 0.2; 
lam = -0.5; B = 6.1; bias = 0.1; 
phi = 0.3; tau_phi = 0.1; lapse = 0.05*2;

params = [lam, sigma_a, sigma_s, sigma_i, B, phi, tau_phi, bias, lapse] 

# LL = LogLikelihood(params, RightClickTimes, LeftClickTimes, Nsteps, rat_choice)

# # write your own tests here
# @test (LL - -0.9972) < 0.0001

# # ###### test2 : Compute Logliklihood of 40 trials
# ntrials = 40
# LLs = SharedArray(Float64, ntrials)
# LL2 = ComputeLL(LLs, params, ratdata["rawdata"], ntrials)

# @test (LL2 - 9.0591) < 0.0001

# # # ###### test3 : Compute Gradients of 40 trials
# LL, LLgrad = ComputeGrad(params, ratdata["rawdata"], ntrials)
# print(LLgrad)

# # ###### test4 : Compute Hessian Matrix of 40 trials
# LL, LLgrad, LLhess = ComputeHess(params, ratdata["rawdata"], ntrials)
# print(LLhess)

# ###### test5 : Model Optimization
# init_params = InitParams()
# result = ModelFitting(init_params, ratdata, ntrials)
# FitSummary(mpath, fname, result)


# ====================== new test ====================== #

# to generalize the model.. 
# any subset of parameters in 12-param model. 
# change the model_likelihood.jl

# 9p / 12p (+3 bias params) 
# 8p / 11p sigma_i = 0
# 10p (without adaptation parameters)

# input should be "set of name of parameters", "and their values" 
# where shall we handle the sigma_s_L .. sigma_s
# if someone specify sigma_s_L/sigma_s_R,then use them
# if someone does not specify both sigma_s_L and sigma_s_R, then use one of them for sigma_s
# if someone does not specify none of sigma_s_L and sigma_s_R, then use default value for sigma_s

# how do we check that each parameter is set or not.
# compare it with default value... for now

ntrials = 40
LLs = SharedArray(Float64, ntrials)

# ===== 9p ===== #

# # known parameter set
# args = ["sigma_a","sigma_s_R","sigma_i","lambda","B","bias","phi","tau_phi","lapse_R"]
# x = [1., 0.1, 0.2, -0.5, 6.1, 0.1, 0.3, 0.1, 0.05*2]

# # LL2 = ComputeLL(LLs, ratdata["rawdata"], ntrials
# #     ;make_dict(args, x)...)
# LL2 = ComputeLL(LLs, ratdata["rawdata"], ntrials, args, x)

# @test (LL2 - 9.0591) < 0.0001

# # ###### test3 : Compute Gradients of 40 trials
# LL, LLgrad = ComputeGrad(ratdata["rawdata"], ntrials, args, x)
# print(LLgrad)

# # ###### test4 : Compute Hessian Matrix of 40 trials
# LL, LLgrad, LLhess = ComputeHess(ratdata["rawdata"], ntrials, args, x)
# print(LLhess)

# # ###### test5 : Model Optimization
# args = ["sigma_a","sigma_s_R","sigma_i","lambda","B","bias","phi","tau_phi","lapse_R"]
# init_params = InitParams(args)
# result = ModelFitting(args, init_params, ratdata, ntrials)
# FitSummary(mpath, fname, result)

# ===== 12p ===== #
args_12p = ["sigma_a","sigma_s_R","sigma_s_L","sigma_i","lambda","B","bias","phi","tau_phi","lapse_R","lapse_L","input_gain_weight"]
x_12p = [1., 0.1, 50, 0.2, -0.5, 6.1, 0.1, 0.3, 0.1, 0.05*2, 0.2, 0.4]

ntrials = 40
LLs = SharedArray(Float64, ntrials)
LL = ComputeLL(LLs, ratdata["rawdata"], ntrials, args_12p, x_12p)
print(LL)

# ###### test3 : Compute Gradients of 40 trials
LL, LLgrad = ComputeGrad(ratdata["rawdata"], ntrials, args_12p, x_12p)
print(LLgrad)

###### test4 : Compute Hessian Matrix of 40 trials
LL, LLgrad, LLhess = ComputeHess(ratdata["rawdata"], ntrials, args_12p, x_12p)
print(LLhess)


###### test5 : Model Optimization
init_params = InitParams(args_12p)
result = ModelFitting(args_12p, init_params, ratdata, ntrials)
FitSummary(mpath, fname, result)


