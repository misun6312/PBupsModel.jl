"""
A package for fitting data from auditory evidence accumulation task (Poisson clicks task) 
to evidence accumulation model.
"""
__precompile__()

module PBupsModel

# 3rd party
import Base.convert

using MAT
using ForwardDiff
import ForwardDiff.DiffBase
# using DiffBase

export 
    
# data_handle
    TrialData,

# model_likelihood 
    LogLikelihood, 

# all_trials
    ComputeLL,
    ComputeGrad,
    ComputeHess,
    Likely_all_trials

# optimization


include("data_handle.jl")
include("model_likelihood.jl")
include("all_trials.jl")

end # module
