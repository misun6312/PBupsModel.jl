"""
A package for fitting data from auditory evidence accumulation task (Poisson clicks task) 
to evidence accumulation model.
"""
module PBupsModel

export TrialData, LogLikelihood

include("data_handle.jl")
include("model_likelihood.jl")

end # module
