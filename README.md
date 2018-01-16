# PBupsModel

The `PBupsModel` package currently provides the Likelihood, Gradients, Hessian matrix of Evidence accumulation model for Poisson Clicks Task using Automatic Differentiation ([Brunton et al. 2013 Science][Bing]).

## Installation

As described in the manual, to [install unregistered packages][unregistered], use `Pkg.clone()` with the repository url:

```julia
Pkg.clone("https://github.com/misun6312/GeneralUtils.jl.git")
Pkg.clone("https://github.com/misun6312/PBupsModel.jl.git")
```

Julia version 0.6 is required (install instructions [here][version]).

## Usage

- `TrialData`: reads Data file in format of Poisson Clicks task. It includes the right and 
left click times on trial and the subject's decision on trial, and duration of trial.

We accept data file in MATLAB `.mat` format (Both in the older v5/v6/v7 format, as well as the newer v7.3 format. This is supported by [MAT.jl](https://github.com/simonster/MAT.jl) package.)
```

The data should have following 4 fields for each trial
	rightbups	Click times on the right 	e.g. [0, 0.0639, 0.0881, 0.0920, 0.1277, 0.1478, 0.1786, 0.1894, 0.2180, 0.2464]
	leftbups	Click times on the left		e.g. [0, 0.3225]
	pokedR		Whether the subject responded left or right (0 -> left, 1 -> right) e.g. 1 
	T  		Total duration of the stimulus	e.g. 0.3619
```

- `LogLikelihood`: computes the log likelihood according to Bing's model, and returns log likelihood for single trial.

```
Model parameters can be accessed by following keywords : 
    sigma_a     a diffusion constant, parameterizing noise in a. square root of accumulator variance per unit time sqrt(click units^2 per second)
    sigma_s_R   parameterizing noise when adding evidence from a right pulse. (incoming sensory evidence). standard deviation introduced with each click (will get scaled by click adaptation)
    sigma_s_L   parameterizing noise when adding evidence from a left pulse. (incoming sensory evidence). standard deviation introduced with each click (will get scaled by click adaptation)
    sigma_i     square root of initial accumulator variance sqrt (click units^2). initial condition for the dynamical equation at t=0.
    lambda      1/accumulator time constant (sec^-1). Positive means unstable, neg means stable
    B           sticky bound height (click units)
    bias        where the decision boundary lies (click units)
    phi         click adaptation/facilitation multiplication parameter
    tau_phi     time constant for recovery from click adaptation (sec)
    lapse_R     The lapse rate parameterizes the probability of making a random response as right choice. (L->R)
    lapse_L     The lapse rate parameterizes the probability of making a random response as left choice. (R->L)
    biased_input unbalanced input gain (sensory neglect)
```

- `ComputeLL`: computes the log likelihood for many trials and returns the sum of log likelihood.
- `ComputeGrad`: returns the gradients.
- `ComputeHess`: returns hessian matrix at given parameter point.

## Example

The simple example below computes a log likelihood of model with example trial.

```julia
using PBupsModel
using MAT

dt = 0.02

# read test data file
ratdata = matread("data/testdata.mat")

# using TrialData load trial data
RightClickTimes, LeftClickTimes, maxT, rat_choice = TrialData(ratdata["rawdata"], 1)

Nsteps = Int(ceil(maxT/dt))

# known parameter set (9-parameter)
args = ["sigma_a","sigma_s_R","sigma_i","lambda","B","bias","phi","tau_phi","lapse_R"]
x = [1., 0.1, 0.2, -0.5, 6.1, 0.1, 0.3, 0.1, 0.05*2]

# Compute Loglikelihood value
LL = LogLikelihood(RightClickTimes, LeftClickTimes, Nsteps, rat_choice
                ;make_dict(args, x)...)

# Compute Loglikelihood value of many trials
ntrials = 1000
LLs = SharedArray(Float64, ntrials)
LL_total = ComputeLL(LLs, ratdata["rawdata"], ntrials, args, x)

# Compute Gradients 
LL, LLgrad = ComputeGrad(ratdata["rawdata"], ntrials, args, x)

# Compute Gradients 
LL, LLgrad, LLhess = ComputeHess(ratdata["rawdata"], ntrials, args, x)

# Model Optimization
args = ["sigma_a","sigma_s_R","sigma_i","lambda","B","bias","phi","tau_phi","lapse_R"]
init_params = InitParams(args)
result = ModelFitting(args, init_params, ratdata, ntrials)
FitSummary(mpath, fname, result)

# known parameter set (12-parameter including bias parameters)
args_12p = ["sigma_a","sigma_s_R","sigma_s_L","sigma_i","lambda","B","bias","phi","tau_phi","lapse_R","lapse_L","input_gain_weight"]
x_12p = [1., 0.1, 50, 0.2, -0.5, 6.1, 0.1, 0.3, 0.1, 0.05*2, 0.2, 0.4]

# Compute Loglikelihood value of many trials
ntrials = 400
LLs = SharedArray(Float64, ntrials)
LL = ComputeLL(LLs, ratdata["rawdata"], ntrials, args_12p, x_12p)
print(LL)

# Compute Gradients 
LL, LLgrad = ComputeGrad(ratdata["rawdata"], ntrials, args_12p, x_12p)
print(LLgrad)

# Compute Gradients 
LL, LLgrad, LLhess = ComputeHess(ratdata["rawdata"], ntrials, args_12p, x_12p)
print(LLhess)


```

## Testing

In a Julia session, run `Pkg.test("PBupsModel")`.


[unregistered]:http://docs.julialang.org/en/release-0.4/manual/packages/#installing-unregistered-packages
[version]:http://julialang.org/downloads/platform.html
[Bing]:http://brodylab.org/publications-2/brunton-et-al-2013
