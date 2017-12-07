# PBupsModel

The `PBupsModel` package currently provides the Likelihood, Gradients, Hessian matrix of Evidence accumulation model for Poisson Clicks Task using Automatic Differentiation ([Brunton et al. 2013 Science][Bing]).

## Installation

As described in the manual, to [install unregistered packages][unregistered], use `Pkg.clone()` with the repository url:

```julia
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
params is a vector whose elements, in order, are
    sigma_a    square root of accumulator variance per unit time sqrt(click units^2 per second)
    sigma_s    standard deviation introduced with each click (will get scaled by click adaptation)
    sigma_i    square root of initial accumulator variance sqrt(click units^2)
    lambda     1/accumulator time constant (sec^-1). Positive means unstable, neg means stable
    B          sticky bound height (click units)
    bias       where the decision boundary lies (click units)
    phi        click adaptation/facilitation multiplication parameter
    tau_phi    time constant for recovery from click adaptation (sec)
    lapse      2*lapse fraction of trials are decided randomly
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

# known parameter set
sigma_a = 1; sigma_s = 0.1; sigma_i = 0.2; 
lam = -0.5; B = 6.1; bias = 0.1; 
phi = 0.3; tau_phi = 0.1; lapse = 0.05*2;
params = [sigma_a, sigma_s, sigma_i, lam, B, bias, phi, tau_phi, lapse]

# Compute Loglikelihood value
LL = LogLikelihood(params, RightClickTimes, LeftClickTimes, Nsteps, rat_choice)

# Compute Loglikelihood value of many trials
ntrials = 1000
LLs = SharedArray(Float64, ntrials)
LL_total = ComputeLL(LLs, params, ratdata["rawdata"], ntrials)

# Compute Gradients 
LL, LLgrad = ComputeGrad(params, ratdata["rawdata"], ntrials)

# Compute Hessian Matrix 
LL, LLgrad, LLhess = ComputeHess(params, ratdata["rawdata"], ntrials)


```

## Testing

In a Julia session, run `Pkg.test("PBupsModel")`.


[unregistered]:http://docs.julialang.org/en/release-0.4/manual/packages/#installing-unregistered-packages
[version]:http://julialang.org/downloads/platform.html
[Bing]:http://brodylab.org/publications-2/brunton-et-al-2013
