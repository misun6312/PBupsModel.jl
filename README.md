# PBupsModel

The `PBupsModel` package currently provides the Likelihood, Gradients, Hessian matrix of Evidence accumulation model for Poisson Clicks Task using Automatic Differentiation:

- `TrialData`: reads Data file in format of Poisson Clicks task. It includes the right and left click times on trial and the subject's decision on trial, and duration of trial.
- `LogLikelihood`: computes the logliklihood of each trial using Bing's model (Brunton et al. 2013 Science) (http://brodylab.org/publications-2/brunton-et-al-2013).

## Installation

As described in the manual, to [install unregistered packages][unregistered], use `Pkg.clone()` with the repository url:

```julia
Pkg.clone("https://github.com/misun6312/PBupsModel.git")
```

Julia version 0.4 or higher is required (install instructions [here][version]).

## Usage

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

# sigma_a = rand()*4.; sigma_s = rand()*4.; sigma_i = rand()*30.;
# lam = randn()*0.4; B = rand()*12.+5.; bias = randn();
# phi = rand()*1.39+0.01; tau_phi = 0.695*rand()+0.005; lapse = rand();

# known parameter set
sigma_a = 1; sigma_s = 0.1; sigma_i = 0.2; 
lam = -0.5; B = 6.1; bias = 0.1; 
phi = 0.3; tau_phi = 0.1; lapse = 0.05*2;
params = [sigma_a, sigma_s, sigma_i, lam, B, bias, phi, tau_phi, lapse]

# Compute Loglikelihood value
LL = LogLikelihood(params, RightClickTimes, LeftClickTimes, Nsteps, rat_choice)

```

## Testing

In a Julia session, run `Pkg.test("PBupsModel")`.


[unregistered]:http://docs.julialang.org/en/release-0.4/manual/packages/#installing-unregistered-packages
[version]:http://julialang.org/downloads/platform.html
