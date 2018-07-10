# for using keyword_vgh()
# change the order of parameters.

# function ComputeLL(LLs::SharedArray{Float64,1}, ratdata, ntrials::Int#, args, x::Vector{T})
function ComputeLL(LLs, ratdata, ntrials::Int#, args, x::Vector{T})
    ;kwargs...)
    LL = 0.

    @sync @parallel for i in 1:ntrials
        RightClickTimes, LeftClickTimes, maxT, rat_choice = TrialData(ratdata, i)
        Nsteps = Int(ceil(maxT/dt))

        # LLs[i] = LogLikelihood(params, RightClickTimes, LeftClickTimes, Nsteps, rat_choice)
        LLs[i] = LogLikelihood(RightClickTimes, LeftClickTimes, Nsteps, rat_choice
                ;kwargs...)#make_dict(args, x)...)
    end

    LL = -sum(LLs)
    return LL 
end

function ComputeLL_bbox(LLs, ratdata, ntrials::Int#, args, x::Vector{T})
    ;kwargs...)
    LL  = 0.
    (k,v) = kwargs[1]
    LLs = SharedArray(typeof(v), ntrials)#zeros(eltype(params),ntrials)

    @sync @parallel for i in 1:ntrials
        RightClickTimes, LeftClickTimes, maxT, rat_choice = TrialData(ratdata, i)
        Nsteps = Int(ceil(maxT/dt))
        LLs[i] = LogLikelihood(RightClickTimes, LeftClickTimes, Nsteps, rat_choice
            ;kwargs...)#;make_dict(args, x)...)
    end
    LL = -sum(LLs)
    return LL
end

function ComputeGrad{T}(ratdata, ntrials::Int, args, x::Vector{T})
    LL        = 0.
    LLgrad    = zeros(T,length(x))
    
    # do we still need wrapper?
    function WrapperLL(;kwargs...)#(params::Vector{T})
        LL  = 0.
        (k,v) = kwargs[1]
        LLs = SharedArray(typeof(v), ntrials)#zeros(eltype(params),ntrials)

        @sync @parallel for i in 1:ntrials
            RightClickTimes, LeftClickTimes, maxT, rat_choice = TrialData(ratdata, i)
            Nsteps = Int(ceil(maxT/dt))
            LLs[i] = LogLikelihood(RightClickTimes, LeftClickTimes, Nsteps, rat_choice
                ;kwargs...)#;make_dict(args, x)...)
        end
        LL = -sum(LLs)
        return LL
    end

    do_hess = false

    # LL = WrapperLL(;make_dict(args, x)...)
    # println(LL)
    LL, LLgrad = GeneralUtils.keyword_vgh((;params...) 
        -> WrapperLL(;params...), args, x, do_hess)

    # LL, LLgrad = GeneralUtils.keyword_vgh(WrapperLL, args, x ,do_hess)

    # LL, LLgrad = GeneralUtils.vgh(WrapperLL, params, do_hess)

    # result =  DiffBase.GradientResult(params)    
    # ForwardDiff.gradient!(result, WrapperLL, params);
    # LL     = DiffBase.value(result)
    # LLgrad = DiffBase.gradient(result)
    return LL, LLgrad
end

function ComputeHess{T}(ratdata, ntrials::Int, args, x::Vector{T})
    LL        = 0.
    LLgrad    = zeros(T,length(x))
    LLhess    = zeros(T,length(x),length(x))
    
    function WrapperLL(;kwargs...)#(params::Vector{T})
        LL  = 0.
        (k,v) = kwargs[1]
        LLs = SharedArray(typeof(v), ntrials)#zeros(eltype(params),ntrials)

        @sync @parallel for i in 1:ntrials
            RightClickTimes, LeftClickTimes, maxT, rat_choice = TrialData(ratdata, i)
            Nsteps = Int(ceil(maxT/dt))
            LLs[i] = LogLikelihood(RightClickTimes, LeftClickTimes, Nsteps, rat_choice
                ;kwargs...)#;make_dict(args, x)...)

        end
        LL = -sum(LLs)
        return LL
    end

    do_hess = true

    # LL = WrapperLL(;make_dict(args, x)...)
    # println(LL)
    LL, LLgrad, LLhess = GeneralUtils.keyword_vgh((;params...) 
        -> WrapperLL(;params...), args, x, do_hess)

    # LL, LLgrad, LLhess = GeneralUtils.vgh(WrapperLL, params)

    # result =  DiffBase.HessianResult(params)    
    # ForwardDiff.hessian!(result, WrapperLL, params);
    # LL     = DiffBase.value(result)
    # LLgrad = DiffBase.gradient(result)
    # LLhess = DiffBase.hessian(result)
    return LL, LLgrad, LLhess
end


function TrialsLikelihood{T}(LL::AbstractArray{T,1},params::Vector, ratdata, ntrials::Int)     
    for i in 1:ntrials
        RightClickTimes, LeftClickTimes, maxT, rat_choice = TrialData(ratdata, i)
        Nsteps = Int(ceil(maxT/dt))

        LL[i] = LogLikelihood(params, RightClickTimes, LeftClickTimes, Nsteps, rat_choice)
    end
end
