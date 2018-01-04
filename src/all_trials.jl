# for using keyword_vgh()
# change the order of parameters.

function ComputeLL(LLs::SharedArray{Float64,1}, ratdata, ntrials::Int, args, x)

    LL = 0.

    @sync @parallel for i in 1:ntrials
        RightClickTimes, LeftClickTimes, maxT, rat_choice = TrialData(ratdata, i)
        Nsteps = Int(ceil(maxT/dt))

        # LLs[i] = LogLikelihood(params, RightClickTimes, LeftClickTimes, Nsteps, rat_choice)
        LogLikelihood(RightClickTimes, LeftClickTimes, Nsteps, rat_choice;
            make_dict(args, x)...)
    end

    LL = -sum(LLs)
    return LL 
end

function ComputeGrad{T}(params::Vector{T}, ratdata, ntrials::Int)
    LL        = 0.
    LLgrad    = zeros(T,length(params))
    
    function WrapperLL{T}(params::Vector{T})
        LL  = 0.
        LLs = SharedArray(eltype(params), ntrials)#zeros(eltype(params),ntrials)

        @sync @parallel for i in 1:ntrials
            RightClickTimes, LeftClickTimes, maxT, rat_choice = TrialData(ratdata, i)
            Nsteps = Int(ceil(maxT/dt))
            LLs[i] = LogLikelihood(params, RightClickTimes, LeftClickTimes, Nsteps, rat_choice)
        end
        LL = -sum(LLs)
        return LL
    end

    do_hess = false
    LL, LLgrad = GeneralUtils.vgh(WrapperLL, params, do_hess)

    LL, LLgrad = GeneralUtils.keyword_vgh((;params...) -> WrapperLL(RightClickTimes, LeftClickTimes, Nsteps;params...)
    ,args, x ,do_hess)



    # result =  DiffBase.GradientResult(params)
    
    # ForwardDiff.gradient!(result, WrapperLL, params);
    
    # LL     = DiffBase.value(result)
    # LLgrad = DiffBase.gradient(result)
    return LL, LLgrad
end

function ComputeHess{T}(params::Vector{T}, ratdata, ntrials::Int)
    LL        = 0.
    LLgrad    = zeros(T,length(params))
    LLhess    = zeros(T,length(params),length(params))
    
    function WrapperLL{T}(params::Vector{T})
        LL  = 0.
        LLs = SharedArray(eltype(params), ntrials)#zeros(eltype(params),ntrials)

        @sync @parallel for i in 1:ntrials
            RightClickTimes, LeftClickTimes, maxT, rat_choice = TrialData(ratdata, i)
            Nsteps = Int(ceil(maxT/dt))
            LLs[i] = LogLikelihood(params, RightClickTimes, LeftClickTimes, Nsteps, rat_choice)
        end
        LL = -sum(LLs)
        return LL
    end

    LL, LLgrad, LLhess = GeneralUtils.vgh(WrapperLL, params)

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
