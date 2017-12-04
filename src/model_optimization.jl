function InitParams(seed_mode=1)

    # Parameters (match the parameter order with original code)

    # random seed    
    if seed_mode == 1
        lam = randn()*.01; 
        sigma_a = rand()*50.; sigma_s = rand()*50.; sigma_i = rand()*30.; 
        B = rand()*12.+5.; 
        phi = rand(); tau_phi = 0.695*rand()+0.005; 
        bias = randn(); lapse = rand()*.5;
    
        params = [lam, sigma_a, sigma_s, sigma_i, B, phi, tau_phi, bias, lapse] 

    # bing's rat avg parameter set
    elseif seed_mode == 2
        params = [-0.2767, 2.0767, 75.9600, 1.9916, 8.9474, 0.1694, 0.0964, -0.0269, 0.0613]

    # simple fixed parameter set    
    elseif seed_mode == 3
        lam = -0.0005; sigma_a = 1.; sigma_s = 0.1; sigma_i = 0.2; 
        B = 6.1; phi = 0.3; tau_phi = 0.1; bias = 0.1; lapse = 0.05*2;

        params = [lam, sigma_a, sigma_s, sigma_i, B, phi, tau_phi, bias, lapse] 
    end

    return params   
end

function ModelFitting(params, ratdata, ntrials)
    # Parameter Constraints 
    l = [-5., 0.,   0.,   0.,  2.,  0.01, 0.005, -5., 0.]
    u = [+5., 200., 200., 40., 32.,  2.5,    1.,  5., 1.]

    function LL_f(params::Vector)
        LLs = SharedArray(Float64, ntrials)
        return ComputeLL(LLs, params, ratdata["rawdata"], ntrials)
    end

    # updated for julia v0.6 (in-place order)
    function LL_g!{T}(grads::Vector{T}, params::Vector{T})
        LL, LLgrad = ComputeGrad(params, ratdata["rawdata"], ntrials)
        for i=1:length(params)
            grads[i] = LLgrad[i]
        end
    end

    function LL_fg!(params::Vector, grads)
        LL, LLgrad = ComputeGrad(params, ratdata["rawdata"], ntrials)
        for i=1:length(params)
            grads[i] = LLgrad[i]
        end
        return LL
    end
    
    function my_line_search!(df, x, s, x_scratch, gr_scratch, lsr, alpha,
        mayterminate, c1::Real = 1e-4, rhohi::Real = 0.5, rholo::Real = 0.1, iterations::Integer = 1_000)
        initial_alpha = 0.5
        LineSearches.bt2!(df, x, s,x_scratch, gr_scratch, lsr, initial_alpha,
                      mayterminate, c1, rhohi, rholo, iterations)
    end

    d4 = OnceDifferentiable(LL_f,LL_g!,params)
                                # LL_fg!)

    tic()
    # history = optimize(d4, params, l, u, Fminbox(); 
    #          optimizer = GradientDescent, iterations = 500, linesearch = my_line_search!, optimizer_o = Optim.Options(g_tol = 1e-12,
    #                                                                         x_tol = 1e-32,
    #                                                                         f_tol = 1e-16,
    #                                                                         iterations = 20,
    #                                                                         store_trace = true,
    #                                                                         show_trace = true,
    #                                                                         extended_trace = true
    #                                                                         ))
    history = optimize(d4, params, l, u, Fminbox(); 
             optimizer = LBFGS, optimizer_o = Optim.Options(g_tol = 1e-12,
                                                                            x_tol = 1e-10,
                                                                            f_tol = 1e-6,                                                                        iterations = 10,
                                                                            store_trace = true,
                                                                            show_trace = true,
                                                                            extended_trace = true))




    fit_time = toc()
    println(history.minimizer)
    println(history)

    ## do a single functional evaluation at best fit parameters and save likely for each trial
    # likely_all = zeros(typeof(sigma_i),ntrials)
    x_bf = history.minimizer #.minimum
    # Likely_all_trials(likely_all, x_bf, ratdata["rawdata"], ntrials)
    LL, LLgrad, LLhess = ComputeHess(x_bf, ratdata["rawdata"], ntrials)
    
    Gs = zeros(length(history.trace),length(params))
    Xs = zeros(length(history.trace),length(params))
    fs = zeros(length(history.trace))

    for i=1:length(history.trace)
        tt = getfield(history.trace[i],:metadata)
        fs[i] = getfield(history.trace[i],:value)
        Gs[i,:] = tt["g(x)"]
        Xs[i,:] = tt["x"]
    end

    D = Dict([("x_init",params),
                ("trials",ntrials),
                ("f",history.minimum), 
                ("x_converged",history.x_converged),
                ("f_converged",history.f_converged),
                ("g_converged",history.g_converged),                            
                ("grad_trace",Gs),
                ("f_trace",fs),
                ("x_trace",Xs),                         
                ("fit_time",fit_time),
                ("x_bf",history.minimizer),
                ("myfval", history.minimum),
                ("hessian", LLhess)
                ])

    # saveto_filename = *("julia_out_",ratname,"_rseed.mat")
    # WriteFile(mpath, filename, D)
    # matwrite(saveto_filename, Dict([("x_init",params),
    #                                 ("trials",ntrials),
    #                                 ("f",history.minimum), 
    #                                 ("x_converged",history.x_converged),
    #                                 ("f_converged",history.f_converged),
    #                                 ("g_converged",history.g_converged),                            ("grad_trace",Gs),
    #                                 ("f_trace",fs),
    #                                 ("x_trace",Xs),                         
    #                                 ("fit_time",fit_time),
    #                                 ("x_bf",history.minimizer),
    #                                 ("myfval", history.minimum),
    #                                 ("hessian", LLhess)
    #                                 ]))
    return D
end

function FitSummary(mpath, filename, D)
    WriteFile(mpath, filename, D)
end

## matwrite -> data_handle?