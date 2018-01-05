# updating for generalizing the model. 
# user will give args to init parameters 
# use 'dictionary' -> each parameter has different range

# utilize make_dict and defaults.

function InitParams(args, seed_mode=1)

    # Parameters (match the parameter order with original code)
    # --> order doesn't matter now, it will be matched with the order of parameters

    x_val = zeros(length(args))

    # random seed    
    if seed_mode == 1
        rseed_param = Dict("sigma_a" => rand()*50., 
        "sigma_s_R" => rand()*50.,
        "sigma_s_L" => rand()*50.,
        "sigma_i" => rand()*30.,
        "lambda" => randn()*.01,
        "B" => rand()*12.+5.,
        "bias" => randn(),
        "phi" => rand(),
        "tau_phi" => 0.695*rand()+0.005,
        "lapse_R" => rand()*.5,
        "lapse_L" => rand()*.5,
        "input_gain_weight" => rand())

        for i in 1:length(args)
            x_val[i] = rseed_param[args[i]]
        end
    # bing's rat avg parameter set
    elseif seed_mode == 2
        params = [-0.2767, 2.0767, 75.9600, 1.9916, 8.9474, 0.1694, 0.0964, -0.0269, 0.0613]
        defaults_param = Dict("sigma_a" => 2.0767, 
        "sigma_s_R" => 75.9600,
        "sigma_s_L" => 75.9600,
        "sigma_i" => 1.9916,
        "lambda" => -0.2767,
        "B" => 8.9474,
        "bias" => -0.0269,
        "phi" => 0.1694,
        "tau_phi" => 0.0964,
        "lapse_R" => 0.0613,
        "lapse_L" => 0.0613,
        "input_gain_weight" => 0.5)

        for i in 1:length(args)
            x_val[i] = defaults_param[args[i]]
        end

    # simple fixed parameter set    
    elseif seed_mode == 3
        simple_param = Dict("sigma_a" => 1., 
        "sigma_s_R" => 0.1,
        "sigma_s_L" => 0.1,
        "sigma_i" => 0.2,
        "lambda" => -0.0005,
        "B" => 6.1,
        "bias" => 0.1,
        "phi" => 0.3,
        "tau_phi" => 0.1,
        "lapse_R" => 0.1,
        "lapse_L" => 0.1,
        "input_gain_weight" => 0.5)

        for i in 1:length(args)
            x_val[i] = simple_param[args[i]]
        end
    end

    return x_val   
end

function ModelFitting(args, x_init, ratdata, ntrials)
    l_b = Dict("sigma_a" => 0., 
        "sigma_s_R" => 0.,
        "sigma_s_L" => 0.,
        "sigma_i" => 0.,
        "lambda" => -5.,
        "B" => 5.,
        "bias" => -5.,
        "phi" => 0.01,
        "tau_phi" => 0.005,
        "lapse_R" => 0.,
        "lapse_L" => 0.,
        "input_gain_weight" => 0.)

    u_b = Dict("sigma_a" => 200., 
        "sigma_s_R" => 200.,
        "sigma_s_L" => 200.,
        "sigma_i" => 40.,
        "lambda" => 5.,
        "B" => 25.,
        "bias" => 5.,
        "phi" => 1.2,
        "tau_phi" => 1.5,
        "lapse_R" => 1.,
        "lapse_L" => 1.,
        "input_gain_weight" => 1.)


    l = zeros(length(args))
    u = zeros(length(args))

    for i in 1:length(args)
        l[i] = l_b[args[i]]
        u[i] = u_b[args[i]] 
    end


    function LL_f(x_init::Vector)
        LLs = SharedArray(Float64, ntrials)
        return ComputeLL(LLs, ratdata["rawdata"], ntrials, args, x_init)
    end

    # updated for julia v0.6 (in-place order)
    function LL_g!{T}(grads::Vector{T}, x_init::Vector{T})
        LL, LLgrad = ComputeGrad(ratdata["rawdata"], ntrials, args, x_init)

        for i=1:length(x_init)
            grads[i] = LLgrad[i]
        end
    end

    function LL_fg!(x_init::Vector, grads)
        LL, LLgrad = ComputeGrad(ratdata["rawdata"], ntrials, args, x_init)

        for i=1:length(x_init)
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

    d4 = OnceDifferentiable(LL_f,LL_g!,x_init)
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
    history = optimize(d4, x_init, l, u, Fminbox(); 
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
    # LL, LLgrad, LLhess = ComputeHess(x_bf, ratdata["rawdata"], ntrials)
    LL, LLgrad, LLhess = ComputeHess(ratdata["rawdata"], ntrials, args, x_bf)

    Gs = zeros(length(history.trace),length(x_init))
    Xs = zeros(length(history.trace),length(x_init))
    fs = zeros(length(history.trace))

    for i=1:length(history.trace)
        tt = getfield(history.trace[i],:metadata)
        fs[i] = getfield(history.trace[i],:value)
        Gs[i,:] = tt["g(x)"]
        Xs[i,:] = tt["x"]
    end

    D = Dict([("x_init",x_init),    
                ("parameters",args),
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