using GeneralUtils
using MAT
using JLD

include("constrained_parabolic_minimization.jl")

"""
pdict = wallwrap(bdict, pdict)
Given bdict, a dictionary of symbols to [minval, maxval] vectors, and pdict, a dictionary of symbols
to values (or, alternatively, an Array of (Symbol, value) tuples], goes through each of the symbols in 
bdict and modifies the corresponding value in pdict putting it through a tanh so the final output lies 
within the limits in bdict.  Returns the new pdict.  Makes a copy of pdict so as not to modify the original.
"""
function wallwrap(bdict, epdict)
    local pdict = two_level_copy(epdict)  # Must be very careful here! I got bit by the bug of forgetting that without
    # an explicit copy() call, Julia does not make copies of the contents of arrays or dictionaries, making it
    # easy to inadvertently modify something one did not intend to perturb.  Note the two_level_copy() call, 
    # necessary to make sure we don't mess up the content of the caller's dictionary.
    
    if typeof(pdict)<:Array
        pdict = Dict(pdict)
    end

    allkeys = keys(bdict)

    for k in allkeys
        local bbox = bdict[k]
        d = 0.5*(bbox[2] - bbox[1])
        m = 0.5*(bbox[2] + bbox[1])

        pdict[k] = bbox[1] + d*(tanh((pdict[k]-m)/d)+1)
    end
    return pdict
end

    
"""
params = vector_wrap(bbox, args, eparams)
Given bdict, a dictionary of symbols to [minval, maxval] vectors, args, an array of strings representing
symbols, and params, an array of values corresponding to the args list, puts each param that has an entry 
in bdict through the tanh-walling mechanism, and returns the result. Does not modify the contents of the 
original params vector (or bdict or args).
"""
function vector_wrap(bbox, args, eparams)
    local params = two_level_copy(eparams)
    pdict = wallwrap(bbox, make_dict(args, params))
    i=1; j=1
    for i=1:length(args)
        if typeof(args[i])<:Array
            params[j:j+args[i][2]-1] = pdict[Symbol(args[i][1])]
            j += args[i][2]-1
        else
            params[j] = pdict[Symbol(args[i])]
        end
    j = j+1
    end
    return params
end


"""
params = inverse_wall(bdict, args, wparams)
Given bdict, a dictionary of symbols to [minval, maxval] vectors, args, an array of strings representing
symbols, and wparams, an array of values corresponding to the args list where each param that has an entry 
in bdict has alreadt been through the tanh-walling mechanism, UNwalls the ones that have a bdict entry and
returns the result. Does not modify the contents of the original params vector (or bdict or args).
"""
function inverse_wall(bdict, args, wparams)
    local params = two_level_copy(wparams)
    pdict = inverse_wall(bdict, make_dict(args, params))
    i=1; j=1
    for i=1:length(args)
        if typeof(args[i])<:Array
            params[j:j+args[i][2]-1] = pdict[Symbol(args[i][1])]
            j += args[i][2]-1
        else
            params[j] = pdict[Symbol(args[i])]
        end
        j = j+1
    end
    return params    
end

    
"""
pdict = inverse_wall(bdict, wdict)
Given bdict, a dictionary of symbols to [minval, maxval] vectors, and wdict, a dictionary of symbols to values
(or vectors of values)  UNwalls the ones that have a bdict entry and
returns the result. Does not modify the contents of any dictionaries.
"""
function inverse_wall(bdict, wdict)
    local pdict = two_level_copy(wdict)

    allkeys = keys(bdict)
    for k in allkeys
        local bbox = bdict[k]
        d = 0.5*(bbox[2] - bbox[1])
        m = 0.5*(bbox[2] + bbox[1])

        pdict[k] = m + d*0.5*log((pdict[k]-bbox[1])./(2*d - pdict[k] + bbox[1]))
    end
    return(pdict)
end

######################################################
#                                                    #
#         BBOX_HESSIAN_KEYWORD_MINIMIZATION          #
#                                                    #
######################################################



function bbox_Hessian_keyword_minimization(seed, args, bbox, func; start_eta=0.1, tol=1e-6, maxiter=400,
    verbose=false, verbose_level=1, verbose_every=1, softbox=true, hardbox=false, report_file="")

    # --- check that saving will be done to a .jld file ---
    if length(report_file)>0 && splitext(report_file)[2] != ".jld"
        if splitext(report_file)[2] == ""
            report_file = resport_file * ".jld"
        else
            error("Sorry, report_file can only write to JLD files, the extension has to be .jld")
        end
    end

    
    # --------- Initializing the trajectory trace and function wrapper--------
 
    traj_increment = 100
    params = 0  # Make sure to have this here so that params stays defined beyond the try/catch
    if ( !(typeof(bbox)<:Dict) ); error("Currently only supporting softbox=true, bbox must be a Dict"); end;
    try
        params = copy(inverse_wall(bbox, args, seed))
    catch
        error("Were all initial param values within the indicated walls?")
    end
    eta = start_eta
    trajectory = zeros(2+length(params), traj_increment); cpm_traj = zeros(2, traj_increment)
    
    ftraj = Array{Any}(3,0)  # will hold gradient, hessian, and further_out,  per iteration

    further_out =[];  # We define this variable here so it will be available for stashing further outputs from func
    
    # Now we define a wrapper around func() to do three things: (a) wallwrap parameters using the softbox method;
    # (b) return as the desired scalar the first output of func; (c) stash in further_out any further outputs of func
    internal_func = (;pars...) -> begin
        fresults = func(;wallwrap(bbox, pars)...)   # note use of bbox external to this begin...end
        if typeof(fresults)<:Tuple
            answer = fresults[1]
            further_out = fresults[2:end]
        else
            answer = fresults
        end
        return answer  # we assume that the first output of func() will always be a scalar, and that's what we return for ForwardDiff
    end

    # --------- END Initializing the trajectory trace --------
    println("-------- end of Initialization --------")

    if verbose
        @printf "%d: eta=%g ps=" 0 eta 
        print_vector(vector_wrap(bbox, args, params))
        @printf "\n"
    end
    
    if softbox
        if !(typeof(bbox)<:Dict); error("bhm: If softbox=true, then bbox must eb a Dict"); end
        cost, grad, hess = keyword_vgh(internal_func, args, params)  # further_out will be mutated
    elseif hardbox
        error("Sorry, no longer supporting hardbox=true")
    else
        error("Sorry, no longer supporting softbox=false")
    end
        
    chessdelta = zeros(size(params))
    
    i=0  # here so variable i is available outside the loop
    for i in [1:maxiter;]
        if i > size(trajectory, 2)
            trajectory = [trajectory zeros(2+length(params), traj_increment)]
            cpm_traj   = [cpm_traj   zeros(2, traj_increment)]
        end
        trajectory[1:2, i]   = [eta;cost]
        trajectory[3:end, i] = vector_wrap(bbox, args, params)
        ftraj = [ftraj [grad, hess, further_out]]

        if length(report_file)>0
            save(report_file, Dict("traj"=>trajectory[:,1:i], "cpm_traj"=>cpm_traj[:,1:i], "ftraj"=>ftraj))
        end
        
        # println(hess);
        hessdelta  = - pinv(hess)*grad
        try
            if verbose && verbose_level >= 2
                @printf("bhm: about to try cpm with grad : "); print_vector_g(grad); print("\n")
                @printf("bhm:   hess :"); print_vector_g(hess[:]); print("\n");
            end
            if verbose && verbose_level >= 2
                cpm_out = constrained_parabolic_minimization(hess, grad'', eta, 
                    maxiter=500, tol=1e-20, do_plot=true, verbose=true)                
            else
                cpm_out = constrained_parabolic_minimization(hess, grad'', eta, maxiter=500, tol=1e-20)
            end
            chessdelta = cpm_out[1]; cpm_traj[1,i] = cpm_out[5]; cpm_traj[2,i] = cpm_out[6]
            jumptype = "not failed"
        catch y
            jumptype = "failed"
            if verbose
                @printf "Constrained parabolic minimization failed with error %s\n" y
                @printf "\n"
                @printf "eta was %g\n" eta
                @printf "grad was\n"
                print_vector(grad)
                @printf "\n\nhess was\n"
                for k in [1:length(grad);]
                    print_vector(hess[k,:])
                    @printf "\n"
                end
                @printf "\n"
                matwrite("error_report.mat", Dict("grad"=>grad, "hess"=>hess, "eta"=>eta))
            end
            break
        end

        if norm(hessdelta) <= eta
            new_params = params + hessdelta
            jumptype = "Newton"
        elseif jumptype != "failed" 
            new_params = params + chessdelta
            jumptype  = "constrained"
        end

        if jumptype != "failed"
            new_cost, new_grad, new_hess = keyword_vgh(internal_func, args, new_params)   # further_out may mutate
            if verbose && verbose_level >=2
                @printf("bhm: had new_params = : "); print_vector_g(vector_wrap(bbox, args, params)); print("\n");
                @printf("bhm: and my bbox was : "); print(bbox); print("\n")
                @printf("bhm: and my wallwrap output was : "); print(wallwrap(bbox, make_dict(args, new_params))); print("\n")
                @printf("bhm: and this produced new_grad : "); print_vector_g(new_grad); print("\n")
                @printf("bhm:   new_hess :"); print_vector_g(new_hess[:]); print("\n");                                        
            end
            
            if abs(new_cost - cost) < tol || eta < tol
                if verbose
                    @printf("About to break -- tol=%g, new_cost-cost=%g, eta=%g\n", tol, new_cost-cost, eta)
                end
                break
            end
        end

        if jumptype == "failed" || new_cost >= cost  
            if verbose
                @printf("eta going down: new_cost-cost=%g and jumptype='%s'\n", new_cost-cost, jumptype)
                if verbose_level >= 2
                    nwp = vector_wrap(bbox, args, new_params); wp = vector_wrap(bbox, args, params)
                    @printf("   vvv: proposed new params were : "); print_vector_g(nwp); print("\n")
                    @printf("   vvv: proposed delta params was : "); print_vector_g(nwp-wp); print("\n")
                    @printf("   vvv: grad was : "); print_vector_g(grad); print("\n")
                    costheta = dot(new_params-params, grad)/(norm(new_params-params)*norm(grad))
                    @printf("   vvv: costheta of proposed jump was %g\n", costheta)
                end
            end
            eta = eta/2
            costheta = NaN
            if eta < tol
                if verbose
                    @printf("About to break -- tol=%g, new_cost-cost=%g, eta=%g\n", tol, new_cost-cost, eta)
                end
                break
            end
        else
            eta = eta*1.1
            costheta = dot(new_params-params, grad)/(norm(new_params-params)*norm(grad))

            params = new_params
            cost = new_cost
            grad = new_grad
            hess = new_hess
        end

        if verbose
            if rem(i, verbose_every)==0
                @printf "%d: eta=%g cost=%g jtype=%s costheta=%.3f ps=" i eta cost jumptype costheta
                print_vector_g(vector_wrap(bbox, args, params))
                @printf "\n"
                if verbose_level >= 3
                    @printf "    At this point, grad is ="
                    print_vector_g(grad)
                    @printf "\n"                
                end
            end
        end
    end

    trajectory = trajectory[:,1:i]; cpm_traj = cpm_traj[:,1:i]
    if length(report_file)>0
        save(report_file, Dict("traj"=>trajectory, "cpm_traj"=>cpm_traj, "ftraj"=>ftraj))
    end
    
    return vector_wrap(bbox, args, params), trajectory, cost, cpm_traj, ftraj, hess
end


function error_ellipse(x_bf, hessian, idx)
    
# x_bf : best-fit parameter for the offset of ellipse
# hessian : Hessian matrix for covariance matrix
# idx : id of parameters to get the covrariance error ellipse

    covariance = inv(hessian[idx,idx]);

    eigenval, eigenvec = eig(covariance);
    println(eigenvec)

    # Get the largest eigenvalue
    # Get the index of the largest eigenvector
    largest_eigenval, largest_eigenvec_ind_c = findmax(eigenval);
    largest_eigenvec = eigenvec[:, largest_eigenvec_ind_c];


    # Get the smallest eigenvector and eigenvalue
    if largest_eigenvec_ind_c == 1
        smallest_eigenval = eigenval[2]
        smallest_eigenvec = eigenvec[:,2];
    else
        smallest_eigenval = eigenval[1]
        smallest_eigenvec = eigenvec[1,:];
    end

    # Calculate the angle between the x-axis and the largest eigenvector
    angle = atan2(largest_eigenvec[2], largest_eigenvec[1]);

    # This angle is between -pi and pi.
    # Let's shift it such that the angle is between 0 and 2pi
    if angle < 0
        angle = angle + 2*pi;
    end


    # % Get the 95% confidence interval error ellipse
    chisquare_val_95 = 2.1459;


    theta_grid = linspace(0,2*pi,200);
    phi = angle;

    # % x0,y0 ellipse centre coordinates
    X0=x_bf[idx[1]];
    Y0=x_bf[idx[2]];
    a=sqrt(largest_eigenval);
    b=sqrt(smallest_eigenval);

    # % the ellipse in x and y coordinates
    ellipse_x_r  = chisquare_val_95*a*cos.( theta_grid );
    ellipse_y_r  = chisquare_val_95*b*sin.( theta_grid );

    # %Define a rotation matrix
    R = [ cos(phi) sin(phi); -sin(phi) cos(phi) ];

    # let's rotate the ellipse to some angle phi
    r_ellipse = [ellipse_x_r ellipse_y_r] * R;

    Xs = r_ellipse[:,1] + X0
    Ys = r_ellipse[:,2] + Y0
    return Xs,Ys
end

function error_ellipse_cov(x_bf, covariance, idx)
    
# x_bf : best-fit parameter for the offset of ellipse
# hessian : Hessian matrix for covariance matrix
# idx : id of parameters to get the covrariance error ellipse

    # covariance = inv(hessian[idx,idx]);

    eigenval, eigenvec = eig(covariance);

    # Get the largest eigenvalue
    # Get the index of the largest eigenvector
    largest_eigenval, largest_eigenvec_ind_c = findmax(eigenval);
    largest_eigenvec = eigenvec[:, largest_eigenvec_ind_c];


    # Get the smallest eigenvector and eigenvalue
    if largest_eigenvec_ind_c == 1
        smallest_eigenval = eigenval[2]
        smallest_eigenvec = eigenvec[:,2];
    else
        smallest_eigenval = eigenval[1]
        smallest_eigenvec = eigenvec[1,:];
    end

    # Calculate the angle between the x-axis and the largest eigenvector
    angle = atan2(largest_eigenvec[2], largest_eigenvec[1]);

    # This angle is between -pi and pi.
    # Let's shift it such that the angle is between 0 and 2pi
    if angle < 0
        angle = angle + 2*pi;
    end


    # % Get the 95% confidence interval error ellipse
    chisquare_val_95 = 2.1459;


    theta_grid = linspace(0,2*pi,200);
    phi = angle;

    # % x0,y0 ellipse centre coordinates
    X0=x_bf[idx[1]];
    Y0=x_bf[idx[2]];
    a=sqrt(largest_eigenval);
    b=sqrt(smallest_eigenval);

    # % the ellipse in x and y coordinates
    ellipse_x_r  = chisquare_val_95*a*cos.( theta_grid );
    ellipse_y_r  = chisquare_val_95*b*sin.( theta_grid );

    # %Define a rotation matrix
    R = [ cos(phi) sin(phi); -sin(phi) cos(phi) ];

    # let's rotate the ellipse to some angle phi
    r_ellipse = [ellipse_x_r ellipse_y_r] * R;


    Xs = r_ellipse[:,1] + X0
    Ys = r_ellipse[:,2] + Y0
    return Xs,Ys
end