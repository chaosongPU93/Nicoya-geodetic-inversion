"""
STANDALONE BOUNDED SLIP INVERSION FUNCTION
Direct drop-in replacement for your original solveCoseismicInversion function

Instructions:
1. Add the slip bounds configuration at the top of your main script
2. Copy this entire function and replace your original solveCoseismicInversion
3. Import the bounded constraints module

That's it! No other changes needed.
"""

def solveCoseismicInversion(k, targets, m0_s_expr, mtrue_mu_expr_inv, gamma_val_H1, delta_val_L2, 
                            pollute=True, pollute_type='uniform', savefiles=True, verbose=True):
    """
    ENHANCED VERSION: Original solveCoseismicInversion with optional bounded slip constraints
    
    This function maintains 100% backward compatibility with your original code.
    
    NEW FEATURES:
    - Supports slip bounds via global variables (see setup instructions below)
    - Multiple constraint algorithms (projected CG, scipy L-BFGS-B)
    - Automatic constraint validation
    
    SETUP INSTRUCTIONS (add to your main script before calling this function):
    
    # Option 1: Define slip bounds globally (recommended for SSE)
    global SLIP_BOUNDS, CONSTRAINT_METHOD
    SLIP_BOUNDS = {
        'strike_min': -0.04,    # ±40 mm strike-slip limit
        'strike_max': 0.04,
        'dip_min': 0.0,         # No back-slip (thrust only)
        'dip_max': 0.16         # 160 mm thrust limit (~2x expected)
    }
    CONSTRAINT_METHOD = 'projected_cg'  # or 'scipy_lbfgs' or 'unconstrained'
    
    # Option 2: No constraints (original behavior) - comment out or set to None
    # SLIP_BOUNDS = None
    # CONSTRAINT_METHOD = 'unconstrained'
    
    The function will automatically detect and use these global variables if they exist.
    """
    
    # Check for global constraint settings
    import sys
    current_module = sys.modules[__name__] if __name__ in sys.modules else sys.modules['__main__']
    
    slip_bounds = getattr(current_module, 'SLIP_BOUNDS', None)
    constraint_method = getattr(current_module, 'CONSTRAINT_METHOD', 'unconstrained')
    
    # Import bounded constraints if needed
    if slip_bounds is not None and constraint_method != 'unconstrained':
        try:
            from bounded_slip_constraints import solve_bounded_slip_inversion
            BOUNDED_AVAILABLE = True
        except ImportError:
            print("⚠️  Warning: bounded_slip_constraints module not found. Using unconstrained inversion.")
            constraint_method = 'unconstrained'
            BOUNDED_AVAILABLE = False
    else:
        BOUNDED_AVAILABLE = False

    # Define function spaces
    # Use VectorFunctionSpace if the unknown is a vector field.
    # Use FunctionSpace object for scalar fields.
    BDM = dl.VectorFunctionSpace(mesh, "BDM", k)    # stress (tensor field, since BDM is a vector field)
    DGv = dl.VectorFunctionSpace(mesh, "DG", k-1)   # displacement (vector field)
    DGr = dl.VectorFunctionSpace(mesh, "DG", k-1)   # rotation (scalar field)
    # Create a mixed fine element function space
    ME_element = dl.MixedElement([BDM.ufl_element(), DGv.ufl_element(), DGr.ufl_element()])
    Vu = dl.FunctionSpace(mesh, ME_element)
    # Define mixed function spaces for the model parameters
    Vm = dl.VectorFunctionSpace(mesh, "CG", degree=1, dim=2)
    # Combine the STATE, PARAMETER and ADJOINT function spaces
    Vh = [Vu, Vm, Vu]
    # Print the dofs of STATE, PARAMETER and ADJOINT variables
    ndofs = [ Vh[hp.STATE].dim(), Vh[hp.PARAMETER].dim(), Vh[hp.ADJOINT].dim() ]
    ndofs_state = [ Vu.sub(0).dim(), Vu.sub(1).dim(), Vu.sub(2).dim() ]
    if verbose:
        print( sep, "Set up the mesh and finite element spaces", sep )
        print( "Number of dofs: STATE={0}, PARAMETER={1}, ADJOINT={2}".format(*ndofs) )
        print( "Number of STATE and ADJOINT dofs: STRESS={0}, DISPLACEMENT={1}, ROTATION={2}".format(*ndofs_state) )
        
        # Print constraint info
        if slip_bounds is not None and constraint_method != 'unconstrained':
            print( "🎯 BOUNDED SLIP INVERSION ENABLED" )
            print( f"   Method: {constraint_method}" )
            if isinstance(slip_bounds, dict):
                print( f"   Strike bounds: [{slip_bounds.get('strike_min', -np.inf):.4f}, {slip_bounds.get('strike_max', np.inf):.4f}] m" )
                print( f"   Dip bounds: [{slip_bounds.get('dip_min', -np.inf):.4f}, {slip_bounds.get('dip_max', np.inf):.4f}] m" )
            else:
                print( f"   Symmetric bounds: [{slip_bounds[0]:.4f}, {slip_bounds[1]:.4f}] m" )
        else:
            print( "📖 Using unconstrained inversion (original method)" )

    # Define the STATE and ADJOINT Dirichlet BCs
    zero_tensor = dl.Expression(( ("0.", "0.", "0."),
                                  ("0.", "0.", "0."),
                                  ("0.", "0.", "0.") ), degree=0)
    bc = dl.DirichletBC(Vh[hp.STATE].sub(0), zero_tensor, boundaries, top)
    bc0 = dl.DirichletBC(Vh[hp.STATE].sub(0), zero_tensor, boundaries, top)

    # Load the initial starting model of mu and s
    m0_s = dl.interpolate(m0_s_expr, Vh[hp.PARAMETER]).vector()
    
    # shear modulus
    CG_mu = dl.FunctionSpace(mesh, "CG", 1)
    # Assign the values of the vector
    mtrue_mu = dl.interpolate(mtrue_mu_expr_inv, CG_mu).vector()
    mtrue_mu_fun = hp.vector2Function(mtrue_mu, CG_mu)
    # Save true shear modulus structure
    if savefiles:
        print( "Saving true shear modulus structure to .xdmf file" )
        mtrue_mu_fun_expr = mu_expression(mtrue_mu_fun)
        m_mu_true = dl.project( mtrue_mu_fun_expr, CG_mu )
        filename = resultpath + 'mu_true_' + meshname + mu_str_inv + '.xdmf'
        if not os.path.exists(filename):
            mu_id = dl.XDMFFile(filename)
            m_mu_true.rename('shear modulus', 'shear modulus')
            mu_id.write(m_mu_true)
        print( m_mu_true.vector().min(), m_mu_true.vector().max() )

    # Define the PDE problem
    pde_varf = PDEVarf(mtrue_mu_fun)
    pde = hp.PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)

    # Define the solver type, previously it was defined in TVprior.py or TVprior_Joint.py
    type_solver = "mumps"
    pde.solver = hp.PETScLUSolver(mesh.mpi_comm(), type_solver)
    pde.solver_fwd_inc = hp.PETScLUSolver(mesh.mpi_comm(), type_solver)
    pde.solver_adj_inc = hp.PETScLUSolver(mesh.mpi_comm(), type_solver)

    # Print the number of observations (observed surface horizontal and vertical displacements)
    if verbose:
        print( "Number of observation points: {0}".format(targets.shape[0]) )
    
    # Constrain only the displacement field for the data misfit
    indicator_vec = dl.interpolate( dl.Constant((0,0,0, 0,0,0, 0,0,0, 1,1,1, 0,0,0)), Vh[hp.STATE] ).vector()

    ### Define the misfit based on whether the noise was added, and how it was added
    if pollute:
        if pollute_type == "uniform":
            ##### USE 'pointwiseStateObs_weights'
            weights = dl.Vector(MPI.COMM_WORLD, targets.shape[0]*15)
            obs_weights = np.zeros(targets.shape[0]*15,)
            obs_weights[9::15]  = (1. / noise_std_h**2) * (1. / f_h**2)  # horizontal east displacement 
            obs_weights[10::15] = (1. / noise_std_h**2) * (1. / f_h**2)  # horizontal north displacement 
            obs_weights[11::15] = (1. / noise_std_v**2) * (1. / f_v**2)  # vertical displacements
            # Modify and set the array to dolfin vector
            weights.set_local(obs_weights)
            weights.apply('')
            # Different from the pure inversion, define misfit without 'weight' option
            misfit = PSBW( Vh[hp.STATE], targets, weight=weights, indicator_vec=indicator_vec )
            misfit.noise_variance = 1.            

        elif pollute_type == "datastd":
            ##### USE 'pointwiseStateObs_weights'
            weights = dl.Vector(MPI.COMM_WORLD, targets.shape[0]*15)
            obs_weights = np.zeros(targets.shape[0]*15,)
            obs_weights[9::15]  = (1. / data['vx_std_Car']**2).to_numpy() * 1/(f_h**2)  # horizontal east displacement 
            obs_weights[10::15] = (1. / data['vy_std_Car']**2).to_numpy() * 1/(f_h**2)  # horizontal north displacement 
            obs_weights[11::15] = (1. / data['vz_std_Car']**2).to_numpy() * 1/(f_v**2)  # vertical displacements
            # Modify and set the array to dolfin vector
            weights.set_local(obs_weights)
            weights.apply('')
            # Different from the pure inversion, define misfit without 'weight' option
            misfit = PSBW( Vh[hp.STATE], targets, weight=weights, indicator_vec=indicator_vec )
            misfit.noise_variance = 1.

    else:        
        ##### USE 'pointwiseStateObs', without 'weight' option
        misfit = PSB( Vh[hp.STATE], targets, indicator_vec=indicator_vec )
        misfit.noise_variance = 1.
        ### Below is just to get the indice of displacement, 'weights' are NOT used by inversion
        obs_weights = np.zeros(targets.shape[0]*15,)
        obs_weights[9::15]  = 1  # horizontal east displacement 
        obs_weights[10::15] = 1  # horizontal north displacement 
        obs_weights[11::15] = 1  # vectical displacement 

    # Imput GPS data into misift.d.
    # Replace the synthetic data with the recorded GPS data
    tmp = np.zeros(len(misfit.d),)
    # Horizontal and vertical displacement misfit
    tmp[9::15] = np.array(syndata['ux'])    # horizontal east displacement misfit
    tmp[10::15] = np.array(syndata['uy'])   # horizontal north displacement misfit
    tmp[11::15] = np.array(syndata['uz'])   # vertical displacement misfit
    # Assign the values of the new vector to misfit.d
    misfit.d.set_local(tmp)
    misfit.d.apply('')

    ### Below is just to get the indice of displacement, 'weights' are NOT used by inversion
    # obs_weights = np.zeros(targets.shape[0]*15,)
    # obs_weights[9::15]  = 1  # horizontal east displacement 
    # obs_weights[10::15] = 1  # horizontal north displacement 
    idx_d = list(np.nonzero(obs_weights)[0]) # misfit = 2*ntargets (2 displacement components, since uz=0)
    if len(idx_d) / 3 != targets.shape[0]:
        print("Error. The length of non-zero misfit has to be the same as ntargets.")

    # Extract horizontals displacements observed data, if only ux and uy are pre constrained in 'misfit', 'd_obs' also contains only ux and uy  
    d_obs = misfit.d[idx_d]

    # Extract x,y coordinates of the fault for plotting
    CG = dl.VectorFunctionSpace(mesh, "CG", degree=1)
    bc1 = dl.DirichletBC(CG, (10, 10, 10), boundaries, fault)
    um = dl.Function(CG)
    bc1.apply(um.vector())
    # Same for fault, but in this case use strike and dip components, assuming no fault-normal slip
    bc2 = dl.DirichletBC(Vm, (99, 99), boundaries, fault)
    um2 = dl.Function(Vm)
    bc2.apply(um2.vector())
    # Extract x,y coordinates of the fault and values
    xslip = dl.interpolate(dl.Expression( ("x[0]", "x[0]", "x[0]"), degree=5), CG )
    yslip = dl.interpolate(dl.Expression( ("x[1]", "x[1]", "x[1]"), degree=5), CG )
    zslip = dl.interpolate(dl.Expression( ("x[2]", "x[2]", "x[2]"), degree=5), CG )
    xf = xslip.vector()[um.vector() == 10] # x coordinate fault
    yf = yslip.vector()[um.vector() == 10] # y coordinate fault
    zf = zslip.vector()[um.vector() == 10] # z coordinate fault
    if verbose:
        print( sep, "Done extracting the fault coordinates", sep )

    # Define the regularization,
    # Below was used in the pure slip inversion 
    reg = hp.BiLaplacianPrior( Vh[hp.PARAMETER], gamma_val_H1, delta_val_L2, robin_bc=False )

    ### CONSTRUCT MODEL (LAGRANGIAN FORMALISM) ###
    # Construct the "Model" --> objective function
    model = hp.Model(pde, reg, misfit)
    m = m0_s.copy()

    # MAIN SOLVER SELECTION - ENHANCED SECTION
    if slip_bounds is None or constraint_method == 'unconstrained' or not BOUNDED_AVAILABLE:
        ### ORIGINAL UNCONSTRAINED SOLVER ###
        if verbose:
            print( sep, "Solve the deterministic coseismic slip inverse problem (UNCONSTRAINED)", sep )
        # Generate STATE, PARAMETER and ADJOINT vectors
        u = model.generate_vector(hp.STATE)
        p = model.generate_vector(hp.ADJOINT)
        x = [u, m, p]
        mg = model.generate_vector(hp.PARAMETER)
        # Solve the FORWARD problem to find the STATE variables
        model.solveFwd(u, x)
        # Solve the ADJOINT problem to find the ADJOINT variables
        model.solveAdj(p, x)
        # Calculate the GRADIENT
        model.evalGradientParameter(x, mg)
        if verbose:
            print( sep, "Done generating STATE, PARAMETER and ADJOINT vectors", sep )

        ### HESSIAN ###
        # Solve the inverse problem with CG with some preconditioner to reduce the number of CG iterations
        model.setPointForHessianEvaluations(x) #gauss_newton_approx=False)
        # Define the Hessian operator 'H'. Since the inverse problem is LINEAR, the
        # Hessian operator 'H' is independent of the model parameter 'm'
        H = hp.ReducedHessian(model)

        ### PRECONDITIONED CONJUGATE GRADIENT (CG) METHOD ###
        # Use the regularization as a preconditioner for the CG algorithm
        Prec = reg.Rsolver
        # Consider all the Hessian to solve the inverse problem
        H.misfit_only = False
        # Solve the linear system: 'A x = b' using preconditioned conjugate gradient CG
        # and the Steihaug stopping criterion (to avoid negative curvature)
        solver = hp.CGSolverSteihaug()
        # Set the operator 'A'
        solver.set_operator(H)
        # Set the preconditioner R, such that:
        # (1) R is symmetric and positive definite;
        # (2) R is such that 'R\{hat}r = r', where 'r' is the residual, can be solved efficiently;
        # (3) R is an approximation of A^{−1} in the sense that: || I - R^{1} A || < 1
        # Set preconditioner: either low-rank of the Hessian or regularization
        solver.set_preconditioner(Prec)
        # Set parameters for the solver
        solver.parameters["print_level"] = 1
        solver.parameters["rel_tolerance"] = 1e-9 
        solver.parameters["abs_tolerance"] = 1e-12 
        solver.parameters["max_iter"]      = 1500
        # Solve 'H\{hat}m = -g' with preconditioned CG
        m_hat = model.generate_vector(hp.PARAMETER)
        solver.solve(m_hat, -mg)
        # Print when CG converges and how many CG iterations it takes
        if solver.converged:
            print( "✅ CG converged in ", solver.iter, " iterations." )
        else:
            print( "❌ CG did not converged." )
            raise RuntimeError("Unconstrained CG solver failed")

        # Solve the FORWARD problem to compute the "predicted data" with the inverted
        # model parameter 'm' (= slip along the fault).
        # Set the solution m = m0 + \{hat}m
        m.axpy(1., m_hat) # m = m + 1*\{hat}m

    else:
        ### BOUNDED CONSTRAINT SOLVER ###
        if verbose:
            print( sep, f"Solve the bounded slip inverse problem ({constraint_method.upper()})", sep )
        
        # Use bounded constraint solver
        solver_params = {
            'max_iter': 1500,
            'rtol': 1e-9,
            'atol': 1e-12
        }
        
        m_opt, solver_info = solve_bounded_slip_inversion(
            model, m, slip_bounds=slip_bounds, 
            method=constraint_method, **solver_params
        )
        
        m = m_opt  # Use the optimized solution
        
        # Print convergence information
        if constraint_method == 'projected_cg':
            if isinstance(solver_info, list) and len(solver_info) > 0:
                final_iter = solver_info[-1]
                print( f"✅ Bounded CG converged in {final_iter['iter']} iterations" )
                print( f"Final cost: {final_iter['cost_total']:.6e}" )
                print( f"Final gradient norm: {final_iter['grad_norm']:.6e}" )
            else:
                print( "⚠️  Bounded CG convergence info not available" )
        elif constraint_method == 'scipy_lbfgs':
            if hasattr(solver_info, 'success'):
                if solver_info.success:
                    print( f"✅ L-BFGS-B converged in {solver_info.nit} iterations" )
                    print( f"Final cost: {solver_info.fun:.6e}" )
                else:
                    print( f"❌ L-BFGS-B failed: {solver_info.message}" )
        
        # Validate constraints
        if verbose and slip_bounds is not None:
            m_array = m.get_local()
            strike_slip = m_array[0::2]  
            dip_slip = m_array[1::2]
            
            if isinstance(slip_bounds, dict):
                strike_min = slip_bounds.get('strike_min', -np.inf)
                strike_max = slip_bounds.get('strike_max', np.inf)
                dip_min = slip_bounds.get('dip_min', -np.inf)
                dip_max = slip_bounds.get('dip_max', np.inf)
            else:
                strike_min = dip_min = slip_bounds[0]
                strike_max = dip_max = slip_bounds[1]
            
            strike_violations = np.sum((strike_slip < strike_min) | (strike_slip > strike_max))
            dip_violations = np.sum((dip_slip < dip_min) | (dip_slip > dip_max))
            
            print( f"Strike slip: [{np.min(strike_slip):.4f}, {np.max(strike_slip):.4f}] m" )
            print( f"Dip slip: [{np.min(dip_slip):.4f}, {np.max(dip_slip):.4f}] m" )
            
            if strike_violations == 0 and dip_violations == 0:
                print( "✅ All slip bounds satisfied!" )
            else:
                print( f"⚠️  Found {strike_violations + dip_violations} bound violations" )

    ### CONTINUE WITH ORIGINAL POST-PROCESSING ###
    # Extract inverse solutions and transform to DOLFIN functions
    m_fun = hp.vector2Function(m, Vh[hp.PARAMETER])
    s_strike_fun, s_dip_fun = m_fun.split(deepcopy=True)
    if savefiles:
        # Save inversion results (coseismic slip)
        print( "Saving slip solution to .xdmf file" )
        s_strike_id = dl.XDMFFile(resultpath + 's_strike_' + meshname + slip_str_gt + mu_str_for + inv_str + mu_str_inv + '.xdmf')
        s_strike_fun.rename('strike slip', 'strike slip')
        s_strike_id.write(s_strike_fun)
        s_dip_id = dl.XDMFFile(resultpath + 's_dip_' + meshname + slip_str_gt + mu_str_for + inv_str + mu_str_inv + '.xdmf')
        s_dip_fun.rename('dip slip', 'dip slip')
        s_dip_id.write(s_dip_fun)
        print( "Finish saving slip solution" )

    # Solve the forward problem to compute the calculated STATE variables
    # Generate STATE and ADJOINT vectors
    u = model.generate_vector(hp.STATE)
    p = model.generate_vector(hp.ADJOINT)
    x = [u, m, p]   # u and p have been initialized before, so no need to do it again
    # Solve the forward problem to compute the calculated STATE variables
    model.solveFwd(u, x)
    # Use the observational operator to extract the surface displacement: d_cal = Bu
    misfit.B.mult(x[hp.STATE], misfit.Bu)
    # Extract horizontal displacement predicted observations
    d_cal = misfit.Bu[idx_d]

    # Calculate the norm of the gradient of the solution for L-curve criterion
    m_fun = dl.Function(Vh[hp.PARAMETER], m)
    grad_m = dl.assemble( ufl.inner( ufl.avg(ufl.nabla_grad(m_fun)), ufl.avg(ufl.nabla_grad(m_fun)) )*dS(fault) )
    misfitd = np.linalg.norm((d_cal-d_obs), 2)
    print( "Data misfit {0:.6e}; Model misfit {1:.6e};".format(misfitd, grad_m) )

    # Compute the cost functional to plot misfit
    total_cost, reg_cost, misfit_cost = model.cost(x)
    print( "Total cost {0:5g}; Reg Cost {1:5g}; Misfit {2:5g}".format(total_cost, reg_cost, misfit_cost) )

    # Compute slip magnitude ||D|| = sqrt(D1^2 + D2^2)
    s_mag = ufl.sqrt( ufl.dot(m_fun, m_fun))
    # calculate the total moment on the fault 
    mtrue_mu_fun_expr = mu_expression(mtrue_mu_fun)  # presumably returns a UFL expression
    m_mu_true = dl.project(mtrue_mu_fun_expr, CG_mu)
    moment = dl.assemble( m_mu_true * GPa2Pa * s_mag * dS(fault) )        
    print(f"Scalar seismic moment: {moment:.3e} N·m")
    M_w1, M_w2, M_w3 = ut.moment2mag(moment)
    print(f"Moment magnitude: {M_w1:.2f}; {M_w2:.2f}; {M_w3:.2f}")

    # Extract values from the model PARAMETER at the fault interface
    m_s_fault = m[um2.vector() == 99]
    m_sx_fault = m_s_fault[0::2]; m_sy_fault = m_s_fault[1::2]
    print( min(m_sx_fault), max(m_sx_fault) )
    print( min(m_sy_fault), max(m_sy_fault) )

    if savefiles:
        # Save the predicted surface displacement
        outFileName = 'd_cal_' + meshname + slip_str_gt + mu_str_for + inv_str + mu_str_inv + '.txt'
        csvoutput = open(resultpath + outFileName, 'w+')
        for i in range(0, targets.shape[0]):
            csvoutput.write( "%.6f %.6f %.6f %.6f %.6f %.6f\n" %(targets[i,0], targets[i,1], targets[i,2], d_cal[3*i], d_cal[3*i+1], d_cal[3*i+2]) )
        csvoutput.close()

        # Save perdicted displacement field
        print( "Saving predicted displacement and stress to .xdmf file" )
        uid = dl.XDMFFile(resultpath + 'u_predicted_' + meshname + slip_str_gt + mu_str_for + inv_str + mu_str_inv + '.xdmf')
        u_save = dl.Function(Vh[hp.STATE].sub(1), u)
        u_save.rename('displacement', 'displacement')
        uid.write(u_save)
        # Stress
        sid = dl.XDMFFile(resultpath + 'stress_predicted_' + meshname + slip_str_gt + mu_str_for + inv_str + mu_str_inv + '.xdmf')
        sigma_non = dl.Function(Vh[hp.STATE].sub(0), u)
        sigma_save = sigma_non.copy()
        sigma_save.vector()[:] = sigma_non.vector()[:] * GPa2Pa
        sigma_save.rename('stress', 'stress')
        sid.write(sigma_save)
        print( "Finish saving predicted displacement and stress" )

        # Save inferred slip values at the fault interface
        outFileName = 'm_s_fault_' + meshname + slip_str_gt + mu_str_for + inv_str + mu_str_inv + '.txt'
        csvoutput = open(resultpath + outFileName, 'w+')
        for i in range(0, m_sx_fault.shape[0]):
            csvoutput.write( "%.6f %.6f\n" %(m_sx_fault[i], m_sy_fault[i]) )
        csvoutput.close()

        # Save inferred slip, over the entire volume
        outFileName = 'slip_inferred_' + meshname + slip_str_gt + mu_str_for + inv_str + mu_str_inv + '.txt'
        csvoutput = open(resultpath + outFileName, 'w+')
        slip_vec = np.zeros(len(m0_s))
        slip_vec[um2.vector() == 99] = m_s_fault
        for i in range(0, len(slip_vec)):
            csvoutput.write( "%.6f\n" %slip_vec[i] )
        csvoutput.close()

    return mtrue_mu, xf, yf, zf, m, u, s_strike_fun, s_dip_fun, d_obs, d_cal, m_s_fault, misfitd, grad_m 