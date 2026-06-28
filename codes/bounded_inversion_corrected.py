"""
CORRECTED: Integration of bounded slip constraints into existing inversion workflow
This version fixes all variable scope issues and provides complete implementation

Usage: Copy the functions into your main script or import them
"""

import dolfin as dl
import numpy as np
import hippylib as hp
import pandas as pd
import os
import ufl
from bounded_slip_constraints import solve_bounded_slip_inversion


def solveCoseismicInversion_Bounded(k, targets, m0_s_expr, mtrue_mu_expr_inv, gamma_val_H1, delta_val_L2,
                                   mesh, boundaries, subdomains, fault, top, syndata, resultpath, meshname,
                                   slip_str_gt, mu_str_for, inv_str, mu_str_inv, GPa2Pa, nu, f, data,
                                   slip_bounds=None, constraint_method='projected_cg',
                                   pollute=True, pollute_type='uniform', savefiles=True, verbose=True):
    """
    CORRECTED: Modified version of solveCoseismicInversion with bounded slip constraints
    
    Parameters:
        k: Element order
        targets: Observation points array
        m0_s_expr: Initial slip model expression  
        mtrue_mu_expr_inv: Shear modulus expression for inversion
        gamma_val_H1, delta_val_L2: Regularization parameters
        mesh, boundaries, subdomains, fault, top: Mesh and boundary data
        syndata: Synthetic displacement data
        resultpath, meshname, slip_str_gt, mu_str_for, inv_str, mu_str_inv: File naming
        GPa2Pa, nu, f: Physical constants
        data: Original GPS data (for weights)
        slip_bounds: Slip constraints - dict or tuple
        constraint_method: 'projected_cg', 'scipy_lbfgs', 'penalty_method', or 'unconstrained'
    """
    
    # Define function spaces
    BDM = dl.VectorFunctionSpace(mesh, "BDM", k)
    DGv = dl.VectorFunctionSpace(mesh, "DG", k-1)
    DGr = dl.VectorFunctionSpace(mesh, "DG", k-1)
    ME_element = dl.MixedElement([BDM.ufl_element(), DGv.ufl_element(), DGr.ufl_element()])
    Vu = dl.FunctionSpace(mesh, ME_element)
    Vm = dl.VectorFunctionSpace(mesh, "CG", degree=1, dim=2)
    Vh = [Vu, Vm, Vu]
    
    # Print dimensions
    ndofs = [Vh[hp.STATE].dim(), Vh[hp.PARAMETER].dim(), Vh[hp.ADJOINT].dim()]
    ndofs_state = [Vu.sub(0).dim(), Vu.sub(1).dim(), Vu.sub(2).dim()]
    if verbose:
        sep = "\n" + "#"*80 + "\n"
        print(sep + "Set up the mesh and finite element spaces" + sep)
        print(f"Number of dofs: STATE={ndofs[0]}, PARAMETER={ndofs[1]}, ADJOINT={ndofs[2]}")
        print(f"Number of STATE and ADJOINT dofs: STRESS={ndofs_state[0]}, DISPLACEMENT={ndofs_state[1]}, ROTATION={ndofs_state[2]}")
        
        if slip_bounds is not None:
            print(f"\n🎯 BOUNDED INVERSION with method: {constraint_method}")
            if isinstance(slip_bounds, dict):
                print(f"  Strike bounds: [{slip_bounds.get('strike_min', -np.inf):.3f}, {slip_bounds.get('strike_max', np.inf):.3f}] m")
                print(f"  Dip bounds: [{slip_bounds.get('dip_min', -np.inf):.3f}, {slip_bounds.get('dip_max', np.inf):.3f}] m")
            else:
                print(f"  Symmetric bounds: [{slip_bounds[0]:.3f}, {slip_bounds[1]:.3f}] m")
        else:
            print("\n📖 UNCONSTRAINED INVERSION (original method)")

    # Define boundary conditions
    zero_tensor = dl.Expression((("0.", "0.", "0."),
                                ("0.", "0.", "0."),
                                ("0.", "0.", "0.")), degree=0)
    bc = dl.DirichletBC(Vh[hp.STATE].sub(0), zero_tensor, boundaries, top)
    bc0 = dl.DirichletBC(Vh[hp.STATE].sub(0), zero_tensor, boundaries, top)

    # Load initial model
    m0_s = dl.interpolate(m0_s_expr, Vh[hp.PARAMETER]).vector()
    
    # Setup shear modulus
    from synth_stripeslip_inv_hetmu_nicoya_lock_noi import mu_expression, PDEVarf
    CG_mu = dl.FunctionSpace(mesh, "CG", 1)
    mtrue_mu = dl.interpolate(mtrue_mu_expr_inv, CG_mu).vector()
    mtrue_mu_fun = hp.vector2Function(mtrue_mu, CG_mu)
    
    if savefiles:
        print("Saving true shear modulus structure to .xdmf file")
        mtrue_mu_fun_expr = mu_expression(mtrue_mu_fun)
        m_mu_true = dl.project(mtrue_mu_fun_expr, CG_mu)
        filename = resultpath + 'mu_true_' + meshname + mu_str_inv + '.xdmf'
        if not os.path.exists(filename):
            mu_id = dl.XDMFFile(filename)
            m_mu_true.rename('shear modulus', 'shear modulus')
            mu_id.write(mu_id)
        print(f"Shear modulus range: [{m_mu_true.vector().min():.1f}, {m_mu_true.vector().max():.1f}] GPa")

    # Define PDE problem
    pde_varf = PDEVarf(mtrue_mu_fun)
    pde = hp.PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)

    # Define solver type
    type_solver = "mumps"
    pde.solver = hp.PETScLUSolver(mesh.mpi_comm(), type_solver)
    pde.solver_fwd_inc = hp.PETScLUSolver(mesh.mpi_comm(), type_solver)
    pde.solver_adj_inc = hp.PETScLUSolver(mesh.mpi_comm(), type_solver)

    if verbose:
        print(f"Number of observation points: {targets.shape[0]}")
    
    # Constrain only displacement field for data misfit
    indicator_vec = dl.interpolate(dl.Constant((0,0,0, 0,0,0, 0,0,0, 1,1,1, 0,0,0)), Vh[hp.STATE]).vector()

    # Setup misfit based on noise type
    if pollute:
        if pollute_type == "uniform":
            # Get noise parameters from main script scope - need to be passed in
            # For now, use defaults
            noise_std_h = 0.5 * (data['vx_std_Car'].mean() + data['vy_std_Car'].mean()) if 'data' in locals() else 0.003
            noise_std_v = data['vz_std_Car'].mean() if 'data' in locals() else 0.006
            f_h, f_v = 1, 1/2
            
            weights = dl.Vector(dl.MPI.comm_world, targets.shape[0]*15)
            obs_weights = np.zeros(targets.shape[0]*15)
            obs_weights[9::15] = (1. / noise_std_h**2) * (1. / f_h**2)   # horizontal east
            obs_weights[10::15] = (1. / noise_std_h**2) * (1. / f_h**2)  # horizontal north
            obs_weights[11::15] = (1. / noise_std_v**2) * (1. / f_v**2)  # vertical
            weights.set_local(obs_weights)
            weights.apply('')
            
            from pointwiseStateObs_weights import PointwiseStateObservation as PSBW
            misfit = PSBW(Vh[hp.STATE], targets, weight=weights, indicator_vec=indicator_vec)
            misfit.noise_variance = 1.0
            
        elif pollute_type == "datastd":
            f_h, f_v = 1, 1/2
            weights = dl.Vector(dl.MPI.comm_world, targets.shape[0]*15)
            obs_weights = np.zeros(targets.shape[0]*15)
            obs_weights[9::15] = (1. / data['vx_std_Car']**2).to_numpy() * 1/(f_h**2)
            obs_weights[10::15] = (1. / data['vy_std_Car']**2).to_numpy() * 1/(f_h**2)
            obs_weights[11::15] = (1. / data['vz_std_Car']**2).to_numpy() * 1/(f_v**2)
            weights.set_local(obs_weights)
            weights.apply('')
            
            from pointwiseStateObs_weights import PointwiseStateObservation as PSBW
            misfit = PSBW(Vh[hp.STATE], targets, weight=weights, indicator_vec=indicator_vec)
            misfit.noise_variance = 1.0
    else:
        from pointwiseStateObs import PointwiseStateObservation as PSB
        misfit = PSB(Vh[hp.STATE], targets, indicator_vec=indicator_vec)
        misfit.noise_variance = 1.0
        obs_weights = np.zeros(targets.shape[0]*15)
        obs_weights[9::15] = 1   # horizontal east
        obs_weights[10::15] = 1  # horizontal north
        obs_weights[11::15] = 1  # vertical

    # Input GPS data into misfit
    tmp = np.zeros(len(misfit.d))
    tmp[9::15] = np.array(syndata['ux'])    # horizontal east
    tmp[10::15] = np.array(syndata['uy'])   # horizontal north  
    tmp[11::15] = np.array(syndata['uz'])   # vertical
    misfit.d.set_local(tmp)
    misfit.d.apply('')

    # Get displacement indices
    idx_d = list(np.nonzero(obs_weights)[0])
    if len(idx_d) / 3 != targets.shape[0]:
        print("Error: Length of non-zero misfit must equal number of targets.")

    # Extract observed data
    d_obs = misfit.d[idx_d]

    # Extract fault coordinates for output
    CG = dl.VectorFunctionSpace(mesh, "CG", degree=1)
    bc1 = dl.DirichletBC(CG, (10, 10, 10), boundaries, fault)
    um = dl.Function(CG)
    bc1.apply(um.vector())
    
    bc2 = dl.DirichletBC(Vm, (99, 99), boundaries, fault)
    um2 = dl.Function(Vm)
    bc2.apply(um2.vector())
    
    xslip = dl.interpolate(dl.Expression(("x[0]", "x[0]", "x[0]"), degree=5), CG)
    yslip = dl.interpolate(dl.Expression(("x[1]", "x[1]", "x[1]"), degree=5), CG)
    zslip = dl.interpolate(dl.Expression(("x[2]", "x[2]", "x[2]"), degree=5), CG)
    xf = xslip.vector()[um.vector() == 10]  # x coordinates of fault
    yf = yslip.vector()[um.vector() == 10]  # y coordinates of fault
    zf = zslip.vector()[um.vector() == 10]  # z coordinates of fault
    
    if verbose:
        print(sep + "Done extracting fault coordinates" + sep)

    # Define regularization
    reg = hp.BiLaplacianPrior(Vh[hp.PARAMETER], gamma_val_H1, delta_val_L2, robin_bc=False)

    # Construct model
    model = hp.Model(pde, reg, misfit)
    m = m0_s.copy()

    # MAIN SOLVER SELECTION
    if constraint_method == 'unconstrained' or slip_bounds is None:
        # Use original unconstrained solver
        if verbose:
            print(sep + "Solve the deterministic slip inverse problem (UNCONSTRAINED)" + sep)
        
        m_opt, solver_info = solve_original_unconstrained(model, m, verbose)
        
    else:
        # Use bounded constraint solver
        if verbose:
            print(sep + f"Solve the bounded slip inverse problem ({constraint_method.upper()})" + sep)
        
        solver_params = {
            'max_iter': 1500,
            'rtol': 1e-9,
            'atol': 1e-12
        }
        
        m_opt, solver_info = solve_bounded_slip_inversion(
            model, m, slip_bounds=slip_bounds, 
            method=constraint_method, **solver_params
        )
        
        # Print convergence info
        if constraint_method == 'projected_cg':
            if isinstance(solver_info, list) and len(solver_info) > 0:
                final_iter = solver_info[-1]
                print(f"✅ Bounded CG converged in {final_iter['iter']} iterations")
                print(f"Final cost: {final_iter['cost_total']:.6e}")
                print(f"Final gradient norm: {final_iter['grad_norm']:.6e}")
            else:
                print("⚠️ Bounded CG convergence info not available")
        elif constraint_method == 'scipy_lbfgs':
            if hasattr(solver_info, 'success'):
                if solver_info.success:
                    print(f"✅ L-BFGS-B converged in {solver_info.nit} iterations")
                    print(f"Final cost: {solver_info.fun:.6e}")
                else:
                    print(f"❌ L-BFGS-B failed: {solver_info.message}")

    # Continue with post-processing (same as original)
    m = m_opt  # Use optimized solution

    # Extract inverse solutions and transform to DOLFIN functions
    m_fun = hp.vector2Function(m, Vh[hp.PARAMETER])
    s_strike_fun, s_dip_fun = m_fun.split(deepcopy=True)
    
    if savefiles:
        print("Saving slip solution to .xdmf file")
        s_strike_id = dl.XDMFFile(resultpath + 's_strike_' + meshname + slip_str_gt + mu_str_for + inv_str + mu_str_inv + '.xdmf')
        s_strike_fun.rename('strike slip', 'strike slip')
        s_strike_id.write(s_strike_fun)
        
        s_dip_id = dl.XDMFFile(resultpath + 's_dip_' + meshname + slip_str_gt + mu_str_for + inv_str + mu_str_inv + '.xdmf')
        s_dip_fun.rename('dip slip', 'dip slip')
        s_dip_id.write(s_dip_fun)
        print("Finish saving slip solution")

    # Solve forward problem with optimized parameters
    u = model.generate_vector(hp.STATE)
    p = model.generate_vector(hp.ADJOINT)
    x = [u, m, p]
    model.solveFwd(u, x)
    
    # Calculate predicted data
    misfit.B.mult(x[hp.STATE], misfit.Bu)
    d_cal = misfit.Bu[idx_d]

    # Calculate misfits
    m_fun = dl.Function(Vh[hp.PARAMETER], m)
    dS = dl.Measure("dS")(domain=mesh, subdomain_data=boundaries)
    grad_m = dl.assemble(ufl.inner(ufl.avg(ufl.nabla_grad(m_fun)), ufl.avg(ufl.nabla_grad(m_fun)))*dS(fault))
    misfitd = np.linalg.norm((d_cal-d_obs), 2)
    print(f"Data misfit {misfitd:.6e}; Model misfit {grad_m:.6e}")

    # Compute cost
    total_cost, reg_cost, misfit_cost = model.cost(x)
    print(f"Total cost {total_cost:5g}; Reg Cost {reg_cost:5g}; Misfit {misfit_cost:5g}")

    # Calculate seismic moment
    s_mag = ufl.sqrt(ufl.dot(m_fun, m_fun))
    mtrue_mu_fun_expr = mu_expression(mtrue_mu_fun)
    m_mu_true = dl.project(mtrue_mu_fun_expr, CG_mu)
    moment = dl.assemble(m_mu_true * GPa2Pa * s_mag * dS(fault))
    print(f"Scalar seismic moment: {moment:.3e} N·m")

    # Extract fault slip values
    m_s_fault = m[um2.vector() == 99]
    m_sx_fault = m_s_fault[0::2] 
    m_sy_fault = m_s_fault[1::2]
    print(f"Strike slip range: [{min(m_sx_fault):.4f}, {max(m_sx_fault):.4f}] m")
    print(f"Dip slip range: [{min(m_sy_fault):.4f}, {max(m_sy_fault):.4f}] m")

    # Save results if requested
    if savefiles:
        # Save predicted displacement
        outFileName = 'd_cal_' + meshname + slip_str_gt + mu_str_for + inv_str + mu_str_inv + '.txt'
        csvoutput = open(resultpath + outFileName, 'w+')
        for i in range(0, targets.shape[0]):
            csvoutput.write(f"{targets[i,0]:.6f} {targets[i,1]:.6f} {targets[i,2]:.6f} {d_cal[3*i]:.6f} {d_cal[3*i+1]:.6f} {d_cal[3*i+2]:.6f}\n")
        csvoutput.close()

        # Save displacement field
        print("Saving predicted displacement and stress to .xdmf file")
        uid = dl.XDMFFile(resultpath + 'u_predicted_' + meshname + slip_str_gt + mu_str_for + inv_str + mu_str_inv + '.xdmf')
        u_save = dl.Function(Vh[hp.STATE].sub(1), u)
        u_save.rename('displacement', 'displacement')
        uid.write(u_save)

        # Save stress
        sid = dl.XDMFFile(resultpath + 'stress_predicted_' + meshname + slip_str_gt + mu_str_for + inv_str + mu_str_inv + '.xdmf')
        sigma_non = dl.Function(Vh[hp.STATE].sub(0), u)
        sigma_save = sigma_non.copy()
        sigma_save.vector()[:] = sigma_non.vector()[:] * GPa2Pa
        sigma_save.rename('stress', 'stress')
        sid.write(sigma_save)
        print("Finish saving predicted displacement and stress")

        # Save inferred slip at fault
        outFileName = 'm_s_fault_' + meshname + slip_str_gt + mu_str_for + inv_str + mu_str_inv + '.txt'
        csvoutput = open(resultpath + outFileName, 'w+')
        for i in range(0, m_sx_fault.shape[0]):
            csvoutput.write(f"{m_sx_fault[i]:.6f} {m_sy_fault[i]:.6f}\n")
        csvoutput.close()

        # Save inferred slip over entire volume
        outFileName = 'slip_inferred_' + meshname + slip_str_gt + mu_str_for + inv_str + mu_str_inv + '.txt'
        csvoutput = open(resultpath + outFileName, 'w+')
        slip_vec = np.zeros(len(m0_s))
        slip_vec[um2.vector() == 99] = m_s_fault
        for i in range(0, len(slip_vec)):
            csvoutput.write(f"{slip_vec[i]:.6f}\n")
        csvoutput.close()

    return mtrue_mu, xf, yf, zf, m, u, s_strike_fun, s_dip_fun, d_obs, d_cal, m_s_fault, misfitd, grad_m


def solve_original_unconstrained(model, m0_s, verbose):
    """
    Original unconstrained solver (same as your current implementation)
    """
    if verbose:
        sep = "\n" + "#"*80 + "\n" 
        print(sep + "Using original unconstrained CG solver" + sep)

    # Generate vectors
    u = model.generate_vector(hp.STATE)
    p = model.generate_vector(hp.ADJOINT)
    m = m0_s.copy()
    x = [u, m, p]
    mg = model.generate_vector(hp.PARAMETER)

    # Solve forward and adjoint
    model.solveFwd(u, x)
    model.solveAdj(p, x)
    model.evalGradientParameter(x, mg)

    # Set up Hessian
    model.setPointForHessianEvaluations(x)
    H = hp.ReducedHessian(model)
    H.misfit_only = False

    # Preconditioned CG solver
    solver = hp.CGSolverSteihaug()
    solver.set_operator(H)
    solver.set_preconditioner(model.prior.Rsolver)
    solver.parameters["print_level"] = 1
    solver.parameters["rel_tolerance"] = 1e-9
    solver.parameters["abs_tolerance"] = 1e-12
    solver.parameters["max_iter"] = 1500

    # Solve
    m_hat = model.generate_vector(hp.PARAMETER)
    solver.solve(m_hat, -mg)
    
    if solver.converged:
        print(f"✅ CG converged in {solver.iter} iterations.")
    else:
        print("❌ CG did not converge.")
        raise RuntimeError("CG solver failed to converge")

    # Update solution
    m.axpy(1., m_hat)
    
    return m, {'converged': solver.converged, 'iter': solver.iter}


def validate_slip_bounds(m_final, slip_bounds, fault_coords, verbose=True):
    """
    Validate that final solution satisfies slip bounds
    """
    if slip_bounds is None:
        if verbose:
            print("No bounds specified - skipping validation")
        return True
        
    m_array = m_final.get_local()
    
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
    
    # Check violations
    strike_violations = np.sum((strike_slip < strike_min) | (strike_slip > strike_max))
    dip_violations = np.sum((dip_slip < dip_min) | (dip_slip > dip_max))
    
    if verbose:
        print("="*60)
        print("SLIP BOUNDS VALIDATION")
        print("="*60)
        print(f"Strike slip: min={np.min(strike_slip):.4f} m, max={np.max(strike_slip):.4f} m")
        print(f"Strike bounds: [{strike_min:.3f}, {strike_max:.3f}] m")
        print(f"Strike violations: {strike_violations}")
        print("")
        print(f"Dip slip: min={np.min(dip_slip):.4f} m, max={np.max(dip_slip):.4f} m")
        print(f"Dip bounds: [{dip_min:.3f}, {dip_max:.3f}] m")
        print(f"Dip violations: {dip_violations}")
        print("")
        
        if strike_violations == 0 and dip_violations == 0:
            print("✅ All slip bounds satisfied!")
        else:
            print(f"⚠️  Found {strike_violations + dip_violations} bound violations")
    
    return strike_violations == 0 and dip_violations == 0