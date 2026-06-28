"""
SLIP TRANSFORMATION UTILITIES - ENHANCED VERSION
Clean, focused utility for bounded slip inversion using multiple transformation functions

This utility class provides slip transformation capabilities with multiple function options
that can be easily imported and used across multiple inversion codes.

Key advantages:
- Multiple transformation function options (tanh, arctan, sigmoid, sqrt)
- No constrained optimization needed (keeps your fast unconstrained solver)  
- Guaranteed bounds satisfaction
- Smooth, differentiable transformations
- Flexible constraint combinations
- Component-wise transformation to avoid UFL splitting issues

Supported transformation functions:
1. tanh(x) - Hyperbolic tangent (default, fastest convergence)
2. arctan(x) - Arctangent (scaled by 2/π)
3. sigmoid(x) - Sigmoid variant: 2/(1+exp(-x)) - 1
4. sqrt(x) - Square root variant: x/sqrt(1+x²)

Author: Claude Code Assistant
Compatible with: FEniCS, hIPPYlib
Created: 2025-10-03 - Enhanced production-ready utility
"""

import dolfin as dl
import ufl
import numpy as np


class SlipTransformation:
    """
    Handle slip transformations with flexible constraint combinations and multiple function options
    Uses various transformation functions: unbounded parameters → bounded slip
    
    Enhanced production-ready utility class for bounded slip inversion.
    Designed to be imported across multiple codes with function selection capability.
    """
    
    def __init__(self, strike_bounds=None, dip_bounds=None, function_type='tanh'):
        """
        Initialize slip transformation
        
        Args:
            strike_bounds: tuple (min, max) or None for unconstrained
            dip_bounds: tuple (min, max) or None for unconstrained
            function_type: str, transformation function type
                          'tanh' - Hyperbolic tangent (default)
                          'arctan' - Arctangent scaled by 2/π  
                          'sigmoid' - 2/(1+exp(-x)) - 1
                          'sqrt' - x/sqrt(1+x²)
            
        Examples:
            # Both components constrained with tanh (default)
            SlipTransformation(strike_bounds=(-0.04, 0.04), dip_bounds=(0.0, 0.16))
            
            # Using arctan transformation
            SlipTransformation(strike_bounds=(-0.04, 0.04), dip_bounds=(0.0, 0.16), 
                             function_type='arctan')
            
            # Only dip constrained with sigmoid (typical for thrust faults)
            SlipTransformation(strike_bounds=None, dip_bounds=(0.0, 0.16), 
                             function_type='sigmoid')
            
            # Using sqrt transformation
            SlipTransformation(strike_bounds=(-0.1, 0.1), dip_bounds=None, 
                             function_type='sqrt')
            
            # No constraints (reverts to original framework)
            SlipTransformation(strike_bounds=None, dip_bounds=None)
        """
        self.strike_bounds = strike_bounds
        self.dip_bounds = dip_bounds
        self.function_type = function_type
        
        # Validate function type
        valid_functions = ['tanh', 'arctan', 'sigmoid', 'sqrt']
        if function_type not in valid_functions:
            raise ValueError(f"function_type must be one of {valid_functions}, got '{function_type}'")
        
        # Determine transformation mode
        self.has_strike_bounds = strike_bounds is not None
        self.has_dip_bounds = dip_bounds is not None
        self.has_any_bounds = self.has_strike_bounds or self.has_dip_bounds
        
        if self.has_strike_bounds:
            self.strike_min, self.strike_max = strike_bounds
            
        if self.has_dip_bounds:
            self.dip_min, self.dip_max = dip_bounds
    
    def __str__(self):
        """String representation for debugging"""
        if not self.has_any_bounds:
            return f"SlipTransformation: UNCONSTRAINED (original framework), function: {self.function_type}"
        
        parts = []
        if self.has_strike_bounds:
            parts.append(f"strike ∈ [{self.strike_min:.4f}, {self.strike_max:.4f}]")
        else:
            parts.append("strike: unconstrained")
            
        if self.has_dip_bounds:
            parts.append(f"dip ∈ [{self.dip_min:.4f}, {self.dip_max:.4f}]")
        else:
            parts.append("dip: unconstrained")
            
        return f"SlipTransformation: {', '.join(parts)}, function: {self.function_type}"
    
    def _apply_transformation_function(self, x):
        """
        Apply the selected transformation function to map (-∞,∞) → (-1,1)
        
        Args:
            x: UFL expression for unbounded parameter
            
        Returns:
            UFL expression mapped to (-1,1) range
        """
        if self.function_type == 'tanh':
            # tanh(x): (-∞,∞) → (-1,1)
            return ufl.tanh(x)
        
        elif self.function_type == 'arctan':
            # 2/π * arctan(x): (-∞,∞) → (-1,1)
            return (2.0 / ufl.pi) * ufl.atan(x)
        
        elif self.function_type == 'sigmoid':
            # 2/(1+exp(-x)) - 1: (-∞,∞) → (-1,1)
            return 2.0 / (1.0 + ufl.exp(-x)) - 1.0
        
        elif self.function_type == 'sqrt':
            # x/sqrt(1+x²): (-∞,∞) → (-1,1)
            return x / ufl.sqrt(1.0 + x**2)
        
        else:
            raise ValueError(f"Unknown function type: {self.function_type}")
    
    def transform_to_physical_slip(self, m):
        """
        Transform unbounded parameters m to bounded physical slip using selected function
        
        Args:
            m: UFL expression or Function for unbounded parameters (2D vector)
            
        Returns:
            UFL expression for physical slip with bounds applied
            
        Note: This method is suitable for numpy post-processing but may cause
        UFL splitting issues. For UFL variational formulations, use the
        component-wise approach shown in the examples.
        """
        if not self.has_any_bounds:
            # No bounds - return original parameters as slip
            return m
            
        # Split into components
        m_strike, m_dip = ufl.split(m)
        
        # Transform strike component
        if self.has_strike_bounds:
            # Apply selected transformation: (-∞,∞) → (-1,1) → (strike_min, strike_max)
            transformed_strike = self._apply_transformation_function(m_strike)
            strike_scaled = (transformed_strike + 1) / 2  # (-1,1) → (0,1)
            s_strike = self.strike_min + (self.strike_max - self.strike_min) * strike_scaled
        else:
            # No bounds - use parameter directly
            s_strike = m_strike
            
        # Transform dip component  
        if self.has_dip_bounds:
            # Apply selected transformation: (-∞,∞) → (-1,1) → (dip_min, dip_max)
            transformed_dip = self._apply_transformation_function(m_dip)
            dip_scaled = (transformed_dip + 1) / 2  # (-1,1) → (0,1)
            s_dip = self.dip_min + (self.dip_max - self.dip_min) * dip_scaled
        else:
            # No bounds - use parameter directly
            s_dip = m_dip
            
        return ufl.as_vector([s_strike, s_dip])
    
    def transform_component_wise(self, m_strike, m_dip):
        """
        Component-wise transformation to avoid UFL splitting issues
        
        Args:
            m_strike: UFL expression for unbounded strike parameter
            m_dip: UFL expression for unbounded dip parameter
            
        Returns:
            tuple: (s_strike, s_dip) UFL expressions for physical slip components
        """
        # Transform strike component
        if self.has_strike_bounds:
            transformed_strike = self._apply_transformation_function(m_strike)
            strike_scaled = (transformed_strike + 1) / 2
            s_strike = self.strike_min + (self.strike_max - self.strike_min) * strike_scaled
        else:
            s_strike = m_strike
            
        # Transform dip component  
        if self.has_dip_bounds:
            transformed_dip = self._apply_transformation_function(m_dip)
            dip_scaled = (transformed_dip + 1) / 2
            s_dip = self.dip_min + (self.dip_max - self.dip_min) * dip_scaled
        else:
            s_dip = m_dip
            
        return s_strike, s_dip
    
    def validate_bounds(self, slip_function, fault_mask=None, verbose=True):
        """
        Validate that transformed slip satisfies bounds
        
        Args:
            slip_function: FEniCS Function containing physical slip values
            fault_mask: Optional boolean mask to extract fault interface values only
            verbose: Print validation results
            
        Returns:
            bool: True if all bounds satisfied
        """
        if not self.has_any_bounds:
            if verbose:
                print("No bounds to validate (unconstrained mode)")
            return True
            
        # Extract slip values
        slip_array = slip_function.vector().get_local()
        
        # Apply fault mask if provided
        if fault_mask is not None:
            slip_array = slip_array[fault_mask]
        
        n_dofs = len(slip_array) // 2
        strike_slip = slip_array[0::2]
        dip_slip = slip_array[1::2]
        
        violations = 0
        
        if verbose:
            print("="*60)
            location_info = "FAULT INTERFACE ONLY" if fault_mask is not None else "ENTIRE MESH"
            print(f"SLIP BOUNDS VALIDATION ({location_info}) - {self.function_type.upper()}")
            print("="*60)
        
        # Check strike bounds
        if self.has_strike_bounds:
            strike_violations = np.sum((strike_slip < self.strike_min) | 
                                     (strike_slip > self.strike_max))
            violations += strike_violations
            
            if verbose:
                print(f"Strike slip: [{np.min(strike_slip):.6f}, {np.max(strike_slip):.6f}] m")
                print(f"Strike bounds: [{self.strike_min:.6f}, {self.strike_max:.6f}] m")
                print(f"Strike violations: {strike_violations}")
        else:
            if verbose:
                print(f"Strike slip: [{np.min(strike_slip):.6f}, {np.max(strike_slip):.6f}] m (unconstrained)")
        
        # Check dip bounds
        if self.has_dip_bounds:
            dip_violations = np.sum((dip_slip < self.dip_min) | 
                                  (dip_slip > self.dip_max))
            violations += dip_violations
            
            if verbose:
                print(f"Dip slip: [{np.min(dip_slip):.6f}, {np.max(dip_slip):.6f}] m")
                print(f"Dip bounds: [{self.dip_min:.6f}, {self.dip_max:.6f}] m") 
                print(f"Dip violations: {dip_violations}")
        else:
            if verbose:
                print(f"Dip slip: [{np.min(dip_slip):.6f}, {np.max(dip_slip):.6f}] m (unconstrained)")
        
        if verbose:
            print("")
            if violations == 0:
                print("✅ ALL BOUNDS SATISFIED!")
            else:
                print(f"⚠️ FOUND {violations} BOUND VIOLATIONS!")
                print("Note: Small violations (~1e-15) are numerical precision artifacts")
        
        return violations == 0


def create_slip_transformer_for_problem(amp, slip_pattern=None, function_type='tanh'):
    """
    Factory function to create appropriate slip transformer for your problem
    
    Args:
        amp: Slip amplitude from your problem setup
        slip_pattern: Optional pattern identifier (for future use)
        function_type: Transformation function type ('tanh', 'arctan', 'sigmoid', 'sqrt')
        
    Returns:
        SlipTransformation instance configured for your problem
    """
    
    if amp < 0.1:  # SSE case
        return SlipTransformation(
            strike_bounds=(-0.5 * amp, 0.5 * amp),    # ±50% of amplitude
            dip_bounds=(0.0, 2.0 * amp),              # 0 to 2x amplitude (thrust only)
            function_type=function_type
        )
    elif amp >= 1.0:  # Large earthquake
        return SlipTransformation(
            strike_bounds=(-10.0, 10.0),              # ±10 m
            dip_bounds=(0.0, 50.0),                   # 0-50 m thrust
            function_type=function_type
        )
    else:  # Moderate case
        return SlipTransformation(
            strike_bounds=(-1.0, 1.0),                # ±1 m
            dip_bounds=(0.0, 5.0),                    # 0-5 m thrust
            function_type=function_type
        )


class BoxConstraintTransformation:
    """
    Direct box constraint transformation for bounded slip inversion
    Uses explicit bounds in optimization solver (like MATLAB lsqlin)
    
    This class provides bounds setup for direct constrained optimization,
    eliminating the smoothing effects of transformation functions.
    """
    
    def __init__(self, strike_bounds=None, dip_bounds=None):
        """
        Initialize box constraint transformation
        
        Args:
            strike_bounds: tuple (min, max) or None for unconstrained
            dip_bounds: tuple (min, max) or None for unconstrained  
        """
        self.strike_bounds = strike_bounds
        self.dip_bounds = dip_bounds
        
        # Determine constraint mode
        self.has_strike_bounds = strike_bounds is not None
        self.has_dip_bounds = dip_bounds is not None
        self.has_any_bounds = self.has_strike_bounds or self.has_dip_bounds
        
        if self.has_strike_bounds:
            self.strike_min, self.strike_max = strike_bounds
            
        if self.has_dip_bounds:
            self.dip_min, self.dip_max = dip_bounds
    
    def __str__(self):
        """String representation for debugging"""
        if not self.has_any_bounds:
            return "BoxConstraintTransformation: UNCONSTRAINED"
        
        parts = []
        if self.has_strike_bounds:
            parts.append(f"strike ∈ [{self.strike_min:.4f}, {self.strike_max:.4f}]")
        else:
            parts.append("strike: unconstrained")
            
        if self.has_dip_bounds:
            parts.append(f"dip ∈ [{self.dip_min:.4f}, {self.dip_max:.4f}]")
        else:
            parts.append("dip: unconstrained")
            
        return f"BoxConstraintTransformation: {', '.join(parts)}"
    
    def create_bounds_vector(self, param_vector, fault_mask=None):
        """
        Create bounds arrays for constrained optimization
        
        Args:
            param_vector: Parameter vector (slip DoFs)
            fault_mask: Optional mask to identify fault interface DoFs
            
        Returns:
            tuple: (lower_bounds, upper_bounds) arrays matching param_vector size
        """
        n_dofs = len(param_vector)
        
        # Initialize with no bounds (None values)
        lower_bounds = np.full(n_dofs, -np.inf)
        upper_bounds = np.full(n_dofs, np.inf)
        
        if not self.has_any_bounds:
            return lower_bounds, upper_bounds
        
        # Apply bounds to fault interface DoFs only (if mask provided)
        if fault_mask is not None:
            fault_indices = np.where(fault_mask)[0]
        else:
            fault_indices = np.arange(n_dofs)
        
        # Apply bounds component-wise (assuming interleaved strike/dip ordering)
        for i in fault_indices:
            component_type = i % 2  # 0 = strike, 1 = dip
            
            if component_type == 0 and self.has_strike_bounds:
                # Strike component
                lower_bounds[i] = self.strike_min
                upper_bounds[i] = self.strike_max
            elif component_type == 1 and self.has_dip_bounds:
                # Dip component  
                lower_bounds[i] = self.dip_min
                upper_bounds[i] = self.dip_max
        
        return lower_bounds, upper_bounds
    
    def create_scipy_bounds(self, param_vector, fault_mask=None):
        """
        Create bounds in scipy.optimize format
        
        Args:
            param_vector: Parameter vector (slip DoFs)
            fault_mask: Optional mask to identify fault interface DoFs
            
        Returns:
            list: Bounds in scipy format [(min, max), ...]
        """
        lower_bounds, upper_bounds = self.create_bounds_vector(param_vector, fault_mask)
        
        bounds = []
        for lb, ub in zip(lower_bounds, upper_bounds):
            if lb == -np.inf and ub == np.inf:
                bounds.append((None, None))
            elif lb == -np.inf:
                bounds.append((None, ub))
            elif ub == np.inf:
                bounds.append((lb, None))
            else:
                bounds.append((lb, ub))
        
        return bounds
    
    def validate_bounds(self, slip_vector, fault_mask=None, verbose=True):
        """
        Validate that slip values satisfy bounds
        
        Args:
            slip_vector: numpy array of slip values
            fault_mask: Optional mask to check only fault interface values
            verbose: Print validation results
            
        Returns:
            bool: True if all bounds satisfied
        """
        if not self.has_any_bounds:
            if verbose:
                print("No bounds to validate (unconstrained mode)")
            return True
        
        # Apply fault mask if provided
        if fault_mask is not None:
            slip_values = slip_vector[fault_mask]
        else:
            slip_values = slip_vector
        
        n_dofs = len(slip_values) // 2
        strike_slip = slip_values[0::2]
        dip_slip = slip_values[1::2]
        
        violations = 0
        
        if verbose:
            print("="*60)
            location_info = "FAULT INTERFACE ONLY" if fault_mask is not None else "ENTIRE MESH"
            print(f"BOX CONSTRAINT VALIDATION ({location_info})")
            print("="*60)
        
        # Check strike bounds
        if self.has_strike_bounds:
            strike_violations = np.sum((strike_slip < self.strike_min) | 
                                     (strike_slip > self.strike_max))
            violations += strike_violations
            
            if verbose:
                print(f"Strike slip: [{np.min(strike_slip):.6f}, {np.max(strike_slip):.6f}] m")
                print(f"Strike bounds: [{self.strike_min:.6f}, {self.strike_max:.6f}] m")
                print(f"Strike violations: {strike_violations}")
        else:
            if verbose:
                print(f"Strike slip: [{np.min(strike_slip):.6f}, {np.max(strike_slip):.6f}] m (unconstrained)")
        
        # Check dip bounds
        if self.has_dip_bounds:
            dip_violations = np.sum((dip_slip < self.dip_min) | 
                                  (dip_slip > self.dip_max))
            violations += dip_violations
            
            if verbose:
                print(f"Dip slip: [{np.min(dip_slip):.6f}, {np.max(dip_slip):.6f}] m")
                print(f"Dip bounds: [{self.dip_min:.6f}, {self.dip_max:.6f}] m") 
                print(f"Dip violations: {dip_violations}")
        else:
            if verbose:
                print(f"Dip slip: [{np.min(dip_slip):.6f}, {np.max(dip_slip):.6f}] m (unconstrained)")
        
        if verbose:
            print("")
            if violations == 0:
                print("✅ ALL BOUNDS SATISFIED!")
            else:
                print(f"⚠️ FOUND {violations} BOUND VIOLATIONS!")
        
        return violations == 0


def compare_transformation_functions():
    """
    Utility function to compare different transformation functions
    Useful for understanding the behavior of each function type
    """
    print("Transformation Function Comparison")
    print("="*50)
    print("Available functions mapping (-∞,∞) → (-1,1):")
    print("1. tanh(x) - Hyperbolic tangent")
    print("2. 2/π * arctan(x) - Scaled arctangent") 
    print("3. 2/(1+exp(-x)) - 1 - Sigmoid variant")
    print("4. x/sqrt(1+x²) - Square root variant")
    print("")
    print("Key differences:")
    print("- tanh: Fastest saturation, steepest at origin")
    print("- arctan: Gradual saturation, linear-like behavior")
    print("- sigmoid: Similar to tanh but different curvature")
    print("- sqrt: Slowest saturation, most linear-like")


if __name__ == "__main__":
    print("Enhanced Slip Transformation Utilities")
    print("="*40)
    print("Utilities for bounded slip inversion with multiple function options:")
    print("1. SlipTransformation: Multiple smooth transformation functions")
    print("2. BoxConstraintTransformation: Direct constraint bounds") 
    print("")
    
    # Show function comparison
    compare_transformation_functions()
    print("")
    
    # Demonstrate different constraint configurations with different functions
    print("Example constraint configurations:")
    
    enhanced_examples = [
        SlipTransformation(strike_bounds=(-0.04, 0.04), dip_bounds=(0.0, 0.16), function_type='tanh'),
        SlipTransformation(strike_bounds=(-0.04, 0.04), dip_bounds=(0.0, 0.16), function_type='arctan'),
        SlipTransformation(strike_bounds=(-0.04, 0.04), dip_bounds=(0.0, 0.16), function_type='sigmoid'),
        SlipTransformation(strike_bounds=(-0.04, 0.04), dip_bounds=(0.0, 0.16), function_type='sqrt'),
        SlipTransformation(strike_bounds=None, dip_bounds=(0.0, 0.16), function_type='tanh'),
    ]
    
    box_examples = [
        BoxConstraintTransformation(strike_bounds=(-0.027, 0.027), dip_bounds=(0.0, 0.0785)),
        BoxConstraintTransformation(strike_bounds=None, dip_bounds=(0.0, 0.0785)),
    ]
    
    print("\nEnhanced Transformation (multiple function options):")
    for i, transformer in enumerate(enhanced_examples, 1):
        print(f"{i}. {transformer}")
        
    print("\nBox Constraint Transformation (hard bounds):")
    for i, transformer in enumerate(box_examples, 1):
        print(f"{i}. {transformer}")