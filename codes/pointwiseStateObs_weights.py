"""
name: pointwiseStateObs_weights.py

This script defines the observation operator in FEniCS-hIPPYlib.

Simone Puel (spuel@utexas.edu)
Creation date : 09/11/2022
Last modified date : 06/18/2023
"""


# Import libraries
from __future__ import absolute_import, division, print_function
import dolfin as dl
import hippylib as hp


# Class that defines the obervation operator 
class PointwiseStateObservation(hp.Misfit):

    """
    This class implements pointwise state observations at given locations.
    It assumes that the state variable is a scalar function.
    """

    def __init__(self, Vh, obs_points, weight, indicator_vec=None):
        """
        Constructor:
        - :code:`Vh` is the finite element space for the state variable.
        - :code:`obs_points` is a 2D array number of points by geometric dimensions that stores the location of the observations.
        """

        if indicator_vec is not None:
            B = hp.assemblePointwiseObservation(Vh, obs_points)
            self.B = DomainRestrictedOperator(indicator_vec, B)
        else:
            self.B = hp.assemblePointwiseObservation(Vh, obs_points)

        self.d = dl.Vector(self.B.mpi_comm())
        self.B.init_vector(self.d, 0)
        self.Bu = dl.Vector(self.B.mpi_comm())
        self.B.init_vector(self.Bu, 0)
        self.noise_variance = 1.
        self.weight = weight # vector of 1 / noise_variances

        self.design_lin = dl.Vector(self.B.mpi_comm())
        self.B.init_vector(self.design_lin, 0)
        self.diff = dl.Vector(self.B.mpi_comm())
        self.B.init_vector(self.diff, 0)


    def cost(self, x):
        self.B.mult(x[hp.STATE], self.Bu)
        self.Bu.axpy(-1., self.d)
        return (.5/self.noise_variance)*self.Bu.inner(self.weight*self.Bu)


    def grad(self, i, x, out):
        if i == hp.STATE:
            self.B.mult(x[hp.STATE], self.Bu)
            self.Bu.axpy(-1., self.d)
            self.B.transpmult(self.weight*self.Bu, out)
            out *= (1./self.noise_variance)
        elif i == hp.PARAMETER:
            out.zero()
        else:
            raise IndexError()


    def setLinearizationPoint(self, x, gauss_newton_approx=False):
        self.diff.zero()
        self.B.mult(x[hp.STATE], self.diff)
        self.diff.axpy(-1., self.d)

        self.design_lin.zero()
        self.design_lin.axpy(1., self.weight)
        return


    def apply_ij(self, i, j, dir, out):
        if i == hp.STATE and j == hp.STATE:
            self.B.mult(dir, self.Bu)
            self.B.transpmult(self.weight*self.Bu, out)
            out *= (1./self.noise_variance)
        else:
            out.zero()


# Define class to restrict the data to just displacements
class DomainRestrictedOperator:
    """
    This class defines a linear operator that zeros out fields in the state vector.
    """
    def __init__(self, indicator_vec, B):
        """
        Constructor:

            :code:`indicator_vec`: vector that allows you to select what part of the state you want to zero out\
            when working with a mixed problem (or a problem whose state has multiple fields).

            :code:`B` is a PETSc matrix that projects the state onto the location of observations.
        """
        self.indicator_vec = indicator_vec
        self.B = B

    def mpi_comm(self):
        return self.B.mpi_comm()

    def init_vector(self, v, dim):
        return self.B.init_vector(v, dim)

    def mult(self, u, y):
        self.B.mult(u*self.indicator_vec, y)

    def transpmult(self, y, u):
        self.B.transpmult(y, u)
        u *= self.indicator_vec
