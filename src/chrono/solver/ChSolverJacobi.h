// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2014 projectchrono.org
// All right reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Authors: Alessandro Tasora, Radu Serban
// =============================================================================

#ifndef CHSOLVERJACOBI_H
#define CHSOLVERJACOBI_H

#include "chrono/solver/ChIterativeSolver.h"

namespace chrono {

/// An iterative solver for VI (VI/CCP/LCP/linear problems,..) based on projective fixed
/// point method, similar to a projected Jacobi method.
/// Note: this method is here mostly for comparison and tests: we suggest you to use the
/// more efficient ChSolverSOR - similar, but faster & converges better.
/// The problem is described by a variational inequality VI(Z*x-d,K):
///
///  | M -Cq'|*|q|- | f|= |0| , l \in Y, c \in Ny, normal cone to Y
///  | Cq -E | |l|  |-b|  |c|
///
/// Also Z symmetric by flipping sign of l_i: |M  Cq'|*| q|-| f|=|0|
///                                           |Cq  E | |-l| |-b| |c|
/// * case linear problem:  all Y_i = R, Ny=0, ex. all bilaterals
/// * case LCP: all Y_i = R+:  c>=0, l>=0, l*c=0
/// * case CCP: Y_i are friction cones

class ChApi ChSolverJacobi : public ChIterativeSolver {
    // Chrono RTTI, needed for serialization
    CH_RTTI(ChSolverJacobi, ChIterativeSolver);

  public:
    ChSolverJacobi(int mmax_iters = 50,       ///< max.number of iterations
                   bool mwarm_start = false,  ///< uses warm start?
                   double mtolerance = 0.0,   ///< tolerance for termination criterion
                   double momega = 0.2        ///< overrelaxation criterion
                   )
        : ChIterativeSolver(mmax_iters, mwarm_start, mtolerance, momega){};

    virtual ~ChSolverJacobi() {}

    /// Performs the solution of the problem.
    /// \return  the maximum constraint violation after termination.
    virtual double Solve(ChSystemDescriptor& sysd  ///< system description with constraints and variables
                         ) override;
};

}  // end namespace chrono

#endif
