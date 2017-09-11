// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2014 projectchrono.org
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Author: Milad Rakhsha
// =============================================================================
//
// Class for solving a linear linear system via iterative methods.//
// =============================================================================

#ifndef CHFSILINEARSOLVER_H_
#define CHFSILINEARSOLVER_H_

#include <typeinfo>  // for usage of C++ typeid
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusparse_v2.h"
namespace chrono {
namespace fsi {
enum solverType { gmres, cr, bicgstab, cg, sap };

typedef char MM_typecode[4];

class ChFsiLinearSolver {
  private:
    double rel_res = 1e-3;
    double abs_res = 1e-6;
    int max_iter = 500;
    bool verbose = false;
    int Iterations = 0;
    double residual = 1e5;
    chrono::fsi::solverType solver;
    int solver_status = 0;

  public:
    ChFsiLinearSolver(){};
    ChFsiLinearSolver(chrono::fsi::solverType msolver,
                      double mrel_res = 1e-8,
                      double mabs_res = 1e-4,
                      int mmax_iter = 1000,
                      bool mverbose = false) {
        solver = msolver;
        rel_res = mrel_res;
        abs_res = mabs_res;
        max_iter = mmax_iter;
        verbose = mverbose;
    };

    virtual ~ChFsiLinearSolver();
    double GetResidual() { return residual; }
    int GetNumIterations() { return Iterations; }
    int GetIterationLimit() { return max_iter; }
    int GetSolverStatus() { return solver_status; }
    void Solve(int SIZE, int NNZ, double* A, uint* ArowIdx, uint* AcolIdx, double* x, double* b);
    void BiCGStab(int SIZE, int NNZ, double* A, uint* ArowIdx, uint* AcolIdx, double* x, double* b);
};

}  // end namespace fsi
}  // end namespace chrono
#endif /* CHFSILINEARSOLVER_H_ */
