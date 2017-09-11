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

static int loadMMSparseMatrix(int& m, int& n, int& nnz, double** aVal, int** aRowInd, int** aColInd) {
    nnz = 5;
    m = 4;
    n = 4;
    *aRowInd = (int*)malloc(5 * sizeof(int));
    *aColInd = (int*)malloc(5 * sizeof(int));
    *aVal = (double*)malloc(5 * sizeof(double));
    (*aVal)[0] = 3000.0;
    (*aVal)[1] = 20.0;
    (*aVal)[2] = 30.0;
    (*aVal)[3] = 11.0;
    (*aVal)[4] = 105.0;

    (*aColInd)[0] = 0;
    (*aColInd)[1] = 1;
    (*aColInd)[2] = 2;
    (*aColInd)[3] = 2;
    (*aColInd)[4] = 3;

    (*aRowInd)[0] = 0;
    (*aRowInd)[1] = 1;
    (*aRowInd)[2] = 3;
    (*aRowInd)[3] = 4;
    (*aRowInd)[4] = 5;

    //    aVal= [ 1, 2, 3, 4 ];
    //    aColInd = [ 0, 0, 2, 1 ];
    //    aRowInd = [ 0, 1, 3, 3, 4 ];

    return 0;
}

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
    int PCG_CUDA(int SIZE, int NNZ, double* A, uint* ArowIdx, uint* AcolIdx, double* x, double* b);
};

}  // end namespace fsi
}  // end namespace chrono
#endif /* CHFSILINEARSOLVER_H_ */
