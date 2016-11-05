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
// Authors: Dario Mangoni, Radu Serban
// =============================================================================
// Interfacing to the Pardiso Sparse Direct Solver from the Intel� MKL Library.
// =============================================================================

#ifndef CHMKLENGINE_H
#define CHMKLENGINE_H

#include <mkl.h>

#include "chrono_mkl/ChApiMkl.h"
#include "chrono/core/ChSparseMatrix.h"

namespace chrono {

/// @addtogroup mkl_module
/// @{

/// Interface class to Intel MKL Pardiso solver.
/// This class wraps the C interface of the solver in order to fit Chrono data structures.
/// This class can still be called by the end-user in order to solve linear systems.
/// See demo_MKL_MklEngine for the related demo.
class ChApiMkl ChMklEngine {
  public:
    ChMklEngine(int pb_size = 0, ChSparseMatrix::SymmetryType matrix_type = ChSparseMatrix::GENERAL);
    ~ChMklEngine();

    enum phase_t {
        COMPLETE = 13,
        ANALYSIS = 11,
        ANALYSIS_NUMFACTORIZATION = 12,
        NUMFACTORIZATION = 22,
        NUMFACTORIZATION_SOLVE = 23,
        SOLVE = 33,
        SOLVE_FORWARD = 331,
        SOLVE_DIAGONAL = 332,
        SOLVE_BACKWARD = 333,
        RELEASE_FACTORS = 0,
        RELEASE_ALL = -1
    };

    /// Set problem dimension.
    void SetProblemSize(int pb_size) { m_n = pb_size; }

    /// Set the problem matrix.
    /// This will also update the problem dimension as well as the matrix symmetry type.
    void SetMatrix(ChSparseMatrix& Z);

    /// Set directly the CSR matrix arrays.
    /// Note that it is implied that the matrix symmetry type is GENERAL.
    void SetMatrix(int pb_size, double* a, int* ia, int* ja);

    /// Set the solution vector.
    /// Note that it is the caller's responsibility to provide an array of appropriate size.
    void SetSolutionVector(ChMatrix<>& x);
    void SetSolutionVector(double* x);

    /// Set the right-hand side vector.
    /// Note that it is the caller's responsibility to ensure that the size is appropriate.
    void SetRhsVector(ChMatrix<>& b);
    void SetRhsVector(double* b);

    /// Set the matrix, as well as the right-hand side and solution arrays.
    void SetProblem(ChSparseMatrix& Z, ChMatrix<>& b, ChMatrix<>& x);

    /// Solver routine.
    int PardisoCall(int phase, int message_level = 0);

    /// Reinitializes the solver to default values.
    void ResetSolver();

    // Output functions

    /// Calculate and return the problem residual res=b-Ax.
    /// Note that it is the caller's responsibility to provide an array of appropriate size.
    void GetResidual(ChMatrix<>& res) const;
    void GetResidual(double* res) const;

    /// Calculate and return the L2-norm of the problem residual, ||b-Ax||.
    double GetResidualNorm() const;

    // Auxiliary functions

    /// Set the value of the specified entry in the Pardiso parameter list.
    void SetIparmValue(int parm_num, int value) { m_iparm[parm_num] = value; }

    /// Return the current value of the specified Pardiso parameter.
    int GetIparmValue(int parm_num) const { return m_iparm[parm_num]; }

    /// Get the Pardiso parameter list.
    int* GetIparmAddress() { return m_iparm; }

    /// Print the current values of the Pardiso solver parameters.
    void PrintPardisoParameters() const;

    // Advanced functions

    /// Enable/disable use of permutation vector.
    /// Indicate to the solver to store the permutation vector and use it in the next calls.
    void UsePermutationVector(bool val);

    void UsePartialSolution(int option = 1, int start_row = 0, int end_row = 0);
    void OutputSchurComplement(int option, int start_row, int end_row = 0);
    void SetPreconditionedCGS(bool val, int L);

  private:
    // Internal functions

    static MKL_INT ConvertMatrixType(ChSparseMatrix::SymmetryType type);
    void resetIparmElement(int iparm_num, int reset_value = 0);

    // Data

    // Matrix in CSR3 format.
    // Note that ChMklEngine does not own this data.
    double* m_a;    ///< pointer to the CSR array of non-zero elements of the A
    MKL_INT* m_ia;  ///< pointer to the CSR array of row indices
    MKL_INT* m_ja;  ///< pointer to the CSR array of columns indices

    // Right-hand side and solution arrays.
    // Note that ChMklEngine does not own this data.
    double* m_b;  ///< rhs vector
    double* m_x;  ///< solution vector

    // Problem properties
    MKL_INT m_n;     ///< (square) matrix size
    MKL_INT m_type;  ///< matrix type
    MKL_INT m_nrhs;  ///< number of rhs vectors

    // Pardiso solver settings
    MKL_INT m_iparm[64];    ///< Pardiso solver parameters
    MKL_INT m_maxfct;         ///< maximum number of numerical factorizations
    std::vector<int> m_perm;  ///< vector in which the permutation is stored
    MKL_INT m_mnum;           ///< 1 <= mnum <= maxfct : which factorizations to use; usually 1

    void* m_pt[64];    ///< Pardiso solver internal data
    int m_last_phase;  ///< cached value for the phase used in the last Pardiso call
};

/// @} mkl_module

}  // end of namespace chrono

#endif