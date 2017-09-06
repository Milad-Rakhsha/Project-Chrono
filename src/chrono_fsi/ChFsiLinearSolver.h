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
#include "helper_cuda.h"
//#include "helper_cusolver.h"
namespace chrono {
namespace fsi {
enum solverType { gmres, cr, bicgstab, cg, sap };

typedef char MM_typecode[4];

// char* mm_typecode_to_str(MM_typecode matcode);
//
// int mm_read_banner(FILE* f, MM_typecode* matcode);
// int mm_read_mtx_crd_size(FILE* f, int* M, int* N, int* nz);
// int mm_read_mtx_array_size(FILE* f, int* M, int* N);
//
// int mm_write_banner(FILE* f, MM_typecode matcode);
// int mm_write_mtx_crd_size(FILE* f, int M, int N, int nz);
// int mm_write_mtx_array_size(FILE* f, int M, int N);
///********************* MM_typecode query fucntions ***************************/
//
//#define mm_is_matrix(typecode) ((typecode)[0] == 'M')
//#define mm_is_sparse(typecode) ((typecode)[1] == 'C')
//#define mm_is_coordinate(typecode) ((typecode)[1] == 'C')
//#define mm_is_dense(typecode) ((typecode)[1] == 'A')
//#define mm_is_array(typecode) ((typecode)[1] == 'A')
//
//#define mm_is_complex(typecode) ((typecode)[2] == 'C')
//#define mm_is_real(typecode) ((typecode)[2] == 'R')
//#define mm_is_pattern(typecode) ((typecode)[2] == 'P')
//#define mm_is_integer(typecode) ((typecode)[2] == 'I')
//
//#define mm_is_symmetric(typecode) ((typecode)[3] == 'S')
//#define mm_is_general(typecode) ((typecode)[3] == 'G')
//#define mm_is_skew(typecode) ((typecode)[3] == 'K')
//#define mm_is_hermitian(typecode) ((typecode)[3] == 'H')
// int mm_is_valid(MM_typecode matcode); /* too complex for a macro */
//
//// profiling the code
//#define TIME_INDIVIDUAL_LIBRARY_CALLS
//
//#define DBICGSTAB_MAX_ULP_ERR 100
//#define DBICGSTAB_EPS 1.E-14f  // 9e-2
//
//#define CLEANUP()                                             \
//    do {                                                      \
//        if (x)                                                \
//            free(x);                                          \
//        if (f)                                                \
//            free(f);                                          \
//        if (r)                                                \
//            free(r);                                          \
//        if (rw)                                               \
//            free(rw);                                         \
//        if (p)                                                \
//            free(p);                                          \
//        if (pw)                                               \
//            free(pw);                                         \
//        if (s)                                                \
//            free(s);                                          \
//        if (t)                                                \
//            free(t);                                          \
//        if (v)                                                \
//            free(v);                                          \
//        if (tx)                                               \
//            free(tx);                                         \
//        if (Aval)                                             \
//            free(Aval);                                       \
//        if (AcolsIndex)                                       \
//            free(AcolsIndex);                                 \
//        if (ArowsIndex)                                       \
//            free(ArowsIndex);                                 \
//        if (Mval)                                             \
//            free(Mval);                                       \
//        if (devPtrX)                                          \
//            checkCudaErrors(cudaFree(devPtrX));               \
//        if (devPtrF)                                          \
//            checkCudaErrors(cudaFree(devPtrF));               \
//        if (devPtrR)                                          \
//            checkCudaErrors(cudaFree(devPtrR));               \
//        if (devPtrRW)                                         \
//            checkCudaErrors(cudaFree(devPtrRW));              \
//        if (devPtrP)                                          \
//            checkCudaErrors(cudaFree(devPtrP));               \
//        if (devPtrS)                                          \
//            checkCudaErrors(cudaFree(devPtrS));               \
//        if (devPtrT)                                          \
//            checkCudaErrors(cudaFree(devPtrT));               \
//        if (devPtrV)                                          \
//            checkCudaErrors(cudaFree(devPtrV));               \
//        if (devPtrAval)                                       \
//            checkCudaErrors(cudaFree(devPtrAval));            \
//        if (devPtrAcolsIndex)                                 \
//            checkCudaErrors(cudaFree(devPtrAcolsIndex));      \
//        if (devPtrArowsIndex)                                 \
//            checkCudaErrors(cudaFree(devPtrArowsIndex));      \
//        if (devPtrMval)                                       \
//            checkCudaErrors(cudaFree(devPtrMval));            \
//        if (stream)                                           \
//            checkCudaErrors(cudaStreamDestroy(stream));       \
//        if (cublasHandle)                                     \
//            checkCudaErrors(cublasDestroy(cublasHandle));     \
//        if (cusparseHandle)                                   \
//            checkCudaErrors(cusparseDestroy(cusparseHandle)); \
//        fflush(stdout);                                       \
//    } while (0)
//
// static int loadMMSparseMatrix(int& m, int& n, int& nnz, double** aVal, int** aRowInd, int** aColInd) {
//    nnz = 5;
//    *aRowInd = (int*)malloc(5 * sizeof(int));
//    *aColInd = (int*)malloc(5 * sizeof(int));
//    *aVal = (double*)malloc(5 * sizeof(double));
//    (*aVal)[0] = 1.0;
//    (*aVal)[1] = 20.0;
//    (*aVal)[2] = 30.0;
//    (*aVal)[3] = 11.0;
//    (*aVal)[4] = 10.0;
//
//    (*aColInd)[0] = 0;
//    (*aColInd)[1] = 1;
//    (*aColInd)[2] = 2;
//    (*aColInd)[3] = 2;
//    (*aColInd)[4] = 3;
//
//    (*aRowInd)[0] = 0;
//    (*aRowInd)[1] = 1;
//    (*aRowInd)[2] = 3;
//    (*aRowInd)[3] = 4;
//    (*aRowInd)[4] = 5;
//
//    m = 4;
//    n = 4;
//    //    aVal= [ 1, 2, 3, 4 ];
//    //    aColInd = [ 0, 0, 2, 1 ];
//    //    aRowInd = [ 0, 1, 3, 3, 4 ];
//
//    return 0;
//}
class ChFsiLinearSolver {
  public:
    ChFsiLinearSolver(){};
    ChFsiLinearSolver(double mrel_res, double mabs_res, int mmax_iter, solverType msolver) {
        solver = msolver;
        rel_res = mrel_res;
        abs_res = mabs_res;
        max_iter = mmax_iter;
    };

    virtual ~ChFsiLinearSolver();
    double rel_res = 1e-3;
    double abs_res = 1e-6;
    int max_iter = 500;
    chrono::fsi::solverType solver;

    int PCG(int SIZE, int NNZ, double* A, int* ArowIdx, int* AcolIdx, double* x, double* b) {
        cublasHandle_t cublasHandle = 0;
        cusparseHandle_t cusparseHandle = 0;
        cusparseMatDescr_t descrA = 0;
        cusparseMatDescr_t descrM = 0;
        cudaStream_t stream = 0;
        cusparseSolveAnalysisInfo_t info_l = 0;
        cusparseSolveAnalysisInfo_t info_u = 0;

        double *r, *r_old, *rh, *p, *Mp, *AMp, *s, *Ms, *AMs;
        double* M = 0;
        int AcolIdxSize = SIZE + 1, ArowIdxSize = NNZ;

        checkCudaErrors(cudaMalloc((void**)&r, sizeof(double) * SIZE));
        checkCudaErrors(cudaMalloc((void**)&r_old, sizeof(double) * SIZE));
        checkCudaErrors(cudaMalloc((void**)&rh, sizeof(double) * SIZE));
        checkCudaErrors(cudaMalloc((void**)&p, sizeof(double) * SIZE));
        checkCudaErrors(cudaMalloc((void**)&Mp, sizeof(double) * SIZE));
        checkCudaErrors(cudaMalloc((void**)&AMp, sizeof(double) * SIZE));
        checkCudaErrors(cudaMalloc((void**)&s, sizeof(double) * SIZE));
        checkCudaErrors(cudaMalloc((void**)&Ms, sizeof(double) * SIZE));
        checkCudaErrors(cudaMalloc((void**)&AMs, sizeof(double) * SIZE));
        checkCudaErrors(cudaMalloc((void**)&M, sizeof(double) * NNZ));

        //====== Get handle to the CUBLAS context ========
        cublasStatus_t cublasStatus;
        cublasStatus = cublasCreate(&cublasHandle);
        checkCudaErrors(cublasStatus);

        //====== Get handle to the CUSPARSE context ======
        cusparseStatus_t cusparseStatus1, cusparseStatus2;
        cusparseStatus1 = cusparseCreate(&cusparseHandle);
        checkCudaErrors(cusparseStatus1);

        //============ initialize CUSPARSE ===============================================
        cusparseStatus1 = cusparseCreate(&cusparseHandle);
        if (cusparseStatus1 != CUSPARSE_STATUS_SUCCESS) {
            fprintf(stderr, "CUSPARSE initialization failed\n");
            return EXIT_FAILURE;
        }

        //============ create three matrix descriptors =======================================
        cusparseStatus1 = cusparseCreateMatDescr(&descrA);
        cusparseStatus2 = cusparseCreateMatDescr(&descrM);
        if ((cusparseStatus1 != CUSPARSE_STATUS_SUCCESS) || (cusparseStatus2 != CUSPARSE_STATUS_SUCCESS)) {
            fprintf(stderr, "!!!! CUSPARSE cusparseCreateMatDescr (coefficient matrix or preconditioner) error\n");
            return EXIT_FAILURE;
        }
        //==========create three matrix descriptors ===========================================
        checkCudaErrors(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
        checkCudaErrors(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
        checkCudaErrors(cusparseSetMatType(descrM, CUSPARSE_MATRIX_TYPE_GENERAL));
        checkCudaErrors(cusparseSetMatIndexBase(descrM, CUSPARSE_INDEX_BASE_ZERO));

        //        //==========create the analysis info (for lower and upper triangular factors)==========
        //        checkCudaErrors(cusparseCreateSolveAnalysisInfo(&info_l));
        //        checkCudaErrors(cusparseCreateSolveAnalysisInfo(&info_u));
        //        checkCudaErrors(cusparseSetMatFillMode(descrM, CUSPARSE_FILL_MODE_LOWER));
        //        checkCudaErrors(cusparseSetMatDiagType(descrM, CUSPARSE_DIAG_TYPE_UNIT));
        //        checkCudaErrors(cusparseDcsrsv_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, SIZE, NNZ,
        //        descrM, A,
        //                                                ArowIdx, AcolIdx, info_l));
        //        checkCudaErrors(cusparseSetMatFillMode(descrM, CUSPARSE_FILL_MODE_UPPER));
        //        checkCudaErrors(cusparseSetMatDiagType(descrM, CUSPARSE_DIAG_TYPE_NON_UNIT));
        //        checkCudaErrors(cusparseDcsrsv_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, SIZE, NNZ,
        //        descrM, A,
        //                                                ArowIdx, AcolIdx, info_u));
        //
        //        //==========Compute the lower and upper triangular factors using CUSPARSE csrilu0 routine
        //        //        int* MrowIdx = ArowIdx;
        //        //        int* McolIdx = AcolIdx;
        //        checkCudaErrors(cusparseDcsrilu0(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, SIZE, descrM, M,
        //        ArowIdx,
        //                                         AcolIdx, info_l));
        //        checkCudaErrors(cudaThreadSynchronize());

        // checkCudaErrors(cusparseDestroySolveAnalysisInfo(info_l));
        // checkCudaErrors(cusparseDestroySolveAnalysisInfo(info_u));

        //===========================Solution=====================================================
        double rho, rho_old, beta, alpha, negalpha, omega, negomega, temp, temp2;
        double nrmr, nrmr0;
        double zero = 0.0;
        double one = 1.0;
        double mone = -1.0;
        int j = 0;
        rho = 1;
        alpha = 1;
        omega = 1;

        // compute initial residual r0=b-Ax0 (using initial guess in x)
        checkCudaErrors(cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, SIZE, SIZE, NNZ, &mone, descrA,
                                       A, ArowIdx, AcolIdx, x, &zero, r));
        checkCudaErrors(cublasDaxpy(cublasHandle, SIZE, &one, b, 1, r, 1));
        checkCudaErrors(cublasDnrm2(cublasHandle, SIZE, r, 1, &nrmr0));
        printf("Initial ||A*x-b||=%f\n", nrmr0);
        nrmr = nrmr0;
        // copy residual r into r^{\hat} and p
        checkCudaErrors(cublasDcopy(cublasHandle, SIZE, r, 1, rh, 1));
        checkCudaErrors(cublasDcopy(cublasHandle, SIZE, r, 1, r_old, 1));
        checkCudaErrors(cublasDcopy(cublasHandle, SIZE, r, 1, p, 1));
        checkCudaErrors(cublasDdot(cublasHandle, SIZE, rh, 1, r, 1, &rho_old));

        for (int iter = 0; iter < this->max_iter; iter++) {
            checkCudaErrors(cublasDcopy(cublasHandle, SIZE, p, 1, Mp, 1));
            //            checkCudaErrors(cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, SIZE, SIZE,
            //            NNZ, &mone,
            //                                           descrM, M, ArowIdx, AcolIdx, x, &zero, p));

            //            // preconditioning step (lower and upper triangular solve)
            //            // Mp=M^(-1)*p
            //            checkCudaErrors(cusparseSetMatFillMode(descrM, CUSPARSE_FILL_MODE_LOWER));
            //            checkCudaErrors(cusparseSetMatDiagType(descrM, CUSPARSE_DIAG_TYPE_UNIT));
            //            checkCudaErrors(cusparseDcsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, SIZE,
            //            &one, descrM,
            //                                                 M, ArowIdx, AcolIdx, info_l, p,
            //                                                 AMp));  // AMp is just dummy vector to save (Ml^-1*p)
            //
            //            checkCudaErrors(cusparseSetMatFillMode(descrM, CUSPARSE_FILL_MODE_UPPER));
            //            checkCudaErrors(cusparseSetMatDiagType(descrM, CUSPARSE_DIAG_TYPE_NON_UNIT));
            //            checkCudaErrors(cusparseDcsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, SIZE,
            //            &one, descrM,
            //                                                 M, ArowIdx, AcolIdx, info_u, AMp,
            //                                                 Mp));  // AMp is just dummy vector to save
            //                                                 (Ml^-1*p),Mu^-1*AMp=Mp
            //
            // AMp=A*Mp
            checkCudaErrors(cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, SIZE, SIZE, NNZ, &one,
                                           descrA, A, ArowIdx, AcolIdx, Mp, &zero, AMp));

            // alpha=rho/(rw'*AMp)
            checkCudaErrors(cublasDdot(cublasHandle, SIZE, rh, 1, AMp, 1, &temp));
            alpha = rho_old / temp;
            negalpha = -(alpha);
            printf("temp=%f, alpha=%f, ", temp, alpha);

            // s = r-alpha*AMp
            checkCudaErrors(cublasDcopy(cublasHandle, SIZE, r, 1, s, 1));
            checkCudaErrors(cublasDaxpy(cublasHandle, SIZE, &negalpha, AMp, 1, s, 1));

            if (nrmr < this->rel_res * nrmr0 || nrmr < this->abs_res) {
                // x = x+ alpha*Mp
                checkCudaErrors(cublasDaxpy(cublasHandle, SIZE, &alpha, Mp, 1, x, 1));
                printf("final Res=%f\n ", nrmr);

                break;
            }

            checkCudaErrors(cublasDcopy(cublasHandle, SIZE, s, 1, Ms, 1));

            //            // Ms=M^(-1)*s
            //            checkCudaErrors(cusparseSetMatFillMode(descrM, CUSPARSE_FILL_MODE_LOWER));
            //            checkCudaErrors(cusparseSetMatDiagType(descrM, CUSPARSE_DIAG_TYPE_UNIT));
            //            checkCudaErrors(cusparseDcsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, SIZE,
            //            &one, descrM,
            //                                                 M, ArowIdx, AcolIdx, info_l, AMs,
            //                                                 Ms));  // AMs is just dummy vector to save (Ml^-1*s)
            //
            //            checkCudaErrors(cusparseSetMatFillMode(descrM, CUSPARSE_FILL_MODE_UPPER));
            //            checkCudaErrors(cusparseSetMatDiagType(descrM, CUSPARSE_DIAG_TYPE_NON_UNIT));
            //            checkCudaErrors(cusparseDcsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, SIZE,
            //            &one, descrM,
            //                                                 M, ArowIdx, AcolIdx, info_u, AMs,
            //                                                 Ms));  // AMs is just dummy vector to save
            //                                                 (Ml^-1*s),Mu^-1*AMs=Ms

            // AMs=A*Ms
            checkCudaErrors(cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, SIZE, SIZE, NNZ, &one,
                                           descrA, A, ArowIdx, AcolIdx, Ms, &zero, AMs));

            // w_new
            checkCudaErrors(cublasDdot(cublasHandle, SIZE, AMs, 1, s, 1, &temp));
            checkCudaErrors(cublasDdot(cublasHandle, SIZE, AMs, 1, AMs, 1, &temp2));
            omega = temp / temp2;

            // x_{j+1} = x_j + alpha*Mp + omega*Ms
            checkCudaErrors(cublasDaxpy(cublasHandle, SIZE, &alpha, Mp, 1, x, 1));
            checkCudaErrors(cublasDaxpy(cublasHandle, SIZE, &omega, Ms, 1, x, 1));
            // r_{j+1} = s_j - omega*AMs
            negomega = -(omega);
            checkCudaErrors(cublasDcopy(cublasHandle, SIZE, r, 1, r_old, 1));
            checkCudaErrors(cublasDcopy(cublasHandle, SIZE, s, 1, r, 1));
            checkCudaErrors(cublasDaxpy(cublasHandle, SIZE, &negomega, AMs, 1, r, 1));
            checkCudaErrors(cublasDnrm2(cublasHandle, SIZE, r, 1, &nrmr));

            // beta_j = (r_{j+1}, r_star) / (r_j, r_star) * (alpha/omega)
            checkCudaErrors(cublasDdot(cublasHandle, SIZE, rh, 1, r, 1, &rho));
            checkCudaErrors(cublasDdot(cublasHandle, SIZE, rh, 1, r_old, 1, &rho_old));
            beta = rho / rho_old * alpha / omega;
            printf("omega=%f, beta_j=%f, ", omega, beta);

            // p_{j+1} = r_{j+1} + beta*(p_j - omega*A*M*p)
            double nbo = -beta * omega;
            checkCudaErrors(cublasDscal(cublasHandle, SIZE, &beta, p, 1));
            checkCudaErrors(cublasDaxpy(cublasHandle, SIZE, &one, r, 1, p, 1));
            checkCudaErrors(cublasDaxpy(cublasHandle, SIZE, &nbo, AMp, 1, p, 1));
            printf("iter=%d, solver tolerance=%f\n", iter, nrmr);
        }

        //        for (int iter = 0; iter < this->max_iter; iter++) {
        //            rho_old = rho;
        //            checkCudaErrors(cublasDdot(cublasHandle, SIZE, rh, 1, r, 1, &rho));
        //            if (iter > 0) {
        //                beta = (rho / rhop) * (alpha / omega);
        //                negomega = -omega;
        //                checkCudaErrors(cublasDaxpy(cublasHandle, SIZE, &negomega, v, 1, p, 1));
        //                checkCudaErrors(cublasDscal(cublasHandle, SIZE, &beta, p, 1));
        //                checkCudaErrors(cublasDaxpy(cublasHandle, SIZE, &one, r, 1, p, 1));
        //            }
        //            // preconditioning step (lower and upper triangular solve)
        //            // pw=M^(-1)*p
        //            checkCudaErrors(cusparseSetMatFillMode(descrM, CUSPARSE_FILL_MODE_LOWER));
        //            checkCudaErrors(cusparseSetMatDiagType(descrM, CUSPARSE_DIAG_TYPE_UNIT));
        //            checkCudaErrors(cusparseDcsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, SIZE, &one,
        //            descrM,
        //                                                 M, ArowIdx, AcolIdx, info_l, p, t));
        //
        //            checkCudaErrors(cusparseSetMatFillMode(descrM, CUSPARSE_FILL_MODE_UPPER));
        //            checkCudaErrors(cusparseSetMatDiagType(descrM, CUSPARSE_DIAG_TYPE_NON_UNIT));
        //            checkCudaErrors(cusparseDcsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, SIZE, &one,
        //            descrM,
        //                                                 M, ArowIdx, AcolIdx, info_u, t, pold));
        //
        //            // v=A*pw
        //            checkCudaErrors(cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, SIZE, SIZE, NNZ,
        //            &one,
        //                                           descrA, A, ArowIdx, AcolIdx, pold, &zero, v));
        //
        //            // alpha=rho/(rw'*v)
        //            checkCudaErrors(cublasDdot(cublasHandle, SIZE, rh, 1, v, 1, &temp));
        //            alpha = rho / temp;
        //            negalpha = -(alpha);
        //
        //            // ??? s = r-alpha*v
        //            checkCudaErrors(cublasDaxpy(cublasHandle, SIZE, &negalpha, v, 1, r, 1));
        //            // x = x+ alpha*pw
        //            checkCudaErrors(cublasDaxpy(cublasHandle, SIZE, &alpha, pold, 1, x, 1));
        //
        //            checkCudaErrors(cublasDnrm2(cublasHandle, SIZE, r, 1, &nrmr));
        //
        //            if (nrmr < this->rel_res * nrmr0) {
        //                j = 5;
        //                break;
        //            }
        //
        //            // t=(Ml)^-1*r
        //            checkCudaErrors(cusparseSetMatFillMode(descrM, CUSPARSE_FILL_MODE_LOWER));
        //            checkCudaErrors(cusparseSetMatDiagType(descrM, CUSPARSE_DIAG_TYPE_UNIT));
        //            checkCudaErrors(cusparseDcsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, SIZE, &one,
        //            descrM,
        //                                                 M, ArowIdx, AcolIdx, info_l, r, t));
        //
        //            // s=(Mu)^-1*t
        //            checkCudaErrors(cusparseSetMatFillMode(descrM, CUSPARSE_FILL_MODE_UPPER));
        //            checkCudaErrors(cusparseSetMatDiagType(descrM, CUSPARSE_DIAG_TYPE_NON_UNIT));
        //            checkCudaErrors(cusparseDcsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, SIZE, &one,
        //            descrM,
        //                                                 M, ArowIdx, AcolIdx, info_u, t, s));
        //
        //            // t=A*s
        //            checkCudaErrors(cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, SIZE, SIZE, NNZ,
        //            &one,
        //                                           descrA, A, ArowIdx, AcolIdx, s, &zero, t));
        //
        //            // w_new
        //            checkCudaErrors(cublasDdot(cublasHandle, SIZE, t, 1, r, 1, &temp));
        //            checkCudaErrors(cublasDdot(cublasHandle, SIZE, t, 1, t, 1, &temp2));
        //            omega = temp / temp2;
        //            // r = s - w* t
        //            negomega = -(omega);
        //            checkCudaErrors(cublasDaxpy(cublasHandle, SIZE, &omega, s, 1, x, 1));
        //            checkCudaErrors(cublasDaxpy(cublasHandle, SIZE, &negomega, t, 1, r, 1));
        //
        //            checkCudaErrors(cublasDnrm2(cublasHandle, SIZE, r, 1, &nrmr));
        //
        //            if (nrmr < this->rel_res * nrmr0) {
        //                iter++;
        //                j = 0;
        //                break;
        //            }
        //            iter++;
        //            printf("iter=%d, solver tolerance=%f\n", iter, nrmr);
        //        }

        return EXIT_SUCCESS;
    }

    //    void gpu_pbicgstab(
    //        cublasHandle_t cublasHandle,
    //        cusparseHandle_t cusparseHandle,
    //        int m,
    //        int n,
    //        int nnz,
    //        const cusparseMatDescr_t descra, /* the coefficient matrix in CSR format */
    //        double* a,
    //        int* ia,
    //        int* ja,
    //        const cusparseMatDescr_t descrm, /* the preconditioner in CSR format, lower & upper triangular factor */
    //        double* vm,
    //        int* im,
    //        int* jm,
    //        cusparseSolveAnalysisInfo_t info_l,
    //        cusparseSolveAnalysisInfo_t info_u, /* the analysis of the lower and upper triangular parts */
    //        double* f,
    //        double* r,
    //        double* rw,
    //        double* p,
    //        double* pw,
    //        double* s,
    //        double* t,
    //        double* v,
    //        double* x,
    //        int maxit,
    //        double tol,
    //        double ttt_sv) {
    //        double rho, rhop, beta, alpha, negalpha, omega, negomega, temp, temp2;
    //        double nrmr, nrmr0;
    //        rho = 0.0;
    //        double zero = 0.0;
    //        double one = 1.0;
    //        double mone = -1.0;
    //        int i = 0;
    //        int j = 0;
    //        double ttl, ttl2, ttu, ttu2, ttm, ttm2;
    //        double ttt_mv = 0.0;
    //        rho = 1;
    //        alpha = 1;
    //        omega = 1;
    //
    //        printf("nnz=%d\n", nnz);
    //
    //        // WARNING: Analysis is done outside of the function (and the time taken by it is passed to the function
    //        in
    //        // variable
    //        // ttt_sv)
    //        // compute initial residual r0=b-Ax0 (using initial guess in x)
    //
    //        checkCudaErrors(cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz, &one, descra,
    //        a, ia,
    //                                       ja, x, &zero, r));
    //
    //        checkCudaErrors(cublasDnrm2(cublasHandle, n, r, 1, &nrmr0));
    //        checkCudaErrors(cublasDscal(cublasHandle, n, &mone, r, 1));
    //        checkCudaErrors(cublasDaxpy(cublasHandle, n, &one, f, 1, r, 1));
    //        printf("||A*x-b||=%f\n", nrmr0);
    //
    //        // copy residual r into r^{\hat} and p
    //        checkCudaErrors(cublasDcopy(cublasHandle, n, r, 1, rw, 1));
    //        checkCudaErrors(cublasDcopy(cublasHandle, n, r, 1, p, 1));
    //        checkCudaErrors(cublasDnrm2(cublasHandle, n, r, 1, &nrmr0));
    //
    //        for (i = 0; i < maxit;) {
    //            rhop = rho;
    //            checkCudaErrors(cublasDdot(cublasHandle, n, rw, 1, r, 1, &rho));
    //            if (i > 0) {
    //                beta = (rho / rhop) * (alpha / omega);
    //                negomega = -omega;
    //                checkCudaErrors(cublasDaxpy(cublasHandle, n, &negomega, v, 1, p, 1));
    //                checkCudaErrors(cublasDscal(cublasHandle, n, &beta, p, 1));
    //                checkCudaErrors(cublasDaxpy(cublasHandle, n, &one, r, 1, p, 1));
    //            }
    //            // preconditioning step (lower and upper triangular solve)
    //            // pw=M^(-1)*p
    //            checkCudaErrors(cusparseSetMatFillMode(descrm, CUSPARSE_FILL_MODE_LOWER));
    //            checkCudaErrors(cusparseSetMatDiagType(descrm, CUSPARSE_DIAG_TYPE_UNIT));
    //            checkCudaErrors(cusparseDcsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, &one,
    //            descrm, vm,
    //                                                 im, jm, info_l, p, t));
    //
    //            checkCudaErrors(cusparseSetMatFillMode(descrm, CUSPARSE_FILL_MODE_UPPER));
    //            checkCudaErrors(cusparseSetMatDiagType(descrm, CUSPARSE_DIAG_TYPE_NON_UNIT));
    //            checkCudaErrors(cusparseDcsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, &one,
    //            descrm, vm,
    //                                                 im, jm, info_u, t, pw));
    //
    //            // v=A*pw
    //            checkCudaErrors(cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz, &one,
    //            descra, a,
    //                                           ia, ja, pw, &zero, v));
    //
    //            // alpha=rho/(rw'*v)
    //            checkCudaErrors(cublasDdot(cublasHandle, n, rw, 1, v, 1, &temp));
    //            alpha = rho / temp;
    //            negalpha = -(alpha);
    //
    //            // ??? s = r-alpha*v
    //            checkCudaErrors(cublasDaxpy(cublasHandle, n, &negalpha, v, 1, r, 1));
    //            // x = x+ alpha*pw
    //            checkCudaErrors(cublasDaxpy(cublasHandle, n, &alpha, pw, 1, x, 1));
    //
    //            checkCudaErrors(cublasDnrm2(cublasHandle, n, r, 1, &nrmr));
    //
    //            if (nrmr < tol * nrmr0) {
    //                j = 5;
    //                break;
    //            }
    //
    //            // t=(Ml)^-1*r
    //            checkCudaErrors(cusparseSetMatFillMode(descrm, CUSPARSE_FILL_MODE_LOWER));
    //            checkCudaErrors(cusparseSetMatDiagType(descrm, CUSPARSE_DIAG_TYPE_UNIT));
    //            checkCudaErrors(cusparseDcsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, &one,
    //            descrm, vm,
    //                                                 im, jm, info_l, r, t));
    //
    //            // s=(Mu)^-1*t
    //            checkCudaErrors(cusparseSetMatFillMode(descrm, CUSPARSE_FILL_MODE_UPPER));
    //            checkCudaErrors(cusparseSetMatDiagType(descrm, CUSPARSE_DIAG_TYPE_NON_UNIT));
    //            checkCudaErrors(cusparseDcsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, &one,
    //            descrm, vm,
    //                                                 im, jm, info_u, t, s));
    //
    //            // t=A*s
    //            checkCudaErrors(cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz, &one,
    //            descra, a,
    //                                           ia, ja, s, &zero, t));
    //
    //            // w_new
    //            checkCudaErrors(cublasDdot(cublasHandle, n, t, 1, r, 1, &temp));
    //            checkCudaErrors(cublasDdot(cublasHandle, n, t, 1, t, 1, &temp2));
    //            omega = temp / temp2;
    //            // r = s - w* t
    //            negomega = -(omega);
    //            checkCudaErrors(cublasDaxpy(cublasHandle, n, &omega, s, 1, x, 1));
    //            checkCudaErrors(cublasDaxpy(cublasHandle, n, &negomega, t, 1, r, 1));
    //
    //            checkCudaErrors(cublasDnrm2(cublasHandle, n, r, 1, &nrmr));
    //
    //            if (nrmr < tol * nrmr0) {
    //                i++;
    //                j = 0;
    //                break;
    //            }
    //            i++;
    //            printf("iter=%d, solver tolerance=%f\n", i, nrmr);
    //        }
    //        printf("Num iter=%d, Final tolerance=%f\n", i, nrmr);
    //        double* x_h = (double*)malloc(m * sizeof(double));
    //        checkCudaErrors(cudaMemcpy(x_h, x, (size_t)(m * sizeof(double)), cudaMemcpyDeviceToHost));
    //        for (int i = 0; i < m; i++) {
    //            printf("x[i]=%f\n", i, x_h[i]);
    //        }
    //    }
    //
    //    int test_bicgstab(double damping, int maxit, double tol) {
    //        cublasHandle_t cublasHandle = 0;
    //        cusparseHandle_t cusparseHandle = 0;
    //        cusparseMatDescr_t descra = 0;
    //        cusparseMatDescr_t descrm = 0;
    //        cudaStream_t stream = 0;
    //        cusparseSolveAnalysisInfo_t info_l = 0;
    //        cusparseSolveAnalysisInfo_t info_u = 0;
    //        cusparseStatus_t status1, status2, status3;
    //        double* devPtrAval = 0;
    //        int* devPtrAcolsIndex = 0;
    //        int* devPtrArowsIndex = 0;
    //        double* devPtrMval = 0;
    //        int* devPtrMcolsIndex = 0;
    //        int* devPtrMrowsIndex = 0;
    //        double* devPtrX = 0;
    //        double* devPtrF = 0;
    //        double* devPtrR = 0;
    //        double* devPtrRW = 0;
    //        double* devPtrP = 0;
    //        double* devPtrPW = 0;
    //        double* devPtrS = 0;
    //        double* devPtrT = 0;
    //        double* devPtrV = 0;
    //        double* Aval = 0;
    //        int* AcolsIndex = 0;
    //        int* ArowsIndex = 0;
    //        double* Mval = 0;
    //        int* MrowsIndex = 0;
    //        int* McolsIndex = 0;
    //        double* x = 0;
    //        double* tx = 0;
    //        double* f = 0;
    //        double* r = 0;
    //        double* rw = 0;
    //        double* p = 0;
    //        double* pw = 0;
    //        double* s = 0;
    //        double* t = 0;
    //        double* v = 0;
    //        int matrixM;
    //        int matrixN;
    //        int matrixSizeAval, matrixSizeAcolsIndex, matrixSizeArowsIndex, mSizeAval, mSizeAcolsIndex,
    //        mSizeArowsIndex;
    //        int arraySizeX, arraySizeF, arraySizeR, arraySizeRW, arraySizeP, arraySizePW, arraySizeS, arraySizeT,
    //            arraySizeV, nnz, mNNZ;
    //        long long flops;
    //        double start, stop;
    //        int num_iterations, nbrTests, count, base, mbase;
    //        double ttt_sv = 0.0;
    //
    //        cublasStatus_t cublasStatus;
    //        cublasStatus = cublasCreate(&cublasHandle);
    //        checkCudaErrors(cublasStatus);
    //
    //        /* Get handle to the CUSPARSE context */
    //        cusparseStatus_t cusparseStatus;
    //        cusparseStatus = cusparseCreate(&cusparseHandle);
    //        checkCudaErrors(cusparseStatus);
    //
    //        loadMMSparseMatrix(matrixM, matrixN, nnz, &Aval, &ArowsIndex, &AcolsIndex);
    //
    //        matrixSizeAval = nnz;
    //        matrixSizeAcolsIndex = matrixSizeAval;
    //        matrixSizeArowsIndex = matrixM + 1;
    //        printf("Aval[0]=%f, ArowsIndex[0]=%d, AcolsIndex[0]=%d\n", Aval[0], ArowsIndex[0], AcolsIndex[0]);
    //
    //        base = ArowsIndex[0];
    //        if (matrixM != matrixN) {
    //            fprintf(stderr, "!!!! matrix MUST be square, error: m=%d != n=%d\n", matrixM, matrixN);
    //            return EXIT_FAILURE;
    //        }
    //
    //        /* set some extra parameters for lower triangular factor */
    //        mNNZ = ArowsIndex[matrixM] - ArowsIndex[0];
    //        mSizeAval = nnz;
    //        mSizeAcolsIndex = mSizeAval;
    //        mSizeArowsIndex = matrixM + 1;
    //        mbase = ArowsIndex[0];
    //        printf("^^^^ matrixM=%d, matrixN=%d, nnz=%d\n", matrixM, matrixN, mNNZ);
    //
    //        /* compressed sparse row */
    //        arraySizeX = matrixN;
    //        arraySizeF = matrixM;
    //        arraySizeR = matrixM;
    //        arraySizeRW = matrixM;
    //        arraySizeP = matrixN;
    //        arraySizePW = matrixN;
    //        arraySizeS = matrixM;
    //        arraySizeT = matrixM;
    //        arraySizeV = matrixM;
    //
    //        /* initialize cublas */
    //
    //        //        if (cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS) {
    //        //            fprintf(stderr, "!!!! CUBLAS initialization error\n");
    //        //            return EXIT_FAILURE;
    //        //        }
    //        /* initialize cusparse */
    //        status1 = cusparseCreate(&cusparseHandle);
    //
    //        if (status1 != CUSPARSE_STATUS_SUCCESS) {
    //            fprintf(stderr, "!!!! CUSPARSE initialization error\n");
    //            return EXIT_FAILURE;
    //        }
    //        /* create three matrix descriptors */
    //        status1 = cusparseCreateMatDescr(&descra);
    //        status2 = cusparseCreateMatDescr(&descrm);
    //        if ((status1 != CUSPARSE_STATUS_SUCCESS) || (status2 != CUSPARSE_STATUS_SUCCESS)) {
    //            fprintf(stderr, "!!!! CUSPARSE cusparseCreateMatDescr (coefficient matrix or preconditioner)
    //            error\n");
    //            return EXIT_FAILURE;
    //        }
    //        /* create the test matrix and vectors on the host */
    //        checkCudaErrors(cusparseSetMatType(descra, CUSPARSE_MATRIX_TYPE_GENERAL));
    //        if (base) {
    //            checkCudaErrors(cusparseSetMatIndexBase(descra, CUSPARSE_INDEX_BASE_ONE));
    //        } else {
    //            checkCudaErrors(cusparseSetMatIndexBase(descra, CUSPARSE_INDEX_BASE_ZERO));
    //            printf("A zero base! good!\n");
    //        }
    //        checkCudaErrors(cusparseSetMatType(descrm, CUSPARSE_MATRIX_TYPE_GENERAL));
    //        if (mbase) {
    //            checkCudaErrors(cusparseSetMatIndexBase(descrm, CUSPARSE_INDEX_BASE_ONE));
    //        } else {
    //            printf("M zero base! good!\n");
    //            checkCudaErrors(cusparseSetMatIndexBase(descrm, CUSPARSE_INDEX_BASE_ZERO));
    //        }
    //
    //        /* allocate device memory for csr matrix and vectors */
    //        checkCudaErrors(cudaMalloc((void**)&devPtrX, sizeof(double) * arraySizeX));
    //        checkCudaErrors(cudaMalloc((void**)&devPtrF, sizeof(double) * arraySizeF));
    //        checkCudaErrors(cudaMalloc((void**)&devPtrR, sizeof(double) * arraySizeR));
    //        checkCudaErrors(cudaMalloc((void**)&devPtrRW, sizeof(double) * arraySizeRW));
    //        checkCudaErrors(cudaMalloc((void**)&devPtrP, sizeof(double) * arraySizeP));
    //        checkCudaErrors(cudaMalloc((void**)&devPtrPW, sizeof(double) * arraySizePW));
    //        checkCudaErrors(cudaMalloc((void**)&devPtrS, sizeof(double) * arraySizeS));
    //        checkCudaErrors(cudaMalloc((void**)&devPtrT, sizeof(double) * arraySizeT));
    //        checkCudaErrors(cudaMalloc((void**)&devPtrV, sizeof(double) * arraySizeV));
    //        checkCudaErrors(cudaMalloc((void**)&devPtrAval, sizeof(double) * matrixSizeAval));
    //        checkCudaErrors(cudaMalloc((void**)&devPtrAcolsIndex, sizeof(int) * matrixSizeAcolsIndex));
    //        checkCudaErrors(cudaMalloc((void**)&devPtrArowsIndex, sizeof(int) * matrixSizeArowsIndex));
    //        checkCudaErrors(cudaMalloc((void**)&devPtrMval, sizeof(double) * mSizeAval));
    //
    //        /* allocate host memory for  vectors */
    //        x = (double*)malloc(arraySizeX * sizeof(double));
    //        f = (double*)malloc(arraySizeF * sizeof(double));
    //        r = (double*)malloc(arraySizeR * sizeof(double));
    //        rw = (double*)malloc(arraySizeRW * sizeof(double));
    //        p = (double*)malloc(arraySizeP * sizeof(double));
    //        pw = (double*)malloc(arraySizePW * sizeof(double));
    //        s = (double*)malloc(arraySizeS * sizeof(double));
    //        t = (double*)malloc(arraySizeT * sizeof(double));
    //        v = (double*)malloc(arraySizeV * sizeof(double));
    //        tx = (double*)malloc(arraySizeX * sizeof(double));
    //        Mval = (double*)malloc(matrixSizeAval * sizeof(double));
    //        if ((!Aval) || (!AcolsIndex) || (!ArowsIndex) || (!Mval) || (!x) || (!f) || (!r) || (!rw) || (!p) || (!pw)
    //        ||
    //            (!s) || (!t) || (!v) || (!tx)) {
    //            CLEANUP();
    //            fprintf(stderr, "!!!! memory allocation error\n");
    //            return EXIT_FAILURE;
    //        }
    //        /* use streams */
    //        int useStream = 0;
    //        if (useStream) {
    //            checkCudaErrors(cudaStreamCreate(&stream));
    //
    //            if (cublasSetStream(cublasHandle, stream) != CUBLAS_STATUS_SUCCESS) {
    //                CLEANUP();
    //                fprintf(stderr, "!!!! cannot set CUBLAS stream\n");
    //                return EXIT_FAILURE;
    //            }
    //            status1 = cusparseSetStream(cusparseHandle, stream);
    //            if (status1 != CUSPARSE_STATUS_SUCCESS) {
    //                CLEANUP();
    //                fprintf(stderr, "!!!! cannot set CUSPARSE stream\n");
    //                return EXIT_FAILURE;
    //            }
    //        }
    //
    //        /* clean memory */
    //        checkCudaErrors(cudaMemset((void*)devPtrX, 10.0, sizeof(double) * arraySizeX));
    //        checkCudaErrors(cudaMemset((void*)devPtrF, 10.0, sizeof(double) * arraySizeF));
    //        checkCudaErrors(cudaMemset((void*)devPtrR, 10, sizeof(double) * arraySizeR));
    //        checkCudaErrors(cudaMemset((void*)devPtrRW, 0, sizeof(double) * arraySizeRW));
    //        checkCudaErrors(cudaMemset((void*)devPtrP, 0, sizeof(double) * arraySizeP));
    //        checkCudaErrors(cudaMemset((void*)devPtrPW, 0, sizeof(double) * arraySizePW));
    //        checkCudaErrors(cudaMemset((void*)devPtrS, 0, sizeof(double) * arraySizeS));
    //        checkCudaErrors(cudaMemset((void*)devPtrT, 0, sizeof(double) * arraySizeT));
    //        checkCudaErrors(cudaMemset((void*)devPtrV, 0, sizeof(double) * arraySizeV));
    //        checkCudaErrors(cudaMemset((void*)devPtrAval, 0.0, sizeof(double) * matrixSizeAval));
    //        checkCudaErrors(cudaMemset((void*)devPtrAcolsIndex, 0, sizeof(int) * matrixSizeAcolsIndex));
    //        checkCudaErrors(cudaMemset((void*)devPtrArowsIndex, 0, sizeof(int) * matrixSizeArowsIndex));
    //        checkCudaErrors(cudaMemset((void*)devPtrMval, 0, sizeof(double) * mSizeAval));
    //
    //        memset(x, 0.0, arraySizeX * sizeof(double));
    //        for (int i = 0; i < arraySizeF; i++) {
    //            f[i] = 1.0;
    //            x[i] = std::rand();
    //        }
    //        //        memset(f, 0.0, arraySizeF * sizeof(double));
    //        memset(r, 0.0, arraySizeR * sizeof(double));
    //        memset(rw, 0.0, arraySizeRW * sizeof(double));
    //        memset(p, 0.0, arraySizeP * sizeof(double));
    //        memset(pw, 0.0, arraySizePW * sizeof(double));
    //        memset(s, 0.0, arraySizeS * sizeof(double));
    //        memset(t, 0.0, arraySizeT * sizeof(double));
    //        memset(v, 0.0, arraySizeV * sizeof(double));
    //        memset(tx, 0.0, arraySizeX * sizeof(double));
    //
    //        // compute the right-hand-side f=A*e, where e=[1, ..., 1]'
    //        for (int i = 0; i < arraySizeP; i++) {
    //            p[i] = 1.0;
    //        }
    //
    //        /* copy the csr matrix and vectors into device memory */
    //        double start_matrix_copy, stop_matrix_copy, start_preconditioner_copy, stop_preconditioner_copy;
    //
    //        start_matrix_copy = second();
    //
    //        checkCudaErrors(
    //            cudaMemcpy(devPtrAval, Aval, (size_t)(matrixSizeAval * sizeof(double)), cudaMemcpyHostToDevice));
    //        checkCudaErrors(
    //            cudaMemcpy(devPtrMval, devPtrAval, (size_t)(matrixSizeAval * sizeof(double)),
    //            cudaMemcpyDeviceToDevice));
    //        checkCudaErrors(cudaMemcpy(devPtrAcolsIndex, AcolsIndex, (size_t)(matrixSizeAcolsIndex * sizeof(int)),
    //                                   cudaMemcpyHostToDevice));
    //        checkCudaErrors(cudaMemcpy(devPtrArowsIndex, ArowsIndex, (size_t)(matrixSizeArowsIndex * sizeof(int)),
    //                                   cudaMemcpyHostToDevice));
    //
    //        stop_matrix_copy = second();
    //
    //        fprintf(stdout, "Copy matrix from CPU to GPU, time(s) = %10.8f\n", stop_matrix_copy - start_matrix_copy);
    //
    //        double one = 1;
    //        double zero = 0;
    //        double norm = 100;
    //        checkCudaErrors(cudaMemcpy(devPtrX, x, (size_t)(arraySizeX * sizeof(double)), cudaMemcpyHostToDevice));
    //        checkCudaErrors(cudaMemcpy(devPtrF, f, (size_t)(arraySizeF * sizeof(double)), cudaMemcpyHostToDevice));
    //        checkCudaErrors(cudaMemcpy(devPtrR, r, (size_t)(arraySizeR * sizeof(double)), cudaMemcpyHostToDevice));
    //
    //        cublasStatus = cublasDdot(cublasHandle, arraySizeF, devPtrF, 1, devPtrF, 1, &norm);
    //        cudaThreadSynchronize();
    //        printf("test ||r||=%f\n", norm);
    //        //        cusparseStatus_t status =
    //        //            cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, 4, 4, 5, &one, descra,
    //        //            devPtrAval,
    //        //                           devPtrArowsIndex, devPtrAcolsIndex, devPtrX, &zero, devPtrR);
    //        //        if (status != CUSPARSE_STATUS_SUCCESS) {
    //        //            CLEANUP();
    //        //            return 1;
    //        //        }
    //
    //        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //        //        int SIZE_test = 4;
    //        //
    //        //        float* Hostx = (float*)malloc(sizeof(float) * SIZE_test);
    //        //        for (int i = 0; i < SIZE_test; i++) {
    //        //            Hostx[i] = 0.135;
    //        //        }
    //        //        /* Get handle to the CUBLAS context */
    //        //        cublasHandle_t cublasHandle_test = 0;
    //        //        cublasStatus_t cublasStatus_test;
    //        //        cublasStatus_test = cublasCreate(&cublasHandle_test);
    //        //        checkCudaErrors(cublasStatus_test);
    //        //        float* devPtr;
    //        //        checkCudaErrors(cudaMalloc((void**)&devPtr, sizeof(float) * SIZE_test));
    //        //        checkCudaErrors(cudaMemset((void*)devPtr, 1.69, sizeof(float) * SIZE_test));
    //        //        cudaMemcpy(devPtr, Hostx, SIZE_test * sizeof(float), cudaMemcpyHostToDevice);
    //        //        float result = 101;
    //        //        cublasStatus_test = cublasSdot(cublasHandle_test, SIZE_test, devPtr, 1, devPtr, 1, &result);
    //        //        cudaThreadSynchronize();
    //        //        printf("after ||r||=%f\n", result);
    //        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //        checkCudaErrors(cudaMemcpy(devPtrRW, rw, (size_t)(arraySizeRW * sizeof(devPtrRW[0])),
    //        cudaMemcpyHostToDevice));
    //        checkCudaErrors(cudaMemcpy(devPtrP, p, (size_t)(arraySizeP * sizeof(devPtrP[0])),
    //        cudaMemcpyHostToDevice));
    //        checkCudaErrors(cudaMemcpy(devPtrPW, pw, (size_t)(arraySizePW * sizeof(devPtrPW[0])),
    //        cudaMemcpyHostToDevice));
    //        checkCudaErrors(cudaMemcpy(devPtrS, s, (size_t)(arraySizeS * sizeof(devPtrS[0])),
    //        cudaMemcpyHostToDevice));
    //        checkCudaErrors(cudaMemcpy(devPtrT, t, (size_t)(arraySizeT * sizeof(devPtrT[0])),
    //        cudaMemcpyHostToDevice));
    //        checkCudaErrors(cudaMemcpy(devPtrV, v, (size_t)(arraySizeV * sizeof(devPtrV[0])),
    //        cudaMemcpyHostToDevice));
    //
    //        /* --- GPU --- */
    //        /* create the analysis info (for lower and upper triangular factors) */
    //        checkCudaErrors(cusparseCreateSolveAnalysisInfo(&info_l));
    //        checkCudaErrors(cusparseCreateSolveAnalysisInfo(&info_u));
    //
    //        /* analyse the lower and upper triangular factors */
    //        double ttl = second();
    //        checkCudaErrors(cusparseSetMatFillMode(descrm, CUSPARSE_FILL_MODE_LOWER));
    //        checkCudaErrors(cusparseSetMatDiagType(descrm, CUSPARSE_DIAG_TYPE_UNIT));
    //        checkCudaErrors(cusparseDcsrsv_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, matrixM, nnz,
    //        descrm,
    //                                                devPtrAval, devPtrArowsIndex, devPtrAcolsIndex, info_l));
    //        checkCudaErrors(cudaThreadSynchronize());
    //        double ttl2 = second();
    //
    //        double ttu = second();
    //        checkCudaErrors(cusparseSetMatFillMode(descrm, CUSPARSE_FILL_MODE_UPPER));
    //        checkCudaErrors(cusparseSetMatDiagType(descrm, CUSPARSE_DIAG_TYPE_NON_UNIT));
    //        checkCudaErrors(cusparseDcsrsv_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, matrixM, nnz,
    //        descrm,
    //                                                devPtrAval, devPtrArowsIndex, devPtrAcolsIndex, info_u));
    //        checkCudaErrors(cudaThreadSynchronize());
    //        double ttu2 = second();
    //        ttt_sv += (ttl2 - ttl) + (ttu2 - ttu);
    //        printf("analysis lower %f (s), upper %f (s) \n", ttl2 - ttl, ttu2 - ttu);
    //
    //        /* compute the lower and upper triangular factors using CUSPARSE csrilu0 routine (on the GPU) */
    //        double start_ilu, stop_ilu;
    //        printf("CUSPARSE csrilu0 ");
    //        start_ilu = second();
    //        devPtrMrowsIndex = devPtrArowsIndex;
    //        devPtrMcolsIndex = devPtrAcolsIndex;
    //        checkCudaErrors(cusparseDcsrilu0(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, matrixM, descra,
    //        devPtrMval,
    //                                         devPtrArowsIndex, devPtrAcolsIndex, info_l));
    //        checkCudaErrors(cudaThreadSynchronize());
    //        stop_ilu = second();
    //        fprintf(stdout, "time(s) = %10.8f \n", stop_ilu - start_ilu);
    //
    //        /* run the test */
    //        num_iterations = 1;  // 10;
    //        start = second() / num_iterations;
    //        for (count = 0; count < num_iterations; count++) {
    //            for (int i = 0; i < nnz; i++) {
    //                printf("a[%d]=%f,", i, Aval[i]);
    //            }
    //            printf("\n");
    //            for (int i = 0; i < matrixM + 1; i++) {
    //                printf("ia[%d]=%d,", i, ArowsIndex[i]);
    //            }
    //            printf("\n");
    //            for (int i = 0; i < nnz; i++) {
    //                printf("ja[%d]=%d,", i, AcolsIndex[i]);
    //            }
    //            printf("\n");
    //
    //            gpu_pbicgstab(cublasHandle, cusparseHandle, matrixM, matrixN, nnz, descra, devPtrAval,
    //            devPtrArowsIndex,
    //                          devPtrAcolsIndex, descrm, devPtrMval, devPtrMrowsIndex, devPtrMcolsIndex, info_l,
    //                          info_u,
    //                          devPtrF, devPtrR, devPtrRW, devPtrP, devPtrPW, devPtrS, devPtrT, devPtrV, devPtrX,
    //                          maxit, tol,
    //                          ttt_sv);
    //
    //            checkCudaErrors(cudaThreadSynchronize());
    //        }
    //        stop = second() / num_iterations;
    //
    //        /* destroy the analysis info (for lower and upper triangular factors) */
    //        checkCudaErrors(cusparseDestroySolveAnalysisInfo(info_l));
    //        checkCudaErrors(cusparseDestroySolveAnalysisInfo(info_u));
    //
    //        /* copy the result into host memory */
    //        checkCudaErrors(cudaMemcpy(tx, devPtrX, (size_t)(arraySizeX * sizeof(tx[0])), cudaMemcpyDeviceToHost));
    //
    //        return EXIT_SUCCESS;
    //    }
};

}  // end namespace fsi
}  // end namespace chrono
#endif /* CHFSILINEARSOLVER_H_ */
