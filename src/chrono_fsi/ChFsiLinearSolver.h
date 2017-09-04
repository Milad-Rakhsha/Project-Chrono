/*
 * ChFsiLinearSolver.h
 *
 *  Created on: Sep 4, 2017
 *      Author: milad
 */

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
#include "helper_cusolver.h"
#include "mmio.h"

#include "mmio_wrapper.h"

#include "helper_cuda.h"

// profiling the code
#define TIME_INDIVIDUAL_LIBRARY_CALLS

#define DBICGSTAB_MAX_ULP_ERR 100
#define DBICGSTAB_EPS 1.E-14f  // 9e-2

class ChFsiLinearSolver {
  public:
    ChFsiLinearSolver();
    virtual ~ChFsiLinearSolver();
    enum solverType { gmres, cr, bicgstab, bicgstab_m, cg, sap };

    static void gpu_pbicgstab(
        cublasHandle_t cublasHandle,
        cusparseHandle_t cusparseHandle,
        int m,
        int n,
        int nnz,
        const cusparseMatDescr_t descra, /* the coefficient matrix in CSR format */
        double* a,
        int* ia,
        int* ja,
        const cusparseMatDescr_t descrm, /* the preconditioner in CSR format, lower & upper triangular factor */
        double* vm,
        int* im,
        int* jm,
        cusparseSolveAnalysisInfo_t info_l,
        cusparseSolveAnalysisInfo_t info_u, /* the analysis of the lower and upper triangular parts */
        double* f,
        double* r,
        double* rw,
        double* p,
        double* pw,
        double* s,
        double* t,
        double* v,
        double* x,
        int maxit,
        double tol,
        double ttt_sv) {
        double rho, rhop, beta, alpha, negalpha, omega, negomega, temp, temp2;
        double nrmr, nrmr0;
        rho = 0.0;
        double zero = 0.0;
        double one = 1.0;
        double mone = -1.0;
        int i = 0;
        int j = 0;
        double ttl, ttl2, ttu, ttu2, ttm, ttm2;
        double ttt_mv = 0.0;

// WARNING: Analysis is done outside of the function (and the time taken by it is passed to the function in variable
// ttt_sv)

// compute initial residual r0=b-Ax0 (using initial guess in x)
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
        checkCudaErrors(cudaThreadSynchronize());
        ttm = second();
#endif

        checkCudaErrors(cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz, &one, descra, a, ia,
                                       ja, x, &zero, r));
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
        cudaThreadSynchronize();
        ttm2 = second();
        ttt_mv += (ttm2 - ttm);
// printf("matvec %f (s)\n",ttm2-ttm);
#endif
        checkCudaErrors(cublasDscal(cublasHandle, n, &mone, r, 1));
        checkCudaErrors(cublasDaxpy(cublasHandle, n, &one, f, 1, r, 1));
        // copy residual r into r^{\hat} and p
        checkCudaErrors(cublasDcopy(cublasHandle, n, r, 1, rw, 1));
        checkCudaErrors(cublasDcopy(cublasHandle, n, r, 1, p, 1));
        checkCudaErrors(cublasDnrm2(cublasHandle, n, r, 1, &nrmr0));
        // printf("gpu, init residual:norm %20.16f\n",nrmr0);

        for (i = 0; i < maxit;) {
            rhop = rho;
            checkCudaErrors(cublasDdot(cublasHandle, n, rw, 1, r, 1, &rho));

            if (i > 0) {
                beta = (rho / rhop) * (alpha / omega);
                negomega = -omega;
                checkCudaErrors(cublasDaxpy(cublasHandle, n, &negomega, v, 1, p, 1));
                checkCudaErrors(cublasDscal(cublasHandle, n, &beta, p, 1));
                checkCudaErrors(cublasDaxpy(cublasHandle, n, &one, r, 1, p, 1));
            }
// preconditioning step (lower and upper triangular solve)
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
            checkCudaErrors(cudaThreadSynchronize());
            ttl = second();
#endif
            checkCudaErrors(cusparseSetMatFillMode(descrm, CUSPARSE_FILL_MODE_LOWER));
            checkCudaErrors(cusparseSetMatDiagType(descrm, CUSPARSE_DIAG_TYPE_UNIT));
            checkCudaErrors(cusparseDcsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, &one, descrm, vm,
                                                 im, jm, info_l, p, t));
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
            checkCudaErrors(cudaThreadSynchronize());
            ttl2 = second();
            ttu = second();
#endif
            checkCudaErrors(cusparseSetMatFillMode(descrm, CUSPARSE_FILL_MODE_UPPER));
            checkCudaErrors(cusparseSetMatDiagType(descrm, CUSPARSE_DIAG_TYPE_NON_UNIT));
            checkCudaErrors(cusparseDcsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, &one, descrm, vm,
                                                 im, jm, info_u, t, pw));
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
            checkCudaErrors(cudaThreadSynchronize());
            ttu2 = second();
            ttt_sv += (ttl2 - ttl) + (ttu2 - ttu);
// printf("solve lower %f (s), upper %f (s) \n",ttl2-ttl,ttu2-ttu);
#endif

// matrix-vector multiplication
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
            checkCudaErrors(cudaThreadSynchronize());
            ttm = second();
#endif

            checkCudaErrors(cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz, &one, descra, a,
                                           ia, ja, pw, &zero, v));
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
            checkCudaErrors(cudaThreadSynchronize());
            ttm2 = second();
            ttt_mv += (ttm2 - ttm);
// printf("matvec %f (s)\n",ttm2-ttm);
#endif

            checkCudaErrors(cublasDdot(cublasHandle, n, rw, 1, v, 1, &temp));
            alpha = rho / temp;
            negalpha = -(alpha);
            checkCudaErrors(cublasDaxpy(cublasHandle, n, &negalpha, v, 1, r, 1));
            checkCudaErrors(cublasDaxpy(cublasHandle, n, &alpha, pw, 1, x, 1));
            checkCudaErrors(cublasDnrm2(cublasHandle, n, r, 1, &nrmr));

            if (nrmr < tol * nrmr0) {
                j = 5;
                break;
            }

// preconditioning step (lower and upper triangular solve)
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
            checkCudaErrors(cudaThreadSynchronize());
            ttl = second();
#endif
            checkCudaErrors(cusparseSetMatFillMode(descrm, CUSPARSE_FILL_MODE_LOWER));
            checkCudaErrors(cusparseSetMatDiagType(descrm, CUSPARSE_DIAG_TYPE_UNIT));
            checkCudaErrors(cusparseDcsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, &one, descrm, vm,
                                                 im, jm, info_l, r, t));
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
            checkCudaErrors(cudaThreadSynchronize());
            ttl2 = second();
            ttu = second();
#endif
            checkCudaErrors(cusparseSetMatFillMode(descrm, CUSPARSE_FILL_MODE_UPPER));
            checkCudaErrors(cusparseSetMatDiagType(descrm, CUSPARSE_DIAG_TYPE_NON_UNIT));
            checkCudaErrors(cusparseDcsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, &one, descrm, vm,
                                                 im, jm, info_u, t, s));
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
            checkCudaErrors(cudaThreadSynchronize());
            ttu2 = second();
            ttt_sv += (ttl2 - ttl) + (ttu2 - ttu);
// printf("solve lower %f (s), upper %f (s) \n",ttl2-ttl,ttu2-ttu);
#endif
// matrix-vector multiplication
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
            checkCudaErrors(cudaThreadSynchronize());
            ttm = second();
#endif

            checkCudaErrors(cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz, &one, descra, a,
                                           ia, ja, s, &zero, t));
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
            checkCudaErrors(cudaThreadSynchronize());
            ttm2 = second();
            ttt_mv += (ttm2 - ttm);
// printf("matvec %f (s)\n",ttm2-ttm);
#endif

            checkCudaErrors(cublasDdot(cublasHandle, n, t, 1, r, 1, &temp));
            checkCudaErrors(cublasDdot(cublasHandle, n, t, 1, t, 1, &temp2));
            omega = temp / temp2;
            negomega = -(omega);
            checkCudaErrors(cublasDaxpy(cublasHandle, n, &omega, s, 1, x, 1));
            checkCudaErrors(cublasDaxpy(cublasHandle, n, &negomega, t, 1, r, 1));

            checkCudaErrors(cublasDnrm2(cublasHandle, n, r, 1, &nrmr));

            if (nrmr < tol * nrmr0) {
                i++;
                j = 0;
                break;
            }
            i++;
        }

#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
        printf("gpu total solve time %f (s), matvec time %f (s)\n", ttt_sv, ttt_mv);
#endif
    }
};

#endif /* CHFSILINEARSOLVER_H_ */
