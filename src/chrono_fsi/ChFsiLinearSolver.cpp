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

#include <chrono_fsi/ChFsiLinearSolver.h>
namespace chrono {
namespace fsi {

ChFsiLinearSolver::~ChFsiLinearSolver() {
    // TODO Auto-generated destructor stub
}
void ChFsiLinearSolver::Solve(int SIZE, int NNZ, double* A, uint* ArowIdx, uint* AcolIdx, double* x, double* b) {
    if (solver == (solverType)bicgstab)
        BiCGStab(SIZE, NNZ, A, ArowIdx, AcolIdx, x, b);
}
void ChFsiLinearSolver::BiCGStab(int SIZE, int NNZ, double* A, uint* ArowIdx, uint* AcolIdx, double* x, double* b) {
    cublasHandle_t cublasHandle = 0;
    cusparseHandle_t cusparseHandle = 0;
    cusparseMatDescr_t descrA = 0;
    cusparseMatDescr_t descrM = 0;
    cudaStream_t stream = 0;
    cusparseSolveAnalysisInfo_t info_l = 0;
    cusparseSolveAnalysisInfo_t info_u = 0;

    double *r, *r_old, *rh, *p, *Mp, *AMp, *s, *Ms, *AMs;
    double* M = 0;

    cudaMalloc((void**)&r, sizeof(double) * SIZE);
    cudaMalloc((void**)&r_old, sizeof(double) * SIZE);
    cudaMalloc((void**)&rh, sizeof(double) * SIZE);
    cudaMalloc((void**)&p, sizeof(double) * SIZE);
    cudaMalloc((void**)&Mp, sizeof(double) * SIZE);
    cudaMalloc((void**)&AMp, sizeof(double) * SIZE);
    cudaMalloc((void**)&s, sizeof(double) * SIZE);
    cudaMalloc((void**)&Ms, sizeof(double) * SIZE);
    cudaMalloc((void**)&AMs, sizeof(double) * SIZE);
    cudaMalloc((void**)&M, sizeof(double) * NNZ);
    cudaThreadSynchronize();

    //    cudaMemset((void*)x, 0, sizeof(double) * SIZE);
    cudaMemset((void*)p, 0, sizeof(double) * SIZE);
    cudaMemset((void*)Mp, 0, sizeof(double) * SIZE);
    cudaMemset((void*)AMp, 0, sizeof(double) * SIZE);
    cudaMemset((void*)s, 0, sizeof(double) * SIZE);
    cudaMemset((void*)Ms, 0, sizeof(double) * SIZE);
    cudaMemset((void*)AMs, 0, sizeof(double) * SIZE);
    cudaThreadSynchronize();

    //====== Get handle to the CUBLAS context ========
    cublasStatus_t cublasStatus;
    cublasStatus = cublasCreate(&cublasHandle);
    cudaThreadSynchronize();

    //====== Get handle to the CUSPARSE context ======
    cusparseStatus_t cusparseStatus1, cusparseStatus2;
    cusparseStatus1 = cusparseCreate(&cusparseHandle);
    cusparseStatus2 = cusparseCreate(&cusparseHandle);
    cudaThreadSynchronize();

    //============ initialize CUBLAS ===============================================
    if (cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! CUBLAS initialization error\n");
        exit(0);
    }
    //============ initialize CUSPARSE ===============================================
    if (cusparseCreate(&cusparseHandle) != CUSPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "CUSPARSE initialization failed\n");
        exit(0);
    }

    //============ create three matrix descriptors =======================================
    cusparseStatus1 = cusparseCreateMatDescr(&descrA);
    cusparseStatus2 = cusparseCreateMatDescr(&descrM);
    if ((cusparseStatus1 != CUSPARSE_STATUS_SUCCESS) || (cusparseStatus2 != CUSPARSE_STATUS_SUCCESS)) {
        fprintf(stderr, "!!!! CUSPARSE cusparseCreateMatDescr (coefficient matrix or preconditioner) error\n");
    }
    cudaThreadSynchronize();

    //    ==========create three matrix descriptors ===========================================
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descrM, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrM, CUSPARSE_INDEX_BASE_ZERO);
    cudaThreadSynchronize();

    //==========create the analysis info (for lower and upper triangular factors)==========
    //    cusparseCreateSolveAnalysisInfo(&info_l);
    //    cusparseCreateSolveAnalysisInfo(&info_u);
    //    cusparseSetMatFillMode(descrM, CUSPARSE_FILL_MODE_LOWER);
    //    cusparseSetMatDiagType(descrM, CUSPARSE_DIAG_TYPE_UNIT);
    //    cusparseDcsrsv_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, SIZE, NNZ, descrM, A, ArowIdx,
    //                            AcolIdx, info_l);
    //    cusparseSetMatFillMode(descrM, CUSPARSE_FILL_MODE_UPPER);
    //    cusparseSetMatDiagType(descrM, CUSPARSE_DIAG_TYPE_NON_UNIT);
    //    cusparseDcsrsv_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, SIZE, NNZ, descrM, A, ArowIdx,
    //                            AcolIdx, info_u);
    //    cudaThreadSynchronize();

    //==========Compute the lower and upper triangular factors using CUSPARSE csrilu0 routine
    //        int* MrowIdx = ArowIdx;
    //        int* McolIdx = AcolIdx;
    //    cusparseDcsrilu0(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, SIZE, descrM, M, ArowIdx,
    //    AcolIdx,
    //                     info_l);
    //    cudaThreadSynchronize();

    //===========================Solution=====================================================
    double rho, rho_old, beta, alpha, negalpha, omega, negomega, temp, temp2;
    double nrmr, nrmr0;
    double zero = 0.0;
    double one = 1.0;
    double mone = -1.0;
    rho = 1;
    alpha = 1;
    omega = 1;

    // compute initial residual r0=b-Ax0 (using initial guess in x)
    cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, SIZE, SIZE, NNZ, &mone, descrA, A, (int*)ArowIdx,
                   (int*)AcolIdx, x, &zero, r);
    cudaThreadSynchronize();
    cublasDaxpy(cublasHandle, SIZE, &one, b, 1, r, 1);
    cudaThreadSynchronize();
    cublasDnrm2(cublasHandle, SIZE, r, 1, &nrmr0);
    nrmr = nrmr0;
    // copy residual r into r^{\hat} and p
    cublasDcopy(cublasHandle, SIZE, r, 1, rh, 1);
    cublasDcopy(cublasHandle, SIZE, r, 1, r_old, 1);
    cudaThreadSynchronize();
    cublasDdot(cublasHandle, SIZE, rh, 1, r, 1, &rho_old);
    cudaThreadSynchronize();

    for (Iterations = 0; Iterations < max_iter; Iterations++) {
        cublasDdot(cublasHandle, SIZE, rh, 1, r_old, 1, &rho);
        cudaThreadSynchronize();

        // beta_j = (r_{j+1}, r_star) / (r_j, r_star) * (alpha/omega)
        beta = rho / rho_old * alpha / omega;
        // p_{j+1} = r_{j+1} + beta*(p_j - omega*A*M*p)

        double nbo = -beta * omega;

        cublasDscal(cublasHandle, SIZE, &beta, p, 1);
        cudaThreadSynchronize();

        cublasDaxpy(cublasHandle, SIZE, &one, r_old, 1, p, 1);
        cudaThreadSynchronize();
        cublasDaxpy(cublasHandle, SIZE, &nbo, AMp, 1, p, 1);
        cudaThreadSynchronize();
        // Mp=M*p
        cublasDcopy(cublasHandle, SIZE, p, 1, Mp, 1);
        cudaThreadSynchronize();

        //        // Mp=M^(-1)*p
        //        cusparseSetMatFillMode(descrM, CUSPARSE_FILL_MODE_LOWER);
        //        cusparseSetMatDiagType(descrM, CUSPARSE_DIAG_TYPE_UNIT);
        //        cusparseDcsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, SIZE, &one, descrM, M,
        //        ArowIdx,
        //                             AcolIdx, info_l, p,
        //                             AMp);  // AMp is just dummy vector to save (Ml^-1*p)
        //        cudaThreadSynchronize();
        //        cusparseSetMatFillMode(descrM, CUSPARSE_FILL_MODE_UPPER);
        //        cusparseSetMatDiagType(descrM, CUSPARSE_DIAG_TYPE_NON_UNIT);
        //        cusparseDcsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, SIZE, &one, descrM, M,
        //        ArowIdx,
        //                             AcolIdx, info_u, AMp,
        //                             Mp);  // AMp is just dummy vector to save (Ml ^ -1 * p), Mu ^ -1 * AMp = Mp
        //
        //        cudaThreadSynchronize();

        // AMp=A*Mp
        cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, SIZE, SIZE, NNZ, &one, descrA, A,
                       (int*)ArowIdx, (int*)AcolIdx, Mp, &zero, AMp);
        cudaThreadSynchronize();

        // alpha=rho/(rh'*AMp)
        cublasDdot(cublasHandle, SIZE, rh, 1, AMp, 1, &temp);
        cudaThreadSynchronize();
        alpha = rho / temp;
        negalpha = -(alpha);
        cublasDnrm2(cublasHandle, SIZE, Mp, 1, &nrmr);
        cudaThreadSynchronize();
        //            nrmr *= alpha;

        if (nrmr < rel_res * nrmr0 || nrmr < abs_res) {
            // x = x+ alpha*Mp
            cublasDaxpy(cublasHandle, SIZE, &alpha, Mp, 1, x, 1);
            cudaThreadSynchronize();
            residual = nrmr;
            solver_status = 1;
            break;
        }

        // s = r_old-alpha*AMp
        cublasDcopy(cublasHandle, SIZE, r_old, 1, s, 1);
        cudaThreadSynchronize();
        cublasDaxpy(cublasHandle, SIZE, &negalpha, AMp, 1, s, 1);
        cudaThreadSynchronize();

        cublasDcopy(cublasHandle, SIZE, s, 1, Ms, 1);
        cudaThreadSynchronize();

        //        // Ms=M^(-1)*s
        //        cusparseSetMatFillMode(descrM, CUSPARSE_FILL_MODE_LOWER);
        //        cusparseSetMatDiagType(descrM, CUSPARSE_DIAG_TYPE_UNIT);
        //        cusparseDcsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, SIZE, &one, descrM, M,
        //        ArowIdx,
        //                             AcolIdx, info_l, AMs,
        //                             Ms);  // AMs is just dummy vector to save
        //                                   //        (Ml ^ -1 * s)
        //
        //        cusparseSetMatFillMode(descrM, CUSPARSE_FILL_MODE_UPPER);
        //        cusparseSetMatDiagType(descrM, CUSPARSE_DIAG_TYPE_NON_UNIT);
        //        cusparseDcsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, SIZE, &one, descrM, M,
        //        ArowIdx,
        //                             AcolIdx, info_u, AMs,
        //                             Ms);  // AMs is just dummy vector to save (Ml ^ -1 * s),Mu ^ -1 * AMs = Ms

        // AMs=A*Ms
        cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, SIZE, SIZE, NNZ, &one, descrA, A,
                       (int*)ArowIdx, (int*)AcolIdx, Ms, &zero, AMs);
        cudaThreadSynchronize();

        // w_new
        cublasDdot(cublasHandle, SIZE, AMs, 1, Ms, 1, &temp);
        cublasDdot(cublasHandle, SIZE, AMs, 1, AMs, 1, &temp2);
        cudaThreadSynchronize();

        omega = temp / temp2;
        //        printf("alpha=%f, temp=%f, beta=%f, rho_old=%f, rho=%f ", alpha, temp, beta, rho_old, rho);

        // x_{j+1} = x_j + alpha*Mp + omega*Ms
        cublasDaxpy(cublasHandle, SIZE, &alpha, Mp, 1, x, 1);
        cudaThreadSynchronize();
        cublasDaxpy(cublasHandle, SIZE, &omega, Ms, 1, x, 1);

        // r_{j+1} = s_j - omega*AMs
        negomega = -(omega);
        cublasDcopy(cublasHandle, SIZE, s, 1, r, 1);
        cudaThreadSynchronize();
        cublasDaxpy(cublasHandle, SIZE, &negomega, AMs, 1, r, 1);
        cudaThreadSynchronize();
        cublasDnrm2(cublasHandle, SIZE, r, 1, &nrmr);

        cublasDcopy(cublasHandle, SIZE, r, 1, r_old, 1);
        rho_old = rho;
        cudaThreadSynchronize();
        if (verbose)
            printf("Iterations=%d\t ||b-A*x||=%.4e\n", Iterations, nrmr);
    }

    cusparseDestroySolveAnalysisInfo(info_l);
    cusparseDestroySolveAnalysisInfo(info_u);
    cudaFree(r);
    cudaFree(r_old);
    cudaFree(rh);
    cudaFree(p);
    cudaFree(Mp);
    cudaFree(AMp);
    cudaFree(s);
    cudaFree(Ms);
    cudaFree(AMs);
    cudaFree(M);
}

}  // end namespace fsi
}  // end namespace chrono
