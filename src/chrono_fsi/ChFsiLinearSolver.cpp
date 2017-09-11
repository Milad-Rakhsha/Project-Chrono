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

int ChFsiLinearSolver::PCG_CUDA(int SIZE, int NNZ, double* A, uint* ArowIdx, uint* AcolIdx, double* x, double* b) {
    cublasHandle_t cublasHandle = 0;
    cusparseHandle_t cusparseHandle = 0;
    cusparseMatDescr_t descrA = 0;
    cusparseMatDescr_t descrM = 0;
    cudaStream_t stream = 0;
    cusparseSolveAnalysisInfo_t info_l = 0;
    cusparseSolveAnalysisInfo_t info_u = 0;

    double *r, *r_old, *rh, *p, *pw, *s, *t, *v;
    double* M = 0;

    cudaMalloc((void**)&r, sizeof(double) * SIZE);
    cudaMalloc((void**)&r_old, sizeof(double) * SIZE);
    cudaMalloc((void**)&rh, sizeof(double) * SIZE);
    cudaMalloc((void**)&p, sizeof(double) * SIZE);
    cudaMalloc((void**)&pw, sizeof(double) * SIZE);
    cudaMalloc((void**)&t, sizeof(double) * SIZE);
    cudaMalloc((void**)&s, sizeof(double) * SIZE);
    cudaMalloc((void**)&v, sizeof(double) * SIZE);
    cudaMalloc((void**)&M, sizeof(double) * NNZ);

    cudaMemset((void*)r, 0, sizeof(double) * SIZE);
    cudaMemset((void*)r_old, 0, sizeof(double) * SIZE);
    cudaMemset((void*)rh, 0, sizeof(double) * SIZE);
    cudaMemset((void*)p, 0, sizeof(double) * SIZE);
    cudaMemset((void*)pw, 0, sizeof(double) * SIZE);
    cudaMemset((void*)t, 0, sizeof(double) * SIZE);
    cudaMemset((void*)s, 0, sizeof(double) * SIZE);
    cudaMemset((void*)v, 0, sizeof(double) * SIZE);
    cudaMemset((void*)M, 0, sizeof(double) * NNZ);

    double* phost = (double*)malloc(SIZE * sizeof(double));
    double* xhost = (double*)malloc(SIZE * sizeof(double));

    //    cudaMemcpy(xhost, x, (size_t)(SIZE * sizeof(double)), cudaMemcpyHostToDevice);
    for (int i = 0; i < SIZE; i++) {
        xhost[i] = 1.0;
    }
    cudaMemcpy(b, xhost, (size_t)(SIZE * sizeof(double)), cudaMemcpyHostToDevice);

    //====== Get handle to the CUBLAS context ========
    cublasStatus_t cublasStatus;
    if (cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUBLAS initialization error\n");
        return EXIT_FAILURE;
    }

    //============ initialize CUSPARSE ===================================================
    cusparseStatus_t cusparseStatus1, cusparseStatus2;
    cusparseStatus1 = cusparseCreate(&cusparseHandle);
    if (cusparseStatus1 != CUSPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "CUSPARSE initialization failed\n");
        return EXIT_FAILURE;
    }

    //============ create three matrix descriptors =======================================
    cusparseStatus1 = cusparseCreateMatDescr(&descrA);
    cusparseStatus2 = cusparseCreateMatDescr(&descrM);
    if ((cusparseStatus1 != CUSPARSE_STATUS_SUCCESS) || (cusparseStatus2 != CUSPARSE_STATUS_SUCCESS)) {
        fprintf(stderr, "CUSPARSE cusparseCreateMatDescr (coefficient matrix or preconditioner) error\n");
        return EXIT_FAILURE;
    }
    //==========create three matrix descriptors ===========================================
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descrM, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrM, CUSPARSE_INDEX_BASE_ZERO);

    //==========create the analysis info (for lower and upper triangular factors)==========
    //        cusparseCreateSolveAnalysisInfo(&info_l));
    //        cusparseCreateSolveAnalysisInfo(&info_u));
    //        cusparseSetMatFillMode(descrM, CUSPARSE_FILL_MODE_LOWER));
    //        cusparseSetMatDiagType(descrM, CUSPARSE_DIAG_TYPE_UNIT));
    //        cusparseDcsrsv_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, SIZE, NNZ,
    //        descrM, A,
    //                                                ArowIdx, AcolIdx, info_l));
    //        cudaThreadSynchronize());
    //
    //        cusparseSetMatFillMode(descrM, CUSPARSE_FILL_MODE_UPPER));
    //        cusparseSetMatDiagType(descrM, CUSPARSE_DIAG_TYPE_NON_UNIT));
    //        cusparseDcsrsv_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, SIZE, NNZ,
    //        descrM, A,
    //                                                ArowIdx, AcolIdx, info_u));
    //        cudaThreadSynchronize());
    //
    //        //==========Compute the lower and upper triangular factors using CUSPARSE csrilu0 routine
    //        //        int* MrowIdx = ArowIdx;
    //        //        int* McolIdx = AcolIdx;
    //        cusparseDcsrilu0(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, SIZE, descrA, M,
    //        ArowIdx,
    //                                         AcolIdx, info_l));
    //        cudaThreadSynchronize());

    //===========================Solution=====================================================
    double rho, rhop, beta, alpha, negalpha, omega, negomega, temp, temp2;
    double nrmr, nrmr0;
    double zero = 0.0;
    double one = 1.0;
    double mone = -1.0;

    // compute initial residual r0=b-Ax0 (using initial guess in x)
    cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, SIZE, SIZE, NNZ, &one, descrA, A, (int*)ArowIdx,
                   (int*)AcolIdx, x, &zero, r);
    cublasDscal(cublasHandle, SIZE, &mone, r, 1);
    cublasDaxpy(cublasHandle, SIZE, &one, b, 1, r, 1);
    // copy residual r into r^{\hat} and p
    cublasDcopy(cublasHandle, SIZE, r, 1, rh, 1);
    cublasDcopy(cublasHandle, SIZE, r, 1, p, 1);
    cublasDnrm2(cublasHandle, SIZE, rh, 1, &nrmr0);
    printf("gpu, init residual:norm %20.16f\n", nrmr0);

    for (int i = 0; i < max_iter;) {
        rhop = rho;
        cublasDdot(cublasHandle, SIZE, rh, 1, r, 1, &rho);

        if (i > 0) {
            beta = (rho / rhop) * (alpha / omega);
            negomega = -omega;
            cublasDaxpy(cublasHandle, SIZE, &negomega, v, 1, p, 1);
            cublasDscal(cublasHandle, SIZE, &beta, p, 1);
            cublasDaxpy(cublasHandle, SIZE, &one, r, 1, p, 1);
        }
        // preconditioning step (lower and upper triangular solve)

        cublasDcopy(cublasHandle, SIZE, p, 1, pw, 1);

        // cusparseSetMatFillMode(descrm,CUSPARSE_FILL_MODE_LOWER));
        // cusparseSetMatDiagType(descrm,CUSPARSE_DIAG_TYPE_UNIT));
        // cusparseDcsrsv_solve(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE,n,&one,descrm,vm,im,jm,info_l,p,t));

        // cusparseSetMatFillMode(descrm,CUSPARSE_FILL_MODE_UPPER));
        // cusparseSetMatDiagType(descrm,CUSPARSE_DIAG_TYPE_NON_UNIT));
        // cusparseDcsrsv_solve(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE,n,&one,descrm,vm,im,jm,info_u,t,pw));

        cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, SIZE, SIZE, NNZ, &one, descrA, A,
                       (int*)ArowIdx, (int*)AcolIdx, pw, &zero, v);

        cublasDdot(cublasHandle, SIZE, rh, 1, v, 1, &temp);
        alpha = rho / temp;
        negalpha = -(alpha);
        cublasDaxpy(cublasHandle, SIZE, &negalpha, v, 1, r, 1);
        cublasDaxpy(cublasHandle, SIZE, &alpha, pw, 1, x, 1);
        cublasDnrm2(cublasHandle, SIZE, r, 1, &nrmr);
        printf("alpha=%f, temp=%f, beta=%f, rhop=%f, rho=%f ", alpha, temp, beta, rhop, rho);

        if (nrmr < abs_res) {
            break;
        }

        // preconditioning step (lower and upper triangular solve)
        cublasDcopy(cublasHandle, SIZE, r, 1, s, 1);

        // cusparseSetMatFillMode(descrm,CUSPARSE_FILL_MODE_LOWER));
        // cusparseSetMatDiagType(descrm,CUSPARSE_DIAG_TYPE_UNIT));
        // cusparseDcsrsv_solve(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE,n,
        // &one,descrm,vm,im,jm,info_l,r,t));

        // cusparseSetMatFillMode(descrm,CUSPARSE_FILL_MODE_UPPER));
        // cusparseSetMatDiagType(descrm,CUSPARSE_DIAG_TYPE_NON_UNIT));
        // cusparseDcsrsv_solve(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE,n,
        // &one,descrm,vm,im,jm,info_u,t,s));

        cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, SIZE, SIZE, NNZ, &one, descrA, A,
                       (int*)ArowIdx, (int*)AcolIdx, s, &zero, t);
        cublasDdot(cublasHandle, SIZE, t, 1, r, 1, &temp);
        cublasDdot(cublasHandle, SIZE, t, 1, t, 1, &temp2);
        omega = temp / temp2;
        negomega = -(omega);
        cublasDaxpy(cublasHandle, SIZE, &omega, s, 1, x, 1);
        cublasDaxpy(cublasHandle, SIZE, &negomega, t, 1, r, 1);

        cublasDnrm2(cublasHandle, SIZE, r, 1, &nrmr);
        printf("iter=%d, tolerance=%f\n", i, nrmr);

        if (nrmr < abs_res) {
            i++;
            break;
        }

        i++;
    }

    cusparseDestroySolveAnalysisInfo(info_l);
    cusparseDestroySolveAnalysisInfo(info_u);
    return EXIT_SUCCESS;
}

}  // end namespace fsi
}  // end namespace chrono
