/*
 * SPHCudaUtils.h
 *
 *      Author: Arman Pazouki, Milad Rakhsha
 */
// ****************************************************************************
// This file contains miscellaneous macros and utilities used in the SPH code.
// ****************************************************************************
#ifndef CH_SPH_GENERAL_CU
#define CH_SPH_GENERAL_CU

// ----------------------------------------------------------------------------
// CUDA headers
// ----------------------------------------------------------------------------
#include "chrono_fsi/ChSphGeneral.cuh"

namespace chrono {
namespace fsi {

void CopyParams_NumberOfObjects(SimParams* paramsH, NumberOfObjects* numObjectsH) {
    cudaMemcpyToSymbolAsync(paramsD, paramsH, sizeof(SimParams));
    cudaMemcpyToSymbolAsync(numObjectsD, numObjectsH, sizeof(NumberOfObjects));
    cudaThreadSynchronize();
}

//--------------------------------------------------------------------------------------------------------------------------------
__global__ void calc_A_tensor(Real* A_tensor,
                              Real* G_tensor,
                              Real4* sortedPosRad,
                              Real4* sortedRhoPreMu,
                              Real* sumWij_inv,
                              uint* csrColInd,
                              uint* numContacts,
                              const int numAllMarkers,
                              volatile bool* isErrorD) {
    uint i_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_idx >= numAllMarkers) {
        return;
    }

    // Remember : we want to solve 6x6 system Bi*l=-[1 0 0 1 0 1]'
    // elements of matrix B depends on tensor A

    uint csrStartIdx = numContacts[i_idx];
    uint csrEndIdx = numContacts[i_idx + 1];
    Real3 posRadA = mR3(sortedPosRad[i_idx]);
    Real h_i = sortedPosRad[i_idx].w;
    Real m_i = pow(h_i * paramsD.MULT_INITSPACE, 3) * paramsD.rho0;
    Real sum_mW = 0;
    Real A_ijk[27] = {0.0};

    Real Gi[9] = {0.0};
    for (int i = 0; i < 9; i++)
        Gi[i] = G_tensor[i_idx * 9 + i];

    // get address in grid
    int3 gridPos = calcGridPos(posRadA);

    for (int count = csrStartIdx; count < csrEndIdx; count++) {
        int j = csrColInd[count];
        Real3 posRadB = mR3(sortedPosRad[j]);
        Real3 rij = Distance(posRadA, posRadB);
        Real d = length(rij);
        if (d > RESOLUTION_LENGTH_MULT * h_i || sortedRhoPreMu[j].w <= -2)
            continue;
        Real h_j = sortedPosRad[j].w;
        Real m_j = pow(h_j * paramsD.MULT_INITSPACE, 3) * paramsD.rho0;
        Real h_ij = 0.5 * (h_j + h_i);
        Real3 grad_ij = GradWh(rij, h_ij);
        Real V_j = sumWij_inv[j];
        Real com_part = 0;
        com_part = (Gi[0] * grad_ij.x + Gi[1] * grad_ij.y + Gi[2] * grad_ij.z) * V_j;
        A_ijk[0] += rij.x * rij.x * com_part;  // 111
        A_ijk[1] += rij.x * rij.y * com_part;  // 112
        A_ijk[2] += rij.x * rij.z * com_part;  // 113
        A_ijk[3] += rij.y * rij.x * com_part;  // 121
        A_ijk[4] += rij.y * rij.y * com_part;  // 122
        A_ijk[5] += rij.y * rij.z * com_part;  // 123
        A_ijk[6] += rij.z * rij.x * com_part;  // 131
        A_ijk[7] += rij.z * rij.y * com_part;  // 132
        A_ijk[8] += rij.z * rij.z * com_part;  // 133
        com_part = (Gi[3] * grad_ij.x + Gi[4] * grad_ij.y + Gi[5] * grad_ij.z) * V_j;
        A_ijk[9] += rij.x * rij.x * com_part;   // 211
        A_ijk[10] += rij.x * rij.y * com_part;  // 212
        A_ijk[11] += rij.x * rij.z * com_part;  // 213
        A_ijk[12] += rij.y * rij.x * com_part;  // 221
        A_ijk[13] += rij.y * rij.y * com_part;  // 222
        A_ijk[14] += rij.y * rij.z * com_part;  // 223
        A_ijk[15] += rij.z * rij.x * com_part;  // 231
        A_ijk[16] += rij.z * rij.y * com_part;  // 232
        A_ijk[17] += rij.z * rij.z * com_part;  // 233
        com_part = (Gi[6] * grad_ij.x + Gi[7] * grad_ij.y + Gi[8] * grad_ij.z) * V_j;
        A_ijk[18] += rij.x * rij.x * com_part;  // 311
        A_ijk[19] += rij.x * rij.y * com_part;  // 312
        A_ijk[20] += rij.x * rij.z * com_part;  // 313
        A_ijk[21] += rij.y * rij.x * com_part;  // 321
        A_ijk[22] += rij.y * rij.y * com_part;  // 322
        A_ijk[23] += rij.y * rij.z * com_part;  // 323
        A_ijk[24] += rij.z * rij.x * com_part;  // 331
        A_ijk[25] += rij.z * rij.y * com_part;  // 332
        A_ijk[26] += rij.z * rij.z * com_part;  // 333
    }

    for (int i = 0; i < 27; i++)
        A_tensor[i_idx * 9 + i] = A_ijk[i];

    //    printf("A_tensor[%d]= %f,%f,%f,%f,%f,%f,%f,%f,%f, %f,%f,%f,%f,%f,%f,%f,%f,%f, %f,%f,%f,%f,%f,%f,%f,%f,%f\n",
    //    i_idx,
    //           A_ijk[0], A_ijk[1], A_ijk[2], A_ijk[3], A_ijk[4], A_ijk[5], A_ijk[6], A_ijk[7], A_ijk[8], A_ijk[9 + 0],
    //           A_ijk[9 + 1], A_ijk[9 + 2], A_ijk[9 + 3], A_ijk[9 + 4], A_ijk[9 + 5], A_ijk[9 + 6], A_ijk[9 + 7],
    //           A_ijk[9 + 8], A_ijk[18 + 0], A_ijk[18 + 1], A_ijk[18 + 2], A_ijk[18 + 3], A_ijk[18 + 4], A_ijk[18 + 5],
    //           A_ijk[18 + 6], A_ijk[18 + 7], A_ijk[18 + 8]);
}
//--------------------------------------------------------------------------------------------------------------------------------
__global__ void calc_L_tensor(Real* A_tensor,
                              Real* L_tensor,
                              Real* G_tensor,
                              Real4* sortedPosRad,
                              Real4* sortedRhoPreMu,
                              Real* sumWij_inv,
                              uint* csrColInd,
                              uint* numContacts,
                              const int numAllMarkers,
                              volatile bool* isErrorD) {
    uint i_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_idx >= numAllMarkers) {
        return;
    }

    if (sortedRhoPreMu[i_idx].w != -1) {
        return;
    }

    // Remember : we want to solve 6x6 system Bi*l=-[1 0 0 1 0 1]'
    // elements of matrix B depends on tensor A

    uint csrStartIdx = numContacts[i_idx];
    uint csrEndIdx = numContacts[i_idx + 1];
    Real3 posRadA = mR3(sortedPosRad[i_idx]);
    Real h_i = sortedPosRad[i_idx].w;
    Real m_i = pow(h_i * paramsD.MULT_INITSPACE, 3) * paramsD.rho0;
    Real B[36] = {0.0};

    Real Gi[9] = {0.0};
    for (int i = 0; i < 9; i++)
        Gi[i] = G_tensor[i_idx * 9 + i];

    Real A_ijk[27] = {0.0};
    for (int i = 0; i < 27; i++)
        A_ijk[i] = A_tensor[i_idx * 27 + i];

    // get address in grid
    int3 gridPos = calcGridPos(posRadA);

    for (int count = csrStartIdx; count < csrEndIdx; count++) {
        int j = csrColInd[count];
        Real3 posRadB = mR3(sortedPosRad[j]);
        Real3 rij = Distance(posRadA, posRadB);
        Real d = length(rij);
        if (d > RESOLUTION_LENGTH_MULT * h_i || sortedRhoPreMu[j].w <= -2)
            continue;
        Real3 eij = rij / d;

        Real h_j = sortedPosRad[j].w;
        Real m_j = pow(h_j * paramsD.MULT_INITSPACE, 3) * paramsD.rho0;
        Real h_ij = 0.5 * (h_j + h_i);
        Real3 grad_ij = GradWh(rij, h_ij);
        Real V_j = sumWij_inv[j];
        Real com_part = 0;
        // mn=11

        Real XX = (eij.x * grad_ij.x);
        Real XY = (eij.x * grad_ij.y + eij.y * grad_ij.x);
        Real XZ = (eij.x * grad_ij.z + eij.z * grad_ij.x);
        Real YY = (eij.y * grad_ij.y);
        Real YZ = (eij.y * grad_ij.z + eij.z * grad_ij.y);
        Real ZZ = (eij.z * grad_ij.z);

        com_part = (A_ijk[0] * eij.x + A_ijk[9] * eij.y + A_ijk[18] * eij.z + rij.x * eij.x) * V_j;
        B[6 * 0 + 0] += com_part * XX;  // 11
        B[6 * 0 + 1] += com_part * XY;  // 12
        B[6 * 0 + 2] += com_part * XZ;  // 13
        B[6 * 0 + 3] += com_part * YY;  // 14
        B[6 * 0 + 4] += com_part * YZ;  // 15
        B[6 * 0 + 5] += com_part * ZZ;  // 15
        // mn=12
        com_part = (A_ijk[1] * eij.x + A_ijk[10] * eij.y + A_ijk[19] * eij.z + rij.x * eij.y) * V_j;
        B[6 * 1 + 0] += com_part * XX;  // 21
        B[6 * 1 + 1] += com_part * XY;  // 22
        B[6 * 1 + 2] += com_part * XZ;  // 23
        B[6 * 1 + 3] += com_part * YY;  // 24
        B[6 * 1 + 4] += com_part * YZ;  // 25
        B[6 * 1 + 5] += com_part * ZZ;  // 25

        // mn=13
        com_part = (A_ijk[2] * eij.x + A_ijk[11] * eij.y + A_ijk[20] * eij.z + rij.x * eij.z) * V_j;
        B[6 * 2 + 0] += com_part * XX;  // 31
        B[6 * 2 + 1] += com_part * XY;  // 32
        B[6 * 2 + 2] += com_part * XZ;  // 33
        B[6 * 2 + 3] += com_part * YY;  // 34
        B[6 * 2 + 4] += com_part * YZ;  // 35
        B[6 * 2 + 5] += com_part * ZZ;  // 36

        // Note that we skip mn=21 since it is similar to mn=12
        // mn=22
        com_part = (A_ijk[4] * eij.x + A_ijk[13] * eij.y + A_ijk[22] * eij.z + rij.y * eij.y) * V_j;
        B[6 * 3 + 0] += com_part * XX;  // 41
        B[6 * 3 + 1] += com_part * XY;  // 42
        B[6 * 3 + 2] += com_part * XZ;  // 43
        B[6 * 3 + 3] += com_part * YY;  // 44
        B[6 * 3 + 4] += com_part * YZ;  // 45
        B[6 * 3 + 5] += com_part * ZZ;  // 46

        // mn=23
        com_part = (A_ijk[5] * eij.x + A_ijk[14] * eij.y + A_ijk[23] * eij.z + rij.y * eij.z) * V_j;
        B[6 * 4 + 0] += com_part * XX;  // 51
        B[6 * 4 + 1] += com_part * XY;  // 52
        B[6 * 4 + 2] += com_part * XZ;  // 53
        B[6 * 4 + 3] += com_part * YY;  // 54
        B[6 * 4 + 4] += com_part * YZ;  // 55
        B[6 * 4 + 5] += com_part * ZZ;  // 56
        // mn=33
        com_part = (A_ijk[8] * eij.x + A_ijk[17] * eij.y + A_ijk[26] * eij.z + rij.z * eij.z) * V_j;
        B[6 * 5 + 0] += com_part * XX;  // 61
        B[6 * 5 + 1] += com_part * XY;  // 62
        B[6 * 5 + 2] += com_part * XZ;  // 63
        B[6 * 5 + 3] += com_part * YY;  // 64
        B[6 * 5 + 4] += com_part * YZ;  // 65
        B[6 * 5 + 5] += com_part * ZZ;  // 66
    }

    inv6xdelta_mn(B, &L_tensor[6 * i_idx]);
    //    printf("L[%d]=%f,%f,%f,%f,%f,%f\n", i_idx, L_tensor[6 * i_idx + 0], L_tensor[6 * i_idx + 1],
    //           L_tensor[6 * i_idx + 2], L_tensor[6 * i_idx + 3], L_tensor[6 * i_idx + 4], L_tensor[6 * i_idx + 5]);
    //    for (uint j = 0; j < 6; j++)
    //        printf("B[%d,[%d]]=%f,%f,%f,%f,%f,%f\n", i_idx, j, B[6 * j + 0], B[6 * j + 1], B[6 * j + 2], B[6 * j + 3],
    //               B[6 * j + 4], B[6 * j + 5]);

    //    printf(
    //        "B[%d]=%f,%f,%f,%f,%f,%f,%f,%f,%f, %f,%f,%f,%f,%f,%f,%f,%f,%f, %f,%f,%f,%f,%f,%f,%f,%f,%f, "
    //        "%f,%f,%f,%f,%f,%f,%f,%f,%f, --- %f,%f,%f,%f,%f,%f\n",
    //        i_idx, B[0], B[1], B[2], B[3], B[4], B[5], B[6], B[7], B[8], B[9], B[10], B[11], B[12], B[13], B[14],
    //        B[15], B[16], B[17], B[18], B[19], B[20], B[21], B[22], B[23], B[24], B[25], B[26], B[27], B[28],
    //        B[29], B[30], B[31], B[32], B[33], 1, B[35], L_tensor[6 * i_idx + 0], L_tensor[6 * i_idx + 1],
    //        L_tensor[6 * i_idx
    //        + 2], L_tensor[6 * i_idx + 3], L_tensor[6 * i_idx + 4], L_tensor[6 * i_idx + 5]);
}

//--------------------------------------------------------------------------------------------------------------------------------
__global__ void calcRho_kernel(Real4* sortedPosRad,
                               Real4* sortedRhoPreMu,
                               Real* sumWij_inv,
                               uint* cellStart,
                               uint* cellEnd,
                               uint* mynumContact,
                               const int numAllMarkers,
                               volatile bool* isErrorD) {
    uint i_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_idx >= numAllMarkers) {
        return;
    }

    if (sortedRhoPreMu[i_idx].w == -2) {
        mynumContact[i_idx] = 1;
        return;
    }

    Real3 posRadA = mR3(sortedPosRad[i_idx]);
    Real h_i = sortedPosRad[i_idx].w;
    Real m_i = pow((h_i * paramsD.MULT_INITSPACE), 3) * paramsD.rho0;
    //    printf("paramsD.MULT_INITSPACE=%f, h_i,m_i ", paramsD.MULT_INITSPACE, h_i, m_i);

    Real sum_mW = 0;
    Real sum_W = 0.0;
    uint mcon = 1;
    // get address in grid
    int3 gridPos = calcGridPos(posRadA);
    //    printf("h_i=%f\t", h_i);
    for (int z = -1; z <= 1; z++) {
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                int3 neighbourPos = gridPos + mI3(x, y, z);
                uint gridHash = calcGridHash(neighbourPos);
                uint startIndex = cellStart[gridHash];
                if (startIndex != 0xffffffff) {
                    uint endIndex = cellEnd[gridHash];
                    for (uint j = startIndex; j < endIndex; j++) {
                        Real3 posRadB = mR3(sortedPosRad[j]);
                        Real3 dist3 = Distance(posRadA, posRadB);
                        Real d = length(dist3);

                        if (d > RESOLUTION_LENGTH_MULT * h_i || sortedRhoPreMu[j].w <= -2)
                            continue;
                        //                        if ((sortedRhoPreMu[j].w == -1.0 || sortedRhoPreMu[i_idx].w == -1.0)
                        //                        && i_idx != j)
                        if (i_idx != j)
                            mcon++;
                        Real h_j = sortedPosRad[j].w;
                        Real m_j = pow(h_j * paramsD.MULT_INITSPACE, 3) * paramsD.rho0;
                        //                        Real W3 = W3h(d, 0.5 * (h_j + h_i));
                        Real W3 = 0.5 * (W3h(d, h_i) + W3h(d, h_j));
                        sum_mW += m_j * W3;
                        sum_W += W3;
                    }
                }
            }
        }
    }
    mynumContact[i_idx] = mcon;
    // Adding neighbor contribution is done!
    sumWij_inv[i_idx] = m_i / sum_mW;
    //    printf("%f, ", m_i / sum_mW);
    sortedRhoPreMu[i_idx].x = sum_mW;

    if ((sortedRhoPreMu[i_idx].x > 2 * paramsD.rho0 || sortedRhoPreMu[i_idx].x < 0) && sortedRhoPreMu[i_idx].w == -1)
        printf("(calcRho_kernel)too large/small density marker %d, rho=%f, sum_W=%f, m_i=%f\n", i_idx,
               sortedRhoPreMu[i_idx].x, sum_W, m_i);
}

//--------------------------------------------------------------------------------------------------------------------------------
__global__ void calcNormalizedRho_kernel(Real4* sortedPosRad,  // input: sorted positions
                                         Real3* sortedVelMas,
                                         Real4* sortedRhoPreMu,
                                         Real* sumWij_inv,
                                         Real* G_i,
                                         Real3* normals,
                                         Real* Color,
                                         uint* cellStart,
                                         uint* cellEnd,
                                         const int numAllMarkers,
                                         volatile bool* isErrorD) {
    uint i_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_idx >= numAllMarkers || sortedRhoPreMu[i_idx].w <= -2) {
        return;
    }
    //    Real3 gravity = paramsD.gravity;
    Real RHO_0 = paramsD.rho0;
    Real IncompressibilityFactor = paramsD.IncompressibilityFactor;
    //    dxi_over_Vi[i_idx] = 1e10;
    if (sortedRhoPreMu[i_idx].w == -2)
        return;
    Real3 posRadA = mR3(sortedPosRad[i_idx]);
    Real h_i = sortedPosRad[i_idx].w;
    //    Real m_i = pow(h_i * paramsD.MULT_INITSPACE, 3) * paramsD.rho0;
    Real sum_mW = 0;
    Real sum_W_sumWij_inv = 0;
    Real C = 0;
    // get address in grid
    int3 gridPos = calcGridPos(posRadA);

    // This is the elements of inverse of G
    Real mGi[9] = {0.0};
    Real theta_i = sortedRhoPreMu[i_idx].w + 1;
    if (theta_i > 1)
        theta_i = 1;
    Real3 mynormals = mR3(0.0);

    //  /// if (gridPos.x == paramsD.gridSize.x-1) printf("****aha %d %d\n", gridPos.x, paramsD.gridSize.x);
    //
    // examine neighbouring cells
    for (int z = -1; z <= 1; z++) {
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                int3 neighbourPos = gridPos + mI3(x, y, z);
                uint gridHash = calcGridHash(neighbourPos);
                // get start of bucket for this cell50
                uint startIndex = cellStart[gridHash];
                if (startIndex != 0xffffffff) {  // cell is not empty
                                                 // iterate over particles in this cell
                    uint endIndex = cellEnd[gridHash];

                    for (uint j = startIndex; j < endIndex; j++) {
                        Real3 posRadB = mR3(sortedPosRad[j]);
                        Real3 dist3 = Distance(posRadA, posRadB);
                        Real3 dv3 = Distance(sortedVelMas[i_idx], sortedVelMas[j]);
                        Real d = length(dist3);
                        Real h_j = sortedPosRad[j].w;
                        Real m_j = pow(h_j * 1, 3) * paramsD.rho0;
                        C += m_j * Color[i_idx] / sortedRhoPreMu[i_idx].x * W3h(d, 0.5 * (h_j + h_i));

                        if (d > RESOLUTION_LENGTH_MULT * h_i || sortedRhoPreMu[j].w <= -2)
                            continue;
                        Real V_j = sumWij_inv[j];

                        Real h_ij = 0.5 * (h_j + h_i);
                        Real W3 = W3h(d, h_ij);
                        Real3 grad_i_wij = GradWh(dist3, h_ij);
                        Real theta_j = sortedRhoPreMu[j].w + 1;
                        if (theta_j > 1)
                            theta_j = 1;

                        if (sortedRhoPreMu[i_idx].w == -3 && sortedRhoPreMu[j].w == -3)
                            mynormals += grad_i_wij * V_j;
                        if (sortedRhoPreMu[i_idx].w != -3)
                            mynormals += (theta_j - theta_i) * grad_i_wij * V_j;

                        mGi[0] -= dist3.x * grad_i_wij.x * V_j;
                        mGi[1] -= dist3.x * grad_i_wij.y * V_j;
                        mGi[2] -= dist3.x * grad_i_wij.z * V_j;
                        mGi[3] -= dist3.y * grad_i_wij.x * V_j;
                        mGi[4] -= dist3.y * grad_i_wij.y * V_j;
                        mGi[5] -= dist3.y * grad_i_wij.z * V_j;
                        mGi[6] -= dist3.z * grad_i_wij.x * V_j;
                        mGi[7] -= dist3.z * grad_i_wij.y * V_j;
                        mGi[8] -= dist3.z * grad_i_wij.z * V_j;
                        sum_mW += m_j * W3;
                        sum_W_sumWij_inv += sumWij_inv[j] * W3;
                    }
                }
            }
        }
    }

    normals[i_idx] = mynormals;

    if (length(mynormals) > EPSILON)
        normals[i_idx] = mynormals / length(mynormals);

    Real Det = (mGi[0] * mGi[4] * mGi[8] - mGi[0] * mGi[5] * mGi[7] - mGi[1] * mGi[3] * mGi[8] +
                mGi[1] * mGi[5] * mGi[6] + mGi[2] * mGi[3] * mGi[7] - mGi[2] * mGi[4] * mGi[6]);
    G_i[i_idx * 9 + 0] = (mGi[4] * mGi[8] - mGi[5] * mGi[7]) / Det;
    G_i[i_idx * 9 + 1] = -(mGi[1] * mGi[8] - mGi[2] * mGi[7]) / Det;
    G_i[i_idx * 9 + 2] = (mGi[1] * mGi[5] - mGi[2] * mGi[4]) / Det;
    G_i[i_idx * 9 + 3] = -(mGi[3] * mGi[8] - mGi[5] * mGi[6]) / Det;
    G_i[i_idx * 9 + 4] = (mGi[0] * mGi[8] - mGi[2] * mGi[6]) / Det;
    G_i[i_idx * 9 + 5] = -(mGi[0] * mGi[5] - mGi[2] * mGi[3]) / Det;
    G_i[i_idx * 9 + 6] = (mGi[3] * mGi[7] - mGi[4] * mGi[6]) / Det;
    G_i[i_idx * 9 + 7] = -(mGi[0] * mGi[7] - mGi[1] * mGi[6]) / Det;
    G_i[i_idx * 9 + 8] = (mGi[0] * mGi[4] - mGi[1] * mGi[3]) / Det;

    //    printf("update G[%d]= %f,%f,%f  %f,%f,%f, %f,%f,%f\n", i_idx, G_i[i_idx * 9 + 0], G_i[i_idx * 9 + 1],
    //           G_i[i_idx * 9 + 2], G_i[i_idx * 9 + 3], G_i[i_idx * 9 + 4], G_i[i_idx * 9 + 5], G_i[i_idx * 9 + 6],
    //           G_i[i_idx * 9 + 7], G_i[i_idx * 9 + 8]);

    if (sortedRhoPreMu[i_idx].x > RHO_0)
        IncompressibilityFactor = 1;

    //    sortedRhoPreMu[i_idx].x = (sum_mW / sum_W_sumWij_inv - RHO_0) * IncompressibilityFactor + RHO_0;

    //    if (sortedRhoPreMu[i_idx].x < EPSILON)
    if ((sortedRhoPreMu[i_idx].x > 5 * RHO_0 || sortedRhoPreMu[i_idx].x < RHO_0 / 5) && sortedRhoPreMu[i_idx].w == -1)
        printf(
            "calcNormalizedRho_kernel-- sortedRhoPreMu[i_idx].w=%f, h=%f, sum_mW=%f, "
            "sum_W_sumWij_inv=%.4e, sortedRhoPreMu[i_idx].x=%.4e\n",
            sortedRhoPreMu[i_idx].w, sortedPosRad[i_idx].w, sum_mW, sum_W_sumWij_inv, sortedRhoPreMu[i_idx].x);

    //    sortedRhoPreMu[i_idx].x = (sum_mW - RHO_0) * IncompressibilityFactor + RHO_0;
    //
    //    if (sortedRhoPreMu[i_idx].x < EPSILON) {
    //        printf("My density is %f, index= %d\n", sortedRhoPreMu[i_idx].x, i_idx);
    //
    //        printf("My position = [%f %f %f]\n", sortedPosRad[i_idx].x, sortedPosRad[i_idx].y,
    //        sortedPosRad[i_idx].z);
    //
    //        *isErrorD = true;
    //        return;
    //    }

    //  if (sortedRhoPreMu[i_idx].w > -1) {
    //    sortedRhoPreMu[i_idx].x = RHO_0;
    //  }
}

//--------------------------------------------------------------------------------------------------------------------------------
__global__ void calcNormalizedRho_Gi_fillInMatrixIndices(Real4* sortedPosRad,  // input: sorted positions
                                                         Real3* sortedVelMas,
                                                         Real4* sortedRhoPreMu,
                                                         Real* sumWij_inv,
                                                         Real* G_i,
                                                         Real3* normals,
                                                         uint* csrColInd,
                                                         uint* numContacts,
                                                         uint* cellStart,
                                                         uint* cellEnd,
                                                         const int numAllMarkers,
                                                         volatile bool* isErrorD) {
    uint i_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_idx >= numAllMarkers) {
        return;
    }
    Real RHO_0 = paramsD.rho0;
    uint csrStartIdx = numContacts[i_idx] + 1;  // Reserve the starting index for the A_ii
    uint csrEndIdx = numContacts[i_idx + 1];
    Real3 posRadA = mR3(sortedPosRad[i_idx]);
    Real h_i = sortedPosRad[i_idx].w;
    Real sum_mW = 0;
    Real sum_W_sumWij_inv = 0;
    // get address in grid
    int3 gridPos = calcGridPos(posRadA);

    csrColInd[csrStartIdx - 1] = i_idx;
    uint nextCol = csrStartIdx;

    if (sortedRhoPreMu[i_idx].w == -2)
        return;

    Real theta_i = sortedRhoPreMu[i_idx].w + 1;
    if (theta_i > 1)
        theta_i = 1;

    Real3 mynormals = mR3(0.0);
    // This is the elements of inverse of G
    Real mGi[9] = {0.0};
    // examine neighbouring cells
    for (int z = -1; z <= 1; z++) {
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                int3 neighbourPos = gridPos + mI3(x, y, z);
                uint gridHash = calcGridHash(neighbourPos);
                // get start of bucket for this cell50
                uint startIndex = cellStart[gridHash];
                if (startIndex != 0xffffffff) {  // cell is not empty
                                                 // iterate over particles in this cell
                    uint endIndex = cellEnd[gridHash];

                    for (uint j = startIndex; j < endIndex; j++) {
                        Real3 posRadB = mR3(sortedPosRad[j]);
                        Real3 rij = Distance(posRadA, posRadB);
                        Real3 dv3 = Distance(sortedVelMas[i_idx], sortedVelMas[j]);
                        Real d = length(rij);
                        Real h_j = sortedPosRad[j].w;
                        Real m_j = pow(h_j * paramsD.MULT_INITSPACE, 3) * paramsD.rho0;
                        Real h_ij = 0.5 * (h_j + h_i);
                        //                        Real W3 = W3h(d, h_ij);
                        //                        Real3 grad_i_wij = GradWh(rij, h_ij);
                        Real W3 = 0.5 * (W3h(d, h_i) + W3h(d, h_j));
                        Real3 grad_i_wij = 0.5 * (GradWh(rij, h_i) + GradWh(rij, h_j));

                        Real V_j = sumWij_inv[j];

                        if (d > RESOLUTION_LENGTH_MULT * h_i || sortedRhoPreMu[j].w <= -2)
                            continue;
                        //                        if (i_idx != j && ((sortedRhoPreMu[i_idx].w == -1.0) ||
                        //                        (sortedRhoPreMu[j].w == -1.0))) {
                        if (i_idx != j) {
                            csrColInd[nextCol] = j;
                            nextCol++;
                        }

                        Real theta_j = sortedRhoPreMu[j].w + 1;
                        if (theta_j > 1)
                            theta_j = 1;

                        if (sortedRhoPreMu[i_idx].w == -3 && sortedRhoPreMu[j].w == -3)
                            mynormals += grad_i_wij * V_j;
                        if (sortedRhoPreMu[i_idx].w != -3)
                            mynormals += (theta_j - theta_i) * grad_i_wij * V_j;

                        mGi[0] -= rij.x * grad_i_wij.x * V_j;
                        mGi[1] -= rij.x * grad_i_wij.y * V_j;
                        mGi[2] -= rij.x * grad_i_wij.z * V_j;
                        mGi[3] -= rij.y * grad_i_wij.x * V_j;
                        mGi[4] -= rij.y * grad_i_wij.y * V_j;
                        mGi[5] -= rij.y * grad_i_wij.z * V_j;
                        mGi[6] -= rij.z * grad_i_wij.x * V_j;
                        mGi[7] -= rij.z * grad_i_wij.y * V_j;
                        mGi[8] -= rij.z * grad_i_wij.z * V_j;
                        //                        sum_mW += m_j * sumWij_inv[j];
                        sum_mW += sortedRhoPreMu[j].x * W3 * V_j;
                        //                        sum_W_sumWij_inv += W3 * sumWij_inv[j];
                    }
                }
            }
        }
    }

    normals[i_idx] = mynormals;

    if (length(mynormals) > EPSILON)
        normals[i_idx] = mynormals / length(mynormals);

    if (sortedRhoPreMu[i_idx].w == -3)
        normals[i_idx] *= -1;
    //    if (sortedRhoPreMu[i_idx].w == -3)
    //        printf("position=%f,%f,%f normals= %f,%f,%f, numContacts=%d\n", sortedPosRad[i_idx].x,
    //        sortedPosRad[i_idx].y,
    //               sortedPosRad[i_idx].z, normals[i_idx].x, normals[i_idx].y, normals[i_idx].z, csrEndIdx -
    //               csrStartIdx);

    Real Det = (mGi[0] * mGi[4] * mGi[8] - mGi[0] * mGi[5] * mGi[7] - mGi[1] * mGi[3] * mGi[8] +
                mGi[1] * mGi[5] * mGi[6] + mGi[2] * mGi[3] * mGi[7] - mGi[2] * mGi[4] * mGi[6]);
    if (abs(Det) < 1e-6 && sortedRhoPreMu[i_idx].w != -3) {
        printf("Gi,");
        for (int i = 0; i < 9; i++)
            G_i[i_idx * 9 + i] = 0.0;
        //        G_i[i_idx * 9 + 0] = 1;
        //        G_i[i_idx * 9 + 4] = 1;
        //        G_i[i_idx * 9 + 8] = 1;
    } else {
        G_i[i_idx * 9 + 0] = (mGi[4] * mGi[8] - mGi[5] * mGi[7]) / Det;
        G_i[i_idx * 9 + 1] = -(mGi[1] * mGi[8] - mGi[2] * mGi[7]) / Det;
        G_i[i_idx * 9 + 2] = (mGi[1] * mGi[5] - mGi[2] * mGi[4]) / Det;
        G_i[i_idx * 9 + 3] = -(mGi[3] * mGi[8] - mGi[5] * mGi[6]) / Det;
        G_i[i_idx * 9 + 4] = (mGi[0] * mGi[8] - mGi[2] * mGi[6]) / Det;
        G_i[i_idx * 9 + 5] = -(mGi[0] * mGi[5] - mGi[2] * mGi[3]) / Det;
        G_i[i_idx * 9 + 6] = (mGi[3] * mGi[7] - mGi[4] * mGi[6]) / Det;
        G_i[i_idx * 9 + 7] = -(mGi[0] * mGi[7] - mGi[1] * mGi[6]) / Det;
        G_i[i_idx * 9 + 8] = (mGi[0] * mGi[4] - mGi[1] * mGi[3]) / Det;
    }
    //    sortedRhoPreMu[i_idx].x = sum_mW / sum_W_sumWij_inv;
    //    sortedRhoPreMu[i_idx].x = sum_mW;

    if ((sortedRhoPreMu[i_idx].x > 5 * RHO_0 || sortedRhoPreMu[i_idx].x < RHO_0 / 5) && sortedRhoPreMu[i_idx].w > -2)
        printf(
            "calcNormalizedRho_kernel-- sortedRhoPreMu[i_idx].w=%f, h=%f, sum_mW=%f, "
            "sum_W_sumWij_inv=%.4e, sortedRhoPreMu[i_idx].x=%.4e\n",
            sortedRhoPreMu[i_idx].w, sortedPosRad[i_idx].w, sum_mW, sum_W_sumWij_inv, sortedRhoPreMu[i_idx].x);
}
//--------------------------------------------------------------------------------------------------------------------------------
__global__ void Function_Gradient_Laplacian_Operator(Real4* sortedPosRad,  // input: sorted positions
                                                     Real3* sortedVelMas,
                                                     Real4* sortedRhoPreMu,
                                                     Real* sumWij_inv,
                                                     Real* G_tensor,
                                                     Real* L_tensor,

                                                     Real* A_L,   /// Laplacian Operator matrix
                                                     Real3* A_G,  /// Gradient Operator matrix
                                                     Real* A_f,   /// Function Operator matrix
                                                     /// A_L, A_G are in system level;
                                                     /// A_G* p gives gradp, A_L*p gives Delta^2p
                                                     uint* csrColInd,
                                                     uint* numContacts,
                                                     const int numAllMarkers,
                                                     volatile bool* isErrorD) {
    uint i_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_idx >= numAllMarkers)
        return;

    if (sortedRhoPreMu[i_idx].w <= -2)
        return;

    Real RHO_0 = paramsD.rho0;
    uint csrStartIdx = numContacts[i_idx];
    uint csrEndIdx = numContacts[i_idx + 1];
    Real3 posRadA = mR3(sortedPosRad[i_idx]);
    Real h_i = sortedPosRad[i_idx].w;
    // get address in grid
    int3 gridPos = calcGridPos(posRadA);
    // This is the elements of inverse of G
    Real mGi[9] = {0.0};
    Real Li[6] = {0.0};
    Real3 LaplacainVi = mR3(0.0);
    Real NormGi = 0;
    Real NormLi = 0;

    for (int i = 0; i < 9; i++) {
        mGi[i] = G_tensor[i_idx * 9 + i];
        NormGi += abs(mGi[i]);
    }
    for (int i = 0; i < 6; i++) {
        Li[i] = L_tensor[i_idx * 6 + i];
        NormLi += abs(Li[i]);
    }

    Real V_i = sumWij_inv[i_idx];
    Real m_i = pow(h_i * paramsD.MULT_INITSPACE, 3) * paramsD.rho0;
    Real rhoi = sortedRhoPreMu[i_idx].x;
    for (int count = csrStartIdx; count < csrEndIdx; count++) {
        int j = csrColInd[count];
        Real3 posRadB = mR3(sortedPosRad[j]);
        Real3 rij = Distance(posRadA, posRadB);
        Real d = length(rij);
        Real3 eij = rij / d;
        Real h_j = sortedPosRad[j].w;
        Real m_j = pow(h_j * paramsD.MULT_INITSPACE, 3) * paramsD.rho0;
        Real h_ij = 0.5 * (h_j + h_i);
        //        Real W3 = W3h(d, h_ij);
        //        Real3 grad_i_wij = GradWh(rij, h_ij);
        Real W3 = 0.5 * (W3h(d, h_i) + W3h(d, h_j));
        Real3 grad_i_wij = 0.5 * (GradWh(rij, h_i) + GradWh(rij, h_j));

        Real V_j = sumWij_inv[j];
        A_f[count] = V_j * W3;
        if (paramsD.Conservative_Form || abs(rhoi - RHO_0) > 0.2 * RHO_0) {
            //            Real3 comm = 1 / V_i * (V_j * V_j + V_i * V_i) / (rhoi + sortedRhoPreMu[j].x) * grad_i_wij;
            //            A_G[count] = rhoi * comm;
            //            A_G[csrStartIdx] += sortedRhoPreMu[j].x * comm;
            Real3 comm = m_j * rhoi * grad_i_wij;
            A_G[count] = 1 / (sortedRhoPreMu[j].x * sortedRhoPreMu[j].x) * comm;
            A_G[csrStartIdx] += 1 / (rhoi * rhoi) * comm;

            //            Real Coeff = V_j;
            //            A_G[count].x = Coeff * (grad_i_wij.x * mGi[0] + grad_i_wij.y * mGi[1] + grad_i_wij.z *
            //            mGi[2]); A_G[count].y = Coeff * (grad_i_wij.x * mGi[3] + grad_i_wij.y * mGi[4] + grad_i_wij.z
            //            * mGi[5]); A_G[count].z = Coeff * (grad_i_wij.x * mGi[6] + grad_i_wij.y * mGi[7] +
            //            grad_i_wij.z * mGi[8]); A_G[csrStartIdx].x += Coeff * (grad_i_wij.x * mGi[0] + grad_i_wij.y *
            //            mGi[1] + grad_i_wij.z * mGi[2]); A_G[csrStartIdx].y += Coeff * (grad_i_wij.x * mGi[3] +
            //            grad_i_wij.y * mGi[4] + grad_i_wij.z * mGi[5]); A_G[csrStartIdx].z += Coeff * (grad_i_wij.x *
            //            mGi[6] + grad_i_wij.y * mGi[7] + grad_i_wij.z * mGi[8]);

        } else {
            Real Coeff = V_j;
            A_G[count].x = Coeff * (grad_i_wij.x * mGi[0] + grad_i_wij.y * mGi[1] + grad_i_wij.z * mGi[2]);
            A_G[count].y = Coeff * (grad_i_wij.x * mGi[3] + grad_i_wij.y * mGi[4] + grad_i_wij.z * mGi[5]);
            A_G[count].z = Coeff * (grad_i_wij.x * mGi[6] + grad_i_wij.y * mGi[7] + grad_i_wij.z * mGi[8]);
            A_G[csrStartIdx].x -= Coeff * (grad_i_wij.x * mGi[0] + grad_i_wij.y * mGi[1] + grad_i_wij.z * mGi[2]);
            A_G[csrStartIdx].y -= Coeff * (grad_i_wij.x * mGi[3] + grad_i_wij.y * mGi[4] + grad_i_wij.z * mGi[5]);
            A_G[csrStartIdx].z -= Coeff * (grad_i_wij.x * mGi[6] + grad_i_wij.y * mGi[7] + grad_i_wij.z * mGi[8]);
        }
    }

    for (int count = csrStartIdx; count < csrEndIdx; count++) {
        int j = csrColInd[count];
        Real3 posRadB = mR3(sortedPosRad[j]);
        Real3 rij = Distance(posRadA, posRadB);
        Real d = length(rij);
        Real3 eij = rij / d;
        Real h_j = sortedPosRad[j].w;
        Real m_j = pow(h_j * paramsD.MULT_INITSPACE, 3) * paramsD.rho0;
        Real h_ij = 0.5 * (h_j + h_i);
        Real W3 = W3h(d, h_ij);
        Real3 grad_ij = GradWh(rij, h_ij);
        Real V_j = sumWij_inv[j];
        //
        if (paramsD.Conservative_Form || abs(rhoi - RHO_0) > 0.2 * RHO_0) {
            Real comm = 2 / V_i * (V_j * V_j + V_i * V_i) * dot(rij, grad_ij) /
                        (d * d + h_ij * h_ij * paramsD.epsMinMarkersDis);
            A_L[count] = -comm;        // j
            A_L[csrStartIdx] += comm;  // i
            //            Real comm = 2 / rhoi * m_j * dot(rij, grad_ij) / (d * d + h_ij * h_ij *
            //            paramsD.epsMinMarkersDis);
            //            A_L[count] = -comm;        // j
            //            A_L[csrStartIdx] += comm;  // i

        } else {
            Real commonterm = 2 / V_j * (V_j * V_j + V_i * V_i) *
                              (Li[0] * eij.x * grad_ij.x + Li[1] * eij.x * grad_ij.y + Li[2] * eij.x * grad_ij.z +
                               Li[1] * eij.y * grad_ij.x + Li[3] * eij.y * grad_ij.y + Li[4] * eij.y * grad_ij.z +
                               Li[2] * eij.z * grad_ij.x + Li[4] * eij.z * grad_ij.y + Li[5] * eij.z * grad_ij.z);

            A_L[count] -= commonterm / (d + h_ij * paramsD.epsMinMarkersDis);        // j
            A_L[csrStartIdx] += commonterm / (d + h_ij * paramsD.epsMinMarkersDis);  // i

            for (int count_in = csrStartIdx; count_in < csrEndIdx; count_in++) {
                A_L[count_in] -= commonterm * dot(A_G[count_in], eij);  // k
            }
        }
    }
}
//--------------------------------------------------------------------------------------------------------------------------------
__global__ void Jacobi_SOR_Iter(Real4* sortedRhoPreMu,
                                Real* A_Matrix,
                                Real3* V_old,
                                Real3* V_new,
                                Real3* b3vec,
                                Real* q_old,  // q=p^(n+1)-p^n
                                Real* q_new,  // q=p^(n+1)-p^n
                                Real* b1vec,
                                const uint* csrColInd,
                                const uint* numContacts,
                                int numAllMarkers,
                                bool _3dvector,
                                volatile bool* isErrorD) {
    uint i_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_idx >= numAllMarkers) {
        return;
    }
    uint startIdx = numContacts[i_idx] + 1;  // Reserve the starting index for the A_ii
    uint endIdx = numContacts[i_idx + 1];

    if (_3dvector) {
        Real3 aij_vj = mR3(0.0);
        for (int myIdx = startIdx; myIdx < endIdx; myIdx++) {
            aij_vj += A_Matrix[myIdx] * V_old[csrColInd[myIdx]];
        }
        //        if (abs(A_Matrix[startIdx - 1]) < EPSILON)
        //            printf(" %d A_Matrix[startIdx - 1]= %f, type=%f \n", i_idx, A_Matrix[startIdx - 1],
        //                   sortedRhoPreMu[i_idx].w);
        V_new[i_idx] = (b3vec[i_idx] - aij_vj) / A_Matrix[startIdx - 1];
    } else {
        Real aij_pj = 0.0;
        for (int myIdx = startIdx; myIdx < endIdx; myIdx++) {
            aij_pj += A_Matrix[myIdx] * q_old[csrColInd[myIdx]];
        }
        //        if (A_Matrix[startIdx - 1] == 0.0)
        //            printf(" %d A_Matrix[startIdx - 1]= %f, type=%f \n", i_idx, A_Matrix[startIdx - 1],
        //                   sortedRhoPreMu[i_idx].w);

        q_new[i_idx] = (b1vec[i_idx] - aij_pj) / A_Matrix[startIdx - 1];
    }
}
//--------------------------------------------------------------------------------------------------------------------------------
__global__ void Update_AND_Calc_Res(Real4* sortedRhoPreMu,
                                    Real3* V_old,
                                    Real3* V_new,
                                    Real* q_old,
                                    Real* q_new,
                                    Real* Residuals,
                                    const int numAllMarkers,
                                    bool _3dvector,
                                    volatile bool* isErrorD) {
    uint i_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_idx >= numAllMarkers) {
        return;
    }
    Real omega = paramsD.PPE_relaxation;
    Real res = 0;
    if (_3dvector) {
        V_new[i_idx] = (1 - omega) * V_old[i_idx] + omega * V_new[i_idx];
        res = length(V_old[i_idx] - V_new[i_idx]);
        V_old[i_idx] = V_new[i_idx];
        //        if (!(isfinite(V_old[i_idx].x) && isfinite(V_old[i_idx].y) && isfinite(V_old[i_idx].z)))
        //            printf(" %d vel= %f,%f,%f\n", i_idx, V_old[i_idx].x, V_old[i_idx].y, V_old[i_idx].z);

    } else {
        q_new[i_idx] = (1 - omega) * q_old[i_idx] + omega * q_new[i_idx];
        res = abs(q_old[i_idx] - q_new[i_idx]);
        q_old[i_idx] = q_new[i_idx];
        //        if (!(isfinite(q_old[i_idx])))
        //            printf(" %d q= %f\n", i_idx, q_old[i_idx]);
    }
    Residuals[i_idx] = res;
}
//--------------------------------------------------------------------------------------------------------------------------------
__global__ void Initialize_Variables(Real4* sortedRhoPreMu,
                                     Real* p_old,
                                     Real3* sortedVelMas,
                                     Real3* V_new,
                                     const int numAllMarkers,
                                     volatile bool* isErrorD) {
    const uint i_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_idx >= numAllMarkers) {
        return;
    }
    if (sortedRhoPreMu[i_idx].w <= -2) {
        return;
    }

    p_old[i_idx] = sortedRhoPreMu[i_idx].y;  // This needs consistency p_old is old but v_new is new !!
    if (sortedRhoPreMu[i_idx].w > -1) {
        sortedVelMas[i_idx] = V_new[i_idx];
    }
}

//--------------------------------------------------------------------------------------------------------------------------------
__global__ void UpdateDensity(Real3* vis_vel,
                              Real3* new_vel,       // Write
                              Real4* sortedPosRad,  // Read
                              Real4* sortedRhoPreMu,
                              Real* sumWij_inv,
                              uint* cellStart,
                              uint* cellEnd,
                              uint numAllMarkers,
                              volatile bool* isErrorD) {
    uint i_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_idx >= numAllMarkers) {
        return;
    }
    if (sortedRhoPreMu[i_idx].w <= -2) {
        sortedRhoPreMu[i_idx].x = 0;
        sortedRhoPreMu[i_idx].y = 0;
        sortedRhoPreMu[i_idx].z = 0;
        return;
    }
    Real dT = paramsD.dT;
    Real rho_plus = 0;
    Real3 Vel_i = new_vel[i_idx];
    Real3 posi = mR3(sortedPosRad[i_idx]);
    if ((sortedRhoPreMu[i_idx].x > 2 * paramsD.rho0 || sortedRhoPreMu[i_idx].x < 0) && sortedRhoPreMu[i_idx].w < 0)
        printf("(UpdateDensity-0)too large/small density marker %d, type=%f\n", i_idx, sortedRhoPreMu[i_idx].w);
    Real h_i = sortedPosRad[i_idx].w;
    //    Real m_i = pow(h_i * paramsD.MULT_INITSPACE, 3) * paramsD.rho0;
    int3 gridPos = calcGridPos(posi);

    Real3 normalizedV_n = mR3(0);
    Real normalizedV_d = 0.0;
    Real sumW = 0.0;

    for (int z = -1; z <= 1; z++) {
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                int3 neighbourPos = gridPos + mI3(x, y, z);
                uint gridHash = calcGridHash(neighbourPos);
                // get start of bucket for this cell
                uint startIndex = cellStart[gridHash];
                uint endIndex = cellEnd[gridHash];
                for (uint j = startIndex; j < endIndex; j++) {
                    Real3 posj = mR3(sortedPosRad[j]);
                    Real3 dist3 = Distance(posi, posj);
                    Real d = length(dist3);
                    if (d > RESOLUTION_LENGTH_MULT * h_i || sortedRhoPreMu[j].w <= -2 ||
                        (sortedRhoPreMu[i_idx].w >= 0 && sortedRhoPreMu[j].w >= 0))
                        continue;
                    Real3 Vel_j = new_vel[j];
                    Real h_j = sortedPosRad[j].w;
                    Real m_j = pow(h_j * paramsD.MULT_INITSPACE, 3) * paramsD.rho0;
                    Real h_ij = 0.5 * (h_j + h_i);
                    Real3 grad_i_wij = GradWh(dist3, h_ij);
                    rho_plus += m_j * dot((Vel_i - Vel_j), grad_i_wij) * sumWij_inv[j];
                    Real Wd = W3h(d, h_ij);
                    sumW += Wd;

                    if (sortedRhoPreMu[j].w != -1)
                        Vel_j = mR3(0.0);

                    normalizedV_n += Vel_j * Wd * m_j / sortedRhoPreMu[j].x;
                    normalizedV_d += Wd * m_j / sortedRhoPreMu[j].x;
                }
            }
        }
    }
    if (abs(sumW) > EPSILON) {
        vis_vel[i_idx] = normalizedV_n / normalizedV_d;
        //        new_vel[i_idx] = paramsD.EPS_XSPH * vis_vel[i_idx] + (1 - paramsD.EPS_XSPH) * new_vel[i_idx]; //race
        //        condition
    }

    sortedRhoPreMu[i_idx].x += rho_plus * dT;
    if ((sortedRhoPreMu[i_idx].x > 2 * paramsD.rho0 || sortedRhoPreMu[i_idx].x < 0) && sortedRhoPreMu[i_idx].w < 0)
        printf("(UpdateDensity-1)too large/small density marker %d, type=%f\n", i_idx, sortedRhoPreMu[i_idx].w);
}

//--------------------------------------------------------------------------------------------------------------------------------
__global__ void Calc_HelperMarkers_normals(Real4* sortedPosRad,
                                           Real4* sortedRhoPreMu,
                                           Real3* helpers_normal,
                                           int* myType,
                                           uint* cellStart,
                                           uint* cellEnd,
                                           uint* gridMarkerIndexD,
                                           int numAllMarkers,
                                           volatile bool* isErrorD) {
    uint i_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_idx >= numAllMarkers) {
        return;
    }
    myType[i_idx] = (int)round(sortedRhoPreMu[i_idx].w);

    if (sortedRhoPreMu[i_idx].w != -3) {
        return;
    }
    int Original_idx = gridMarkerIndexD[i_idx];
    Real3 posi = mR3(sortedPosRad[i_idx]);
    Real h_i = sortedPosRad[i_idx].w;

    int j_pre = i_idx;
    int num_samples = 0;
    Real3 my_normal = mR3(0.1);

    //    int3 gridPos = calcGridPos(posi);
    //    for (int z = -2; z <= 2; z++) {
    //        for (int y = -2; y <= 2; y++) {
    //            for (int x = -2; x <= 2; x++) {
    //                int3 neighbourPos = gridPos + mI3(x, y, z);
    //                uint gridHash = calcGridHash(neighbourPos);
    //                uint startIndex = cellStart[gridHash];
    //                uint endIndex = cellEnd[gridHash];
    //                for (uint j = startIndex; j < endIndex; j++) {
    //                    if (sortedRhoPreMu[j].w != -3) {
    //                        continue;
    //                    }
    //                    if (j_pre == i_idx) {
    //                        j_pre = j;
    //                        continue;
    //                    } else {
    //                        Real3 posj = mR3(sortedPosRad[j]);
    //                        Real3 posjpre = mR3(sortedPosRad[j_pre]);
    //                        Real3 temp = cross(posj - posi, posjpre - posi);
    //                        temp = temp / length(temp);
    //                        num_samples++;
    //                        my_normal += temp;
    //                        j_pre = i_idx;
    //                    }
    //                }
    //            }
    //        }
    //    }

    Real3 cent = mR3(-0.005, 0, 0.205);
    my_normal = posi - cent;
    my_normal = mR3(my_normal.x, 0.0, my_normal.z);
    //    Real3 test = (posi - cent);
    //    test.y = 0;

    //    if (posi.y > 0.038 && r < 0.098)
    //        my_normal = mR3(0, 1, 0);
    //    else if (posi.y > 0.038 && r > 0.098)
    //        my_normal = normalize(my_normal + mR3(0, 0.05, 0));
    //    else if (posi.y < -0.038 && length(my_normal) < 0.098)
    //        my_normal = mR3(0, -1, 0);
    //    else if (posi.y < -0.038 && r > 0.098)
    //        my_normal = normalize(my_normal + mR3(0, -0.05, 0));

    //    Real3 cent = mR3(-0.005, 0, 0.195);
    //    my_normal = posi - cent;
    //    normalize(my_normal);

    helpers_normal[Original_idx] = my_normal / length(my_normal);

    //    printf("Original_idx=%d, p=(%f,%f,%f), Ni[0]=%f, Ni[1]=%f, Ni[2]=%f\n", Original_idx, posi.x, posi.y, posi.z,
    //           helpers_normal[Original_idx].x, helpers_normal[Original_idx].y, helpers_normal[Original_idx].z);
}
//--------------------------------------------------------------------------------------------------------------------------------
__global__ void Calc_Splits_and_Merges(Real4* sortedPosRad,
                                       Real4* sortedRhoPreMu,
                                       Real3* sortedVelMas,
                                       Real3* helpers_normal,

                                       Real* A_L,   /// Laplacian Operator matrix
                                       Real3* A_G,  /// Gradient Operator matrix
                                       Real* A_f,   /// Function Operator matrix
                                       const uint* csrColInd,
                                       const uint* numContacts,

                                       uint* splitMe,
                                       uint* MergeMe,
                                       int* myType,

                                       uint* gridMarkerIndexD,
                                       Real fineResolution,
                                       Real coarseResolution,
                                       int numAllMarkers,
                                       int limit,
                                       volatile bool* isErrorD) {
    uint i_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int Original_idx = gridMarkerIndexD[i_idx];
    if (i_idx >= numAllMarkers) {
        return;
    }

    if (sortedRhoPreMu[i_idx].w != -3) {
        return;
    }

    Real3 posi = mR3(sortedPosRad[i_idx]);

    //    MergingMarkers1[Original_idx] = mI4(i_idx);
    //    MergingMarkers2[Original_idx] = mI4(i_idx);
    Real3 normal = helpers_normal[Original_idx];
    //    Real3 normal = helpers_normal[i_idx];
    //    sortedVelMas[i_idx] = normal;
    //    Real3 normal = posi - mR3(-0.005, 0.0, 0.195);

    //    sortedRhoPreMu[i_idx].x = normal.x;
    //    sortedRhoPreMu[i_idx].y = normal.y;
    //    sortedRhoPreMu[i_idx].z = normal.z;

    uint Ni[8] = {i_idx};
    uint mySplits = 0;
    uint myMerges = 0;
    uint csrStartIdx = numContacts[i_idx];
    uint csrEndIdx = numContacts[i_idx + 1];
    for (int count = csrStartIdx; count < csrEndIdx; count++) {
        uint j = csrColInd[count];
        Real3 posj = mR3(sortedPosRad[j]);
        Real3 dist3 = Distance(posj, posi);
        Real d = length(dist3);
        Real3 velj = sortedVelMas[j];
        Real cosT = dot(dist3, normal) / (length(dist3) * length(normal));
        Real cosTv = dot(velj, normal) / (length(velj) * length(normal));

        //                    Real y2 = d * d * cosT * cosT;
        //                    Real b2 = coarseResolution * coarseResolution;
        Real x2 = d * d * (1 - cosT * cosT);
        //                    Real a2 = fineResolution * fineResolution;
        //                    y2 / b2 + x2 / a2 < 1

        ////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////
        /*x2 < 9 / 4 * fineResolution * fineResolution &&*/
        /*|| (d * cosT < fineResolution * 1.5 && dot(velj, normal) > 0))*/
        /*||
             (d * cosT < fineResolution * 1 && dot(velj, normal) > 0.8 * length(velj)))*/

        //                    if (abs(sortedPosRad[j].w - fineResolution) < EPSILON && (MergeMe[j] ==
        //                    i_idx)
        //                    &&
        //                        (MergeMe[j] == i_idx) && myType[j] == -1 && dot(dist3, normal) > 0 &&
        //                        x2 < fineResolution * fineResolution) {
        //////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////
        //                    if (sortedRhoPreMu[j].w == -1 && abs(sortedPosRad[j].w - fineResolution) <
        //                    EPSILON &&
        //                        myType[j] == -1 && MergeMe[j] == 0 && splitMe[j] == 0 && d <
        //                        coarseResolution && cosTv > 0.2) {
        //        (dot(dist3, normal) > 0 || (d < 2 * fineResolution && cosTv > 0.2)) &&
        if (abs(sortedPosRad[j].w - fineResolution) < EPSILON && (MergeMe[j] == 0) && myType[j] == -1 &&
            (sortedRhoPreMu[j].w == -1) && cosT > 0 && cosTv > 0) {
            uint p = 9;
            if (d < length(Distance(posi, mR3(sortedPosRad[Ni[7]]))) || Ni[7] == i_idx)
                p = 8;
            if (d < length(Distance(posi, mR3(sortedPosRad[Ni[6]]))) || Ni[6] == i_idx)
                p = 7;
            if (d < length(Distance(posi, mR3(sortedPosRad[Ni[5]]))) || Ni[5] == i_idx)
                p = 6;
            if (d < length(Distance(posi, mR3(sortedPosRad[Ni[4]]))) || Ni[4] == i_idx)
                p = 5;
            if (d < length(Distance(posi, mR3(sortedPosRad[Ni[3]]))) || Ni[3] == i_idx)
                p = 4;
            if (d < length(Distance(posi, mR3(sortedPosRad[Ni[2]]))) || Ni[2] == i_idx)
                p = 3;
            if (d < length(Distance(posi, mR3(sortedPosRad[Ni[1]]))) || Ni[1] == i_idx)
                p = 2;
            if (d < length(Distance(posi, mR3(sortedPosRad[Ni[0]]))) || Ni[0] == i_idx)
                p = 1;

            if (p < 9) {
                atomicCAS(&MergeMe[j], 0, i_idx);
                if (MergeMe[j] != i_idx)
                    continue;
                // Release the last one since it is going to be replaced
                if (Ni[7] != i_idx)
                    atomicExch(&MergeMe[Ni[7]], 0);
            }
            if (p == 8) {
                Ni[7] = j;
            } else if (p == 7) {
                Ni[7] = Ni[6];
                Ni[6] = j;
            } else if (p == 6) {
                Ni[7] = Ni[6];
                Ni[6] = Ni[5];
                Ni[5] = j;
            } else if (p == 5) {
                Ni[7] = Ni[6];
                Ni[6] = Ni[5];
                Ni[5] = Ni[4];
                Ni[4] = j;
            } else if (p == 4) {
                Ni[7] = Ni[6];
                Ni[6] = Ni[5];
                Ni[5] = Ni[4];
                Ni[4] = Ni[3];
                Ni[3] = j;
            } else if (p == 3) {
                Ni[7] = Ni[6];
                Ni[6] = Ni[5];
                Ni[5] = Ni[4];
                Ni[4] = Ni[3];
                Ni[3] = Ni[2];
                Ni[2] = j;
            } else if (p == 2) {
                Ni[7] = Ni[6];
                Ni[6] = Ni[5];
                Ni[5] = Ni[4];
                Ni[4] = Ni[3];
                Ni[3] = Ni[2];
                Ni[2] = Ni[1];
                Ni[1] = j;
            } else if (p == 1) {
                Ni[7] = Ni[6];
                Ni[6] = Ni[5];
                Ni[5] = Ni[4];
                Ni[4] = Ni[3];
                Ni[3] = Ni[2];
                Ni[2] = Ni[1];
                Ni[1] = Ni[0];
                Ni[0] = j;
            }
        } else if (abs(sortedPosRad[j].w - coarseResolution) < EPSILON && splitMe[j] == 0 && myType[j] == -1 &&
                   d < coarseResolution && cosTv < 0.0 && cosT < 0) {
            //                    } else if (abs(sortedPosRad[j].w - coarseResolution) < EPSILON &&
            //                    splitMe[j] == 0 &&
            //                               MergeMe[j] == 0 && myType[j] == -1 && sortedRhoPreMu[j].w
            //                               == -1
            //                               && d < 1 * coarseResolution && cosTv < -0.5) {
            //                        atomicCAS(&splitMe[j], 0, i_idx);
        }
    }

    // note that this can cause race condition if two helper markers try to merge the same marker Ni
    if (Ni[7] != i_idx && Ni[6] != i_idx && Ni[5] != i_idx && Ni[4] != i_idx && Ni[3] != i_idx && Ni[2] != i_idx &&
        Ni[1] != i_idx && Ni[0] != i_idx) {
        if (MergeMe[Ni[0]] != i_idx || MergeMe[Ni[1]] != i_idx || MergeMe[Ni[2]] != i_idx || MergeMe[Ni[3]] != i_idx ||
            MergeMe[Ni[4]] != i_idx || MergeMe[Ni[5]] != i_idx || MergeMe[Ni[6]] != i_idx || MergeMe[Ni[7]] != i_idx) {
            printf("RACE CONDITION in merging! Please revise the spacing or the merging scheme.\n");
            //            *isErrorD = true;
        }

        //        printf("idx=%d Start=%d, End=%d, merging %d,%d,%d,%d,%d,%d,%d,%d\n", i_idx, csrStartIdx, csrEndIdx,
        //        Ni[0],
        //                 Ni[1], Ni[2], Ni[3], Ni[4], Ni[5], Ni[6], Ni[7]);

        if (sortedPosRad[Ni[0]].w != fineResolution || sortedPosRad[Ni[1]].w != fineResolution ||
            sortedPosRad[Ni[2]].w != fineResolution || sortedPosRad[Ni[3]].w != fineResolution ||
            sortedPosRad[Ni[4]].w != fineResolution || sortedPosRad[Ni[5]].w != fineResolution ||
            sortedPosRad[Ni[6]].w != fineResolution || sortedPosRad[Ni[7]].w != fineResolution) {
            printf("ops something went wrong!.\n");
            //            *isErrorD = true;
        }

        Real4 center = 0.125 * (sortedPosRad[Ni[0]] + sortedPosRad[Ni[1]] + sortedPosRad[Ni[2]] + sortedPosRad[Ni[3]] +
                                sortedPosRad[Ni[4]] + sortedPosRad[Ni[5]] + sortedPosRad[Ni[6]] + sortedPosRad[Ni[7]]);
        Real4 rpmt =
            0.125 * (sortedRhoPreMu[Ni[0]] + sortedRhoPreMu[Ni[1]] + sortedRhoPreMu[Ni[2]] + sortedRhoPreMu[Ni[3]] +
                     sortedRhoPreMu[Ni[4]] + sortedRhoPreMu[Ni[5]] + sortedRhoPreMu[Ni[6]] + sortedRhoPreMu[Ni[7]]);

        Real3 myGradVx[8] = {mR3(0.0)};
        Real3 myGradVy[8] = {mR3(0.0)};
        Real3 myGradVz[8] = {mR3(0.0)};
        Real3 myGradrho[8] = {mR3(0.0)};
        Real3 myGradp[8] = {mR3(0.0)};

        for (int par = 0; par < 8; par++) {
            uint csrStartIdx = numContacts[Ni[par]];
            uint csrEndIdx = numContacts[Ni[par] + 1];
            for (int count = csrStartIdx; count < csrEndIdx; count++) {
                int j = csrColInd[count];
                myGradVx[par] += A_G[count] * sortedVelMas[j].x;
                myGradVy[par] += A_G[count] * sortedVelMas[j].y;
                myGradVz[par] += A_G[count] * sortedVelMas[j].z;
                myGradrho[par] += A_G[count] * sortedRhoPreMu[j].x;
                myGradp[par] += A_G[count] * sortedRhoPreMu[j].y;
            }
            Real3 delta_r = mR3(center - sortedPosRad[Ni[par]]);
            sortedVelMas[Ni[par]].x += dot(myGradVx[par], delta_r);
            sortedVelMas[Ni[par]].y += dot(myGradVy[par], delta_r);
            sortedVelMas[Ni[par]].z += dot(myGradVz[par], delta_r);
            sortedRhoPreMu[Ni[par]].x += dot(myGradrho[par], delta_r);
            sortedRhoPreMu[Ni[par]].y += dot(myGradp[par], delta_r);
        }

        Real3 Vel = mR3(0.0);
        rpmt = mR4(0.0);
        center = mR4(0.0);
        for (int par = 0; par < 8; par++) {
            center += sortedPosRad[Ni[par]];
            Vel += sortedVelMas[Ni[par]];
            rpmt += sortedRhoPreMu[Ni[par]];
            sortedRhoPreMu[Ni[par]] = mR4(paramsD.rho0, 1e-20, paramsD.mu0, -2);
            sortedPosRad[Ni[par]] = mR4(0.0, 0.0, -0.8, coarseResolution);
            sortedVelMas[Ni[par]] = mR3(0.0);
        }

        center /= 8.0;
        Vel /= 8.0;
        rpmt /= 8.0;
        sortedPosRad[Ni[0]] = mR4(mR3(center), fineResolution);
        sortedVelMas[Ni[0]] = Vel;
        sortedRhoPreMu[Ni[0]] = mR4(mR3(rpmt), -1);

        if (sortedPosRad[Ni[0]].w != fineResolution || sortedRhoPreMu[Ni[0]].w != -1) {
            printf("ops something went wrong!!!!\n");
            *isErrorD = true;
        }
    }
}
//--------------------------------------------------------------------------------------------------------------------------------
__global__ void Split(Real4* sortedPosRad,
                      Real4* sortedRhoPreMu,
                      Real3* sortedVelMas,
                      Real3* helpers_normal,

                      Real* A_L,   /// Laplacian Operator matrix
                      Real3* A_G,  /// Gradient Operator matrix
                      Real* A_f,   /// Function Operator matrix
                      const uint* csrColInd,
                      const uint* numContacts,

                      uint* splitMe,
                      uint* MergeMe,
                      int* myType,

                      uint* gridMarkerIndexD,
                      Real fineResolution,
                      Real coarseResolution,
                      int numAllMarkers,
                      int limit,
                      volatile bool* isErrorD) {
    uint i_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int Original_idx = gridMarkerIndexD[i_idx];
    if (i_idx >= numAllMarkers) {
        return;
    }
    if (splitMe[i_idx] == 0)
        return;

    Real3 posi = mR3(sortedPosRad[i_idx]);
    Real3 veli = sortedVelMas[i_idx];

    //    if (abs(sortedPosRad[i_idx].w - coarseResolution) < EPSILON && sortedRhoPreMu[i_idx].w == -1 &&
    //        splitMe[i_idx] != 0 && myType[i_idx] == -1) {
    //        uint temp = 0;
    //
    //        MergeMe[i_idx] = -1;
    //        uint childMarkers[7] = {0};
    //        temp = 0;
    //        uint k = 0;
    //        int numpass = 0;
    //        Real4 RhoPreMu = sortedRhoPreMu[i_idx];
    //
    //        while (k < 7) {
    //            atomicCAS(&myType[temp], -2, i_idx);
    //            if (myType[temp] == i_idx) {
    //                childMarkers[k] = temp;
    //                k++;
    //            }
    //            if (temp > numAllMarkers) {
    //                if (numpass < 2) {
    //                    numpass++;
    //                    temp = 0;
    //                    continue;
    //                }
    //                break;
    //                *isErrorD = true;
    //            }
    //            temp++;
    //        }
    //        for (int n = 0; n < 7; n++) {
    //            sortedVelMas[childMarkers[n]] = veli;
    //            sortedRhoPreMu[childMarkers[n]] = RhoPreMu;
    //            sumWij_inv[childMarkers[n]] = sumWij_inv[i_idx];
    //
    //            for (int l = 0; l < 9; l++)
    //                G_i[childMarkers[n] * 9 + l] = G_i[i_idx * 9 + l];
    //
    //            for (int l = 0; l < 6; l++)
    //                L_i[childMarkers[n] * 6 + l] = L_i[i_idx * 6 + l];
    //        }
    //        Real3 center = mR3(sortedPosRad[i_idx]);
    //        Real h = fineResolution;
    //
    //        childMarkers[7] = i_idx;
    //        Real d = 0.5 * fineResolution;
    //        // This is recommended for our confuguration from Vacondio et. al
    //        //                Real d = 0.72 * coarseResolution / 1.7321;
    //        sortedPosRad[childMarkers[0]] = mR4(center + d * mR3(-1, -1, -1), h);
    //        sortedPosRad[childMarkers[1]] = mR4(center + d * mR3(+1, -1, -1), h);
    //        sortedPosRad[childMarkers[2]] = mR4(center + d * mR3(+1, +1, -1), h);
    //        sortedPosRad[childMarkers[3]] = mR4(center + d * mR3(-1, +1, -1), h);
    //        sortedPosRad[childMarkers[4]] = mR4(center + d * mR3(-1, -1, +1), h);
    //        sortedPosRad[childMarkers[5]] = mR4(center + d * mR3(+1, -1, +1), h);
    //        sortedPosRad[childMarkers[6]] = mR4(center + d * mR3(+1, +1, +1), h);
    //        sortedPosRad[childMarkers[7]] = mR4(center + d * mR3(-1, +1, +1), h);
    //
    //        Real3 myGradrho[8] = {0};
    //        Real rho0 = +sortedRhoPreMu[Ni[7]].x;
    //
    //        grad_scalar(Ni[0], sortedPosRad, sortedRhoPreMu, sumWij_inv, G_i, sortedRhoPreMu, myGrad[0], cellStart,
    //                    cellEnd);
    //        grad_scalar(Ni[1], sortedPosRad, sortedRhoPreMu, sumWij_inv, G_i, sortedRhoPreMu, myGrad[1], cellStart,
    //                    cellEnd);
    //        grad_scalar(Ni[2], sortedPosRad, sortedRhoPreMu, sumWij_inv, G_i, sortedRhoPreMu, myGrad[2], cellStart,
    //                    cellEnd);
    //        grad_scalar(Ni[3], sortedPosRad, sortedRhoPreMu, sumWij_inv, G_i, sortedRhoPreMu, myGrad[3], cellStart,
    //                    cellEnd);
    //        grad_scalar(Ni[4], sortedPosRad, sortedRhoPreMu, sumWij_inv, G_i, sortedRhoPreMu, myGrad[4], cellStart,
    //                    cellEnd);
    //        grad_scalar(Ni[5], sortedPosRad, sortedRhoPreMu, sumWij_inv, G_i, sortedRhoPreMu, myGrad[5], cellStart,
    //                    cellEnd);
    //        grad_scalar(Ni[6], sortedPosRad, sortedRhoPreMu, sumWij_inv, G_i, sortedRhoPreMu, myGrad[6], cellStart,
    //                    cellEnd);
    //        grad_scalar(Ni[7], sortedPosRad, sortedRhoPreMu, sumWij_inv, G_i, sortedRhoPreMu, myGrad[7], cellStart,
    //                    cellEnd);
    //        sortedRhoPreMu[Ni[0]].x += dot(myGrad[0], mR3(sortedPosRad[Ni[0]]) - center);
    //        sortedRhoPreMu[Ni[1]].x += dot(myGrad[1], mR3(sortedPosRad[Ni[1]]) - center);
    //        sortedRhoPreMu[Ni[2]].x += dot(myGrad[2], mR3(sortedPosRad[Ni[2]]) - center);
    //        sortedRhoPreMu[Ni[3]].x += dot(myGrad[3], mR3(sortedPosRad[Ni[3]]) - center);
    //        sortedRhoPreMu[Ni[4]].x += dot(myGrad[4], mR3(sortedPosRad[Ni[4]]) - center);
    //        sortedRhoPreMu[Ni[5]].x += dot(myGrad[5], mR3(sortedPosRad[Ni[5]]) - center);
    //        sortedRhoPreMu[Ni[6]].x += dot(myGrad[6], mR3(sortedPosRad[Ni[6]]) - center);
    //        sortedRhoPreMu[Ni[7]].x += dot(myGrad[7], mR3(sortedPosRad[Ni[7]]) - center);
    //
    //        Real3 myGradx[8] = {0};
    //        Real3 myGrady[8] = {0};
    //        Real3 myGradz[8] = {0};
    //        grad_vector(Ni[0], sortedPosRad, sortedRhoPreMu, sumWij_inv, G_i, sortedVelMas, myGradx[0], myGrady[0],
    //                    myGradz[0], cellStart, cellEnd);
    //        grad_vector(Ni[1], sortedPosRad, sortedRhoPreMu, sumWij_inv, G_i, sortedVelMas, myGradx[1], myGrady[1],
    //                    myGradz[1], cellStart, cellEnd);
    //        grad_vector(Ni[2], sortedPosRad, sortedRhoPreMu, sumWij_inv, G_i, sortedVelMas, myGradx[2], myGrady[2],
    //                    myGradz[2], cellStart, cellEnd);
    //        grad_vector(Ni[3], sortedPosRad, sortedRhoPreMu, sumWij_inv, G_i, sortedVelMas, myGradx[3], myGrady[3],
    //                    myGradz[3], cellStart, cellEnd);
    //        grad_vector(Ni[4], sortedPosRad, sortedRhoPreMu, sumWij_inv, G_i, sortedVelMas, myGradx[4], myGrady[4],
    //                    myGradz[4], cellStart, cellEnd);
    //        grad_vector(Ni[5], sortedPosRad, sortedRhoPreMu, sumWij_inv, G_i, sortedVelMas, myGradx[5], myGrady[5],
    //                    myGradz[5], cellStart, cellEnd);
    //        grad_vector(Ni[6], sortedPosRad, sortedRhoPreMu, sumWij_inv, G_i, sortedVelMas, myGradx[6], myGrady[6],
    //                    myGradz[6], cellStart, cellEnd);
    //        grad_vector(Ni[7], sortedPosRad, sortedRhoPreMu, sumWij_inv, G_i, sortedVelMas, myGradx[7], myGrady[7],
    //                    myGradz[7], cellStart, cellEnd);
    //
    //        sortedVelMas[Ni[0]].x += dot(myGradx[0], mR3(sortedPosRad[Ni[0]]) - center);
    //        sortedVelMas[Ni[1]].x += dot(myGradx[1], mR3(sortedPosRad[Ni[1]]) - center);
    //        sortedVelMas[Ni[2]].x += dot(myGradx[2], mR3(sortedPosRad[Ni[2]]) - center);
    //        sortedVelMas[Ni[3]].x += dot(myGradx[3], mR3(sortedPosRad[Ni[3]]) - center);
    //        sortedVelMas[Ni[4]].x += dot(myGradx[4], mR3(sortedPosRad[Ni[4]]) - center);
    //        sortedVelMas[Ni[5]].x += dot(myGradx[5], mR3(sortedPosRad[Ni[5]]) - center);
    //        sortedVelMas[Ni[6]].x += dot(myGradx[6], mR3(sortedPosRad[Ni[6]]) - center);
    //        sortedVelMas[Ni[7]].x += dot(myGradx[7], mR3(sortedPosRad[Ni[7]]) - center);
    //
    //        sortedVelMas[Ni[0]].y += dot(myGrady[0], mR3(sortedPosRad[Ni[0]]) - center);
    //        sortedVelMas[Ni[1]].y += dot(myGrady[1], mR3(sortedPosRad[Ni[1]]) - center);
    //        sortedVelMas[Ni[2]].y += dot(myGrady[2], mR3(sortedPosRad[Ni[2]]) - center);
    //        sortedVelMas[Ni[3]].y += dot(myGrady[3], mR3(sortedPosRad[Ni[3]]) - center);
    //        sortedVelMas[Ni[4]].y += dot(myGrady[4], mR3(sortedPosRad[Ni[4]]) - center);
    //        sortedVelMas[Ni[5]].y += dot(myGrady[5], mR3(sortedPosRad[Ni[5]]) - center);
    //        sortedVelMas[Ni[6]].y += dot(myGrady[6], mR3(sortedPosRad[Ni[6]]) - center);
    //        sortedVelMas[Ni[7]].y += dot(myGrady[7], mR3(sortedPosRad[Ni[7]]) - center);
    //
    //        sortedVelMas[Ni[0]].z += dot(myGradz[0], mR3(sortedPosRad[Ni[0]]) - center);
    //        sortedVelMas[Ni[1]].z += dot(myGradz[1], mR3(sortedPosRad[Ni[1]]) - center);
    //        sortedVelMas[Ni[2]].z += dot(myGradz[2], mR3(sortedPosRad[Ni[2]]) - center);
    //        sortedVelMas[Ni[3]].z += dot(myGradz[3], mR3(sortedPosRad[Ni[3]]) - center);
    //        sortedVelMas[Ni[4]].z += dot(myGradz[4], mR3(sortedPosRad[Ni[4]]) - center);
    //        sortedVelMas[Ni[5]].z += dot(myGradz[5], mR3(sortedPosRad[Ni[5]]) - center);
    //        sortedVelMas[Ni[6]].z += dot(myGradz[6], mR3(sortedPosRad[Ni[6]]) - center);
    //        sortedVelMas[Ni[7]].z += dot(myGradz[7], mR3(sortedPosRad[Ni[7]]) - center);
    //
    //        //        printf("extrapolate vel of marker %d with %f\n", Ni[0], dot(myGradx[0], mR3(sortedPosRad[Ni[0]])
    //        //        - center));
    //
    //        if (temp > numAllMarkers)
    //            printf(
    //                "Reached the limit of the ghost markers. Please increase the number of ghost "
    //                "markers.\n");
    //    }
}
//--------------------------------------------------------------------------------------------------------------------------------

}  // namespace fsi
}  // namespace chrono
#endif
