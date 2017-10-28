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
// Author:  Milad Rakhsha
// =============================================================================
//
// Base class for processing sph force in fsi system.//
// =============================================================================

#ifndef CH_SPHKERNELS_CU_
#define CH_SPHKERNELS_CU_
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>
#include "chrono_fsi/ChSphGeneral.cuh"
#include "chrono_fsi/solver6x6.cuh"

namespace chrono {
namespace fsi {
//--------------------------------------------------------------------------------------------------------------------------------
inline __device__ void grad_scalar(int i_idx,
                                   Real4* sortedPosRad,  // input: sorted positions
                                   Real4* sortedRhoPreMu,
                                   Real* sumWij_inv,
                                   Real* G_i,
                                   Real4* Scalar,
                                   Real3& myGrad,
                                   uint* cellStart,
                                   uint* cellEnd) {
    // Note that this function only calculates the gradient of the first element of the Scalar;
    // This is hard coded like this for now because usually rho appears in Real4 structure
    Real3 posRadA = mR3(sortedPosRad[i_idx]);
    Real h_i = sortedPosRad[i_idx].w;
    int3 gridPos = calcGridPos(posRadA);

    //    printf("update G[%d]= %f,%f,%f  %f,%f,%f, %f,%f,%f\n", i_idx, G_i[i_idx * 9 + 0], G_i[i_idx * 9 + 1],
    //           G_i[i_idx * 9 + 2], G_i[i_idx * 9 + 3], G_i[i_idx * 9 + 4], G_i[i_idx * 9 + 5], G_i[i_idx * 9 + 6],
    //           G_i[i_idx * 9 + 7], G_i[i_idx * 9 + 8]);

    // This is the elements of inverse of G
    Real mGi[9];
    for (int n = 0; n < 9; n++)
        mGi[n] = G_i[i_idx * 9 + n];

    Real3 grad_si = mR3(0.);
    for (int z = -1; z <= 1; z++) {
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                int3 neighbourPos = gridPos + mI3(x, y, z);
                uint gridHash = calcGridHash(neighbourPos);
                // get start of bucket for this cell50
                uint startIndex = cellStart[gridHash];
                if (startIndex != 0xffffffff) {  // cell is not empty
                    uint endIndex = cellEnd[gridHash];

                    for (uint j = startIndex; j < endIndex; j++) {
                        Real3 posRadB = mR3(sortedPosRad[j]);
                        Real3 dist3 = Distance(posRadA, posRadB);
                        Real d = length(dist3);
                        if (d > RESOLUTION_LENGTH_MULT * h_i || sortedRhoPreMu[j].w <= -2)
                            continue;

                        Real h_j = sortedPosRad[j].w;
                        Real h_ij = 0.5 * (h_j + h_i);
                        Real W3 = W3h(d, h_ij);
                        Real3 grad_i_wij = GradWh(dist3, h_ij);
                        Real V_j = sumWij_inv[j];
                        Real3 common_part = mR3(0);
                        common_part.x = grad_i_wij.x * mGi[0] + grad_i_wij.y * mGi[1] + grad_i_wij.z * mGi[2];
                        common_part.y = grad_i_wij.x * mGi[3] + grad_i_wij.y * mGi[4] + grad_i_wij.z * mGi[5];
                        common_part.z = grad_i_wij.x * mGi[6] + grad_i_wij.y * mGi[7] + grad_i_wij.z * mGi[8];
                        grad_si += common_part * (Scalar[j].x - Scalar[i_idx].x) * V_j;
                    }
                }
            }
        }
    }
    myGrad = grad_si;
    //    printf("grad_scalar[%d]= %f,%f,%f\n", i_idx, myGrad.x, myGrad.y, myGrad.z);
}

//--------------------------------------------------------------------------------------------------------------------------------
inline __device__ void grad_vector(int i_idx,
                                   Real4* sortedPosRad,  // input: sorted positions
                                   Real4* sortedRhoPreMu,
                                   Real* sumWij_inv,
                                   Real* G_i,
                                   Real3* Vector,
                                   Real3& myGradx,
                                   Real3& myGrady,
                                   Real3& myGradz,
                                   uint* cellStart,
                                   uint* cellEnd) {
    Real3 posRadA = mR3(sortedPosRad[i_idx]);
    Real h_i = sortedPosRad[i_idx].w;
    int3 gridPos = calcGridPos(posRadA);

    // This is the elements of inverse of G
    Real mGi[9];
    for (int n = 0; n < 9; n++)
        mGi[n] = G_i[i_idx * 9 + n];

    Real3 common_part = mR3(0.);
    Real3 grad_Vx = mR3(0.);
    Real3 grad_Vy = mR3(0.);
    Real3 grad_Vz = mR3(0.);

    for (int z = -1; z <= 1; z++) {
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                int3 neighbourPos = gridPos + mI3(x, y, z);
                uint gridHash = calcGridHash(neighbourPos);
                // get start of bucket for this cell50
                uint startIndex = cellStart[gridHash];
                if (startIndex != 0xffffffff) {  // cell is not empty
                    uint endIndex = cellEnd[gridHash];

                    for (uint j = startIndex; j < endIndex; j++) {
                        Real3 posRadB = mR3(sortedPosRad[j]);
                        Real3 dist3 = Distance(posRadA, posRadB);
                        Real d = length(dist3);
                        if (d > RESOLUTION_LENGTH_MULT * h_i || sortedRhoPreMu[j].w <= -2)
                            continue;

                        Real h_j = sortedPosRad[j].w;
                        Real h_ij = 0.5 * (h_j + h_i);
                        Real W3 = W3h(d, h_ij);
                        Real3 grad_i_wij = GradWh(dist3, h_ij);
                        Real V_j = sumWij_inv[j];
                        common_part.x = grad_i_wij.x * mGi[0] + grad_i_wij.y * mGi[1] + grad_i_wij.z * mGi[2];
                        common_part.y = grad_i_wij.x * mGi[3] + grad_i_wij.y * mGi[4] + grad_i_wij.z * mGi[5];
                        common_part.z = grad_i_wij.x * mGi[6] + grad_i_wij.y * mGi[7] + grad_i_wij.z * mGi[8];
                        grad_Vx += common_part * (Vector[i_idx].x - Vector[j].x) * V_j;
                        grad_Vy += common_part * (Vector[i_idx].y - Vector[j].y) * V_j;
                        grad_Vz += common_part * (Vector[i_idx].z - Vector[j].z) * V_j;
                    }
                }
            }
        }
    }
    myGradx = grad_Vx;
    myGrady = grad_Vy;
    myGradz = grad_Vz;
    //    printf("grad_vector[%d]= %f,%f,%f  %f,%f,%f, %f,%f,%f\n", i_idx, myGradx.x, myGradx.y, myGradx.z, myGrady.x,
    //           myGrady.y, myGrady.z, myGradz.x, myGradz.y, myGradz.z);
}
//--------------------------------------------------------------------------------------------------------------------------------
inline __global__ void calc_A_tensor(Real* A_tensor,
                                     Real* G_tensor,
                                     Real4* sortedPosRad,
                                     Real4* sortedRhoPreMu,
                                     Real* sumWij_inv,
                                     uint* cellStart,
                                     uint* cellEnd,
                                     const int numAllMarkers,
                                     volatile bool* isErrorD) {
    uint i_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_idx >= numAllMarkers) {
        return;
    }

    // Remember : we want to solve 6x6 system Bi*l=-[1 0 0 1 0 1]'
    // elements of matrix B depends on tensor A
    Real3 posRadA = mR3(sortedPosRad[i_idx]);
    Real h_i = sortedPosRad[i_idx].w;
    Real m_i = h_i * h_i * h_i * paramsD.rho0;
    Real sum_mW = 0;
    Real A_ijk[27] = {0.0};

    Real Gi[9] = {0.0};
    for (int i = 0; i < 9; i++)
        Gi[i] = G_tensor[i_idx * 9 + i];

    // get address in grid
    int3 gridPos = calcGridPos(posRadA);

    for (int z = -1; z <= 1; z++) {
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                int3 neighbourPos = gridPos + mI3(x, y, z);
                uint gridHash = calcGridHash(neighbourPos);
                // get start of bucket for this cell
                uint startIndex = cellStart[gridHash];
                if (startIndex != 0xffffffff) {
                    uint endIndex = cellEnd[gridHash];
                    for (uint j = startIndex; j < endIndex; j++) {
                        Real3 posRadB = mR3(sortedPosRad[j]);
                        Real3 rij = Distance(posRadA, posRadB);
                        Real d = length(rij);
                        if (d > RESOLUTION_LENGTH_MULT * h_i || sortedRhoPreMu[j].w <= -2)
                            continue;
                        Real h_j = sortedPosRad[j].w;
                        Real m_j = h_j * h_j * h_j * paramsD.rho0;
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
                }
            }
        }
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
inline __global__ void calc_L_tensor(Real* A_tensor,
                                     Real* L_tensor,
                                     Real* G_tensor,
                                     Real4* sortedPosRad,
                                     Real4* sortedRhoPreMu,
                                     Real* sumWij_inv,
                                     uint* cellStart,
                                     uint* cellEnd,
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
    Real3 posRadA = mR3(sortedPosRad[i_idx]);
    Real h_i = sortedPosRad[i_idx].w;
    Real m_i = h_i * h_i * h_i * paramsD.rho0;
    Real B[36] = {0.0};

    Real Gi[9] = {0.0};
    for (int i = 0; i < 9; i++)
        Gi[i] = G_tensor[i_idx * 9 + i];

    Real A_ijk[27] = {0.0};
    for (int i = 0; i < 27; i++)
        A_ijk[i] = A_tensor[i_idx * 27 + i];

    // get address in grid
    int3 gridPos = calcGridPos(posRadA);

    for (int z = -1; z <= 1; z++) {
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                int3 neighbourPos = gridPos + mI3(x, y, z);
                uint gridHash = calcGridHash(neighbourPos);
                // get start of bucket for this cell
                uint startIndex = cellStart[gridHash];
                if (startIndex != 0xffffffff) {  // cell is not empty
                    uint endIndex = cellEnd[gridHash];
                    for (uint j = startIndex; j < endIndex; j++) {
                        Real3 posRadB = mR3(sortedPosRad[j]);
                        Real3 rij = Distance(posRadA, posRadB);
                        Real d = length(rij);
                        if (d > RESOLUTION_LENGTH_MULT * h_i || sortedRhoPreMu[j].w <= -2)
                            continue;
                        Real3 eij = rij / d;

                        Real h_j = sortedPosRad[j].w;
                        Real m_j = h_j * h_j * h_j * paramsD.rho0;
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
                }
            }
        }
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
inline __global__ void calcRho_kernel(Real4* sortedPosRad,  // input: sorted positionsmin(
                                      Real4* sortedRhoPreMu,
                                      Real* sumWij_inv,
                                      uint* cellStart,
                                      uint* cellEnd,
                                      const int numAllMarkers,
                                      volatile bool* isErrorD) {
    uint i_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_idx >= numAllMarkers) {
        return;
    }
    Real3 posRadA = mR3(sortedPosRad[i_idx]);
    Real h_i = sortedPosRad[i_idx].w;
    Real m_i = h_i * h_i * h_i * paramsD.rho0;
    Real sum_mW = 0;
    Real sum_W = 0.0;

    // get address in grid
    int3 gridPos = calcGridPos(posRadA);

    for (int z = -1; z <= 1; z++) {
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                int3 neighbourPos = gridPos + mI3(x, y, z);
                uint gridHash = calcGridHash(neighbourPos);
                // get start of bucket for this cell
                uint startIndex = cellStart[gridHash];
                //                if (startIndex > numAllMarkers)
                //                    printf("wow!startIndex=%d\n", startIndex);
                if (startIndex != 0xffffffff) {  // cell is not empty
                    uint endIndex = cellEnd[gridHash];
                    //                    if (endIndex > numAllMarkers)
                    //                        printf("wow!endIndex=%d\n", endIndex);

                    for (uint j = startIndex; j < endIndex; j++) {
                        Real3 posRadB = mR3(sortedPosRad[j]);
                        Real3 dist3 = Distance(posRadA, posRadB);
                        Real d = length(dist3);
                        if (d > RESOLUTION_LENGTH_MULT * h_i || sortedRhoPreMu[j].w <= -2)
                            continue;
                        Real h_j = sortedPosRad[j].w;
                        Real m_j = h_j * h_j * h_j * paramsD.rho0;
                        Real W3 = W3h(d, 0.5 * (h_j + h_i));
                        //                        Real W3 = 0.5 * (W3h(d, h_i) + W3h(d, h_j));
                        sum_mW += m_j * W3;
                        sum_W += W3;
                    }
                }
            }
        }
    }

    // Adding neighbor contribution is done!
    sumWij_inv[i_idx] = m_i / sum_mW;
    sortedRhoPreMu[i_idx].x = sum_mW;

    //    if (sumWij_inv[i_idx] > 1e-5 && sortedRhoPreMu[i_idx].w > -2)
    //        printf("sum_mW=%f,  sumWij_inv[i_idx]=%.4e\n", sum_mW, sumWij_inv[i_idx]);
    //
    //    //    sortedRhoPreMu[i_idx].x = sum_mW;
    //
    //    if ((sortedRhoPreMu[i_idx].x > 2 * paramsD.rho0 || sortedRhoPreMu[i_idx].x < 0) && sortedRhoPreMu[i_idx].w
    //    == -1)
    //        printf("(calcRho_kernel)too large/small density marker %d, rho=%f\n", i_idx, sortedRhoPreMu[i_idx].x);
}

//--------------------------------------------------------------------------------------------------------------------------------
inline __global__ void calcNormalizedRho_kernel(Real4* sortedPosRad,  // input: sorted positions
                                                Real3* sortedVelMas,
                                                Real4* sortedRhoPreMu,
                                                Real* sumWij_inv,
                                                Real* G_i,
                                                Real* dxi_over_Vi,
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

    Real3 posRadA = mR3(sortedPosRad[i_idx]);
    Real h_i = sortedPosRad[i_idx].w;
    //    Real m_i = h_i * h_i * h_i * paramsD.rho0;
    Real sum_mW = 0;
    Real sum_W_sumWij_inv = 0;
    Real C = 0;
    // get address in grid
    int3 gridPos = calcGridPos(posRadA);

    // This is the elements of inverse of G
    Real mGi[9] = {0.0};

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
                        Real m_j = h_j * h_j * h_j * paramsD.rho0;
                        C += m_j * Color[i_idx] / sortedRhoPreMu[i_idx].x * W3h(d, 0.5 * (h_j + h_i));
                        //                        Real particle_particle_n_CFL = abs(dot(dv3, dist3)) / d;
                        //                        Real particle_particle = length(dv3);
                        //                        Real particle_n_CFL = abs(dot(sortedVelMas[i_idx], dist3)) / d;
                        //                        Real particle_CFL = length(sortedVelMas[i_idx]);

                        //                        if (i_idx != j)
                        //                            dxi_over_Vi[i_idx] = fminf(d / particle_CFL,
                        //                            dxi_over_Vi[i_idx]);

                        if (d > RESOLUTION_LENGTH_MULT * h_i || sortedRhoPreMu[j].w <= -2)
                            continue;
                        Real h_ij = 0.5 * (h_j + h_i);
                        Real W3 = W3h(d, h_ij);
                        Real3 grad_i_wij = GradWh(dist3, h_ij);
                        Real V_j = sumWij_inv[j];
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

    sortedRhoPreMu[i_idx].x = (sum_mW / sum_W_sumWij_inv - RHO_0) * IncompressibilityFactor + RHO_0;

    //    if (sortedRhoPreMu[i_idx].x < EPSILON)
    if ((sortedRhoPreMu[i_idx].x > 5 * RHO_0 || sortedRhoPreMu[i_idx].x < RHO_0 / 5) && sortedRhoPreMu[i_idx].w > -2)
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

////--------------------------------------------------------------------------------------------------------------------------------
inline __device__ void BCE_Vel_Acc(int i_idx,
                                   Real3& myAcc,
                                   Real3& V_prescribed,

                                   Real4* sortedPosRad,
                                   int4 updatePortion,
                                   uint* gridMarkerIndexD,

                                   Real4* velMassRigid_fsiBodies_D,
                                   Real3* accRigid_fsiBodies_D,
                                   uint* rigidIdentifierD,

                                   Real3* pos_fsi_fea_D,
                                   Real3* vel_fsi_fea_D,
                                   Real3* acc_fsi_fea_D,
                                   uint* FlexIdentifierD,
                                   const int numFlex1D,
                                   uint2* CableElementsNodes,
                                   uint4* ShellelementsNodes) {
    int Original_idx = gridMarkerIndexD[i_idx];

    //  Real MASS;

    // See if this belongs to boundary
    if (Original_idx >= updatePortion.x && Original_idx < updatePortion.y) {
        myAcc = mR3(0.0);
        V_prescribed = mR3(0.0);
        if (paramsD.Apply_BC_U)
            V_prescribed = user_BC_U(mR3(sortedPosRad[i_idx]));

    } else if (Original_idx >= updatePortion.y && Original_idx < updatePortion.z) {
        int rigidIndex = rigidIdentifierD[Original_idx - updatePortion.y];
        V_prescribed = mR3(velMassRigid_fsiBodies_D[rigidIndex].x, velMassRigid_fsiBodies_D[rigidIndex].y,
                           velMassRigid_fsiBodies_D[rigidIndex].z);
        myAcc = mR3(accRigid_fsiBodies_D[rigidIndex].x, accRigid_fsiBodies_D[rigidIndex].y,
                    accRigid_fsiBodies_D[rigidIndex].z);

        // Or not, Flexible bodies for sure
    } else if (Original_idx >= updatePortion.z && Original_idx < updatePortion.w) {
        int FlexIndex = FlexIdentifierD[Original_idx - updatePortion.z];

        if (FlexIndex < numFlex1D) {
            int nA = CableElementsNodes[FlexIndex].y;
            int nB = CableElementsNodes[FlexIndex].x;

            Real3 pos_fsi_fea_D_nA = pos_fsi_fea_D[nA];
            Real3 pos_fsi_fea_D_nB = pos_fsi_fea_D[nB];

            Real3 vel_fsi_fea_D_nA = vel_fsi_fea_D[nA];
            Real3 vel_fsi_fea_D_nB = vel_fsi_fea_D[nB];

            Real3 acc_fsi_fea_D_nA = acc_fsi_fea_D[nA];
            Real3 acc_fsi_fea_D_nB = acc_fsi_fea_D[nB];

            Real3 dist3 = mR3(sortedPosRad[Original_idx]) - pos_fsi_fea_D_nA;
            Real3 x_dir = (pos_fsi_fea_D_nB - pos_fsi_fea_D_nA);
            Real Cable_x = length(x_dir);
            x_dir = x_dir / length(x_dir);
            Real dx = dot(dist3, x_dir);

            Real2 N_shell = Cables_ShapeFunctions(dx / Cable_x);
            Real NA = N_shell.x;
            Real NB = N_shell.y;

            V_prescribed = NA * vel_fsi_fea_D_nA + NB * vel_fsi_fea_D_nB;
            myAcc = NA * acc_fsi_fea_D_nA + NB * acc_fsi_fea_D_nB;
        }
        if (FlexIndex >= numFlex1D) {
            int nA = ShellelementsNodes[FlexIndex - numFlex1D].x;
            int nB = ShellelementsNodes[FlexIndex - numFlex1D].y;
            int nC = ShellelementsNodes[FlexIndex - numFlex1D].z;
            int nD = ShellelementsNodes[FlexIndex - numFlex1D].w;

            Real3 pos_fsi_fea_D_nA = pos_fsi_fea_D[nA];
            Real3 pos_fsi_fea_D_nB = pos_fsi_fea_D[nB];
            Real3 pos_fsi_fea_D_nC = pos_fsi_fea_D[nC];
            Real3 pos_fsi_fea_D_nD = pos_fsi_fea_D[nD];

            Real3 vel_fsi_fea_D_nA = vel_fsi_fea_D[nA];
            Real3 vel_fsi_fea_D_nB = vel_fsi_fea_D[nB];
            Real3 vel_fsi_fea_D_nC = vel_fsi_fea_D[nC];
            Real3 vel_fsi_fea_D_nD = vel_fsi_fea_D[nD];

            Real3 acc_fsi_fea_D_nA = acc_fsi_fea_D[nA];
            Real3 acc_fsi_fea_D_nB = acc_fsi_fea_D[nB];
            Real3 acc_fsi_fea_D_nC = acc_fsi_fea_D[nC];
            Real3 acc_fsi_fea_D_nD = acc_fsi_fea_D[nD];

            Real3 Shell_center = 0.25 * (pos_fsi_fea_D_nA + pos_fsi_fea_D_nB + pos_fsi_fea_D_nC + pos_fsi_fea_D_nD);
            Real3 dist3 = mR3(sortedPosRad[Original_idx]) - Shell_center;
            Real Shell_x =
                0.25 * (length(pos_fsi_fea_D_nB - pos_fsi_fea_D_nA) + length(pos_fsi_fea_D_nD - pos_fsi_fea_D_nC));
            Real Shell_y =
                0.25 * (length(pos_fsi_fea_D_nD - pos_fsi_fea_D_nA) + length(pos_fsi_fea_D_nC - pos_fsi_fea_D_nB));
            Real2 FlexSPH_MeshPos_Natural = mR2(dist3.x / Shell_x, dist3.y / Shell_y);

            Real4 N_shell = Shells_ShapeFunctions(FlexSPH_MeshPos_Natural.x, FlexSPH_MeshPos_Natural.y);
            Real NA = N_shell.x;
            Real NB = N_shell.y;
            Real NC = N_shell.z;
            Real ND = N_shell.w;
            V_prescribed =
                NA * vel_fsi_fea_D_nA + NB * vel_fsi_fea_D_nB + NC * vel_fsi_fea_D_nC + ND * vel_fsi_fea_D_nD;
            myAcc = NA * acc_fsi_fea_D_nA + NB * acc_fsi_fea_D_nB + NC * acc_fsi_fea_D_nC + ND * acc_fsi_fea_D_nD;
        }
    } else {
        printf("i_idx=%d, Original_idx:%d was not found \n\n", i_idx, Original_idx);
    }
}
//--------------------------------------------------------------------------------------------------------------------------------
inline __global__ void Initialize_Variables(Real4* sortedRhoPreMu,
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
inline __global__ void UpdateDensity(Real3* vis_vel,
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
    //    Real m_i = h_i * h_i * h_i * paramsD.rho0;
    int3 gridPos = calcGridPos(posi);

    Real3 normalizedV_n = mR3(0);
    Real normalizedV_d = 0.0;

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
                        (sortedRhoPreMu[i_idx].w > 0 && sortedRhoPreMu[j].w > 0))
                        continue;
                    Real3 Vel_j = new_vel[j];
                    Real h_j = sortedPosRad[j].w;
                    Real m_j = h_j * h_j * h_j * paramsD.rho0;
                    Real h_ij = 0.5 * (h_j + h_i);
                    Real3 grad_i_wij = GradWh(dist3, h_ij);
                    rho_plus += m_j * dot((Vel_i - Vel_j), grad_i_wij) * sumWij_inv[j];
                    Real Wd = W3h(d, h_ij);
                    if (sortedRhoPreMu[j].w == -1) {
                        normalizedV_n += Vel_j * Wd * m_j / sortedRhoPreMu[j].x;
                        normalizedV_d += Wd * m_j / sortedRhoPreMu[j].x;
                    }
                }
            }
        }
    }
    if (normalizedV_d > EPSILON && sortedRhoPreMu[i_idx].w == -1)
        vis_vel[i_idx] = normalizedV_n / normalizedV_d;
    sortedRhoPreMu[i_idx].x += rho_plus * dT;
    if ((sortedRhoPreMu[i_idx].x > 2 * paramsD.rho0 || sortedRhoPreMu[i_idx].x < 0) && sortedRhoPreMu[i_idx].w < 0)
        printf("(UpdateDensity-1)too large/small density marker %d, type=%f\n", i_idx, sortedRhoPreMu[i_idx].w);
}

//--------------------------------------------------------------------------------------------------------------------------------
inline __global__ void Calc_HelperMarkers_normals(Real4* sortedPosRad,
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

    Real3 cent = mR3(-0.005, 0, 0.195);
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

    //    printf("Original_idx=%d, p=(%f,%f,%f), n1=%f, n2=%f, n3=%f\n", Original_idx, posi.x, posi.y, posi.z,
    //           helpers_normal[Original_idx].x, helpers_normal[Original_idx].y, helpers_normal[Original_idx].z);
}
//--------------------------------------------------------------------------------------------------------------------------------
inline __global__ void Calc_Splits_and_Merges(Real4* sortedPosRad,
                                              Real4* sortedRhoPreMu,
                                              Real3* sortedVelMas,
                                              Real3* helpers_normal,
                                              Real* sumWij_inv,
                                              Real* G_i,
                                              uint* splitMe,
                                              uint* MergeMe,
                                              int* myType,
                                              uint* cellStart,
                                              uint* cellEnd,
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
    //    Real3 normal = posi - mR3(-0.005, 0.0, 0.195);

    sortedRhoPreMu[i_idx].x = normal.x;
    sortedRhoPreMu[i_idx].y = normal.y;
    sortedRhoPreMu[i_idx].z = normal.z;

    int3 gridPos = calcGridPos(posi);

    //    printf("Original_idx=%d, p=(%f,%f,%f), n1=%f, n2=%f, n3=%f\n", Original_idx, posi.x, posi.y, posi.z,
    //           helpers_normal[Original_idx].x, helpers_normal[Original_idx].y, helpers_normal[Original_idx].z);
    //    uint N1, N2, N3, N4, N5, N6, N7, N8;
    uint N1 = i_idx;
    uint N2 = i_idx;
    uint N3 = i_idx;
    uint N4 = i_idx;
    uint N5 = i_idx;
    uint N6 = i_idx;
    uint N7 = i_idx;
    uint N8 = i_idx;
    uint mySplits = 0;
    uint myMerges = 0;
    for (int z = -1; z <= 1; z++) {
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                //                int x = 0, y = 0, z = 0;
                int3 neighbourPos = gridPos + mI3(x, y, z);
                uint gridHash = calcGridHash(neighbourPos);
                uint startIndex = cellStart[gridHash];
                uint endIndex = cellEnd[gridHash];
                for (uint j = startIndex; j < endIndex; j++) {
                    if (sortedRhoPreMu[j].w != -1 || j == i_idx)
                        continue;

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
                    if (abs(sortedPosRad[j].w - fineResolution) < EPSILON && (MergeMe[j] == 0) &&
                        (sortedRhoPreMu[j].w == -1) && posj.x > 0 && dot(dist3, normal) > 0) {
                        //                        if (x != 0 || y != 0 || z != 0)
                        //                            continue;
                        uint p = 9;
                        if (d < length(Distance(posi, mR3(sortedPosRad[N8]))) || N8 == i_idx)
                            p = 8;
                        if (d < length(Distance(posi, mR3(sortedPosRad[N7]))) || N7 == i_idx)
                            p = 7;
                        if (d < length(Distance(posi, mR3(sortedPosRad[N6]))) || N6 == i_idx)
                            p = 6;
                        if (d < length(Distance(posi, mR3(sortedPosRad[N5]))) || N5 == i_idx)
                            p = 5;
                        if (d < length(Distance(posi, mR3(sortedPosRad[N4]))) || N4 == i_idx)
                            p = 4;
                        if (d < length(Distance(posi, mR3(sortedPosRad[N3]))) || N3 == i_idx)
                            p = 3;
                        if (d < length(Distance(posi, mR3(sortedPosRad[N2]))) || N2 == i_idx)
                            p = 2;
                        if (d < length(Distance(posi, mR3(sortedPosRad[N1]))) || N1 == i_idx)
                            p = 1;

                        //            if (abs(sortedPosRad[j].w - coarseResolution) < EPSILON &&
                        //            length(sortedPosRad[j]-sortedPosRad[j])<fineResolution)

                        // Release the last one if
                        if (p < 9) {
                            //                            atomicCAS(&MergeMe[j], 0, i_idx);
                            if (MergeMe[j] != i_idx)
                                continue;
                            // Release the last one since it is going to be replaced
                            //                            if (N8 != i_idx)
                            //                                atomicExch(&MergeMe[N8], 0);
                        }
                        if (p == 8) {
                            N8 = j;
                        } else if (p == 7) {
                            N8 = N7;
                            N7 = j;
                        } else if (p == 6) {
                            N8 = N7;
                            N7 = N6;
                            N6 = j;
                        } else if (p == 5) {
                            N8 = N7;
                            N7 = N6;
                            N6 = N5;
                            N5 = j;
                        } else if (p == 4) {
                            N8 = N7;
                            N7 = N6;
                            N6 = N5;
                            N5 = N4;
                            N4 = j;
                        } else if (p == 3) {
                            N8 = N7;
                            N7 = N6;
                            N6 = N5;
                            N5 = N4;
                            N4 = N3;
                            N3 = j;
                        } else if (p == 2) {
                            N8 = N7;
                            N7 = N6;
                            N6 = N5;
                            N5 = N4;
                            N4 = N3;
                            N3 = N2;
                            N2 = j;
                        } else if (p == 1) {
                            N8 = N7;
                            N7 = N6;
                            N6 = N5;
                            N5 = N4;
                            N4 = N3;
                            N3 = N2;
                            N2 = N1;
                            N1 = j;
                        }
                    } else if (abs(sortedPosRad[j].w - coarseResolution) < EPSILON && sortedRhoPreMu[j].w == -1 &&
                               splitMe[j] == 0 && myType[j] == -1 && sortedRhoPreMu[j].w == -1 &&
                               d < 2 * coarseResolution && cosTv < 0.0 && dot(dist3, normal) < 0 && posj.x < 0) {
                        //                    } else if (abs(sortedPosRad[j].w - coarseResolution) < EPSILON &&
                        //                    splitMe[j] == 0 &&
                        //                               MergeMe[j] == 0 && myType[j] == -1 && sortedRhoPreMu[j].w
                        //                               == -1
                        //                               && d < 1 * coarseResolution && cosTv < -0.5) {
                        //                        atomicCAS(&splitMe[j], 0, i_idx);
                    }
                }
            }
        }  // namespace fsi
    }      // namespace chrono

    // note that this can cause race condition if two helper markers try to merge the same marker Ni
    if (N8 != i_idx && N7 != i_idx && N6 != i_idx && N5 != i_idx && N4 != i_idx && N3 != i_idx && N2 != i_idx &&
        N1 != i_idx) {
        if (MergeMe[N1] != i_idx || MergeMe[N2] != i_idx || MergeMe[N3] != i_idx || MergeMe[N4] != i_idx ||
            MergeMe[N5] != i_idx || MergeMe[N6] != i_idx || MergeMe[N7] != i_idx || MergeMe[N8] != i_idx) {
            printf("RACE CONDITION in merging! Please revise the spacing or the merging scheme.\n");
            *isErrorD = true;
        }

        //        printf("idx=%d merging %d,%d,%d,%d,%d,%d,%d,%d\n", i_idx, N1, N2, N3, N4, N5, N6, N7, N8);
        if (sortedPosRad[N1].w != fineResolution || sortedPosRad[N2].w != fineResolution ||
            sortedPosRad[N3].w != fineResolution || sortedPosRad[N4].w != fineResolution ||
            sortedPosRad[N5].w != fineResolution || sortedPosRad[N6].w != fineResolution ||
            sortedPosRad[N7].w != fineResolution || sortedPosRad[N8].w != fineResolution) {
            printf("ops something went wrong!.\n");
            *isErrorD = true;
        }

        Real4 center = 0.125 * (sortedPosRad[N1] + sortedPosRad[N2] + sortedPosRad[N3] + sortedPosRad[N4] +
                                sortedPosRad[N5] + sortedPosRad[N6] + sortedPosRad[N7] + sortedPosRad[N8]);

        Real3 myGrad[8];
        grad_scalar(N1, sortedPosRad, sortedRhoPreMu, sumWij_inv, G_i, sortedRhoPreMu, myGrad[0], cellStart, cellEnd);
        grad_scalar(N2, sortedPosRad, sortedRhoPreMu, sumWij_inv, G_i, sortedRhoPreMu, myGrad[1], cellStart, cellEnd);
        grad_scalar(N3, sortedPosRad, sortedRhoPreMu, sumWij_inv, G_i, sortedRhoPreMu, myGrad[2], cellStart, cellEnd);
        grad_scalar(N4, sortedPosRad, sortedRhoPreMu, sumWij_inv, G_i, sortedRhoPreMu, myGrad[3], cellStart, cellEnd);
        grad_scalar(N5, sortedPosRad, sortedRhoPreMu, sumWij_inv, G_i, sortedRhoPreMu, myGrad[4], cellStart, cellEnd);
        grad_scalar(N6, sortedPosRad, sortedRhoPreMu, sumWij_inv, G_i, sortedRhoPreMu, myGrad[5], cellStart, cellEnd);
        grad_scalar(N7, sortedPosRad, sortedRhoPreMu, sumWij_inv, G_i, sortedRhoPreMu, myGrad[6], cellStart, cellEnd);
        grad_scalar(N8, sortedPosRad, sortedRhoPreMu, sumWij_inv, G_i, sortedRhoPreMu, myGrad[7], cellStart, cellEnd);
        sortedRhoPreMu[N1].x -= dot(myGrad[0], mR3(sortedPosRad[N1] - center));
        sortedRhoPreMu[N2].x -= dot(myGrad[1], mR3(sortedPosRad[N2] - center));
        sortedRhoPreMu[N3].x -= dot(myGrad[2], mR3(sortedPosRad[N3] - center));
        sortedRhoPreMu[N4].x -= dot(myGrad[3], mR3(sortedPosRad[N4] - center));
        sortedRhoPreMu[N5].x -= dot(myGrad[4], mR3(sortedPosRad[N5] - center));
        sortedRhoPreMu[N6].x -= dot(myGrad[5], mR3(sortedPosRad[N6] - center));
        sortedRhoPreMu[N7].x -= dot(myGrad[6], mR3(sortedPosRad[N7] - center));
        sortedRhoPreMu[N8].x -= dot(myGrad[7], mR3(sortedPosRad[N8] - center));

        Real4 rpmt = 0.125 * (sortedRhoPreMu[N1] + sortedRhoPreMu[N2] + sortedRhoPreMu[N3] + sortedRhoPreMu[N4] +
                              sortedRhoPreMu[N5] + sortedRhoPreMu[N6] + sortedRhoPreMu[N7] + sortedRhoPreMu[N8]);

        if (rpmt.x > 1.0 * paramsD.rho0)
            return;

        sortedRhoPreMu[N1] = rpmt;

        Real3 myGradx[8];
        Real3 myGrady[8];
        Real3 myGradz[8];
        grad_vector(N1, sortedPosRad, sortedRhoPreMu, sumWij_inv, G_i, sortedVelMas, myGradx[0], myGrady[0], myGradz[0],
                    cellStart, cellEnd);
        grad_vector(N2, sortedPosRad, sortedRhoPreMu, sumWij_inv, G_i, sortedVelMas, myGradx[1], myGrady[1], myGradz[1],
                    cellStart, cellEnd);
        grad_vector(N3, sortedPosRad, sortedRhoPreMu, sumWij_inv, G_i, sortedVelMas, myGradx[2], myGrady[2], myGradz[2],
                    cellStart, cellEnd);
        grad_vector(N4, sortedPosRad, sortedRhoPreMu, sumWij_inv, G_i, sortedVelMas, myGradx[3], myGrady[3], myGradz[3],
                    cellStart, cellEnd);
        grad_vector(N5, sortedPosRad, sortedRhoPreMu, sumWij_inv, G_i, sortedVelMas, myGradx[4], myGrady[4], myGradz[4],
                    cellStart, cellEnd);
        grad_vector(N6, sortedPosRad, sortedRhoPreMu, sumWij_inv, G_i, sortedVelMas, myGradx[5], myGrady[5], myGradz[5],
                    cellStart, cellEnd);
        grad_vector(N7, sortedPosRad, sortedRhoPreMu, sumWij_inv, G_i, sortedVelMas, myGradx[6], myGrady[6], myGradz[6],
                    cellStart, cellEnd);
        grad_vector(N8, sortedPosRad, sortedRhoPreMu, sumWij_inv, G_i, sortedVelMas, myGradx[7], myGrady[7], myGradz[7],
                    cellStart, cellEnd);

        sortedVelMas[N1].x -= dot(myGradx[0], mR3(sortedPosRad[N1] - center));
        sortedVelMas[N2].x -= dot(myGradx[1], mR3(sortedPosRad[N2] - center));
        sortedVelMas[N3].x -= dot(myGradx[2], mR3(sortedPosRad[N3] - center));
        sortedVelMas[N4].x -= dot(myGradx[3], mR3(sortedPosRad[N4] - center));
        sortedVelMas[N5].x -= dot(myGradx[4], mR3(sortedPosRad[N5] - center));
        sortedVelMas[N6].x -= dot(myGradx[5], mR3(sortedPosRad[N6] - center));
        sortedVelMas[N7].x -= dot(myGradx[6], mR3(sortedPosRad[N7] - center));
        sortedVelMas[N8].x -= dot(myGradx[7], mR3(sortedPosRad[N8] - center));

        sortedVelMas[N1].y -= dot(myGrady[0], mR3(sortedPosRad[N1] - center));
        sortedVelMas[N2].y -= dot(myGrady[1], mR3(sortedPosRad[N2] - center));
        sortedVelMas[N3].y -= dot(myGrady[2], mR3(sortedPosRad[N3] - center));
        sortedVelMas[N4].y -= dot(myGrady[3], mR3(sortedPosRad[N4] - center));
        sortedVelMas[N5].y -= dot(myGrady[4], mR3(sortedPosRad[N5] - center));
        sortedVelMas[N6].y -= dot(myGrady[5], mR3(sortedPosRad[N6] - center));
        sortedVelMas[N7].y -= dot(myGrady[6], mR3(sortedPosRad[N7] - center));
        sortedVelMas[N8].y -= dot(myGrady[7], mR3(sortedPosRad[N8] - center));

        sortedVelMas[N1].z -= dot(myGradz[0], mR3(sortedPosRad[N1] - center));
        sortedVelMas[N2].z -= dot(myGradz[1], mR3(sortedPosRad[N2] - center));
        sortedVelMas[N3].z -= dot(myGradz[2], mR3(sortedPosRad[N3] - center));
        sortedVelMas[N4].z -= dot(myGradz[3], mR3(sortedPosRad[N4] - center));
        sortedVelMas[N5].z -= dot(myGradz[4], mR3(sortedPosRad[N5] - center));
        sortedVelMas[N6].z -= dot(myGradz[5], mR3(sortedPosRad[N6] - center));
        sortedVelMas[N7].z -= dot(myGradz[6], mR3(sortedPosRad[N7] - center));
        sortedVelMas[N8].z -= dot(myGradz[7], mR3(sortedPosRad[N8] - center));

        sortedVelMas[N1] = 0.125 * (sortedVelMas[N1] + sortedVelMas[N2] + sortedVelMas[N3] + sortedVelMas[N4] +
                                    sortedVelMas[N5] + sortedVelMas[N6] + sortedVelMas[N7] + sortedVelMas[N8]);
        sortedPosRad[N1] = center;

        if (sortedPosRad[N1].w != fineResolution || sortedRhoPreMu[N1].w != -1) {
            printf("ops something went wrong!!!!\n");
            *isErrorD = true;
        }
        sortedPosRad[N1].w = coarseResolution;
        sortedRhoPreMu[N1].w = -1;
        sortedRhoPreMu[N2].w = -2;
        sortedRhoPreMu[N3].w = -2;
        sortedRhoPreMu[N4].w = -2;
        sortedRhoPreMu[N5].w = -2;
        sortedRhoPreMu[N6].w = -2;
        sortedRhoPreMu[N7].w = -2;
        sortedRhoPreMu[N8].w = -2;
        sortedPosRad[N2] = mR4(mR3(0.0, 0.0, -0.8), coarseResolution);
        sortedPosRad[N3] = mR4(mR3(0.0, 0.0, -0.8), coarseResolution);
        sortedPosRad[N4] = mR4(mR3(0.0, 0.0, -0.8), coarseResolution);
        sortedPosRad[N5] = mR4(mR3(0.0, 0.0, -0.8), coarseResolution);
        sortedPosRad[N6] = mR4(mR3(0.0, 0.0, -0.8), coarseResolution);
        sortedPosRad[N7] = mR4(mR3(0.0, 0.0, -0.8), coarseResolution);
        sortedPosRad[N8] = mR4(mR3(0.0, 0.0, -0.8), coarseResolution);
        sortedVelMas[N2] = mR3(0.0);
        sortedVelMas[N3] = mR3(0.0);
        sortedVelMas[N4] = mR3(0.0);
        sortedVelMas[N5] = mR3(0.0);
        sortedVelMas[N6] = mR3(0.0);
        sortedVelMas[N7] = mR3(0.0);
        sortedVelMas[N8] = mR3(0.0);
    }
}
//--------------------------------------------------------------------------------------------------------------------------------
inline __global__ void Split(Real4* sortedPosRad,
                             Real4* sortedRhoPreMu,
                             Real3* sortedVelMas,
                             Real3* helpers_normal,
                             Real* sumWij_inv,
                             Real* G_i,
                             Real* L_i,
                             uint* splitMe,
                             uint* MergeMe,
                             int* myType,
                             uint* cellStart,
                             uint* cellEnd,
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

    if (abs(sortedPosRad[i_idx].w - coarseResolution) < EPSILON && sortedRhoPreMu[i_idx].w == -1 &&
        splitMe[i_idx] != 0 && myType[i_idx] == -1) {
        uint temp = 0;

        MergeMe[i_idx] = -1;
        uint childMarkers[7] = {0};
        temp = 0;
        uint k = 0;
        int numpass = 0;
        Real4 RhoPreMu = sortedRhoPreMu[i_idx];

        while (k < 7) {
            //            atomicCAS(&myType[temp], -2, i_idx);
            if (myType[temp] == i_idx) {
                childMarkers[k] = temp;
                k++;
            }
            if (temp > numAllMarkers) {
                if (numpass < 2) {
                    temp = 0;
                    continue;
                }
                break;
                *isErrorD = true;
            }
            temp++;
        }
        for (int n = 0; n < 7; n++) {
            sortedVelMas[childMarkers[n]] = veli;
            sortedRhoPreMu[childMarkers[n]] = RhoPreMu;
            sumWij_inv[childMarkers[n]] = sumWij_inv[i_idx];

            for (int l = 0; l < 9; l++)
                G_i[childMarkers[n] * 9 + l] = G_i[i_idx * 9 + l];

            for (int l = 0; l < 6; l++)
                L_i[childMarkers[n] * 6 + l] = L_i[i_idx * 6 + l];
        }
        Real3 center = mR3(sortedPosRad[i_idx]);
        Real h = fineResolution;

        int N1 = childMarkers[0];
        int N2 = childMarkers[1];
        int N3 = childMarkers[2];
        int N4 = childMarkers[3];
        int N5 = childMarkers[4];
        int N6 = childMarkers[5];
        int N7 = childMarkers[6];
        int N8 = i_idx;
        Real d = 0.5 * fineResolution;
        // This is recommended for our confuguration from Vacondio et. al
        //                Real d = 0.72 * coarseResolution / 1.7321;
        sortedPosRad[N1] = mR4(center + d * mR3(-1, -1, -1), h);
        sortedPosRad[N2] = mR4(center + d * mR3(+1, -1, -1), h);
        sortedPosRad[N3] = mR4(center + d * mR3(+1, +1, -1), h);
        sortedPosRad[N4] = mR4(center + d * mR3(-1, +1, -1), h);
        sortedPosRad[N5] = mR4(center + d * mR3(-1, -1, +1), h);
        sortedPosRad[N6] = mR4(center + d * mR3(+1, -1, +1), h);
        sortedPosRad[N7] = mR4(center + d * mR3(+1, +1, +1), h);
        sortedPosRad[N8] = mR4(center + d * mR3(-1, +1, +1), h);

        Real3 myGrad[8] = {0};
        Real rho0 = +sortedRhoPreMu[N8].x;
        grad_scalar(N1, sortedPosRad, sortedRhoPreMu, sumWij_inv, G_i, sortedRhoPreMu, myGrad[0], cellStart, cellEnd);
        grad_scalar(N2, sortedPosRad, sortedRhoPreMu, sumWij_inv, G_i, sortedRhoPreMu, myGrad[1], cellStart, cellEnd);
        grad_scalar(N3, sortedPosRad, sortedRhoPreMu, sumWij_inv, G_i, sortedRhoPreMu, myGrad[2], cellStart, cellEnd);
        grad_scalar(N4, sortedPosRad, sortedRhoPreMu, sumWij_inv, G_i, sortedRhoPreMu, myGrad[3], cellStart, cellEnd);
        grad_scalar(N5, sortedPosRad, sortedRhoPreMu, sumWij_inv, G_i, sortedRhoPreMu, myGrad[4], cellStart, cellEnd);
        grad_scalar(N6, sortedPosRad, sortedRhoPreMu, sumWij_inv, G_i, sortedRhoPreMu, myGrad[5], cellStart, cellEnd);
        grad_scalar(N7, sortedPosRad, sortedRhoPreMu, sumWij_inv, G_i, sortedRhoPreMu, myGrad[6], cellStart, cellEnd);
        grad_scalar(N8, sortedPosRad, sortedRhoPreMu, sumWij_inv, G_i, sortedRhoPreMu, myGrad[7], cellStart, cellEnd);
        sortedRhoPreMu[N1].x += dot(myGrad[0], mR3(sortedPosRad[N1]) - center);
        sortedRhoPreMu[N2].x += dot(myGrad[1], mR3(sortedPosRad[N2]) - center);
        sortedRhoPreMu[N3].x += dot(myGrad[2], mR3(sortedPosRad[N3]) - center);
        sortedRhoPreMu[N4].x += dot(myGrad[3], mR3(sortedPosRad[N4]) - center);
        sortedRhoPreMu[N5].x += dot(myGrad[4], mR3(sortedPosRad[N5]) - center);
        sortedRhoPreMu[N6].x += dot(myGrad[5], mR3(sortedPosRad[N6]) - center);
        sortedRhoPreMu[N7].x += dot(myGrad[6], mR3(sortedPosRad[N7]) - center);
        sortedRhoPreMu[N8].x += dot(myGrad[7], mR3(sortedPosRad[N8]) - center);

        Real3 myGradx[8] = {0};
        Real3 myGrady[8] = {0};
        Real3 myGradz[8] = {0};
        grad_vector(N1, sortedPosRad, sortedRhoPreMu, sumWij_inv, G_i, sortedVelMas, myGradx[0], myGrady[0], myGradz[0],
                    cellStart, cellEnd);
        grad_vector(N2, sortedPosRad, sortedRhoPreMu, sumWij_inv, G_i, sortedVelMas, myGradx[1], myGrady[1], myGradz[1],
                    cellStart, cellEnd);
        grad_vector(N3, sortedPosRad, sortedRhoPreMu, sumWij_inv, G_i, sortedVelMas, myGradx[2], myGrady[2], myGradz[2],
                    cellStart, cellEnd);
        grad_vector(N4, sortedPosRad, sortedRhoPreMu, sumWij_inv, G_i, sortedVelMas, myGradx[3], myGrady[3], myGradz[3],
                    cellStart, cellEnd);
        grad_vector(N5, sortedPosRad, sortedRhoPreMu, sumWij_inv, G_i, sortedVelMas, myGradx[4], myGrady[4], myGradz[4],
                    cellStart, cellEnd);
        grad_vector(N6, sortedPosRad, sortedRhoPreMu, sumWij_inv, G_i, sortedVelMas, myGradx[5], myGrady[5], myGradz[5],
                    cellStart, cellEnd);
        grad_vector(N7, sortedPosRad, sortedRhoPreMu, sumWij_inv, G_i, sortedVelMas, myGradx[6], myGrady[6], myGradz[6],
                    cellStart, cellEnd);
        grad_vector(N8, sortedPosRad, sortedRhoPreMu, sumWij_inv, G_i, sortedVelMas, myGradx[7], myGrady[7], myGradz[7],
                    cellStart, cellEnd);

        sortedVelMas[N1].x += dot(myGradx[0], mR3(sortedPosRad[N1]) - center);
        sortedVelMas[N2].x += dot(myGradx[1], mR3(sortedPosRad[N2]) - center);
        sortedVelMas[N3].x += dot(myGradx[2], mR3(sortedPosRad[N3]) - center);
        sortedVelMas[N4].x += dot(myGradx[3], mR3(sortedPosRad[N4]) - center);
        sortedVelMas[N5].x += dot(myGradx[4], mR3(sortedPosRad[N5]) - center);
        sortedVelMas[N6].x += dot(myGradx[5], mR3(sortedPosRad[N6]) - center);
        sortedVelMas[N7].x += dot(myGradx[6], mR3(sortedPosRad[N7]) - center);
        sortedVelMas[N8].x += dot(myGradx[7], mR3(sortedPosRad[N8]) - center);

        sortedVelMas[N1].y += dot(myGrady[0], mR3(sortedPosRad[N1]) - center);
        sortedVelMas[N2].y += dot(myGrady[1], mR3(sortedPosRad[N2]) - center);
        sortedVelMas[N3].y += dot(myGrady[2], mR3(sortedPosRad[N3]) - center);
        sortedVelMas[N4].y += dot(myGrady[3], mR3(sortedPosRad[N4]) - center);
        sortedVelMas[N5].y += dot(myGrady[4], mR3(sortedPosRad[N5]) - center);
        sortedVelMas[N6].y += dot(myGrady[5], mR3(sortedPosRad[N6]) - center);
        sortedVelMas[N7].y += dot(myGrady[6], mR3(sortedPosRad[N7]) - center);
        sortedVelMas[N8].y += dot(myGrady[7], mR3(sortedPosRad[N8]) - center);

        sortedVelMas[N1].z += dot(myGradz[0], mR3(sortedPosRad[N1]) - center);
        sortedVelMas[N2].z += dot(myGradz[1], mR3(sortedPosRad[N2]) - center);
        sortedVelMas[N3].z += dot(myGradz[2], mR3(sortedPosRad[N3]) - center);
        sortedVelMas[N4].z += dot(myGradz[3], mR3(sortedPosRad[N4]) - center);
        sortedVelMas[N5].z += dot(myGradz[4], mR3(sortedPosRad[N5]) - center);
        sortedVelMas[N6].z += dot(myGradz[5], mR3(sortedPosRad[N6]) - center);
        sortedVelMas[N7].z += dot(myGradz[6], mR3(sortedPosRad[N7]) - center);
        sortedVelMas[N8].z += dot(myGradz[7], mR3(sortedPosRad[N8]) - center);

        //        printf("extrapolate vel of marker %d with %f\n", N1, dot(myGradx[0], mR3(sortedPosRad[N1])
        //        - center));

        if (temp > numAllMarkers)
            printf(
                "Reached the limit of the ghost markers. Please increase the number of ghost "
                "markers.\n");
    }
}
//--------------------------------------------------------------------------------------------------------------------------------

}  // namespace fsi
}  // namespace chrono
#endif
