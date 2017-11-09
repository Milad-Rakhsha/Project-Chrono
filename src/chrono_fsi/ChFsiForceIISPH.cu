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
#include <thrust/extrema.h>
#include <thrust/sort.h>
#include "chrono_fsi/ChFsiForceIISPH.cuh"

//==========================================================================================================================================
namespace chrono {
namespace fsi {

ChFsiForceIISPH::ChFsiForceIISPH(
    ChBce* otherBceWorker,                   ///< Pointer to the ChBce object that handles BCE markers
    SphMarkerDataD* otherSortedSphMarkersD,  ///< Information of markers in the sorted array on device
    ProximityDataD*
        otherMarkersProximityD,           ///< Pointer to the object that holds the proximity of the markers on device
    FsiGeneralData* otherFsiGeneralData,  ///< Pointer to the sph general data
    SimParams* otherParamsH,              ///< Pointer to the simulation parameters on host
    NumberOfObjects* otherNumObjects      ///< Pointer to number of objects, fluid and boundary markers, etc.
    )
    : ChFsiForceParallel(otherBceWorker,
                         otherSortedSphMarkersD,
                         otherMarkersProximityD,
                         otherFsiGeneralData,
                         otherParamsH,
                         otherNumObjects) {}
//--------------------------------------------------------------------------------------------------------------------------------
ChFsiForceIISPH::~ChFsiForceIISPH() {}
//--------------------------------------------------------------------------------------------------------------------------------
void ChFsiForceIISPH::Finalize() {
    ChFsiForceParallel::Finalize();
    cudaMemcpyToSymbolAsync(paramsD, paramsH, sizeof(SimParams));
    cudaMemcpyToSymbolAsync(numObjectsD, numObjectsH, sizeof(NumberOfObjects));
    cudaMemcpyFromSymbol(paramsH, paramsD, sizeof(SimParams));
    cudaThreadSynchronize();
}
//--------------------------------------------------------------------------------------------------------------------------------
__global__ void V_i_np__AND__d_ii_kernel(Real4* sortedPosRad,  // input: sorted positions
                                         Real3* sortedVelMas,
                                         Real4* sortedRhoPreMu,
                                         Real3* d_ii,
                                         Real3* V_i_np,
                                         Real* sumWij_inv,
                                         Real* G_tensor,
                                         Real* L_tensor,
                                         uint* cellStart,
                                         uint* cellEnd,
                                         const int numAllMarkers,
                                         volatile bool* isErrorD) {
    uint i_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_idx >= numAllMarkers || sortedRhoPreMu[i_idx].w <= -2) {
        return;
    }
    //    sortedRhoPreMu[i_idx].x = sortedRhoPreMu[i_idx].x / sumWij_inv[i_idx];
    Real h_i = sortedPosRad[i_idx].w;
    Real m_i = h_i * h_i * h_i * paramsD.rho0;

    Real mu_0 = paramsD.mu0;
    Real epsilon = paramsD.epsMinMarkersDis;
    Real dT = paramsD.dT;
    Real3 gravity = paramsD.gravity;
    Real RHO_0 = paramsD.rho0;
    if (sortedRhoPreMu[i_idx].x < EPSILON) {
        printf("My density is %f,ref density= %f\n", sortedRhoPreMu[i_idx].x, RHO_0);
    }

    Real3 posi = mR3(sortedPosRad[i_idx]);
    Real3 Veli = sortedVelMas[i_idx];
    Real Rhoi = sortedRhoPreMu[i_idx].x;
    Real3 My_d_ii = mR3(0);
    Real3 My_F_i_np = mR3(0);

    Real3 myGradvx = mR3(0);
    Real3 myGradvy = mR3(0);
    Real3 myGradvz = mR3(0);
    Real Gi[9] = {0.0};
    Real Li[6] = {0.0};
    Real3 LaplacainVi = mR3(0.0);
    for (int i = 0; i < 9; i++)
        Gi[i] = G_tensor[i_idx * 9 + i];
    for (int i = 0; i < 6; i++)
        Li[i] = L_tensor[i_idx * 6 + i];

    grad_vector(i_idx, sortedPosRad, sortedRhoPreMu, sumWij_inv, G_tensor, sortedVelMas, myGradvx, myGradvy, myGradvz,
                cellStart, cellEnd);

    // get address in grid
    int3 gridPos = calcGridPos(posi);

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
                        Real3 posj = mR3(sortedPosRad[j]);
                        Real3 rij = Distance(posi, posj);
                        Real d = length(rij);

                        if (d > RESOLUTION_LENGTH_MULT * h_i || sortedRhoPreMu[j].w <= -2 || i_idx == j)
                            continue;
                        Real3 eij = rij / d;

                        Real3 Velj = sortedVelMas[j];
                        Real Rhoj = sortedRhoPreMu[j].x;
                        Real h_j = sortedPosRad[j].w;

                        if (Rhoj == 0) {
                            printf("Bug F_i_np__AND__d_ii_kernel i=%d j=%d, hi=%f, hj=%f\n", i_idx, j, h_i, h_j);
                        }

                        Real m_j = h_j * h_j * h_j * paramsD.rho0;
                        Real h_ij = 0.5 * (h_j + h_i);
                        Real3 grad_ij = GradWh(rij, h_ij);
                        My_d_ii += m_j * (-(dT * dT) / (Rhoi * Rhoi)) * grad_ij;

                        Real Rho_bar = (Rhoj + Rhoi) * 0.5;
                        Real3 V_ij = (Veli - Velj);
                        //                        Real nu = mu_0 * paramsD.HSML * 320 / Rho_bar;
                        //            Real3 muNumerator = nu * fmin(0.0, dot(rij, V_ij)) * grad_i_wij;
                        Real3 muNumerator = 2 * mu_0 * dot(rij, grad_ij) * V_ij;
                        Real muDenominator = (Rho_bar * Rho_bar) * (d * d + h_ij * h_ij * epsilon);
                        My_F_i_np += m_j * muNumerator / muDenominator;

                        Real Wd = W3h(d, h_ij);
                        My_F_i_np -= paramsD.kappa / m_i * m_j * Wd * rij;

                        Real Vj = sumWij_inv[j];
                        Real commonterm =
                            2 * Vj *
                            (Li[0] * eij.x * grad_ij.x + Li[1] * eij.x * grad_ij.y + Li[2] * eij.x * grad_ij.z +
                             Li[1] * eij.y * grad_ij.x + Li[3] * eij.y * grad_ij.y + Li[4] * eij.y * grad_ij.z +
                             Li[2] * eij.z * grad_ij.x + Li[4] * eij.z * grad_ij.y + Li[5] * eij.z * grad_ij.z);

                        LaplacainVi.x += commonterm * (V_ij.x / d - dot(eij, myGradvx));
                        LaplacainVi.y += commonterm * (V_ij.y / d - dot(eij, myGradvy));
                        LaplacainVi.z += commonterm * (V_ij.y / d - dot(eij, myGradvz));
                    }
                }
            }
        }
    }

    My_F_i_np = mu_0 * LaplacainVi;

    My_F_i_np *= m_i;

    My_F_i_np += m_i * gravity;
    d_ii[i_idx] = My_d_ii;
    V_i_np[i_idx] = (My_F_i_np * dT + Veli);  // This does not contain m_0?
}
//--------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------
__global__ void Rho_np_AND_a_ii_AND_sum_m_GradW(Real4* sortedPosRad,
                                                Real4* sortedRhoPreMu,
                                                Real* rho_np,  // Write
                                                Real* a_ii,    // Write
                                                Real* p_old,   // Write
                                                Real3* V_np,   // Read
                                                Real3* d_ii,   // Read
                                                Real3* sum_m_GradW,
                                                uint* cellStart,
                                                uint* cellEnd,
                                                const int numAllMarkers,
                                                volatile bool* isErrorD) {
    uint i_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_idx >= numAllMarkers || sortedRhoPreMu[i_idx].w <= -2) {
        return;
    }

    Real h_i = sortedPosRad[i_idx].w;
    Real m_i = h_i * h_i * h_i * paramsD.rho0;

    Real3 posi = mR3(sortedPosRad[i_idx]);
    Real3 Veli_np = V_np[i_idx];
    Real Rho_i = sortedRhoPreMu[i_idx].x;
    Real3 my_d_ii = d_ii[i_idx];
    Real rho_temp = 0;
    Real my_a_ii = 0;
    Real3 My_sum_m_gradW = mR3(0);
    Real dT = paramsD.dT;
    // get address in gridj
    int3 gridPos = calcGridPos(posi);

    //
    // examine neighbouring cells
    for (int z = -1; z <= 1; z++) {
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                int3 neighbourPos = gridPos + mI3(x, y, z);
                uint gridHash = calcGridHash(neighbourPos);
                // get start of bucket for this cell
                uint startIndex = cellStart[gridHash];
                if (startIndex != 0xffffffff) {  // cell is not empty
                    // iterate over particles in this cell
                    uint endIndex = cellEnd[gridHash];

                    for (uint j = startIndex; j < endIndex; j++) {
                        Real3 posj = mR3(sortedPosRad[j]);
                        Real3 dist3 = Distance(posi, posj);
                        Real d = length(dist3);
                        if (d > RESOLUTION_LENGTH_MULT * h_i || sortedRhoPreMu[j].w <= -2 || i_idx == j)
                            continue;
                        Real h_j = sortedPosRad[j].w;
                        Real m_j = h_j * h_j * h_j * paramsD.rho0;
                        Real h_ij = 0.5 * (h_j + h_i);
                        Real3 Velj_np = V_np[j];
                        Real3 grad_i_wij = GradWh(dist3, h_ij);
                        rho_temp += m_j * dot((Veli_np - Velj_np), grad_i_wij);
                        Real3 d_ji = m_i * (-(dT * dT) / (Rho_i * Rho_i)) * (-grad_i_wij);
                        my_a_ii += m_j * dot((my_d_ii - d_ji), grad_i_wij);
                        My_sum_m_gradW += m_j * grad_i_wij;
                    }
                }
            }
        }
    }
    rho_np[i_idx] = dT * rho_temp + sortedRhoPreMu[i_idx].x;

    a_ii[i_idx] = my_a_ii;
    sum_m_GradW[i_idx] = My_sum_m_gradW;

    p_old[i_idx] = sortedRhoPreMu[i_idx].y;  // = 1000;  // Note that this is outside of the for loop
}
//--------------------------------------------------------------------------------------------------------------------------------

__global__ void Calc_dij_pj(Real3* dij_pj,  // write
                            Real3* F_p,     // Write
                            Real3* d_ii,    // Read
                            Real4* sortedPosRad,
                            Real3* sortedVelMas,
                            Real4* sortedRhoPreMu,
                            Real* p_old,
                            uint* cellStart,
                            uint* cellEnd,
                            const int numAllMarkers,
                            volatile bool* isErrorD) {
    uint i_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_idx >= numAllMarkers || sortedRhoPreMu[i_idx].w <= -2) {
        return;
    }
    Real h_i = sortedPosRad[i_idx].w;
    Real m_i = h_i * h_i * h_i * paramsD.rho0;

    Real3 my_F_p = mR3(0);
    Real p_i_old = p_old[i_idx];
    Real3 pos_i = mR3(sortedPosRad[i_idx]);
    Real Rho_i = sortedRhoPreMu[i_idx].x;
    if (sortedRhoPreMu[i_idx].x < EPSILON) {
        printf("(Calc_dij_pj) My density is %f in Calc_dij_pj\n", sortedRhoPreMu[i_idx].x);
    }
    Real dT = paramsD.dT;

    Real3 My_dij_pj = mR3(0);
    int3 gridPos = calcGridPos(pos_i);
    for (int z = -1; z <= 1; z++) {
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                int3 neighbourPos = gridPos + mI3(x, y, z);
                uint gridHash = calcGridHash(neighbourPos);
                // get start of bucket for this cell
                uint startIndex = cellStart[gridHash];
                if (startIndex != 0xffffffff) {  // cell is not empty
                    // iterate over particles in this cell
                    uint endIndex = cellEnd[gridHash];

                    for (uint j = startIndex; j < endIndex; j++) {
                        Real3 pos_j = mR3(sortedPosRad[j]);
                        Real3 dist3 = Distance(pos_i, pos_j);
                        Real d = length(dist3);
                        ////CHECK THIS CONDITION!!!
                        if (d > RESOLUTION_LENGTH_MULT * h_i || sortedRhoPreMu[j].w <= -2 || i_idx == j)
                            continue;
                        Real h_j = sortedPosRad[j].w;
                        Real m_j = h_j * h_j * h_j * paramsD.rho0;
                        Real h_ij = 0.5 * (h_j + h_i);
                        Real3 grad_i_wij = GradWh(dist3, h_ij);
                        Real Rho_j = sortedRhoPreMu[j].x;
                        Real p_j_old = p_old[j];
                        My_dij_pj += m_j * (-(dT * dT) / (Rho_j * Rho_j)) * grad_i_wij * p_j_old;
                        my_F_p += m_j * ((p_i_old / (Rho_i * Rho_i)) + (p_j_old / (Rho_j * Rho_j))) * grad_i_wij;
                    }
                }
            }
        }
    }
    dij_pj[i_idx] = My_dij_pj;
    F_p[i_idx] = -m_i * my_F_p;
}

////--------------------------------------------------------------------------------------------------------------------------------
__global__ void CalcNumber_Contacts(uint* numContacts,
                                    Real4* sortedPosRad,
                                    Real4* sortedRhoPreMu,
                                    uint* cellStart,
                                    uint* cellEnd,
                                    const int numAllMarkers,
                                    volatile bool* isErrorD) {
    uint i_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_idx >= numAllMarkers) {
        return;
    }
    if (sortedRhoPreMu[i_idx].w <= -2) {
        numContacts[i_idx] = 1;
        return;
    }

    Real h_i = sortedPosRad[i_idx].w;
    //    Real m_i = h_i * h_i * h_i * paramsD.rho0;

    int myType = sortedRhoPreMu[i_idx].w;
    Real3 pos_i = mR3(sortedPosRad[i_idx]);

    uint numCol[800];
    int counter = 1;
    numCol[0] = i_idx;  // The first one is always the idx of the marker itself

    int3 gridPos = calcGridPos(pos_i);
    for (int z = -1; z <= 1; z++) {
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                int3 neighbourPos = gridPos + mI3(x, y, z);
                uint gridHash = calcGridHash(neighbourPos);
                // get start of bucket for this cell
                uint startIndex = cellStart[gridHash];
                if (startIndex != 0xffffffff) {
                    // iterate over particles in this cell
                    uint endIndex = cellEnd[gridHash];
                    for (uint j = startIndex; j < endIndex; j++) {
                        Real3 pos_j = mR3(sortedPosRad[j]);
                        Real3 dist3 = Distance(pos_i, pos_j);
                        Real d = length(dist3);
                        if (d > RESOLUTION_LENGTH_MULT * h_i || sortedRhoPreMu[j].w <= -2 || i_idx == j)
                            continue;
                        bool AlreadyHave = false;
                        for (uint findCol = 1; findCol <= counter; findCol++) {
                            if (numCol[findCol] == j) {
                                AlreadyHave = true;
                                continue;
                            }
                        }

                        // Room for improvment ...
                        if (!AlreadyHave) {
                            numCol[counter] = j;
                            counter++;
                            if (myType >= 0 && sortedRhoPreMu[j].w >= 0)  // Do not count BCE-BCE interactions...
                                counter--;
                        }

                        if (myType != -1)  // For BCE no need to go deeper than this...
                            continue;

                        Real h_j = sortedPosRad[j].w;
                        int3 gridPosJ = calcGridPos(pos_j);

                        for (int zz = -1; zz <= 1; zz++) {
                            for (int yy = -1; yy <= 1; yy++) {
                                for (int xx = -1; xx <= 1; xx++) {
                                    int3 neighbourPosJ = gridPosJ + mI3(xx, yy, zz);
                                    uint gridHashJ = calcGridHash(neighbourPosJ);
                                    uint startIndexJ = cellStart[gridHashJ];
                                    if (startIndexJ != 0xffffffff) {  // cell is not empty
                                        uint endIndexJ = cellEnd[gridHashJ];
                                        for (uint k = startIndexJ; k < endIndexJ; k++) {
                                            Real3 pos_k = mR3(sortedPosRad[k]);
                                            Real3 dist3jk = Distance(pos_j, pos_k);
                                            Real djk = length(dist3jk);
                                            if (djk > RESOLUTION_LENGTH_MULT * h_j || k == j || k == i_idx ||
                                                sortedRhoPreMu[k].w <= -2)
                                                continue;
                                            bool AlreadyHave2 = false;
                                            for (uint findCol = 1; findCol <= counter; findCol++) {
                                                if (numCol[findCol] == k) {
                                                    AlreadyHave2 = true;
                                                    continue;
                                                }
                                            }
                                            if (!AlreadyHave2) {
                                                numCol[counter] = k;
                                                counter++;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        ///////////////////////////////
                    }
                }
            }
        }
    }

    numContacts[i_idx] = counter + 10;
}

////--------------------------------------------------------------------------------------------------------------------------------
__global__ void Calc_summGradW(Real3* summGradW,  // write
                               Real4* sortedPosRad,
                               Real4* sortedRhoPreMu,
                               uint* cellStart,
                               uint* cellEnd,
                               const int numAllMarkers,
                               volatile bool* isErrorD) {
    uint i_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_idx >= numAllMarkers) {
        return;
    }
    if (sortedRhoPreMu[i_idx].w <= -2) {
        return;
    }
    Real h_i = sortedPosRad[i_idx].w;
    //    Real m_i = h_i * h_i * h_i * paramsD.rho0;

    Real3 pos_i = mR3(sortedPosRad[i_idx]);
    Real3 My_summgradW = mR3(0);
    //    Real dT = paramsD.dT;
    int3 gridPos = calcGridPos(pos_i);
    for (int z = -1; z <= 1; z++) {
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                int3 neighbourPos = gridPos + mI3(x, y, z);
                uint gridHash = calcGridHash(neighbourPos);
                // get start of bucket for this cell
                uint startIndex = cellStart[gridHash];
                if (startIndex != 0xffffffff) {  // cell is not empty
                    // iterate over particles in this cell
                    uint endIndex = cellEnd[gridHash];

                    for (uint j = startIndex; j < endIndex; j++) {
                        Real3 pos_j = mR3(sortedPosRad[j]);
                        Real3 dist3 = Distance(pos_i, pos_j);
                        Real d = length(dist3);
                        if (d > RESOLUTION_LENGTH_MULT * h_i || sortedRhoPreMu[j].w <= -2 || i_idx == j)
                            continue;
                        Real h_j = sortedPosRad[j].w;
                        Real m_j = h_j * h_j * h_j * paramsD.rho0;
                        Real h_ij = 0.5 * (h_j + h_i);
                        Real3 grad_i_wij = GradWh(dist3, h_ij);
                        My_summgradW += m_j * grad_i_wij;
                    }
                }
            }
        }
    }
    summGradW[i_idx] = My_summgradW;
}

////--------------------------------------------------------------------------------------------------------------------------------
__device__ void Calc_BC_aij_Bi(const uint i_idx,
                               Real* csrValA,
                               uint* csrColIndA,
                               unsigned long int* GlobalcsrColIndA,
                               uint* numContacts,
                               ///> The above 4 vectors are used for CSR form.
                               Real* a_ii,  // write
                               Real* B_i,
                               Real4* sortedPosRad,
                               Real3* sortedVelMas,
                               const Real4* sortedRhoPreMu,
                               Real3* V_new,
                               Real* p_old,

                               Real4* velMassRigid_fsiBodies_D,
                               Real3* accRigid_fsiBodies_D,
                               uint* rigidIdentifierD,

                               Real3* pos_fsi_fea_D,
                               Real3* vel_fsi_fea_D,
                               Real3* acc_fsi_fea_D,
                               uint* FlexIdentifierD,
                               const int numFlex1D,
                               uint2* CableElementsNodes,
                               uint4* ShellelementsNodes,

                               int4 updatePortion,
                               uint* gridMarkerIndexD,
                               const uint* cellStart,
                               const uint* cellEnd,
                               const int numAllMarkers,
                               bool IsSPARSE) {
    uint csrStartIdx = numContacts[i_idx] + 1;
    uint csrEndIdx = numContacts[i_idx + 1];

    Real h_i = sortedPosRad[i_idx].w;
    //    Real m_i = h_i * h_i * h_i * paramsD.rho0;

    Real3 gravity = paramsD.gravity;
    //  if (bceIndex >= numObjectsD.numRigid_SphMarkers) {
    //    return;
    //  }

    //  int Original_idx = gridMarkerIndexD[i_idx];
    Real3 myAcc;
    Real3 V_prescribed;

    if (!(sortedRhoPreMu[i_idx].w >= 0 && sortedRhoPreMu[i_idx].w <= 2))
        printf("type of marker is %f\n", sortedRhoPreMu[i_idx].w);

    BCE_Vel_Acc(i_idx, myAcc, V_prescribed, sortedPosRad, updatePortion, gridMarkerIndexD, velMassRigid_fsiBodies_D,
                accRigid_fsiBodies_D, rigidIdentifierD, pos_fsi_fea_D, vel_fsi_fea_D, acc_fsi_fea_D, FlexIdentifierD,
                numFlex1D, CableElementsNodes, ShellelementsNodes);

    for (int c = csrStartIdx; c < csrEndIdx; c++) {
        csrValA[c] = 0;
        csrColIndA[c] = i_idx;
        GlobalcsrColIndA[c] = i_idx + numAllMarkers * i_idx;
    }

    // if ((csrEndIdx - csrStartIdx) != uint(0)) {
    Real3 numeratorv = mR3(0);
    Real denumenator = 0;
    Real pRHS = 0;
    //  Real Rho_i = sortedRhoPreMu[i_idx].x;
    Real3 pos_i = mR3(sortedPosRad[i_idx]);
    // get address in grid
    int3 gridPos = calcGridPos(pos_i);

    uint counter = 0;
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
                        Real3 pos_j = mR3(sortedPosRad[j]);
                        Real3 dist3 = Distance(pos_i, pos_j);
                        Real d = length(dist3);
                        if (d > RESOLUTION_LENGTH_MULT * h_i || sortedRhoPreMu[j].w <= -2 || sortedRhoPreMu[j].w != -1)
                            continue;
                        Real3 Vel_j = sortedVelMas[j];
                        Real h_j = sortedPosRad[j].w;
                        Real h_ij = 0.5 * (h_j + h_i);
                        Real Wd = W3h(d, h_ij);
                        numeratorv += Vel_j * Wd;
                        pRHS += dot(gravity - myAcc, dist3) * sortedRhoPreMu[j].x * Wd;
                        denumenator += Wd;
                        csrValA[counter + csrStartIdx] = -Wd;
                        csrColIndA[counter + csrStartIdx] = j;
                        GlobalcsrColIndA[counter + csrStartIdx] = j + numAllMarkers * i_idx;
                        counter++;
                    }
                }
            }
        }
    }

    //  if (abs(Scaling) < EPSILON)
    //    Scaling = -paramsD.HSML;

    if (abs(denumenator) < EPSILON) {
        V_new[i_idx] = 2 * V_prescribed;
        B_i[i_idx] = 0;

        csrValA[csrStartIdx - 1] = a_ii[i_idx];
        csrColIndA[csrStartIdx - 1] = i_idx;
        GlobalcsrColIndA[csrStartIdx - 1] = i_idx + numAllMarkers * i_idx;
    } else {
        Real Scaling = a_ii[i_idx] / denumenator;

        V_new[i_idx] = 2 * V_prescribed - numeratorv / denumenator;
        B_i[i_idx] = pRHS;
        csrValA[csrStartIdx - 1] = denumenator;
        csrColIndA[csrStartIdx - 1] = i_idx;
        GlobalcsrColIndA[csrStartIdx - 1] = i_idx + numAllMarkers * i_idx;

        for (int i = csrStartIdx - 1; i < csrEndIdx; i++)
            csrValA[i] *= Scaling;
        B_i[i_idx] *= Scaling;
    }

    sortedVelMas[i_idx] = V_new[i_idx];
}
//--------------------------------------------------------------------------------------------------------------------------------

////--------------------------------------------------------------------------------------------------------------------------------
__device__ void Calc_fluid_aij_Bi(const uint i_idx,
                                  Real* csrValA,
                                  uint* csrColIndA,
                                  unsigned long int* GlobalcsrColIndA,
                                  uint* numContacts,
                                  ///> The above 4 vectors are used for CSR form.
                                  Real* B_i,
                                  Real3* d_ii,   // Read
                                  Real* a_ii,    // Read
                                  Real* rho_np,  // Read
                                  Real3* summGradW,
                                  Real4* sortedPosRad,
                                  Real4* sortedRhoPreMu,
                                  uint* cellStart,
                                  uint* cellEnd,
                                  const int numAllMarkers,
                                  bool IsSPARSE) {
    Real3 pos_i = mR3(sortedPosRad[i_idx]);
    Real dT = paramsD.dT;

    int counter = 0;  // There is always one non-zero at each row- The marker itself
    B_i[i_idx] = paramsD.rho0 - rho_np[i_idx];

    uint csrStartIdx = numContacts[i_idx] + 1;  // Reserve the starting index for the A_ii
    uint csrEndIdx = numContacts[i_idx + 1];

    Real h_i = sortedPosRad[i_idx].w;
    //    Real m_i = h_i * h_i * h_i * paramsD.rho0;

    //  for (int c = csrStartIdx; c < csrEndIdx; c++) {
    //    csrValA[c] = a_ii[i_idx];
    //    csrColIndA[c] = i_idx;
    //    GlobalcsrColIndA[c] = i_idx + numAllMarkers * i_idx;
    //  }

    int3 gridPos = calcGridPos(pos_i);
    for (int z = -1; z <= 1; z++) {
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                int3 neighbourPos = gridPos + mI3(x, y, z);
                uint gridHash = calcGridHash(neighbourPos);
                // get start of bucket for this cell
                uint startIndex = cellStart[gridHash];
                if (startIndex != 0xffffffff) {  // cell is not empty
                    // iterate over particles in this cell
                    uint endIndex = cellEnd[gridHash];
                    //          Real Rho_i = sortedRhoPreMu[i_idx].x;

                    for (uint j = startIndex; j < endIndex; j++) {
                        Real3 pos_j = mR3(sortedPosRad[j]);
                        Real3 dist3 = Distance(pos_i, pos_j);
                        Real d = length(dist3);
                        if (d > RESOLUTION_LENGTH_MULT * h_i || sortedRhoPreMu[j].w <= -2 || i_idx == j)
                            continue;
                        Real h_j = sortedPosRad[j].w;
                        Real h_ij = 0.5 * (h_j + h_i);
                        Real3 grad_i_wij = GradWh(dist3, h_ij);
                        Real Rho_j = sortedRhoPreMu[j].x;
                        Real m_j = h_j * h_j * h_j * paramsD.rho0;
                        Real3 d_it = m_j * (-(dT * dT) / (Rho_j * Rho_j)) * grad_i_wij;
                        Real My_a_ij_1 = m_j * dot(d_it, summGradW[i_idx]);
                        Real My_a_ij_2 = m_j * dot(d_ii[j], grad_i_wij);
                        Real My_a_ij_12 = My_a_ij_1 - My_a_ij_2;
                        bool DONE1 = false;

                        for (uint findCol = csrStartIdx; findCol < csrEndIdx; findCol++) {
                            if (csrColIndA[findCol] == j) {
                                csrValA[findCol] += My_a_ij_12;
                                csrColIndA[findCol] = j;
                                GlobalcsrColIndA[findCol] = j + numAllMarkers * i_idx;
                                DONE1 = true;
                                continue;
                            }
                        }
                        if (!DONE1) {
                            csrValA[counter + csrStartIdx] += My_a_ij_12;
                            csrColIndA[counter + csrStartIdx] = j;
                            GlobalcsrColIndA[counter + csrStartIdx] = j + numAllMarkers * i_idx;
                            counter++;
                        }
                        int3 gridPosJ = calcGridPos(pos_j);
                        for (int zz = -1; zz <= 1; zz++) {
                            for (int yy = -1; yy <= 1; yy++) {
                                for (int xx = -1; xx <= 1; xx++) {
                                    int3 neighbourPosJ = gridPosJ + mI3(xx, yy, zz);
                                    uint gridHashJ = calcGridHash(neighbourPosJ);
                                    uint startIndexJ = cellStart[gridHashJ];
                                    if (startIndexJ != 0xffffffff) {  // cell is not empty
                                        uint endIndexJ = cellEnd[gridHashJ];
                                        for (uint k = startIndexJ; k < endIndexJ; k++) {
                                            Real3 pos_k = mR3(sortedPosRad[k]);
                                            Real3 dist3jk = Distance(pos_j, pos_k);
                                            Real djk = length(dist3jk);
                                            if (djk > RESOLUTION_LENGTH_MULT * h_j || k == j || k == i_idx ||
                                                sortedRhoPreMu[k].w <= -2)
                                                continue;
                                            Real h_k = sortedPosRad[j].w;
                                            Real h_jk = 0.5 * (h_j + h_k);
                                            Real3 grad_j_wjk = GradWh(dist3jk, h_jk);
                                            Real m_k = pow(sortedPosRad[k].w, 3) * paramsD.rho0;
                                            Real Rho_k = sortedRhoPreMu[k].x;
                                            Real3 d_jk = m_k * (-(dT * dT) / (Rho_k * Rho_k)) * grad_j_wjk;
                                            Real My_a_ij_3 = m_j * dot(d_jk, grad_i_wij);
                                            bool DONE2 = false;

                                            for (uint findCol = csrStartIdx; findCol < csrEndIdx; findCol++) {
                                                if (csrColIndA[findCol] == k) {
                                                    csrValA[findCol] -= My_a_ij_3;
                                                    csrColIndA[findCol] = k;
                                                    GlobalcsrColIndA[findCol] = k + numAllMarkers * i_idx;
                                                    DONE2 = true;
                                                    continue;
                                                }
                                            }
                                            if (!DONE2) {
                                                csrValA[counter + csrStartIdx] -= My_a_ij_3;
                                                csrColIndA[counter + csrStartIdx] = k;
                                                GlobalcsrColIndA[counter + csrStartIdx] = k + numAllMarkers * i_idx;
                                                counter++;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        ///////////////////////////////
                    }
                }
            }
        }
    }

    for (int myIdx = csrStartIdx; myIdx < csrEndIdx; myIdx++) {
        if (csrColIndA[myIdx] == i_idx)
            csrValA[myIdx] = a_ii[i_idx];
    }

    csrValA[csrStartIdx - 1] = a_ii[i_idx];
    csrColIndA[csrStartIdx - 1] = i_idx;
    GlobalcsrColIndA[csrStartIdx - 1] = i_idx + numAllMarkers * i_idx;

    if (sortedRhoPreMu[i_idx].x < 0.999 * paramsD.rho0) {
        csrValA[csrStartIdx - 1] = a_ii[i_idx];
        for (int myIdx = csrStartIdx; myIdx < csrEndIdx; myIdx++) {
            csrValA[myIdx] = 0.0;
            B_i[i_idx] = 0.0;
        }
    }

    Real RHS = B_i[i_idx];
    B_i[i_idx] = RHS;  // fminf(0.0, RHS);
}

//--------------------------------------------------------------------------------------------------------------------------------
__global__ void FormAXB(Real* csrValA,
                        uint* csrColIndA,
                        unsigned long int* GlobalcsrColIndA,

                        uint* numContacts,
                        ///> The above 4 vectors are used for CSR form.
                        Real* a_ij,   // write
                        Real* B_i,    // write
                        Real3* d_ii,  // Read
                        Real* a_ii,   // Read
                        Real3* summGradW,
                        Real4* sortedPosRad,
                        Real3* sortedVelMas,
                        Real4* sortedRhoPreMu,
                        Real3* V_new,
                        Real* p_old,
                        Real* rho_np,

                        Real4* velMassRigid_fsiBodies_D,
                        Real3* accRigid_fsiBodies_D,
                        uint* rigidIdentifierD,

                        Real3* pos_fsi_fea_D,
                        Real3* vel_fsi_fea_D,
                        Real3* acc_fsi_fea_D,
                        uint* FlexIdentifierD,
                        const int numFlex1D,
                        uint2* CableElementsNodes,
                        uint4* ShellelementsNodes,

                        int4 updatePortion,
                        uint* gridMarkerIndexD,
                        uint* cellStart,
                        uint* cellEnd,
                        const int numAllMarkers,
                        bool IsSPARSE,
                        volatile bool* isError) {
    uint i_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_idx >= numAllMarkers) {
        return;
    }

    //    Real m_0 = paramsD.markerMass;
    //    Real RHO_0 = paramsD.rho0;
    //    Real dT = paramsD.dT;
    //    Real3 gravity = paramsD.gravity;

    int TYPE_OF_NARKER = sortedRhoPreMu[i_idx].w;

    if (TYPE_OF_NARKER <= -2) {
        B_i[i_idx] = 0;
        uint csrStartIdx = numContacts[i_idx];
        // This needs to be check to see if it messes up the condition number of the matrix
        csrValA[csrStartIdx] = 1.0;
        csrColIndA[csrStartIdx] = i_idx;
        GlobalcsrColIndA[csrStartIdx] = i_idx + numAllMarkers * i_idx;
    } else if (TYPE_OF_NARKER == -1) {
        Calc_fluid_aij_Bi(i_idx, csrValA, csrColIndA, GlobalcsrColIndA, numContacts, B_i, d_ii, a_ii, rho_np, summGradW,
                          sortedPosRad, sortedRhoPreMu, cellStart, cellEnd, numAllMarkers, true);

    } else if (TYPE_OF_NARKER > -1)
        Calc_BC_aij_Bi(i_idx, csrValA, csrColIndA, GlobalcsrColIndA, numContacts, a_ii, B_i, sortedPosRad, sortedVelMas,

                       sortedRhoPreMu, V_new, p_old,

                       velMassRigid_fsiBodies_D, accRigid_fsiBodies_D, rigidIdentifierD,

                       pos_fsi_fea_D, vel_fsi_fea_D, acc_fsi_fea_D, FlexIdentifierD, numFlex1D, CableElementsNodes,
                       ShellelementsNodes,

                       updatePortion, gridMarkerIndexD, cellStart, cellEnd, numAllMarkers, true);
}

//--------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------
__global__ void Calc_Pressure_AXB_USING_CSR(Real* csrValA,
                                            Real* a_ii,
                                            uint* csrColIndA,
                                            uint* numContacts,
                                            Real4* sortedRhoPreMu,
                                            Real* sumWij_inv,
                                            Real3* sortedVelMas,
                                            Real3* V_new,
                                            Real* p_old,
                                            Real* B_i,  // Read
                                            const int numAllMarkers,
                                            volatile bool* isErrorD) {
    uint i_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_idx >= numAllMarkers) {
        return;
    }
    if (sortedRhoPreMu[i_idx].w <= -2) {
        return;
    }

    //    Real RHO_0 = paramsD.rho0;
    //    bool ClampPressure = paramsD.ClampPressure;
    //    Real Max_Pressure = paramsD.Max_Pressure;
    uint startIdx = numContacts[i_idx] + 1;  // numContacts[i_idx] is the diagonal itself
    uint endIdx = numContacts[i_idx + 1];

    Real aij_pj = 0;
    //  Real error = aij_pj + sortedRhoPreMu[i_idx].y * csrValA[startIdx - 1] - B_i[i_idx];

    for (int myIdx = startIdx; myIdx < endIdx; myIdx++) {
        if (csrColIndA[myIdx] != i_idx)
            aij_pj += csrValA[myIdx] * p_old[csrColIndA[myIdx]];
    }
    Real RHS = B_i[i_idx];
    sortedRhoPreMu[i_idx].y = (RHS - aij_pj) / csrValA[startIdx - 1];

    //  sortedRhoPreMu[i_idx].y = (ClampPressure && sortedRhoPreMu[i_idx].y < 0.0) ? 0.0 : sortedRhoPreMu[i_idx].y;

    if (!isfinite(aij_pj)) {
        printf("a_ij *p_j became Nan in Calc_Pressure_AXB_USING_CSR ");
    }
}

////--------------------------------------------------------------------------------------------------------------------------------
__global__ void Calc_Pressure(Real* a_ii,     // Read
                              Real3* d_ii,    // Read
                              Real3* dij_pj,  // Read
                              Real* rho_np,   // Read
                              Real* rho_p,    // Write
                              Real3* F_p,
                              Real4* sortedPosRad,
                              Real3* sortedVelMas,
                              Real4* sortedRhoPreMu,

                              Real4* velMassRigid_fsiBodies_D,
                              Real3* accRigid_fsiBodies_D,
                              uint* rigidIdentifierD,

                              Real3* pos_fsi_fea_D,
                              Real3* vel_fsi_fea_D,
                              Real3* acc_fsi_fea_D,
                              uint* FlexIdentifierD,
                              const int numFlex1D,
                              uint2* CableElementsNodes,
                              uint4* ShellelementsNodes,

                              int4 updatePortion,
                              uint* gridMarkerIndexD,

                              Real* p_old,
                              Real3* V_new,
                              uint* cellStart,
                              uint* cellEnd,
                              const int numAllMarkers,
                              volatile bool* isErrorD) {
    uint i_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_idx >= numAllMarkers) {
        return;
    }
    if (sortedRhoPreMu[i_idx].w <= -2) {
        return;
    }

    Real h_i = sortedPosRad[i_idx].w;
    Real m_i = h_i * h_i * h_i * paramsD.rho0;

    Real RHO_0 = paramsD.rho0;
    Real dT = paramsD.dT;
    Real3 gravity = paramsD.gravity;
    bool ClampPressure = paramsD.ClampPressure;

    if (sortedRhoPreMu[i_idx].x < EPSILON) {
        printf("(Calc_Pressure)My density is %f in Calc_Pressure\n", sortedRhoPreMu[i_idx].x);
    }
    int myType = sortedRhoPreMu[i_idx].w;
    Real Rho_i = sortedRhoPreMu[i_idx].x;
    Real p_i = p_old[i_idx];
    Real3 pos_i = mR3(sortedPosRad[i_idx]);
    Real p_new = 0;
    Real my_rho_p = 0;
    Real3 F_i_p = F_p[i_idx];

    if (myType == -1) {
        if (Rho_i < 0.95 * RHO_0) {
            p_new = 0;
        } else {
            Real3 my_dij_pj = dij_pj[i_idx];
            Real sum_dij_pj = 0;  // This is the first summation  term in the expression for the pressure.
            Real sum_djj_pj = 0;  // This is the second summation term in the expression for the pressure.
            Real sum_djk_pk = 0;  // This is the last summation term in the expression for the pressure.
            int3 gridPosI = calcGridPos(pos_i);
            for (int z = -1; z <= 1; z++) {
                for (int y = -1; y <= 1; y++) {
                    for (int x = -1; x <= 1; x++) {
                        int3 neighbourPosI = gridPosI + mI3(x, y, z);
                        uint gridHashI = calcGridHash(neighbourPosI);
                        // get start of bucket for this cell
                        uint startIndexI = cellStart[gridHashI];
                        if (startIndexI != 0xffffffff) {
                            uint endIndexI = cellEnd[gridHashI];
                            for (uint j = startIndexI; j < endIndexI; j++) {
                                Real3 pos_j = mR3(sortedPosRad[j]);
                                Real3 dist3ij = Distance(pos_i, pos_j);
                                Real dij = length(dist3ij);
                                if (dij > RESOLUTION_LENGTH_MULT * paramsD.HSML || i_idx == j ||
                                    sortedRhoPreMu[j].w <= -2)
                                    continue;
                                //                Real Rho_j = sortedRhoPreMu[j].x;
                                Real p_j_old = p_old[j];
                                Real h_j = sortedPosRad[j].w;
                                Real m_j = h_j * h_j * h_j * paramsD.rho0;

                                Real3 djj = d_ii[j];
                                Real3 F_j_p = F_p[j];

                                Real h_ij = 0.5 * (h_j + h_i);
                                Real3 grad_i_wij = GradWh(dist3ij, h_ij);
                                Real3 d_ji = m_i * (-(dT * dT) / (Rho_i * Rho_i)) * (-grad_i_wij);
                                Real3 djk_pk = dij_pj[j] - d_ji * p_i;
                                sum_dij_pj += m_j * dot(my_dij_pj, grad_i_wij);
                                sum_djj_pj += m_j * dot(djj, grad_i_wij) * p_j_old;
                                sum_djk_pk += m_j * dot(djk_pk, grad_i_wij);
                                my_rho_p += (dT * dT) * m_j * dot((F_i_p / m_i - F_j_p / m_j), grad_i_wij);
                            }
                        }
                    }
                }
            }

            Real RHS = fminf(0.0, RHO_0 - rho_np[i_idx]);
            //      Real RHS = RHO_0 - rho_np[i_idx];

            Real aij_pj = +sum_dij_pj - sum_djj_pj - sum_djk_pk;
            p_new = (RHS - aij_pj) / a_ii[i_idx];
            //      sortedRhoPreMu[i_idx].x = aij_pj + p_new * a_ii[i_idx] + RHO_0 - RHS;
        }
    } else {  // Do Adami BC

        Real3 myAcc = mR3(0);
        Real3 V_prescribed = mR3(0);
        BCE_Vel_Acc(i_idx, myAcc, V_prescribed, sortedPosRad, updatePortion, gridMarkerIndexD, velMassRigid_fsiBodies_D,
                    accRigid_fsiBodies_D, rigidIdentifierD, pos_fsi_fea_D, vel_fsi_fea_D, acc_fsi_fea_D,
                    FlexIdentifierD, numFlex1D, CableElementsNodes, ShellelementsNodes);

        Real3 numeratorv = mR3(0);
        Real denumenator = 0;
        Real numeratorp = 0;
        Real3 Vel_i;

        // get address in grid
        int3 gridPos = calcGridPos(pos_i);
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
                            Real3 pos_j = mR3(sortedPosRad[j]);
                            Real3 dist3 = Distance(pos_i, pos_j);
                            Real d = length(dist3);
                            if (d > RESOLUTION_LENGTH_MULT * paramsD.HSML || sortedRhoPreMu[j].w != -1)
                                continue;
                            // OLD VELOCITY IS SHOULD BE OBDATED NOT THE NEW ONE!!!!!
                            Real3 Vel_j = sortedVelMas[j];
                            Real p_j = p_old[j];
                            Real3 F_j_p = F_p[j];
                            Real h_j = sortedPosRad[j].w;
                            Real m_j = h_j * h_j * h_j * paramsD.rho0;
                            Real h_ij = 0.5 * (h_j + h_i);
                            Real Wd = W3h(d, h_ij);
                            numeratorv += Vel_j * Wd;
                            numeratorp += p_j * Wd + dot(gravity - myAcc, dist3) * sortedRhoPreMu[j].x * Wd;
                            denumenator += Wd;
                            Real3 TobeUsed = (F_i_p / m_i - F_j_p / m_j);
                            my_rho_p += (dT * dT) * m_j * dot(TobeUsed, GradWh(dist3, h_ij));

                            if (isnan(numeratorp))
                                printf("Something is wrong here..., %f\n", numeratorp);
                        }
                    }
                }
            }
        }
        if (abs(denumenator) < EPSILON) {
            p_new = 0;
            Vel_i = 2 * V_prescribed;
        } else {
            Vel_i = 2 * V_prescribed - numeratorv / denumenator;
            p_new = numeratorp / denumenator;
        }

        V_new[i_idx] = Vel_i;
    }
    if (ClampPressure && p_new < 0.0)
        p_new = 0.0;
    rho_p[i_idx] = my_rho_p;
    sortedRhoPreMu[i_idx].y = p_new;
}

//--------------------------------------------------------------------------------------------------------------------------------
__global__ void Update_AND_Calc_Res(Real3* sortedVelMas,
                                    Real4* sortedRhoPreMu,
                                    Real* p_old,
                                    Real3* V_new,
                                    Real* rho_p,
                                    Real* rho_np,
                                    Real* Residuals,
                                    const int numAllMarkers,
                                    const int Iteration,
                                    Real params_relaxation,
                                    bool IsSPARSE,
                                    volatile bool* isErrorD) {
    uint i_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (i_idx >= numAllMarkers) {
        return;
    }
    if (sortedRhoPreMu[i_idx].w <= -2) {
        return;
    }

    //  p_i = (1 - relax) * p_old_i + relax * p_i;
    sortedRhoPreMu[i_idx].y = (1 - params_relaxation) * p_old[i_idx] + params_relaxation * sortedRhoPreMu[i_idx].y;
    //  Real AbsRes = abs(sortedRhoPreMu[i_idx].y - p_old[i_idx]);

    //  Real Updated_rho = rho_np[i_idx] + rho_p[i_idx];
    //  Real rho_res = abs(1000 - sortedRhoPreMu[i_idx].x);  // Hard-coded for now
    Real p_res = 0;
    //  p_res = abs(sortedRhoPreMu[i_idx].y - p_old[i_idx]) / (abs(p_old[i_idx]) + 0.00001);
    p_res = abs(sortedRhoPreMu[i_idx].y - p_old[i_idx]);

    Residuals[i_idx] = p_res;
}

//--------------------------------------------------------------------------------------------------------------------------------
__global__ void CalcForces(Real3* new_vel,  // Write
                           Real4* derivVelRhoD,
                           Real4* sortedPosRad,  // Read
                           Real3* sortedVelMas,  // Read
                           Real4* sortedRhoPreMu,
                           Real* sumWij_inv,
                           Real* G_tensor,
                           Real* L_tensor,
                           Real3* r_shift,
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

    if (sortedRhoPreMu[i_idx].w > -1) {
        return;
    }

    Real mu_0 = paramsD.mu0;
    Real h_i = sortedPosRad[i_idx].w;
    Real m_i = h_i * h_i * h_i * paramsD.rho0;

    Real dT = paramsD.dT;
    Real3 gravity = paramsD.gravity;
    Real epsilon = paramsD.epsMinMarkersDis;
    Real3 posi = mR3(sortedPosRad[i_idx]);
    Real3 Veli = sortedVelMas[i_idx];
    Real p_i = sortedRhoPreMu[i_idx].y;
    Real rho_i = sortedRhoPreMu[i_idx].x;
    Real3 F_i_mu = mR3(0);
    Real3 F_i_surface_tension = mR3(0);
    Real3 F_i_p = mR3(0);
    if ((sortedRhoPreMu[i_idx].x > 3 * paramsD.rho0 || sortedRhoPreMu[i_idx].x < 0) && sortedRhoPreMu[i_idx].w < 0)
        printf("too large/small density marker %d, type=%f\n", i_idx, sortedRhoPreMu[i_idx].w);

    Real r0 = 0;
    int Ni = 0;
    Real mi_bar = 0;
    Real3 inner_sum = mR3(0);

    int3 gridPos = calcGridPos(posi);
    Real3 myGradvx = mR3(0);
    Real3 myGradvy = mR3(0);
    Real3 myGradvz = mR3(0);
    Real Gi[9] = {0.0};
    Real Li[6] = {0.0};
    Real3 LaplacainVi = mR3(0.0);
    for (int i = 0; i < 9; i++)
        Gi[i] = G_tensor[i_idx * 9 + i];
    for (int i = 0; i < 6; i++)
        Li[i] = L_tensor[i_idx * 6 + i];

    grad_vector(i_idx, sortedPosRad, sortedRhoPreMu, sumWij_inv, G_tensor, sortedVelMas, myGradvx, myGradvy, myGradvz,
                cellStart, cellEnd);

    // get address in grid
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
                    Real3 rij = Distance(posi, posj);
                    Real d = length(rij);
                    if (d > RESOLUTION_LENGTH_MULT * h_i || sortedRhoPreMu[j].w <= -2 || i_idx == j)
                        continue;
                    Real3 eij = rij / d;
                    Real h_j = sortedPosRad[j].w;
                    Real m_j = h_j * h_j * h_j * paramsD.rho0;

                    mi_bar += m_j;
                    Ni++;
                    r0 += d;
                    inner_sum += m_j * rij / (d * d * d);

                    Real h_ij = 0.5 * (h_j + h_i);
                    Real Wd = m_j * W3h(d, h_ij);
                    Real3 grad_ij = GradWh(rij, h_ij);

                    Real3 Velj = sortedVelMas[j];
                    Real p_j = sortedRhoPreMu[j].y;
                    Real rho_j = sortedRhoPreMu[j].x;

                    Real3 V_ij = (Veli - Velj);
                    // Only Consider (fluid-fluid + fluid-solid) or Solid-Fluid Interaction
                    if (sortedRhoPreMu[i_idx].w < 0 || (sortedRhoPreMu[i_idx].w > 0 && sortedRhoPreMu[j].w < 0))
                        F_i_p += -m_j * ((p_i / (rho_i * rho_i)) + (p_j / (rho_j * rho_j))) * grad_ij;

                    Real Rho_bar = (rho_j + rho_i) * 0.5;
                    //                    Real nu = mu_0 * paramsD.HSML * 320 / Rho_bar;
                    //          Real3 muNumerator = nu * fminf(0.0, dot(rij, V_ij)) * grad_ij;
                    Real3 muNumerator = 2 * mu_0 * dot(rij, grad_ij) * V_ij;

                    Real muDenominator = (Rho_bar * Rho_bar) * (d * d + paramsD.HSML * paramsD.HSML * epsilon);
                    // Only Consider (fluid-fluid + fluid-solid) or Solid-Fluid Interaction
                    if (sortedRhoPreMu[i_idx].w < 0 || (sortedRhoPreMu[i_idx].w > 0 && sortedRhoPreMu[j].w < 0))
                        F_i_mu += m_j * muNumerator / muDenominator;
                    if (!isfinite(length(F_i_mu))) {
                        printf("F_i_np in CalcForces returns Nan or Inf");
                    }

                    F_i_surface_tension += m_j * Wd * rij;

                    Real Vj = sumWij_inv[j];
                    Real commonterm =
                        2 * Vj *
                        (Li[0] * eij.x * grad_ij.x + Li[1] * eij.x * grad_ij.y + Li[2] * eij.x * grad_ij.z +
                         Li[1] * eij.y * grad_ij.x + Li[3] * eij.y * grad_ij.y + Li[4] * eij.y * grad_ij.z +
                         Li[2] * eij.z * grad_ij.x + Li[4] * eij.z * grad_ij.y + Li[5] * eij.z * grad_ij.z);

                    LaplacainVi.x += commonterm * (V_ij.x / d - dot(eij, myGradvx));
                    LaplacainVi.y += commonterm * (V_ij.y / d - dot(eij, myGradvy));
                    LaplacainVi.z += commonterm * (V_ij.y / d - dot(eij, myGradvz));

                    if (!isfinite(length(LaplacainVi))) {
                        printf("LaplacainVi in CalcForces returns Nan or Inf");
                    }
                }
            }
        }
    }

    r0 /= Ni;
    r_shift[i_idx] = 0.5 * r0 * r0 * paramsD.v_Max * dT / mi_bar * inner_sum;

    F_i_surface_tension = -F_i_surface_tension * paramsD.kappa / m_i;
    // Forces are per unit mass at this point.
    //    derivVelRhoD[i_idx] = mR4((F_i_p + F_i_mu + F_i_surface_tension) * m_i);
    derivVelRhoD[i_idx] = mR4((F_i_p + mu_0 * LaplacainVi + F_i_surface_tension) * m_i);

    // Add the gravity forces only to the fluid markers
    if ((int)sortedRhoPreMu[i_idx].w == -1)
        derivVelRhoD[i_idx] = derivVelRhoD[i_idx] + mR4(gravity) * m_i;

    new_vel[i_idx] = Veli + dT * mR3(derivVelRhoD[i_idx]) / m_i + r_shift[i_idx] / dT;
    sortedVelMas[i_idx] = new_vel[i_idx];
}

//--------------------------------------------------------------------------------------------------------------------------------
__global__ void FinalizePressure(Real4* sortedPosRad,  // Read
                                 Real4* sortedRhoPreMu,
                                 Real* p_old,
                                 Real3* F_p,  // Write
                                 uint* cellStart,
                                 uint* cellEnd,
                                 uint numAllMarkers,
                                 Real p_shift,
                                 volatile bool* isErrorD) {
    uint i_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_idx >= numAllMarkers) {
        return;
    }
    if (sortedRhoPreMu[i_idx].w <= -2) {
        return;
    }
    //  Real m_0 = paramsD.markerMass;
    //  Real3 posi = sortedPosRad[i_idx];
    //  Real Rho_i = sortedRhoPreMu[i_idx].x;
    //  Real3 my_F_p = mR3(0);
    //  Real p_i = sortedRhoPreMu[i_idx].y;
    //  Real sum_pW = 0;
    //  Real sum_W = 0;

    // if (p_old[i_idx] > 0)
    // sortedRhoPreMu[i_idx].y = p_old[i_idx];
    // else
    // sortedRhoPreMu[i_idx].y = 0;

    //  if (p_shift < 0)
    sortedRhoPreMu[i_idx].y = p_old[i_idx] + ((paramsD.ClampPressure) ? paramsD.BASEPRES : 0.0);  //- p_shift;

    //  if (p_old[i_idx] < 0)
    //    sortedRhoPreMu[i_idx].y = (p_old[i_idx] > 0) ? p_old[i_idx] : 0.0;

    //  if (sortedRhoPreMu[i_idx].y > paramsD.Max_Pressure)
    //    sortedRhoPreMu[i_idx].y = paramsD.Max_Pressure;

    //  // get address in grid
    //  int3 gridPos = calcGridPos(posi);
    //  for (int z = -1; z <= 1; z++) {
    //    for (int y = -1; y <= 1; y++) {
    //      for (int x = -1; x <= 1; x++) {
    //        int3 neighbourPos = gridPos + mI3(x, y, z);
    //        uint gridHash = calcGridHash(neighbourPos);
    //        // get start of bucket for this cell
    //        uint startIndex = cellStart[gridHash];
    //        uint endIndex = cellEnd[gridHash];
    //
    //        for (uint j = startIndex; j < endIndex; j++) {
    //          Real3 posJ = sortedPosRad[j];
    //          Real3 dist3 = Distance(posi, posJ);
    //          Real d = length(dist3);
    //          if (d > RESOLUTION_LENGTH_MULT * paramsD.HSML || i_idx == j)
    //            continue;
    //          Real3 grad_i_wij = GradW(dist3);
    //          Real p_j = p_old[j];
    //          Real Rho_j = sortedRhoPreMu[j].x;
    //          my_F_p += -m_0 * m_0 * ((p_old[i_idx] / (Rho_i * Rho_i)) + (p_old[j] / (Rho_j * Rho_j))) *
    //          grad_i_wij;
    //        }
    //      }
    //    }
    //  }
    //  F_p[i_idx] = my_F_p;

    //  if (abs(sum_W) < EPSILON) {
    //    sortedRhoPreMu[i_idx].y = 0;
    //  } else {
    //    sortedRhoPreMu[i_idx].y = sum_pW / sum_W;
    //  }
}

//--------------------------------------------------------------------------------------------------------------------------------

// ChFsiForceI2SPH::ChFsiForceI2SPH() {}

// ChFsiForceIISPH::ChFsiForceIISPH() {}
// ChFsiForceIISPH::~ChFsiForceIISPH() {}
// ChFsiForceIISPH::ChFsiForceIISPH(
//    ChBce* otherBceWorker,                   ///< Pointer to the ChBce object that handles BCE markers
//    SphMarkerDataD* otherSortedSphMarkersD,  ///< Information of markers in the sorted array on device
//    ProximityDataD*
//        otherMarkersProximityD,           ///< Pointer to the object that holds the proximity of the markers on device
//    FsiGeneralData* otherFsiGeneralData,  ///< Pointer to the sph general data
//    SimParams* otherParamsH,              ///< Pointer to the simulation parameters on host
//    NumberOfObjects* otherNumObjects      ///< Pointer to number of objects, fluid and boundary markers, etc.
//    )
//    : ChFsiForceParallel(otherBceWorker,
//                         otherSortedSphMarkersD,
//                         otherMarkersProximityD,
//                         otherFsiGeneralData,
//                         otherParamsH,
//                         otherNumObjects) {
//    printf("ChFsiForceIISPH::ChFsiForceIISPH constructor\n");
//}

void ChFsiForceIISPH::calcPressureIISPH(thrust::device_vector<Real4> velMassRigid_fsiBodies_D,
                                        thrust::device_vector<Real3> accRigid_fsiBodies_D,
                                        thrust::device_vector<Real3> pos_fsi_fea_D,
                                        thrust::device_vector<Real3> vel_fsi_fea_D,
                                        thrust::device_vector<Real3> acc_fsi_fea_D,
                                        thrust::device_vector<Real>& sumWij_inv,
                                        thrust::device_vector<Real>& G_i,
                                        thrust::device_vector<Real>& L_i,
                                        thrust::device_vector<Real>& Color) {
    //    Real RES = paramsH->PPE_res;

    PPE_SolutionType mySolutionType = paramsH->PPE_Solution_type;

    double total_step_timeClock = clock();
    bool *isErrorH, *isErrorD;
    isErrorH = (bool*)malloc(sizeof(bool));
    cudaMalloc((void**)&isErrorD, sizeof(bool));
    *isErrorH = false;
    cudaMemcpy(isErrorD, isErrorH, sizeof(bool), cudaMemcpyHostToDevice);
    //------------------------------------------------------------------------
    // thread per particle
    uint numThreads, numBlocks;
    int numAllMarkers = numObjectsH->numAllMarkers;
    computeGridSize(numAllMarkers, 256, numBlocks, numThreads);

    printf("numBlocks: %d, numThreads: %d, numAllMarker:%d \n", numBlocks, numThreads, numAllMarkers);

    *isErrorH = false;
    cudaMemcpy(isErrorD, isErrorH, sizeof(bool), cudaMemcpyHostToDevice);

    //    calcRho_kernel<<<numBlocks, numThreads>>>(
    //        mR4CAST(sortedSphMarkersD->posRadD), mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(sumWij_inv),
    //        U1CAST(markersProximityD->cellStartD), U1CAST(markersProximityD->cellEndD), numAllMarkers, isErrorD);
    //
    //    cudaThreadSynchronize();
    //    cudaCheckError();
    //    cudaMemcpy(isErrorH, isErrorD, sizeof(bool), cudaMemcpyDeviceToHost);
    //    if (*isErrorH == true) {
    //        throw std::runtime_error("Error! program crashed after calcRho_kernel!\n");
    //    }
    //
    //    thrust::device_vector<Real> dxi_over_Vi(numAllMarkers);
    //    thrust::fill(dxi_over_Vi.begin(), dxi_over_Vi.end(), 0);
    //
    //    calcNormalizedRho_kernel<<<numBlocks, numThreads>>>(
    //        mR4CAST(sortedSphMarkersD->posRadD), mR3CAST(sortedSphMarkersD->velMasD),
    //        mR4CAST(sortedSphMarkersD->rhoPresMuD),  // input: sorted velocities
    //        R1CAST(sumWij_inv), R1CAST(G_i), R1CAST(dxi_over_Vi), R1CAST(Color),
    //        U1CAST(markersProximityD->cellStartD), U1CAST(markersProximityD->cellEndD), numAllMarkers, isErrorD);
    //    cudaThreadSynchronize();
    //    cudaCheckError();
    //    cudaMemcpy(isErrorH, isErrorD, sizeof(bool), cudaMemcpyDeviceToHost);
    //    if (*isErrorH == true) {
    //        throw std::runtime_error("Error! program crashed after calcNormalizedRho_kernel!\n");
    //    }
    /*
        if (paramsH->Adaptive_time_stepping) {
          int position = thrust::min_element(dxi_over_Vi.begin(), dxi_over_Vi.end()) - dxi_over_Vi.begin();
          Real min_dxi_over_Vi = dxi_over_Vi[position];

          Real dt = paramsH->Co_number * min_dxi_over_Vi;
          printf("Min dxi_over_Vi of fluid particles to boundary is: %f. Time step based on Co=%f is %f\n",
       min_dxi_over_Vi,
                 paramsH->Co_number, dt);
          // I am doing this to prevent very low time steps (when it requires it to save data at the current
       step)
          // Because if will have to do two time steps either way
          if (dt / paramsH->dT_Max > 0.7 && dt / paramsH->dT_Max < 1)
            paramsH->dT = paramsH->dT_Max * 0.5;
          else
            paramsH->dT = fminf((float)dt, paramsH->dT_Max);

          std::cout << "time step is set to min(dt_Co,dT_Max)= " << paramsH->dT << "\n";
        }
        */
    thrust::device_vector<Real3> d_ii(numAllMarkers);
    thrust::device_vector<Real3> V_np(numAllMarkers);
    thrust::fill(d_ii.begin(), d_ii.end(), mR3(0.0));
    thrust::fill(V_np.begin(), V_np.end(), mR3(0.0));
    *isErrorH = false;
    cudaMemcpy(isErrorD, isErrorH, sizeof(bool), cudaMemcpyHostToDevice);
    V_i_np__AND__d_ii_kernel<<<numBlocks, numThreads>>>(
        mR4CAST(sortedSphMarkersD->posRadD), mR3CAST(sortedSphMarkersD->velMasD),
        mR4CAST(sortedSphMarkersD->rhoPresMuD), mR3CAST(d_ii), mR3CAST(V_np), R1CAST(sumWij_inv), R1CAST(G_i),
        R1CAST(L_i), U1CAST(markersProximityD->cellStartD), U1CAST(markersProximityD->cellEndD), numAllMarkers,
        isErrorD);

    cudaThreadSynchronize();
    cudaCheckError();
    cudaMemcpy(isErrorH, isErrorD, sizeof(bool), cudaMemcpyDeviceToHost);
    if (*isErrorH == true) {
        throw std::runtime_error("Error! program crashed after F_i_np__AND__d_ii_kernel!\n");
    }

    thrust::device_vector<Real> a_ii(numAllMarkers);
    thrust::device_vector<Real> rho_np(numAllMarkers);
    thrust::device_vector<Real> p_old(numAllMarkers);
    thrust::fill(a_ii.begin(), a_ii.end(), 0.0);
    thrust::fill(rho_np.begin(), rho_np.end(), 0.0);
    thrust::fill(p_old.begin(), p_old.end(), 0.0);
    thrust::device_vector<Real3> summGradW(numAllMarkers);

    *isErrorH = false;
    cudaMemcpy(isErrorD, isErrorH, sizeof(bool), cudaMemcpyHostToDevice);
    Rho_np_AND_a_ii_AND_sum_m_GradW<<<numBlocks, numThreads>>>(
        mR4CAST(sortedSphMarkersD->posRadD), mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(rho_np), R1CAST(a_ii),
        R1CAST(p_old), mR3CAST(V_np), mR3CAST(d_ii), mR3CAST(summGradW), U1CAST(markersProximityD->cellStartD),
        U1CAST(markersProximityD->cellEndD), numAllMarkers, isErrorD);

    cudaThreadSynchronize();
    cudaCheckError();
    cudaMemcpy(isErrorH, isErrorD, sizeof(bool), cudaMemcpyDeviceToHost);
    if (*isErrorH == true) {
        throw std::runtime_error("Error! program crashed after F_i_np__AND__d_ii_kernel!\n");
    }

    thrust::device_vector<Real3> V_new(numAllMarkers);
    thrust::fill(V_new.begin(), V_new.end(), mR3(0.0));
    thrust::device_vector<Real> a_ij;
    thrust::device_vector<Real> B_i(numAllMarkers);
    thrust::device_vector<uint> csrColIndA;
    thrust::device_vector<uint> numContacts(numAllMarkers);
    thrust::device_vector<unsigned long int> GlobalcsrColIndA;
    thrust::device_vector<Real> csrValA;

    double durationFormAXB;

    int numFlexbodies = +numObjectsH->numFlexBodies1D + numObjectsH->numFlexBodies2D;

    int haveGhost = (numObjectsH->numGhostMarkers > 0) ? 1 : 0;
    int haveHelper = (numObjectsH->numHelperMarkers > 0) ? 1 : 0;

    int4 updatePortion =
        mI4(fsiGeneralData->referenceArray[haveGhost + haveHelper + 0].y,  // end of fluid
            fsiGeneralData->referenceArray[haveGhost + haveHelper + 1].y,  // end of boundary
            fsiGeneralData->referenceArray[haveGhost + haveHelper + 1 + numObjectsH->numRigidBodies].y,
            fsiGeneralData->referenceArray[haveGhost + haveHelper + 1 + numObjectsH->numRigidBodies + numFlexbodies].y);

    uint NNZ;
    if (mySolutionType == FORM_SPARSE_MATRIX) {
        thrust::fill(a_ij.begin(), a_ij.end(), 0.0);
        thrust::fill(B_i.begin(), B_i.end(), 0.0);
        thrust::fill(summGradW.begin(), summGradW.end(), mR3(0.0));
        thrust::fill(numContacts.begin(), numContacts.end(), 0.0);
        //------------------------------------------------------------------------
        //------------- MatrixJacobi
        //------------------------------------------------------------------------

        bool SPARSE_FLAG = true;
        double FormAXBClock = clock();
        thrust::device_vector<Real> Residuals(numAllMarkers);
        thrust::fill(Residuals.begin(), Residuals.end(), 1.0);
        thrust::device_vector<Real> rho_p(numAllMarkers);
        thrust::fill(rho_p.begin(), rho_p.end(), 0.0);

        *isErrorH = false;
        cudaMemcpy(isErrorD, isErrorH, sizeof(bool), cudaMemcpyHostToDevice);
        CalcNumber_Contacts<<<numBlocks, numThreads>>>(
            U1CAST(numContacts), mR4CAST(sortedSphMarkersD->posRadD), mR4CAST(sortedSphMarkersD->rhoPresMuD),
            U1CAST(markersProximityD->cellStartD), U1CAST(markersProximityD->cellEndD), numAllMarkers, isErrorD);

        cudaThreadSynchronize();
        cudaCheckError();
        cudaMemcpy(isErrorH, isErrorD, sizeof(bool), cudaMemcpyDeviceToHost);
        if (*isErrorH == true) {
            throw std::runtime_error("Error! program crashed after CalcNumber_Contacts!\n");
        }
        uint MAX_CONTACT = thrust::reduce(numContacts.begin(), numContacts.end(), 0, thrust::maximum<Real>());
        std::cout << "Max contact between SPH particles: " << MAX_CONTACT << std::endl;

        uint LastVal = numContacts[numAllMarkers - 1];
        thrust::exclusive_scan(numContacts.begin(), numContacts.end(), numContacts.begin());
        numContacts.push_back(LastVal + numContacts[numAllMarkers - 1]);
        NNZ = numContacts[numAllMarkers];

        csrValA.resize(NNZ);
        csrColIndA.resize(NNZ);
        GlobalcsrColIndA.resize(NNZ);

        thrust::fill(csrValA.begin(), csrValA.end(), 0.0);
        thrust::fill(GlobalcsrColIndA.begin(), GlobalcsrColIndA.end(), 0.0);
        thrust::fill(csrColIndA.begin(), csrColIndA.end(), 0.0);

        cudaMemcpy(isErrorD, isErrorH, sizeof(bool), cudaMemcpyHostToDevice);

        std::cout << "updatePortion of  BC: " << updatePortion.x << " " << updatePortion.y << " " << updatePortion.z
                  << " " << updatePortion.w << "\n ";

        FormAXB<<<numBlocks, numThreads>>>(
            R1CAST(csrValA), U1CAST(csrColIndA), LU1CAST(GlobalcsrColIndA), U1CAST(numContacts), R1CAST(a_ij),
            R1CAST(B_i), mR3CAST(d_ii), R1CAST(a_ii), mR3CAST(summGradW), mR4CAST(sortedSphMarkersD->posRadD),
            mR3CAST(sortedSphMarkersD->velMasD), mR4CAST(sortedSphMarkersD->rhoPresMuD), mR3CAST(V_new), R1CAST(p_old),
            R1CAST(rho_np),

            mR4CAST(velMassRigid_fsiBodies_D), mR3CAST(accRigid_fsiBodies_D), U1CAST(fsiGeneralData->rigidIdentifierD),

            mR3CAST(pos_fsi_fea_D), mR3CAST(vel_fsi_fea_D), mR3CAST(acc_fsi_fea_D),
            U1CAST(fsiGeneralData->FlexIdentifierD), numObjectsH->numFlexBodies1D,
            U2CAST(fsiGeneralData->CableElementsNodes), U4CAST(fsiGeneralData->ShellElementsNodes),

            updatePortion, U1CAST(markersProximityD->gridMarkerIndexD), U1CAST(markersProximityD->cellStartD),
            U1CAST(markersProximityD->cellEndD), numAllMarkers, SPARSE_FLAG, isErrorD);

        cudaThreadSynchronize();
        cudaCheckError();
        cudaMemcpy(isErrorH, isErrorD, sizeof(bool), cudaMemcpyDeviceToHost);
        if (*isErrorH == true) {
            throw std::runtime_error("Error! program crashed after F_i_np__AND__d_ii_kernel!\n");
        }

        durationFormAXB = (clock() - FormAXBClock) / (double)CLOCKS_PER_SEC;
    }
    //------------------------------------------------------------------------
    //------------- Iterative loop
    //------------------------------------------------------------------------
    int Iteration = 0;
    Real MaxRes = 100;
    thrust::device_vector<Real> Residuals(numAllMarkers);
    thrust::fill(Residuals.begin(), Residuals.end(), 1.0);
    thrust::device_vector<Real3> dij_pj(numAllMarkers);
    thrust::fill(dij_pj.begin(), dij_pj.end(), mR3(0.0));
    thrust::device_vector<Real3> F_p(numAllMarkers);
    thrust::fill(F_p.begin(), F_p.end(), mR3(0.0));
    thrust::device_vector<Real> rho_p(numAllMarkers);
    thrust::fill(rho_p.begin(), rho_p.end(), 0.0);

    double LinearSystemClock = clock();

    ChFsiLinearSolver myLS(paramsH->LinearSolver, paramsH->LinearSolver_Rel_Tol, paramsH->LinearSolver_Abs_Tol,
                           paramsH->LinearSolver_Max_Iter, paramsH->Verbose_monitoring);
    if (paramsH->USE_LinearSolver) {
        if (paramsH->PPE_Solution_type != FORM_SPARSE_MATRIX) {
            printf(
                "You should paramsH->PPE_Solution_type == FORM_SPARSE_MATRIX in order to use the "
                "chrono_fsi linear "
                "solvers\n");
            exit(0);
        }

        myLS.Solve(numAllMarkers, NNZ, R1CAST(csrValA), U1CAST(numContacts), U1CAST(csrColIndA), (double*)R1CAST(p_old),
                   R1CAST(B_i));
        cudaCheckError();
    }

    while ((MaxRes > paramsH->LinearSolver_Rel_Tol || Iteration < 3) && paramsH->USE_Iterative_solver &&
           Iteration < paramsH->LinearSolver_Max_Iter) {
        *isErrorH = false;
        cudaMemcpy(isErrorD, isErrorH, sizeof(bool), cudaMemcpyHostToDevice);
        Initialize_Variables<<<numBlocks, numThreads>>>(mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(p_old),
                                                        mR3CAST(sortedSphMarkersD->velMasD), mR3CAST(V_new),
                                                        numAllMarkers, isErrorD);
        cudaThreadSynchronize();
        cudaCheckError();
        cudaMemcpy(isErrorH, isErrorD, sizeof(bool), cudaMemcpyDeviceToHost);
        if (*isErrorH == true) {
            throw std::runtime_error("Error! program crashed after Initialize_Variables!\n");
        }

        if (mySolutionType == MATRIX_FREE) {
            *isErrorH = false;
            cudaMemcpy(isErrorD, isErrorH, sizeof(bool), cudaMemcpyHostToDevice);
            Calc_dij_pj<<<numBlocks, numThreads>>>(
                mR3CAST(dij_pj), mR3CAST(F_p), mR3CAST(d_ii), mR4CAST(sortedSphMarkersD->posRadD),
                mR3CAST(sortedSphMarkersD->velMasD), mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(p_old),
                U1CAST(markersProximityD->cellStartD), U1CAST(markersProximityD->cellEndD), numAllMarkers, isErrorD);
            cudaThreadSynchronize();
            cudaCheckError();
            cudaMemcpy(isErrorH, isErrorD, sizeof(bool), cudaMemcpyDeviceToHost);
            if (*isErrorH == true) {
                throw std::runtime_error("Error! program crashed after Calc_dij_pj!\n");
            }

            *isErrorH = false;
            cudaMemcpy(isErrorD, isErrorH, sizeof(bool), cudaMemcpyHostToDevice);
            Calc_Pressure<<<numBlocks, numThreads>>>(
                R1CAST(a_ii), mR3CAST(d_ii), mR3CAST(dij_pj), R1CAST(rho_np), R1CAST(rho_p), mR3CAST(F_p),
                mR4CAST(sortedSphMarkersD->posRadD), mR3CAST(sortedSphMarkersD->velMasD),
                mR4CAST(sortedSphMarkersD->rhoPresMuD), mR4CAST(velMassRigid_fsiBodies_D),
                mR3CAST(accRigid_fsiBodies_D), U1CAST(fsiGeneralData->rigidIdentifierD), mR3CAST(pos_fsi_fea_D),
                mR3CAST(vel_fsi_fea_D), mR3CAST(acc_fsi_fea_D), U1CAST(fsiGeneralData->FlexIdentifierD),
                numObjectsH->numFlexBodies1D, U2CAST(fsiGeneralData->CableElementsNodes),
                U4CAST(fsiGeneralData->ShellElementsNodes), updatePortion, U1CAST(markersProximityD->gridMarkerIndexD),
                R1CAST(p_old), mR3CAST(V_new), U1CAST(markersProximityD->cellStartD),
                U1CAST(markersProximityD->cellEndD), numAllMarkers, isErrorD);

            cudaThreadSynchronize();
            cudaCheckError();
            cudaMemcpy(isErrorH, isErrorD, sizeof(bool), cudaMemcpyDeviceToHost);
            if (*isErrorH == true) {
                throw std::runtime_error("Error! program crashed after Calc_Pressure!\n");
            }
        }

        if (mySolutionType == FORM_SPARSE_MATRIX) {
            *isErrorH = false;
            cudaMemcpy(isErrorD, isErrorH, sizeof(bool), cudaMemcpyHostToDevice);
            Calc_Pressure_AXB_USING_CSR<<<numBlocks, numThreads>>>(
                R1CAST(csrValA), R1CAST(a_ii), U1CAST(csrColIndA), U1CAST(numContacts),
                mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(sumWij_inv), mR3CAST(sortedSphMarkersD->velMasD),
                mR3CAST(V_new), R1CAST(p_old), R1CAST(B_i), numAllMarkers, isErrorD);
            cudaThreadSynchronize();
            cudaCheckError();
            cudaMemcpy(isErrorH, isErrorD, sizeof(bool), cudaMemcpyDeviceToHost);
            if (*isErrorH == true) {
                throw std::runtime_error("Error! program crashed after Iterative_pressure_update!\n");
            }
        }
        *isErrorH = false;
        cudaMemcpy(isErrorD, isErrorH, sizeof(bool), cudaMemcpyHostToDevice);

        Update_AND_Calc_Res<<<numBlocks, numThreads>>>(
            mR3CAST(sortedSphMarkersD->velMasD), mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(p_old), mR3CAST(V_new),
            R1CAST(rho_p), R1CAST(rho_np), R1CAST(Residuals), numAllMarkers, Iteration, paramsH->PPE_relaxation, false,
            isErrorD);
        cudaThreadSynchronize();
        cudaCheckError();
        cudaMemcpy(isErrorH, isErrorD, sizeof(bool), cudaMemcpyDeviceToHost);
        if (*isErrorH == true) {
            throw std::runtime_error("Error! program crashed after Iterative_pressure_update!\n");
        }

        Iteration++;

        thrust::device_vector<Real>::iterator iter = thrust::max_element(Residuals.begin(), Residuals.end());
        unsigned int position = iter - Residuals.begin();
        MaxRes = *iter;

        //    Real PMAX = thrust::reduce(p_old.begin(), p_old.end(), 0.0, thrust::maximum<Real>());
        //    MaxRes = thrust::reduce(Residuals.begin(), Residuals.end(), 0.0, thrust::plus<Real>()) /
        //    numObjectsH->numAllMarkers;
        //    MaxRes = thrust::reduce(Residuals.begin(), Residuals.end(), 0.0, thrust::maximum<Real>());
        //      Real R_np = thrust::reduce(rho_np.begin(), rho_np.end(), 0.0, thrust::plus<Real>()) /
        //      rho_np.size();
        //      Real R_p = thrust::reduce(rho_p.begin(), rho_p.end(), 0.0, thrust::plus<Real>()) /
        //      rho_p.size();
        //
        if (paramsH->Verbose_monitoring)
            printf("Iter= %d, Res= %f\n", Iteration, MaxRes);
    }

    //  thrust::device_vector<Real>::iterator iter = thrust::min_element(p_old.begin(), p_old.end());
    //  unsigned int position = iter - p_old.begin();
    //  Real shift_p = *iter;
    Real shift_p = 0;

    // This must be run if cusp solver is used
    if (paramsH->ClampPressure || paramsH->USE_LinearSolver) {
        //    printf("Shifting pressure values by %f\n", -shift_p);
        *isErrorH = false;
        cudaMemcpy(isErrorD, isErrorH, sizeof(bool), cudaMemcpyHostToDevice);
        FinalizePressure<<<numBlocks, numThreads>>>(
            mR4CAST(sortedSphMarkersD->posRadD), mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(p_old), mR3CAST(F_p),
            U1CAST(markersProximityD->cellStartD), U1CAST(markersProximityD->cellEndD), numAllMarkers, shift_p,
            isErrorD);
        cudaThreadSynchronize();
        cudaCheckError();
        cudaMemcpy(isErrorH, isErrorD, sizeof(bool), cudaMemcpyDeviceToHost);
        if (*isErrorH == true) {
            throw std::runtime_error("Error! program crashed after FinalizePressure!\n");
        }
    }
    double durationLinearSystem = (clock() - LinearSystemClock) / (double)CLOCKS_PER_SEC;
    double durationtotal_step_time = (clock() - total_step_timeClock) / (double)CLOCKS_PER_SEC;

    printf("---------------IISPH CLOCK-------------------\n");
    printf(" Total: %f \n FormAXB: %f\n Linear System: %f \n", durationtotal_step_time, durationFormAXB,
           durationLinearSystem);
    if (!paramsH->USE_LinearSolver)
        printf(" Iter (Jacobi+SOR)# = %d, to Res= %f \n", Iteration, MaxRes);
    if (paramsH->USE_LinearSolver)
        if (myLS.GetSolverStatus()) {
            std::cout << " Solver converged to " << myLS.GetResidual() << " tolerance";
            std::cout << " after " << myLS.GetNumIterations() << " iterations" << std::endl;
        } else {
            std::cout << "Failed to converge after " << myLS.GetIterationLimit() << " iterations";
            std::cout << " (" << myLS.GetResidual() << " final residual)" << std::endl;
        }

    //------------------------------------------------------------------------
    //------------------------------------------------------------------------
    cudaFree(isErrorD);
    free(isErrorH);
    // dxi_over_Vi.clear();
    p_old.clear();
    d_ii.clear();
    V_np.clear();
    V_new.clear();
    a_ii.clear();
    rho_np.clear();
    rho_p.clear();
    F_p.clear();
    a_ii.clear();
    a_ij.clear();
    B_i.clear();
    dij_pj.clear();
    summGradW.clear();
    csrColIndA.clear();
    GlobalcsrColIndA.clear();
    csrValA.clear();
    V_new.clear();
    Residuals.clear();
    numContacts.clear();
}

void ChFsiForceIISPH::ForceImplicitSPH(SphMarkerDataD* otherSphMarkersD,
                                       FsiBodiesDataD* otherFsiBodiesD,
                                       FsiMeshDataD* otherFsiMeshD) {
    std::cout << "dT in ForceSPH before calcPressure: " << paramsH->dT << "\n";

    sphMarkersD = otherSphMarkersD;
    int numAllMarkers = numObjectsH->numAllMarkers;
    int numHelperMarkers = numObjectsH->numHelperMarkers;
    fsiCollisionSystem->ArrangeData(sphMarkersD);
    printf("ForceIISPH paramsH=%f,%f,%f\n", paramsH->cellSize.x, paramsH->cellSize.y, paramsH->cellSize.z);

    bool *isErrorH, *isErrorD, *isErrorD2;

    isErrorH = (bool*)malloc(sizeof(bool));
    cudaMalloc((void**)&isErrorD, sizeof(bool));
    cudaMalloc((void**)&isErrorD2, sizeof(bool));
    *isErrorH = false;
    cudaMemcpy(isErrorD, isErrorH, sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(isErrorD2, isErrorH, sizeof(bool), cudaMemcpyHostToDevice);

    uint numThreads, numBlocks;
    computeGridSize(numAllMarkers, 256, numBlocks, numThreads);

    thrust::device_vector<Real> Color(numAllMarkers);
    thrust::fill(Color.begin(), Color.end(), 1.0e10);
    thrust::device_vector<Real> _sumWij_inv(numAllMarkers);
    thrust::fill(_sumWij_inv.begin(), _sumWij_inv.end(), 1e-3);
    thrust::device_vector<Real> G_i(numAllMarkers * 9);
    thrust::fill(G_i.begin(), G_i.end(), 0);
    thrust::device_vector<Real> A_i(numAllMarkers * 27);
    thrust::fill(A_i.begin(), A_i.end(), 0);
    thrust::device_vector<Real> L_i(numAllMarkers * 6);
    thrust::fill(L_i.begin(), L_i.end(), 0);
    *isErrorH = false;
    cudaMemcpy(isErrorD, isErrorH, sizeof(bool), cudaMemcpyHostToDevice);

    //    calcRho_kernel<<<numBlocks, numThreads>>>(
    //        mR4CAST(sortedSphMarkersD->posRadD), mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(_sumWij_inv),
    //        U1CAST(markersProximityD->cellStartD), U1CAST(markersProximityD->cellEndD), numAllMarkers, isErrorD);
    //
    //    cudaThreadSynchronize();
    //    cudaCheckError();
    //    cudaMemcpy(isErrorH, isErrorD, sizeof(bool), cudaMemcpyDeviceToHost);
    //    if (*isErrorH == true) {
    //        throw std::runtime_error("Error! program crashed after calcRho_kernel!\n");
    //    }
    //
    //    thrust::device_vector<Real> dxi_over_Vi(numAllMarkers);
    //    thrust::fill(dxi_over_Vi.begin(), dxi_over_Vi.end(), 0);
    //
    //    calcNormalizedRho_kernel<<<numBlocks, numThreads>>>(
    //        mR4CAST(sortedSphMarkersD->posRadD), mR3CAST(sortedSphMarkersD->velMasD),
    //        mR4CAST(sortedSphMarkersD->rhoPresMuD),  // input: sorted velocities
    //        R1CAST(_sumWij_inv), R1CAST(G_i), R1CAST(dxi_over_Vi), R1CAST(Color),
    //        U1CAST(markersProximityD->cellStartD), U1CAST(markersProximityD->cellEndD), numAllMarkers, isErrorD);
    //    cudaThreadSynchronize();
    //    cudaCheckError();
    //    cudaMemcpy(isErrorH, isErrorD, sizeof(bool), cudaMemcpyDeviceToHost);
    //    if (*isErrorH == true) {
    //        throw std::runtime_error("Error! program crashed after calcNormalizedRho_kernel!\n");
    //    }
    //
    //    thrust::device_vector<uint> SplitMe(numAllMarkers);
    //    thrust::device_vector<uint> MergeMe(numAllMarkers);
    //    thrust::device_vector<uint> myType(numAllMarkers);
    //    thrust::device_vector<Real3> helpers_normal(numHelperMarkers);
    //
    //    thrust::fill(SplitMe.begin(), SplitMe.end(), 0);
    //    thrust::fill(MergeMe.begin(), MergeMe.end(), 0);
    //    thrust::fill(myType.begin(), myType.end(), 0);
    //    thrust::fill(helpers_normal.begin(), helpers_normal.end(), mR3(0));
    //    cudaThreadSynchronize();
    //
    //    Calc_HelperMarkers_normals<<<numBlocks, numThreads>>>(
    //        mR4CAST(sortedSphMarkersD->posRadD), mR4CAST(sortedSphMarkersD->rhoPresMuD), mR3CAST(helpers_normal),
    //        I1CAST(myType), U1CAST(markersProximityD->cellStartD), U1CAST(markersProximityD->cellEndD),
    //        U1CAST(markersProximityD->gridMarkerIndexD), numAllMarkers, isErrorD2);
    //    cudaThreadSynchronize();
    //    cudaCheckError();
    //    cudaMemcpy(isErrorH, isErrorD2, sizeof(bool), cudaMemcpyDeviceToHost);
    //    if (*isErrorH == true) {
    //        throw std::runtime_error("Error! program crashed in Calc_HelperMarkers_normals!\n");
    //    }
    //    int haveGhost = (numObjectsH->numGhostMarkers > 0) ? 1 : 0;
    //    int haveHelper = (numObjectsH->numHelperMarkers > 0) ? 1 : 0;
    //    int limit = numObjectsH->numFluidMarkers + numObjectsH->numGhostMarkers + numObjectsH->numHelperMarkers;
    //
    //    Calc_Splits_and_Merges<<<numBlocks, numThreads>>>(
    //        mR4CAST(sortedSphMarkersD->posRadD), mR4CAST(sortedSphMarkersD->rhoPresMuD),
    //        mR3CAST(sortedSphMarkersD->velMasD), mR3CAST(helpers_normal), R1CAST(_sumWij_inv), R1CAST(G_i),
    //        U1CAST(SplitMe), U1CAST(MergeMe), I1CAST(myType), U1CAST(markersProximityD->cellStartD),
    //        U1CAST(markersProximityD->cellEndD), U1CAST(markersProximityD->gridMarkerIndexD), paramsH->HSML / 2.0,
    //        paramsH->HSML, numAllMarkers, limit, isErrorD2);
    //    cudaThreadSynchronize();
    //    cudaCheckError();
    //    cudaMemcpy(isErrorH, isErrorD2, sizeof(bool), cudaMemcpyDeviceToHost);
    //    if (*isErrorH == true) {
    //        throw std::runtime_error("Error! program crashed in Calc_Split_and_Merges!\n");
    //    }
    //
    //    Split<<<numBlocks, numThreads>>>(mR4CAST(sortedSphMarkersD->posRadD),
    //    mR4CAST(sortedSphMarkersD->rhoPresMuD),
    //                                     mR3CAST(sortedSphMarkersD->velMasD), mR3CAST(helpers_normal),
    //                                     R1CAST(_sumWij_inv), R1CAST(G_i), U1CAST(SplitMe), U1CAST(MergeMe),
    //                                     I1CAST(myType), U1CAST(markersProximityD->cellStartD),
    //                                     U1CAST(markersProximityD->cellEndD),
    //                                     U1CAST(markersProximityD->gridMarkerIndexD), paramsH->HSML / 2.0,
    //                                     paramsH->HSML, numAllMarkers, limit, isErrorD2);
    //    cudaThreadSynchronize();
    //    cudaCheckError();
    //    cudaMemcpy(isErrorH, isErrorD2, sizeof(bool), cudaMemcpyDeviceToHost);
    //    if (*isErrorH == true) {
    //        throw std::runtime_error("Error! program crashed in Calc_Split_and_Merges!\n");
    //    }
    //
    thrust::device_vector<Real> mcontac(numAllMarkers);
    thrust::fill(mcontac.begin(), mcontac.end(), 0);
    calcRho_kernel<<<numBlocks, numThreads>>>(
        mR4CAST(sortedSphMarkersD->posRadD), mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(_sumWij_inv),
        U1CAST(markersProximityD->cellStartD), U1CAST(markersProximityD->cellEndD), U1CAST(mcontac), numAllMarkers,
        isErrorD);
    cudaThreadSynchronize();
    cudaCheckError();
    cudaMemcpy(isErrorH, isErrorD, sizeof(bool), cudaMemcpyDeviceToHost);
    if (*isErrorH == true) {
        throw std::runtime_error("Error! program crashed after calcRho_kernel!\n");
    }

    calcNormalizedRho_kernel<<<numBlocks, numThreads>>>(
        mR4CAST(sortedSphMarkersD->posRadD), mR3CAST(sortedSphMarkersD->velMasD),
        mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(_sumWij_inv), R1CAST(G_i), R1CAST(_sumWij_inv), R1CAST(Color),
        U1CAST(markersProximityD->cellStartD), U1CAST(markersProximityD->cellEndD), numAllMarkers, isErrorD);
    cudaThreadSynchronize();
    cudaCheckError();
    cudaMemcpy(isErrorH, isErrorD, sizeof(bool), cudaMemcpyDeviceToHost);
    if (*isErrorH == true) {
        throw std::runtime_error("Error! program crashed after calcNormalizedRho_kernel!\n");
    }
    //

    calc_A_tensor<<<numBlocks, numThreads>>>(R1CAST(A_i), R1CAST(G_i), mR4CAST(sortedSphMarkersD->posRadD),
                                             mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(_sumWij_inv),
                                             U1CAST(markersProximityD->cellStartD), U1CAST(markersProximityD->cellEndD),
                                             numAllMarkers, isErrorD);

    cudaThreadSynchronize();
    cudaCheckError();
    cudaMemcpy(isErrorH, isErrorD, sizeof(bool), cudaMemcpyDeviceToHost);
    if (*isErrorH == true) {
        throw std::runtime_error("Error! program crashed after calcRho_kernel!\n");
    }

    calc_L_tensor<<<numBlocks, numThreads>>>(R1CAST(A_i), R1CAST(L_i), R1CAST(G_i), mR4CAST(sortedSphMarkersD->posRadD),
                                             mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(_sumWij_inv),
                                             U1CAST(markersProximityD->cellStartD), U1CAST(markersProximityD->cellEndD),
                                             numAllMarkers, isErrorD);
    cudaThreadSynchronize();
    cudaCheckError();
    cudaMemcpy(isErrorH, isErrorD, sizeof(bool), cudaMemcpyDeviceToHost);
    if (*isErrorH == true) {
        throw std::runtime_error("Error! program crashed after calcRho_kernel!\n");
    }

    calcPressureIISPH(otherFsiBodiesD->velMassRigid_fsiBodies_D, otherFsiBodiesD->accRigid_fsiBodies_D,
                      otherFsiMeshD->pos_fsi_fea_D, otherFsiMeshD->vel_fsi_fea_D, otherFsiMeshD->acc_fsi_fea_D,
                      _sumWij_inv, G_i, L_i, Color);

    //------------------------------------------------------------------------
    // thread per particle
    //  std::cout << "dT in ForceSPH after calcPressure: " << paramsH->dT << "\n";
    double CalcForcesClock = clock();

    thrust::fill(vel_IISPH_Sorted_D.begin(), vel_IISPH_Sorted_D.end(), mR3(0.0));
    thrust::fill(derivVelRhoD_Sorted_D.begin(), derivVelRhoD_Sorted_D.end(), mR4(0.0));
    thrust::fill(vel_XSPH_Sorted_D.begin(), vel_XSPH_Sorted_D.end(), mR3(0.0));
    thrust::device_vector<Real3> dr_shift(numAllMarkers);
    thrust::fill(dr_shift.begin(), dr_shift.end(), mR3(0.0));

    CalcForces<<<numBlocks, numThreads>>>(mR3CAST(vel_XSPH_Sorted_D), mR4CAST(derivVelRhoD_Sorted_D),
                                          mR4CAST(sortedSphMarkersD->posRadD), mR3CAST(sortedSphMarkersD->velMasD),
                                          mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(_sumWij_inv), R1CAST(G_i),
                                          R1CAST(L_i), mR3CAST(dr_shift), U1CAST(markersProximityD->cellStartD),
                                          U1CAST(markersProximityD->cellEndD), numAllMarkers, isErrorD);
    cudaThreadSynchronize();
    cudaCheckError();

    cudaMemcpy(isErrorH, isErrorD, sizeof(bool), cudaMemcpyDeviceToHost);
    if (*isErrorH == true) {
        throw std::runtime_error("Error! program crashed in CalcForces!\n");
    }
    double calcforce = (clock() - CalcForcesClock) / (double)CLOCKS_PER_SEC;
    printf(" Force Computation: %f \n", calcforce);
    double UpdateClock = clock();

    thrust::device_vector<uint> SplitMe(numAllMarkers);
    thrust::device_vector<uint> MergeMe(numAllMarkers);
    thrust::device_vector<uint> myType(numAllMarkers);
    thrust::device_vector<Real3> helpers_normal(numHelperMarkers);
    //
    thrust::fill(SplitMe.begin(), SplitMe.end(), 0);
    thrust::fill(MergeMe.begin(), MergeMe.end(), 0);
    thrust::fill(myType.begin(), myType.end(), 0);
    thrust::fill(helpers_normal.begin(), helpers_normal.end(), mR3(0));
    cudaThreadSynchronize();

    Calc_HelperMarkers_normals<<<numBlocks, numThreads>>>(
        mR4CAST(sortedSphMarkersD->posRadD), mR4CAST(sortedSphMarkersD->rhoPresMuD), mR3CAST(helpers_normal),
        I1CAST(myType), U1CAST(markersProximityD->cellStartD), U1CAST(markersProximityD->cellEndD),
        U1CAST(markersProximityD->gridMarkerIndexD), numAllMarkers, isErrorD2);
    cudaThreadSynchronize();
    cudaCheckError();
    cudaMemcpy(isErrorH, isErrorD2, sizeof(bool), cudaMemcpyDeviceToHost);
    if (*isErrorH == true) {
        throw std::runtime_error("Error! program crashed in Calc_HelperMarkers_normals!\n");
    }
    double CalcSplitMerge = clock();

    int haveGhost = (numObjectsH->numGhostMarkers > 0) ? 1 : 0;
    int haveHelper = (numObjectsH->numHelperMarkers > 0) ? 1 : 0;
    int limit = numObjectsH->numFluidMarkers + numObjectsH->numGhostMarkers + numObjectsH->numHelperMarkers;

    Calc_Splits_and_Merges<<<numBlocks, numThreads>>>(
        mR4CAST(sortedSphMarkersD->posRadD), mR4CAST(sortedSphMarkersD->rhoPresMuD),
        mR3CAST(sortedSphMarkersD->velMasD), mR3CAST(helpers_normal), R1CAST(_sumWij_inv), R1CAST(G_i), U1CAST(SplitMe),
        U1CAST(MergeMe), I1CAST(myType), U1CAST(markersProximityD->cellStartD), U1CAST(markersProximityD->cellEndD),
        U1CAST(markersProximityD->gridMarkerIndexD), paramsH->HSML / 2.0, paramsH->HSML, numAllMarkers, limit,
        isErrorD2);
    cudaThreadSynchronize();
    cudaCheckError();
    cudaMemcpy(isErrorH, isErrorD2, sizeof(bool), cudaMemcpyDeviceToHost);
    if (*isErrorH == true) {
        throw std::runtime_error("Error! program crashed in Calc_Split_and_Merges!\n");
    }

    Split<<<numBlocks, numThreads>>>(mR4CAST(sortedSphMarkersD->posRadD), mR4CAST(sortedSphMarkersD->rhoPresMuD),
                                     mR3CAST(sortedSphMarkersD->velMasD), mR3CAST(helpers_normal), R1CAST(_sumWij_inv),
                                     R1CAST(G_i), R1CAST(L_i), U1CAST(SplitMe), U1CAST(MergeMe), I1CAST(myType),
                                     U1CAST(markersProximityD->cellStartD), U1CAST(markersProximityD->cellEndD),
                                     U1CAST(markersProximityD->gridMarkerIndexD), paramsH->HSML / 2.0, paramsH->HSML,
                                     numAllMarkers, limit, isErrorD2);
    cudaThreadSynchronize();
    cudaCheckError();
    cudaMemcpy(isErrorH, isErrorD2, sizeof(bool), cudaMemcpyDeviceToHost);
    if (*isErrorH == true) {
        throw std::runtime_error("Error! program crashed in Calc_Split_and_Merges!\n");
    }

    UpdateDensity<<<numBlocks, numThreads>>>(
        mR3CAST(vel_IISPH_Sorted_D), mR3CAST(vel_XSPH_Sorted_D), mR4CAST(sortedSphMarkersD->posRadD),
        mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(_sumWij_inv), U1CAST(markersProximityD->cellStartD),
        U1CAST(markersProximityD->cellEndD), numAllMarkers, isErrorD);

    cudaThreadSynchronize();
    cudaCheckError();

    cudaMemcpy(isErrorH, isErrorD, sizeof(bool), cudaMemcpyDeviceToHost);
    if (*isErrorH == true) {
        throw std::runtime_error("Error! program crashed in CalcForces!\n");
    }
    CopySortedToOriginal_NonInvasive_R3(fsiGeneralData->vel_XSPH_D, vel_XSPH_Sorted_D,
                                        markersProximityD->gridMarkerIndexD);
    CopySortedToOriginal_NonInvasive_R3(fsiGeneralData->vel_IISPH_D, vel_IISPH_Sorted_D,
                                        markersProximityD->gridMarkerIndexD);
    CopySortedToOriginal_NonInvasive_R4(sphMarkersD->posRadD, sortedSphMarkersD->posRadD,
                                        markersProximityD->gridMarkerIndexD);
    CopySortedToOriginal_NonInvasive_R3(sphMarkersD->velMasD, sortedSphMarkersD->velMasD,
                                        markersProximityD->gridMarkerIndexD);
    CopySortedToOriginal_NonInvasive_R4(sphMarkersD->rhoPresMuD, sortedSphMarkersD->rhoPresMuD,
                                        markersProximityD->gridMarkerIndexD);
    CopySortedToOriginal_NonInvasive_R4(fsiGeneralData->derivVelRhoD, derivVelRhoD_Sorted_D,
                                        markersProximityD->gridMarkerIndexD);
    printf(" Update information: %f \n", (clock() - UpdateClock) / (double)CLOCKS_PER_SEC);
    printf("----------------------------------------------\n");

    _sumWij_inv.clear();
    helpers_normal.clear();
    Color.clear();
    //
}

}  // namespace fsi
}  // namespace chrono
