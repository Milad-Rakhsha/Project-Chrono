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
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

#include "chrono_fsi/ChFsiForceI2SPH.cuh"

//==========================================================================================================================================
namespace chrono {
namespace fsi {
// extern __constant__ SimParams paramsD;
// extern __constant__ NumberOfObjects numObjectsD;

struct compare_Real3_mag {
    __host__ __device__ bool operator()(Real3 lhs, Real3 rhs) { return length(lhs) < length(rhs); }
};

// double precision atomic add function
__device__ inline double datomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;

    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}
//--------------------------------------------------------------------------------------------------------------------------------
__global__ void V_star_Predictor(Real4* sortedPosRad,  // input: sorted positions
                                 Real3* sortedVelMas,
                                 Real4* sortedRhoPreMu,
                                 Real* A_Matrix,
                                 Real3* b,
                                 Real3* v_old,

                                 const Real* A_L,
                                 const Real3* A_G,
                                 const Real* A_f,

                                 const Real* sumWij_inv,
                                 Real3* normals,

                                 const uint* csrColInd,
                                 const uint* numContacts,

                                 Real4* velMassRigid_fsiBodies_D,
                                 Real3* accRigid_fsiBodies_D,
                                 uint* rigidIdentifierD,

                                 Real3* pos_fsi_fea_D,
                                 Real3* vel_fsi_fea_D,
                                 Real3* acc_fsi_fea_D,
                                 uint* FlexIdentifierD,
                                 int numFlex1D,
                                 uint2* CableElementsNodes,
                                 uint4* ShellelementsNodes,

                                 int4 updatePortion,
                                 uint* gridMarkerIndexD,

                                 int numAllMarkers,
                                 volatile bool* isErrorD) {
    uint i_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_idx >= numAllMarkers) {
        return;
    }

    uint csrStartIdx = numContacts[i_idx];
    uint csrEndIdx = numContacts[i_idx + 1];

    bool Fluid_Marker = sortedRhoPreMu[i_idx].w == -1.0;
    bool Boundary_Marker = sortedRhoPreMu[i_idx].w > -1.0;

    if (sortedRhoPreMu[i_idx].w <= -2) {
        A_Matrix[csrStartIdx] = 1;
        b[i_idx] = mR3(0.0);
        return;
    }

    Real rho0 = paramsD.rho0;
    Real nu0 = paramsD.mu0 / rho0;
    Real dt = paramsD.dT;
    Real rhoi = sortedRhoPreMu[i_idx].x;
    Real3 grad_rho_i = mR3(0.0);

    for (int count = csrStartIdx; count < csrEndIdx; count++) {
        int j = csrColInd[count];
        grad_rho_i += A_G[count] * sortedRhoPreMu[j].x;
    }

    if (Fluid_Marker) {
        Real3 rhs = mR3(0.0);
        for (int count = csrStartIdx; count < csrEndIdx; count++) {
            int j = csrColInd[count];

            A_Matrix[count] =
                paramsD.mu0 * -(0.5 / rhoi * A_L[count] - 0.5 / (rhoi * rhoi) * dot(grad_rho_i, A_G[count]));

            rhs += paramsD.mu0 * (0.5 / rhoi * A_L[count] - 0.5 / (rhoi * rhoi) * dot(grad_rho_i, A_G[count])) *
                   sortedVelMas[j];
        }
        A_Matrix[csrStartIdx] += 1 / dt;
        b[i_idx] = rhs + sortedVelMas[i_idx] / dt  //forward euler term from lhs
                   + paramsD.gravity;              // body force

    } else if (Boundary_Marker) {
        Real h_i = sortedPosRad[i_idx].w;
        Real3 posRadA = mR3(sortedPosRad[i_idx]);
        Real den = 0.0;
        for (uint count = csrStartIdx + 1; count < csrEndIdx; count++) {
            uint j = csrColInd[count];
            if (sortedRhoPreMu[j].w != -1)
                continue;
            Real3 posRadB = mR3(sortedPosRad[j]);
            Real3 rij = Distance(posRadA, posRadB);
            Real h_j = sortedPosRad[j].w;
            Real h_ij = 0.5 * (h_j + h_i);
            Real W3 = W3h(length(rij), h_ij);
            A_Matrix[count] = W3;
            // A_Matrix[count] = A_f[count];
            den = den + W3;
        }

        Real3 myAcc = mR3(0);
        Real3 V_prescribed = mR3(0);
        BCE_Vel_Acc(i_idx, myAcc, V_prescribed, sortedPosRad, updatePortion, gridMarkerIndexD, velMassRigid_fsiBodies_D,
                    accRigid_fsiBodies_D, rigidIdentifierD, pos_fsi_fea_D, vel_fsi_fea_D, acc_fsi_fea_D,
                    FlexIdentifierD, numFlex1D, CableElementsNodes, ShellelementsNodes);

        if (abs(den) < EPSILON) {
            A_Matrix[csrStartIdx] = 1.0;
            b[i_idx] = V_prescribed;
        } else {
            A_Matrix[csrStartIdx] = den;
            b[i_idx] = 2 * V_prescribed * den;
        }
    }

    v_old[i_idx] = sortedVelMas[i_idx];
}
//--------------------------------------------------------------------------------------------------------------------------------
__global__ void Pressure_Equation(Real4* sortedPosRad,  // input: sorted positions
                                  Real3* sortedVelMas,
                                  Real4* sortedRhoPreMu,
                                  Real* A_Matrix,
                                  Real* Bi,
                                  Real3* Vstar,
                                  Real* q_old,

                                  const Real* A_f,
                                  const Real* A_L,
                                  const Real3* A_G,
                                  const Real* sumWij_inv,
                                  Real3* Normals,
                                  const uint* csrColInd,
                                  const uint* numContacts,

                                  Real4* velMassRigid_fsiBodies_D,
                                  Real3* accRigid_fsiBodies_D,
                                  uint* rigidIdentifierD,

                                  Real3* pos_fsi_fea_D,
                                  Real3* vel_fsi_fea_D,
                                  Real3* acc_fsi_fea_D,
                                  uint* FlexIdentifierD,
                                  int numFlex1D,
                                  uint2* CableElementsNodes,
                                  uint4* ShellelementsNodes,

                                  int4 updatePortion,
                                  uint* gridMarkerIndexD,
                                  int numAllMarkers,
                                  int FixedMarker,
                                  volatile bool* isErrorD) {
    uint i_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_idx >= numAllMarkers) {
        return;
    }

    uint csrStartIdx = numContacts[i_idx];
    uint csrEndIdx = numContacts[i_idx + 1];

    bool Fluid_Marker = sortedRhoPreMu[i_idx].w == -1.0;
    bool Boundary_Marker = sortedRhoPreMu[i_idx].w > -1.0;

    if (sortedRhoPreMu[i_idx].w <= -2) {
        A_Matrix[csrStartIdx] = 1.0;
        Bi[i_idx] = 0.0;
        return;
    }

    Real3 gravity = paramsD.gravity;
    Real dt = paramsD.dT;
    Real3 grad_rho_i = mR3(0.0);
    Real div_vi_star = 0;

    for (int count = csrStartIdx; count < csrEndIdx; count++) {
        int j = csrColInd[count];
        div_vi_star += dot(A_G[count], Vstar[j]);
        grad_rho_i += A_G[count] * sortedRhoPreMu[j].x;
    }

    Real rhoi = sortedRhoPreMu[i_idx].x;
    //    Real rhoi_star = rhoi - rhoi * div_vi_star * dt;
    Real rhoi_star = rhoi + dot(grad_rho_i, Vstar[i_idx] * dt);

    //======================== Interior ===========================
    if (Fluid_Marker) {
        //        if (sortedRhoPreMu[i_idx].x < 0.80 * paramsD.rho0 && sortedRhoPreMu[i_idx].x > 0.80 * paramsD.rho0) {
        //            //        if (i_idx == FixedMarker) {
        //            A_Matrix[csrStartIdx] = 1.0;
        //            //
        //            for (int count = csrStartIdx; count < csrEndIdx; count++) {
        //                A_Matrix[count] = -A_f[count];
        //            }
        //        } else if (sortedRhoPreMu[i_idx].x < 0.99 * paramsD.rho0) {
        //            A_Matrix[csrStartIdx] = 1.0;
        //            Bi[i_idx] = 0.0;
        if (i_idx == -1) {
            A_Matrix[csrStartIdx] = 1.0;
            Bi[i_idx] = 0.0;
        } else {
            for (int count = csrStartIdx; count < csrEndIdx; count++) {
                A_Matrix[count] = 1.0 / rhoi * A_L[count] - 1.0 / (rhoi * rhoi) * dot(grad_rho_i, A_G[count]);
            }
            Bi[i_idx] = 1.0 * div_vi_star / dt + 0.00 * (paramsD.rho0 - rhoi_star) / paramsD.rho0 / (dt * dt);
        }

        //======================= Boundary ===========================
    } else if (Boundary_Marker && paramsD.bceType != ADAMI) {
        Real3 my_normal = Normals[i_idx];
        bool haveFluid = false;
        Real temp = 0;

        for (int count = csrStartIdx; count < csrEndIdx; count++) {
            if (sortedRhoPreMu[csrColInd[count]].w == -1.0)
                haveFluid = true;
        }

        if (!haveFluid) {
            for (int count = csrStartIdx + 1; count < csrEndIdx; count++) {
                A_Matrix[count] = 0.0;
            }
            A_Matrix[csrStartIdx] = 1.0;
            Bi[i_idx] = 1000.0;
        } else {
            for (int count = csrStartIdx; count < csrEndIdx; count++) {
                uint j = csrColInd[count];
                if (sortedRhoPreMu[j].w == -1.0)
                    A_Matrix[count] = dot(A_G[count], my_normal);
            }
            if (A_Matrix[csrStartIdx] < 1e-3) {
                A_Matrix[csrStartIdx] = 1;  // temp / (csrEndIdx - csrStartIdx) / 5;
                Bi[i_idx] = 1e3;            // * dot(paramsD.gravity, my_normal);
            }
        }

        //======================= Boundary Adami===========================
    } else if (Boundary_Marker && paramsD.bceType == ADAMI) {
        Real h_i = sortedPosRad[i_idx].w;
        Real Vi = sumWij_inv[i_idx];
        Real3 posRadA = mR3(sortedPosRad[i_idx]);
        Real3 myAcc = mR3(0);
        Real3 V_prescribed = mR3(0);
        BCE_Vel_Acc(i_idx, myAcc, V_prescribed, sortedPosRad, updatePortion, gridMarkerIndexD, velMassRigid_fsiBodies_D,
                    accRigid_fsiBodies_D, rigidIdentifierD, pos_fsi_fea_D, vel_fsi_fea_D, acc_fsi_fea_D,
                    FlexIdentifierD, numFlex1D, CableElementsNodes, ShellelementsNodes);
        Real pRHS = 0.0;
        Real den = 0.0;

        for (int count = csrStartIdx + 1; count < csrEndIdx; count++) {
            uint j = csrColInd[count];
            if (sortedRhoPreMu[j].w != -1.0)
                continue;
            Real3 posRadB = mR3(sortedPosRad[j]);
            Real3 rij = Distance(posRadA, posRadB);
            Real h_j = sortedPosRad[j].w;
            Real h_ij = 0.5 * (h_j + h_i);
            Real W3 = W3h(length(rij), h_ij);
            A_Matrix[count] = -W3;
            den += W3;
            pRHS += dot(gravity - myAcc, rij) * sortedRhoPreMu[j].x * W3;
        }

        if (abs(den) > EPSILON) {
            A_Matrix[csrStartIdx] = den;
            Bi[i_idx] = pRHS;
            for (int count = csrStartIdx; count < csrEndIdx; count++) {
                A_Matrix[count] /= den;
            }
            Bi[i_idx] /= den;
        } else {
            A_Matrix[csrStartIdx] = 1;
            Bi[i_idx] = 0;
        }
    }

    //    q_old[i_idx] = sortedRhoPreMu[i_idx].y;
}
//--------------------------------------------------------------------------------------------------------------------------------
__global__ void Velocity_Correction_and_update(Real4* sortedPosRad,
                                               Real4* sortedRhoPreMu,
                                               Real3* sortedVelMas,
                                               Real3* Vstar,
                                               Real* q_i,  // q=p^(n+1)-p^n
                                               const Real3* A_G,
                                               const uint* csrColInd,
                                               const uint* numContacts,
                                               int numAllMarkers,
                                               const Real MaxVel,
                                               volatile bool* isErrorD) {
    uint i_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_idx >= numAllMarkers) {
        return;
    }

    uint csrStartIdx = numContacts[i_idx];
    uint csrEndIdx = numContacts[i_idx + 1];
    Real3 grad_q_i = mR3(0.0);
    Real3 grad_p_nPlus1 = mR3(0.0);
    Real3 inner_sum = mR3(0.0), shift_r = mR3(0.0);
    Real mi_bar = 0.0, r0 = 0.0;
    for (int count = csrStartIdx; count < csrEndIdx; count++) {
        uint j = csrColInd[count];
        grad_q_i += A_G[count] * q_i[j];
        grad_p_nPlus1 += A_G[count] * (sortedRhoPreMu[j].y + q_i[j]);
        Real3 rij = Distance(mR3(sortedPosRad[i_idx]), mR3(sortedPosRad[j]));
        Real d = length(rij);
        if (count == csrStartIdx)
            continue;
        Real m_j = pow(sortedPosRad[j].w * paramsD.MULT_INITSPACE, 3) * paramsD.rho0;
        mi_bar += m_j;
        r0 += d;
        inner_sum += m_j * rij / (d * d * d);
    }
    if (sortedRhoPreMu[i_idx].w == -1.0)
        r0 /= (csrEndIdx - csrStartIdx - 1);
    shift_r = 0.5 * r0 * r0 * length(MaxVel) * paramsD.dT / mi_bar * inner_sum;

    Real3 V_new = Vstar[i_idx] - paramsD.dT / sortedRhoPreMu[i_idx].x * grad_q_i;

    Real4 x_new = sortedPosRad[i_idx] + mR4(paramsD.dT / 2 * (V_new + sortedVelMas[i_idx]), 0.0);

    sortedVelMas[i_idx] = V_new;

    //    sortedRhoPreMu[i_idx].y += q_i[i_idx] + dot(grad_p_nPlus1, mR3(x_new - sortedPosRad[i_idx]));

    sortedRhoPreMu[i_idx].y = q_i[i_idx];

    if (sortedRhoPreMu[i_idx].w == -1.0) {
        sortedPosRad[i_idx] = x_new;
    }

    Real3 grad_p = mR3(0.0);
    Real3 grad_rho = mR3(0.0);
    Real3 grad_ux = mR3(0.0);
    Real3 grad_uy = mR3(0.0);
    Real3 grad_uz = mR3(0.0);

    for (int count = csrStartIdx; count < csrEndIdx; count++) {
        uint j = csrColInd[count];
        grad_p += A_G[count] * sortedRhoPreMu[i_idx].y;
        grad_rho += A_G[count] * sortedRhoPreMu[i_idx].x;
        grad_ux += A_G[count] * sortedVelMas[i_idx].x;
        grad_uy += A_G[count] * sortedVelMas[i_idx].y;
        grad_uz += A_G[count] * sortedVelMas[i_idx].z;
    }

    if (true && sortedRhoPreMu[i_idx].w == -1.0) {
        sortedPosRad[i_idx] += mR4(shift_r, 0.0);
        sortedRhoPreMu[i_idx].y += dot(shift_r, grad_p);
        sortedRhoPreMu[i_idx].x += dot(shift_r, grad_rho);
        sortedVelMas[i_idx].x += dot(shift_r, grad_ux);
        sortedVelMas[i_idx].y += dot(shift_r, grad_uy);
        sortedVelMas[i_idx].z += dot(shift_r, grad_uz);
    }

    if (!(isfinite(sortedPosRad[i_idx].x) && isfinite(sortedPosRad[i_idx].y) && isfinite(sortedPosRad[i_idx].z))) {
        printf("Error! particle %d position is NAN: thrown from Velocity_Correction_and_update  %f,%f,%f,%f\n", i_idx,
               sortedPosRad[i_idx].x, sortedPosRad[i_idx].y, sortedPosRad[i_idx].z, sortedPosRad[i_idx].w);
    }
    if (!(isfinite(sortedRhoPreMu[i_idx].x) && isfinite(sortedRhoPreMu[i_idx].y) &&
          isfinite(sortedRhoPreMu[i_idx].z))) {
        printf("Error! particle %d rhoPreMu is NAN: thrown from Velocity_Correction_and_update ! %f,%f,%f,%f\n", i_idx,
               sortedRhoPreMu[i_idx].x, sortedRhoPreMu[i_idx].y, sortedRhoPreMu[i_idx].z, sortedRhoPreMu[i_idx].w);
    }

    if (!(isfinite(sortedVelMas[i_idx].x) && isfinite(sortedVelMas[i_idx].y) && isfinite(sortedVelMas[i_idx].z))) {
        printf("Error! particle %d velocity is NAN: thrown from Velocity_Correction_and_update !%f,%f,%f\n", i_idx,
               sortedVelMas[i_idx].x, sortedVelMas[i_idx].y, sortedVelMas[i_idx].z);
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
        if (A_Matrix[startIdx - 1] == 0.0)
            printf(" %d A_Matrix[startIdx - 1]= %f, type=%f \n", i_idx, A_Matrix[startIdx - 1],
                   sortedRhoPreMu[i_idx].w);
        V_new[i_idx] = (b3vec[i_idx] - aij_vj) / A_Matrix[startIdx - 1];
    } else {
        Real aij_pj = 0.0;
        for (int myIdx = startIdx; myIdx < endIdx; myIdx++) {
            aij_pj += A_Matrix[myIdx] * q_old[csrColInd[myIdx]];
        }
        if (A_Matrix[startIdx - 1] == 0.0)
            printf(" %d A_Matrix[startIdx - 1]= %f, type=%f \n", i_idx, A_Matrix[startIdx - 1],
                   sortedRhoPreMu[i_idx].w);

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
        if (!(isfinite(V_old[i_idx].x) && isfinite(V_old[i_idx].y) && isfinite(V_old[i_idx].z)))
            printf(" %d vel= %f,%f,%f\n", i_idx, V_old[i_idx].x, V_old[i_idx].y, V_old[i_idx].z);

    } else {
        q_new[i_idx] = (1 - omega) * q_old[i_idx] + omega * q_new[i_idx];
        res = abs(q_old[i_idx] - q_new[i_idx]);
        q_old[i_idx] = q_new[i_idx];
        if (!(isfinite(q_old[i_idx])))
            printf(" %d q= %f\n", i_idx, q_old[i_idx]);
    }
    Residuals[i_idx] = res;
}
//==========================================================================================================================================
//==========================================================================================================================================
//==========================================================================================================================================

ChFsiForceI2SPH::ChFsiForceI2SPH(
    ChBce* otherBceWorker,                   ///< Pointer to the ChBce object that handles BCE markers
    SphMarkerDataD* otherSortedSphMarkersD,  ///< Information of markers in the sorted array on device
    ProximityDataD*
        otherMarkersProximityD,           ///< Pointer to the object that holds the proximity of the markers on device
    FsiGeneralData* otherFsiGeneralData,  ///< Pointer to the sph general data
    SimParams* otherParamsH,              ///< Pointer to the simulation parameters on host
    NumberOfObjects* otherNumObjects      ///< Pointer to number of objects, fluid and boundary markers, etc.
    )
    : ChFsiForce(otherBceWorker,
                 otherSortedSphMarkersD,
                 otherMarkersProximityD,
                 otherFsiGeneralData,
                 otherParamsH,
                 otherNumObjects) {
    CopyParams_NumberOfObjects(paramsH, numObjectsH);
}

ChFsiForceI2SPH::~ChFsiForceI2SPH() {}

void ChFsiForceI2SPH::Finalize() {
    ChFsiForce::Finalize();
    cudaMemcpyToSymbolAsync(paramsD, paramsH, sizeof(SimParams));
    cudaMemcpyToSymbolAsync(numObjectsD, numObjectsH, sizeof(NumberOfObjects));
    cudaMemcpyFromSymbol(paramsH, paramsD, sizeof(SimParams));
    cudaThreadSynchronize();
    CopyParams_NumberOfObjects(paramsH, numObjectsH);
    int numAllMarkers = numObjectsH->numAllMarkers;
    _sumWij_inv.resize(numAllMarkers);
    Normals.resize(numAllMarkers);
    G_i.resize(numAllMarkers * 9);
    A_i.resize(numAllMarkers * 27);
    L_i.resize(numAllMarkers * 6);
    Contact_i.resize(numAllMarkers);
    V_star_new.resize(numAllMarkers);
    V_star_old.resize(numAllMarkers);
    q_new.resize(numAllMarkers);
    q_old.resize(numAllMarkers);
    b1Vector.resize(numAllMarkers);
    b3Vector.resize(numAllMarkers);
    Residuals.resize(numAllMarkers);
}
//==========================================================================================================================================
//==========================================================================================================================================
//==========================================================================================================================================
void ChFsiForceI2SPH::ForceImplicitSPH(SphMarkerDataD* otherSphMarkersD,
                                       FsiBodiesDataD* otherFsiBodiesD,
                                       FsiMeshDataD* otherFsiMeshD) {
    std::cout << "dT in ForceImplicitSPH: " << paramsH->dT << "\n";
    CopyParams_NumberOfObjects(paramsH, numObjectsH);
    sphMarkersD = otherSphMarkersD;
    int numAllMarkers = numObjectsH->numAllMarkers;
    int numHelperMarkers = numObjectsH->numHelperMarkers;
    fsiCollisionSystem->ArrangeData(sphMarkersD);
    printf("ForceI2SPH numAllMarkers:%d,numHelperMarkers=%d\n", numAllMarkers, numHelperMarkers);
    bool *isErrorH, *isErrorD, *isErrorD2;
    isErrorH = (bool*)malloc(sizeof(bool));
    cudaMalloc((void**)&isErrorD, sizeof(bool));
    cudaMalloc((void**)&isErrorD2, sizeof(bool));
    *isErrorH = false;
    cudaMemcpy(isErrorD, isErrorH, sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(isErrorD2, isErrorH, sizeof(bool), cudaMemcpyHostToDevice);

    Contact_i.resize(numAllMarkers);
    uint numThreads, numBlocks;
    computeGridSize(numAllMarkers, 256, numBlocks, numThreads);
    thrust::fill(Contact_i.begin(), Contact_i.end(), 0);
    thrust::fill(_sumWij_inv.begin(), _sumWij_inv.end(), 1e-3);
    thrust::fill(A_i.begin(), A_i.end(), 0);
    thrust::fill(L_i.begin(), L_i.end(), 0);
    thrust::fill(G_i.begin(), G_i.end(), 0);

    //============================================================================================================
    *isErrorH = false;
    cudaMemcpy(isErrorD, isErrorH, sizeof(bool), cudaMemcpyHostToDevice);
    calcRho_kernel<<<numBlocks, numThreads>>>(
        mR4CAST(sortedSphMarkersD->posRadD), mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(_sumWij_inv),
        U1CAST(markersProximityD->cellStartD), U1CAST(markersProximityD->cellEndD), U1CAST(Contact_i), numAllMarkers,
        isErrorD);
    ChDeviceUtils::Sync_CheckError(isErrorH, isErrorD, "calcRho_kernel");

    uint LastVal = Contact_i[numAllMarkers - 1];
    thrust::exclusive_scan(Contact_i.begin(), Contact_i.end(), Contact_i.begin());
    Contact_i.push_back(LastVal + Contact_i[numAllMarkers - 1]);
    int NNZ = Contact_i[numAllMarkers];
    csrValGradient.resize(NNZ);
    csrValLaplacian.resize(NNZ);
    csrValFunciton.resize(NNZ);
    AMatrix.resize(NNZ);
    csrColInd.resize(NNZ);
    thrust::fill(csrValGradient.begin(), csrValGradient.end(), mR3(0.0));
    thrust::fill(csrValLaplacian.begin(), csrValLaplacian.end(), 0.0);
    thrust::fill(csrValFunciton.begin(), csrValFunciton.end(), 0.0);
    thrust::fill(csrColInd.begin(), csrColInd.end(), 0.0);
    thrust::fill(AMatrix.begin(), AMatrix.end(), 0.0);

    calcNormalizedRho_Gi_fillInMatrixIndices<<<numBlocks, numThreads>>>(
        mR4CAST(sortedSphMarkersD->posRadD), mR3CAST(sortedSphMarkersD->velMasD),
        mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(_sumWij_inv), R1CAST(G_i), mR3CAST(Normals), U1CAST(csrColInd),
        U1CAST(Contact_i), U1CAST(markersProximityD->cellStartD), U1CAST(markersProximityD->cellEndD), numAllMarkers,
        isErrorD);
    ChDeviceUtils::Sync_CheckError(isErrorH, isErrorD, "calcNormalizedRho_Gi_fillInMatrixIndices");
    //    //
    //    CopySortedToOriginal_NonInvasive_R3(fsiGeneralData->vel_IISPH_D, sortedSphMarkersD->velMasD,
    //                                        markersProximityD->gridMarkerIndexD);
    //    CopySortedToOriginal_NonInvasive_R3(sphMarkersD->velMasD, sortedSphMarkersD->velMasD,
    //                                        markersProximityD->gridMarkerIndexD);
    //
    //    return;

    //============================================================================================================
    double A_L_Tensor_GradLaplacian = clock();
    printf(" calc_A_tensor+");
    calc_A_tensor<<<numBlocks, numThreads>>>(R1CAST(A_i), R1CAST(G_i), mR4CAST(sortedSphMarkersD->posRadD),
                                             mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(_sumWij_inv),
                                             U1CAST(markersProximityD->cellStartD), U1CAST(markersProximityD->cellEndD),
                                             numAllMarkers, isErrorD);
    ChDeviceUtils::Sync_CheckError(isErrorH, isErrorD, "calc_A_tensor");
    printf("calc_L_tensor+");
    calc_L_tensor<<<numBlocks, numThreads>>>(R1CAST(A_i), R1CAST(L_i), R1CAST(G_i), mR4CAST(sortedSphMarkersD->posRadD),
                                             mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(_sumWij_inv),
                                             U1CAST(markersProximityD->cellStartD), U1CAST(markersProximityD->cellEndD),
                                             numAllMarkers, isErrorD);
    ChDeviceUtils::Sync_CheckError(isErrorH, isErrorD, "calc_L_tensor");
    printf("Gradient_Laplacian_Operator: ");
    Function_Gradient_Laplacian_Operator<<<numBlocks, numThreads>>>(
        mR4CAST(sortedSphMarkersD->posRadD), mR3CAST(sortedSphMarkersD->velMasD),
        mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(_sumWij_inv), R1CAST(G_i), R1CAST(L_i), R1CAST(csrValLaplacian),
        mR3CAST(csrValGradient), R1CAST(csrValFunciton), U1CAST(csrColInd), U1CAST(Contact_i), numAllMarkers, isErrorD);
    ChDeviceUtils::Sync_CheckError(isErrorH, isErrorD, "Gradient_Laplacian_Operator");
    double Gradient_Laplacian_Operator = (clock() - A_L_Tensor_GradLaplacian) / (double)CLOCKS_PER_SEC;
    printf("%f (s)\n", Gradient_Laplacian_Operator);
    int numFlexbodies = +numObjectsH->numFlexBodies1D + numObjectsH->numFlexBodies2D;
    int haveGhost = (numObjectsH->numGhostMarkers > 0) ? 1 : 0;
    int haveHelper = (numObjectsH->numHelperMarkers > 0) ? 1 : 0;
    int4 updatePortion =
        mI4(fsiGeneralData->referenceArray[haveGhost + haveHelper + 0].y,  // end of fluid
            fsiGeneralData->referenceArray[haveGhost + haveHelper + 1].y,  // end of boundary
            fsiGeneralData->referenceArray[haveGhost + haveHelper + 1 + numObjectsH->numRigidBodies].y,
            fsiGeneralData->referenceArray[haveGhost + haveHelper + 1 + numObjectsH->numRigidBodies + numFlexbodies].y);

    thrust::fill(V_star_old.begin(), V_star_old.end(), mR3(0.0));
    thrust::fill(V_star_new.begin(), V_star_new.end(), mR3(0.0));
    thrust::fill(q_old.begin(), q_old.end(), 0.0);
    thrust::fill(q_new.begin(), q_new.end(), 0.0);
    thrust::fill(b1Vector.begin(), b1Vector.end(), 0.0);
    thrust::fill(b3Vector.begin(), b3Vector.end(), mR3(0.0));
    thrust::fill(Residuals.begin(), Residuals.end(), 0.0);
    //============================================V_star_Predictor===============================================
    double LinearSystemClock_V = clock();
    V_star_Predictor<<<numBlocks, numThreads>>>(
        mR4CAST(sortedSphMarkersD->posRadD), mR3CAST(sortedSphMarkersD->velMasD),
        mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(AMatrix), mR3CAST(b3Vector), mR3CAST(V_star_old),
        R1CAST(csrValLaplacian), mR3CAST(csrValGradient), R1CAST(csrValFunciton), R1CAST(_sumWij_inv), mR3CAST(Normals),
        U1CAST(csrColInd), U1CAST(Contact_i),

        mR4CAST(otherFsiBodiesD->velMassRigid_fsiBodies_D), mR3CAST(otherFsiBodiesD->accRigid_fsiBodies_D),
        U1CAST(fsiGeneralData->rigidIdentifierD),

        mR3CAST(otherFsiMeshD->pos_fsi_fea_D), mR3CAST(otherFsiMeshD->vel_fsi_fea_D),
        mR3CAST(otherFsiMeshD->acc_fsi_fea_D), U1CAST(fsiGeneralData->FlexIdentifierD),

        numObjectsH->numFlexBodies1D, U2CAST(fsiGeneralData->CableElementsNodes),
        U4CAST(fsiGeneralData->ShellElementsNodes), updatePortion, U1CAST(markersProximityD->gridMarkerIndexD),
        numAllMarkers, isErrorD);
    ChDeviceUtils::Sync_CheckError(isErrorH, isErrorD, "V_star_Predictor");

    int Iteration = 0;
    Real MaxRes = 100;

    while ((MaxRes > paramsH->LinearSolver_Rel_Tol || Iteration < 3) && Iteration < paramsH->LinearSolver_Max_Iter) {
        Jacobi_SOR_Iter<<<numBlocks, numThreads>>>(mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(AMatrix),
                                                   mR3CAST(V_star_old), mR3CAST(V_star_new), mR3CAST(b3Vector),
                                                   R1CAST(q_old), R1CAST(q_new), R1CAST(b1Vector), U1CAST(csrColInd),
                                                   U1CAST(Contact_i), numAllMarkers, true, isErrorD);
        ChDeviceUtils::Sync_CheckError(isErrorH, isErrorD, "Jacobi_SOR_Iter");

        Update_AND_Calc_Res<<<numBlocks, numThreads>>>(mR4CAST(sortedSphMarkersD->rhoPresMuD), mR3CAST(V_star_old),
                                                       mR3CAST(V_star_new), R1CAST(q_old), R1CAST(q_new),
                                                       R1CAST(Residuals), numAllMarkers, true, isErrorD);
        ChDeviceUtils::Sync_CheckError(isErrorH, isErrorD, "Update_AND_Calc_Res");
        Iteration++;
        thrust::device_vector<Real>::iterator iter = thrust::max_element(Residuals.begin(), Residuals.end());
        unsigned int position = iter - Residuals.begin();
        MaxRes = *iter;

        if (paramsH->Verbose_monitoring)
            printf("Iter= %.4d, Res= %.4e\n", Iteration, MaxRes);
    }

    thrust::device_vector<Real3>::iterator iter =
        thrust::max_element(V_star_new.begin(), V_star_new.end(), compare_Real3_mag());
    unsigned int position = iter - V_star_new.begin();
    Real MaxVel = length(*iter);

    uint FixedMarker = 5679;
    Real temp = 0;
    //    for (int i = 0; i < numAllMarkers; i++) {
    //        if (Real4(sortedSphMarkersD->rhoPresMuD[i]).w == -1 && Real4(sortedSphMarkersD->posRadD[i]).x > temp)
    //        {
    //            FixedMarker = i;
    //            //            break;
    //        }
    //    }
    printf("Fixed marker %d\n", FixedMarker);

    double V_star_Predictor = (clock() - LinearSystemClock_V) / (double)CLOCKS_PER_SEC;
    printf(" V_star_Predictor Equation: %f (sec) - Final Residual=%.3e\n", V_star_Predictor, MaxRes);
    //==================================================Pressure_Equation==============================================
    Iteration = 0;
    MaxRes = 100;
    double LinearSystemClock_p = clock();
    thrust::fill(AMatrix.begin(), AMatrix.end(), 0.0);
    Pressure_Equation<<<numBlocks, numThreads>>>(
        mR4CAST(sortedSphMarkersD->posRadD), mR3CAST(sortedSphMarkersD->velMasD),
        mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(AMatrix), R1CAST(b1Vector), mR3CAST(V_star_new), R1CAST(q_new),
        R1CAST(csrValFunciton), R1CAST(csrValLaplacian), mR3CAST(csrValGradient), R1CAST(_sumWij_inv), mR3CAST(Normals),
        U1CAST(csrColInd), U1CAST(Contact_i),

        mR4CAST(otherFsiBodiesD->velMassRigid_fsiBodies_D), mR3CAST(otherFsiBodiesD->accRigid_fsiBodies_D),
        U1CAST(fsiGeneralData->rigidIdentifierD),

        mR3CAST(otherFsiMeshD->pos_fsi_fea_D), mR3CAST(otherFsiMeshD->vel_fsi_fea_D),
        mR3CAST(otherFsiMeshD->acc_fsi_fea_D), U1CAST(fsiGeneralData->FlexIdentifierD),

        numObjectsH->numFlexBodies1D, U2CAST(fsiGeneralData->CableElementsNodes),
        U4CAST(fsiGeneralData->ShellElementsNodes), updatePortion, U1CAST(markersProximityD->gridMarkerIndexD),
        numAllMarkers, FixedMarker, isErrorD);
    ChDeviceUtils::Sync_CheckError(isErrorH, isErrorD, "Pressure_Equation");

    if (paramsH->USE_LinearSolver) {
        ChFsiLinearSolver myLS(paramsH->LinearSolver, paramsH->LinearSolver_Abs_Tol, paramsH->LinearSolver_Abs_Tol,
                               paramsH->LinearSolver_Max_Iter, paramsH->Verbose_monitoring);
        if (paramsH->PPE_Solution_type != FORM_SPARSE_MATRIX) {
            printf(
                "You should paramsH->PPE_Solution_type == FORM_SPARSE_MATRIX in order to use the "
                "chrono_fsi linear "
                "solvers\n");
            exit(0);
        }
        myLS.Solve(numAllMarkers, NNZ, R1CAST(AMatrix), U1CAST(Contact_i), U1CAST(csrColInd), (double*)R1CAST(q_new),
                   R1CAST(b1Vector));
        cudaCheckError();
        if (myLS.GetSolverStatus()) {
            std::cout << " Linear solver converged to " << myLS.GetResidual() << " tolerance";
            std::cout << " after " << myLS.GetNumIterations() << " iterations" << std::endl;
        } else {
            std::cout << "Failed to converge after " << myLS.GetIterationLimit() << " iterations";
            std::cout << " (" << myLS.GetResidual() << " final residual)" << std::endl;
        }
    } else {
        while ((MaxRes > paramsH->LinearSolver_Abs_Tol || Iteration < 3) && paramsH->USE_Iterative_solver &&
               Iteration < paramsH->LinearSolver_Max_Iter) {
            Jacobi_SOR_Iter<<<numBlocks, numThreads>>>(
                mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(AMatrix), mR3CAST(V_star_old), mR3CAST(V_star_new),
                mR3CAST(b3Vector), R1CAST(q_old), R1CAST(q_new), R1CAST(b1Vector), U1CAST(csrColInd), U1CAST(Contact_i),
                numAllMarkers, false, isErrorD);
            ChDeviceUtils::Sync_CheckError(isErrorH, isErrorD, "Jacobi_SOR_Iter");

            Update_AND_Calc_Res<<<numBlocks, numThreads>>>(mR4CAST(sortedSphMarkersD->rhoPresMuD), mR3CAST(V_star_old),
                                                           mR3CAST(V_star_new), R1CAST(q_old), R1CAST(q_new),
                                                           R1CAST(Residuals), numAllMarkers, false, isErrorD);
            ChDeviceUtils::Sync_CheckError(isErrorH, isErrorD, "Update_AND_Calc_Res");
            Iteration++;
            thrust::device_vector<Real>::iterator iter = thrust::max_element(Residuals.begin(), Residuals.end());
            unsigned int position = iter - Residuals.begin();
            MaxRes = *iter;

            if (paramsH->Verbose_monitoring)
                printf("Iter= %.4d, Res= %.4e\n", Iteration, MaxRes);
        }
    }
    double Pressure_Computation = (clock() - LinearSystemClock_p) / (double)CLOCKS_PER_SEC;
    printf(" Pressure Poisson Equation: %f (sec) - Final Residual=%.3e\n", Pressure_Computation, MaxRes);
    //==================================Velocity_Correction_and_update============================================
    double updateClock = clock();
    Velocity_Correction_and_update<<<numBlocks, numThreads>>>(
        mR4CAST(sortedSphMarkersD->posRadD), mR4CAST(sortedSphMarkersD->rhoPresMuD),
        mR3CAST(sortedSphMarkersD->velMasD), mR3CAST(V_star_new), R1CAST(q_new), mR3CAST(csrValGradient),
        U1CAST(csrColInd), U1CAST(Contact_i), numAllMarkers, MaxVel, isErrorD);
    ChDeviceUtils::Sync_CheckError(isErrorH, isErrorD, "Velocity_Correction_and_update");
    double updateComputation = (clock() - updateClock) / (double)CLOCKS_PER_SEC;
    printf(" Pressure Poisson Equation: %f (sec)\n", updateComputation);
    //============================================================================================================
    CopySortedToOriginal_NonInvasive_R3(fsiGeneralData->vel_IISPH_D, sortedSphMarkersD->velMasD,
                                        markersProximityD->gridMarkerIndexD);
    CopySortedToOriginal_NonInvasive_R3(sphMarkersD->velMasD, sortedSphMarkersD->velMasD,
                                        markersProximityD->gridMarkerIndexD);
    CopySortedToOriginal_NonInvasive_R4(sphMarkersD->rhoPresMuD, sortedSphMarkersD->rhoPresMuD,
                                        markersProximityD->gridMarkerIndexD);
    CopySortedToOriginal_NonInvasive_R4(sphMarkersD->posRadD, sortedSphMarkersD->posRadD,
                                        markersProximityD->gridMarkerIndexD);
    csrValGradient.clear();
    csrValLaplacian.clear();
    csrValFunciton.clear();
    AMatrix.clear();
    Contact_i.clear();
    csrColInd.clear();
}
}  // namespace fsi
}  // namespace chrono
