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
                                 Real* vx,
                                 Real* vy,
                                 Real* vz,

                                 const Real* A_L,
                                 const Real3* A_G,
                                 const Real* sumWij_inv,
                                 const uint* csrColInd,
                                 unsigned long int* GlobalcsrColInd,
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

    int TYPE_OF_NARKER = sortedRhoPreMu[i_idx].w;
    if (TYPE_OF_NARKER <= -2) {
        A_Matrix[csrStartIdx] = 1;
        b[i_idx] = mR3(0.0);
        vx[i_idx] = b[i_idx].x;
        vy[i_idx] = b[i_idx].y;
        vz[i_idx] = b[i_idx].z;
        return;
    }

    Real rho0 = paramsD.rho0;
    Real nu0 = paramsD.mu0 / rho0;
    Real dt = paramsD.dT;
    Real3 posRadA = mR3(sortedPosRad[i_idx]);

    if (TYPE_OF_NARKER == -1) {
        Real3 rhs = mR3(0.0);
        for (int count = csrStartIdx; count < csrEndIdx; count++) {
            int j = csrColInd[count];
            A_Matrix[count] = -nu0 / 2.0 * A_L[count];
            rhs += -1.0 / sortedRhoPreMu[j].x * A_G[count] * sortedRhoPreMu[j].y +
                   nu0 / 2.0 * A_L[count] * sortedVelMas[j];
        }
        b[i_idx] = rhs + paramsD.gravity + sortedVelMas[i_idx] / dt;
        A_Matrix[csrStartIdx] += 1 / dt;
    } else if (TYPE_OF_NARKER > -1) {
        uint havefluid = 0;
        Real Vi = sumWij_inv[i_idx];
        for (int count = csrStartIdx; count < csrEndIdx; count++) {
            int j = csrColInd[count];
            Real3 posRadB = mR3(sortedPosRad[j]);
            Real3 rij = Distance(posRadA, posRadB);
            Real h_j = sortedPosRad[j].w;
            Real h_i = sortedPosRad[i_idx].w;
            Real h_ij = 0.5 * (h_j + h_i);
            Real W3 = W3h(length(rij), h_ij);
            if (sortedRhoPreMu[j].w == -1) {
                A_Matrix[count] = W3 * Vi;
                havefluid++;
            }
        }
        Real3 myAcc = mR3(0);
        Real3 V_prescribed = mR3(0);
        BCE_Vel_Acc(i_idx, myAcc, V_prescribed, sortedPosRad, updatePortion, gridMarkerIndexD, velMassRigid_fsiBodies_D,
                    accRigid_fsiBodies_D, rigidIdentifierD, pos_fsi_fea_D, vel_fsi_fea_D, acc_fsi_fea_D,
                    FlexIdentifierD, numFlex1D, CableElementsNodes, ShellelementsNodes);

        if (havefluid == 0) {
            A_Matrix[csrStartIdx] = W3h(0, paramsD.HSML) * sumWij_inv[i_idx];
            b[i_idx] = mR3(0.0);
        } else {
            A_Matrix[csrStartIdx] += 1;
            b[i_idx] = 2 * V_prescribed;
        }
    }

    vx[i_idx] = b[i_idx].x;
    vy[i_idx] = b[i_idx].y;
    vz[i_idx] = b[i_idx].z;
}
//--------------------------------------------------------------------------------------------------------------------------------
__global__ void Pressure_Equation(Real4* sortedPosRad,  // input: sorted positions
                                  Real3* sortedVelMas,
                                  Real4* sortedRhoPreMu,
                                  Real* A_Matrix,
                                  Real* Bi,
                                  Real3* Vstar,

                                  const Real* A_L,
                                  const Real3* A_G,
                                  const Real* sumWij_inv,
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

    int TYPE_OF_NARKER = sortedRhoPreMu[i_idx].w;
    if (TYPE_OF_NARKER <= -2) {
        A_Matrix[csrStartIdx] = 1;
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

    if (TYPE_OF_NARKER == -1) {
        Real rhoi = sortedRhoPreMu[i_idx].x;
        for (int count = csrStartIdx; count < csrEndIdx; count++) {
            A_Matrix[count] = rhoi * A_L[count] - dot(grad_rho_i, A_G[count]);
        }
        Bi[i_idx] = div_vi_star / dt * rhoi * rhoi;
    } else if (TYPE_OF_NARKER > -1) {
        bool havefluid = false;
        Real h_i = sortedPosRad[i_idx].w;
        Real3 posRadA = mR3(sortedPosRad[i_idx]);
        Real3 myAcc = mR3(0);
        Real3 V_prescribed = mR3(0);
        BCE_Vel_Acc(i_idx, myAcc, V_prescribed, sortedPosRad, updatePortion, gridMarkerIndexD, velMassRigid_fsiBodies_D,
                    accRigid_fsiBodies_D, rigidIdentifierD, pos_fsi_fea_D, vel_fsi_fea_D, acc_fsi_fea_D,
                    FlexIdentifierD, numFlex1D, CableElementsNodes, ShellelementsNodes);
        Real pRHS = 0.0;
        for (int count = csrStartIdx; count < csrEndIdx; count++) {
            int j = csrColInd[count];
            if (sortedRhoPreMu[j].w == -1) {
                Real3 posRadB = mR3(sortedPosRad[j]);
                Real3 rij = Distance(posRadA, posRadB);
                Real h_j = sortedPosRad[j].w;
                Real h_ij = 0.5 * (h_j + h_i);
                Real W3 = W3h(length(rij), h_ij);
                A_Matrix[count] = -W3;
                havefluid = true;
                pRHS += dot(gravity - myAcc, rij) * sortedRhoPreMu[j].x * W3;
            }
        }
        if (havefluid) {
            A_Matrix[csrStartIdx] = 1 / sumWij_inv[i_idx];
            Bi[i_idx] = pRHS;
        } else {
            A_Matrix[csrStartIdx] = 1;
            Bi[i_idx] = 0;
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
        V_new[i_idx] = (b3vec[i_idx] - aij_vj) / A_Matrix[startIdx - 1];
    } else {
        Real aij_pj = 0.0;
        for (int myIdx = startIdx; myIdx < endIdx; myIdx++) {
            aij_pj += A_Matrix[myIdx] * q_old[csrColInd[myIdx]];
        }
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
    Real p_res = 0;
    if (_3dvector) {
        V_new[i_idx] = (1 - omega) * V_old[i_idx] + omega * V_new[i_idx];
        p_res = length(V_old[i_idx] - V_new[i_idx]);
        V_old[i_idx] = V_new[i_idx];
    } else {
        q_new[i_idx] = (1 - omega) * q_old[i_idx] + omega * q_new[i_idx];
        p_res = q_old[i_idx] - q_new[i_idx];
        q_old[i_idx] = q_new[i_idx];
    }
    Residuals[i_idx] = p_res;
    //    if (p_res > 1e5)
    //        printf("idx=%d, type=%f\n", i_idx, sortedRhoPreMu[i_idx].w);
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
    : ChFsiForceParallel(otherBceWorker,
                         otherSortedSphMarkersD,
                         otherMarkersProximityD,
                         otherFsiGeneralData,
                         otherParamsH,
                         otherNumObjects) {
    CopyParams_NumberOfObjects(paramsH, numObjectsH);
}

ChFsiForceI2SPH::~ChFsiForceI2SPH() {}

void ChFsiForceI2SPH::Finalize() {
    ChFsiForceParallel::Finalize();
    cudaMemcpyToSymbolAsync(paramsD, paramsH, sizeof(SimParams));
    cudaMemcpyToSymbolAsync(numObjectsD, numObjectsH, sizeof(NumberOfObjects));
    cudaMemcpyFromSymbol(paramsH, paramsD, sizeof(SimParams));
    cudaThreadSynchronize();
    CopyParams_NumberOfObjects(paramsH, numObjectsH);

    //    int numAllMarkers = numObjectsH->numAllMarkers;
    //    _sumWij_inv.resize(numAllMarkers);
    //    Color.resize(numAllMarkers);
    //    G_i.resize(numAllMarkers * 9);
    //    A_i.resize(numAllMarkers * 27);
    //    L_i.resize(numAllMarkers * 6);
    //    Contact_i.resize(numAllMarkers);
    //
    //    thrust::fill(Contact_i.begin(), Contact_i.end(), 1e-3);
    //    thrust::fill(_sumWij_inv.begin(), _sumWij_inv.end(), 1e-3);
    //    thrust::fill(A_i.begin(), A_i.end(), 0);
    //    thrust::fill(L_i.begin(), L_i.end(), 0);
    //    thrust::fill(G_i.begin(), G_i.end(), 0);
}
//==========================================================================================================================================
//==========================================================================================================================================
//==========================================================================================================================================
void ChFsiForceI2SPH::ForceImplicitSPH(SphMarkerDataD* otherSphMarkersD,
                                       FsiBodiesDataD* otherFsiBodiesD,
                                       FsiMeshDataD* otherFsiMeshD) {
    std::cout << "dT in ForceSPH before calcPressure: " << paramsH->dT << "\n";
    CopyParams_NumberOfObjects(paramsH, numObjectsH);

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

    thrust::device_vector<Real> _sumWij_inv;
    thrust::device_vector<uint> Contact_i;
    thrust::device_vector<Real> G_i;
    thrust::device_vector<Real> A_i;
    thrust::device_vector<Real> L_i;
    thrust::device_vector<Real> Color;
    thrust::device_vector<uint> csrColInd;
    thrust::device_vector<unsigned long int> GlobalcsrColInd;
    thrust::device_vector<Real> csrValLaplacian;
    thrust::device_vector<Real3> csrValGradient;
    thrust::device_vector<Real> AMatrix;
    thrust::device_vector<Real3> V_Star;

    _sumWij_inv.resize(numAllMarkers);
    Color.resize(numAllMarkers);
    G_i.resize(numAllMarkers * 9);
    A_i.resize(numAllMarkers * 27);
    L_i.resize(numAllMarkers * 6);
    Contact_i.resize(numAllMarkers);
    thrust::fill(Contact_i.begin(), Contact_i.end(), 0);
    thrust::fill(_sumWij_inv.begin(), _sumWij_inv.end(), 1e-3);
    thrust::fill(A_i.begin(), A_i.end(), 0);
    thrust::fill(L_i.begin(), L_i.end(), 0);
    thrust::fill(G_i.begin(), G_i.end(), 0);
    fsiCollisionSystem->ArrangeData(sphMarkersD);
    printf("ForceIISPH paramsH=%f,%f,%f\n", paramsH->cellSize.x, paramsH->cellSize.y, paramsH->cellSize.z);

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
    AMatrix.resize(NNZ);
    csrColInd.resize(NNZ);
    GlobalcsrColInd.resize(NNZ);
    thrust::fill(csrValGradient.begin(), csrValGradient.end(), mR3(0.0));
    thrust::fill(csrValLaplacian.begin(), csrValLaplacian.end(), 0.0);
    thrust::fill(GlobalcsrColInd.begin(), GlobalcsrColInd.end(), 0.0);
    thrust::fill(csrColInd.begin(), csrColInd.end(), 0.0);

    calcNormalizedRho_Gi_fillInMatrixIndices<<<numBlocks, numThreads>>>(
        mR4CAST(sortedSphMarkersD->posRadD), mR3CAST(sortedSphMarkersD->velMasD),
        mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(_sumWij_inv), R1CAST(G_i), U1CAST(csrColInd),
        LU1CAST(GlobalcsrColInd), U1CAST(Contact_i), U1CAST(markersProximityD->cellStartD),
        U1CAST(markersProximityD->cellEndD), numAllMarkers, isErrorD);
    ChDeviceUtils::Sync_CheckError(isErrorH, isErrorD, "calcNormalizedRho_Gi_fillInMatrixIndices");
    printf("=======================================\n");
    double A_L_Tensor_GradLaplacian = clock();
    calc_A_tensor<<<numBlocks, numThreads>>>(R1CAST(A_i), R1CAST(G_i), mR4CAST(sortedSphMarkersD->posRadD),
                                             mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(_sumWij_inv),
                                             U1CAST(markersProximityD->cellStartD), U1CAST(markersProximityD->cellEndD),
                                             numAllMarkers, isErrorD);
    ChDeviceUtils::Sync_CheckError(isErrorH, isErrorD, "calc_A_tensor");

    calc_L_tensor<<<numBlocks, numThreads>>>(R1CAST(A_i), R1CAST(L_i), R1CAST(G_i), mR4CAST(sortedSphMarkersD->posRadD),
                                             mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(_sumWij_inv),
                                             U1CAST(markersProximityD->cellStartD), U1CAST(markersProximityD->cellEndD),
                                             numAllMarkers, isErrorD);
    ChDeviceUtils::Sync_CheckError(isErrorH, isErrorD, "calc_L_tensor");

    Gradient_Laplacian_Operator<<<numBlocks, numThreads>>>(
        mR4CAST(sortedSphMarkersD->posRadD), mR3CAST(sortedSphMarkersD->velMasD),
        mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(_sumWij_inv), R1CAST(G_i), R1CAST(L_i), R1CAST(csrValLaplacian),
        mR3CAST(csrValGradient), U1CAST(csrColInd), LU1CAST(GlobalcsrColInd), U1CAST(Contact_i), numAllMarkers,
        isErrorD);
    ChDeviceUtils::Sync_CheckError(isErrorH, isErrorD, "Gradient_Laplacian_Operator");

    int numFlexbodies = +numObjectsH->numFlexBodies1D + numObjectsH->numFlexBodies2D;
    int haveGhost = (numObjectsH->numGhostMarkers > 0) ? 1 : 0;
    int haveHelper = (numObjectsH->numHelperMarkers > 0) ? 1 : 0;
    int4 updatePortion =
        mI4(fsiGeneralData->referenceArray[haveGhost + haveHelper + 0].y,  // end of fluid
            fsiGeneralData->referenceArray[haveGhost + haveHelper + 1].y,  // end of boundary
            fsiGeneralData->referenceArray[haveGhost + haveHelper + 1 + numObjectsH->numRigidBodies].y,
            fsiGeneralData->referenceArray[haveGhost + haveHelper + 1 + numObjectsH->numRigidBodies + numFlexbodies].y);

    thrust::device_vector<Real3> b3Vector(numAllMarkers);
    thrust::fill(b3Vector.begin(), b3Vector.end(), mR3(0.0));

    thrust::device_vector<Real> vx(numAllMarkers);
    thrust::device_vector<Real> vy(numAllMarkers);
    thrust::device_vector<Real> vz(numAllMarkers);
    thrust::device_vector<Real> xvec(numAllMarkers);
    double Gradient_Laplacian_Operator = (clock() - A_L_Tensor_GradLaplacian) / (double)CLOCKS_PER_SEC;
    printf(" Gradient_Laplacian_Operator Computation: %f \n", Gradient_Laplacian_Operator);

    printf("=======================================\n");
    double LinearSystemClock_V = clock();
    V_star_Predictor<<<numBlocks, numThreads>>>(
        mR4CAST(sortedSphMarkersD->posRadD), mR3CAST(sortedSphMarkersD->velMasD),
        mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(AMatrix), mR3CAST(b3Vector), R1CAST(vx), R1CAST(vy), R1CAST(vz),
        R1CAST(csrValLaplacian), mR3CAST(csrValGradient), R1CAST(_sumWij_inv), U1CAST(csrColInd),
        LU1CAST(GlobalcsrColInd), U1CAST(Contact_i),

        mR4CAST(otherFsiBodiesD->velMassRigid_fsiBodies_D), mR3CAST(otherFsiBodiesD->accRigid_fsiBodies_D),
        U1CAST(fsiGeneralData->rigidIdentifierD),

        mR3CAST(otherFsiMeshD->pos_fsi_fea_D), mR3CAST(otherFsiMeshD->vel_fsi_fea_D),
        mR3CAST(otherFsiMeshD->acc_fsi_fea_D), U1CAST(fsiGeneralData->FlexIdentifierD),

        numObjectsH->numFlexBodies1D, U2CAST(fsiGeneralData->CableElementsNodes),
        U4CAST(fsiGeneralData->ShellElementsNodes), updatePortion, U1CAST(markersProximityD->gridMarkerIndexD),
        numAllMarkers, isErrorD);
    ChDeviceUtils::Sync_CheckError(isErrorH, isErrorD, "V_star_Predictor");

    thrust::device_vector<Real3> V_star_new(numAllMarkers);
    thrust::device_vector<Real3> V_star_old(numAllMarkers);
    thrust::fill(V_star_old.begin(), V_star_old.end(), mR3(0.0));

    thrust::device_vector<Real> q_new(numAllMarkers);
    thrust::device_vector<Real> q_old(numAllMarkers);
    thrust::fill(q_old.begin(), q_old.end(), 0.0);

    thrust::device_vector<Real> b1Vector(numAllMarkers);
    thrust::fill(b1Vector.begin(), b1Vector.end(), 0.0);
    thrust::device_vector<Real> Residuals(numAllMarkers);
    int Iteration = 0;
    Real MaxRes = 100;

    while ((MaxRes > paramsH->LinearSolver_Rel_Tol || Iteration < 3) && paramsH->USE_Iterative_solver &&
           Iteration < paramsH->LinearSolver_Max_Iter) {
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

    //    ChFsiLinearSolver myLS(paramsH->LinearSolver, paramsH->LinearSolver_Rel_Tol,
    //    paramsH->LinearSolver_Abs_Tol,
    //                           paramsH->LinearSolver_Max_Iter, paramsH->Verbose_monitoring);
    //    if (paramsH->USE_LinearSolver) {
    //        if (paramsH->PPE_Solution_type != FORM_SPARSE_MATRIX) {
    //            printf(
    //                "You should paramsH->PPE_Solution_type == FORM_SPARSE_MATRIX in order to use the "
    //                "chrono_fsi linear "
    //                "solvers\n");
    //            exit(0);
    //        }
    //        thrust::fill(xvec.begin(), xvec.end(), 0.0);
    //
    //        myLS.Solve(numAllMarkers, NNZ, R1CAST(AMatrix), U1CAST(Contact_i), U1CAST(csrColInd),
    //        (double*)R1CAST(xvec),
    //                   R1CAST(vx));
    //        cudaCheckError();
    //    }

    double V_star_Predictor = (clock() - LinearSystemClock_V) / (double)CLOCKS_PER_SEC;
    printf(" V_star_Predictor Computation: %f \n", V_star_Predictor);

    printf("=======================================\n");
    thrust::fill(AMatrix.begin(), AMatrix.end(), 0.0);

    Iteration = 0;
    MaxRes = 100;
    double LinearSystemClock_p = clock();
    Pressure_Equation<<<numBlocks, numThreads>>>(
        mR4CAST(sortedSphMarkersD->posRadD), mR3CAST(sortedSphMarkersD->velMasD),
        mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(AMatrix), R1CAST(b1Vector), mR3CAST(V_star_new),
        R1CAST(csrValLaplacian), mR3CAST(csrValGradient), R1CAST(_sumWij_inv), U1CAST(csrColInd), U1CAST(Contact_i),

        mR4CAST(otherFsiBodiesD->velMassRigid_fsiBodies_D), mR3CAST(otherFsiBodiesD->accRigid_fsiBodies_D),
        U1CAST(fsiGeneralData->rigidIdentifierD),

        mR3CAST(otherFsiMeshD->pos_fsi_fea_D), mR3CAST(otherFsiMeshD->vel_fsi_fea_D),
        mR3CAST(otherFsiMeshD->acc_fsi_fea_D), U1CAST(fsiGeneralData->FlexIdentifierD),

        numObjectsH->numFlexBodies1D, U2CAST(fsiGeneralData->CableElementsNodes),
        U4CAST(fsiGeneralData->ShellElementsNodes), updatePortion, U1CAST(markersProximityD->gridMarkerIndexD),
        numAllMarkers, isErrorD);
    ChDeviceUtils::Sync_CheckError(isErrorH, isErrorD, "Pressure_Equation");

    while ((MaxRes > paramsH->LinearSolver_Rel_Tol || Iteration < 3) && paramsH->USE_Iterative_solver &&
           Iteration < paramsH->LinearSolver_Max_Iter) {
        Jacobi_SOR_Iter<<<numBlocks, numThreads>>>(mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(AMatrix),
                                                   mR3CAST(V_star_old), mR3CAST(V_star_new), mR3CAST(b3Vector),
                                                   R1CAST(q_old), R1CAST(q_new), R1CAST(b1Vector), U1CAST(csrColInd),
                                                   U1CAST(Contact_i), numAllMarkers, false, isErrorD);
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

    double Pressure_Computation = (clock() - LinearSystemClock_p) / (double)CLOCKS_PER_SEC;
    printf(" Pressure Computation: %f \n", Pressure_Computation);
    printf("=======================================\n");

    //    thrust::device_vector<Real> AMatrixHost(NNZ);
    //    thrust::copy(AMatrix.begin(), AMatrix.end(), AMatrixHost.begin());
    //
    //    thrust::device_vector<Real> csrColIndHost(NNZ);
    //    thrust::copy(csrColInd.begin(), csrColInd.end(), csrColIndHost.begin());
    //
    //    thrust::device_vector<Real> Contact_iHost(NNZ);
    //    thrust::copy(Contact_i.begin(), Contact_i.end(), Contact_iHost.begin());
    //
    //    static int fileCounter = 0;
    //    std::string name = std::string("AMat") + std::string(".txt");
    //    std::ofstream ostrean;
    //    ostrean.open(name);
    //    std::stringstream sstream;
    //
    //    sstream << "%%MatrixMarket matrix coordinate real general"
    //            << "\n";
    //    sstream << numAllMarkers << " " << numAllMarkers << " " << NNZ << "\n";
    //
    //    for (int i = 0; i < numAllMarkers; i++) {
    //        for (int j = Contact_iHost[i]; j < Contact_iHost[i + 1]; j++) {
    //            sstream << i + 1 << " " << csrColIndHost[j] + 1 << " " << AMatrixHost[j] << "\n";
    //        }
    //        std::cout << i << "\n";
    //    }
    //    ostrean << sstream.str();
    //    ostrean.close();

    Contact_i.clear();
    Color.clear();
    _sumWij_inv.clear();
    G_i.clear();
    A_i.clear();
    L_i.clear();
}  // namespace fsi

}  // namespace fsi
}  // namespace chrono
