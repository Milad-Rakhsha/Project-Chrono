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

#include "chrono_fsi/ChFsiForceXSPH.cuh"

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
//==========================================================================================================================================
__global__ void NS_RHS_Predictor(Real4* sortedPosRad_tn,
                                 Real4* sortedPosRad,
                                 Real4* sortedRhoPreMu_tn,
                                 Real4* sortedRhoPreMu,
                                 Real3* sortedVelMas_tn,
                                 Real3* sortedVelMas,
                                 Real3* V_HalfStep,
                                 Real4* X_HalfStep,
                                 const Real3* A_G,
                                 const Real* A_L,
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

    bool Fluid_Marker = sortedRhoPreMu[i_idx].w == -1.0;
    bool Boundary_Marker = sortedRhoPreMu[i_idx].w > -1.0;

    if (sortedRhoPreMu[i_idx].w <= -2) {
        return;
    }

    Real3 grad_p_i = mR3(0.0);
    Real3 Laplacian_v_i = mR3(0.0);
    Real3 RHS = mR3(0.0);
    for (int count = csrStartIdx; count < csrEndIdx; count++) {
        int j = csrColInd[count];
        grad_p_i += A_G[count] * sortedRhoPreMu[j].y;
        Laplacian_v_i += A_L[count] * sortedVelMas[j];
    }

    RHS = -1 / sortedRhoPreMu[i_idx].x * grad_p_i                  // pressure gradient
          + paramsD.mu0 / sortedRhoPreMu[i_idx].x * Laplacian_v_i  // viscous term;
          + paramsD.gravity;                                       // body force

    V_HalfStep[i_idx] = sortedVelMas_tn[i_idx] + paramsD.dT / 2 * RHS;
    X_HalfStep[i_idx] = sortedPosRad_tn[i_idx] + paramsD.dT / 2 * mR4(V_HalfStep[i_idx], 0.0);
}

//==========================================================================================================================================
__global__ void Update(Real4* sortedPosRad,
                       Real4* sortedRhoPreMu,
                       Real3* sortedVelMas,
                       Real3* V_HalfStep,
                       Real4* X_HalfStep,
                       int numAllMarkers,
                       const Real MaxVel,
                       volatile bool* isErrorD) {
    uint i_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_idx >= numAllMarkers) {
        return;
    }

    bool Fluid_Marker = sortedRhoPreMu[i_idx].w == -1.0;
    bool Boundary_Marker = sortedRhoPreMu[i_idx].w > -1.0;

    if (sortedRhoPreMu[i_idx].w <= -2) {
        return;
    }

    if (Fluid_Marker) {
        sortedVelMas[i_idx] = 2 * V_HalfStep[i_idx] - sortedVelMas[i_idx];
        sortedPosRad[i_idx] = mR4(mR3(2 * X_HalfStep[i_idx] - sortedPosRad[i_idx]), sortedPosRad[i_idx].w);
    }
}
//==========================================================================================================================================
__global__ void Boundary_Conditions(Real4* sortedPosRad,
                                    Real4* sortedRhoPreMu,
                                    Real3* sortedVelMas,
                                    const Real3* A_G,
                                    const Real* A_L,
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
    if (sortedRhoPreMu[i_idx].w == -1.0) {
        return;
    }
    uint csrStartIdx = numContacts[i_idx];
    uint csrEndIdx = numContacts[i_idx + 1];

    Real h_i = sortedPosRad[i_idx].w;
    Real3 posRadA = mR3(sortedPosRad[i_idx]);
    Real3 myAcc = mR3(0);
    Real3 V_prescribed = mR3(0);
    BCE_Vel_Acc(i_idx, myAcc, V_prescribed, sortedPosRad, updatePortion, gridMarkerIndexD, velMassRigid_fsiBodies_D,
                accRigid_fsiBodies_D, rigidIdentifierD, pos_fsi_fea_D, vel_fsi_fea_D, acc_fsi_fea_D, FlexIdentifierD,
                numFlex1D, CableElementsNodes, ShellelementsNodes);
    Real pRHS1 = 0.0;
    Real pRHS2 = 0.0;

    Real den = 0.0;
    Real3 numV = mR3(0);

    for (int count = csrStartIdx + 1; count < csrEndIdx; count++) {
        uint j = csrColInd[count];
        if (sortedRhoPreMu[j].w != -1.0)
            continue;
        Real3 posRadB = mR3(sortedPosRad[j]);
        Real3 rij = Distance(posRadA, posRadB);
        Real h_j = sortedPosRad[j].w;
        Real h_ij = 0.5 * (h_j + h_i);
        Real W3 = W3h(length(rij), h_ij);
        numV += W3 * sortedVelMas[j];
        den += W3;
        pRHS1 += W3 * sortedRhoPreMu[j].y;
        pRHS2 += dot(paramsD.gravity - myAcc, rij) * sortedRhoPreMu[j].x * W3;
    }

    if (abs(den) > EPSILON) {
        sortedVelMas[i_idx] = 2 * V_prescribed - numV / den;
        sortedRhoPreMu[i_idx].y = (pRHS1 + pRHS2) / den;
    } else {
        sortedVelMas[i_idx] = mR3(0);
        sortedRhoPreMu[i_idx].y = 1000;
    }
}
//==========================================================================================================================================

ChFsiForceXSPH::ChFsiForceXSPH(
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

ChFsiForceXSPH::~ChFsiForceXSPH() {}

void ChFsiForceXSPH::Finalize() {
    ChFsiForce::Finalize();
    cudaMemcpyToSymbolAsync(paramsD, paramsH, sizeof(SimParams));
    cudaMemcpyToSymbolAsync(numObjectsD, numObjectsH, sizeof(NumberOfObjects));
    cudaMemcpyFromSymbol(paramsH, paramsD, sizeof(SimParams));
    cudaThreadSynchronize();
    CopyParams_NumberOfObjects(paramsH, numObjectsH);
}
//==========================================================================================================================================
//==========================================================================================================================================
//==========================================================================================================================================
void ChFsiForceXSPH::ForceSPH(SphMarkerDataD* otherSphMarkersD,
                              FsiBodiesDataD* otherFsiBodiesD,
                              FsiMeshDataD* otherFsiMeshD) {
    std::cout << "dT in ForceImplicitSPH: " << paramsH->dT << "\n";
    CopyParams_NumberOfObjects(paramsH, numObjectsH);

    sphMarkersD = otherSphMarkersD;
    int numAllMarkers = numObjectsH->numAllMarkers;
    int numHelperMarkers = numObjectsH->numHelperMarkers;
    fsiCollisionSystem->ArrangeData(sphMarkersD);
    printf("ForceXSPH numAllMarkers:%d,numHelperMarkers=%d\n", numAllMarkers, numHelperMarkers);

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
    thrust::device_vector<uint> csrColInd;
    thrust::device_vector<Real> csrValLaplacian;
    thrust::device_vector<Real3> csrValGradient;
    thrust::device_vector<Real> csrValFunciton;
    thrust::device_vector<Real> AMatrix;
    thrust::device_vector<Real3> V_Star;
    thrust::device_vector<Real3> Normals;

    _sumWij_inv.resize(numAllMarkers);
    Normals.resize(numAllMarkers);
    G_i.resize(numAllMarkers * 9);
    A_i.resize(numAllMarkers * 27);
    L_i.resize(numAllMarkers * 6);
    Contact_i.resize(numAllMarkers);
    thrust::fill(Contact_i.begin(), Contact_i.end(), 0);
    thrust::fill(_sumWij_inv.begin(), _sumWij_inv.end(), 1e-3);
    thrust::fill(A_i.begin(), A_i.end(), 0);
    thrust::fill(L_i.begin(), L_i.end(), 0);
    thrust::fill(G_i.begin(), G_i.end(), 0);

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

    calcNormalizedRho_Gi_fillInMatrixIndices<<<numBlocks, numThreads>>>(
        mR4CAST(sortedSphMarkersD->posRadD), mR3CAST(sortedSphMarkersD->velMasD),
        mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(_sumWij_inv), R1CAST(G_i), mR3CAST(Normals), U1CAST(csrColInd),
        U1CAST(Contact_i), U1CAST(markersProximityD->cellStartD), U1CAST(markersProximityD->cellEndD), numAllMarkers,
        isErrorD);
    ChDeviceUtils::Sync_CheckError(isErrorH, isErrorD, "calcNormalizedRho_Gi_fillInMatrixIndices");
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

    thrust::device_vector<Real3> V_HalfStep(numAllMarkers);
    thrust::device_vector<Real4> X_HalfStep(numAllMarkers);
    thrust::device_vector<Real4> RPMT_HalfStep(numAllMarkers);

    thrust::fill(V_HalfStep.begin(), V_HalfStep.end(), mR3(0.0));
    thrust::fill(X_HalfStep.begin(), X_HalfStep.end(), mR4(0.0));
    thrust::fill(RPMT_HalfStep.begin(), RPMT_HalfStep.end(), mR4(0.0));

    //============================================================================================================

    CopySortedToOriginal_NonInvasive_R3(fsiGeneralData->vel_IISPH_D, sortedSphMarkersD->velMasD,
                                        markersProximityD->gridMarkerIndexD);
    CopySortedToOriginal_NonInvasive_R3(sphMarkersD->velMasD, sortedSphMarkersD->velMasD,
                                        markersProximityD->gridMarkerIndexD);
    CopySortedToOriginal_NonInvasive_R4(sphMarkersD->rhoPresMuD, sortedSphMarkersD->rhoPresMuD,
                                        markersProximityD->gridMarkerIndexD);
    CopySortedToOriginal_NonInvasive_R4(sphMarkersD->posRadD, sortedSphMarkersD->posRadD,
                                        markersProximityD->gridMarkerIndexD);

    _sumWij_inv.clear();
    Contact_i.clear();
    AMatrix.clear();
    csrColInd.clear();
    G_i.clear();
    A_i.clear();
    L_i.clear();
    csrValLaplacian.clear();
    csrValGradient.clear();
    V_HalfStep.clear();
    X_HalfStep.clear();
    RPMT_HalfStep.clear();

}  // namespace fsi

}  // namespace fsi
}  // namespace chrono
