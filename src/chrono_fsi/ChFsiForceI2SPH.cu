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
#include "chrono_fsi/ChFsiForceI2SPH.cuh"
#include "chrono_fsi/ChParams.cuh"
#include "chrono_fsi/ChSphGeneral.cu"

//==========================================================================================================================================
namespace chrono {
namespace fsi {
//extern __constant__ SimParams paramsD;
//extern __constant__ NumberOfObjects numObjectsD;

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
inline __global__ void V_star_Predictor(Real4* sortedPosRad,  // input: sorted positions
                                        Real3* sortedVelMas,
                                        Real4* sortedRhoPreMu,
                                        Real* A_Matrix,
                                        Real3* b,
                                        Real* A_L,
                                        Real3* A_G,
                                        Real* sumWij_inv,
                                        uint* csrColInd,
                                        unsigned long int* GlobalcsrColInd,
                                        uint* numContacts,

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

                                        const int numAllMarkers,
                                        volatile bool* isErrorD) {
    uint i_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_idx >= numAllMarkers || sortedRhoPreMu[i_idx].w <= -2) {
        return;
    }
    Real rho0 = paramsD.rho0;
    Real nu0 = paramsD.mu0 / rho0;
    Real dt = paramsD.dT;
    int TYPE_OF_NARKER = sortedRhoPreMu[i_idx].w;
    Real3 posRadA = mR3(sortedPosRad[i_idx]);

    uint csrStartIdx = numContacts[i_idx] + 1;  // Reserve the starting index for the A_ii
    uint csrEndIdx = numContacts[i_idx + 1];

    if (TYPE_OF_NARKER == -1) {
        for (int count = csrStartIdx - 1; count < csrEndIdx; count++) {
            int j = csrColInd[count];
            A_Matrix[count] = +nu0 / 2 * A_L[count];
            Real3 rhsj = +1 / rho0 * A_G[count] * sortedRhoPreMu[j].y - nu0 / 2 * A_L[count] * sortedVelMas[j];
            datomicAdd(&b[j].x, rhsj.x);
            datomicAdd(&b[j].y, rhsj.y);
            datomicAdd(&b[j].z, rhsj.z);
        }
    } else if (TYPE_OF_NARKER > -1) {
        for (int count = csrStartIdx - 1; count < csrEndIdx; count++) {
            int j = csrColInd[count];
            Real3 posRadB = mR3(sortedPosRad[j]);
            Real3 rij = Distance(posRadA, posRadB);
            Real h_j = sortedPosRad[j].w;
            Real h_i = sortedPosRad[i_idx].w;
            Real h_ij = 0.5 * (h_j + h_i);
            Real W3 = W3h(length(rij), h_ij);
            if (sortedRhoPreMu[j].w == -1)
                A_Matrix[count] = W3 * sumWij_inv[i_idx];
        }
    }

    if (TYPE_OF_NARKER == -1) {
        A_Matrix[csrStartIdx - 1] = 1;
        b[csrStartIdx - 1] = mR3(0.0);
    } else if (TYPE_OF_NARKER == -1) {
        A_Matrix[csrStartIdx - 1] += 1 / dt;
        b[csrStartIdx - 1] -= paramsD.gravity - mR3(1 / dt);
    } else if (TYPE_OF_NARKER > -1) {
        Real3 myAcc = mR3(0);
        Real3 V_prescribed = mR3(0);
        BCE_Vel_Acc(i_idx, myAcc, V_prescribed, sortedPosRad, updatePortion, gridMarkerIndexD, velMassRigid_fsiBodies_D,
                    accRigid_fsiBodies_D, rigidIdentifierD, pos_fsi_fea_D, vel_fsi_fea_D, acc_fsi_fea_D,
                    FlexIdentifierD, numFlex1D, CableElementsNodes, ShellelementsNodes);
        A_Matrix[csrStartIdx - 1] = 1;
        b[csrStartIdx - 1] = 2 * V_prescribed;
    }
}  // namespace fsi

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
                         otherNumObjects) {}

ChFsiForceI2SPH::~ChFsiForceI2SPH() {}

void ChFsiForceI2SPH::Finalize() {
    ChFsiForceParallel::Finalize();
    cudaMemcpyToSymbolAsync(paramsD, paramsH, sizeof(SimParams));
    cudaMemcpyToSymbolAsync(numObjectsD, numObjectsH, sizeof(NumberOfObjects));
    cudaMemcpyFromSymbol(paramsH, paramsD, sizeof(SimParams));
    cudaThreadSynchronize();
    CopyParams_NumberOfObjects(paramsH, numObjectsH);

    int numAllMarkers = numObjectsH->numAllMarkers;
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

void ChFsiForceI2SPH::ForceImplicitSPH(SphMarkerDataD* otherSphMarkersD,
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
    thrust::device_vector<Real3> bVector;

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
    bVector.resize(numAllMarkers);
    thrust::fill(bVector.begin(), bVector.end(), mR3(0.0));
    V_star_Predictor<<<numBlocks, numThreads>>>(
        mR4CAST(sortedSphMarkersD->posRadD), mR3CAST(sortedSphMarkersD->velMasD),
        mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(AMatrix), mR3CAST(bVector), R1CAST(csrValLaplacian),
        mR3CAST(csrValGradient), R1CAST(_sumWij_inv), U1CAST(csrColInd), LU1CAST(GlobalcsrColInd), U1CAST(Contact_i),

        mR4CAST(otherFsiBodiesD->velMassRigid_fsiBodies_D), mR3CAST(otherFsiBodiesD->accRigid_fsiBodies_D),
        U1CAST(fsiGeneralData->rigidIdentifierD),

        mR3CAST(otherFsiMeshD->pos_fsi_fea_D), mR3CAST(otherFsiMeshD->vel_fsi_fea_D),
        mR3CAST(otherFsiMeshD->acc_fsi_fea_D), U1CAST(fsiGeneralData->FlexIdentifierD),

        numObjectsH->numFlexBodies1D, U2CAST(fsiGeneralData->CableElementsNodes),
        U4CAST(fsiGeneralData->ShellElementsNodes), updatePortion, U1CAST(markersProximityD->gridMarkerIndexD),
        numAllMarkers, isErrorD);
    ChDeviceUtils::Sync_CheckError(isErrorH, isErrorD, "V_star_Predictor");

    Contact_i.clear();
    Color.clear();
    _sumWij_inv.clear();
    G_i.clear();
    A_i.clear();
    L_i.clear();
}

}  // namespace fsi
}  // namespace chrono
