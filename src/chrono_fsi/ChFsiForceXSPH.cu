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

//==========================================================================================================================================
__global__ void calculate_pressure(Real4* sortedRhoPreMu,
                                   const Real v_max,
                                   const Real Min_rho,
                                   const int numAllMarkers,
                                   volatile bool* isErrorD) {
    uint i_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_idx >= numAllMarkers) {
        return;
    }

    sortedRhoPreMu[i_idx].y = EOS_new(sortedRhoPreMu[i_idx].x, v_max, Min_rho);
}
//==========================================================================================================================================
__global__ void NS_RHS_Predictor(Real4* sortedPosRad,
                                 Real4* sortedPosRad_old,
                                 Real4* sortedRhoPreMu,
                                 Real4* sortedRhoPreMu_old,
                                 Real3* sortedVelMas,
                                 Real3* sortedVelMas_old,
                                 Real3* NS_RHS,
                                 const Real3* A_G,
                                 const Real* A_L,
                                 const uint* csrColInd,
                                 const uint* numContacts,
                                 const Real delta_t,
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
        return;
    }

    Real3 grad_p_i = mR3(0.0);
    Real3 Laplacian_v_i = mR3(0.0);
    Real3 RHS = mR3(0.0);
    for (int count = csrStartIdx; count < csrEndIdx; count++) {
        int j = csrColInd[count];
        grad_p_i += A_G[count] * sortedRhoPreMu_old[j].y;
        Laplacian_v_i += A_L[count] * sortedVelMas_old[j];
    }

    RHS = -1 / sortedRhoPreMu_old[i_idx].x * grad_p_i                  // pressure gradient
          + paramsD.mu0 / sortedRhoPreMu_old[i_idx].x * Laplacian_v_i  // viscous term;
          + paramsD.gravity;                                           // body force

    NS_RHS[i_idx] = RHS;
    if (Fluid_Marker)
        sortedPosRad[i_idx] = sortedPosRad_old[i_idx] + delta_t / 2 * mR4(sortedVelMas_old[i_idx], 0.0);
    sortedVelMas[i_idx] = sortedVelMas_old[i_idx] + delta_t / 2 * RHS;
}
//==========================================================================================================================================
__global__ void Update(Real4* PosRad_tn,
                       Real4* RhoPreMu_tn,
                       Real3* VelMas_tn,
                       Real3* VelMas_next,
                       Real4* PosRad_next,
                       Real3* fthalf,
                       const Real delta_t,
                       int numAllMarkers,
                       // const Real MaxVel,
                       volatile bool* isErrorD) {
    uint i_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_idx >= numAllMarkers) {
        return;
    }

    bool Fluid_Marker = (RhoPreMu_tn[i_idx].w == -1.0);

    if (RhoPreMu_tn[i_idx].w <= -2) {
        return;
    }

    if (Fluid_Marker) {
        VelMas_next[i_idx] = VelMas_tn[i_idx] + delta_t * fthalf[i_idx];
        PosRad_next[i_idx] =
            mR4(mR3(PosRad_tn[i_idx]) + delta_t * VelMas_tn[i_idx] + 0.5 * delta_t * delta_t * fthalf[i_idx],
                PosRad_tn[i_idx].w);
    }
}
//==========================================================================================================================================
__global__ void Shifting_r(Real4* sortedPosRad,
                           Real4* sortedPosRad_old,
                           Real4* sortedRhoPreMu,
                           Real4* sortedRhoPreMu_old,
                           Real3* sortedVelMas,
                           Real3* sortedVelMas_old,
                           Real3* sortedVisualVel,
                           const Real* A_f,
                           const Real3* A_G,
                           const uint* csrColInd,
                           const uint* numContacts,
                           const Real MaxVel,
                           const Real delta_t,
                           int numAllMarkers,
                           volatile bool* isErrorD) {
    uint i_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_idx >= numAllMarkers) {
        return;
    }

    bool Fluid_Marker = (sortedRhoPreMu[i_idx].w == -1.0);

    uint csrStartIdx = numContacts[i_idx];
    uint csrEndIdx = numContacts[i_idx + 1];

    if (sortedRhoPreMu[i_idx].w <= -2) {
        return;
    }

    if (Fluid_Marker) {
        Real3 inner_sum = mR3(0.0), shift_r = mR3(0.0);
        Real mi_bar = 0.0, r0 = 0.0;
        Real3 posRadA = mR3(sortedPosRad[i_idx]);
        Real h_i = sortedPosRad[i_idx].w;
        Real3 grad_i_wij = mR3(0.0);
        Real summation = 0.0;

        for (int count = csrStartIdx + 1; count < csrEndIdx; count++) {
            uint j = csrColInd[count];
            Real3 posRadB = mR3(sortedPosRad_old[j]);
            Real h_j = sortedPosRad_old[j].w;
            Real h_ij = 0.5 * (h_j + h_i);
            Real3 rij = Distance(posRadA, posRadB);
            Real d = length(rij);
            Real m_j = pow(sortedPosRad_old[j].w * paramsD.MULT_INITSPACE, 3) * paramsD.rho0;
            mi_bar += m_j;
            r0 += d;
            inner_sum += m_j * rij / (d * d * d);

            Real3 grad_i_wij = GradWh(rij, h_ij);
            summation += m_j * dot((sortedVelMas_old[i_idx] - sortedVelMas_old[j]), grad_i_wij);
        }
        r0 /= (csrEndIdx - csrStartIdx - 1);
        mi_bar /= (csrEndIdx - csrStartIdx);
        shift_r = paramsD.beta_shifting * r0 * r0 * length(MaxVel) * delta_t / mi_bar * inner_sum;
        sortedRhoPreMu[i_idx].x += delta_t * summation;
        sortedRhoPreMu[i_idx].y = Eos(sortedRhoPreMu_old[i_idx].x, sortedRhoPreMu_old[i_idx].w);

        Real3 grad_p = mR3(0.0);
        Real3 grad_rho = mR3(0.0);
        Real3 grad_ux = mR3(0.0);
        Real3 grad_uy = mR3(0.0);
        Real3 grad_uz = mR3(0.0);
        Real3 vis_vel = mR3(0.0);

        for (int count = csrStartIdx; count < csrEndIdx; count++) {
            uint j = csrColInd[count];
            grad_rho += A_G[count] * sortedRhoPreMu_old[i_idx].x;
            grad_p += A_G[count] * sortedRhoPreMu_old[i_idx].y;
            grad_ux += A_G[count] * sortedVelMas_old[i_idx].x;
            grad_uy += A_G[count] * sortedVelMas_old[i_idx].y;
            grad_uz += A_G[count] * sortedVelMas_old[i_idx].z;
            vis_vel += A_f[count] * sortedVelMas_old[i_idx];
        }

        sortedPosRad[i_idx] += mR4(shift_r, 0.0);
        sortedRhoPreMu[i_idx].x += dot(shift_r, grad_rho);
        sortedRhoPreMu[i_idx].y += dot(shift_r, grad_p);
        sortedVelMas[i_idx].x += dot(shift_r, grad_ux);
        sortedVelMas[i_idx].y += dot(shift_r, grad_uy);
        sortedVelMas[i_idx].z += dot(shift_r, grad_uz);
        sortedVisualVel[i_idx] = vis_vel;
    }
}
//==========================================================================================================================================
__global__ void Boundary_Conditions(Real4* sortedPosRad,
                                    Real4* sortedRhoPreMu,
                                    Real4* sortedRhoPreMu_old,
                                    Real3* sortedVelMas,
                                    Real3* sortedVelMas_old,

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
        numV += W3 * sortedVelMas_old[j];
        den += W3;
        pRHS1 += W3 * sortedRhoPreMu_old[j].y;
        pRHS2 += dot(paramsD.gravity - myAcc, rij) * sortedRhoPreMu_old[j].x * W3;
    }

    if (abs(den) > 1e-14) {
        sortedVelMas[i_idx] = 2 * V_prescribed - numV / den;
        sortedRhoPreMu[i_idx].y = (pRHS1 + pRHS2) / den;
    } else {
        sortedVelMas[i_idx] = 2 * V_prescribed;
        sortedRhoPreMu[i_idx].y = paramsD.BASEPRES;
    }
}

//==========================================================================================================================================
__global__ void Update_Vel_XSPH(Real4* sortedPosRad,
                                Real4* sortedRhoPreMu,
                                Real3* sortedVelMas,
                                Real3* sortedVelMas_old,
                                Real3* vis_vel,

                                const uint* csrColInd,
                                const uint* numContacts,
                                uint numAllMarkers,
                                volatile bool* isErrorD) {
    uint i_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_idx >= numAllMarkers) {
        return;
    }
    bool Fluid_Marker = (sortedRhoPreMu[i_idx].w == -1.0);

    if (sortedRhoPreMu[i_idx].w <= -2) {
        return;
    }

    Real3 deltaV = mR3(0);
    Real3 posRadA = mR3(sortedPosRad[i_idx]);
    Real h_i = sortedPosRad[i_idx].w;

    uint csrStartIdx = numContacts[i_idx];
    uint csrEndIdx = numContacts[i_idx + 1];
    if (Fluid_Marker) {
        Real rho_i = sortedRhoPreMu[i_idx].x;
        Real3 vel_i = sortedVelMas[i_idx];
        for (int count = csrStartIdx; count < csrEndIdx; count++) {
            uint j = csrColInd[count];
            Real3 posRadB = mR3(sortedPosRad[j]);
            Real3 rij = Distance(posRadA, posRadB);
            Real h_j = sortedPosRad[j].w;
            Real m_j = pow(h_j * paramsD.MULT_INITSPACE, 3) * paramsD.rho0;
            Real h_ij = 0.5 * (h_j + h_i);
            Real W3 = W3h(length(rij), h_ij);
            deltaV += 2.0f * m_j / (rho_i + sortedRhoPreMu[j].x) * (sortedVelMas_old[j] - vel_i) * W3;
            vis_vel[i_idx] += m_j / sortedRhoPreMu[j].x * (sortedVelMas_old[j]) * W3;
        }
        //        sortedVelMas[i_idx] += paramsD.EPS_XSPH * deltaV;
        sortedVelMas[i_idx] = (1 - paramsD.EPS_XSPH) * sortedVelMas_old[i_idx] + paramsD.EPS_XSPH * vis_vel[i_idx];
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

    SphMarkerDataD SphMarkerDataD1 = *otherSphMarkersD;

    int numAllMarkers = numObjectsH->numAllMarkers;
    int numHelperMarkers = numObjectsH->numHelperMarkers;

    fsiCollisionSystem->ArrangeData(otherSphMarkersD);
    printf("ForceXSPH numAllMarkers:%d,numHelperMarkers=%d\n", numAllMarkers, numHelperMarkers);
    thrust::device_vector<Real3>::iterator iter =
        thrust::max_element(sortedSphMarkersD->velMasD.begin(), sortedSphMarkersD->velMasD.end(), compare_Real3_mag());
    Real MaxVel = length(*iter);

    paramsH->v_Max = MaxVel;
    if (paramsH->Adaptive_time_stepping) {
        Real dt_CFL = paramsH->Co_number * paramsH->HSML / (MaxVel);
        Real dt_nu = 0.125 * paramsH->HSML * paramsH->HSML / (paramsH->mu0 / paramsH->rho0);
        Real dt_body = 0.125 * std::sqrt(paramsH->HSML / length(paramsH->bodyForce3 + paramsH->gravity));
        Real dt = std::min(dt_body, std::min(dt_CFL, dt_nu));
        if (dt / paramsH->dT_Max > 0.7 && dt / paramsH->dT_Max < 1)
            paramsH->dT = paramsH->dT_Max * 0.5;
        else
            paramsH->dT = std::min(dt, paramsH->dT_Max);

        //        if (paramsH->dT < 1e-5)
        //            paramsH->dT = 1e-5;
        CopyParams_NumberOfObjects(paramsH, numObjectsH);

        printf(" time step=%.3e, dt_Max=%.3e, dt_CFL=%.3e (CFL=%.2g), dt_nu=%.3e, dt_body=%.3e\n", paramsH->dT,
               paramsH->dT_Max, dt_CFL, paramsH->Co_number, dt_nu, dt_body);
    }

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

    thrust::device_vector<Real3> ft(numAllMarkers, mR3(0));

    *isErrorH = false;
    cudaMemcpy(isErrorD, isErrorH, sizeof(bool), cudaMemcpyHostToDevice);
    //============================================================================================================
    // some initialization
    // 1st call
    calcRho_kernel<<<numBlocks, numThreads>>>(
        mR4CAST(sortedSphMarkersD->posRadD), mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(_sumWij_inv),
        U1CAST(markersProximityD->cellStartD), U1CAST(markersProximityD->cellEndD), U1CAST(Contact_i), numAllMarkers,
        isErrorD);
    ChDeviceUtils::Sync_CheckError(isErrorH, isErrorD, "calcRho_kernel");
    //============================================================================================================

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
    thrust::device_vector<Real3> vel_vis(numAllMarkers);
    thrust::fill(vel_vis.begin(), vel_vis.end(), mR3(0.0));
    //============================================================================================================
    calcNormalizedRho_Gi_fillInMatrixIndices<<<numBlocks, numThreads>>>(
        mR4CAST(sortedSphMarkersD->posRadD), mR3CAST(sortedSphMarkersD->velMasD),
        mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(_sumWij_inv), R1CAST(G_i), mR3CAST(Normals), U1CAST(csrColInd),
        U1CAST(Contact_i), U1CAST(markersProximityD->cellStartD), U1CAST(markersProximityD->cellEndD), numAllMarkers,
        isErrorD);
    ChDeviceUtils::Sync_CheckError(isErrorH, isErrorD, "calcNormalizedRho_Gi_fillInMatrixIndices");

    //============================================================================================================
    double A_L_Tensor_GradLaplacian = clock();
    printf("calc_A_tensor+");
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
    //============================================================================================================
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
    //============================================================================================================
    thrust::device_vector<Real4>::iterator iter_rho = thrust::min_element(
        sortedSphMarkersD->rhoPresMuD.begin(), sortedSphMarkersD->rhoPresMuD.end(), compare_Real4_x());
    Real Min_rho = length(*iter_rho);
    calculate_pressure<<<numBlocks, numThreads>>>(mR4CAST(sortedSphMarkersD->rhoPresMuD), MaxVel, Min_rho,
                                                  numAllMarkers, isErrorD);
    ChDeviceUtils::Sync_CheckError(isErrorH, isErrorD, "calculate_pressure-1");
    //============================================================================================================
    thrust::device_vector<Real4> rhoPresMuD_old = sortedSphMarkersD->rhoPresMuD;
    thrust::device_vector<Real4> posRadD_old = sortedSphMarkersD->posRadD;
    thrust::device_vector<Real3> velMasD_old = sortedSphMarkersD->velMasD;
    Update_Vel_XSPH<<<numBlocks, numThreads>>>(
        mR4CAST(sortedSphMarkersD->posRadD), mR4CAST(sortedSphMarkersD->rhoPresMuD),
        mR3CAST(sortedSphMarkersD->velMasD), mR3CAST(velMasD_old), mR3CAST(vel_vis), U1CAST(csrColInd),
        U1CAST(Contact_i), numAllMarkers, isErrorD);
    ChDeviceUtils::Sync_CheckError(isErrorH, isErrorD, "Update_Vel_XSPH-1");

    // consider boundary condition
    //============================================================================================================
    velMasD_old = sortedSphMarkersD->velMasD;
    rhoPresMuD_old = sortedSphMarkersD->rhoPresMuD;

    printf("numObjectsH->numFlexBodies1D=%d\n", numObjectsH->numFlexBodies1D);
    Boundary_Conditions<<<numBlocks, numThreads>>>(
        mR4CAST(sortedSphMarkersD->posRadD), mR4CAST(sortedSphMarkersD->rhoPresMuD), mR4CAST(rhoPresMuD_old),
        mR3CAST(sortedSphMarkersD->velMasD), mR3CAST(velMasD_old), mR3CAST(csrValGradient), R1CAST(csrValLaplacian),
        U1CAST(csrColInd), U1CAST(Contact_i),

        mR4CAST(otherFsiBodiesD->velMassRigid_fsiBodies_D), mR3CAST(otherFsiBodiesD->accRigid_fsiBodies_D),
        U1CAST(fsiGeneralData->rigidIdentifierD),

        mR3CAST(otherFsiMeshD->pos_fsi_fea_D), mR3CAST(otherFsiMeshD->vel_fsi_fea_D),
        mR3CAST(otherFsiMeshD->acc_fsi_fea_D), U1CAST(fsiGeneralData->FlexIdentifierD),

        numObjectsH->numFlexBodies1D, U2CAST(fsiGeneralData->CableElementsNodes),
        U4CAST(fsiGeneralData->ShellElementsNodes), updatePortion, U1CAST(markersProximityD->gridMarkerIndexD),
        numAllMarkers, isErrorD);
    ChDeviceUtils::Sync_CheckError(isErrorH, isErrorD, "Boundary_Conditions-1");
    //============================================================================================================
    velMasD_old = sortedSphMarkersD->velMasD;
    NS_RHS_Predictor<<<numBlocks, numThreads>>>(mR4CAST(sortedSphMarkersD->posRadD), mR4CAST(posRadD_old),
                                                mR4CAST(sortedSphMarkersD->rhoPresMuD), mR4CAST(rhoPresMuD_old),
                                                mR3CAST(sortedSphMarkersD->velMasD), mR3CAST(velMasD_old), mR3CAST(ft),
                                                mR3CAST(csrValGradient), R1CAST(csrValLaplacian), U1CAST(csrColInd),
                                                U1CAST(Contact_i), paramsH->dT, numAllMarkers, isErrorD);
    ChDeviceUtils::Sync_CheckError(isErrorH, isErrorD, "NS_RHS_Predictor-1");
    // 2nd call to calc f and v,x from predicted v and x
    CopySortedToOriginal_NonInvasive_R3(otherSphMarkersD->velMasD, sortedSphMarkersD->velMasD,
                                        markersProximityD->gridMarkerIndexD);
    CopySortedToOriginal_NonInvasive_R4(otherSphMarkersD->rhoPresMuD, sortedSphMarkersD->rhoPresMuD,
                                        markersProximityD->gridMarkerIndexD);
    CopySortedToOriginal_NonInvasive_R4(otherSphMarkersD->posRadD, sortedSphMarkersD->posRadD,
                                        markersProximityD->gridMarkerIndexD);

    fsiCollisionSystem->ArrangeData(otherSphMarkersD);
    //============================================================================================================
    calcRho_kernel<<<numBlocks, numThreads>>>(
        mR4CAST(sortedSphMarkersD->posRadD), mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(_sumWij_inv),
        U1CAST(markersProximityD->cellStartD), U1CAST(markersProximityD->cellEndD), U1CAST(Contact_i), numAllMarkers,
        isErrorD);
    ChDeviceUtils::Sync_CheckError(isErrorH, isErrorD, "calcRho_kernel-2");
    LastVal = Contact_i[numAllMarkers - 1];
    thrust::exclusive_scan(Contact_i.begin(), Contact_i.end(), Contact_i.begin());
    Contact_i.push_back(LastVal + Contact_i[numAllMarkers - 1]);
    NNZ = Contact_i[numAllMarkers];
    csrValGradient.resize(NNZ);
    csrValLaplacian.resize(NNZ);
    csrValFunciton.resize(NNZ);
    AMatrix.resize(NNZ);
    csrColInd.resize(NNZ);
    thrust::fill(csrValGradient.begin(), csrValGradient.end(), mR3(0.0));
    thrust::fill(csrValLaplacian.begin(), csrValLaplacian.end(), 0.0);
    thrust::fill(csrValFunciton.begin(), csrValFunciton.end(), 0.0);
    thrust::fill(csrColInd.begin(), csrColInd.end(), 0.0);
    //============================================================================================================
    calcNormalizedRho_Gi_fillInMatrixIndices<<<numBlocks, numThreads>>>(
        mR4CAST(sortedSphMarkersD->posRadD), mR3CAST(sortedSphMarkersD->velMasD),
        mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(_sumWij_inv), R1CAST(G_i), mR3CAST(Normals), U1CAST(csrColInd),
        U1CAST(Contact_i), U1CAST(markersProximityD->cellStartD), U1CAST(markersProximityD->cellEndD), numAllMarkers,
        isErrorD);
    ChDeviceUtils::Sync_CheckError(isErrorH, isErrorD, "calcNormalizedRho_Gi_fillInMatrixIndices-2");
    //============================================================================================================
    printf("calc_A_tensor+");
    calc_A_tensor<<<numBlocks, numThreads>>>(R1CAST(A_i), R1CAST(G_i), mR4CAST(sortedSphMarkersD->posRadD),
                                             mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(_sumWij_inv),
                                             U1CAST(markersProximityD->cellStartD), U1CAST(markersProximityD->cellEndD),
                                             numAllMarkers, isErrorD);
    ChDeviceUtils::Sync_CheckError(isErrorH, isErrorD, "calc_A_tensor-2");
    printf("calc_L_tensor+");
    calc_L_tensor<<<numBlocks, numThreads>>>(R1CAST(A_i), R1CAST(L_i), R1CAST(G_i), mR4CAST(sortedSphMarkersD->posRadD),
                                             mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(_sumWij_inv),
                                             U1CAST(markersProximityD->cellStartD), U1CAST(markersProximityD->cellEndD),
                                             numAllMarkers, isErrorD);
    ChDeviceUtils::Sync_CheckError(isErrorH, isErrorD, "calc_L_tensor-2");
    printf("Gradient_Laplacian_Operator: ");
    //============================================================================================================
    Function_Gradient_Laplacian_Operator<<<numBlocks, numThreads>>>(
        mR4CAST(sortedSphMarkersD->posRadD), mR3CAST(sortedSphMarkersD->velMasD),
        mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(_sumWij_inv), R1CAST(G_i), R1CAST(L_i), R1CAST(csrValLaplacian),
        mR3CAST(csrValGradient), R1CAST(csrValFunciton), U1CAST(csrColInd), U1CAST(Contact_i), numAllMarkers, isErrorD);
    ChDeviceUtils::Sync_CheckError(isErrorH, isErrorD, "Gradient_Laplacian_Operator-2");
    // //============================================================================================================
    // calculate_pressure<<<numBlocks, numThreads>>>(mR4CAST(sortedSphMarkersD->rhoPresMuD), numAllMarkers, isErrorD);
    // ChDeviceUtils::Sync_CheckError(isErrorH, isErrorD, "calculate_pressure-2");
    //============================================================================================================
    velMasD_old = sortedSphMarkersD->velMasD;
    Update_Vel_XSPH<<<numBlocks, numThreads>>>(
        mR4CAST(sortedSphMarkersD->posRadD), mR4CAST(sortedSphMarkersD->rhoPresMuD),
        mR3CAST(sortedSphMarkersD->velMasD), mR3CAST(velMasD_old), mR3CAST(vel_vis), U1CAST(csrColInd),
        U1CAST(Contact_i), numAllMarkers, isErrorD);
    ChDeviceUtils::Sync_CheckError(isErrorH, isErrorD, "Update_Vel_XSPH-2");
    // consider boundary condition
    //============================================================================================================
    velMasD_old = sortedSphMarkersD->velMasD;
    rhoPresMuD_old = sortedSphMarkersD->rhoPresMuD;
    Boundary_Conditions<<<numBlocks, numThreads>>>(
        mR4CAST(sortedSphMarkersD->posRadD), mR4CAST(sortedSphMarkersD->rhoPresMuD), mR4CAST(rhoPresMuD_old),
        mR3CAST(sortedSphMarkersD->velMasD), mR3CAST(velMasD_old), mR3CAST(csrValGradient), R1CAST(csrValLaplacian),
        U1CAST(csrColInd), U1CAST(Contact_i),

        mR4CAST(otherFsiBodiesD->velMassRigid_fsiBodies_D), mR3CAST(otherFsiBodiesD->accRigid_fsiBodies_D),
        U1CAST(fsiGeneralData->rigidIdentifierD),

        mR3CAST(otherFsiMeshD->pos_fsi_fea_D), mR3CAST(otherFsiMeshD->vel_fsi_fea_D),
        mR3CAST(otherFsiMeshD->acc_fsi_fea_D), U1CAST(fsiGeneralData->FlexIdentifierD),

        numObjectsH->numFlexBodies1D, U2CAST(fsiGeneralData->CableElementsNodes),
        U4CAST(fsiGeneralData->ShellElementsNodes), updatePortion, U1CAST(markersProximityD->gridMarkerIndexD),
        numAllMarkers, isErrorD);
    ChDeviceUtils::Sync_CheckError(isErrorH, isErrorD, "Boundary_Conditions-2");
    //============================================================================================================
    rhoPresMuD_old = sortedSphMarkersD->rhoPresMuD;
    posRadD_old = sortedSphMarkersD->posRadD;
    velMasD_old = sortedSphMarkersD->velMasD;
    NS_RHS_Predictor<<<numBlocks, numThreads>>>(mR4CAST(sortedSphMarkersD->posRadD), mR4CAST(posRadD_old),
                                                mR4CAST(sortedSphMarkersD->rhoPresMuD), mR4CAST(rhoPresMuD_old),
                                                mR3CAST(sortedSphMarkersD->velMasD), mR3CAST(velMasD_old), mR3CAST(ft),
                                                mR3CAST(csrValGradient), R1CAST(csrValLaplacian), U1CAST(csrColInd),
                                                U1CAST(Contact_i), paramsH->dT, numAllMarkers, isErrorD);
    ChDeviceUtils::Sync_CheckError(isErrorH, isErrorD, "NS_RHS_Predictor-2");
    //============================================================================================================
    thrust::device_vector<Real3> ft_unsorted(numAllMarkers);
    thrust::fill(ft_unsorted.begin(), ft_unsorted.end(), mR3(0.0));

    CopySortedToOriginal_NonInvasive_R3(ft_unsorted, ft, markersProximityD->gridMarkerIndexD);
    // otherSphMarkersD->velMasD = SphMarkerDataD1.velMasD;
    // otherSphMarkersD->posRadD = SphMarkerDataD1.posRadD;
    //    otherSphMarkersD->rhoPresMuD = SphMarkerDataD1.rhoPresMuD;
    // CopySortedToOriginal_NonInvasive_R3(otherSphMarkersD->velMasD, sortedSphMarkersD->velMasD,
    //                                     markersProximityD->gridMarkerIndexD);
    CopySortedToOriginal_NonInvasive_R4(otherSphMarkersD->rhoPresMuD, sortedSphMarkersD->rhoPresMuD,
                                        markersProximityD->gridMarkerIndexD);
    //============================================================================================================
    Update<<<numBlocks, numThreads>>>(mR4CAST(SphMarkerDataD1.posRadD), mR4CAST(SphMarkerDataD1.rhoPresMuD),
                                      mR3CAST(SphMarkerDataD1.velMasD), mR3CAST(otherSphMarkersD->velMasD),
                                      mR4CAST(otherSphMarkersD->posRadD), mR3CAST(ft_unsorted), paramsH->dT,
                                      numAllMarkers, isErrorD);
    ChDeviceUtils::Sync_CheckError(isErrorH, isErrorD, "Update");

    iter = thrust::max_element(otherSphMarkersD->velMasD.begin(), otherSphMarkersD->velMasD.end(), compare_Real3_mag());
    MaxVel = length(*iter);

    fsiCollisionSystem->ArrangeData(otherSphMarkersD);
    // 3rd call
    //============================================================================================================
    calcRho_kernel<<<numBlocks, numThreads>>>(
        mR4CAST(sortedSphMarkersD->posRadD), mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(_sumWij_inv),
        U1CAST(markersProximityD->cellStartD), U1CAST(markersProximityD->cellEndD), U1CAST(Contact_i), numAllMarkers,
        isErrorD);
    ChDeviceUtils::Sync_CheckError(isErrorH, isErrorD, "calcRho_kernel-3");
    LastVal = Contact_i[numAllMarkers - 1];
    thrust::exclusive_scan(Contact_i.begin(), Contact_i.end(), Contact_i.begin());
    Contact_i.push_back(LastVal + Contact_i[numAllMarkers - 1]);
    NNZ = Contact_i[numAllMarkers];
    csrValGradient.resize(NNZ);
    csrValLaplacian.resize(NNZ);
    csrValFunciton.resize(NNZ);
    AMatrix.resize(NNZ);
    csrColInd.resize(NNZ);
    thrust::fill(csrValGradient.begin(), csrValGradient.end(), mR3(0.0));
    thrust::fill(csrValLaplacian.begin(), csrValLaplacian.end(), 0.0);
    thrust::fill(csrValFunciton.begin(), csrValFunciton.end(), 0.0);
    thrust::fill(csrColInd.begin(), csrColInd.end(), 0.0);
    //============================================================================================================
    calcNormalizedRho_Gi_fillInMatrixIndices<<<numBlocks, numThreads>>>(
        mR4CAST(sortedSphMarkersD->posRadD), mR3CAST(sortedSphMarkersD->velMasD),
        mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(_sumWij_inv), R1CAST(G_i), mR3CAST(Normals), U1CAST(csrColInd),
        U1CAST(Contact_i), U1CAST(markersProximityD->cellStartD), U1CAST(markersProximityD->cellEndD), numAllMarkers,
        isErrorD);
    ChDeviceUtils::Sync_CheckError(isErrorH, isErrorD, "calcNormalizedRho_Gi_fillInMatrixIndices-3");
    printf("calc_A_tensor+");
    //============================================================================================================
    calc_A_tensor<<<numBlocks, numThreads>>>(R1CAST(A_i), R1CAST(G_i), mR4CAST(sortedSphMarkersD->posRadD),
                                             mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(_sumWij_inv),
                                             U1CAST(markersProximityD->cellStartD), U1CAST(markersProximityD->cellEndD),
                                             numAllMarkers, isErrorD);
    ChDeviceUtils::Sync_CheckError(isErrorH, isErrorD, "calc_A_tensor-3");
    //============================================================================================================
    calc_L_tensor<<<numBlocks, numThreads>>>(R1CAST(A_i), R1CAST(L_i), R1CAST(G_i), mR4CAST(sortedSphMarkersD->posRadD),
                                             mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(_sumWij_inv),
                                             U1CAST(markersProximityD->cellStartD), U1CAST(markersProximityD->cellEndD),
                                             numAllMarkers, isErrorD);
    ChDeviceUtils::Sync_CheckError(isErrorH, isErrorD, "calc_L_tensor-3");
    printf("Gradient_Laplacian_Operator: ");
    //============================================================================================================
    Function_Gradient_Laplacian_Operator<<<numBlocks, numThreads>>>(
        mR4CAST(sortedSphMarkersD->posRadD), mR3CAST(sortedSphMarkersD->velMasD),
        mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(_sumWij_inv), R1CAST(G_i), R1CAST(L_i), R1CAST(csrValLaplacian),
        mR3CAST(csrValGradient), R1CAST(csrValFunciton), U1CAST(csrColInd), U1CAST(Contact_i), numAllMarkers, isErrorD);
    ChDeviceUtils::Sync_CheckError(isErrorH, isErrorD, "Gradient_Laplacian_Operator-3");

    //============================================================================================================
    rhoPresMuD_old = sortedSphMarkersD->rhoPresMuD;
    posRadD_old = sortedSphMarkersD->posRadD;
    velMasD_old = sortedSphMarkersD->velMasD;
    Shifting_r<<<numBlocks, numThreads>>>(mR4CAST(sortedSphMarkersD->posRadD), mR4CAST(posRadD_old),
                                          mR4CAST(sortedSphMarkersD->rhoPresMuD), mR4CAST(rhoPresMuD_old),
                                          mR3CAST(sortedSphMarkersD->velMasD), mR3CAST(velMasD_old), mR3CAST(vel_vis),
                                          R1CAST(csrValFunciton), mR3CAST(csrValGradient), U1CAST(csrColInd),
                                          U1CAST(Contact_i), MaxVel, paramsH->dT, numAllMarkers, isErrorD);
    ChDeviceUtils::Sync_CheckError(isErrorH, isErrorD, "Shifting_r");

    Real4_x unary_op(paramsH->rho0);
    thrust::plus<Real> binary_op;
    Real Ave_density = thrust::transform_reduce(sortedSphMarkersD->rhoPresMuD.begin(),
                                                sortedSphMarkersD->rhoPresMuD.end(), unary_op, 0.0, binary_op) /
                       (numObjectsH->numFluidMarkers * paramsH->rho0);
    Real Re = paramsH->L_Characteristic * paramsH->rho0 * MaxVel / paramsH->mu0;
    printf("Ave_density_Err=%.3e, Re=%.1f\n", Ave_density, Re);

    CopySortedToOriginal_NonInvasive_R3(fsiGeneralData->vis_vel_SPH_D, vel_vis, markersProximityD->gridMarkerIndexD);
    CopySortedToOriginal_NonInvasive_R3(otherSphMarkersD->velMasD, sortedSphMarkersD->velMasD,
                                        markersProximityD->gridMarkerIndexD);
    CopySortedToOriginal_NonInvasive_R4(otherSphMarkersD->rhoPresMuD, sortedSphMarkersD->rhoPresMuD,
                                        markersProximityD->gridMarkerIndexD);
    CopySortedToOriginal_NonInvasive_R4(otherSphMarkersD->posRadD, sortedSphMarkersD->posRadD,
                                        markersProximityD->gridMarkerIndexD);
    //    CopySortedToOriginal_NonInvasive_R3(fsiGeneralData->vel_IISPH_D, sortedSphMarkersD->velMasD,
    //                                        markersProximityD->gridMarkerIndexD);
    //    CopySortedToOriginal_NonInvasive_R3(otherSphMarkersD->velMasD, sortedSphMarkersD->velMasD,
    //                                        markersProximityD->gridMarkerIndexD);

    //    CopySortedToOriginal_NonInvasive_R4(otherSphMarkersD->posRadD, sortedSphMarkersD->posRadD,
    //                                        markersProximityD->gridMarkerIndexD);
    //============================================================================================================

    _sumWij_inv.clear();
    Contact_i.clear();
    AMatrix.clear();
    csrColInd.clear();
    G_i.clear();
    A_i.clear();
    L_i.clear();
    csrValLaplacian.clear();
    csrValGradient.clear();
    ft_unsorted.clear();
    ft.clear();

}  // namespace fsi

}  // namespace fsi
}  // namespace chrono
