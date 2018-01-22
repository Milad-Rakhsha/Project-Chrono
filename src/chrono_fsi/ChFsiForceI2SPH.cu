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
//--------------------------------------------------------------------------------------------------------------------------------
__global__ void V_star_Predictor(Real4* sortedPosRad,  // input: sorted positions
                                 Real3* sortedVelMas,
                                 Real4* sortedRhoPreMu,
                                 Real* A_Matrix,
                                 Real3* Bi,
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
                                 Real delta_t,

                                 volatile bool* isErrorD) {
    uint i_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_idx >= numAllMarkers) {
        return;
    }
    Real CN = 0.5;

    uint csrStartIdx = numContacts[i_idx];
    uint csrEndIdx = numContacts[i_idx + 1];

    bool Fluid_Marker = sortedRhoPreMu[i_idx].w == -1.0;
    bool Boundary_Marker = sortedRhoPreMu[i_idx].w > -1.0;

    if (sortedRhoPreMu[i_idx].w <= -2) {
        A_Matrix[csrStartIdx] = 1;
        Bi[i_idx] = mR3(0.0);
        return;
    }

    Real rho0 = paramsD.rho0;
    Real rhoi = sortedRhoPreMu[i_idx].x;
    Real3 grad_rho_i = mR3(0.0);

    for (int count = csrStartIdx; count < csrEndIdx; count++) {
        int j = csrColInd[count];
        grad_rho_i += A_G[count] * sortedRhoPreMu[j].x;
    }
    if (Fluid_Marker) {
        //======================== Interior ===========================
        // Navier-Stokes
        Real3 rhs = mR3(0.0);
        for (int count = csrStartIdx; count < csrEndIdx; count++) {
            int j = csrColInd[count];
            A_Matrix[count] = -CN * paramsD.mu0 / rhoi * A_L[count];
            rhs += -1 / rhoi * A_G[count] * sortedRhoPreMu[j].y *
                       !paramsD.USE_NonIncrementalProjection +                    // Pressure Gradient
                   (1 - CN) * paramsD.mu0 / rhoi * A_L[count] * sortedVelMas[j];  // viscous term;
        }
        A_Matrix[csrStartIdx] += 1 / delta_t;
        Bi[i_idx] = rhs + sortedVelMas[i_idx] / delta_t      //forward euler term from lhs
                    + paramsD.gravity + paramsD.bodyForce3;  // body force

        //======================== Inflow/outflow =====================
        if (paramsD.ApplyInFlowOutFlow) {
            //            if (sortedPosRad[i_idx].x >= paramsD.outflow.x &&
            //                sortedPosRad[i_idx].x <= paramsD.outflow.x + paramsD.HSML * 5) {
            //                clearRow3(i_idx, csrStartIdx, csrEndIdx, A_Matrix, Bi);
            //                for (int count = csrStartIdx; count < csrEndIdx; count++) {
            //                    uint j = csrColInd[count];
            //                    A_Matrix[count] = -dot(A_G[count], mR3(1.0, 0, 0));
            //                    A_Matrix[csrStartIdx] += +dot(A_G[count], mR3(1.0, 0, 0));
            //                }
            //                Bi[i_idx] = mR3(0.0);
            //                if (abs(A_Matrix[csrStartIdx]) < EPSILON) {
            //                    clearRow3(i_idx, csrStartIdx, csrEndIdx, A_Matrix, Bi);
            //                    A_Matrix[csrStartIdx] = 1.0;
            //                }
            //                // Inlet
            //            } else
            if (sortedPosRad[i_idx].x < paramsD.inflow.x &&
                sortedPosRad[i_idx].x > paramsD.inflow.x - paramsD.HSML * 5) {
                clearRow3(i_idx, csrStartIdx, csrEndIdx, A_Matrix, Bi);
                //                for (int count = csrStartIdx; count < csrEndIdx; count++) {
                //                    uint j = csrColInd[count];
                //                    A_Matrix[count] = A_f[count];
                //            }
                A_Matrix[csrStartIdx] = 1.0;
                Bi[i_idx] = paramsD.V_in;
            }
        }

    } else if (Boundary_Marker) {
        //======================== Boundary ===========================
        Real h_i = sortedPosRad[i_idx].w;
        Real3 posRadA = mR3(sortedPosRad[i_idx]);
        Real den = 0.0;
        Real3 gradp = mR3(0);

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
            gradp += A_G[count] * sortedPosRad[j].y;
        }

        Real3 myAcc = mR3(0);
        Real3 V_prescribed = mR3(0);

        BCE_Vel_Acc(i_idx, myAcc, V_prescribed, sortedPosRad, updatePortion, gridMarkerIndexD, velMassRigid_fsiBodies_D,
                    accRigid_fsiBodies_D, rigidIdentifierD, pos_fsi_fea_D, vel_fsi_fea_D, acc_fsi_fea_D,
                    FlexIdentifierD, numFlex1D, CableElementsNodes, ShellelementsNodes);

        if (abs(den) < EPSILON) {
            A_Matrix[csrStartIdx] = 1.0;
            Bi[i_idx] = V_prescribed;
        } else {
            A_Matrix[csrStartIdx] = den;
            Bi[i_idx] = 2 * V_prescribed * den +
                        gradp / sortedRhoPreMu[i_idx].x * delta_t * paramsD.USE_NonIncrementalProjection;
        }
    }

    v_old[i_idx] = sortedVelMas[i_idx];

    if (abs(A_Matrix[csrStartIdx]) < EPSILON)
        printf("V_star_Predictor %d A_Matrix[csrStartIdx]= %f, type=%f \n", i_idx, A_Matrix[csrStartIdx],
               sortedRhoPreMu[i_idx].w);
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
                                  Real delta_t,
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

    Real rho0 = paramsD.rho0;

    Real3 gravity = paramsD.gravity + paramsD.bodyForce3;
    Real3 grad_rho_i = mR3(0.0);
    Real div_vi_star = 0;

    // Calculating the div.v* and grad(rho)
    for (int count = csrStartIdx; count < csrEndIdx; count++) {
        int j = csrColInd[count];
        div_vi_star += dot(A_G[count], Vstar[j]);
        grad_rho_i += A_G[count] * sortedRhoPreMu[j].x;
    }

    Real rhoi = sortedRhoPreMu[i_idx].x;
    Real rhoi_star = rhoi - rhoi * div_vi_star * delta_t;
    //    Real rhoi_star = rhoi + dot(grad_rho_i, Vstar[i_idx] * delta_t);

    //======================== Interior ===========================
    if (Fluid_Marker) {
        for (int count = csrStartIdx; count < csrEndIdx; count++) {
            A_Matrix[count] = 1 / rhoi * A_L[count] - 1.0 / (rhoi * rhoi) * dot(grad_rho_i, A_G[count]);
        }

        Bi[i_idx] = div_vi_star / delta_t;

        //======================== Free-Surface =====================
        //        if (sortedRhoPreMu[i_idx].x < 0.95 * paramsD.rho0) {
        //            clearRow(i_idx, csrStartIdx, csrEndIdx, A_Matrix, Bi);
        //            //            for (int count = csrStartIdx; count < csrEndIdx; count++) {
        //            //                uint j = csrColInd[count];
        //            //                A_Matrix[count] = A_f[count];
        //            //            }
        //            A_Matrix[csrStartIdx] += 1.0;
        //            Bi[i_idx] = -sortedRhoPreMu[i_idx].y * !paramsD.USE_NonIncrementalProjection;
        //        }

        //======================== Inflow/outflow =====================
        if (paramsD.ApplyInFlowOutFlow) {
            //            if (sortedPosRad[i_idx].x <= paramsD.inflow.x &&
            //                sortedPosRad[i_idx].x > paramsD.inflow.x - paramsD.HSML * 5) {
            //                //                clearRow(i_idx, csrStartIdx, csrEndIdx, A_Matrix, Bi);
            //                //                for (int count = csrStartIdx; count < csrEndIdx; count++) {
            //                //                    uint j = csrColInd[count];
            //                //                    A_Matrix[count] = -dot(A_G[count], mR3(1, 0, 0));
            //                //                    A_Matrix[csrStartIdx] += +dot(A_G[count], mR3(1, 0, 0));
            //                //                }
            //                //                Bi[i_idx] = 0;
            //
            //                //                A_Matrix[csrStartIdx] = 1.0;
            //                //                Bi[i_idx] += 0;
            //
            //            } else

            if (sortedPosRad[i_idx].x > paramsD.outflow.x &&
                sortedPosRad[i_idx].x < paramsD.outflow.x + paramsD.HSML * 5) {
                clearRow(i_idx, csrStartIdx, csrEndIdx, A_Matrix, Bi);
                //                for (int count = csrStartIdx; count < csrEndIdx; count++) {
                //                    uint j = csrColInd[count];
                //                    A_Matrix[count] = A_f[count];
                //                }
                A_Matrix[csrStartIdx] = 1.0;
                Bi[i_idx] = 0.0;
            }
        }
        //======================= Boundary ===========================
    } else if (Boundary_Marker && paramsD.bceType != ADAMI) {
        Real3 my_normal = Normals[i_idx];
        bool haveFluid = false;
        for (int count = csrStartIdx; count < csrEndIdx; count++) {
            uint j = csrColInd[count];
            if (sortedRhoPreMu[j].w == -1.0 || dot(my_normal, mR3(sortedPosRad[i_idx] - sortedPosRad[j])) > 0) {
                A_Matrix[count] = -dot(A_G[count], my_normal);
                A_Matrix[csrStartIdx] += +dot(A_G[count], my_normal);
            }
        }
        Bi[i_idx] = 0;
        if (abs(A_Matrix[csrStartIdx]) < 1e-6) {
            clearRow(i_idx, csrStartIdx, csrEndIdx, A_Matrix, Bi);
            for (int count = csrStartIdx; count < csrEndIdx; count++)
                A_Matrix[count] = A_f[count];
            A_Matrix[csrStartIdx] -= 1.0;
            Bi[i_idx] = 0.0;
        }

        //======================= Boundary Adami===========================
    } else if ((Boundary_Marker) && paramsD.bceType == ADAMI && paramsD.USE_NonIncrementalProjection) {
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

        for (int count = csrStartIdx; count < csrEndIdx; count++) {
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
            Bi[i_idx] = paramsD.BASEPRES;
        }
    }

    if (paramsD.USE_NonIncrementalProjection)
        q_old[i_idx] = sortedRhoPreMu[i_idx].y;

    if (abs(A_Matrix[csrStartIdx]) < EPSILON)
        printf("Pressure_Equation %d A_Matrix[csrStartIdx]= %f, type=%f \n", i_idx, A_Matrix[csrStartIdx],
               sortedRhoPreMu[i_idx].w);
}
//--------------------------------------------------------------------------------------------------------------------------------
__global__ void Velocity_Correction_and_update(Real4* sortedPosRad,
                                               Real4* sortedRhoPreMu,
                                               Real3* sortedVelMas,
                                               Real3* sortedVisVel,
                                               Real3* Vstar,
                                               Real* q_i,  // q=p^(n+1)-p^n
                                               const Real* A_f,
                                               const Real3* A_G,
                                               const uint* csrColInd,
                                               const uint* numContacts,
                                               int numAllMarkers,
                                               const Real MaxVel,
                                               Real delta_t,
                                               volatile bool* isErrorD) {
    uint i_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_idx >= numAllMarkers) {
        return;
    }

    uint csrStartIdx = numContacts[i_idx];
    uint csrEndIdx = numContacts[i_idx + 1];
    Real3 grad_q_i = mR3(0.0);
    Real3 grad_p_nPlus1 = mR3(0.0);
    Real divV_star = 0;
    Real3 inner_sum = mR3(0.0), shift_r = mR3(0.0);
    Real mi_bar = 0.0, r0 = 0.0;
    for (int count = csrStartIdx; count < csrEndIdx; count++) {
        uint j = csrColInd[count];
        grad_q_i += A_G[count] * q_i[j];
        divV_star += dot(A_G[count], Vstar[j]);
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

    if (sortedRhoPreMu[i_idx].w == -1.0) {
        r0 /= (csrEndIdx - csrStartIdx);
        mi_bar /= (csrEndIdx - csrStartIdx);
    }

    shift_r = paramsD.beta_shifting * r0 * r0 * length(MaxVel) * delta_t * inner_sum / mi_bar;

    Real3 V_new = Vstar[i_idx] - delta_t / sortedRhoPreMu[i_idx].x * grad_q_i;

    Real4 x_new = sortedPosRad[i_idx] + mR4(delta_t / 2 * (V_new + sortedVelMas[i_idx]), 0.0);

    sortedVelMas[i_idx] = V_new;

    if (paramsD.USE_NonIncrementalProjection)
        sortedRhoPreMu[i_idx].y += q_i[i_idx];
    else
        sortedRhoPreMu[i_idx].y += q_i[i_idx] + dot(grad_p_nPlus1, mR3(x_new - sortedPosRad[i_idx]));

    if (sortedRhoPreMu[i_idx].w == -1.0) {
        sortedPosRad[i_idx] = x_new;
    }
    //
    //    Real3 grad_p = mR3(0.0);
    //    Real3 grad_rho = mR3(0.0);
    //    Real3 grad_ux = mR3(0.0);
    //    Real3 grad_uy = mR3(0.0);
    //    Real3 grad_uz = mR3(0.0);
    //
    //    for (int count = csrStartIdx; count < csrEndIdx; count++) {
    //        uint j = csrColInd[count];
    //        grad_p += A_G[count] * sortedRhoPreMu[i_idx].y;
    //        grad_rho += A_G[count] * sortedRhoPreMu[i_idx].x;
    //        grad_ux += A_G[count] * sortedVelMas[i_idx].x;
    //        grad_uy += A_G[count] * sortedVelMas[i_idx].y;
    //        grad_uz += A_G[count] * sortedVelMas[i_idx].z;
    //    }
    //
    //    if (sortedRhoPreMu[i_idx].w == -1.0) {
    //        sortedPosRad[i_idx] += mR4(shift_r, 0.0);
    //        sortedRhoPreMu[i_idx].y += dot(shift_r, grad_p);
    //        sortedRhoPreMu[i_idx].x += dot(shift_r, grad_rho);
    //        sortedVelMas[i_idx].x += dot(shift_r, grad_ux);
    //        sortedVelMas[i_idx].y += dot(shift_r, grad_uy);
    //        sortedVelMas[i_idx].z += dot(shift_r, grad_uz);
    //    }

    //    Real3 vis_vel = mR3(0.0);
    //
    //    for (int count = csrStartIdx; count < csrEndIdx; count++) {
    //        uint j = csrColInd[count];
    //        vis_vel += A_f[count] * (sortedVelMas[j]);
    //    }

    //    sortedVisVel[i_idx] = vis_vel;
    //    sortedVelMas[i_idx] = paramsD.EPS_XSPH * vis_vel + (1 - paramsD.EPS_XSPH) * sortedVelMas[i_idx];
}
//--------------------------------------------------------------------------------------------------------------------------------
__global__ void Shifting(Real4* sortedPosRad,
                         Real4* sortedRhoPreMu,
                         Real3* sortedVelMas,
                         Real3* sortedVisVel,
                         const Real* A_f,
                         const Real3* A_G,
                         const uint* csrColInd,
                         const uint* numContacts,
                         int numAllMarkers,
                         const Real MaxVel,
                         Real delta_t,
                         volatile bool* isErrorD) {
    uint i_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_idx >= numAllMarkers) {
        return;
    }

    uint csrStartIdx = numContacts[i_idx];
    uint csrEndIdx = numContacts[i_idx + 1];
    Real3 inner_sum = mR3(0.0), shift_r = mR3(0.0);
    Real mi_bar = 0.0, r0 = 0.0, v_bar = 0.0;

    for (int count = csrStartIdx; count < csrEndIdx; count++) {
        uint j = csrColInd[count];
        Real3 rij = Distance(mR3(sortedPosRad[i_idx]), mR3(sortedPosRad[j]));
        Real d = length(rij);
        if (count == csrStartIdx)
            continue;

        v_bar += length(A_f[count] * (sortedVelMas[j]));
        Real m_j = pow(sortedPosRad[j].w * paramsD.MULT_INITSPACE, 3) * paramsD.rho0;
        mi_bar += m_j;
        r0 += d;
        //        inner_sum += rij / (d * d * d);
        inner_sum += m_j * rij / (d * d * d);
    }

    if (sortedRhoPreMu[i_idx].w == -1.0) {
        r0 /= (csrEndIdx - csrStartIdx);
        mi_bar /= (csrEndIdx - csrStartIdx);
    }

    shift_r = paramsD.beta_shifting * r0 * r0 * length(MaxVel) * delta_t * inner_sum / mi_bar;
    //    shift_r = paramsD.beta_shifting * r0 * r0 * length(MaxVel) * delta_t * inner_sum;

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

    if (sortedRhoPreMu[i_idx].w == -1.0) {
        sortedPosRad[i_idx] += mR4(shift_r, 0.0);
        sortedRhoPreMu[i_idx].y += dot(shift_r, grad_p);
        sortedRhoPreMu[i_idx].x += dot(shift_r, grad_rho);
        sortedVelMas[i_idx].x += dot(shift_r, grad_ux);
        sortedVelMas[i_idx].y += dot(shift_r, grad_uy);
        sortedVelMas[i_idx].z += dot(shift_r, grad_uz);
    }

    Real3 vis_vel = mR3(0.0);

    for (int count = csrStartIdx; count < csrEndIdx; count++) {
        uint j = csrColInd[count];
        if (sortedRhoPreMu[j].w == -1.0)
            vis_vel += A_f[count] * (sortedVelMas[j]);
    }

    sortedVisVel[i_idx] = vis_vel;
    sortedVelMas[i_idx] = paramsD.EPS_XSPH * vis_vel + (1 - paramsD.EPS_XSPH) * sortedVelMas[i_idx];
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
    isErrorH = (bool*)malloc(sizeof(bool));
    cudaMalloc((void**)&isErrorD, sizeof(bool));
    cudaMalloc((void**)&isErrorD2, sizeof(bool));
    *isErrorH = false;
    cudaMemcpy(isErrorD, isErrorH, sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(isErrorD2, isErrorH, sizeof(bool), cudaMemcpyHostToDevice);
}

void ChFsiForceI2SPH::PreProcessor(SphMarkerDataD* sortedSphMarkersD, bool print, bool calcLaplacianOperator) {
    numAllMarkers = numObjectsH->numAllMarkers;
    Contact_i.resize(numAllMarkers);
    uint numThreads, numBlocks;
    computeGridSize(numAllMarkers, 128, numBlocks, numThreads);
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

    calcNormalizedRho_Gi_fillInMatrixIndices<<<numBlocks, numThreads>>>(
        mR4CAST(sortedSphMarkersD->posRadD), mR3CAST(sortedSphMarkersD->velMasD),
        mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(_sumWij_inv), R1CAST(G_i), mR3CAST(Normals), U1CAST(csrColInd),
        U1CAST(Contact_i), U1CAST(markersProximityD->cellStartD), U1CAST(markersProximityD->cellEndD), numAllMarkers,
        isErrorD);
    ChDeviceUtils::Sync_CheckError(isErrorH, isErrorD, "calcNormalizedRho_Gi_fillInMatrixIndices");

    //============================================================================================================
    double A_L_Tensor_GradLaplacian = clock();
    if (print)
        printf(" calc_A_tensor+");

    if (calcLaplacianOperator) {
        calc_A_tensor<<<numBlocks, numThreads>>>(R1CAST(A_i), R1CAST(G_i), mR4CAST(sortedSphMarkersD->posRadD),
                                                 mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(_sumWij_inv),
                                                 U1CAST(csrColInd), U1CAST(Contact_i), numAllMarkers, isErrorD);
        ChDeviceUtils::Sync_CheckError(isErrorH, isErrorD, "calc_A_tensor");
        if (print)
            printf("calc_L_tensor+");
        calc_L_tensor<<<numBlocks, numThreads>>>(R1CAST(A_i), R1CAST(L_i), R1CAST(G_i),
                                                 mR4CAST(sortedSphMarkersD->posRadD),
                                                 mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(_sumWij_inv),
                                                 U1CAST(csrColInd), U1CAST(Contact_i), numAllMarkers, isErrorD);
        ChDeviceUtils::Sync_CheckError(isErrorH, isErrorD, "calc_L_tensor");
    }
    if (print)
        printf("Gradient_Laplacian_Operator: ");
    Function_Gradient_Laplacian_Operator<<<numBlocks, numThreads>>>(
        mR4CAST(sortedSphMarkersD->posRadD), mR3CAST(sortedSphMarkersD->velMasD),
        mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(_sumWij_inv), R1CAST(G_i), R1CAST(L_i), R1CAST(csrValLaplacian),
        mR3CAST(csrValGradient), R1CAST(csrValFunciton), U1CAST(csrColInd), U1CAST(Contact_i), numAllMarkers, isErrorD);
    ChDeviceUtils::Sync_CheckError(isErrorH, isErrorD, "Gradient_Laplacian_Operator");
    double Gradient_Laplacian_Operator = (clock() - A_L_Tensor_GradLaplacian) / (double)CLOCKS_PER_SEC;
    if (print)
        printf("%f (s)\n", Gradient_Laplacian_Operator);
}

//==========================================================================================================================================
//==========================================================================================================================================
//==========================================================================================================================================
void ChFsiForceI2SPH::ForceImplicitSPH(SphMarkerDataD* otherSphMarkersD,
                                       FsiBodiesDataD* otherFsiBodiesD,
                                       FsiMeshDataD* otherFsiMeshD) {
    CopyParams_NumberOfObjects(paramsH, numObjectsH);

    sphMarkersD = otherSphMarkersD;
    fsiCollisionSystem->ArrangeData(sphMarkersD);

    thrust::device_vector<Real3>::iterator iter =
        thrust::max_element(sortedSphMarkersD->velMasD.begin(), sortedSphMarkersD->velMasD.end(), compare_Real3_mag());
    Real MaxVel = length(*iter);

    if (paramsH->Adaptive_time_stepping) {
        Real dt_CFL = paramsH->Co_number * paramsH->HSML / MaxVel;
        Real dt_nu = 0.125 * paramsH->HSML * paramsH->HSML / (paramsH->mu0 / paramsH->rho0);
        Real dt_body = 0.125 * std::sqrt(paramsH->HSML / length(paramsH->bodyForce3 + paramsH->gravity));
        Real dt = std::min(dt_body, std::min(dt_CFL, dt_nu));
        if (dt / paramsH->dT_Max > 0.7 && dt / paramsH->dT_Max < 1)
            paramsH->dT = paramsH->dT_Max * 0.5;
        else
            paramsH->dT = std::min(dt, paramsH->dT_Max);

        CopyParams_NumberOfObjects(paramsH, numObjectsH);

        printf(" time step=%.3e, dt_Max=%.3e, dt_CFL=%.3e (CFL=%.2g), dt_nu=%.3e, dt_body=%.3e\n", paramsH->dT,
               paramsH->dT_Max, dt_CFL, paramsH->Co_number, dt_nu, dt_body);
    }
    int numHelperMarkers = numObjectsH->numHelperMarkers;
    int numFlexbodies = +numObjectsH->numFlexBodies1D + numObjectsH->numFlexBodies2D;
    int haveGhost = (numObjectsH->numGhostMarkers > 0) ? 1 : 0;
    int haveHelper = (numObjectsH->numHelperMarkers > 0) ? 1 : 0;
    int4 updatePortion =
        mI4(fsiGeneralData->referenceArray[haveGhost + haveHelper + 0].y,  // end of fluid
            fsiGeneralData->referenceArray[haveGhost + haveHelper + 1].y,  // end of boundary
            fsiGeneralData->referenceArray[haveGhost + haveHelper + 1 + numObjectsH->numRigidBodies].y,
            fsiGeneralData->referenceArray[haveGhost + haveHelper + 1 + numObjectsH->numRigidBodies + numFlexbodies].y);
    printf("ForceI2SPH numAllMarkers:%d,numHelperMarkers=%d\n", numAllMarkers, numHelperMarkers);

    //=====calcRho_kernel=== calc_A_tensor==calc_L_tensor==Function_Gradient_Laplacian_Operator=================
    ChFsiForceI2SPH::PreProcessor(sortedSphMarkersD);
    //    return;

    //==========================================================================================================
    uint numThreads, numBlocks;
    computeGridSize(numAllMarkers, 256, numBlocks, numThreads);

    thrust::fill(V_star_old.begin(), V_star_old.end(), mR3(0.0));
    thrust::fill(V_star_new.begin(), V_star_new.end(), mR3(0.0));
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
        numAllMarkers, paramsH->dT, isErrorD);
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
    //
    //    thrust::device_vector<Real3>::iterator iter =
    //        thrust::max_element(V_star_new.begin(), V_star_new.end(), compare_Real3_mag());
    //    unsigned int position = iter - V_star_new.begin();
    //    Real MaxVel = length(*iter);
    //
    uint FixedMarker = 0;
    //    Real temp = 0;
    //    for (int i = 0; i < numAllMarkers; i++) {
    //        if (Real4(sortedSphMarkersD->rhoPresMuD[i]).w == -1 && Real4(sortedSphMarkersD->posRadD[i]).x > temp)
    //        {
    //            FixedMarker = i;
    //            //            break;
    //        }
    //    }
    //    printf("Fixed marker %d\n", FixedMarker);

    double V_star_Predictor = (clock() - LinearSystemClock_V) / (double)CLOCKS_PER_SEC;
    printf(" V_star_Predictor Equation: %f (sec) - Final Residual=%.3e - #Iter=%d\n", V_star_Predictor, MaxRes,
           Iteration);
    //==================================================Pressure_Equation==============================================
    Iteration = 0;
    MaxRes = 100;
    double LinearSystemClock_p = clock();
    thrust::fill(AMatrix.begin(), AMatrix.end(), 0.0);
    thrust::fill(b1Vector.begin(), b1Vector.end(), 0.0);
    thrust::fill(q_old.begin(), q_old.end(), 0.0);
    thrust::fill(q_new.begin(), q_new.end(), 0.0);

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
        numAllMarkers, FixedMarker, paramsH->dT, isErrorD);
    ChDeviceUtils::Sync_CheckError(isErrorH, isErrorD, "Pressure_Equation");

    if (paramsH->USE_LinearSolver) {
        ChFsiLinearSolver myLS(paramsH->LinearSolver, 0.0, paramsH->LinearSolver_Abs_Tol,
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
        MaxRes = myLS.GetResidual();
        Iteration = myLS.GetNumIterations();
        //        if (myLS.GetSolverStatus()) {
        //            std::cout << " Linear solver converged to " << myLS.GetResidual() << " tolerance";
        //            std::cout << " after " << myLS.GetNumIterations() << " iterations" << std::endl;
        //        } else {
        //            std::cout << "Failed to converge after " << myLS.GetNumIterations() << " iterations";
        //            std::cout << " (" << myLS.GetResidual() << " final residual)" << std::endl;
        //        }
    } else {
        thrust::fill(Residuals.begin(), Residuals.end(), 0.0);
        while ((MaxRes > paramsH->LinearSolver_Abs_Tol || Iteration < 3) &&
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
    printf(" Pressure Poisson Equation: %f (sec) - Final Residual=%.3e - #Iter=%d\n", Pressure_Computation, MaxRes,
           Iteration);
    //==================================Velocity_Correction_and_update============================================
    double updateClock = clock();
    thrust::fill(vel_vis_Sorted_D.begin(), vel_vis_Sorted_D.end(), mR3(0.0));
    Velocity_Correction_and_update<<<numBlocks, numThreads>>>(
        mR4CAST(sortedSphMarkersD->posRadD), mR4CAST(sortedSphMarkersD->rhoPresMuD),
        mR3CAST(sortedSphMarkersD->velMasD), mR3CAST(vel_vis_Sorted_D), mR3CAST(V_star_new), R1CAST(q_new),
        R1CAST(csrValFunciton), mR3CAST(csrValGradient), U1CAST(csrColInd), U1CAST(Contact_i), numAllMarkers, MaxVel,
        paramsH->dT, isErrorD);
    ChDeviceUtils::Sync_CheckError(isErrorH, isErrorD, "Velocity_Correction_and_update");

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

    double SPLIT_MERGE = clock();

    //    Calc_HelperMarkers_normals<<<numBlocks, numThreads>>>(
    //        mR4CAST(sortedSphMarkersD->posRadD), mR4CAST(sortedSphMarkersD->rhoPresMuD), mR3CAST(helpers_normal),
    //        I1CAST(myType), U1CAST(markersProximityD->cellStartD), U1CAST(markersProximityD->cellEndD),
    //        U1CAST(markersProximityD->gridMarkerIndexD), numAllMarkers, isErrorD2);
    //    ChDeviceUtils::Sync_CheckError(isErrorH, isErrorD, "Calc_HelperMarkers_normals");
    //
    //    double CalcSplitMerge = clock();
    //
    //    int limit = numObjectsH->numFluidMarkers + numObjectsH->numGhostMarkers + numObjectsH->numHelperMarkers;
    //
    //    Calc_Splits_and_Merges<<<numBlocks, numThreads>>>(
    //        mR4CAST(sortedSphMarkersD->posRadD), mR4CAST(sortedSphMarkersD->rhoPresMuD),
    //        mR3CAST(sortedSphMarkersD->velMasD), mR3CAST(helpers_normal), R1CAST(csrValLaplacian),
    //        mR3CAST(csrValGradient), R1CAST(csrValFunciton), U1CAST(csrColInd), U1CAST(Contact_i), U1CAST(SplitMe),
    //        U1CAST(MergeMe), I1CAST(myType), U1CAST(markersProximityD->gridMarkerIndexD), paramsH->HSML / 2.0,
    //        paramsH->HSML, numAllMarkers, limit, isErrorD2);
    //    ChDeviceUtils::Sync_CheckError(isErrorH, isErrorD, "Calc_Splits_and_Merges");
    //    printf(" Merging and ");
    //    CopySortedToOriginal_NonInvasive_R3(fsiGeneralData->vis_vel_SPH_D, sortedSphMarkersD->velMasD,
    //                                        markersProximityD->gridMarkerIndexD);
    //    return;

    //    Split<<<numBlocks, numThreads>>>(mR4CAST(sortedSphMarkersD->posRadD), mR4CAST(sortedSphMarkersD->rhoPresMuD),
    //                                     mR3CAST(sortedSphMarkersD->velMasD), mR3CAST(Normals),
    //                                     R1CAST(csrValLaplacian), mR3CAST(csrValGradient), R1CAST(csrValFunciton),
    //                                     U1CAST(csrColInd), U1CAST(Contact_i), U1CAST(SplitMe), U1CAST(MergeMe),
    //                                     I1CAST(myType), U1CAST(markersProximityD->gridMarkerIndexD), paramsH->HSML
    //                                     / 2.0, paramsH->HSML, numAllMarkers, limit, isErrorD2);
    //    ChDeviceUtils::Sync_CheckError(isErrorH, isErrorD, "Split");
    printf("Spliting : %f \n", (clock() - SPLIT_MERGE) / (double)CLOCKS_PER_SEC);
    //
    CopySortedToOriginal_NonInvasive_R3(fsiGeneralData->vis_vel_SPH_D, vel_vis_Sorted_D,
                                        markersProximityD->gridMarkerIndexD);
    CopySortedToOriginal_NonInvasive_R3(sphMarkersD->velMasD, sortedSphMarkersD->velMasD,
                                        markersProximityD->gridMarkerIndexD);
    CopySortedToOriginal_NonInvasive_R4(sphMarkersD->rhoPresMuD, sortedSphMarkersD->rhoPresMuD,
                                        markersProximityD->gridMarkerIndexD);
    CopySortedToOriginal_NonInvasive_R4(sphMarkersD->posRadD, sortedSphMarkersD->posRadD,
                                        markersProximityD->gridMarkerIndexD);

    fsiCollisionSystem->ArrangeData(sphMarkersD);
    ChFsiForceI2SPH::PreProcessor(sortedSphMarkersD, false, false);
    Shifting<<<numBlocks, numThreads>>>(mR4CAST(sortedSphMarkersD->posRadD), mR4CAST(sortedSphMarkersD->rhoPresMuD),
                                        mR3CAST(sortedSphMarkersD->velMasD), mR3CAST(vel_vis_Sorted_D),
                                        R1CAST(csrValFunciton), mR3CAST(csrValGradient), U1CAST(csrColInd),
                                        U1CAST(Contact_i), numAllMarkers, MaxVel, paramsH->dT, isErrorD);
    ChDeviceUtils::Sync_CheckError(isErrorH, isErrorD, "Shifting");
    Real4_x unary_op(paramsH->rho0);
    thrust::plus<Real> binary_op;
    Real Ave_density = thrust::transform_reduce(sphMarkersD->rhoPresMuD.begin(), sphMarkersD->rhoPresMuD.end(),
                                                unary_op, 0.0, binary_op) /
                       (numObjectsH->numFluidMarkers * paramsH->rho0);
    double updateComputation = (clock() - updateClock) / (double)CLOCKS_PER_SEC;
    Real Re = paramsH->L_Characteristic * paramsH->rho0 * MaxVel / paramsH->mu0;
    printf(" Velocity_Correction_and_update: %f (sec), Ave_density_Err=%.3e, Re=%.1f\n", updateComputation, Ave_density,
           Re);

    //============================================================================================================
    CopySortedToOriginal_NonInvasive_R3(fsiGeneralData->vis_vel_SPH_D, vel_vis_Sorted_D,
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
