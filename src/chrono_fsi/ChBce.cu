// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2014 projectchrono.org
// All right reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Author: Arman Pazouki
// =============================================================================
//
// Base class for processing boundary condition enforcing (bce) markers forces
// in fsi system.//
// =============================================================================

#include "chrono_fsi/ChBce.cuh"  //for FsiGeneralData
#include "chrono_fsi/ChSphGeneral.cuh"

namespace chrono {
namespace fsi {

//__device__ void Shells_ShapeFunctions(Real& NA, Real& NB, Real& NC, Real& ND, Real x, Real y) {
//  NA = 0.25 * (1.0 - x) * (1.0 - y);
//  NB = 0.25 * (1.0 + x) * (1.0 - y);
//  NC = 0.25 * (1.0 + x) * (1.0 + y);
//  ND = 0.25 * (1.0 - x) * (1.0 + y);
//}
// double precision atomic add function
__device__ double atomicAdd(double* address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;

  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);

  return __longlong_as_double(old);
}
//--------------------------------------------------------------------------------------------------------------------------------
__global__ void Populate_RigidSPH_MeshPos_LRF_kernel(Real3* rigidSPH_MeshPos_LRF_D,
                                                     Real3* posRadD,
                                                     uint* rigidIdentifierD,
                                                     Real3* posRigidD,
                                                     Real4* qD) {
  uint index = blockIdx.x * blockDim.x + threadIdx.x;
  int numRigid_SphMarkers = numObjectsD.numRigid_SphMarkers;
  if (index >= numObjectsD.numRigid_SphMarkers) {
    return;
  }
  int rigidIndex = rigidIdentifierD[index];
  uint rigidMarkerIndex = index + numObjectsD.startRigidMarkers;  // updatePortion = [start, end]
                                                                  // index of the update portion
  Real4 q4 = qD[rigidIndex];
  Real3 a1, a2, a3;
  RotationMatirixFromQuaternion(a1, a2, a3, q4);
  Real3 dist3 = posRadD[rigidMarkerIndex] - posRigidD[rigidIndex];
  Real3 dist3LF = InverseRotate_By_RotationMatrix_DeviceHost(a1, a2, a3, dist3);
  rigidSPH_MeshPos_LRF_D[index] = dist3LF;
}

//--------------------------------------------------------------------------------------------------------------------------------
__global__ void Populate_FlexSPH_MeshPos_LRF_kernel(Real3* FlexSPH_MeshPos_LRF_D,
                                                    Real3* posRadD,
                                                    uint* FlexIdentifierD,
                                                    const int numFlex1D,
                                                    uint2* CableElementsNodes,
                                                    uint4* ShellElementsNodes,
                                                    Real3* pos_fsi_fea_D,
                                                    Real Spacing) {
  uint index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= numObjectsD.numFlex_SphMarkers) {
    return;
  }
  int numFlexSphMarkers = numObjectsD.numFlex_SphMarkers;

  int FlexIndex = FlexIdentifierD[index];
  uint FlexMarkerIndex = index + numObjectsD.startFlexMarkers;  // updatePortion = [start, end]
  //  printf("FlexInd ex=%d, FlexMarkerIndex=%d\n", FlexIndex, FlexMarkerIndex);

  if (FlexIndex < numFlex1D) {
    uint2 cableNodes = CableElementsNodes[FlexIndex];
    //    printf("FlexIndex=%d, CableElementsNodes[FlexIndex]=%d,%d\n", FlexIndex, cableNodes.x, cableNodes.y);

    Real3 pos_fsi_fea_D_nA = pos_fsi_fea_D[cableNodes.x];
    Real3 pos_fsi_fea_D_nB = pos_fsi_fea_D[cableNodes.y];
    Real3 dist3 = posRadD[FlexMarkerIndex] - pos_fsi_fea_D_nA;
    Real3 x_dir = pos_fsi_fea_D_nB - pos_fsi_fea_D_nA;
    Real Cable_x = length(x_dir);
    //    printf("dist3= (%f, %f, %f)\n", dist3.x, dist3.y, dist3.z);
    x_dir = x_dir / length(x_dir);
    Real norm_dir_length = length(cross(dist3, x_dir));

    Real3 y_dir = mR3(-x_dir.y, x_dir.x, 0) + mR3(-x_dir.z, 0, x_dir.x) + mR3(0, -x_dir.z, x_dir.y);
    y_dir = y_dir / length(y_dir);
    Real3 z_dir = cross(x_dir, y_dir);
    //    printf("x_dir= (%f, %f, %f), y_dir= (%f, %f, %f), z_dir= (%f, %f, %f)\n", x_dir.x, x_dir.y, x_dir.z, y_dir.x,
    //           y_dir.y, y_dir.z, z_dir.x, z_dir.y, z_dir.z);
    Real dx = dot(dist3, x_dir);
    Real dy = dot(dist3, y_dir);
    Real dz = dot(dist3, z_dir);
    if (abs(dy) > 0)
      dy /= Spacing;
    if (abs(dz) > 0)
      dz /= Spacing;

    FlexSPH_MeshPos_LRF_D[index] = mR3(dx / Cable_x, dy, dz);

    //    printf("FlexSPH_MeshPos_LRF_D[%d]=%f, %f, %f, Cable_x=%f, dx=%f\n", index, FlexSPH_MeshPos_LRF_D[index].x,
    //           FlexSPH_MeshPos_LRF_D[index].y, FlexSPH_MeshPos_LRF_D[index].z, Cable_x, dx);
  }
  if (FlexIndex >= numFlex1D) {
    uint4 shellNodes = ShellElementsNodes[FlexIndex - numFlex1D];
    //    printf("FlexIndex=%d, ShellElementsNodes[FlexIndex]=%d,%d,%d,%d\n", FlexIndex, shellNodes.x, shellNodes.y,
    //           shellNodes.z, shellNodes.w);

    Real3 pos_fsi_fea_D_nA = pos_fsi_fea_D[shellNodes.x];
    Real3 pos_fsi_fea_D_nB = pos_fsi_fea_D[shellNodes.y];
    Real3 pos_fsi_fea_D_nC = pos_fsi_fea_D[shellNodes.z];
    Real3 pos_fsi_fea_D_nD = pos_fsi_fea_D[shellNodes.w];
    //    printf(" %d shellNodes %d %d %d %d\n", FlexIndex, shellNodes.x, shellNodes.y, shellNodes.z, shellNodes.w);
    Real3 Shell_center = 0.25 * (pos_fsi_fea_D_nA + pos_fsi_fea_D_nB + pos_fsi_fea_D_nC + pos_fsi_fea_D_nD);

    Real3 dist3 = posRadD[FlexMarkerIndex] - Shell_center;
    Real Shell_x = 0.25 * (length(pos_fsi_fea_D_nB - pos_fsi_fea_D_nA) + length(pos_fsi_fea_D_nC - pos_fsi_fea_D_nD));

    Real Shell_y = 0.25 * (length(pos_fsi_fea_D_nD - pos_fsi_fea_D_nA) + length(pos_fsi_fea_D_nC - pos_fsi_fea_D_nB));

    Real3 x_dir = (pos_fsi_fea_D_nB - pos_fsi_fea_D_nA + (pos_fsi_fea_D_nC - pos_fsi_fea_D_nD));

    Real3 y_dir = (pos_fsi_fea_D_nD - pos_fsi_fea_D_nA + (pos_fsi_fea_D_nC - pos_fsi_fea_D_nB));

    Real3 Normal = normalize(cross(x_dir, y_dir));
    Real zSide;

    zSide = dot(Normal, dist3) / Spacing;

    Real dx = dot(dist3, x_dir) / length(x_dir);
    Real dy = dot(dist3, y_dir) / length(y_dir);
    //  printf(" FlexMarkerIndex:%d dist3=%f,%f,%f, normal= %f,%f,%f, zside= %f\n", FlexMarkerIndex, dist3.x, dist3.y,
    //         dist3.z, Normal.x, Normal.y, Normal.z, zSide);

    FlexSPH_MeshPos_LRF_D[index] = mR3(dx / Shell_x, dy / Shell_y, zSide);
  }
}
//--------------------------------------------------------------------------------------------------------------------------------
__global__ void Calc_Flex_FSI_ForcesD(Real3* FlexSPH_MeshPos_LRF_D,
                                      uint* FlexIdentifierD,
                                      const int numFlex1D,
                                      uint2* CableElementsNodes,  // This is the connectivity of FEA mesh.
                                      uint4* ShellElementsNodes,  // This is the connectivity of FEA mesh.
                                      Real4* derivVelRhoD,
                                      Real3* pos_fsi_fea_D,
                                      Real3* Flex_FSI_ForcesD) {
  uint index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= numObjectsD.numFlex_SphMarkers) {
    return;
  }
  int numFlexSphMarkers = numObjectsD.numFlex_SphMarkers;

  int FlexIndex = FlexIdentifierD[index];
  uint FlexMarkerIndex = index + numObjectsD.startFlexMarkers;  // updatePortion = [start, end]
  if (FlexIndex < numFlex1D) {
    Real2 N_cable = Cables_ShapeFunctions(FlexSPH_MeshPos_LRF_D[index].x);
    Real NA = N_cable.x;
    Real NB = N_cable.y;

    int nA = CableElementsNodes[FlexIndex].x;
    int nB = CableElementsNodes[FlexIndex].y;
    // printf("idx=%d, N.A= %f, N.B=%f to nodes nA=%d, nB=%d\n",index,NA,NB,nA,nB);

    atomicAdd(&(Flex_FSI_ForcesD[nA].x), NA * (double)derivVelRhoD[FlexMarkerIndex].x);
    atomicAdd(&(Flex_FSI_ForcesD[nA].y), NA * (double)derivVelRhoD[FlexMarkerIndex].y);
    atomicAdd(&(Flex_FSI_ForcesD[nA].z), NA * (double)derivVelRhoD[FlexMarkerIndex].z);

    atomicAdd(&(Flex_FSI_ForcesD[nB].x), NB * (double)derivVelRhoD[FlexMarkerIndex].x);
    atomicAdd(&(Flex_FSI_ForcesD[nB].y), NB * (double)derivVelRhoD[FlexMarkerIndex].y);
    atomicAdd(&(Flex_FSI_ForcesD[nB].z), NB * (double)derivVelRhoD[FlexMarkerIndex].z);
  }
  if (FlexIndex >= numFlex1D) {
    Real4 N_shell = Shells_ShapeFunctions(FlexSPH_MeshPos_LRF_D[index].x, FlexSPH_MeshPos_LRF_D[index].y);
    Real NA = N_shell.x;
    Real NB = N_shell.y;
    Real NC = N_shell.z;
    Real ND = N_shell.w;

    int nA = ShellElementsNodes[FlexIndex - numFlex1D].x;
    int nB = ShellElementsNodes[FlexIndex - numFlex1D].y;
    int nC = ShellElementsNodes[FlexIndex - numFlex1D].z;
    int nD = ShellElementsNodes[FlexIndex - numFlex1D].w;

    atomicAdd(&(Flex_FSI_ForcesD[nA].x), NA * (double)derivVelRhoD[FlexMarkerIndex].x);
    atomicAdd(&(Flex_FSI_ForcesD[nA].y), NA * (double)derivVelRhoD[FlexMarkerIndex].y);
    atomicAdd(&(Flex_FSI_ForcesD[nA].z), NA * (double)derivVelRhoD[FlexMarkerIndex].z);

    atomicAdd(&(Flex_FSI_ForcesD[nB].x), NB * (double)derivVelRhoD[FlexMarkerIndex].x);
    atomicAdd(&(Flex_FSI_ForcesD[nB].y), NB * (double)derivVelRhoD[FlexMarkerIndex].y);
    atomicAdd(&(Flex_FSI_ForcesD[nB].z), NB * (double)derivVelRhoD[FlexMarkerIndex].z);

    atomicAdd(&(Flex_FSI_ForcesD[nC].x), NC * (double)derivVelRhoD[FlexMarkerIndex].x);
    atomicAdd(&(Flex_FSI_ForcesD[nC].y), NC * (double)derivVelRhoD[FlexMarkerIndex].y);
    atomicAdd(&(Flex_FSI_ForcesD[nC].z), NC * (double)derivVelRhoD[FlexMarkerIndex].z);

    atomicAdd(&(Flex_FSI_ForcesD[nD].x), ND * (double)derivVelRhoD[FlexMarkerIndex].x);
    atomicAdd(&(Flex_FSI_ForcesD[nD].y), ND * (double)derivVelRhoD[FlexMarkerIndex].y);
    atomicAdd(&(Flex_FSI_ForcesD[nD].z), ND * (double)derivVelRhoD[FlexMarkerIndex].z);
  }
}
//--------------------------------------------------------------------------------------------------------------------------------
// collide a particle against all other particles in a given cell
// Arman : revisit equation 10 of tech report, is it only on fluid or it is on
// all markers
__device__ void BCE_modification_Share(Real3& sumVW,
                                       Real& sumWAll,
                                       Real3& sumRhoRW,
                                       Real& sumPW,
                                       Real& sumWFluid,
                                       int& isAffectedV,
                                       int& isAffectedP,
                                       int3 gridPos,
                                       Real3 posRadA,
                                       Real3* sortedPosRad,
                                       Real3* sortedVelMas,
                                       Real4* sortedRhoPreMu,
                                       uint* cellStart,
                                       uint* cellEnd) {
  uint gridHash = calcGridHash(gridPos);
  // get start of bucket for this cell
  uint startIndex = cellStart[gridHash];
  if (startIndex != 0xffffffff) {  // cell is not empty
    // iterate over particles in this cell
    uint endIndex = cellEnd[gridHash];

    for (uint j = startIndex; j < endIndex; j++) {
      Real3 posRadB = sortedPosRad[j];
      Real3 dist3 = Distance(posRadA, posRadB);
      Real d = length(dist3);
      Real4 rhoPresMuB = sortedRhoPreMu[j];
      if (d > RESOLUTION_LENGTH_MULT * paramsD.HSML || rhoPresMuB.w > -.1)
        continue;

      Real Wd = W3(d);
      Real WdOvRho = Wd / rhoPresMuB.x;
      isAffectedV = 1;
      Real3 velMasB = sortedVelMas[j];
      sumVW += velMasB * WdOvRho;
      sumWAll += WdOvRho;

      isAffectedP = 1;
      sumRhoRW += rhoPresMuB.x * dist3 * WdOvRho;
      sumPW += rhoPresMuB.y * WdOvRho;
      sumWFluid += WdOvRho;
    }
  }
}

//--------------------------------------------------------------------------------------------------------------------------------
__global__ void new_BCE_VelocityPressure(Real3* velMas_ModifiedBCE,    // input: sorted velocities
                                         Real4* rhoPreMu_ModifiedBCE,  // input: sorted velocities
                                         Real3* sortedPosRad,          // input: sorted positions
                                         Real3* sortedVelMas,          // input: sorted velocities
                                         Real4* sortedRhoPreMu,
                                         uint* cellStart,
                                         uint* cellEnd,
                                         uint* mapOriginalToSorted,
                                         Real3* bceAcc,
                                         int2 updatePortion,
                                         volatile bool* isErrorD) {
  uint bceIndex = blockIdx.x * blockDim.x + threadIdx.x;
  uint sphIndex = bceIndex + updatePortion.x;  // updatePortion = [start, end] index of the update portion
  if (sphIndex >= updatePortion.y) {
    return;
  }

  uint idA = mapOriginalToSorted[sphIndex];
  Real4 rhoPreMuA = sortedRhoPreMu[idA];
  Real3 posRadA = sortedPosRad[idA];
  Real3 velMasA = sortedVelMas[idA];
  int isAffectedV = 0;
  int isAffectedP = 0;

  Real3 sumVW = mR3(0);
  Real sumWAll = 0;
  Real3 sumRhoRW = mR3(0);
  Real sumPW = 0;
  Real sumWFluid = 0;

  // get address in grid
  int3 gridPos = calcGridPos(posRadA);

  /// if (gridPos.x == paramsD.gridSize.x-1) printf("****aha %d %d\n",
  /// gridPos.x, paramsD.gridSize.x);

  // examine neighbouring cells
  for (int z = -1; z <= 1; z++) {
    for (int y = -1; y <= 1; y++) {
      for (int x = -1; x <= 1; x++) {
        int3 neighbourPos = gridPos + mI3(x, y, z);
        BCE_modification_Share(sumVW, sumWAll, sumRhoRW, sumPW, sumWFluid, isAffectedV, isAffectedP, neighbourPos,
                               posRadA, sortedPosRad, sortedVelMas, sortedRhoPreMu, cellStart, cellEnd);
      }
    }
  }

  if (isAffectedV) {
    Real3 modifiedBCE_v = 2 * velMasA - sumVW / sumWAll;
    velMas_ModifiedBCE[bceIndex] = modifiedBCE_v;
  }
  if (isAffectedP) {
    // pressure
    Real3 a3 = mR3(0);
    if (fabs(rhoPreMuA.w) > 0) {  // rigid BCE
      int rigidBceIndex = sphIndex - numObjectsD.startRigidMarkers;
      if (rigidBceIndex < 0 || rigidBceIndex >= numObjectsD.numRigid_SphMarkers) {
        printf(
            "Error! marker index out of bound: thrown from "
            "SDKCollisionSystem.cu, new_BCE_VelocityPressure !\n");
        *isErrorD = true;
        return;
      }
      a3 = bceAcc[rigidBceIndex];
    }
    Real pressure = (sumPW + dot(paramsD.gravity - a3, sumRhoRW)) / sumWFluid;  //(in fact:  (paramsD.gravity -
    // aW), but aW for moving rigids
    // is hard to calc. Assume aW is
    // zero for now
    Real density = InvEos(pressure);
    rhoPreMu_ModifiedBCE[bceIndex] = mR4(density, pressure, rhoPreMuA.z, rhoPreMuA.w);
  }
}
//--------------------------------------------------------------------------------------------------------------------------------
// calculate marker acceleration, required in ADAMI
__global__ void calcBceAcceleration_kernel(Real3* bceAcc,
                                           Real4* q_fsiBodies_D,
                                           Real3* accRigid_fsiBodies_D,
                                           Real3* omegaVelLRF_fsiBodies_D,
                                           Real3* omegaAccLRF_fsiBodies_D,
                                           Real3* rigidSPH_MeshPos_LRF_D,
                                           const uint* rigidIdentifierD) {
  uint bceIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (bceIndex >= numObjectsD.numRigid_SphMarkers) {
    return;
  }

  int rigidBodyIndex = rigidIdentifierD[bceIndex];
  Real3 acc3 = accRigid_fsiBodies_D[rigidBodyIndex];  // linear acceleration (CM)

  Real4 q4 = q_fsiBodies_D[rigidBodyIndex];
  Real3 a1, a2, a3;
  RotationMatirixFromQuaternion(a1, a2, a3, q4);
  Real3 wVel3 = omegaVelLRF_fsiBodies_D[rigidBodyIndex];
  Real3 rigidSPH_MeshPos_LRF = rigidSPH_MeshPos_LRF_D[bceIndex];
  Real3 wVelCrossS = cross(wVel3, rigidSPH_MeshPos_LRF);
  Real3 wVelCrossWVelCrossS = cross(wVel3, wVelCrossS);
  acc3 += dot(a1, wVelCrossWVelCrossS), dot(a2, wVelCrossWVelCrossS),
      dot(a3,
          wVelCrossWVelCrossS);  // centrigugal acceleration

  Real3 wAcc3 = omegaAccLRF_fsiBodies_D[rigidBodyIndex];
  Real3 wAccCrossS = cross(wAcc3, rigidSPH_MeshPos_LRF);
  acc3 += dot(a1, wAccCrossS), dot(a2, wAccCrossS), dot(a3, wAccCrossS);  // tangential acceleration

  //	printf("linear acc %f %f %f point acc %f %f %f \n", accRigid3.x,
  // accRigid3.y, accRigid3.z, acc3.x, acc3.y,
  // acc3.z);
  bceAcc[bceIndex] = acc3;
}
//--------------------------------------------------------------------------------------------------------------------------------
// updates the rigid body particles
__global__ void UpdateRigidMarkersPositionVelocityD(Real3* posRadD,
                                                    Real3* velMasD,
                                                    const Real3* rigidSPH_MeshPos_LRF_D,
                                                    const uint* rigidIdentifierD,
                                                    Real3* posRigidD,
                                                    Real4* velMassRigidD,
                                                    Real3* omegaLRF_D,
                                                    Real4* qD) {
  uint index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= numObjectsD.numRigid_SphMarkers) {
    return;
  }
  uint rigidMarkerIndex = index + numObjectsD.startRigidMarkers;  // updatePortion = [start, end]
                                                                  // index of the update portion
  int rigidBodyIndex = rigidIdentifierD[index];

  Real4 q4 = qD[rigidBodyIndex];
  Real3 a1, a2, a3;
  RotationMatirixFromQuaternion(a1, a2, a3, q4);

  Real3 rigidSPH_MeshPos_LRF = rigidSPH_MeshPos_LRF_D[index];

  // position
  Real3 p_Rigid = posRigidD[rigidBodyIndex];
  posRadD[rigidMarkerIndex] =
      p_Rigid + mR3(dot(a1, rigidSPH_MeshPos_LRF), dot(a2, rigidSPH_MeshPos_LRF), dot(a3, rigidSPH_MeshPos_LRF));

  // velocity
  Real4 vM_Rigid = velMassRigidD[rigidBodyIndex];
  Real3 omega3 = omegaLRF_D[rigidBodyIndex];
  Real3 omegaCrossS = cross(omega3, rigidSPH_MeshPos_LRF);
  velMasD[rigidMarkerIndex] = mR3(vM_Rigid) + dot(a1, omegaCrossS), dot(a2, omegaCrossS), dot(a3, omegaCrossS);
}
//--------------------------------------------------------------------------------------------------------------------------------
// Real3 *posRadD, uint *FlexIdentifierD, Real3 *posFlex_fsiBodies_nA_D, Real3 *posFlex_fsiBodies_nB_D,
//    Real3 *posFlex_fsiBodies_nC_D, Real3 *posFlex_fsiBodies_nD_D

__global__ void UpdateFlexMarkersPositionVelocityAccD(Real3* posRadD,
                                                      Real3* FlexSPH_MeshPos_LRF_D,
                                                      Real3* velMasD,
                                                      const uint* FlexIdentifierD,
                                                      const int numFlex1D,
                                                      uint2* CableElementsNodes,
                                                      uint4* ShellelementsNodes,
                                                      Real3* pos_fsi_fea_D,
                                                      Real3* vel_fsi_fea_D,
                                                      Real Spacing) {
  uint index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= numObjectsD.numFlex_SphMarkers) {
    return;
  }
  int numFlexSphMarkers = numObjectsD.numFlex_SphMarkers;

  int FlexIndex = FlexIdentifierD[index];
  //  printf(" %d FlexIndex= %d\n", index, FlexIndex);

  uint FlexMarkerIndex = index + numObjectsD.startFlexMarkers;  // updatePortion = [start, end]

  if (FlexIndex < numFlex1D) {
    uint2 CableNodes = CableElementsNodes[FlexIndex];
    Real3 pos_fsi_fea_D_nA = pos_fsi_fea_D[CableNodes.x];
    Real3 pos_fsi_fea_D_nB = pos_fsi_fea_D[CableNodes.y];
    //    printf("CableElementsNodes[%d]=%d,%d\n", FlexIndex, CableNodes.x, CableNodes.y);

    Real3 x_dir = pos_fsi_fea_D_nB - pos_fsi_fea_D_nA;

    x_dir = x_dir / length(x_dir);
    Real3 y_dir = mR3(-x_dir.y, x_dir.x, 0) + mR3(-x_dir.z, 0, x_dir.x) + mR3(0, -x_dir.z, x_dir.y);
    y_dir = y_dir / length(y_dir);
    Real3 z_dir = cross(x_dir, y_dir);

    Real2 N_cable = Cables_ShapeFunctions(FlexSPH_MeshPos_LRF_D[index].x);
    Real NA = N_cable.x;
    Real NB = N_cable.y;

    Real3 vel_fsi_fea_D_nA = vel_fsi_fea_D[CableNodes.x];
    Real3 vel_fsi_fea_D_nB = vel_fsi_fea_D[CableNodes.y];

    posRadD[FlexMarkerIndex] = NA * pos_fsi_fea_D_nA + NB * pos_fsi_fea_D_nB +
                               FlexSPH_MeshPos_LRF_D[index].y * y_dir * Spacing +
                               FlexSPH_MeshPos_LRF_D[index].z * z_dir * Spacing;

    velMasD[FlexMarkerIndex] = NA * vel_fsi_fea_D_nA + NB * vel_fsi_fea_D_nB;
  }

  if (FlexIndex >= numFlex1D) {
    uint4 shellNodes = ShellelementsNodes[FlexIndex - numFlex1D];
    Real3 pos_fsi_fea_D_nA = pos_fsi_fea_D[shellNodes.x];
    Real3 pos_fsi_fea_D_nB = pos_fsi_fea_D[shellNodes.y];
    Real3 pos_fsi_fea_D_nC = pos_fsi_fea_D[shellNodes.z];
    Real3 pos_fsi_fea_D_nD = pos_fsi_fea_D[shellNodes.w];

    Real3 Shell_center = 0.25 * (pos_fsi_fea_D_nA + pos_fsi_fea_D_nB + pos_fsi_fea_D_nC + pos_fsi_fea_D_nD);

    Real3 dist3 = FlexSPH_MeshPos_LRF_D[index] - Shell_center;
    //  printf(" %d dist3= %f,%f,%f center= %f,%f,%f\n", FlexMarkerIndex, dist3.x, dist3.y, dist3.z, Shell_center.x,
    //         Shell_center.y, Shell_center.z);

    Real3 x_dir = ((pos_fsi_fea_D_nB - pos_fsi_fea_D_nA) + (pos_fsi_fea_D_nC - pos_fsi_fea_D_nD));

    Real3 y_dir = ((pos_fsi_fea_D_nD - pos_fsi_fea_D_nA) + (pos_fsi_fea_D_nC - pos_fsi_fea_D_nB));

    Real3 Normal = normalize(cross(x_dir, y_dir));
    //  printf(" %d Shell (x,y)= %f,%f\n", FlexMarkerIndex, Shell_x, Shell_y);
    //
    //  Real2 FlexSPH_MeshPos_Natural = mR2(dist3.x / length(x_dir) / 4.0, dist3.y / length(y_dir) / 4.0);
    //
    //  printf(" %d FlexSPH_MeshPos_Natural= %f,%f,%f\n", FlexMarkerIndex, FlexSPH_MeshPos_Natural.x,
    //         FlexSPH_MeshPos_Natural.y, FlexSPH_MeshPos_LRF_D[index].z);

    Real4 N_shell = Shells_ShapeFunctions(FlexSPH_MeshPos_LRF_D[index].x, FlexSPH_MeshPos_LRF_D[index].y);
    Real NA = N_shell.x;
    Real NB = N_shell.y;
    Real NC = N_shell.z;
    Real ND = N_shell.w;

    Real3 vel_fsi_fea_D_nA = vel_fsi_fea_D[shellNodes.x];
    Real3 vel_fsi_fea_D_nB = vel_fsi_fea_D[shellNodes.y];
    Real3 vel_fsi_fea_D_nC = vel_fsi_fea_D[shellNodes.z];
    Real3 vel_fsi_fea_D_nD = vel_fsi_fea_D[shellNodes.w];

    posRadD[FlexMarkerIndex] = NA * pos_fsi_fea_D_nA + NB * pos_fsi_fea_D_nB + NC * pos_fsi_fea_D_nC +
                               ND * pos_fsi_fea_D_nD + Normal * FlexSPH_MeshPos_LRF_D[index].z * Spacing;

    velMasD[FlexMarkerIndex] =
        NA * vel_fsi_fea_D_nA + NB * vel_fsi_fea_D_nB + NC * vel_fsi_fea_D_nC + ND * vel_fsi_fea_D_nD;
  }
}

//--------------------------------------------------------------------------------------------------------------------------------
__global__ void Calc_Rigid_FSI_ForcesD(Real3* rigid_FSI_ForcesD, Real4* totalSurfaceInteractionRigid4) {
  uint rigidSphereA = blockIdx.x * blockDim.x + threadIdx.x;
  if (rigidSphereA >= numObjectsD.numRigidBodies) {
    return;
  }
  Real3 force3 = paramsD.markerMass * mR3(totalSurfaceInteractionRigid4[rigidSphereA]);
  rigid_FSI_ForcesD[rigidSphereA] = force3;
}
//--------------------------------------------------------------------------------------------------------------------------------
__global__ void Calc_Markers_TorquesD(Real3* torqueMarkersD,
                                      Real4* derivVelRhoD,
                                      Real3* posRadD,
                                      uint* rigidIdentifierD,
                                      Real3* posRigidD) {
  uint index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= numObjectsD.numRigid_SphMarkers) {
    return;
  }
  uint rigidMarkerIndex = index + numObjectsD.startRigidMarkers;
  Real3 dist3 = Distance(posRadD[rigidMarkerIndex], posRigidD[rigidIdentifierD[index]]);
  torqueMarkersD[index] = paramsD.markerMass * cross(dist3, mR3(derivVelRhoD[rigidMarkerIndex]));  // paramsD.markerMass
                                                                                                   // is multiplied to
                                                                                                   // convert
  // from SPH acceleration to force
}
//--------------------------------------------------------------------------------------------------------------------------------
ChBce::ChBce(SphMarkerDataD* otherSortedSphMarkersD,
             ProximityDataD* otherMarkersProximityD,
             FsiGeneralData* otherFsiGeneralData,
             SimParams* otherParamsH,
             NumberOfObjects* otherNumObjects)
    : sortedSphMarkersD(otherSortedSphMarkersD),
      markersProximityD(otherMarkersProximityD),
      fsiGeneralData(otherFsiGeneralData),
      paramsH(otherParamsH),
      numObjectsH(otherNumObjects) {}
//--------------------------------------------------------------------------------------------------------------------------------
void ChBce::Finalize(SphMarkerDataD* sphMarkersD, FsiBodiesDataD* fsiBodiesD, FsiMeshDataD* fsiMeshD) {
  cudaMemcpyToSymbolAsync(paramsD, paramsH, sizeof(SimParams));
  cudaMemcpyToSymbolAsync(numObjectsD, numObjectsH, sizeof(NumberOfObjects));

  totalSurfaceInteractionRigid4.resize(numObjectsH->numRigidBodies);
  dummyIdentify.resize(numObjectsH->numRigidBodies);
  torqueMarkersD.resize(numObjectsH->numRigid_SphMarkers);

  // Resizing the arrays used to modify the BCE velocity and pressure according
  // to ADAMI

  int numFlexAndRigidAndBoundaryMarkers =
      fsiGeneralData
          ->referenceArray[2 + numObjectsH->numRigidBodies + numObjectsH->numFlexBodies1D +
                           numObjectsH->numFlexBodies2D - 1]
          .y -
      fsiGeneralData->referenceArray[0].y;
  printf(" numFlexAndRigidAndBoundaryMarkers= %d, All= %d\n", numFlexAndRigidAndBoundaryMarkers,
         numObjectsH->numBoundaryMarkers + numObjectsH->numRigid_SphMarkers + numObjectsH->numFlex_SphMarkers);

  if ((numObjectsH->numBoundaryMarkers + numObjectsH->numRigid_SphMarkers + numObjectsH->numFlex_SphMarkers) !=
      numFlexAndRigidAndBoundaryMarkers) {
    throw std::runtime_error("Error! number of flex and rigid and boundary markers are saved incorrectly!\n");
  }
  velMas_ModifiedBCE.resize(numFlexAndRigidAndBoundaryMarkers);
  rhoPreMu_ModifiedBCE.resize(numFlexAndRigidAndBoundaryMarkers);

  // Populate local position of BCE markers
  Populate_RigidSPH_MeshPos_LRF(sphMarkersD, fsiBodiesD);

  Populate_FlexSPH_MeshPos_LRF(sphMarkersD, fsiMeshD);
}
//--------------------------------------------------------------------------------------------------------------------------------
ChBce::~ChBce() {
  // TODO
}

////--------------------------------------------------------------------------------------------------------------------------------
void ChBce::MakeRigidIdentifier() {
  if (numObjectsH->numRigidBodies > 0) {
    for (int rigidSphereA = 0; rigidSphereA < numObjectsH->numRigidBodies; rigidSphereA++) {
      int4 referencePart = fsiGeneralData->referenceArray[2 + rigidSphereA];
      if (referencePart.z != 1) {
        printf(
            " Error! in accessing rigid bodies. Reference array indexing is "
            "wrong\n");
        return;
      }
      int2 updatePortion = mI2(referencePart);  // first two component of the
      thrust::fill(fsiGeneralData->rigidIdentifierD.begin() + (updatePortion.x - numObjectsH->startRigidMarkers),
                   fsiGeneralData->rigidIdentifierD.begin() + (updatePortion.y - numObjectsH->startRigidMarkers),
                   rigidSphereA);
    }
  }
}
////--------------------------------------------------------------------------------------------------------------------------------
void ChBce::MakeFlexIdentifier() {
  if ((numObjectsH->numFlexBodies1D + numObjectsH->numFlexBodies2D) > 0) {
    fsiGeneralData->FlexIdentifierD.resize(numObjectsH->numFlex_SphMarkers);

    for (int CableNum = 0; CableNum < numObjectsH->numFlexBodies1D; CableNum++) {
      int4 referencePart = fsiGeneralData->referenceArray_FEA[CableNum];
      //      printf(" Item Index for this Flex body is %d. ", 2 + numObjectsH->numRigidBodies + CableNum);
      //      printf(" .x=%d, .y=%d, .z=%d, .w=%d", referencePart.x, referencePart.y, referencePart.z, referencePart.w);

      if (referencePart.z != 2) {
        printf(
            " Error! in accessing flex bodies. Reference array indexing is "
            "wrong\n");
        return;
      }
      int2 updatePortion = mI2(referencePart);
      thrust::fill(fsiGeneralData->FlexIdentifierD.begin() + (updatePortion.x - numObjectsH->startFlexMarkers),
                   fsiGeneralData->FlexIdentifierD.begin() + (updatePortion.y - numObjectsH->startFlexMarkers),
                   CableNum);

      //      printf("From %d to %d FlexIdentifierD=%d\n", updatePortion.x, updatePortion.y, CableNum);
    }

    for (int shellNum = 0; shellNum < numObjectsH->numFlexBodies2D; shellNum++) {
      int4 referencePart = fsiGeneralData->referenceArray_FEA[numObjectsH->numFlexBodies1D + shellNum];
      //      printf(" Item Index for this Flex body is %d. ",
      //             2 + numObjectsH->numRigidBodies + numObjectsH->numFlexBodies1D + shellNum);
      //      printf(" .x=%d, .y=%d, .z=%d, .w=%d", referencePart.x, referencePart.y, referencePart.z, referencePart.w);

      if (referencePart.z != 3) {
        printf(
            " Error! in accessing flex bodies. Reference array indexing is "
            "wrong\n");
        return;
      }
      int2 updatePortion = mI2(referencePart);
      thrust::fill(fsiGeneralData->FlexIdentifierD.begin() + (updatePortion.x - numObjectsH->startFlexMarkers),
                   fsiGeneralData->FlexIdentifierD.begin() + (updatePortion.y - numObjectsH->startFlexMarkers),
                   shellNum + numObjectsH->numFlexBodies1D);

      //      printf("From %d to %d FlexIdentifierD=%d\n", updatePortion.x, updatePortion.y,
      //             shellNum + numObjectsH->numFlexBodies1D);
    }
  }
}
////--------------------------------------------------------------------------------------------------------------------------------

void ChBce::Populate_RigidSPH_MeshPos_LRF(SphMarkerDataD* sphMarkersD, FsiBodiesDataD* fsiBodiesD) {
  if (numObjectsH->numRigidBodies == 0) {
    return;
  }

  MakeRigidIdentifier();

  uint nBlocks_numRigid_SphMarkers;
  uint nThreads_SphMarkers;
  computeGridSize(numObjectsH->numRigid_SphMarkers, 256, nBlocks_numRigid_SphMarkers, nThreads_SphMarkers);

  Populate_RigidSPH_MeshPos_LRF_kernel<<<nBlocks_numRigid_SphMarkers, nThreads_SphMarkers>>>(
      mR3CAST(fsiGeneralData->rigidSPH_MeshPos_LRF_D), mR3CAST(sphMarkersD->posRadD),
      U1CAST(fsiGeneralData->rigidIdentifierD), mR3CAST(fsiBodiesD->posRigid_fsiBodies_D),
      mR4CAST(fsiBodiesD->q_fsiBodies_D));
  cudaThreadSynchronize();
  cudaCheckError();

  UpdateRigidMarkersPositionVelocity(sphMarkersD, fsiBodiesD);
}
////--------------------------------------------------------------------------------------------------------------------------------

void ChBce::Populate_FlexSPH_MeshPos_LRF(SphMarkerDataD* sphMarkersD, FsiMeshDataD* fsiMeshD) {
  if ((numObjectsH->numFlexBodies1D + numObjectsH->numFlexBodies2D) == 0) {
    return;
  }

  MakeFlexIdentifier();

  uint nBlocks_numFlex_SphMarkers;
  uint nThreads_SphMarkers;
  computeGridSize(numObjectsH->numFlex_SphMarkers, 256, nBlocks_numFlex_SphMarkers, nThreads_SphMarkers);

  //  printf(
  //      "size of FlexSPH_MeshPos_LRF_D = %d and posRadD=%d, in "
  //      "ChBce::Populate_FlexSPH_MeshPos_LRF\n",
  //      fsiGeneralData->FlexSPH_MeshPos_LRF_D.size(), sphMarkersD->posRadD.size());
  //
  //  printf(
  //      "size of FlexIdentifierD = %d, numObjectsH->numFlexBodies1D =%d, numObjectsH->numFlexBodies2D "
  //      "=%d,in ChBce::Populate_FlexSPH_MeshPos_LRF\n",
  //      fsiGeneralData->FlexIdentifierD.size(), numObjectsH->numFlexBodies1D, numObjectsH->numFlexBodies2D);
  //
  //  printf(
  //      "size of CableElementsNodes = %d and ShellElementsNodes=%d, fsiMeshD->pos_fsi_fea_D =%d, in "
  //      "ChBce::Populate_FlexSPH_MeshPos_LRF\n",
  //      fsiGeneralData->CableElementsNodes.size(), fsiGeneralData->ShellElementsNodes.size(),
  //      fsiMeshD->pos_fsi_fea_D.size());

  Populate_FlexSPH_MeshPos_LRF_kernel<<<nBlocks_numFlex_SphMarkers, nThreads_SphMarkers>>>(
      mR3CAST(fsiGeneralData->FlexSPH_MeshPos_LRF_D), mR3CAST(sphMarkersD->posRadD),
      U1CAST(fsiGeneralData->FlexIdentifierD), numObjectsH->numFlexBodies1D, U2CAST(fsiGeneralData->CableElementsNodes),
      U4CAST(fsiGeneralData->ShellElementsNodes), mR3CAST(fsiMeshD->pos_fsi_fea_D),
      paramsH->HSML * paramsH->MULT_INITSPACE_Shells);

  cudaThreadSynchronize();
  cudaCheckError();

  UpdateFlexMarkersPositionVelocity(sphMarkersD, fsiMeshD);
}
//--------------------------------------------------------------------------------------------------------------------------------
void ChBce::RecalcSortedVelocityPressure_BCE(thrust::device_vector<Real3>& velMas_ModifiedBCE,
                                             thrust::device_vector<Real4>& rhoPreMu_ModifiedBCE,
                                             const thrust::device_vector<Real3>& sortedPosRad,
                                             const thrust::device_vector<Real3>& sortedVelMas,
                                             const thrust::device_vector<Real4>& sortedRhoPreMu,
                                             const thrust::device_vector<uint>& cellStart,
                                             const thrust::device_vector<uint>& cellEnd,
                                             const thrust::device_vector<uint>& mapOriginalToSorted,
                                             const thrust::device_vector<Real3>& bceAcc,
                                             int2 updatePortion) {
  bool *isErrorH, *isErrorD;
  isErrorH = (bool*)malloc(sizeof(bool));
  cudaMalloc((void**)&isErrorD, sizeof(bool));
  *isErrorH = false;
  cudaMemcpy(isErrorD, isErrorH, sizeof(bool), cudaMemcpyHostToDevice);
  //------------------------------------------------------------------------

  // thread per particle
  uint numThreads, numBlocks;
  computeGridSize(updatePortion.y - updatePortion.x, 64, numBlocks, numThreads);

  new_BCE_VelocityPressure<<<numBlocks, numThreads>>>(
      mR3CAST(velMas_ModifiedBCE),
      mR4CAST(rhoPreMu_ModifiedBCE),  // input: sorted velocities
      mR3CAST(sortedPosRad), mR3CAST(sortedVelMas), mR4CAST(sortedRhoPreMu), U1CAST(cellStart), U1CAST(cellEnd),
      U1CAST(mapOriginalToSorted), mR3CAST(bceAcc), updatePortion, isErrorD);

  cudaThreadSynchronize();
  cudaCheckError()

      //------------------------------------------------------------------------
      cudaMemcpy(isErrorH, isErrorD, sizeof(bool), cudaMemcpyDeviceToHost);
  if (*isErrorH == true) {
    throw std::runtime_error("Error! program crashed in  new_BCE_VelocityPressure!\n");
  }
  cudaFree(isErrorD);
  free(isErrorH);
}
//--------------------------------------------------------------------------------------------------------------------------------
void ChBce::CalcBceAcceleration(thrust::device_vector<Real3>& bceAcc,
                                const thrust::device_vector<Real4>& q_fsiBodies_D,
                                const thrust::device_vector<Real3>& accRigid_fsiBodies_D,
                                const thrust::device_vector<Real3>& omegaVelLRF_fsiBodies_D,
                                const thrust::device_vector<Real3>& omegaAccLRF_fsiBodies_D,
                                const thrust::device_vector<Real3>& rigidSPH_MeshPos_LRF_D,
                                const thrust::device_vector<uint>& rigidIdentifierD,
                                int numRigid_SphMarkers) {
  // thread per particle
  uint numThreads, numBlocks;
  computeGridSize(numRigid_SphMarkers, 64, numBlocks, numThreads);

  calcBceAcceleration_kernel<<<numBlocks, numThreads>>>(
      mR3CAST(bceAcc), mR4CAST(q_fsiBodies_D), mR3CAST(accRigid_fsiBodies_D), mR3CAST(omegaVelLRF_fsiBodies_D),
      mR3CAST(omegaAccLRF_fsiBodies_D), mR3CAST(rigidSPH_MeshPos_LRF_D), U1CAST(rigidIdentifierD));

  cudaThreadSynchronize();
  cudaCheckError();
}
//--------------------------------------------------------------------------------------------------------------------------------
void ChBce::ModifyBceVelocity(SphMarkerDataD* sphMarkersD, FsiBodiesDataD* fsiBodiesD) {
  // modify BCE velocity and pressure
  int numRigidAndBoundaryMarkers =
      fsiGeneralData->referenceArray[2 + numObjectsH->numRigidBodies - 1].y - fsiGeneralData->referenceArray[0].y;
  if ((numObjectsH->numBoundaryMarkers + numObjectsH->numRigid_SphMarkers) != numRigidAndBoundaryMarkers) {
    throw std::runtime_error(
        "Error! number of rigid and boundary markers are "
        "saved incorrectly. Thrown from "
        "ModifyBceVelocity!\n");
  }
  if (!(velMas_ModifiedBCE.size() == numRigidAndBoundaryMarkers &&
        rhoPreMu_ModifiedBCE.size() == numRigidAndBoundaryMarkers)) {
    throw std::runtime_error(
        "Error! size error velMas_ModifiedBCE and "
        "rhoPreMu_ModifiedBCE. Thrown from "
        "ModifyBceVelocity!\n");
  }
  int2 updatePortion =
      mI2(fsiGeneralData->referenceArray[0].y, fsiGeneralData->referenceArray[2 + numObjectsH->numRigidBodies - 1].y);
  if (paramsH->bceType == ADAMI) {
    thrust::device_vector<Real3> bceAcc(numObjectsH->numRigid_SphMarkers);
    if (numObjectsH->numRigid_SphMarkers > 0) {
      CalcBceAcceleration(bceAcc, fsiBodiesD->q_fsiBodies_D, fsiBodiesD->accRigid_fsiBodies_D,
                          fsiBodiesD->omegaVelLRF_fsiBodies_D, fsiBodiesD->omegaAccLRF_fsiBodies_D,
                          fsiGeneralData->rigidSPH_MeshPos_LRF_D, fsiGeneralData->rigidIdentifierD,
                          numObjectsH->numRigid_SphMarkers);
    }
    RecalcSortedVelocityPressure_BCE(velMas_ModifiedBCE, rhoPreMu_ModifiedBCE, sortedSphMarkersD->posRadD,
                                     sortedSphMarkersD->velMasD, sortedSphMarkersD->rhoPresMuD,
                                     markersProximityD->cellStartD, markersProximityD->cellEndD,
                                     markersProximityD->mapOriginalToSorted, bceAcc, updatePortion);
    bceAcc.clear();
  } else {
    thrust::copy(sphMarkersD->velMasD.begin() + updatePortion.x, sphMarkersD->velMasD.begin() + updatePortion.y,
                 velMas_ModifiedBCE.begin());
    thrust::copy(sphMarkersD->rhoPresMuD.begin() + updatePortion.x, sphMarkersD->rhoPresMuD.begin() + updatePortion.y,
                 rhoPreMu_ModifiedBCE.begin());
  }
}
//--------------------------------------------------------------------------------------------------------------------------------
// applies the time step to the current quantities and saves the new values into
// variable with the same name and '2' and
// the end
// precondition: for the first step of RK2, all variables with '2' at the end
// have the values the same as those without
// '2' at the end.
void ChBce::Rigid_Forces_Torques(SphMarkerDataD* sphMarkersD, FsiBodiesDataD* fsiBodiesD) {
  // Arman: InitSystem has to be called before this point to set the number of
  // objects

  if (numObjectsH->numRigidBodies == 0) {
    return;
  }
  //################################################### make force and torque
  // arrays
  //####### Force (Acceleration)
  if (totalSurfaceInteractionRigid4.size() != numObjectsH->numRigidBodies ||
      dummyIdentify.size() != numObjectsH->numRigidBodies ||
      torqueMarkersD.size() != numObjectsH->numRigid_SphMarkers) {
    throw std::runtime_error(
        "Error! wrong size: totalSurfaceInteractionRigid4 "
        "or torqueMarkersD or dummyIdentify. Thrown from "
        "Rigid_Forces_Torques!\n");
  }

  thrust::fill(totalSurfaceInteractionRigid4.begin(), totalSurfaceInteractionRigid4.end(), mR4(0));
  thrust::fill(torqueMarkersD.begin(), torqueMarkersD.end(), mR3(0));

  thrust::equal_to<uint> binary_pred;

  //** forces on BCE markers of each rigid body are accumulated at center.
  //"totalSurfaceInteractionRigid4" is got built.
  (void)thrust::reduce_by_key(fsiGeneralData->rigidIdentifierD.begin(), fsiGeneralData->rigidIdentifierD.end(),
                              fsiGeneralData->derivVelRhoD.begin() + numObjectsH->startRigidMarkers,
                              dummyIdentify.begin(), totalSurfaceInteractionRigid4.begin(), binary_pred,
                              thrust::plus<Real4>());
  thrust::fill(fsiGeneralData->rigid_FSI_ForcesD.begin(), fsiGeneralData->rigid_FSI_ForcesD.end(), mR3(0));

  uint nBlock_UpdateRigid;
  uint nThreads_rigidParticles;
  computeGridSize(numObjectsH->numRigidBodies, 128, nBlock_UpdateRigid, nThreads_rigidParticles);

  //** accumulated BCE forces at center are transformed to acceleration of rigid
  // body "rigid_FSI_ForcesD".
  //"rigid_FSI_ForcesD" gets built.
  Calc_Rigid_FSI_ForcesD<<<nBlock_UpdateRigid, nThreads_rigidParticles>>>(mR3CAST(fsiGeneralData->rigid_FSI_ForcesD),
                                                                          mR4CAST(totalSurfaceInteractionRigid4));
  cudaThreadSynchronize();
  cudaCheckError();

  //####### Torque
  uint nBlocks_numRigid_SphMarkers;
  uint nThreads_SphMarkers;
  computeGridSize(numObjectsH->numRigid_SphMarkers, 256, nBlocks_numRigid_SphMarkers, nThreads_SphMarkers);

  //** the current position of the rigid, 'posRigidD', is used to calculate the
  // moment of BCE acceleration at the rigid
  //*** body center (i.e. torque/mass). "torqueMarkersD" gets built.
  Calc_Markers_TorquesD<<<nBlocks_numRigid_SphMarkers, nThreads_SphMarkers>>>(
      mR3CAST(torqueMarkersD), mR4CAST(fsiGeneralData->derivVelRhoD), mR3CAST(sphMarkersD->posRadD),
      U1CAST(fsiGeneralData->rigidIdentifierD), mR3CAST(fsiBodiesD->posRigid_fsiBodies_D));
  cudaThreadSynchronize();
  cudaCheckError();

  (void)thrust::reduce_by_key(fsiGeneralData->rigidIdentifierD.begin(), fsiGeneralData->rigidIdentifierD.end(),
                              torqueMarkersD.begin(), dummyIdentify.begin(), fsiGeneralData->rigid_FSI_TorquesD.begin(),
                              binary_pred, thrust::plus<Real3>());
}
////////////////////////////////////////////////////////////////////////////////////////////////////////
void ChBce::Flex_Forces(SphMarkerDataD* sphMarkersD, FsiMeshDataD* fsiMeshD) {
  if ((numObjectsH->numFlexBodies1D + numObjectsH->numFlexBodies2D) == 0) {
    return;
  }

  thrust::fill(fsiGeneralData->Flex_FSI_ForcesD.begin(), fsiGeneralData->Flex_FSI_ForcesD.end(), mR3(0));

  // Markers' forces cannot be simply reduced by "reduce_by_key" since the forces should be interpolated position
  // coordinates

  uint nBlocks_numFlex_SphMarkers;
  uint nThreads_SphMarkers;
  computeGridSize(numObjectsH->numFlex_SphMarkers, 256, nBlocks_numFlex_SphMarkers, nThreads_SphMarkers);
  // Use the shape function to find the generalized forces on FE nodes
  //  Real2 *FlexSPH_MeshPos_LRF_D, uint *FlexIdentifierD, Real4 *derivVelRhoD,

  Calc_Flex_FSI_ForcesD<<<nBlocks_numFlex_SphMarkers, nThreads_SphMarkers>>>(
      mR3CAST(fsiGeneralData->FlexSPH_MeshPos_LRF_D), U1CAST(fsiGeneralData->FlexIdentifierD),
      numObjectsH->numFlexBodies1D, U2CAST(fsiGeneralData->CableElementsNodes),
      U4CAST(fsiGeneralData->ShellElementsNodes), mR4CAST(fsiGeneralData->derivVelRhoD),
      mR3CAST(fsiMeshD->pos_fsi_fea_D), mR3CAST(fsiGeneralData->Flex_FSI_ForcesD));
  cudaThreadSynchronize();
  cudaCheckError();
}
//--------------------------------------------------------------------------------------------------------------------------------
void ChBce::UpdateRigidMarkersPositionVelocity(SphMarkerDataD* sphMarkersD, FsiBodiesDataD* fsiBodiesD) {
  if (numObjectsH->numRigidBodies == 0) {
    return;
  }

  uint nBlocks_numRigid_SphMarkers;
  uint nThreads_SphMarkers;
  computeGridSize(numObjectsH->numRigid_SphMarkers, 256, nBlocks_numRigid_SphMarkers, nThreads_SphMarkers);

  // Arman: InitSystem has to be called before this lunch to set numObjectsD

  //################################################### update BCE markers
  // position
  //** "posRadD2"/"velMasD2" associated to BCE markers are updated based on new
  // rigid body (position,
  // orientation)/(velocity, angular velocity)
  UpdateRigidMarkersPositionVelocityD<<<nBlocks_numRigid_SphMarkers, nThreads_SphMarkers>>>(
      mR3CAST(sphMarkersD->posRadD), mR3CAST(sphMarkersD->velMasD), mR3CAST(fsiGeneralData->rigidSPH_MeshPos_LRF_D),
      U1CAST(fsiGeneralData->rigidIdentifierD), mR3CAST(fsiBodiesD->posRigid_fsiBodies_D),
      mR4CAST(fsiBodiesD->velMassRigid_fsiBodies_D), mR3CAST(fsiBodiesD->omegaVelLRF_fsiBodies_D),
      mR4CAST(fsiBodiesD->q_fsiBodies_D));
  cudaThreadSynchronize();
  cudaCheckError();
}
//--------------------------------------------------------------------------------------------------------------------------------
void ChBce::UpdateFlexMarkersPositionVelocity(SphMarkerDataD* sphMarkersD, FsiMeshDataD* fsiMeshD) {
  if ((numObjectsH->numFlexBodies1D + numObjectsH->numFlexBodies2D) == 0) {
    return;
  }

  uint nBlocks_numFlex_SphMarkers;
  uint nThreads_SphMarkers;
  printf("UpdateFlexMarkersPositionVelocity..\n");

  computeGridSize(numObjectsH->numFlex_SphMarkers, 256, nBlocks_numFlex_SphMarkers, nThreads_SphMarkers);
  UpdateFlexMarkersPositionVelocityAccD<<<nBlocks_numFlex_SphMarkers, nThreads_SphMarkers>>>(
      mR3CAST(sphMarkersD->posRadD), mR3CAST(fsiGeneralData->FlexSPH_MeshPos_LRF_D), mR3CAST(sphMarkersD->velMasD),
      U1CAST(fsiGeneralData->FlexIdentifierD), numObjectsH->numFlexBodies1D, U2CAST(fsiGeneralData->CableElementsNodes),
      U4CAST(fsiGeneralData->ShellElementsNodes), mR3CAST(fsiMeshD->pos_fsi_fea_D), mR3CAST(fsiMeshD->vel_fsi_fea_D),
      paramsH->HSML * paramsH->MULT_INITSPACE_Shells);
  cudaThreadSynchronize();
  cudaCheckError();
}

}  // end namespace fsi
}  // end namespace chrono
