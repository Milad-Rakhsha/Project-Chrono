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
// Base class for processing sph force in fsi system.//
// =============================================================================
#include "chrono_fsi/ChDeviceUtils.cuh"
#include "chrono_fsi/ChFsiForceParallel.cuh"
#include "chrono_fsi/ChSphGeneral.cuh"

//#include "chrono_fsi/custom_math.h"

#include <thrust/sort.h>
#include <thrust/extrema.h>

#include <cusp/csr_matrix.h>
#include <cusp/print.h>
#include <cusp/monitor.h>
//==========================================================================================================================================

#include <cusp/krylov/cg.h>
#include <cusp/krylov/bicgstab.h>

#include <cusp/krylov/gmres.h>

#include <cusp/krylov/cg_m.h>
#include <cusp/krylov/cr.h>
#include <cusp/krylov/bicg.h>
#include <cusp/precond/aggregation/smoothed_aggregation.h>
#include <cusp/precond/diagonal.h>
#include <cusp/relaxation/jacobi.h>
#include <cusp/multiply.h>
#include <cusp/precond/ainv.h>

namespace chrono {
namespace fsi {

//--------------------------------------------------------------------------------------------------------------------------------

//// double precision atomic add function
//__device__ double atomicAdd(double* address, double val) {
//  unsigned long long int* address_as_ull = (unsigned long long int*)address;
//
//  unsigned long long int old = *address_as_ull, assumed;
//
//  do {
//    assumed = old;
//    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
//  } while (assumed != old);
//
//  return __longlong_as_double(old);
//}

//__device__ void Shells_ShapeFunctions(Real& NA, Real& NB, Real& NC, Real& ND, Real x, Real y) {
//  NA = 0.25 * (1.0 - x) * (1.0 - y);
//  NB = 0.25 * (1.0 + x) * (1.0 - y);
//  NC = 0.25 * (1.0 + x) * (1.0 + y);
//  ND = 0.25 * (1.0 - x) * (1.0 + y);
//}
//--------------------------------------------------------------------------------------------------------------------------------
// collide a particle against all other particles in a given cell
__device__ Real3 deltaVShare(int3 gridPos,
                             uint index,
                             Real3 posRadA,
                             Real3 velMasA,
                             Real4 rhoPresMuA,
                             Real3* sortedPosRad,
                             Real3* sortedVelMas,
                             Real4* sortedRhoPreMu,
                             uint* cellStart,
                             uint* cellEnd) {
  uint gridHash = calcGridHash(gridPos);
  // get start of bucket for this cell
  Real3 deltaV = mR3(0.0f);

  uint startIndex = cellStart[gridHash];
  if (startIndex != 0xffffffff) {  // cell is not empty
    // iterate over particles in this cell
    uint endIndex = cellEnd[gridHash];

    for (uint j = startIndex; j < endIndex; j++) {
      if (j != index) {  // check not colliding with self
        Real3 posRadB = sortedPosRad[j];
        Real3 dist3 = Distance(posRadA, posRadB);
        Real d = length(dist3);
        if (d > RESOLUTION_LENGTH_MULT * paramsD.HSML)
          continue;
        Real4 rhoPresMuB = sortedRhoPreMu[j];
        if (rhoPresMuB.w > -.1)
          continue;  //# B must be fluid (A was checked originally and it is
                     // fluid at this point), accoring to
        // colagrossi (2003), the other phase (i.e. rigid) should not be
        // considered)
        Real multRho = 2.0f / (rhoPresMuA.x + rhoPresMuB.x);
        Real3 velMasB = sortedVelMas[j];
        deltaV += paramsD.markerMass * (velMasB - velMasA) * W3(d) * multRho;
      }
    }
  }
  return deltaV;
}
//--------------------------------------------------------------------------------------------------------------------------------
// modify pressure for body force
__device__ __inline__ void modifyPressure(Real4& rhoPresMuB, const Real3& dist3Alpha) {
  // body force in x direction
  rhoPresMuB.y = (dist3Alpha.x > 0.5 * paramsD.boxDims.x) ? (rhoPresMuB.y - paramsD.deltaPress.x) : rhoPresMuB.y;
  rhoPresMuB.y = (dist3Alpha.x < -0.5 * paramsD.boxDims.x) ? (rhoPresMuB.y + paramsD.deltaPress.x) : rhoPresMuB.y;
  // body force in x direction
  rhoPresMuB.y = (dist3Alpha.y > 0.5 * paramsD.boxDims.y) ? (rhoPresMuB.y - paramsD.deltaPress.y) : rhoPresMuB.y;
  rhoPresMuB.y = (dist3Alpha.y < -0.5 * paramsD.boxDims.y) ? (rhoPresMuB.y + paramsD.deltaPress.y) : rhoPresMuB.y;
  // body force in x direction
  rhoPresMuB.y = (dist3Alpha.z > 0.5 * paramsD.boxDims.z) ? (rhoPresMuB.y - paramsD.deltaPress.z) : rhoPresMuB.y;
  rhoPresMuB.y = (dist3Alpha.z < -0.5 * paramsD.boxDims.z) ? (rhoPresMuB.y + paramsD.deltaPress.z) : rhoPresMuB.y;
}
//--------------------------------------------------------------------------------------------------------------------------------
/**
 * @brief DifVelocityRho
 * @details  See SDKCollisionSystem.cuh
 */
__device__ inline Real4 DifVelocityRho(Real3& dist3,
                                       Real& d,
                                       Real3 posRadA,
                                       Real3 posRadB,
                                       Real3& velMasA,
                                       Real3& vel_XSPH_A,
                                       Real3& velMasB,
                                       Real3& vel_XSPH_B,
                                       Real4& rhoPresMuA,
                                       Real4& rhoPresMuB,
                                       Real multViscosity) {
  Real3 gradW = GradW(dist3);

  // Real vAB_Dot_rAB = dot(velMasA - velMasB, dist3);

  //	//*** Artificial viscosity type 1.1
  //	Real alpha = .001;
  //	Real c_ab = 10 * paramsD.v_Max; //Ma = .1;//sqrt(7.0f * 10000 /
  //((rhoPresMuA.x + rhoPresMuB.x) / 2.0f));
  //	//Real h = paramsD.HSML;
  //	Real rho = .5f * (rhoPresMuA.x + rhoPresMuB.x);
  //	Real nu = alpha * paramsD.HSML * c_ab / rho;

  //	//*** Artificial viscosity type 1.2
  //	Real nu = 22.8f * paramsD.mu0 / 2.0f / (rhoPresMuA.x * rhoPresMuB.x);
  //	Real3 derivV = -paramsD.markerMass * (
  //		rhoPresMuA.y / (rhoPresMuA.x * rhoPresMuA.x) + rhoPresMuB.y /
  //(rhoPresMuB.x * rhoPresMuB.x)
  //		- nu * vAB_Dot_rAB / ( d * d + paramsD.epsMinMarkersDis *
  // paramsD.HSML * paramsD.HSML )
  //		) * gradW;
  //	return mR4(derivV,
  //		rhoPresMuA.x * paramsD.markerMass / rhoPresMuB.x * dot(vel_XSPH_A -
  // vel_XSPH_B, gradW));

  //*** Artificial viscosity type 2
  Real rAB_Dot_GradW = dot(dist3, gradW);
  Real rAB_Dot_GradW_OverDist = rAB_Dot_GradW / (d * d + paramsD.epsMinMarkersDis * paramsD.HSML * paramsD.HSML);
  Real3 derivV = -paramsD.markerMass *
                     (rhoPresMuA.y / (rhoPresMuA.x * rhoPresMuA.x) + rhoPresMuB.y / (rhoPresMuB.x * rhoPresMuB.x)) *
                     gradW +
                 paramsD.markerMass * (8.0f * multViscosity) * paramsD.mu0 *
                     pow(rhoPresMuA.x + rhoPresMuB.x, Real(-2)) * rAB_Dot_GradW_OverDist * (velMasA - velMasB);
  Real derivRho = rhoPresMuA.x * paramsD.markerMass / rhoPresMuB.x * dot(vel_XSPH_A - vel_XSPH_B, gradW);
  //	Real zeta = 0;//.05;//.1;
  //	Real derivRho = rhoPresMuA.x * paramsD.markerMass * invrhoPresMuBx *
  //(dot(vel_XSPH_A - vel_XSPH_B, gradW)
  //			+ zeta * paramsD.HSML * (10 * paramsD.v_Max) * 2 * (rhoPresMuB.x
  /// rhoPresMuA.x - 1) *
  // rAB_Dot_GradW_OverDist
  //			);

  //--------------------------------
  // Ferrari Modification
  derivRho = paramsD.markerMass * dot(vel_XSPH_A - vel_XSPH_B, gradW);
  Real cA = FerrariCi(rhoPresMuA.x);
  Real cB = FerrariCi(rhoPresMuB.x);
  derivRho -= rAB_Dot_GradW / (d + paramsD.epsMinMarkersDis * paramsD.HSML) * max(cA, cB) / rhoPresMuB.x *
              (rhoPresMuB.x - rhoPresMuA.x);

  //--------------------------------
  return mR4(derivV, derivRho);

  //	//*** Artificial viscosity type 1.3
  //	Real rAB_Dot_GradW = dot(dist3, gradW);
  //	Real3 derivV = -paramsD.markerMass * (rhoPresMuA.y / (rhoPresMuA.x *
  // rhoPresMuA.x) + rhoPresMuB.y /
  //(rhoPresMuB.x *
  // rhoPresMuB.x)) * gradW
  //		+ paramsD.markerMass / (rhoPresMuA.x * rhoPresMuB.x) * 2.0f *
  // paramsD.mu0 * rAB_Dot_GradW / ( d * d +
  // paramsD.epsMinMarkersDis * paramsD.HSML * paramsD.HSML ) * (velMasA -
  // velMasB);
  //	return mR4(derivV,
  //		rhoPresMuA.x * paramsD.markerMass / rhoPresMuB.x * dot(vel_XSPH_A -
  // vel_XSPH_B, gradW));
}
//--------------------------------------------------------------------------------------------------------------------------------
// collide a particle against all other particles in a given cell
__device__ Real4 collideCell(int3 gridPos,
                             uint index,
                             Real3 posRadA,
                             Real3 velMasA,
                             Real3 vel_XSPH_A,
                             Real4 rhoPresMuA,
                             Real3* sortedPosRad,
                             Real3* sortedVelMas,
                             Real3* vel_XSPH_Sorted_D,
                             Real4* sortedRhoPreMu,
                             Real3* velMas_ModifiedBCE,
                             Real4* rhoPreMu_ModifiedBCE,
                             uint* gridMarkerIndex,
                             uint* cellStart,
                             uint* cellEnd) {
  uint gridHash = calcGridHash(gridPos);
  // get start of bucket for this cell
  Real4 derivVelRho = mR4(0);

  uint startIndex = cellStart[gridHash];
  if (startIndex == 0xffffffff) {  // cell is not empty
    return derivVelRho;
  }
  // iterate over particles in this cell
  uint endIndex = cellEnd[gridHash];

  for (uint j = startIndex; j < endIndex; j++) {
    if (j != index) {  // check not colliding with self
      Real3 posRadB = sortedPosRad[j];
      Real3 dist3Alpha = posRadA - posRadB;
      //			Real3 dist3 = Distance(posRadA, posRadB);
      Real3 dist3 = Modify_Local_PosB(posRadB, posRadA);
      Real d = length(dist3);
      if (d > RESOLUTION_LENGTH_MULT * paramsD.HSML)
        continue;

      Real4 rhoPresMuB = sortedRhoPreMu[j];
      //			// old version. When rigid-rigid contact used to
      // be handled within fluid
      //			if ((fabs(rhoPresMuB.w - rhoPresMuA.w) < .1)
      //					&& rhoPresMuA.w > -.1) {
      //				continue;
      //			}
      if (rhoPresMuA.w > -.1 && rhoPresMuB.w > -.1) {  // no rigid-rigid force
        continue;
      }

      modifyPressure(rhoPresMuB, dist3Alpha);
      Real3 velMasB = sortedVelMas[j];
      if (rhoPresMuB.w > -.1) {
        int bceIndexB = gridMarkerIndex[j] - (numObjectsD.numFluidMarkers);
        if (!(bceIndexB >= 0 && bceIndexB < numObjectsD.numBoundaryMarkers + numObjectsD.numRigid_SphMarkers)) {
          printf("Error! bceIndex out of bound, collideD !\n");
        }
        rhoPresMuB = rhoPreMu_ModifiedBCE[bceIndexB];
        velMasB = velMas_ModifiedBCE[bceIndexB];
      }
      Real multViscosit = 1;
      Real4 derivVelRhoAB = mR4(0.0f);
      Real3 vel_XSPH_B = vel_XSPH_Sorted_D[j];
      derivVelRhoAB = DifVelocityRho(dist3, d, posRadA, posRadB, velMasA, vel_XSPH_A, velMasB, vel_XSPH_B, rhoPresMuA,
                                     rhoPresMuB, multViscosit);
      derivVelRho += derivVelRhoAB;
    }
  }

  // ff1
  //	if (rhoPresMuA.w > 0) printf("force value %f %f %f\n", 1e20*derivV.x,
  // 1e20*derivV.y, 1e20*derivV.z);
  return derivVelRho;
}

//--------------------------------------------------------------------------------------------------------------------------------
__global__ void newVel_XSPH_D(Real3* vel_XSPH_Sorted_D,  // output: new velocity
                              Real3* sortedPosRad,       // input: sorted positions
                              Real3* sortedVelMas,       // input: sorted velocities
                              Real4* sortedRhoPreMu,
                              uint* gridMarkerIndex,  // input: sorted particle indices
                              uint* cellStart,
                              uint* cellEnd,
                              uint numAllMarkers,
                              volatile bool* isErrorD) {
  uint index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= numAllMarkers)
    return;

  // read particle data from sorted arrays

  Real4 rhoPreMuA = sortedRhoPreMu[index];
  Real3 velMasA = sortedVelMas[index];
  if (rhoPreMuA.w > -0.1) {  // v_XSPH is calculated only for fluid markers. Keep
                             // unchanged if not fluid.
    vel_XSPH_Sorted_D[index] = velMasA;
    return;
  }

  Real3 posRadA = sortedPosRad[index];
  Real3 deltaV = mR3(0);

  // get address in grid
  int3 gridPos = calcGridPos(posRadA);

  /// if (gridPos.x == paramsD.gridSize.x-1) printf("****aha %d %d\n",
  /// gridPos.x, paramsD.gridSize.x);

  // examine neighbouring cells
  for (int z = -1; z <= 1; z++) {
    for (int y = -1; y <= 1; y++) {
      for (int x = -1; x <= 1; x++) {
        int3 neighbourPos = gridPos + mI3(x, y, z);
        deltaV += deltaVShare(neighbourPos, index, posRadA, velMasA, rhoPreMuA, sortedPosRad, sortedVelMas,
                              sortedRhoPreMu, cellStart, cellEnd);
      }
    }
  }
  //   // write new velocity back to original unsorted location
  // sortedVel_XSPH[index] = velMasA + paramsD.EPS_XSPH * deltaV;

  // write new velocity back to original unsorted location
  // uint originalIndex = gridMarkerIndex[index];
  Real3 vXSPH = velMasA + paramsD.EPS_XSPH * deltaV;
  if (!(isfinite(vXSPH.x) && isfinite(vXSPH.y) && isfinite(vXSPH.z))) {
    printf(
        "Error! particle vXSPH is NAN: thrown from SDKCollisionSystem.cu, "
        "newVel_XSPH_D !\n");
    *isErrorD = true;
  }
  vel_XSPH_Sorted_D[index] = vXSPH;
}
//--------------------------------------------------------------------------------------------------------------------------------
__global__ void collideD(Real4* sortedDerivVelRho_fsi_D,  // output: new velocity
                         Real3* sortedPosRad,             // input: sorted positions
                         Real3* sortedVelMas,             // input: sorted velocities
                         Real3* vel_XSPH_Sorted_D,
                         Real4* sortedRhoPreMu,
                         Real3* velMas_ModifiedBCE,
                         Real4* rhoPreMu_ModifiedBCE,
                         uint* gridMarkerIndex,
                         uint* cellStart,
                         uint* cellEnd,
                         uint numAllMarkers,
                         volatile bool* isErrorD) {
  uint index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= numAllMarkers)
    return;

  // read particle data from sorted arrays
  Real3 posRadA = sortedPosRad[index];
  Real3 velMasA = sortedVelMas[index];
  Real4 rhoPreMuA = sortedRhoPreMu[index];

  // *** comment these couple of lines since we don't want the force on the
  // rigid (or boundary) be influenced by ADAMi
  // *** method since it would cause large forces. ADAMI method is used only to
  // calculate forces on the fluid markers
  // (A)
  // *** near the boundary or rigid (B).
  //	if (rhoPreMuA.w > -.1) {
  //		int bceIndex = gridMarkerIndex[index] -
  //(numObjectsD.numFluidMarkers);
  //		if (!(bceIndex >= 0 && bceIndex < numObjectsD.numBoundaryMarkers +
  // numObjectsD.numRigid_SphMarkers)) {
  //			printf("Error! bceIndex out of bound, collideD !\n");
  //			*isErrorD = true;
  //		}
  //		rhoPreMuA = rhoPreMu_ModifiedBCE[bceIndex];
  //		velMasA = velMas_ModifiedBCE[bceIndex];
  //	}

  //	uint originalIndex = gridMarkerIndex[index];
  Real3 vel_XSPH_A = vel_XSPH_Sorted_D[index];
  Real4 derivVelRho = sortedDerivVelRho_fsi_D[index];

  // get address in grid
  int3 gridPos = calcGridPos(posRadA);

  // examine neighbouring cells
  for (int x = -1; x <= 1; x++) {
    for (int y = -1; y <= 1; y++) {
      for (int z = -1; z <= 1; z++) {
        derivVelRho += collideCell(gridPos + mI3(x, y, z), index, posRadA, velMasA, vel_XSPH_A, rhoPreMuA, sortedPosRad,
                                   sortedVelMas, vel_XSPH_Sorted_D, sortedRhoPreMu, velMas_ModifiedBCE,
                                   rhoPreMu_ModifiedBCE, gridMarkerIndex, cellStart, cellEnd);
      }
    }
  }

  // write new velocity back to original unsorted location
  // *** let's tweak a little bit :)
  if (!(isfinite(derivVelRho.x) && isfinite(derivVelRho.y) && isfinite(derivVelRho.z))) {
    printf(
        "Error! particle derivVel is NAN: thrown from "
        "SDKCollisionSystem.cu, collideD !\n");
    *isErrorD = true;
  }
  if (!(isfinite(derivVelRho.w))) {
    printf(
        "Error! particle derivRho is NAN: thrown from "
        "SDKCollisionSystem.cu, collideD !\n");
    *isErrorD = true;
  }
  sortedDerivVelRho_fsi_D[index] = derivVelRho;
}
//--------------------------------------------------------------------------------------------------------------------------------

ChFsiForceParallel::ChFsiForceParallel(ChBce* otherBceWorker,
                                       SphMarkerDataD* otherSortedSphMarkersD,
                                       ProximityDataD* otherMarkersProximityD,
                                       FsiGeneralData* otherFsiGeneralData,
                                       SimParams* otherParamsH,
                                       NumberOfObjects* otherNumObjects)
    : bceWorker(otherBceWorker),
      sortedSphMarkersD(otherSortedSphMarkersD),
      markersProximityD(otherMarkersProximityD),
      fsiGeneralData(otherFsiGeneralData),
      paramsH(otherParamsH),
      numObjectsH(otherNumObjects) {
  fsiCollisionSystem = new ChCollisionSystemFsi(sortedSphMarkersD, markersProximityD, paramsH, numObjectsH);

  sphMarkersD = NULL;
}

//--------------------------------------------------------------------------------------------------------------------------------

void ChFsiForceParallel::Finalize() {
  cudaMemcpyToSymbolAsync(paramsD, paramsH, sizeof(SimParams));
  cudaMemcpyToSymbolAsync(numObjectsD, numObjectsH, sizeof(NumberOfObjects));
  vel_XSPH_Sorted_D.resize(numObjectsH->numAllMarkers);
  derivVelRhoD_Sorted_D.resize(numObjectsH->numAllMarkers);
  fsiCollisionSystem->Finalize();
}
//--------------------------------------------------------------------------------------------------------------------------------

ChFsiForceParallel::~ChFsiForceParallel() {
  delete fsiCollisionSystem;
}
//--------------------------------------------------------------------------------------------------------------------------------
// use invasive to avoid one extra copy. However, keep in mind that sorted is
// changed.
void ChFsiForceParallel::CopySortedToOriginal_Invasive_R3(thrust::device_vector<Real3>& original,
                                                          thrust::device_vector<Real3>& sorted,
                                                          const thrust::device_vector<uint>& gridMarkerIndex) {
  thrust::device_vector<uint> dummyMarkerIndex = gridMarkerIndex;
  thrust::sort_by_key(dummyMarkerIndex.begin(), dummyMarkerIndex.end(), sorted.begin());
  dummyMarkerIndex.clear();
  thrust::copy(sorted.begin(), sorted.end(), original.begin());
}
//--------------------------------------------------------------------------------------------------------------------------------
void ChFsiForceParallel::CopySortedToOriginal_NonInvasive_R3(thrust::device_vector<Real3>& original,
                                                             const thrust::device_vector<Real3>& sorted,
                                                             const thrust::device_vector<uint>& gridMarkerIndex) {
  thrust::device_vector<Real3> dummySorted = sorted;
  CopySortedToOriginal_Invasive_R3(original, dummySorted, gridMarkerIndex);
}
//--------------------------------------------------------------------------------------------------------------------------------
// use invasive to avoid one extra copy. However, keep in mind that sorted is
// changed.
void ChFsiForceParallel::CopySortedToOriginal_Invasive_R4(thrust::device_vector<Real4>& original,
                                                          thrust::device_vector<Real4>& sorted,
                                                          const thrust::device_vector<uint>& gridMarkerIndex) {
  thrust::device_vector<uint> dummyMarkerIndex = gridMarkerIndex;
  thrust::sort_by_key(dummyMarkerIndex.begin(), dummyMarkerIndex.end(), sorted.begin());
  dummyMarkerIndex.clear();
  thrust::copy(sorted.begin(), sorted.end(), original.begin());
}
//--------------------------------------------------------------------------------------------------------------------------------
void ChFsiForceParallel::CopySortedToOriginal_NonInvasive_R4(thrust::device_vector<Real4>& original,
                                                             thrust::device_vector<Real4>& sorted,
                                                             const thrust::device_vector<uint>& gridMarkerIndex) {
  thrust::device_vector<Real4> dummySorted = sorted;
  CopySortedToOriginal_Invasive_R4(original, dummySorted, gridMarkerIndex);
}

//--------------------------------------------------------------------------------------------------------------------------------

void ChFsiForceParallel::CalculateXSPH_velocity() {
  /* Calculate vel_XSPH */
  if (vel_XSPH_Sorted_D.size() != numObjectsH->numAllMarkers) {
    printf("vel_XSPH_Sorted_D.size() %d numObjectsH->numAllMarkers %d \n", vel_XSPH_Sorted_D.size(),
           numObjectsH->numAllMarkers);
    throw std::runtime_error(
        "Error! size error vel_XSPH_Sorted_D Thrown from "
        "CalculateXSPH_velocity!\n");
  }

  bool *isErrorH, *isErrorD;
  isErrorH = (bool*)malloc(sizeof(bool));
  cudaMalloc((void**)&isErrorD, sizeof(bool));
  *isErrorH = false;
  cudaMemcpy(isErrorD, isErrorH, sizeof(bool), cudaMemcpyHostToDevice);
  //------------------------------------------------------------------------
  /* thread per particle */
  uint numThreads, numBlocks;
  computeGridSize(numObjectsH->numAllMarkers, 64, numBlocks, numThreads);

  /* Execute the kernel */
  newVel_XSPH_D<<<numBlocks, numThreads>>>(
      mR3CAST(vel_XSPH_Sorted_D), mR3CAST(sortedSphMarkersD->posRadD), mR3CAST(sortedSphMarkersD->velMasD),
      mR4CAST(sortedSphMarkersD->rhoPresMuD), U1CAST(markersProximityD->gridMarkerIndexD),
      U1CAST(markersProximityD->cellStartD), U1CAST(markersProximityD->cellEndD), numObjectsH->numAllMarkers, isErrorD);

  cudaThreadSynchronize();
  cudaCheckError();
  //------------------------------------------------------------------------
  cudaMemcpy(isErrorH, isErrorD, sizeof(bool), cudaMemcpyDeviceToHost);
  if (*isErrorH == true) {
    throw std::runtime_error("Error! program crashed in  newVel_XSPH_D!\n");
  }
  cudaFree(isErrorD);
  free(isErrorH);
}

//--------------------------------------------------------------------------------------------------------------------------------

/**
 * @brief Wrapper function for collide
 * @details
 * 		See SDKCollisionSystem.cuh for informaton on collide
 */
void ChFsiForceParallel::collide(

    thrust::device_vector<Real4>& sortedDerivVelRho_fsi_D,
    thrust::device_vector<Real3>& sortedPosRad,
    thrust::device_vector<Real3>& sortedVelMas,
    thrust::device_vector<Real3>& vel_XSPH_Sorted_D,
    thrust::device_vector<Real4>& sortedRhoPreMu,
    thrust::device_vector<Real3>& velMas_ModifiedBCE,
    thrust::device_vector<Real4>& rhoPreMu_ModifiedBCE,

    thrust::device_vector<uint>& gridMarkerIndex,
    thrust::device_vector<uint>& cellStart,
    thrust::device_vector<uint>& cellEnd) {
  bool *isErrorH, *isErrorD;
  isErrorH = (bool*)malloc(sizeof(bool));
  cudaMalloc((void**)&isErrorD, sizeof(bool));
  *isErrorH = false;
  cudaMemcpy(isErrorD, isErrorH, sizeof(bool), cudaMemcpyHostToDevice);
  //------------------------------------------------------------------------
  // thread per particle
  uint numThreads, numBlocks;
  computeGridSize(numObjectsH->numAllMarkers, 64, numBlocks, numThreads);

  // execute the kernel
  collideD<<<numBlocks, numThreads>>>(mR4CAST(sortedDerivVelRho_fsi_D), mR3CAST(sortedPosRad), mR3CAST(sortedVelMas),
                                      mR3CAST(vel_XSPH_Sorted_D), mR4CAST(sortedRhoPreMu), mR3CAST(velMas_ModifiedBCE),
                                      mR4CAST(rhoPreMu_ModifiedBCE), U1CAST(gridMarkerIndex), U1CAST(cellStart),
                                      U1CAST(cellEnd), numObjectsH->numAllMarkers, isErrorD);

  cudaThreadSynchronize();
  cudaCheckError();
  //------------------------------------------------------------------------
  cudaMemcpy(isErrorH, isErrorD, sizeof(bool), cudaMemcpyDeviceToHost);
  if (*isErrorH == true) {
    throw std::runtime_error("Error! program crashed in  collideD!\n");
  }
  cudaFree(isErrorD);
  free(isErrorH);

  //					// unroll sorted index to have the location of original
  // particles in the sorted
  // arrays
  //					thrust::device_vector<uint> dummyIndex =
  // gridMarkerIndex;
  //					thrust::sort_by_key(dummyIndex.begin(),
  // dummyIndex.end(),
  //							derivVelRhoD.begin());
  //					dummyIndex.clear();
}
//--------------------------------------------------------------------------------------------------------------------------------

void ChFsiForceParallel::CollideWrapper() {
  thrust::device_vector<Real4> m_dSortedDerivVelRho_fsi_D(
      numObjectsH->numAllMarkers);  // Store Rho, Pressure, Mu of each particle
                                    // in the device memory
  thrust::fill(m_dSortedDerivVelRho_fsi_D.begin(), m_dSortedDerivVelRho_fsi_D.end(), mR4(0));

  collide(m_dSortedDerivVelRho_fsi_D, sortedSphMarkersD->posRadD, sortedSphMarkersD->velMasD, vel_XSPH_Sorted_D,
          sortedSphMarkersD->rhoPresMuD, bceWorker->velMas_ModifiedBCE, bceWorker->rhoPreMu_ModifiedBCE,
          markersProximityD->gridMarkerIndexD, markersProximityD->cellStartD, markersProximityD->cellEndD);

  CopySortedToOriginal_Invasive_R3(fsiGeneralData->vel_XSPH_D, vel_XSPH_Sorted_D, markersProximityD->gridMarkerIndexD);
  CopySortedToOriginal_Invasive_R4(fsiGeneralData->derivVelRhoD, m_dSortedDerivVelRho_fsi_D,
                                   markersProximityD->gridMarkerIndexD);

  m_dSortedDerivVelRho_fsi_D.clear();
  // vel_XSPH_Sorted_D.clear();
}
//--------------------------------------------------------------------------------------------------------------------------------
void ChFsiForceParallel::AddGravityToFluid() {
  // add gravity to fluid markers
  /* Add outside forces. Don't add gravity to rigids, BCE, and boundaries, it is
   * added in ChSystem */
  Real3 totalFluidBodyForce3 = paramsH->bodyForce3 + paramsH->gravity;
  thrust::device_vector<Real4> bodyForceD(numObjectsH->numAllMarkers);
  thrust::fill(bodyForceD.begin(), bodyForceD.end(), mR4(totalFluidBodyForce3));
  thrust::transform(fsiGeneralData->derivVelRhoD.begin() + fsiGeneralData->referenceArray[0].x,
                    fsiGeneralData->derivVelRhoD.begin() + fsiGeneralData->referenceArray[0].y, bodyForceD.begin(),
                    fsiGeneralData->derivVelRhoD.begin() + fsiGeneralData->referenceArray[0].x, thrust::plus<Real4>());
  bodyForceD.clear();
}
//--------------------------------------------------------------------------------------------------------------------------------

void ChFsiForceParallel::ForceSPH(SphMarkerDataD* otherSphMarkersD, FsiBodiesDataD* otherFsiBodiesD) {
  // Arman: Change this function by getting in the arrays of the current stage:
  // useful for RK2. array pointers need to
  // be private members
  sphMarkersD = otherSphMarkersD;

  fsiCollisionSystem->ArrangeData(sphMarkersD);
  bceWorker->ModifyBceVelocity(sphMarkersD, otherFsiBodiesD);
  CalculateXSPH_velocity();
  CollideWrapper();
  AddGravityToFluid();
}

//--------------------------------------------------------------------------------------------------------------------------------
__global__ void calcRho_kernel(Real3* sortedPosRad,  // input: sorted positionsmin(
                               Real* nonNormalRho,
                               uint* cellStart,
                               uint* cellEnd,
                               const int numAllMarkers,
                               const Real RHO_0,
                               const Real m_0,
                               volatile bool* isErrorD) {
  uint i_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (i_idx >= numAllMarkers) {
    return;
  }
  Real3 posRadA = sortedPosRad[i_idx];
  Real sum_mW = 0;
  // get address in grid
  int3 gridPos = calcGridPos(posRadA);
  //
  //  /// if (gridPos.x == paramsD.gridSize.x-1) printf("****aha %d %d\n", gridPos.x, paramsD.gridSize.x);
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
          uint endIndex = cellEnd[gridHash];
          for (uint j = startIndex; j < endIndex; j++) {
            Real3 posRadB = sortedPosRad[j];
            Real3 dist3 = Distance(posRadA, posRadB);
            Real d = length(dist3);
            if (d > RESOLUTION_LENGTH_MULT * paramsD.HSML)
              continue;
            Real Wd = m_0 * W3(d);
            sum_mW += Wd;
          }
        }
      }
    }
  }

  // Adding neighbor contribution is done!
  nonNormalRho[i_idx] = sum_mW;
}

//--------------------------------------------------------------------------------------------------------------------------------
__global__ void calcNormalizedRho_kernel(Real3* sortedPosRad,  // input: sorted positions
                                         Real3* sortedVelMas,
                                         Real4* sortedRhoPreMu,
                                         Real* nonNormalRho,
                                         Real* dxi_over_Vi,
                                         uint* cellStart,
                                         uint* cellEnd,
                                         const int numAllMarkers,
                                         const Real RHO_0,
                                         const Real m_0,
                                         Real IncompressibilityFactor,
                                         volatile bool* isErrorD) {
  uint i_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (i_idx >= numAllMarkers) {
    return;
  }
  Real3 posRadA = sortedPosRad[i_idx];
  dxi_over_Vi[i_idx] = 1e10;

  Real sum_mW = 0;
  Real sum_mW_over_Rho = 0;
  // get address in grid
  int3 gridPos = calcGridPos(posRadA);
  //
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
            Real3 posRadB = sortedPosRad[j];
            Real3 dist3 = Distance(posRadA, posRadB);
            Real3 dv3 = Distance(sortedVelMas[i_idx], sortedVelMas[j]);
            Real d = length(dist3);
            Real particle_particle_n_CFL = abs(dot(dv3, dist3)) / d;
            Real particle_particle = length(dv3);
            Real particle_n_CFL = abs(dot(sortedVelMas[i_idx], dist3)) / d;
            Real particle_CFL = length(sortedVelMas[i_idx]);

            if (i_idx != j)
              dxi_over_Vi[i_idx] = fminf(d / particle_CFL, dxi_over_Vi[i_idx]);

            if (d > RESOLUTION_LENGTH_MULT * paramsD.HSML)
              continue;
            sum_mW += m_0 * W3(d);
            sum_mW_over_Rho += m_0 * W3(d) / nonNormalRho[j];
          }
        }
      }
    }
  }

  if (sortedRhoPreMu[i_idx].x > RHO_0)
    IncompressibilityFactor = 1;

  //  sortedRhoPreMu[i_idx].x = (sum_mW / sum_mW_over_Rho - RHO_0) * IncompressibilityFactor + RHO_0;
  sortedRhoPreMu[i_idx].x = (sum_mW - RHO_0) * IncompressibilityFactor + RHO_0;

  if (sortedRhoPreMu[i_idx].x < EPSILON) {
    printf("My density is %f, index= %d\n", sortedRhoPreMu[i_idx].x, i_idx);

    printf("My position = [%f %f %f]\n", sortedPosRad[i_idx].x, sortedPosRad[i_idx].y, sortedPosRad[i_idx].z);

    *isErrorD = true;
    return;
  }

  //  if (sortedRhoPreMu[i_idx].w > -1) {
  //    sortedRhoPreMu[i_idx].x = RHO_0;
  //  }
}
//--------------------------------------------------------------------------------------------------------------------------------
__global__ void V_i_np__AND__d_ii_kernel(Real3* sortedPosRad,  // input: sorted positions
                                         Real3* sortedVelMas,
                                         Real4* sortedRhoPreMu,
                                         Real3* d_ii,
                                         Real3* V_i_np,
                                         uint* cellStart,
                                         uint* cellEnd,
                                         const int numAllMarkers,
                                         const Real m_0,
                                         const Real mu_0,
                                         const Real RHO_0,
                                         const Real epsilon,
                                         const Real dT,
                                         const Real3 gravity,
                                         volatile bool* isErrorD) {
  uint i_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (i_idx >= numAllMarkers) {
    return;
  }
  if (sortedRhoPreMu[i_idx].x < EPSILON) {
    printf("My density is %f,ref density= %f\n", sortedRhoPreMu[i_idx].x, RHO_0);
  }
  Real3 posi = sortedPosRad[i_idx];
  Real3 Veli = sortedVelMas[i_idx];
  Real Rhoi = sortedRhoPreMu[i_idx].x;
  Real3 My_d_ii = mR3(0);
  Real3 My_F_i_np = mR3(0);

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
            if (i_idx == j)
              continue;
            Real3 posj = sortedPosRad[j];
            Real3 Velj = sortedVelMas[j];
            Real Rhoj = sortedRhoPreMu[j].x;
            if (Rhoj == 0) {
              printf("Bug F_i_np__AND__d_ii_kernel i=%d j=%d\n", i_idx, j);
            }
            Real3 dist3 = Distance(posi, posj);
            Real d = length(dist3);
            ////CHECK THIS CONDITION!!!
            if (d > RESOLUTION_LENGTH_MULT * paramsD.HSML)
              continue;
            Real3 grad_i_wij = GradW(dist3);
            My_d_ii += m_0 * (-(dT * dT) / (Rhoi * Rhoi)) * grad_i_wij;
            Real Rho_bar = (Rhoj + Rhoi) * 0.5;
            Real3 V_ij = (Veli - Velj);
            Real3 muNumerator = 2 * mu_0 * dot(dist3, grad_i_wij) * V_ij;
            Real muDenominator = (Rho_bar * Rho_bar) * (d * d + paramsD.HSML * paramsD.HSML * epsilon);
            My_F_i_np += m_0 * muNumerator / muDenominator;
          }
        }
      }
    }
  }
  My_F_i_np += m_0 * gravity;
  d_ii[i_idx] = My_d_ii;
  V_i_np[i_idx] = (My_F_i_np * dT + Veli);  // This does not contain m_0?
}
//--------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------
__global__ void Rho_np_AND_a_ii(Real3* sortedPosRad,
                                Real4* sortedRhoPreMu,
                                Real* rho_np,  // Write
                                Real* a_ii,    // Write
                                Real* p_old,   // Write
                                Real3* V_np,   // Read
                                Real3* d_ii,   // Read
                                uint* cellStart,
                                uint* cellEnd,
                                const int numAllMarkers,
                                const Real m_0,
                                const Real dT,
                                volatile bool* isErrorD) {
  uint i_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (i_idx >= numAllMarkers) {
    return;
  }
  if (sortedRhoPreMu[i_idx].x < EPSILON) {
    printf("My density is %f,ref density= %f\n", sortedRhoPreMu[i_idx].x, 1000);
  }
  Real3 posi = sortedPosRad[i_idx];
  Real3 Veli_np = V_np[i_idx];
  Real Rho_i = sortedRhoPreMu[i_idx].x;
  Real3 my_d_ii = d_ii[i_idx];
  Real rho_temp = 0;
  Real my_a_ii = 0;
  // get address in grid
  int3 gridPos = calcGridPos(posi);

  //  /// if (gridPos.x == paramsD.gridSize.x-1) printf("****aha %d %d\n", gridPos.x, paramsD.gridSize.x);
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
            Real3 posj = sortedPosRad[j];
            Real3 dist3 = Distance(posi, posj);
            Real d = length(dist3);
            if (d > RESOLUTION_LENGTH_MULT * paramsD.HSML || i_idx == j)
              continue;
            Real3 Velj_np = V_np[j];
            Real3 grad_i_wij = GradW(dist3);
            rho_temp += m_0 * dot((Veli_np - Velj_np), grad_i_wij);
            Real3 d_ji = m_0 * (-(dT * dT) / (Rho_i * Rho_i)) * (-grad_i_wij);
            my_a_ii += m_0 * dot((my_d_ii - d_ji), grad_i_wij);
          }
        }
      }
    }
  }
  rho_np[i_idx] = dT * rho_temp + sortedRhoPreMu[i_idx].x;
  a_ii[i_idx] = my_a_ii;
  //  sortedRhoPreMu[i_idx].y = 1000;  // Note that this is outside of the for loop
}
//--------------------------------------------------------------------------------------------------------------------------------

__global__ void Calc_dij_pj(Real3* dij_pj,  // write
                            Real3* F_p,     // Write
                            Real3* d_ii,    // Read
                            Real3* sortedPosRad,
                            Real3* sortedVelMas,
                            Real4* sortedRhoPreMu,
                            Real* p_old,
                            uint* cellStart,
                            uint* cellEnd,
                            const int numAllMarkers,
                            const Real m_0,
                            const Real dT,
                            volatile bool* isErrorD) {
  uint i_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (i_idx >= numAllMarkers) {
    return;
  }
  Real3 my_F_p = mR3(0);
  Real p_i_old = p_old[i_idx];
  Real3 pos_i = sortedPosRad[i_idx];
  Real Rho_i = sortedRhoPreMu[i_idx].x;
  if (sortedRhoPreMu[i_idx].x < EPSILON) {
    printf("My density is %f in Calc_dij_pj\n", sortedRhoPreMu[i_idx].x);
  }

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
            Real3 pos_j = sortedPosRad[j];
            Real3 dist3 = Distance(pos_i, pos_j);
            Real d = length(dist3);
            ////CHECK THIS CONDITION!!!
            if (d > RESOLUTION_LENGTH_MULT * paramsD.HSML || i_idx == j)
              continue;
            Real3 grad_i_wij = GradW(dist3);
            Real Rho_j = sortedRhoPreMu[j].x;
            Real p_j_old = p_old[j];
            My_dij_pj += m_0 * (-(dT * dT) / (Rho_j * Rho_j)) * grad_i_wij * p_j_old;
            my_F_p += -m_0 * m_0 * ((p_i_old / (Rho_i * Rho_i)) + (p_j_old / (Rho_j * Rho_j))) * grad_i_wij;
          }
        }
      }
    }
  }
  dij_pj[i_idx] = My_dij_pj;
  F_p[i_idx] = my_F_p;
}

////--------------------------------------------------------------------------------------------------------------------------------
__global__ void CalcNumber_Contacts(unsigned long int* numContacts,
                                    Real3* sortedPosRad,
                                    Real4* sortedRhoPreMu,
                                    uint* cellStart,
                                    uint* cellEnd,
                                    const int numAllMarkers,
                                    volatile bool* isErrorD) {
  uint i_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (i_idx >= numAllMarkers) {
    return;
  }
  int myType = sortedRhoPreMu[i_idx].w;
  Real3 pos_i = sortedPosRad[i_idx];
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
        if (startIndex != 0xffffffff) {  // cell is not empty
          // iterate over particles in this cell
          uint endIndex = cellEnd[gridHash];
          for (uint j = startIndex; j < endIndex; j++) {
            Real3 pos_j = sortedPosRad[j];
            Real3 dist3 = Distance(pos_i, pos_j);
            Real d = length(dist3);
            if (d > RESOLUTION_LENGTH_MULT * paramsD.HSML || i_idx == j)
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
              if (myType == 0 && sortedRhoPreMu[j].w == 0)  // Do not count BCE-BCE interactions...
                counter--;
            }
            if (myType > -1)  // For BCE no need to go deeper than this...
              continue;
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
                      Real3 pos_k = sortedPosRad[k];
                      Real3 dist3jk = Distance(pos_j, pos_k);
                      Real djk = length(dist3jk);
                      if (djk > RESOLUTION_LENGTH_MULT * paramsD.HSML || k == j || k == i_idx)
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
                               Real3* sortedPosRad,
                               Real4* sortedRhoPreMu,
                               uint* cellStart,
                               uint* cellEnd,
                               const int numAllMarkers,
                               const Real m_0,
                               const Real dT,
                               volatile bool* isErrorD) {
  uint i_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (i_idx >= numAllMarkers) {
    return;
  }
  Real3 pos_i = sortedPosRad[i_idx];
  if (sortedRhoPreMu[i_idx].x < EPSILON) {
    printf("My density is %f in Calc_dij\n", sortedRhoPreMu[i_idx].x);
  }

  Real3 My_dij = mR3(0);
  Real3 My_summgradW = mR3(0);

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
            Real3 pos_j = sortedPosRad[j];
            Real3 dist3 = Distance(pos_i, pos_j);
            Real d = length(dist3);
            ////CHECK THIS CONDITION!!!
            if (d > RESOLUTION_LENGTH_MULT * paramsD.HSML || i_idx == j)
              continue;
            Real3 grad_i_wij = GradW(dist3);
            Real Rho_j = sortedRhoPreMu[j].x;
            My_dij += m_0 * (-(dT * dT) / (Rho_j * Rho_j)) * grad_i_wij;
            My_summgradW = m_0 * grad_i_wij;
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
                               unsigned long int* numContacts,
                               ///> The above 4 vectors are used for CSR form.
                               Real* a_ii,  // write
                               Real* B_i,
                               const Real3* sortedPosRad,
                               const Real3* sortedVelMas,
                               const Real4* sortedRhoPreMu,
                               Real3* V_new,
                               Real* p_old,

                               Real3* bceAcc,
                               Real4* velMassRigid_fsiBodies_D,
                               Real3* accRigid_fsiBodies_D,
                               uint* rigidIdentifierD,

                               Real3* pos_fsi_fea_D,
                               Real3* vel_fsi_fea_D,
                               Real3* acc_fsi_fea_D,
                               uint* FlexIdentifierD,
                               uint4* ShellelementsNodes,

                               int4 updatePortion,
                               uint* gridMarkerIndexD,
                               const uint* cellStart,
                               const uint* cellEnd,
                               const int numAllMarkers,
                               const Real m_0,
                               const Real3 gravity,
                               bool IsSPARSE) {
  uint csrStartIdx = numContacts[i_idx] + 1;
  uint csrEndIdx = numContacts[i_idx + 1];

  //  if (bceIndex >= numObjectsD.numRigid_SphMarkers) {
  //    return;
  //  }

  int Original_idx = gridMarkerIndexD[i_idx];
  Real3 myAcc;
  Real3 V_prescribed;
  Real MASS;

  // See if this belongs to boundary
  if (Original_idx >= updatePortion.x && Original_idx < updatePortion.y) {
    myAcc = mR3(0.0);
    V_prescribed = mR3(0.0);
    MASS = m_0;
    // Or not maybe Rigid bodies
  } else if (Original_idx >= updatePortion.y && Original_idx < updatePortion.z) {
    int rigidIndex = rigidIdentifierD[Original_idx - updatePortion.y];
    V_prescribed = mR3(velMassRigid_fsiBodies_D[rigidIndex].x, velMassRigid_fsiBodies_D[rigidIndex].y,
                       velMassRigid_fsiBodies_D[rigidIndex].z);
    myAcc =
        mR3(accRigid_fsiBodies_D[rigidIndex].x, accRigid_fsiBodies_D[rigidIndex].y, accRigid_fsiBodies_D[rigidIndex].z);

    MASS = velMassRigid_fsiBodies_D[rigidIndex].w;
    // Or not, Flexible bodies for sure
  } else if (Original_idx >= updatePortion.z && Original_idx < updatePortion.w) {
    int FlexIndex = FlexIdentifierD[Original_idx - updatePortion.z];
    //    printf("My FlexIndex is %d \n", FlexIndex);
    int nA = ShellelementsNodes[FlexIndex].x;
    int nB = ShellelementsNodes[FlexIndex].y;
    int nC = ShellelementsNodes[FlexIndex].z;
    int nD = ShellelementsNodes[FlexIndex].w;

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
    Real3 dist3 = sortedPosRad[Original_idx] - Shell_center;
    Real Shell_x = 0.25 * (length(pos_fsi_fea_D_nB - pos_fsi_fea_D_nA) + length(pos_fsi_fea_D_nD - pos_fsi_fea_D_nC));
    Real Shell_y = 0.25 * (length(pos_fsi_fea_D_nD - pos_fsi_fea_D_nA) + length(pos_fsi_fea_D_nC - pos_fsi_fea_D_nB));
    Real2 FlexSPH_MeshPos_Natural = mR2(dist3.x / Shell_x, dist3.y / Shell_y);

    Real4 N_shell = Shells_ShapeFunctions(FlexSPH_MeshPos_Natural.x, FlexSPH_MeshPos_Natural.y);
    Real NA = N_shell.x;
    Real NB = N_shell.y;
    Real NC = N_shell.z;
    Real ND = N_shell.w;
    V_prescribed = NA * vel_fsi_fea_D_nA + NB * vel_fsi_fea_D_nB + NC * vel_fsi_fea_D_nC + ND * vel_fsi_fea_D_nD;
    myAcc = NA * acc_fsi_fea_D_nA + NB * acc_fsi_fea_D_nB + NC * acc_fsi_fea_D_nC + ND * acc_fsi_fea_D_nD;

  } else {
    printf("i_idx=%d, Original_idx:%d was not found\n\n", i_idx, Original_idx);
  }

  for (int c = csrStartIdx - 1; c < csrEndIdx; c++) {
    csrValA[c] = 0;
    csrColIndA[c] = i_idx;
    GlobalcsrColIndA[c] = i_idx + numAllMarkers * i_idx;
  }

  // if ((csrEndIdx - csrStartIdx) != uint(0)) {
  Real3 numeratorv = mR3(0);
  Real denumenator = 0;
  Real pRHS = 0;
  Real Rho_i = sortedRhoPreMu[i_idx].x;
  Real3 pos_i = sortedPosRad[i_idx];
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
            Real3 pos_j = sortedPosRad[j];
            Real3 dist3 = Distance(pos_i, pos_j);
            Real d = length(dist3);
            if (d > RESOLUTION_LENGTH_MULT * paramsD.HSML || sortedRhoPreMu[j].w != -1)
              continue;
            Real3 Vel_j = sortedVelMas[j];
            Real Wd = W3(d);
            numeratorv += Vel_j * Wd;
            pRHS += dot(gravity - myAcc, dist3) * Rho_i * Wd;
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

  if (abs(denumenator) < EPSILON) {
    V_new[i_idx] = 2 * V_prescribed;
    B_i[i_idx] = 0;
    csrValA[csrStartIdx - 1] = 1;
    csrColIndA[csrStartIdx - 1] = i_idx;
    GlobalcsrColIndA[csrStartIdx - 1] = i_idx + numAllMarkers * i_idx;
  } else {
    V_new[i_idx] = 2 * V_prescribed - numeratorv / denumenator;
    B_i[i_idx] = pRHS;
    csrValA[csrStartIdx - 1] = denumenator;
    csrColIndA[csrStartIdx - 1] = i_idx;
    GlobalcsrColIndA[csrStartIdx - 1] = i_idx + numAllMarkers * i_idx;
  }
}
//--------------------------------------------------------------------------------------------------------------------------------

////--------------------------------------------------------------------------------------------------------------------------------
__device__ void Calc_fluid_aij_Bi(const uint i_idx,
                                  Real* csrValA,
                                  uint* csrColIndA,
                                  unsigned long int* GlobalcsrColIndA,
                                  unsigned long int* numContacts,
                                  ///> The above 4 vectors are used for CSR form.
                                  Real* B_i,
                                  Real3* d_ii,   // Read
                                  Real* a_ii,    // Read
                                  Real* rho_np,  // Read
                                  Real3* summGradW,
                                  Real3* sortedPosRad,
                                  Real4* sortedRhoPreMu,
                                  uint* cellStart,
                                  uint* cellEnd,
                                  const int numAllMarkers,
                                  const Real m_0,
                                  const Real RHO_0,
                                  const Real dT,
                                  bool IsSPARSE) {
  Real3 pos_i = sortedPosRad[i_idx];
  if (sortedRhoPreMu[i_idx].x < EPSILON) {
    printf("My density is %f in Calc_dij_pj\n", sortedRhoPreMu[i_idx].x);
  }

  //   printf("%d is fluid\n", i_idx);

  int counter = 0;  // There is always one non-zero at each row- The marker itself
  B_i[i_idx] = RHO_0 - rho_np[i_idx];

  //  printf("%d is fluid\n", i_idx);

  uint csrStartIdx = numContacts[i_idx] + 1;  // Reserve the starting index for the A_ii
  uint csrEndIdx = numContacts[i_idx + 1];

  if (IsSPARSE) {
    for (int c = csrStartIdx; c < csrEndIdx; c++) {
      csrValA[c] = 0;
      csrColIndA[c] = i_idx;
      GlobalcsrColIndA[c] = i_idx + numAllMarkers * i_idx;
    }
  }
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
            Real3 pos_j = sortedPosRad[j];
            Real3 dist3 = Distance(pos_i, pos_j);
            Real d = length(dist3);
            if (d > RESOLUTION_LENGTH_MULT * paramsD.HSML || i_idx == j)
              continue;
            Real3 grad_i_wij = GradW(dist3);
            Real Rho_j = sortedRhoPreMu[j].x;
            Real3 d_ij = m_0 * (-(dT * dT) / (Rho_j * Rho_j)) * grad_i_wij;
            Real My_a_ij_1 = dot(d_ij, summGradW[i_idx]);
            Real My_a_ij_2 = m_0 * dot(d_ii[j], grad_i_wij);
            float My_a_ij_12 = (float)My_a_ij_1 - (float)My_a_ij_2;
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
                      Real3 pos_k = sortedPosRad[k];
                      Real3 dist3jk = Distance(pos_j, pos_k);
                      Real djk = length(dist3jk);
                      if (djk > RESOLUTION_LENGTH_MULT * paramsD.HSML || k == j || k == i_idx)
                        continue;
                      Real Rho_k = sortedRhoPreMu[k].x;
                      Real3 grad_j_wjk = GradW(dist3jk);
                      Real3 d_jk = m_0 * (-(dT * dT) / (Rho_k * Rho_k)) * grad_j_wjk;
                      float My_a_ij_3 = dot(d_jk, m_0 * grad_i_wij);
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

  csrValA[csrStartIdx - 1] = a_ii[i_idx];
  csrColIndA[csrStartIdx - 1] = i_idx;
  GlobalcsrColIndA[csrStartIdx - 1] = i_idx + numAllMarkers * i_idx;
}

//--------------------------------------------------------------------------------------------------------------------------------
__global__ void FormAXB(Real* csrValA,
                        uint* csrColIndA,
                        unsigned long int* GlobalcsrColIndA,
                        unsigned long int* numContacts,
                        ///> The above 4 vectors are used for CSR form.
                        Real* a_ij,   // write
                        Real* a_ij3,  // write
                        Real* B_i,    // write
                        Real3* d_ii,  // Read
                        Real* a_ii,   // Read
                        Real3* summGradW,
                        Real3* sortedPosRad,
                        Real3* sortedVelMas,
                        Real4* sortedRhoPreMu,
                        Real3* V_new,
                        Real* p_old,
                        Real* rho_np,

                        Real3* bceAcc,
                        Real4* velMassRigid_fsiBodies_D,
                        Real3* accRigid_fsiBodies_D,
                        uint* rigidIdentifierD,

                        Real3* pos_fsi_fea_D,
                        Real3* vel_fsi_fea_D,
                        Real3* acc_fsi_fea_D,
                        uint* FlexIdentifierD,
                        uint4* ShellelementsNodes,

                        int4 updatePortion,
                        uint* gridMarkerIndexD,
                        uint* cellStart,
                        uint* cellEnd,
                        const int numAllMarkers,
                        const Real m_0,
                        const Real RHO_0,
                        const Real dT,
                        const Real3 gravity,
                        bool IsSPARSE,
                        volatile bool* isError) {
  uint i_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (i_idx >= numAllMarkers) {
    return;
  }
  int TYPE_OF_NARKER = sortedRhoPreMu[i_idx].w;
  if (TYPE_OF_NARKER == -1)
    Calc_fluid_aij_Bi(i_idx, csrValA, csrColIndA, GlobalcsrColIndA, numContacts, B_i, d_ii, a_ii, rho_np, summGradW,
                      sortedPosRad, sortedRhoPreMu, cellStart, cellEnd, numAllMarkers, m_0, RHO_0, dT, true);
  else
    Calc_BC_aij_Bi(i_idx, csrValA, csrColIndA, GlobalcsrColIndA, numContacts, a_ii, B_i, sortedPosRad, sortedVelMas,

                   sortedRhoPreMu, V_new, p_old,

                   bceAcc, velMassRigid_fsiBodies_D, accRigid_fsiBodies_D, rigidIdentifierD,

                   pos_fsi_fea_D, vel_fsi_fea_D, acc_fsi_fea_D, FlexIdentifierD, ShellelementsNodes,

                   updatePortion, gridMarkerIndexD, cellStart, cellEnd, numAllMarkers, m_0, gravity, true);
}

//--------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------
__global__ void Calc_Pressure_AXB_USING_CSR(Real* csrValA,
                                            Real* a_ii,
                                            uint* csrColIndA,
                                            unsigned long int* numContacts,
                                            Real4* sortedRhoPreMu,
                                            Real* nonNormalRho,
                                            Real3* sortedVelMas,
                                            Real3* V_new,
                                            Real* p_old,
                                            Real* B_i,  // Read
                                            Real RHO_0,
                                            const int numAllMarkers,
                                            bool ClampPressure,
                                            volatile bool* isErrorD) {
  uint i_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (i_idx >= numAllMarkers)
    return;

  uint startIdx = numContacts[i_idx] + 1;  // numContacts[i_idx] is the diagonal itself
  uint endIdx = numContacts[i_idx + 1];

  if (sortedRhoPreMu[i_idx].x < EPSILON) {
    printf("My density is %f in Calc_Pressure_AXB\n", sortedRhoPreMu[i_idx].x);
  }

  Real aij_pj = 0;
  if ((sortedRhoPreMu[i_idx].x < 0.950 * RHO_0)) {
    sortedRhoPreMu[i_idx].y = 0.0;
    //    sortedRhoPreMu[i_idx].x = RHO_0;
  } else {
    for (int myIdx = startIdx; myIdx < endIdx; myIdx++) {
      if (i_idx == csrColIndA[myIdx])
        continue;
      aij_pj += csrValA[myIdx] * p_old[csrColIndA[myIdx]];
      if (!isfinite(aij_pj)) {
        printf("csrColIndA[myIdx]=%d, myIdx=%d\ncsrValA[myIdx]=%f, p_old[csrColIndA[myIdx]=%f\n", csrColIndA[myIdx],
               myIdx, csrValA[myIdx], p_old[csrColIndA[myIdx]]);
      }
    }
    Real RHS = fminf(0.0, B_i[i_idx]);
    //    Real RHS = B_i[i_idx];

    sortedRhoPreMu[i_idx].y = (RHS - aij_pj) / csrValA[startIdx - 1];
  }

  if (ClampPressure && sortedRhoPreMu[i_idx].y < 0.0)
    sortedRhoPreMu[i_idx].y = 0.0;
  /// This updates the velocity but it is done here since its faster
  if (sortedRhoPreMu[i_idx].w > -1) {
    sortedVelMas[i_idx] = V_new[i_idx];
  }
}
//--------------------------------------------------------------------------------------------------------------------------------
__global__ void PrepareForCusp(Real* csrValA,
                               uint* csrColIndA,
                               unsigned long int* numContacts,
                               Real4* sortedRhoPreMu,
                               Real* B_i,  // Read
                               Real RHO_0,
                               const int numAllMarkers,
                               volatile bool* isErrorD) {
  uint i_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (i_idx >= numAllMarkers)
    return;
  uint startIdx = numContacts[i_idx] + 1;  // numContacts[i_idx] is the diagonal itself
  uint endIdx = numContacts[i_idx + 1];

  if ((sortedRhoPreMu[i_idx].x < 0.998 * RHO_0) && (sortedRhoPreMu[i_idx].w == -1)) {
    sortedRhoPreMu[i_idx].y = 0.0;

    for (int myIdx = startIdx; myIdx < endIdx; myIdx++) {
      if (i_idx == csrColIndA[myIdx])
        csrValA[myIdx] = 1e5;
    }
  } else {
    Real RHS = fminf(0.0, B_i[i_idx]);
    B_i[i_idx] = RHS;
  }
}

////--------------------------------------------------------------------------------------------------------------------------------
////--------------------------------------------------------------------------------------------------------------------------------
__global__ void Calc_Pressure(Real* a_ii,     // Read
                              Real3* d_ii,    // Read
                              Real3* dij_pj,  // Read
                              Real* rho_np,   // Read
                              Real* rho_p,    // Write
                              Real3* F_p,
                              Real3* sortedPosRad,
                              Real3* sortedVelMas,
                              Real4* sortedRhoPreMu,
                              Real* p_old,
                              Real3* V_new,
                              uint* cellStart,
                              uint* cellEnd,
                              const int numAllMarkers,
                              const Real m_0,
                              const Real RHO_0,
                              const Real dT,
                              const Real3 gravity,
                              volatile bool* isErrorD) {
  uint i_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (i_idx >= numAllMarkers)
    return;

  if (sortedRhoPreMu[i_idx].x < EPSILON) {
    printf("My density is %f in Calc_Pressure\n", sortedRhoPreMu[i_idx].x);
  }
  int myType = sortedRhoPreMu[i_idx].w;
  Real Rho_i = sortedRhoPreMu[i_idx].x;
  Real p_i = p_old[i_idx];
  Real3 pos_i = sortedPosRad[i_idx];
  Real p_new = 0;
  Real my_rho_p = 0;
  Real3 F_i_p = F_p[i_idx];

  if (myType < 0) {
    if (Rho_i < 0.998 * RHO_0) {
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
                Real3 pos_j = sortedPosRad[j];
                Real3 dist3ij = Distance(pos_i, pos_j);
                Real dij = length(dist3ij);
                if (dij > RESOLUTION_LENGTH_MULT * paramsD.HSML || i_idx == j)
                  continue;
                //                Real Rho_j = sortedRhoPreMu[j].x;
                Real p_j_old = p_old[j];
                Real3 djj = d_ii[j];
                Real3 F_j_p = F_p[j];
                Real3 grad_i_wij = GradW(dist3ij);
                Real3 d_ji = m_0 * (-(dT * dT) / (Rho_i * Rho_i)) * (-grad_i_wij);
                Real3 djk_pk = dij_pj[j] - d_ji * p_i;
                sum_dij_pj += m_0 * dot(my_dij_pj, grad_i_wij);
                sum_djj_pj += m_0 * dot(djj, grad_i_wij) * p_j_old;
                sum_djk_pk += m_0 * dot(djk_pk, grad_i_wij);
                my_rho_p += (dT * dT) * m_0 * dot((F_i_p / m_0 - F_j_p / m_0), grad_i_wij);
              }
            }
          }
        }
      }

      Real RHS = fminf(0.0, RHO_0 - rho_np[i_idx]);
      Real aij_pj = +sum_dij_pj - sum_djj_pj - sum_djk_pk;
      p_new = (RHS - aij_pj) / a_ii[i_idx];
      //      sortedRhoPreMu[i_idx].x = aij_pj + p_new * a_ii[i_idx] + RHO_0 - RHS;
    }
  } else {  // Do Adami BC

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
              Real3 pos_j = sortedPosRad[j];
              Real3 dist3 = Distance(pos_i, pos_j);
              Real d = length(dist3);
              if (d > RESOLUTION_LENGTH_MULT * paramsD.HSML || sortedRhoPreMu[j].w != -1)
                continue;
              // OLD VELOCITY IS SHOULD BE OBDATED NOT THE NEW ONE!!!!!
              Real3 Vel_j = sortedVelMas[j];
              Real p_j = p_old[j];
              Real3 F_j_p = F_p[j];

              Real Wd = W3(d);
              numeratorv += Vel_j * Wd;
              numeratorp += p_j * Wd + dot(gravity, dist3) * Rho_i * Wd;
              denumenator += Wd;
              Real3 TobeUsed = (F_i_p / m_0 - F_j_p / m_0);
              my_rho_p += (dT * dT) * m_0 * dot(TobeUsed, GradW(dist3));

              if (isnan(numeratorp))
                printf("Something is wrong here..., %f\n", numeratorp);
            }
          }
        }
      }
    }
    if (abs(denumenator) < EPSILON) {
      p_new = 0;
      Vel_i = mR3(0);

    } else {
      Vel_i = -numeratorv / denumenator;
      p_new = numeratorp / denumenator;

      if (isnan(denumenator) || isnan(numeratorp))
        printf("I cheated, something is wrong though ...\n");
    }
    V_new[i_idx] = Vel_i;
  }
  rho_p[i_idx] = my_rho_p;
  sortedRhoPreMu[i_idx].y = p_new;
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

  p_old[i_idx] = sortedRhoPreMu[i_idx].y;  // This needs consistency p_old is old but v_new is new !!
  if (sortedRhoPreMu[i_idx].w > -1) {
    sortedVelMas[i_idx] = V_new[i_idx];
  }
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
  if (sortedRhoPreMu[i_idx].x < EPSILON) {
    printf("My density is %f in Update_AND_Calc_Res\n", sortedRhoPreMu[i_idx].x);
  }

  //  p_i = (1 - relax) * p_old_i + relax * p_i;
  sortedRhoPreMu[i_idx].y = (1 - params_relaxation) * p_old[i_idx] + params_relaxation * sortedRhoPreMu[i_idx].y;
  //  Real AbsRes = abs(sortedRhoPreMu[i_idx].y - p_old[i_idx]);

  Real Updated_rho = rho_np[i_idx] + rho_p[i_idx];
  Real rho_res = abs(1000 - sortedRhoPreMu[i_idx].x);  // Hard-coded for now
  Real p_res = 0;
  p_res = abs(sortedRhoPreMu[i_idx].y - p_old[i_idx]) / (abs(p_old[i_idx]) + 0.00001);

  Residuals[i_idx] = p_res;
  //	Residuals[i_idx] = max(p_res, 0.0);
  //  sortedRhoPreMu[i_idx].x = Updated_rho;

  if (sortedRhoPreMu[i_idx].w > -1) {
    Residuals[i_idx] = 0.0;
    sortedVelMas[i_idx] = V_new[i_idx];
  }
}

//--------------------------------------------------------------------------------------------------------------------------------
__global__ void CalcForces(Real3* new_vel,  // Write
                           Real4* derivVelRhoD,
                           Real3* sortedPosRad,  // Read
                           Real3* sortedVelMas,  // Read
                           Real4* sortedRhoPreMu,
                           uint* cellStart,
                           uint* cellEnd,
                           uint numAllMarkers,
                           Real m_0,
                           Real mu_0,
                           Real epsilon,
                           Real dT,
                           Real3 gravity,
                           volatile bool* isErrorD) {
  uint i_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (i_idx >= numAllMarkers)
    return;

  Real3 posi = sortedPosRad[i_idx];
  Real3 Veli = sortedVelMas[i_idx];
  Real p_i = sortedRhoPreMu[i_idx].y;
  Real rho_i = sortedRhoPreMu[i_idx].x;
  Real3 F_i_np = mR3(0);
  Real3 F_i_p = mR3(0);
  // get address in grid
  int3 gridPos = calcGridPos(posi);
  for (int z = -1; z <= 1; z++) {
    for (int y = -1; y <= 1; y++) {
      for (int x = -1; x <= 1; x++) {
        int3 neighbourPos = gridPos + mI3(x, y, z);
        uint gridHash = calcGridHash(neighbourPos);
        // get start of bucket for this cell
        uint startIndex = cellStart[gridHash];
        uint endIndex = cellEnd[gridHash];
        for (uint j = startIndex; j < endIndex; j++) {
          Real3 posj = sortedPosRad[j];
          Real3 dist3 = Distance(posi, posj);
          Real d = length(dist3);
          if (d > RESOLUTION_LENGTH_MULT * paramsD.HSML || i_idx == j)
            continue;
          Real3 Velj = sortedVelMas[j];
          Real p_j = sortedRhoPreMu[j].y;
          Real rho_j = sortedRhoPreMu[j].x;

          Real3 grad_i_wij = GradW(dist3);
          Real3 V_ij = (Veli - Velj);
          // Only Consider (fluid-fluid + fluid-solid) or Solid-Fluid Interaction
          if (sortedRhoPreMu[i_idx].w == -1 || (sortedRhoPreMu[i_idx].w == 2 && sortedRhoPreMu[j].w == -1))
            F_i_p += -m_0 * ((p_i / (rho_i * rho_i)) + (p_j / (rho_j * rho_j))) * grad_i_wij;

          Real Rho_bar = (rho_j + rho_i) * 0.5;
          Real3 muNumerator = 2 * mu_0 * dot(dist3, grad_i_wij) * V_ij;
          Real muDenominator = (Rho_bar * Rho_bar) * (d * d + paramsD.HSML * paramsD.HSML * epsilon);
          // Only Consider (fluid-fluid + fluid-solid) or Solid-Fluid Interaction
          if (sortedRhoPreMu[i_idx].w == -1 || (sortedRhoPreMu[i_idx].w == 2 && sortedRhoPreMu[j].w == -1))
            F_i_np += m_0 * muNumerator / muDenominator;
          if (!isfinite(length(F_i_np))) {
            printf("F_i_np in CalcForces returns Nan or Inf");
          }
        }
      }
    }
  }
  // Forces are per unit mass at this point.
  derivVelRhoD[i_idx] = mR4((F_i_p + F_i_np) * m_0);
  //  if (sortedRhoPreMu[i_idx].w == 1)
  //    printf("Total force on FlexMArker %d= %f,%f,%f\n", i_idx, derivVelRhoD[i_idx].x, derivVelRhoD[i_idx].y,
  //           derivVelRhoD[i_idx].z);

  new_vel[i_idx] = Veli + dT * (F_i_p + F_i_np) + gravity * dT;
}
//--------------------------------------------------------------------------------------------------------------------------------
__global__ void PrepPressure(Real3* sortedPosRad,  // Read
                             Real4* sortedRhoPreMu,
                             Real* p_old,
                             uint* cellStart,
                             uint* cellEnd,
                             uint numAllMarkers,
                             volatile bool* isErrorD) {
  uint i_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (i_idx >= numAllMarkers)
    return;

  Real3 posi = sortedPosRad[i_idx];
  Real p_i = sortedRhoPreMu[i_idx].y;
  Real sum_pW = 0;
  Real sum_W = 0;

  // get address in grid
  int3 gridPos = calcGridPos(posi);
  for (int z = -1; z <= 1; z++) {
    for (int y = -1; y <= 1; y++) {
      for (int x = -1; x <= 1; x++) {
        int3 neighbourPos = gridPos + mI3(x, y, z);
        uint gridHash = calcGridHash(neighbourPos);
        // get start of bucket for this cell
        uint startIndex = cellStart[gridHash];
        uint endIndex = cellEnd[gridHash];

        for (uint j = startIndex; j < endIndex; j++) {
          Real3 posJ = sortedPosRad[j];
          Real3 dist3 = Distance(posi, posJ);
          Real d = length(dist3);
          if (d > RESOLUTION_LENGTH_MULT * paramsD.HSML || i_idx == j)
            continue;
          Real p_j = p_old[j];
          sum_pW += p_j * W3(d);
          sum_W += W3(d);
        }
      }
    }
  }

  if (abs(sum_W) < EPSILON) {
    sortedRhoPreMu[i_idx].y = 0;
  } else {
    sortedRhoPreMu[i_idx].y = sum_pW / sum_W;
  }
}

//--------------------------------------------------------------------------------------------------------------------------------

void ChFsiForceParallel::calcPressureIISPH(thrust::device_vector<Real3>& bceAcc,
                                           thrust::device_vector<Real4> velMassRigid_fsiBodies_D,
                                           thrust::device_vector<Real3> accRigid_fsiBodies_D,
                                           thrust::device_vector<Real3> pos_fsi_fea_D,
                                           thrust::device_vector<Real3> vel_fsi_fea_D,
                                           thrust::device_vector<Real3> acc_fsi_fea_D) {
  Real RES = paramsH->PPE_res;
  PPE_SolutionType mySolutionType = paramsH->PPE_Solution_type;
  //  std::cout << "size of RIGID IDEN: " << fsiGeneralData->rigidIdentifierD.size() << "\n";
  //
  //  for (int i = 0; i < fsiGeneralData->rigidIdentifierD.size(); i++) {
  //    std::cout << "rigid iden: i=" << i << " is: " << fsiGeneralData->rigidIdentifierD[i] << "\n";
  //  }

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
  thrust::device_vector<Real> nonNormRho(numAllMarkers);

  *isErrorH = false;
  cudaMemcpy(isErrorD, isErrorH, sizeof(bool), cudaMemcpyHostToDevice);

  calcRho_kernel<<<numBlocks, numThreads>>>(mR3CAST(sortedSphMarkersD->posRadD), R1CAST(nonNormRho),
                                            U1CAST(markersProximityD->cellStartD), U1CAST(markersProximityD->cellEndD),
                                            numAllMarkers, paramsH->rho0, paramsH->markerMass, isErrorD);

  cudaThreadSynchronize();
  cudaCheckError();
  cudaMemcpy(isErrorH, isErrorD, sizeof(bool), cudaMemcpyDeviceToHost);
  if (*isErrorH == true) {
    throw std::runtime_error("Error! program crashed after F_i_np__AND__d_ii_kernel!\n");
  }
  thrust::device_vector<Real> dxi_over_Vi(numAllMarkers);
  thrust::fill(dxi_over_Vi.begin(), dxi_over_Vi.end(), 0);

  calcNormalizedRho_kernel<<<numBlocks, numThreads>>>(
      mR3CAST(sortedSphMarkersD->posRadD), mR3CAST(sortedSphMarkersD->velMasD),
      mR4CAST(sortedSphMarkersD->rhoPresMuD),  // input: sorted velocities
      R1CAST(nonNormRho), R1CAST(dxi_over_Vi), U1CAST(markersProximityD->cellStartD),
      U1CAST(markersProximityD->cellEndD), numAllMarkers, paramsH->rho0, paramsH->markerMass,
      paramsH->IncompressibilityFactor, isErrorD);

  // This is mandatory to sync here
  cudaThreadSynchronize();
  cudaCheckError();
  cudaMemcpy(isErrorH, isErrorD, sizeof(bool), cudaMemcpyDeviceToHost);
  if (*isErrorH == true) {
    throw std::runtime_error("Error! program crashed after calcNormalizedRho_kernel!\n");
  }
  if (paramsH->Adaptive_time_stepping) {
    int position = thrust::min_element(dxi_over_Vi.begin(), dxi_over_Vi.end()) - dxi_over_Vi.begin();
    Real min_dxi_over_Vi = dxi_over_Vi[position];

    Real dt = paramsH->Co_number * min_dxi_over_Vi;
    printf("Min dxi_over_Vi of fluid particles to boundary is: %f. Time step based on Co=%f is %f\n", min_dxi_over_Vi,
           paramsH->Co_number, dt);
    // I am doing this to prevent very low time steps (when it requires it to save data at the current step)
    // Because if will have to do two time steps either way
    if (dt / paramsH->dT_Max > 0.7 && dt / paramsH->dT_Max < 1)
      paramsH->dT = paramsH->dT_Max * 0.5;
    else
      paramsH->dT = fminf((float)dt, paramsH->dT_Max);

    std::cout << "time step is set to min(dt_Co,dT_Max)= " << paramsH->dT << "\n";
  }
  thrust::device_vector<Real3> d_ii(numAllMarkers);
  thrust::device_vector<Real3> V_np(numAllMarkers);
  thrust::fill(d_ii.begin(), d_ii.end(), mR3(0));
  thrust::fill(V_np.begin(), V_np.end(), mR3(0));
  *isErrorH = false;
  cudaMemcpy(isErrorD, isErrorH, sizeof(bool), cudaMemcpyHostToDevice);
  V_i_np__AND__d_ii_kernel<<<numBlocks, numThreads>>>(
      mR3CAST(sortedSphMarkersD->posRadD), mR3CAST(sortedSphMarkersD->velMasD), mR4CAST(sortedSphMarkersD->rhoPresMuD),
      mR3CAST(d_ii), mR3CAST(V_np), U1CAST(markersProximityD->cellStartD), U1CAST(markersProximityD->cellEndD),
      numAllMarkers, paramsH->markerMass, paramsH->mu0, paramsH->rho0, paramsH->epsMinMarkersDis, paramsH->dT,
      paramsH->gravity, isErrorD);

  cudaThreadSynchronize();
  cudaCheckError();
  cudaMemcpy(isErrorH, isErrorD, sizeof(bool), cudaMemcpyDeviceToHost);
  if (*isErrorH == true) {
    throw std::runtime_error("Error! program crashed after F_i_np__AND__d_ii_kernel!\n");
  }

  thrust::device_vector<Real> a_ii(numAllMarkers);
  thrust::device_vector<Real> rho_np(numAllMarkers);
  thrust::device_vector<Real> p_old(numAllMarkers);
  thrust::fill(a_ii.begin(), a_ii.end(), 0);
  thrust::fill(rho_np.begin(), rho_np.end(), 0);
  thrust::fill(p_old.begin(), p_old.end(), 0);

  *isErrorH = false;
  cudaMemcpy(isErrorD, isErrorH, sizeof(bool), cudaMemcpyHostToDevice);
  Rho_np_AND_a_ii<<<numBlocks, numThreads>>>(
      mR3CAST(sortedSphMarkersD->posRadD), mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(rho_np), R1CAST(a_ii),
      R1CAST(p_old), mR3CAST(V_np), mR3CAST(d_ii), U1CAST(markersProximityD->cellStartD),
      U1CAST(markersProximityD->cellEndD), numAllMarkers, paramsH->markerMass, paramsH->dT, isErrorD);

  cudaThreadSynchronize();
  cudaCheckError();
  cudaMemcpy(isErrorH, isErrorD, sizeof(bool), cudaMemcpyDeviceToHost);
  if (*isErrorH == true) {
    throw std::runtime_error("Error! program crashed after F_i_np__AND__d_ii_kernel!\n");
  }

  thrust::device_vector<Real3> V_new(numAllMarkers);
  thrust::fill(V_new.begin(), V_new.end(), mR3(0));
  thrust::device_vector<Real> a_ij;
  thrust::device_vector<Real> B_i(numAllMarkers);
  thrust::device_vector<Real> a_ij3;
  thrust::device_vector<Real3> summGradW(numAllMarkers);
  thrust::device_vector<uint> csrColIndA;
  thrust::device_vector<uint> row_indices;
  thrust::device_vector<unsigned long int> GlobalcsrColIndA;
  thrust::device_vector<Real> csrValA;
  thrust::device_vector<unsigned long int> numContacts(numAllMarkers);
  double durationFormAXB;

  //  int4 updatePortion = mI4(fsiGeneralData->referenceArray[1].x, fsiGeneralData->referenceArray[1].y,
  //                           fsiGeneralData->referenceArray[1 + numObjectsH->numRigidBodies].y, 0);

  int4 updatePortion =
      mI4(fsiGeneralData->referenceArray[0].y, fsiGeneralData->referenceArray[1].y,
          fsiGeneralData->referenceArray[1 + numObjectsH->numRigidBodies].y,
          fsiGeneralData->referenceArray[1 + numObjectsH->numRigidBodies + numObjectsH->numFlexBodies].y);

  uint NNZ;
  if (mySolutionType == SPARSE_MATRIX_JACOBI) {
    thrust::fill(a_ij.begin(), a_ij.end(), 0);
    thrust::fill(B_i.begin(), B_i.end(), 0);
    thrust::fill(a_ij3.begin(), a_ij3.end(), 0);
    thrust::fill(summGradW.begin(), summGradW.end(), mR3(0));
    thrust::fill(numContacts.begin(), numContacts.end(), 0);
    //------------------------------------------------------------------------
    //------------- MatrixJacobi
    //------------------------------------------------------------------------

    bool SPARSE_FLAG = true;
    double FormAXBClock = clock();
    thrust::device_vector<Real> Residuals(numAllMarkers);
    thrust::fill(Residuals.begin(), Residuals.end(), 1);
    thrust::device_vector<Real> rho_p(numAllMarkers);
    thrust::fill(rho_p.begin(), rho_p.end(), 0);

    *isErrorH = false;
    cudaMemcpy(isErrorD, isErrorH, sizeof(bool), cudaMemcpyHostToDevice);
    Calc_summGradW<<<numBlocks, numThreads>>>(
        mR3CAST(summGradW), mR3CAST(sortedSphMarkersD->posRadD), mR4CAST(sortedSphMarkersD->rhoPresMuD),
        U1CAST(markersProximityD->cellStartD), U1CAST(markersProximityD->cellEndD), numAllMarkers, paramsH->markerMass,
        paramsH->dT, isErrorD);
    cudaThreadSynchronize();
    cudaCheckError();
    cudaMemcpy(isErrorH, isErrorD, sizeof(bool), cudaMemcpyDeviceToHost);
    if (*isErrorH == true) {
      throw std::runtime_error("Error! program crashed after Calc_dij_pj!\n");
    }

    *isErrorH = false;
    cudaMemcpy(isErrorD, isErrorH, sizeof(bool), cudaMemcpyHostToDevice);
    CalcNumber_Contacts<<<numBlocks, numThreads>>>(
        LU1CAST(numContacts), mR3CAST(sortedSphMarkersD->posRadD), mR4CAST(sortedSphMarkersD->rhoPresMuD),
        U1CAST(markersProximityD->cellStartD), U1CAST(markersProximityD->cellEndD), numAllMarkers, isErrorD);

    cudaThreadSynchronize();
    cudaCheckError();
    cudaMemcpy(isErrorH, isErrorD, sizeof(bool), cudaMemcpyDeviceToHost);
    if (*isErrorH == true) {
      throw std::runtime_error("Error! program crashed after F_i_np__AND__d_ii_kernel!\n");
    }
    uint MAX_CONTACT = thrust::reduce(numContacts.begin(), numContacts.end(), 0, thrust::maximum<Real>());

    uint LastVal = numContacts[numAllMarkers - 1];
    thrust::exclusive_scan(numContacts.begin(), numContacts.end(), numContacts.begin());
    numContacts.push_back(LastVal + numContacts[numAllMarkers - 1]);
    NNZ = numContacts[numAllMarkers];

    csrValA.resize(NNZ);
    csrColIndA.resize(NNZ);
    GlobalcsrColIndA.resize(NNZ);

    thrust::fill(csrValA.begin(), csrValA.end(), 0);
    thrust::fill(GlobalcsrColIndA.begin(), GlobalcsrColIndA.end(), 0);
    thrust::fill(csrColIndA.begin(), csrColIndA.end(), 0);
    cudaMemcpy(isErrorD, isErrorH, sizeof(bool), cudaMemcpyHostToDevice);

    //    printf("size of gridMarkerIndexD: %d\n", markersProximityD->gridMarkerIndexD.size());
    //    printf("size of bceAcc: %d\n", bceAcc.size());
    //    printf("numBlocks: %d, numThreads: %d", numBlocks, numThreads);
    //    printf("max thread: %d", numBlocks * numThreads);

    std::cout << "updatePortion of  BC: " << updatePortion.x << " " << updatePortion.y << " " << updatePortion.z << " "
              << updatePortion.w << "\n ";

    FormAXB<<<numBlocks, numThreads>>>(
        R1CAST(csrValA), U1CAST(csrColIndA), LU1CAST(GlobalcsrColIndA), LU1CAST(numContacts), R1CAST(a_ij),
        R1CAST(a_ij3), R1CAST(B_i), mR3CAST(d_ii), R1CAST(a_ii), mR3CAST(summGradW),
        mR3CAST(sortedSphMarkersD->posRadD), mR3CAST(sortedSphMarkersD->velMasD),
        mR4CAST(sortedSphMarkersD->rhoPresMuD), mR3CAST(V_new), R1CAST(p_old), R1CAST(rho_np),

        mR3CAST(bceAcc), mR4CAST(velMassRigid_fsiBodies_D), mR3CAST(accRigid_fsiBodies_D),
        U1CAST(fsiGeneralData->rigidIdentifierD),

        mR3CAST(pos_fsi_fea_D), mR3CAST(vel_fsi_fea_D), mR3CAST(acc_fsi_fea_D), U1CAST(fsiGeneralData->FlexIdentifierD),
        U4CAST(fsiGeneralData->ShellelementsNodes),

        updatePortion, U1CAST(markersProximityD->gridMarkerIndexD), U1CAST(markersProximityD->cellStartD),
        U1CAST(markersProximityD->cellEndD), numAllMarkers, paramsH->markerMass, paramsH->rho0, paramsH->dT,
        paramsH->gravity, SPARSE_FLAG, isErrorD);

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
  Real MaxRes = 1;
  thrust::device_vector<Real> Residuals(numAllMarkers);
  thrust::fill(Residuals.begin(), Residuals.end(), 1);
  thrust::device_vector<Real3> dij_pj(numAllMarkers);
  thrust::fill(dij_pj.begin(), dij_pj.end(), mR3(0));
  thrust::device_vector<Real3> F_p(numAllMarkers);
  thrust::fill(F_p.begin(), F_p.end(), mR3(0));
  thrust::device_vector<Real> rho_p(numAllMarkers);
  thrust::fill(rho_p.begin(), rho_p.end(), 0);
  double LinearSystemClock = clock();

  //	printf("--------IISPH CLOCK-----------\n");
  //	printf(" FormAXB: %f \n", durationFormAXB);

  while (MaxRes > RES || Iteration < 3) {
    *isErrorH = false;
    cudaMemcpy(isErrorD, isErrorH, sizeof(bool), cudaMemcpyHostToDevice);
    Initialize_Variables<<<numBlocks, numThreads>>>(mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(p_old),
                                                    mR3CAST(sortedSphMarkersD->velMasD), mR3CAST(V_new), numAllMarkers,
                                                    isErrorD);
    cudaThreadSynchronize();
    cudaCheckError();
    cudaMemcpy(isErrorH, isErrorD, sizeof(bool), cudaMemcpyDeviceToHost);
    if (*isErrorH == true) {
      throw std::runtime_error("Error! program crashed after Initialize_Variables!\n");
    }

    if (mySolutionType == IterativeJacobi) {
      *isErrorH = false;
      cudaMemcpy(isErrorD, isErrorH, sizeof(bool), cudaMemcpyHostToDevice);
      Calc_dij_pj<<<numBlocks, numThreads>>>(mR3CAST(dij_pj), mR3CAST(F_p), mR3CAST(d_ii),
                                             mR3CAST(sortedSphMarkersD->posRadD), mR3CAST(sortedSphMarkersD->velMasD),
                                             mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(p_old),
                                             U1CAST(markersProximityD->cellStartD), U1CAST(markersProximityD->cellEndD),
                                             numAllMarkers, paramsH->markerMass, paramsH->dT, isErrorD);
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
          mR3CAST(sortedSphMarkersD->posRadD), mR3CAST(sortedSphMarkersD->velMasD),
          mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(p_old), mR3CAST(V_new), U1CAST(markersProximityD->cellStartD),
          U1CAST(markersProximityD->cellEndD), numAllMarkers, paramsH->markerMass, paramsH->rho0, paramsH->dT,
          paramsH->gravity, isErrorD);
      cudaThreadSynchronize();
      cudaCheckError();
      cudaMemcpy(isErrorH, isErrorD, sizeof(bool), cudaMemcpyDeviceToHost);
      if (*isErrorH == true) {
        throw std::runtime_error("Error! program crashed after Calc_Pressure!\n");
      }
    }

    if (mySolutionType == SPARSE_MATRIX_JACOBI) {
      *isErrorH = false;
      cudaMemcpy(isErrorD, isErrorH, sizeof(bool), cudaMemcpyHostToDevice);
      Calc_Pressure_AXB_USING_CSR<<<numBlocks, numThreads>>>(
          R1CAST(csrValA), R1CAST(a_ii), U1CAST(csrColIndA), LU1CAST(numContacts),
          mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(nonNormRho), mR3CAST(sortedSphMarkersD->velMasD),
          mR3CAST(V_new), R1CAST(p_old), R1CAST(B_i), paramsH->rho0, numAllMarkers, paramsH->ClampPressure, isErrorD);
      cudaThreadSynchronize();
      cudaCheckError();
      cudaMemcpy(isErrorH, isErrorD, sizeof(bool), cudaMemcpyDeviceToHost);
      if (*isErrorH == true) {
        throw std::runtime_error("Error! program crashed after Iterative_pressure_update!\n");
      }
    }
    *isErrorH = false;
    cudaMemcpy(isErrorD, isErrorH, sizeof(bool), cudaMemcpyHostToDevice);

    Update_AND_Calc_Res<<<numBlocks, numThreads>>>(mR3CAST(sortedSphMarkersD->velMasD),
                                                   mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(p_old),
                                                   mR3CAST(V_new), R1CAST(rho_p), R1CAST(rho_np), R1CAST(Residuals),
                                                   numAllMarkers, Iteration, paramsH->PPE_relaxation, false, isErrorD);
    cudaThreadSynchronize();
    cudaCheckError();
    cudaMemcpy(isErrorH, isErrorD, sizeof(bool), cudaMemcpyDeviceToHost);
    if (*isErrorH == true) {
      throw std::runtime_error("Error! program crashed after Iterative_pressure_update!\n");
    }

    Iteration++;
    //			      MaxRes = thrust::reduce(Residuals.begin(), Residuals.end(), 0.0,
    // thrust::maximum<Real>());
    //    Real PMAX = thrust::reduce(p_old.begin(), p_old.end(), 0.0, thrust::maximum<Real>());
    MaxRes =
        thrust::reduce(Residuals.begin(), Residuals.end(), 0.0, thrust::plus<Real>()) / numObjectsH->numFluidMarkers;

    //      Real R_np = thrust::reduce(rho_np.begin(), rho_np.end(), 0.0, thrust::plus<Real>()) / rho_np.size();
    //      Real R_p = thrust::reduce(rho_p.begin(), rho_p.end(), 0.0, thrust::plus<Real>()) / rho_p.size();
    //
    //			printf("Iter= %d, Res= %f\n", Iteration,
    //					MaxRes);

    //    if (paramsH->USE_CUSP && Iteration > 20)
    //      break;
  }

  if (paramsH->USE_CUSP) {
    *isErrorH = false;
    cudaMemcpy(isErrorD, isErrorH, sizeof(bool), cudaMemcpyHostToDevice);
    PrepareForCusp<<<numBlocks, numThreads>>>(R1CAST(csrValA), U1CAST(csrColIndA), LU1CAST(numContacts),
                                              mR4CAST(sortedSphMarkersD->rhoPresMuD), R1CAST(B_i), paramsH->rho0,
                                              numAllMarkers, isErrorD);
    cudaThreadSynchronize();
    cudaCheckError();
    cudaMemcpy(isErrorH, isErrorD, sizeof(bool), cudaMemcpyDeviceToHost);
    if (*isErrorH == true) {
      throw std::runtime_error("Error! program crashed after Iterative_pressure_update!\n");
    }

    cusp::csr_matrix<unsigned long int, double, cusp::device_memory> AMatrix(numAllMarkers, numAllMarkers, NNZ);
    AMatrix.row_offsets = numContacts;
    AMatrix.column_indices = csrColIndA;
    AMatrix.values = csrValA;

    cusp::array1d<double, cusp::device_memory> x = p_old;
    cusp::array1d<double, cusp::device_memory> b = B_i;
    // set stopping criteria:

    cusp::convergence_monitor<double> monitor(b, 1000, 1e-6, 1e-5);
    //    cusp::default_monitor<double> monitor(b, 1000, 1e-6, RES);

    //    cusp::precond::diagonal<double, cusp::device_memory> M2(AMatrix);
    cusp::identity_operator<double, cusp::device_memory> M(AMatrix.num_rows, AMatrix.num_rows);

    // solve the linear system A * x = b with the Conjugate Gradient method
    //    cusp::krylov::cg(AMatrix, x, b, monitor, M);
    int restart = 200;
    //    cusp::krylov::cg_m(AMatrix, x, b, restart, monitor, M);
    cusp::krylov::gmres(AMatrix, x, b, restart, monitor);
    //    cusp::krylov::bicgstab(AMatrix, x, b, monitor);
    //    cusp::krylov::cr(AMatrix, x, b, monitor);
  }

  //  *isErrorH = false;
  //  cudaMemcpy(isErrorD, isErrorH, sizeof(bool), cudaMemcpyHostToDevice);
  //
  //  PrepPressure<<<numBlocks, numThreads>>>(mR3CAST(sortedSphMarkersD->posRadD),
  //  mR4CAST(sortedSphMarkersD->rhoPresMuD),
  //                                          R1CAST(p_old), U1CAST(markersProximityD->cellStartD),
  //                                          U1CAST(markersProximityD->cellEndD), numAllMarkers, isErrorD);
  //  cudaThreadSynchronize();
  //  cudaCheckError();
  //  cudaMemcpy(isErrorH, isErrorD, sizeof(bool), cudaMemcpyDeviceToHost);
  //  if (*isErrorH == true) {
  //    throw std::runtime_error("Error! program crashed after PrepPressure!\n");
  //  }
  double durationLinearSystem = (clock() - LinearSystemClock) / (double)CLOCKS_PER_SEC;
  double durationtotal_step_time = (clock() - total_step_timeClock) / (double)CLOCKS_PER_SEC;
  //	printf(" Linear System: %f \n Total: %f \n ", durationLinearSystem,
  //			durationtotal_step_time);

  printf("\---------------IISPH CLOCK-------------------\n");
  printf(" Total: %f \n FormAXB: %f\n Linear System: %f \n", durationtotal_step_time, durationFormAXB,
         durationLinearSystem);
  printf("Iter# = %d, Res= %f \n", Iteration, MaxRes);
  printf("----------------------------------------------\n");

  //------------------------------------------------------------------------
  //------------------------------------------------------------------------
  cudaFree(isErrorD);
  free(isErrorH);
  nonNormRho.clear();
  d_ii.clear();
  V_np.clear();
  a_ii.clear();
  rho_np.clear();
  a_ii.clear();
  a_ij.clear();
  B_i.clear();
  a_ij3.clear();
  summGradW.clear();
  csrColIndA.clear();
  GlobalcsrColIndA.clear();
  csrValA.clear();
  V_new.clear();
  numContacts.clear();
}

void ChFsiForceParallel::ForceIISPH(SphMarkerDataD* otherSphMarkersD,
                                    FsiBodiesDataD* otherFsiBodiesD,
                                    FsiShellsDataD* otherFsiShellsD,
                                    FsiMeshDataD* otherFsiMeshD) {
  std::cout << "dT in ForceSPH brfore calcPressure: " << paramsH->dT << "\n";

  sphMarkersD = otherSphMarkersD;

  fsiCollisionSystem->ArrangeData(sphMarkersD);
  //  bceWorker->ModifyBceVelocity(sphMarkersD, otherFsiBodiesD);

  thrust::device_vector<Real3> bceAcc(numObjectsH->numRigid_SphMarkers);
  //
  //  if (numObjectsH->numRigid_SphMarkers > 0) {
  //    bceWorker->CalcBceAcceleration(bceAcc, otherFsiBodiesD->q_fsiBodies_D, otherFsiBodiesD->accRigid_fsiBodies_D,
  //                                   otherFsiBodiesD->omegaVelLRF_fsiBodies_D,
  //                                   otherFsiBodiesD->omegaAccLRF_fsiBodies_D,
  //                                   fsiGeneralData->rigidSPH_MeshPos_LRF_D, fsiGeneralData->rigidIdentifierD,
  //                                   numObjectsH->numRigid_SphMarkers);
  //  }

  calcPressureIISPH(bceAcc, otherFsiBodiesD->velMassRigid_fsiBodies_D, otherFsiBodiesD->accRigid_fsiBodies_D,
                    otherFsiMeshD->pos_fsi_fea_D, otherFsiMeshD->vel_fsi_fea_D, otherFsiMeshD->acc_fsi_fea_D);

  bceAcc.clear();

  int numAllMarkers = numObjectsH->numBoundaryMarkers + numObjectsH->numFluidMarkers;
  uint numThreads, numBlocks;
  computeGridSize(numAllMarkers, 256, numBlocks, numThreads);
  bool *isErrorH, *isErrorD;

  isErrorH = (bool*)malloc(sizeof(bool));
  cudaMalloc((void**)&isErrorD, sizeof(bool));
  *isErrorH = false;
  cudaMemcpy(isErrorD, isErrorH, sizeof(bool), cudaMemcpyHostToDevice);
  //------------------------------------------------------------------------
  // thread per particle
  thrust::fill(vel_XSPH_Sorted_D.begin(), vel_XSPH_Sorted_D.end(), mR3(0));
  std::cout << "dT in ForceSPH after calcPressure: " << paramsH->dT << "\n";

  CalcForces<<<numBlocks, numThreads>>>(
      mR3CAST(vel_XSPH_Sorted_D), mR4CAST(derivVelRhoD_Sorted_D), mR3CAST(sortedSphMarkersD->posRadD),
      mR3CAST(sortedSphMarkersD->velMasD), mR4CAST(sortedSphMarkersD->rhoPresMuD),
      U1CAST(markersProximityD->cellStartD), U1CAST(markersProximityD->cellEndD), numAllMarkers, paramsH->markerMass,
      paramsH->mu0, paramsH->epsMinMarkersDis, paramsH->dT, paramsH->gravity, isErrorD);
  cudaThreadSynchronize();
  cudaCheckError();

  cudaMemcpy(isErrorH, isErrorD, sizeof(bool), cudaMemcpyDeviceToHost);
  if (*isErrorH == true) {
    throw std::runtime_error("Error! program crashed in CalcForces!\n");
  }

  CopySortedToOriginal_NonInvasive_R3(fsiGeneralData->vel_XSPH_D, vel_XSPH_Sorted_D,
                                      markersProximityD->gridMarkerIndexD);
  CopySortedToOriginal_NonInvasive_R3(sphMarkersD->posRadD, sortedSphMarkersD->posRadD,
                                      markersProximityD->gridMarkerIndexD);
  CopySortedToOriginal_NonInvasive_R3(sphMarkersD->velMasD, sortedSphMarkersD->velMasD,
                                      markersProximityD->gridMarkerIndexD);
  CopySortedToOriginal_NonInvasive_R4(sphMarkersD->rhoPresMuD, sortedSphMarkersD->rhoPresMuD,
                                      markersProximityD->gridMarkerIndexD);
  CopySortedToOriginal_NonInvasive_R4(fsiGeneralData->derivVelRhoD, derivVelRhoD_Sorted_D,
                                      markersProximityD->gridMarkerIndexD);
  //
}

}  // end namespace fsi
}  // end namespace chrono
