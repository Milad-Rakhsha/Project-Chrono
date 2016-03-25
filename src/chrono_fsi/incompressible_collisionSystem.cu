/*
 * SDKCollisionSystem.cu
 *
 *  Created on: Mar 2, 2013
 *      Author: Arman Pazouki, Milad Rakhsha
 */
#include <stdexcept>
#include <thrust/sort.h>
#include "chrono_fsi/incompressible_collisionSystem.cuh"

//#include "extraOptionalFunctions.cuh"
//#include "SDKCollisionSystemAdditional.cuh"

/**
 * @brief calcGridHash
 * @details  See SDKCollisionSystem.cuh
 */
__device__ int3 calcGridPos(Real3 p) {
  int3 gridPos;
  gridPos.x = floor((p.x - paramsD.worldOrigin.x) / paramsD.cellSize.x);
  gridPos.y = floor((p.y - paramsD.worldOrigin.y) / paramsD.cellSize.y);
  gridPos.z = floor((p.z - paramsD.worldOrigin.z) / paramsD.cellSize.z);
  return gridPos;
}

/**
 * @brief calcGridHash
 * @details  See SDKCollisionSystem.cuh
 */
__device__ uint calcGridHash(int3 gridPos) {
  gridPos.x -= ((gridPos.x >= paramsD.gridSize.x) ? paramsD.gridSize.x : 0);
  gridPos.y -= ((gridPos.y >= paramsD.gridSize.y) ? paramsD.gridSize.y : 0);
  gridPos.z -= ((gridPos.z >= paramsD.gridSize.z) ? paramsD.gridSize.z : 0);

  gridPos.x += ((gridPos.x < 0) ? paramsD.gridSize.x : 0);
  gridPos.y += ((gridPos.y < 0) ? paramsD.gridSize.y : 0);
  gridPos.z += ((gridPos.z < 0) ? paramsD.gridSize.z : 0);

  return __umul24(__umul24(gridPos.z, paramsD.gridSize.y), paramsD.gridSize.x) +
         __umul24(gridPos.y, paramsD.gridSize.x) + gridPos.x;
}

/**
 * @brief calcGridHash
 * @details  See SDKCollisionSystem.cuh
 */
__device__ void DifVelocityRho_implicit(Real3& dist3,
                                        Real& d,
                                        Real3 posRadA,
                                        Real3 posRadB,
                                        Real3& velMasA,
                                        Real3& velMasB,
                                        Real4& rhoPresMuA,
                                        Real4& rhoPresMuB,
                                        Real multViscosity) {
  // TODO
  // Milad: need to change this
}

//--------------------------------------------------------------------------------------------------------------------------------
// collide a particle against all other particles in a given cell
// Arman : revisit equation 10 of tech report, is it only on fluid or it is on all markers
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
// collide a particle against all other particles in a given cell
__device__ Real4 collideCell_implicit(int3 gridPos,
                                      uint index,
                                      Real3 posRadA,
                                      Real3 velMasA,
                                      Real4 rhoPresMuA,
                                      Real3* sortedPosRad,
                                      Real3* sortedVelMas,
                                      Real4* sortedRhoPreMu,
                                      Real3* velMas_ModifiedBCE,
                                      Real4* rhoPreMu_ModifiedBCE,
                                      uint* gridMarkerIndex,
                                      uint* cellStart,
                                      uint* cellEnd) {
  //  uint gridHash = calcGridHash(gridPos);
  //  // get start of bucket for this cell
  //  Real4 derivVelRho = mR4(0);
  //
  //  uint startIndex = FETCH(cellStart, gridHash);
  //  if (startIndex == 0xffffffff) {  // cell is not empty
  //    return derivVelRho;
  //  }
  //  // iterate over particles in this cell
  //  uint endIndex = FETCH(cellEnd, gridHash);
  //
  //  for (uint j = startIndex; j < endIndex; j++) {
  //    if (j != index) {  // check not colliding with self
  //      Real3 posRadB = FETCH(sortedPosRad, j);
  //      Real3 dist3Alpha = posRadA - posRadB;
  //      //			Real3 dist3 = Distance(posRadA, posRadB);
  //      Real3 dist3 = Modify_Local_PosB(posRadB, posRadA);
  //      Real d = length(dist3);
  //      if (d > RESOLUTION_LENGTH_MULT * paramsD.HSML)
  //        continue;
  //
  //      Real4 rhoPresMuB = FETCH(sortedRhoPreMu, j);
  //      //			// old version. When rigid-rigid contact used to be handled within fluid
  //      //			if ((fabs(rhoPresMuB.w - rhoPresMuA.w) < .1)
  //      //					&& rhoPresMuA.w > -.1) {
  //      //				continue;
  //      //			}
  //      if (rhoPresMuA.w > -.1 && rhoPresMuB.w > -.1) {  // no rigid-rigid force
  //        continue;
  //      }
  //
  //      modifyPressure(rhoPresMuB, dist3Alpha);
  //      Real3 velMasB = FETCH(sortedVelMas, j);
  //      if (rhoPresMuB.w > -.1) {
  //        int bceIndexB = gridMarkerIndex[j] - (numObjectsD.numFluidMarkers);
  //        if (!(bceIndexB >= 0 && bceIndexB < numObjectsD.numBoundaryMarkers + numObjectsD.numRigid_SphMarkers)) {
  //          printf("Error! bceIndex out of bound, collideD !\n");
  //        }
  //        rhoPresMuB = rhoPreMu_ModifiedBCE[bceIndexB];
  //        velMasB = velMas_ModifiedBCE[bceIndexB];
  //      }
  //      Real multViscosit = 1;
  //      Real4 derivVelRhoAB = mR4(0.0f);
  //      derivVelRhoAB =
  //          DifVelocityRho_implicit(dist3, d, posRadA, posRadB, velMasA, velMasB, rhoPresMuA, rhoPresMuB,
  //          multViscosit);
  //      derivVelRho += derivVelRhoAB;
  //    }
  //  }
  //
  //  // ff1
  //  //	if (rhoPresMuA.w > 0) printf("force value %f %f %f\n", 1e20*derivV.x, 1e20*derivV.y, 1e20*derivV.z);
  //  return derivVelRho;
  return mR4(0);
}
//--------------------------------------------------------------------------------------------------------------------------------

/**
 * @brief calcHashD
 * @details
 * 		 1. Get particle index. Determine by the block and thread we are in.
 * 		 2. From x,y,z position determine which bin it is in.
 * 		 3. Calculate hash from bin index.
 * 		 4. Store hash and particle index associated with it.
 *
 * @param gridMarkerHash
 * @param gridMarkerIndex
 * @param posRad
 * @param numAllMarkers
 */
__global__ void calcHashD(uint* gridMarkerHash,   // output
                          uint* gridMarkerIndex,  // output
                          Real3* posRad,          // input: positions
                          uint numAllMarkers,
                          volatile bool* isErrorD) {
  /* Calculate the index of where the particle is stored in posRad. */
  uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (index >= numAllMarkers)
    return;

  Real3 p = posRad[index];

  if (!(isfinite(p.x) && isfinite(p.y) && isfinite(p.z))) {
    printf("Error! particle position is NAN: thrown from SDKCollisionSystem.cu, calcHashD !\n");
    *isErrorD = true;
    return;
  }

  /* Check particle is inside the domain. */
  Real3 boxCorner = paramsD.worldOrigin;
  if (p.x < boxCorner.x || p.y < boxCorner.y || p.z < boxCorner.z) {
    printf(
        "Out of Min Boundary, point %f %f %f, boundary min: %f %f %f. Thrown from SDKCollisionSystem.cu, calcHashD !\n",
        p.x, p.y, p.z, boxCorner.x, boxCorner.y, boxCorner.z);
    *isErrorD = true;
    return;
  }
  boxCorner = paramsD.worldOrigin + paramsD.boxDims;
  if (p.x > boxCorner.x || p.y > boxCorner.y || p.z > boxCorner.z) {
    printf(
        "Out of max Boundary, point %f %f %f, boundary max: %f %f %f. Thrown from SDKCollisionSystem.cu, calcHashD !\n",
        p.x, p.y, p.z, boxCorner.x, boxCorner.y, boxCorner.z);
    *isErrorD = true;
    return;
  }

  /* Get x,y,z bin index in grid */
  int3 gridPos = calcGridPos(p);
  /* Calculate a hash from the bin index */
  uint hash = calcGridHash(gridPos);

  /* Store grid hash */
  gridMarkerHash[index] = hash;
  /* Store particle index associated to the hash we stored in gridMarkerHash */
  gridMarkerIndex[index] = index;
}

/**
 * @brief reorderDataAndFindCellStartD
 * @details See SDKCollisionSystem.cuh for more info
 */
__global__ void reorderDataAndFindCellStartD(
    uint* cellStart,      // output: cell start index
    uint* cellEnd,        // output: cell end index
    Real3* sortedPosRad,  // output: sorted positions
    Real3* sortedVelMas,  // output: sorted velocities
    Real4* sortedRhoPreMu,
    uint* gridMarkerHash,       // input: sorted grid hashes
    uint* gridMarkerIndex,      // input: sorted particle indices
    uint* mapOriginalToSorted,  // mapOriginalToSorted[originalIndex] = originalIndex
    Real3* oldPosRad,           // input: sorted position array
    Real3* oldVelMas,           // input: sorted velocity array
    Real4* oldRhoPreMu,
    uint numAllMarkers) {
  extern __shared__ uint sharedHash[];  // blockSize + 1 elements
  /* Get the particle index the current thread is supposed to be looking at. */
  uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  uint hash;
  /* handle case when no. of particles not multiple of block size */
  if (index < numAllMarkers) {
    hash = gridMarkerHash[index];
    /* Load hash data into shared memory so that we can look at neighboring particle's hash
     * value without loading two hash values per thread
     */
    sharedHash[threadIdx.x + 1] = hash;

    if (index > 0 && threadIdx.x == 0) {
      /* first thread in block must load neighbor particle hash */
      sharedHash[0] = gridMarkerHash[index - 1];
    }
  }

  __syncthreads();

  if (index < numAllMarkers) {
    /* If this particle has a different cell index to the previous particle then it must be
     * the first particle in the cell, so store the index of this particle in the cell. As it
     * isn't the first particle, it must also be the cell end of the previous particle's cell
     */
    if (index == 0 || hash != sharedHash[threadIdx.x]) {
      cellStart[hash] = index;
      if (index > 0)
        cellEnd[sharedHash[threadIdx.x]] = index;
    }

    if (index == numAllMarkers - 1) {
      cellEnd[hash] = index + 1;
    }

    /* Now use the sorted index to reorder the pos and vel data */
    uint originalIndex = gridMarkerIndex[index];  // map sorted to original
    mapOriginalToSorted[index] = index;           // will be sorted outside. Alternatively, you could have
    // mapOriginalToSorted[originalIndex] = index; without need to sort. But that
    // is not thread safe
    Real3 posRad = FETCH(oldPosRad, originalIndex);  // macro does either global read or texture fetch
    Real3 velMas = FETCH(oldVelMas, originalIndex);  // see particles_kernel.cuh
    Real4 rhoPreMu = FETCH(oldRhoPreMu, originalIndex);

    if (!(isfinite(posRad.x) && isfinite(posRad.y) && isfinite(posRad.z))) {
      printf("Error! particle position is NAN: thrown from SDKCollisionSystem.cu, reorderDataAndFindCellStartD !\n");
    }
    if (!(isfinite(velMas.x) && isfinite(velMas.y) && isfinite(velMas.z))) {
      printf("Error! particle velocity is NAN: thrown from SDKCollisionSystem.cu, reorderDataAndFindCellStartD !\n");
    }
    if (!(isfinite(rhoPreMu.x) && isfinite(rhoPreMu.y) && isfinite(rhoPreMu.z) && isfinite(rhoPreMu.w))) {
      printf("Error! particle rhoPreMu is NAN: thrown from SDKCollisionSystem.cu, reorderDataAndFindCellStartD !\n");
    }
    sortedPosRad[index] = posRad;
    sortedVelMas[index] = velMas;
    sortedRhoPreMu[index] = rhoPreMu;
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
  Real4 rhoPreMuA = FETCH(sortedRhoPreMu, idA);
  Real3 posRadA = FETCH(sortedPosRad, idA);
  Real3 velMasA = FETCH(sortedVelMas, idA);
  int isAffectedV = 0;
  int isAffectedP = 0;

  Real3 sumVW = mR3(0);
  Real sumWAll = 0;
  Real3 sumRhoRW = mR3(0);
  Real sumPW = 0;
  Real sumWFluid = 0;

  // get address in grid
  int3 gridPos = calcGridPos(posRadA);

  /// if (gridPos.x == paramsD.gridSize.x-1) printf("****aha %d %d\n", gridPos.x, paramsD.gridSize.x);

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
        printf("Error! marker index out of bound: thrown from SDKCollisionSystem.cu, new_BCE_VelocityPressure !\n");
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
// updates the fluid particles' properties, i.e. velocity, density, pressure, position
__global__ void UpdateFluidD_implicit(Real3* posRadD,
                                      Real3* velMasD,
                                      Real4* rhoPresMuD,
                                      Real4* derivVelRhoD,
                                      int2 updatePortion,
                                      Real dT,
                                      volatile bool* isErrorD) {
  uint index = blockIdx.x * blockDim.x + threadIdx.x;
  index += updatePortion.x;  // updatePortion = [start, end] index of the update portion
  if (index >= updatePortion.y) {
    return;
  }
  Real4 derivVelRho = derivVelRhoD[index];
  Real4 rhoPresMu = rhoPresMuD[index];

  if (rhoPresMu.w < 0) {
    //-------------
    // ** position
    //-------------

    Real3 velMas = velMasD[index];
    Real3 posRad = posRadD[index];
    Real3 updatedPositon = posRad + velMas * dT;
    if (!(isfinite(updatedPositon.x) && isfinite(updatedPositon.y) && isfinite(updatedPositon.z))) {
      printf("Error! particle position is NAN: thrown from SDKCollisionSystem.cu, UpdateFluidDKernel !\n");
      *isErrorD = true;
      return;
    }
    posRadD[index] = updatedPositon;  // posRadD updated

    //-------------
    // ** velocity
    //-------------

    Real3 updatedVelocity = velMas + mR3(derivVelRho) * dT;

    if (!(isfinite(updatedVelocity.x) && isfinite(updatedVelocity.y) && isfinite(updatedVelocity.z))) {
      if (paramsD.enableAggressiveTweak) {
        updatedVelocity = mR3(0);
      } else {
        printf("Error! particle updatedVelocity is NAN: thrown from SDKCollisionSystem.cu, UpdateFluidDKernel !\n");
        *isErrorD = true;
        return;
      }
    }
    // 2*** let's tweak a little bit :)
    if (length(updatedVelocity) > paramsD.tweakMultV * paramsD.HSML / paramsD.dT && paramsD.enableTweak) {
      updatedVelocity *= (paramsD.tweakMultV * paramsD.HSML / paramsD.dT) / length(updatedVelocity);
    }
    // 2*** end tweak

    velMasD[index] = updatedVelocity;
  }
  // 3*** let's tweak a little bit :)
  if (!(isfinite(derivVelRho.w))) {
    if (paramsD.enableAggressiveTweak) {
      derivVelRho.w = 0;
    } else {
      printf("Error! particle derivVelRho.w is NAN: thrown from SDKCollisionSystem.cu, UpdateFluidDKernel !\n");
      *isErrorD = true;
      return;
    }
  }
  if (fabs(derivVelRho.w) > paramsD.tweakMultRho * paramsD.rho0 / paramsD.dT && paramsD.enableTweak) {
    derivVelRho.w *=
        (paramsD.tweakMultRho * paramsD.rho0 / paramsD.dT) / fabs(derivVelRho.w);  // to take care of the sign as well
  }
  // 2*** end tweak
  Real rho2 = rhoPresMu.x + derivVelRho.w * dT;  // rho update. (i.e. rhoPresMu.x), still not wriiten to global matrix
  rhoPresMu.y = Eos(rho2, rhoPresMu.w);
  rhoPresMu.x = rho2;
  if (!(isfinite(rhoPresMu.x) && isfinite(rhoPresMu.y) && isfinite(rhoPresMu.z) && isfinite(rhoPresMu.w))) {
    printf("Error! particle rho pressure is NAN: thrown from SDKCollisionSystem.cu, UpdateFluidDKernel !\n");
    *isErrorD = true;
    return;
  }
  rhoPresMuD[index] = rhoPresMu;  // rhoPresMuD updated
}

/**
 * @brief Copies the sortedVelXSPH to velXSPH according to indexing
 * @details [long description]
 *
 * @param vel_XSPH_D
 * @param vel_XSPH_Sorted_D Pointer to new sorted vel_XSPH vector
 * @param m_dGridMarkerIndex List of indeces used to sort vel_XSPH_D
 */

__global__ void CopySorted_vXSPH_dVdRho_to_original_kernel(Real3* vel_XSPH_D,
                                                           Real4* derivVelRhoD,
                                                           Real3* vel_XSPH_Sorted_D,
                                                           Real4* sortedDerivVelRho_fsi_D,
                                                           uint* mapOriginalToSorted) {
  uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (index >= numObjectsD.numAllMarkers)
    return;
  vel_XSPH_D[index] = vel_XSPH_Sorted_D[mapOriginalToSorted[index]];
  derivVelRhoD[index] = sortedDerivVelRho_fsi_D[mapOriginalToSorted[index]];
}

//--------------------------------------------------------------------------------------------------------------------------------
// updates the fluid particles' properties, i.e. velocity, density, pressure, position
__global__ void UpdateKernelBoundary(Real4* rhoPresMuD, Real4* derivVelRhoD, int2 updatePortion, Real dT) {
  uint index = blockIdx.x * blockDim.x + threadIdx.x;
  index += updatePortion.x;  // updatePortion = [start, end] index of the update portion
  if (index >= updatePortion.y) {
    return;
  }

  Real4 derivVelRho = derivVelRhoD[index];
  Real4 rhoPresMu = rhoPresMuD[index];
  Real rho2 = rhoPresMu.x + derivVelRho.w * dT;  // rho update. (i.e. rhoPresMu.x), still not wriiten to global matrix
  rhoPresMu.y = Eos(rho2, rhoPresMu.w);
  rhoPresMu.x = rho2;
  if (!(isfinite(rhoPresMu.x) && isfinite(rhoPresMu.y) && isfinite(rhoPresMu.z) && isfinite(rhoPresMu.w))) {
    printf("Error! particle rp is NAN: thrown from SDKCollisionSystem.cu, UpdateKernelBoundary !\n");
  }
  rhoPresMuD[index] = rhoPresMu;  // rhoPresMuD updated
}

//--------------------------------------------------------------------------------------------------------------------------------
// applies periodic BC along x
__global__ void ApplyPeriodicBoundaryXKernel(Real3* posRadD, Real4* rhoPresMuD) {
  uint index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= numObjectsD.numAllMarkers) {
    return;
  }
  Real4 rhoPresMu = rhoPresMuD[index];
  if (fabs(rhoPresMu.w) < .1) {
    return;
  }  // no need to do anything if it is a boundary particle
  Real3 posRad = posRadD[index];
  if (posRad.x > paramsD.cMax.x) {
    posRad.x -= (paramsD.cMax.x - paramsD.cMin.x);
    posRadD[index] = posRad;
    if (rhoPresMu.w < -.1) {
      rhoPresMu.y = rhoPresMu.y + paramsD.deltaPress.x;
      rhoPresMuD[index] = rhoPresMu;
    }
    return;
  }
  if (posRad.x < paramsD.cMin.x) {
    posRad.x += (paramsD.cMax.x - paramsD.cMin.x);
    posRadD[index] = posRad;
    if (rhoPresMu.w < -.1) {
      rhoPresMu.y = rhoPresMu.y - paramsD.deltaPress.x;
      rhoPresMuD[index] = rhoPresMu;
    }
    return;
  }
}
//--------------------------------------------------------------------------------------------------------------------------------
// applies periodic BC along y
__global__ void ApplyPeriodicBoundaryYKernel(Real3* posRadD, Real4* rhoPresMuD) {
  uint index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= numObjectsD.numAllMarkers) {
    return;
  }
  Real4 rhoPresMu = rhoPresMuD[index];
  if (fabs(rhoPresMu.w) < .1) {
    return;
  }  // no need to do anything if it is a boundary particle
  Real3 posRad = posRadD[index];
  if (posRad.y > paramsD.cMax.y) {
    posRad.y -= (paramsD.cMax.y - paramsD.cMin.y);
    posRadD[index] = posRad;
    if (rhoPresMu.w < -.1) {
      rhoPresMu.y = rhoPresMu.y + paramsD.deltaPress.y;
      rhoPresMuD[index] = rhoPresMu;
    }
    return;
  }
  if (posRad.y < paramsD.cMin.y) {
    posRad.y += (paramsD.cMax.y - paramsD.cMin.y);
    posRadD[index] = posRad;
    if (rhoPresMu.w < -.1) {
      rhoPresMu.y = rhoPresMu.y - paramsD.deltaPress.y;
      rhoPresMuD[index] = rhoPresMu;
    }
    return;
  }
}
//--------------------------------------------------------------------------------------------------------------------------------
// applies periodic BC along z
__global__ void ApplyPeriodicBoundaryZKernel(Real3* posRadD, Real4* rhoPresMuD) {
  uint index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= numObjectsD.numAllMarkers) {
    return;
  }
  Real4 rhoPresMu = rhoPresMuD[index];
  if (fabs(rhoPresMu.w) < .1) {
    return;
  }  // no need to do anything if it is a boundary particle
  Real3 posRad = posRadD[index];
  if (posRad.z > paramsD.cMax.z) {
    posRad.z -= (paramsD.cMax.z - paramsD.cMin.z);
    posRadD[index] = posRad;
    if (rhoPresMu.w < -.1) {
      rhoPresMu.y = rhoPresMu.y + paramsD.deltaPress.z;
      rhoPresMuD[index] = rhoPresMu;
    }
    return;
  }
  if (posRad.z < paramsD.cMin.z) {
    posRad.z += (paramsD.cMax.z - paramsD.cMin.z);
    posRadD[index] = posRad;
    if (rhoPresMu.w < -.1) {
      rhoPresMu.y = rhoPresMu.y - paramsD.deltaPress.z;
      rhoPresMuD[index] = rhoPresMu;
    }
    return;
  }
}

//%%%%%%%%%%%%%%%%%%%%%%%%
//--------------------------------------------------------------------------------------------------------------------------------
void allocateArray(void** devPtr, size_t size) {
  cudaMalloc(devPtr, size);
}
//--------------------------------------------------------------------------------------------------------------------------------
void freeArray(void* devPtr) {
  cudaFree(devPtr);
}

/**
 * @brief iDivUp
 * @details Round a / b to nearest higher integer value
 *
 * @param a numerator
 * @param b denominator
 *
 * @return ceil(a/b)
 */
uint iDivUp(uint a, uint b) {
  return (a % b != 0) ? (a / b + 1) : (a / b);
}

/**
 * @brief computeGridSize
 * @details Compute grid and thread block size for a given number of elements
 *
 * @param n Total number of elements. Each elements needs a thread to be computed
 * @param blockSize Number of threads per block.
 * @param numBlocks output
 * @param numThreads Output: number of threads per block
 */
void computeGridSize(uint n, uint blockSize, uint& numBlocks, uint& numThreads) {
  uint n2 = (n == 0) ? 1 : n;
  numThreads = min(blockSize, n2);
  numBlocks = iDivUp(n2, numThreads);
}

/**
 * @brief [brief description]
 * @details [long description]
 *
 * @param hostParams [description]
 * @param numObjects [description]
 */
void setParameters(SimParams* hostParams, NumberOfObjects* numObjects) {
  // copy parameters to constant memory
  cudaMemcpyToSymbolAsync(paramsD, hostParams, sizeof(SimParams));
  cudaMemcpyToSymbolAsync(numObjectsD, numObjects, sizeof(NumberOfObjects));
}

/**
 * @brief Wrapper function for calcHashD
 * @details See SDKCollisionSystem.cuh for more info
 */
void calcHash(thrust::device_vector<uint>& gridMarkerHash,
              thrust::device_vector<uint>& gridMarkerIndex,
              thrust::device_vector<Real3>& posRad,
              int numAllMarkers) {
  bool *isErrorH, *isErrorD;
  isErrorH = (bool*)malloc(sizeof(bool));
  cudaMalloc((void**)&isErrorD, sizeof(bool));
  *isErrorH = false;
  cudaMemcpy(isErrorD, isErrorH, sizeof(bool), cudaMemcpyHostToDevice);
  //------------------------------------------------------------------------
  /* Is there a need to optimize the number of threads used at once? */
  uint numThreads, numBlocks;
  computeGridSize(numAllMarkers, 256, numBlocks, numThreads);
  /* Execute Kernel */
  calcHashD<<<numBlocks, numThreads>>>(U1CAST(gridMarkerHash), U1CAST(gridMarkerIndex), mR3CAST(posRad), numAllMarkers,
                                       isErrorD);

  /* Check for errors in kernel execution */
  cudaThreadSynchronize();
  cudaCheckError();
  //------------------------------------------------------------------------
  cudaMemcpy(isErrorH, isErrorD, sizeof(bool), cudaMemcpyDeviceToHost);
  if (*isErrorH == true) {
    throw std::runtime_error("Error! program crashed in  calcHashD!\n");
  }
  cudaFree(isErrorD);
  free(isErrorH);
}

/**
 * @brief Wrapper function for reorderDataAndFindCellStartD
 * @details
 * 		See SDKCollisionSystem.cuh for brief.
 */
void reorderDataAndFindCellStart(thrust::device_vector<uint>& cellStart,
                                 thrust::device_vector<uint>& cellEnd,
                                 thrust::device_vector<Real3>& sortedPosRad,
                                 thrust::device_vector<Real3>& sortedVelMas,
                                 thrust::device_vector<Real4>& sortedRhoPreMu,

                                 thrust::device_vector<uint>& gridMarkerHash,
                                 thrust::device_vector<uint>& gridMarkerIndex,

                                 thrust::device_vector<uint>& mapOriginalToSorted,

                                 thrust::device_vector<Real3>& oldPosRad,
                                 thrust::device_vector<Real3>& oldVelMas,
                                 thrust::device_vector<Real4>& oldRhoPreMu,
                                 uint numAllMarkers,
                                 uint numCells) {
  uint numThreads, numBlocks;
  computeGridSize(numAllMarkers, 256, numBlocks, numThreads);  //?$ 256 is blockSize

  /* Set all cells to empty */
  //	cudaMemset(U1CAST(cellStart), 0xffffffff, numCells * sizeof(uint));
  thrust::fill(cellStart.begin(), cellStart.end(), 0);
  thrust::fill(cellEnd.begin(), cellEnd.end(), 0);

  //#if USE_TEX
  //#if 0
  //    cutilSafeCall(cudaBindTexture(0, oldPosTex, oldPosRad, numAllMarkers*sizeof(Real4)));
  //    cutilSafeCall(cudaBindTexture(0, oldVelTex, oldVelMas, numAllMarkers*sizeof(Real4)));
  //#endif

  uint smemSize = sizeof(uint) * (numThreads + 1);
  reorderDataAndFindCellStartD<<<numBlocks, numThreads, smemSize>>>(
      U1CAST(cellStart), U1CAST(cellEnd), mR3CAST(sortedPosRad), mR3CAST(sortedVelMas), mR4CAST(sortedRhoPreMu),
      U1CAST(gridMarkerHash), U1CAST(gridMarkerIndex), U1CAST(mapOriginalToSorted), mR3CAST(oldPosRad),
      mR3CAST(oldVelMas), mR4CAST(oldRhoPreMu), numAllMarkers);
  cudaThreadSynchronize();
  cudaCheckError();

  // unroll sorted index to have the location of original particles in the sorted arrays
  thrust::device_vector<uint> dummyIndex = gridMarkerIndex;
  thrust::sort_by_key(dummyIndex.begin(), dummyIndex.end(), mapOriginalToSorted.begin());
  dummyIndex.clear();
  //#if USE_TEX
  //#if 0
  //    cutilSafeCall(cudaUnbindTexture(oldPosTex));
  //    cutilSafeCall(cudaUnbindTexture(oldVelTex));
  //#endif
}
//--------------------------------------------------------------------------------------------------------------------------------
__global__ void calcNormalizedRho_kernel(Real3* sortedPosRad,  // input: sorted positions
                                         Real4* sortedRhoPreMu,
                                         uint* cellStart,
                                         uint* cellEnd,
                                         int2 updatePortion,
                                         const int numAllMarkers,
                                         const Real RHO_0,
                                         const Real m_0,
                                         volatile bool* isErrorD) {
  uint i_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (i_idx > numAllMarkers) {
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
        //        if (i_idx == 43)
        //          printf("Going to next cell ...\n");
        int3 neighbourPos = gridPos + mI3(x, y, z);
        uint gridHash = calcGridHash(neighbourPos);
        // get start of bucket for this cell
        uint startIndex = cellStart[gridHash];
        if (startIndex != 0xffffffff) {  // cell is not empty
                                         // iterate over particles in this cell
          uint endIndex = cellEnd[gridHash];

          for (uint j = startIndex; j < endIndex; j++) {
            //            if (i_idx == j)
            //              continue;
            Real3 posRadB = sortedPosRad[j];
            Real3 dist3 = Distance(posRadA, posRadB);
            Real d = length(dist3);
            if (d > RESOLUTION_LENGTH_MULT * paramsD.HSML)
              continue;
            Real Wd = m_0 * W3(d);
            //            if (i_idx == 43)
            //              printf("id= %d, wd= %f, q_i=%f idA %d idB %d\n ", i_idx, Wd, d / paramsD.HSML, i_idx, j);
            sum_mW += Wd;
          }
        }
      }
    }
  }

  // Adding neighbor contribution is done!
  Real IncompressibilityFactor = 1;
  sortedRhoPreMu[i_idx].x = (sum_mW - RHO_0) * IncompressibilityFactor + RHO_0;
  if (sortedRhoPreMu[i_idx].x < EPSILON) {
    printf("My density is %f,ref density= %f\n", sortedRhoPreMu[i_idx].x, RHO_0);
  }
}
//--------------------------------------------------------------------------------------------------------------------------------
__global__ void F_i_np__AND__d_ii_kernel(Real3* sortedPosRad,  // input: sorted positions
                                         Real3* sortedVelMas,
                                         Real4* sortedRhoPreMu,
                                         Real3* d_ii,
                                         Real3* F_i_np,
                                         uint* cellStart,
                                         uint* cellEnd,
                                         int2 updatePortion,
                                         const int numAllMarkers,
                                         const Real m_0,
                                         const Real mu_0,
                                         const Real RHO_0,
                                         const Real epsilon,
                                         const Real dT,
                                         volatile bool* isErrorD) {
  uint i_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (i_idx > numAllMarkers) {
    return;
  }

  Real3 posi = sortedPosRad[i_idx];
  Real3 Veli = sortedVelMas[i_idx];
  Real Rhoi = sortedRhoPreMu[i_idx].x;
  Real3 My_d_ii = mR3(0);
  Real3 My_F_i_np = mR3(0);

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
            Real3 grad_a_wab = GradW(dist3) / (paramsD.HSML * d) * dist3;
            My_d_ii += m_0 * (-(dT * dT) / (Rhoi * Rhoi)) * grad_a_wab;
            Real Rho_bar = (Rhoj + Rhoi) * 0.5;
            Real3 V_ij = (Velj - Veli);
            Real3 muNumerator = -2 * mu_0 * dot(dist3, grad_a_wab) * V_ij;
            Real muDenominator = (Rho_bar * Rho_bar) * (d * d + paramsD.HSML * paramsD.HSML * epsilon);
            My_F_i_np += muNumerator / muDenominator;
          }
        }
      }
    }
  }
  // Adding neighbor contribution is done!
  d_ii[i_idx] = My_d_ii;
  F_i_np[i_idx] = My_F_i_np;

  //  if (i_idx == 1) {
  //    for (int j = 1; j < numAllMarkers; j++)
  //      printf("Density j = %d in F_i_np__AND__d_ii_kernel is %f\n", j, sortedRhoPreMu[j].x);
  //  }
}
//--------------------------------------------------------------------------------------------------------------------------------
__device__ void Calc_c_i(Real3& c_i,
                         const uint i_idx,
                         const Real3 pos_i,
                         const Real3 Vel_i,
                         const Real Rho_i,
                         Real3* sortedPosRad,
                         Real3* sortedVelMas,
                         Real4* sortedRhoPreMu,
                         Real* p_old,
                         uint* cellStart,
                         uint* cellEnd,
                         const Real m_0,
                         const Real dT) {
  if (sortedRhoPreMu[i_idx].x < EPSILON) {
    printf("My density is %f in Calc_c_i\n", sortedRhoPreMu[i_idx].x);
  }

  //  printf("i_idx= %d, My density is %f in Calc_c_i\n", i_idx, sortedRhoPreMu[i_idx].x);
  //  for (int j = 1; j < 5000; j++)
  //    if (sortedRhoPreMu[j].x == 0) {
  //      printf("Density is %f\n", sortedRhoPreMu[j].x);
  //    }

  Real3 My_c_i = mR3(0);
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
            if (i_idx == j)
              continue;
            Real3 pos_j = sortedPosRad[j];
            Real3 dist3 = Distance(pos_i, pos_j);
            Real d = length(dist3);
            ////CHECK THIS CONDITION!!!
            if (d > RESOLUTION_LENGTH_MULT * paramsD.HSML)
              continue;
            Real Rho_j = sortedRhoPreMu[j].x;
            if (Rho_j == 0) {
              //              printf("Bug here Calc_c_i i=%d j=%d\n", i_idx, j);
            }
            Real p_j_old = p_old[j];
            Real3 grad_a_wab = GradW(dist3) / (paramsD.HSML * d) * dist3;
            My_c_i += m_0 * (-(dT * dT) / (Rho_j * Rho_j)) * grad_a_wab * p_j_old;
          }
        }
      }
    }
  }
  c_i = My_c_i;
  //  for (int j = i_idx; j < i_idx + 10; j++)
  //    if (sortedRhoPreMu[j].x < EPSILON) {
  //      printf("Density in Calc_c_i %f\n", sortedRhoPreMu[j].x);
  //    }
}

////--------------------------------------------------------------------------------------------------------------------------------
//
__device__ void Calc_Others(Real& rho_np_i,
                            Real& cprime_i,
                            Real& dprime_i,
                            Real& eprime_i,
                            Real& a_ii,
                            const Real3 c_i,
                            Real3* d_ii,
                            Real3* F_i_np,
                            const uint i_idx,
                            const Real3 pos_i,
                            const Real3 Vel_i,
                            const Real Rho_i,
                            Real3* sortedPosRad,
                            Real3* sortedVelMas,
                            Real4* sortedRhoPreMu,
                            Real* p_old,
                            uint* cellStart,
                            uint* cellEnd,
                            const Real m_0,
                            const Real mu_0,
                            const Real RHO_0,
                            const Real epsilon,
                            const Real dT) {
  if (sortedRhoPreMu[i_idx].x == 0) {
    printf("My density is %f in Iterative_pressure_update\n", sortedRhoPreMu[i_idx].x);
  }
  Real my_rho_np = 0;
  Real my_cprime = 0;
  Real my_dprime = 0;
  Real my_eprime = 0;
  Real my_a_ii = 0;
  //  printf("I am inside the Calc_Others my id= %d\n", i_idx);

  int3 gridPosI = calcGridPos(pos_i);
  for (int z = -1; z <= 1; z++) {
    for (int y = -1; y <= 1; y++) {
      for (int x = -1; x <= 1; x++) {
        int3 neighbourPosI = gridPosI + mI3(x, y, z);
        uint gridHashI = calcGridHash(neighbourPosI);
        // get start of bucket for this cell
        uint startIndexI = cellStart[gridHashI];
        if (startIndexI != 0xffffffff) {  // cell is not empty
          // iterate over particles in this cell
          uint endIndexI = cellEnd[gridHashI];

          for (uint j = startIndexI; j < endIndexI; j++) {
            if (i_idx == j)
              continue;
            Real3 pos_j = sortedPosRad[j];
            Real3 dist3ij = Distance(pos_i, pos_j);
            Real dij = length(dist3ij);
            ////CHECK THIS CONDITION!!!
            if (dij > RESOLUTION_LENGTH_MULT * paramsD.HSML)
              continue;
            Real3 Vel_j = sortedVelMas[j];
            Real Rho_j = sortedRhoPreMu[j].x;
            Real p_j_old = p_old[j];
            //            if (Rho_j == 0)
            //              printf("Bug here i=%d j=%d\n", i_idx, j);
            //============================= Calc F_j_np, d_j, and e_j for the j
            Real3 F_np_j = mR3(0);
            Real3 d_j = mR3(0);
            Real3 e_j = mR3(0);

            int3 gridPosJ = calcGridPos(pos_j);
            for (int zz = -1; zz <= 1; zz++) {
              for (int yy = -1; yy <= 1; yy++) {
                for (int xx = -1; xx <= 1; xx++) {
                  int3 neighbourPosJ = gridPosJ + mI3(xx, yy, zz);
                  uint gridHashJ = calcGridHash(neighbourPosJ);
                  //                  // get start of bucket for this cell
                  uint startIndexJ = cellStart[gridHashJ];
                  if (startIndexJ != 0xffffffff) {  // cell is not empty
                                                    //                    // iterate over particles in this cell
                    uint endIndexJ = cellEnd[gridHashJ];
                    for (uint k = startIndexJ; k < endIndexJ; k++) {
                      if (k == j)
                        continue;
                      Real3 pos_k = sortedPosRad[k];

                      Real3 dist3jk = Distance(pos_j, pos_k);
                      Real djk = length(dist3jk);
                      if (djk > RESOLUTION_LENGTH_MULT * paramsD.HSML)
                        continue;
                      Real3 Vel_k = sortedVelMas[k];
                      Real p_k_old = p_old[k];
                      Real Rho_k = sortedRhoPreMu[k].x;
                      Real3 V_jk = (Vel_j - Vel_k);
                      Real3 grad_j_wjk = GradW(dist3jk) / (paramsD.HSML * djk) * dist3jk;
                      Real Rho_bar = (Rho_j + Rho_k) * 0.5;
                      Real3 muNumerator = -2 * mu_0 * dot(dist3jk, grad_j_wjk) * V_jk;
                      Real muDenominator = (Rho_bar * Rho_bar) * (djk * djk + paramsD.HSML * paramsD.HSML * epsilon);
                      F_np_j += m_0 * m_0 * muNumerator / muDenominator;
                      d_j += m_0 * grad_j_wjk;
                      if (Rho_k == 0)
                        printf("Bug here i=%d j=%d k=%d\n", i_idx, j, k);
                      if (k != i_idx)
                        e_j += 1 / (Rho_k * Rho_k) * m_0 * grad_j_wjk * p_k_old;
                    }
                  }
                }
              }
            }
            d_j = d_j * (-(dT * dT) / (Rho_j * Rho_j)) * p_j_old;
            e_j = e_j * (-dT * dT);
            //============================= done with calculation for j
            Real3 grad_i_wij = GradW(dist3ij) / (paramsD.HSML * dij) * dist3ij;

            Real3 V_ij = (Vel_i - Vel_j);
            Real3 F_np_i = F_i_np[i_idx];
            Real3 my_d_ii = d_ii[i_idx];
            Real3 toBeUsed = (V_ij + (F_np_i / m_0 - F_np_j / m_0) * dT);
            my_rho_np += m_0 * dot(toBeUsed, grad_i_wij) * dT;
            my_cprime += m_0 * dot(c_i, grad_i_wij);
            my_dprime += m_0 * dot(d_j, grad_i_wij);
            my_eprime += m_0 * dot(e_j, grad_i_wij);
            Real3 d_ji_temp = m_0 * (-(dT * dT) / (Rho_i * Rho_i)) * (-grad_i_wij);
            my_a_ii += m_0 * dot((my_d_ii - d_ji_temp), grad_i_wij);
          }
        }
      }
    }
  }
  rho_np_i = my_rho_np;
  cprime_i = my_cprime;
  dprime_i = my_dprime;
  eprime_i = my_eprime;
  a_ii = my_a_ii == 0 ? 1e10 : my_a_ii;
  if (my_a_ii == 0) {
    printf("a_ii= %f, rho_np_i= %f, cprime_i= %f, dprime_i= %f, eprime_i= %f\n", my_a_ii, my_rho_np, my_cprime,
           my_dprime, my_eprime);
  }
}
////--------------------------------------------------------------------------------------------------------------------------------
__device__ void Calc_BC_ADAMI(Real& pn,
                              Real3& Vn,
                              const uint i_idx,
                              const Real3 pos_i,
                              const Real Rho_i,
                              Real3* sortedPosRad,
                              Real3* sortedVelMas,
                              Real4* sortedRhoPreMu,
                              Real* p_old,
                              Real3* V_new,
                              uint* cellStart,
                              uint* cellEnd,
                              const Real m_0,
                              const Real mu_0,
                              const Real RHO_0,
                              const Real epsilon,
                              const Real dT,
                              const Real3 g) {
  Real3 numeratorv = mR3(0);
  Real denumenator = 0;
  Real numeratorp = 0;
  Real p_i;
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
            if (sortedRhoPreMu[j].w == -1)  // NO BCE BCE forces...
              continue;

            Real3 pos_j = sortedPosRad[j];
            Real3 dist3 = Distance(pos_i, pos_j);
            Real d = length(dist3);
            if (d > RESOLUTION_LENGTH_MULT * paramsD.HSML)
              continue;
            // OLD VELOCITY IS SHOULD BE OBDATED NOT THE NEW ONE!!!!!
            Real3 Vel_j = sortedVelMas[j];
            Real p_j = p_old[j];
            Real Wd = W3(d);
            if (isnan(numeratorp))

              numeratorv += Vel_j * Wd;
            numeratorp += p_j * Wd + dot(g, dist3) * Rho_i * Wd;
            denumenator += Wd;
            if (isnan(numeratorp))
              printf("Something is wrong here..., %f\n", numeratorp);
          }
        }
      }
    }
  }
  if (denumenator < EPSILON) {
    p_i = 0;
    Vel_i = mR3(0);
    if (isnan(denumenator) || isnan(numeratorp))
      printf("Something is wrong though ...\n");

  } else {
    Vel_i = -numeratorv / denumenator;
    p_i = numeratorp / denumenator;
  }
  pn = p_i;
  // IS THIS FAST???
  Vn = Vel_i;
}
//--------------------------------------------------------------------------------------------------------------------------------
__global__ void Iterative_pressure_update(Real3* sortedPosRad,  // input: sorted positions
                                          Real4* sortedRhoPreMu,
                                          Real* p_old,
                                          Real3* sortedVelMas,
                                          Real3* V_new,
                                          Real3* d_ii,
                                          Real3* F_i_np,
                                          uint* cellStart,
                                          uint* cellEnd,
                                          const int2 updatePortion,
                                          const int numAllMarkers,
                                          const Real m_0,
                                          const Real mu_0,
                                          const Real RHO_0,
                                          const Real epsilon,
                                          const Real dT,
                                          const Real3 g,
                                          volatile bool* isErrorD) {
  const uint i_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (i_idx > numAllMarkers) {
    return;
  }
  //  if (i_idx == 1) {
  //    for (int j = 1; j < numAllMarkers; j++) {
  //      Real4 Inja = sortedRhoPreMu[j];
  //      printf("Density j = %d in Iterative_pressure_update is %f\n", j, sortedRhoPreMu[j].x);
  //    }
  //  }

  //  if (sortedRhoPreMu[i_idx].x == 0) {
  //    printf("My density is %f,ref density= %f\n", sortedRhoPreMu[i_idx].x, RHO_0);
  //  }
  const Real3 pos_i = sortedPosRad[i_idx];
  // Should this be constant or not??
  Real Rho_i = sortedRhoPreMu[i_idx].x;
  const int my_type = sortedRhoPreMu[i_idx].w;
  Real3 Vel_i = sortedVelMas[i_idx];
  Real3 Vn;
  Real pn;
  //
  if (my_type == -1) {
    if (Rho_i > 0.95 * RHO_0) {
      /// DO IISPH expression here
      Real3 c_i = mR3(0);
      Real rho_np_i = 0;
      Real cprime_i = 0;
      Real dprime_i = 0;
      Real eprime_i = 0;
      Real a_ii = 0;
      //------------------------------------ Calculating c_i ----------------------------------------------------
      Calc_c_i(c_i, i_idx, pos_i, Vel_i, Rho_i, sortedPosRad, sortedVelMas, sortedRhoPreMu, p_old, cellStart, cellEnd,
               m_0, dT);
      if (sortedRhoPreMu[i_idx].x < EPSILON) {
        printf("My density is %f,ref density= %f\n", sortedRhoPreMu[i_idx].x, RHO_0);
      }
      //------------------------------------ Calculating other terms -----------------------------------------------
      // This function calculates rho_np, cprime_i, dprime_i,eprime_i
      Calc_Others(rho_np_i, cprime_i, dprime_i, eprime_i, a_ii, c_i, d_ii, F_i_np, i_idx, pos_i, Vel_i, Rho_i,
                  sortedPosRad, sortedVelMas, sortedRhoPreMu, p_old, cellStart, cellEnd, m_0, mu_0, RHO_0, epsilon, dT);
      //-------------------------------------------------------------------------------------------------------------
      pn = (RHO_0 - (Rho_i + rho_np_i) - cprime_i + dprime_i + eprime_i) / a_ii;
    } else {
      /// Free Surface BC on fluid
      pn = 0;
    }
    if (abs(pn) > 1e5)
      printf("Prone to divergence for IISPH solver\n ");
  } else if (my_type == 0) {
    Calc_BC_ADAMI(pn, Vn, i_idx, pos_i, Rho_i, sortedPosRad, sortedVelMas, sortedRhoPreMu, p_old, V_new, cellStart,
                  cellEnd, m_0, mu_0, RHO_0, epsilon, dT, g);
  }

  sortedRhoPreMu[i_idx].y = pn;
  V_new[i_idx] = (my_type == 0 ? Vn : mR3(0));
}
//--------------------------------------------------------------------------------------------------------------------------------
__global__ void Update_AND_Calc_Res(Real3* sortedVelMas,
                                    Real4* sortedRhoPreMu,
                                    Real* p_old,
                                    Real3* V_new,
                                    Real* Residuals,
                                    const int numAllMarkers,
                                    const int Iteration,
                                    volatile bool* isErrorD) {
  uint i_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (i_idx > numAllMarkers) {
    return;
  }
  Real relax;
  if (Iteration < 100)  // Over Relax to speed up
    relax = 0.5;
  else if (Iteration > 100 && Iteration < 200)  // No relaxation
    relax = 0.5;
  else  // Under-relaxation
    relax = 0.5;

  Real P_OLD_TEMP = sortedRhoPreMu[i_idx].y;
  //  p_i = (1 - relax) * p_old_i + relax * p_i;
  sortedRhoPreMu[i_idx].y = (1 - relax) * p_old[i_idx] + relax * sortedRhoPreMu[i_idx].y;
  Residuals[i_idx] = abs(sortedRhoPreMu[i_idx].y - p_old[i_idx]) / (p_old[i_idx] == 0 ? 1e10 : abs(p_old[i_idx]));
  p_old[i_idx] = P_OLD_TEMP;
  if (sortedRhoPreMu[i_idx].w == 0)
    sortedVelMas[i_idx] = V_new[i_idx];
}
//--------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------
//-------------------------------------Pressure Solver------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------
void calcPressureIISPH(const thrust::device_vector<Real3>& sortedPosRad,
                       thrust::device_vector<Real3>& sortedVelMas,
                       thrust::device_vector<Real4>& sortedRhoPreMu,
                       const thrust::device_vector<uint> cellStart,
                       const thrust::device_vector<uint> cellEnd,
                       const thrust::device_vector<uint> mapOriginalToSorted,
                       const SimParams paramsH,
                       const NumberOfObjects& numObjects,
                       const int2 updatePortion,
                       const Real dT,
                       const Real RES) {
  bool *isErrorH, *isErrorD;
  isErrorH = (bool*)malloc(sizeof(bool));
  cudaMalloc((void**)&isErrorD, sizeof(bool));
  *isErrorH = false;
  cudaMemcpy(isErrorD, isErrorH, sizeof(bool), cudaMemcpyHostToDevice);
  //------------------------------------------------------------------------
  // thread per particle
  uint numThreads, numBlocks;
  int numAllMarkers = numObjects.numBoundaryMarkers + numObjects.numFluidMarkers;
  computeGridSize(numAllMarkers, 256, numBlocks, numThreads);
  printf("numBlocks: %d, numThreads: %d, numAllMarker:%d \n", numBlocks, numThreads, numAllMarkers);
  //------------------------------------------------------------------------
  //------------------------------------------------------------------------
  //-------------Calculation of Densities
  //------------------------------------------------------------------------

  calcNormalizedRho_kernel<<<numBlocks, numThreads>>>(mR3CAST(sortedPosRad),
                                                      mR4CAST(sortedRhoPreMu),  // input: sorted velocities
                                                      U1CAST(cellStart), U1CAST(cellEnd), updatePortion, numAllMarkers,
                                                      paramsH.rho0, paramsH.markerMass, isErrorD);

  // This is mandatory to sync here
  cudaThreadSynchronize();
  cudaCheckError();
  cudaMemcpy(isErrorH, isErrorD, sizeof(bool), cudaMemcpyDeviceToHost);
  if (*isErrorH == true) {
    throw std::runtime_error("Error! program crashed after calcNormalizedRho_kernel!\n");
  }

  //------------------------------------------------------------------------
  //------------------------------------------------------------------------
  //-------------Calculation of F_i_np, and d_ii
  //------------------------------------------------------------------------
  thrust::device_vector<Real3> d_ii(numAllMarkers);
  thrust::device_vector<Real3> F_i_np(numAllMarkers);
  *isErrorH = false;
  cudaMemcpy(isErrorD, isErrorH, sizeof(bool), cudaMemcpyHostToDevice);
  F_i_np__AND__d_ii_kernel<<<numBlocks, numThreads>>>(
      mR3CAST(sortedPosRad), mR3CAST(sortedVelMas), mR4CAST(sortedRhoPreMu), mR3CAST(d_ii), mR3CAST(F_i_np),
      U1CAST(cellStart), U1CAST(cellEnd), updatePortion, numAllMarkers, paramsH.markerMass, paramsH.mu0, paramsH.rho0,
      paramsH.epsMinMarkersDis, dT, isErrorD);

  cudaThreadSynchronize();
  cudaCheckError();
  cudaMemcpy(isErrorH, isErrorD, sizeof(bool), cudaMemcpyDeviceToHost);
  if (*isErrorH == true) {
    throw std::runtime_error("Error! program crashed after F_i_np__AND__d_ii_kernel!\n");
  }

  //------------------------------------------------------------------------
  //------------- Iterative loop
  //------------------------------------------------------------------------
  int Iteration = 0;
  Real MaxRes = 1;
  thrust::device_vector<Real> p_old(numAllMarkers);  // This has P_old
  thrust::fill(p_old.begin(), p_old.end(), 0);
  thrust::device_vector<Real> Residuals(numAllMarkers);  // This has res
  thrust::fill(Residuals.begin(), Residuals.end(), 1);
  thrust::device_vector<Real3> V_new(numAllMarkers);  // This has res

  while (MaxRes > RES || Iteration < 3) {
    *isErrorH = false;
    cudaMemcpy(isErrorD, isErrorH, sizeof(bool), cudaMemcpyHostToDevice);
    Iterative_pressure_update<<<numBlocks, numThreads>>>(
        mR3CAST(sortedPosRad), mR4CAST(sortedRhoPreMu), R1CAST(p_old), mR3CAST(sortedVelMas), mR3CAST(V_new),
        mR3CAST(d_ii), mR3CAST(F_i_np), U1CAST(cellStart), U1CAST(cellEnd), updatePortion, numAllMarkers,
        paramsH.markerMass, paramsH.mu0, paramsH.rho0, paramsH.epsMinMarkersDis, dT, paramsH.gravity, isErrorD);
    cudaThreadSynchronize();
    cudaCheckError();
    cudaMemcpy(isErrorH, isErrorD, sizeof(bool), cudaMemcpyDeviceToHost);
    if (*isErrorH == true) {
      throw std::runtime_error("Error! program crashed after Iterative_pressure_update!\n");
    }

    *isErrorH = false;
    cudaMemcpy(isErrorD, isErrorH, sizeof(bool), cudaMemcpyHostToDevice);
    Update_AND_Calc_Res<<<numBlocks, numThreads>>>(mR3CAST(sortedVelMas), mR4CAST(sortedRhoPreMu), R1CAST(p_old),
                                                   mR3CAST(V_new), R1CAST(Residuals), numAllMarkers, Iteration,
                                                   isErrorD);
    cudaThreadSynchronize();
    cudaCheckError();
    cudaMemcpy(isErrorH, isErrorD, sizeof(bool), cudaMemcpyDeviceToHost);
    if (*isErrorH == true) {
      throw std::runtime_error("Error! program crashed after Iterative_pressure_update!\n");
    }

    Iteration++;
    MaxRes = thrust::reduce(Residuals.begin(), Residuals.end(), 0.0, thrust::plus<Real>()) / Residuals.size();
    printf("Iteration= %d, MaxRes= %f\n", Iteration, MaxRes);
  }

  //------------------------------------------------------------------------
  //------------------------------------------------------------------------
  //------------------------------------------------------------------------
  cudaFree(isErrorD);
  free(isErrorH);
  //--------------------------------------------------------------------------------------------------------------------------------
  //-------------------------------------End Of Pressure Solver------------------------------------------------------
  //--------------------------------------------------------------------------------------------------------------------------------
}

//--------------------------------------------------------------------------------------------------------------------------------
__global__ void CalcForces(Real3* new_Pos,       // input: sorted positions
                           Real3* new_vel,       // input: sorted velocities,
                           Real3* sortedPosRad,  // input: sorted positions
                           Real3* sortedVelMas,  // input: sorted velocities
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
  uint i_idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i_idx > numAllMarkers)
    return;
  // Do this only for fluid
  if (sortedRhoPreMu[i_idx].w > -1)
    return;

  Real3 posi = sortedPosRad[i_idx];
  Real3 Veli = sortedVelMas[i_idx];
  Real p_i = sortedRhoPreMu[i_idx].y;
  Real rho_i = sortedRhoPreMu[i_idx].x;
  Real3 F_i_np, F_i_p;
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
            Real3 dist3 = Distance(posj, posi);
            Real d = length(dist3);
            if (d > RESOLUTION_LENGTH_MULT * paramsD.HSML)
              continue;
            Real3 Velj = sortedVelMas[j];
            Real p_j = sortedRhoPreMu[j].y;
            Real rho_j = sortedRhoPreMu[j].x;
            if (rho_j < EPSILON) {
              printf("j= %d returns NAN", j);
              rho_j = 1000;
            }
            Real3 grad_i_wij = GradW(dist3) / (paramsD.HSML * d) * dist3;
            Real3 V_ij = (Veli - Velj);
            F_i_p += (m_0 * ((p_i / (rho_i * rho_i)) + (p_j / (rho_j * rho_j)))) * grad_i_wij;

            Real Rho_bar = (rho_j + rho_i) * 0.5;
            Real3 muNumerator = -2 * mu_0 * dot(dist3, grad_i_wij) * V_ij;
            Real muDenominator = (Rho_bar * Rho_bar) * (d * d + paramsD.HSML * paramsD.HSML * epsilon);
            F_i_np += m_0 * m_0 * muNumerator / muDenominator;
          }
        }
      }
    }
  }
  F_i_p = -m_0 * F_i_p;
  F_i_np = m_0 * F_i_np;

  new_vel[i_idx] = Veli + dT / m_0 * (F_i_p + F_i_np) + gravity * dT;
  new_Pos[i_idx] = posi + dT * Veli;
  // Adding neighbor contribution is done!
}
//--------------------------------------------------------------------------------------------------------------------------------
__global__ void Update_Fluid_State(Real3* new_Pos,       // input: sorted positions
                                   Real3* new_vel,       // input: sorted velocities,
                                   Real3* sortedPosRad,  // input: sorted positions
                                   Real3* sortedVelMas,
                                   Real4* sortedRhoPreMu,
                                   const uint numAllMarkers,  // input: sorted velocities
                                   volatile bool* isErrorD) {
  uint i_idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i_idx > numAllMarkers)
    return;
  if (sortedRhoPreMu[i_idx].w > -1)
    return;
  sortedPosRad[i_idx] = new_Pos[i_idx];
  sortedVelMas[i_idx] = new_vel[i_idx];
}

//%%%%%%%%%%%%%%%%%%%%%%%%
//--------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------
void RecalcSortedVelocityPressure_BCE(thrust::device_vector<Real3>& velMas_ModifiedBCE,
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
//--------------------------------------------------------------------------------------------------------------------------------

/**
 * @brief Wrapper function for collide
 * @details
 * 		See SDKCollisionSystem.cuh for informaton on collide
 */
void Update_FluidIISPH(thrust::device_vector<Real3>& sortedPosRad,
                       thrust::device_vector<Real3>& sortedVelMas,
                       thrust::device_vector<Real4>& sortedRhoPreMu,
                       thrust::device_vector<uint>& cellStart,
                       thrust::device_vector<uint>& cellEnd,
                       uint numAllMarkers,
                       const SimParams paramsH,
                       Real dT) {
  thrust::device_vector<Real3> New_vel(numAllMarkers);  // Store Velocities of each particle in the device memory
  thrust::device_vector<Real3> New_pos(numAllMarkers);  // Store positions of each particle in the device memory

  bool *isErrorH, *isErrorD;
  isErrorH = (bool*)malloc(sizeof(bool));
  cudaMalloc((void**)&isErrorD, sizeof(bool));
  *isErrorH = false;
  cudaMemcpy(isErrorD, isErrorH, sizeof(bool), cudaMemcpyHostToDevice);
  //------------------------------------------------------------------------
  // thread per particle
  uint numThreads, numBlocks;
  computeGridSize(numAllMarkers, 64, numBlocks, numThreads);
  CalcForces<<<numBlocks, numThreads>>>(mR3CAST(New_vel), mR3CAST(New_pos), mR3CAST(sortedPosRad),
                                        mR3CAST(sortedVelMas), mR4CAST(sortedRhoPreMu), U1CAST(cellStart),
                                        U1CAST(cellEnd), numAllMarkers, paramsH.markerMass, paramsH.mu0,
                                        paramsH.epsMinMarkersDis, dT, paramsH.gravity, isErrorD);
  cudaThreadSynchronize();
  cudaCheckError();
  cudaMemcpy(isErrorH, isErrorD, sizeof(bool), cudaMemcpyDeviceToHost);
  if (*isErrorH == true) {
    throw std::runtime_error("Error! program crashed in CalcForces!\n");
  }
  //------------------------------------------------------------------------

  *isErrorH = false;
  cudaMemcpy(isErrorD, isErrorH, sizeof(bool), cudaMemcpyHostToDevice);
  // thread per particle
  Update_Fluid_State<<<numBlocks, numThreads>>>(mR3CAST(New_vel), mR3CAST(New_pos), mR3CAST(sortedPosRad),
                                                mR3CAST(sortedVelMas), mR4CAST(sortedRhoPreMu), numAllMarkers,
                                                isErrorD);
  cudaThreadSynchronize();
  cudaCheckError();

  cudaMemcpy(isErrorH, isErrorD, sizeof(bool), cudaMemcpyDeviceToHost);
  if (*isErrorH == true) {
    throw std::runtime_error("Error! program crashed in  Update_Fluid_State!\n");
  }
  //------------------------------------------------------------------------

  cudaFree(isErrorD);
  free(isErrorH);
}

//--------------------------------------------------------------------------------------------------------------------------------
// updates the fluid particles by calling UpdateFluidD
void UpdateFluid_implicit(thrust::device_vector<Real3>& posRadD,
                          thrust::device_vector<Real3>& velMasD,
                          thrust::device_vector<Real4>& rhoPresMuD,
                          thrust::device_vector<Real4>& derivVelRhoD,
                          const thrust::host_vector<int4>& referenceArray,
                          Real dT) {
  //	int4 referencePortion = referenceArray[0];
  //	if (referencePortion.z != -1) {
  //		printf("error in UpdateFluid, accessing non fluid\n");
  //		return;
  //	}
  //	int2 updatePortion = mI2(referencePortion);
  int2 updatePortion = mI2(0, referenceArray[referenceArray.size() - 1].y);
  // int2 updatePortion = mI2(referenceArray[0].x, referenceArray[0].y);

  bool *isErrorH, *isErrorD;
  isErrorH = (bool*)malloc(sizeof(bool));
  cudaMalloc((void**)&isErrorD, sizeof(bool));
  *isErrorH = false;
  cudaMemcpy(isErrorD, isErrorH, sizeof(bool), cudaMemcpyHostToDevice);
  //------------------------
  uint nBlock_UpdateFluid, nThreads;
  computeGridSize(updatePortion.y - updatePortion.x, 128, nBlock_UpdateFluid, nThreads);
  UpdateFluidD_implicit<<<nBlock_UpdateFluid, nThreads>>>(mR3CAST(posRadD), mR3CAST(velMasD), mR4CAST(rhoPresMuD),
                                                          mR4CAST(derivVelRhoD), updatePortion, dT, isErrorD);
  cudaThreadSynchronize();
  cudaCheckError();
  //------------------------
  cudaMemcpy(isErrorH, isErrorD, sizeof(bool), cudaMemcpyDeviceToHost);
  if (*isErrorH == true) {
    throw std::runtime_error("Error! program crashed in  UpdateFluidD!\n");
  }
  cudaFree(isErrorD);
  free(isErrorH);
}

//--------------------------------------------------------------------------------------------------------------------------------
// use invasive to avoid one extra copy. However, keep in mind that sorted is changed.
void CopySortedToOriginal_Invasive_R3(thrust::device_vector<Real3>& original,
                                      thrust::device_vector<Real3>& sorted,
                                      const thrust::device_vector<uint>& gridMarkerIndex) {
  thrust::device_vector<uint> dummyMarkerIndex = gridMarkerIndex;
  thrust::sort_by_key(dummyMarkerIndex.begin(), dummyMarkerIndex.end(), sorted.begin());
  dummyMarkerIndex.clear();
  thrust::copy(sorted.begin(), sorted.end(), original.begin());
}
//--------------------------------------------------------------------------------------------------------------------------------
void CopySortedToOriginal_NonInvasive_R3(thrust::device_vector<Real3>& original,
                                         thrust::device_vector<Real3>& sorted,
                                         const thrust::device_vector<uint>& gridMarkerIndex) {
  thrust::device_vector<Real3> dummySorted = sorted;
  CopySortedToOriginal_Invasive_R3(original, dummySorted, gridMarkerIndex);
}
//--------------------------------------------------------------------------------------------------------------------------------
// use invasive to avoid one extra copy. However, keep in mind that sorted is changed.
void CopySortedToOriginal_Invasive_R4(thrust::device_vector<Real4>& original,
                                      thrust::device_vector<Real4>& sorted,
                                      const thrust::device_vector<uint>& gridMarkerIndex) {
  thrust::device_vector<uint> dummyMarkerIndex = gridMarkerIndex;
  thrust::sort_by_key(dummyMarkerIndex.begin(), dummyMarkerIndex.end(), sorted.begin());
  dummyMarkerIndex.clear();
  thrust::copy(sorted.begin(), sorted.end(), original.begin());
}
//--------------------------------------------------------------------------------------------------------------------------------
void CopySortedToOriginal_NonInvasive_R4(thrust::device_vector<Real4>& original,
                                         thrust::device_vector<Real4>& sorted,
                                         const thrust::device_vector<uint>& gridMarkerIndex) {
  thrust::device_vector<Real4> dummySorted = sorted;
  CopySortedToOriginal_Invasive_R4(original, dummySorted, gridMarkerIndex);
}

//--------------------------------------------------------------------------------------------------------------------------------
// updates the fluid particles by calling UpdateBoundary
void UpdateBoundary(thrust::device_vector<Real4>& rhoPresMuD,
                    thrust::device_vector<Real4>& derivVelRhoD,
                    const thrust::host_vector<int4>& referenceArray,
                    Real dT) {
  int4 referencePortion = referenceArray[1];
  if (referencePortion.z != 0) {
    printf("error in UpdateBoundary, accessing non boundary\n");
    return;
  }
  int2 updatePortion = mI2(referencePortion);

  uint nBlock_UpdateFluid, nThreads;
  computeGridSize(updatePortion.y - updatePortion.x, 128, nBlock_UpdateFluid, nThreads);
  UpdateKernelBoundary<<<nBlock_UpdateFluid, nThreads>>>(mR4CAST(rhoPresMuD), mR4CAST(derivVelRhoD), updatePortion, dT);
  cudaThreadSynchronize();
  cudaCheckError();
}

/**
 * @brief ApplyBoundarySPH_Markers
 * @details
 * 		See SDKCollisionSystem.cuh for more info
 */
void ApplyBoundarySPH_Markers(thrust::device_vector<Real3>& posRadD,
                              thrust::device_vector<Real4>& rhoPresMuD,
                              int numAllMarkers) {
  uint nBlock_NumSpheres, nThreads_SphMarkers;
  computeGridSize(numAllMarkers, 256, nBlock_NumSpheres, nThreads_SphMarkers);
  ApplyPeriodicBoundaryXKernel<<<nBlock_NumSpheres, nThreads_SphMarkers>>>(mR3CAST(posRadD), mR4CAST(rhoPresMuD));
  cudaThreadSynchronize();
  cudaCheckError();
  // these are useful anyway for out of bound particles
  ApplyPeriodicBoundaryYKernel<<<nBlock_NumSpheres, nThreads_SphMarkers>>>(mR3CAST(posRadD), mR4CAST(rhoPresMuD));
  cudaThreadSynchronize();
  cudaCheckError();
  ApplyPeriodicBoundaryZKernel<<<nBlock_NumSpheres, nThreads_SphMarkers>>>(mR3CAST(posRadD), mR4CAST(rhoPresMuD));
  cudaThreadSynchronize();
  cudaCheckError();

  //	SetOutputPressureToZero_X<<<nBlock_NumSpheres, nThreads_SphMarkers>>>(mR3CAST(posRadD),
  // mR4CAST(rhoPresMuD));
  //    cudaThreadSynchronize();
  //    cudaCheckError();
}
