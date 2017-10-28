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
//
// Base class for processing sph force in fsi system.//
// =============================================================================
#include <thrust/extrema.h>
#include <thrust/sort.h>
#include "chrono_fsi/ChFsiForceI2SPH.cuh"
#include "chrono_fsi/ChFsiForceParallel.cuh"

//==========================================================================================================================================
namespace chrono {
namespace fsi {

ChFsiForceI2SPH::~ChFsiForceI2SPH() {}
ChFsiForceI2SPH::ChFsiForceI2SPH(
    ChBce* otherBceWorker,                   ///< Pointer to the ChBce object that handles BCE markers
    SphMarkerDataD* otherSortedSphMarkersD,  ///< Information of markers in the sorted array on device
    ProximityDataD* otherMarkersProximityD,  ///< Pointer to the object that holds the proximity of the
                                             ///< markers on device
    FsiGeneralData* otherFsiGeneralData,     ///< Pointer to the sph general data
    SimParams* otherParamsH,                 ///< Pointer to the simulation parameters on host
    NumberOfObjects* otherNumObjects         ///< Pointer to number of objects, fluid and boundary markers, etc.
    )
    : ChFsiForceParallel(otherBceWorker,
                         otherSortedSphMarkersD,
                         otherMarkersProximityD,
                         otherFsiGeneralData,
                         otherParamsH,
                         otherNumObjects) {}

void ChFsiForceI2SPH::ForceIISPH(SphMarkerDataD* otherSphMarkersD,
                                 FsiBodiesDataD* otherFsiBodiesD,
                                 FsiMeshDataD* otherFsiMeshD) {}

}  // namespace fsi
}  // namespace chrono
