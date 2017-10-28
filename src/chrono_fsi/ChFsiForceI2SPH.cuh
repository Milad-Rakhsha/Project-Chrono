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

#ifndef CH_FSI_FORCEI2SPH_H_
#define CH_FSI_FORCEI2SPH_H_
#include "chrono_fsi/ChApiFsi.h"
#include "chrono_fsi/ChDeviceUtils.cuh"
#include "chrono_fsi/ChFsiForceParallel.cuh"
#include "chrono_fsi/ChSphGeneral.cuh"

namespace chrono {
namespace fsi {

/// @addtogroup fsi_physics
/// @{

/// @brief Child class of ChForceParallel that implements the I2SPH method.
class CH_FSI_API ChFsiForceI2SPH : public ChFsiForceParallel {
  public:
    ChFsiForceI2SPH(
        ChBce* otherBceWorker,                   ///< Pointer to the ChBce object that handles BCE markers
        SphMarkerDataD* otherSortedSphMarkersD,  ///< Information of markers in the sorted array on device
        ProximityDataD*
            otherMarkersProximityD,  ///< Pointer to the object that holds the proximity of the markers on device
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
    virtual ~ChFsiForceI2SPH() {}
    void Finalize() override;

  private:
    void ForceImplicitSPH(SphMarkerDataD* otherSphMarkersD,
                          FsiBodiesDataD* otherFsiBodiesD,
                          FsiMeshDataD* fsiMeshD) override;
};

/// @} fsi_physics

}  // end namespace fsi
}  // end namespace chrono

#endif
