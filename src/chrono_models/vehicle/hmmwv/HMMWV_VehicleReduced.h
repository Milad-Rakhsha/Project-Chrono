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
// Authors: Radu Serban, Justin Madsen
// =============================================================================
//
// HMMWV 9-body vehicle model...
//
// =============================================================================

#ifndef HMMWV_VEHICLE_REDUCED_H
#define HMMWV_VEHICLE_REDUCED_H

#include "chrono_models/ChApiModels.h"
#include "chrono_models/vehicle/hmmwv/HMMWV_Vehicle.h"
#include "chrono_models/vehicle/hmmwv/HMMWV_Chassis.h"
#include "chrono_models/vehicle/hmmwv/HMMWV_BrakeSimple.h"
#include "chrono_models/vehicle/hmmwv/HMMWV_DoubleWishboneReduced.h"
#include "chrono_models/vehicle/hmmwv/HMMWV_Driveline2WD.h"
#include "chrono_models/vehicle/hmmwv/HMMWV_Driveline4WD.h"
#include "chrono_models/vehicle/hmmwv/HMMWV_SimpleDriveline.h"
#include "chrono_models/vehicle/hmmwv/HMMWV_RackPinion.h"
#include "chrono_models/vehicle/hmmwv/HMMWV_Wheel.h"

namespace chrono {
namespace vehicle {
namespace hmmwv {

class CH_MODELS_API HMMWV_VehicleReduced : public HMMWV_Vehicle {
  public:
    HMMWV_VehicleReduced(const bool fixed = false,
                         DrivelineType driveType = DrivelineType::AWD,
                         ChMaterialSurfaceBase::ContactMethod contactMethod = ChMaterialSurfaceBase::DVI);

    HMMWV_VehicleReduced(ChSystem* system, const bool fixed = false, DrivelineType driveType = DrivelineType::AWD);

    ~HMMWV_VehicleReduced();

    virtual void Initialize(const ChCoordsys<>& chassisPos, double chassisFwdVel = 0) override;

  private:
    void Create(bool fixed);
};

}  // end namespace hmmwv
}  // end namespace vehicle
}  // end namespace chrono

#endif
