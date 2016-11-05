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
// Authors: Alessandro Tasora, Radu Serban
// =============================================================================
//
// HMMWV 2WD driveline model based on ChShaft objects.
//
// =============================================================================

#ifndef HMMWV_DRIVELINE_2WD_H
#define HMMWV_DRIVELINE_2WD_H

#include "chrono_vehicle/wheeled_vehicle/driveline/ChShaftsDriveline2WD.h"

#include "chrono_models/ChApiModels.h"

namespace chrono {
namespace vehicle {
namespace hmmwv {

class CH_MODELS_API HMMWV_Driveline2WD : public ChShaftsDriveline2WD {
  public:
    HMMWV_Driveline2WD(const std::string& name);

    ~HMMWV_Driveline2WD() {}

    virtual double GetDriveshaftInertia() const override { return m_driveshaft_inertia; }
    virtual double GetDifferentialBoxInertia() const override { return m_differentialbox_inertia; }

    virtual double GetConicalGearRatio() const override { return m_conicalgear_ratio; }
    virtual double GetDifferentialRatio() const override { return m_differential_ratio; }

  private:
    // Shaft inertias.
    static const double m_driveshaft_inertia;
    static const double m_differentialbox_inertia;

    // Gear ratios.
    static const double m_conicalgear_ratio;
    static const double m_differential_ratio;
};

}  // end namespace hmmwv
}  // end namespace vehicle
}  // end namespace chrono

#endif
