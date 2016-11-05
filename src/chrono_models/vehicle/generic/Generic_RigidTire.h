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
// Authors: Radu Serban
// =============================================================================
//
// Generic rigid tire subsystem
//
// =============================================================================

#ifndef GENERIC_RIGID_TIRE_H
#define GENERIC_RIGID_TIRE_H

#include "chrono_vehicle/wheeled_vehicle/tire/ChRigidTire.h"

#include "chrono_models/ChApiModels.h"

namespace chrono {
namespace vehicle {
namespace generic {

class CH_MODELS_API Generic_RigidTire : public ChRigidTire {
  public:
    Generic_RigidTire(const std::string& name);

    ~Generic_RigidTire() {}

    virtual double GetRadius() const override { return m_radius; }
    virtual double GetWidth() const override { return m_width; }

  private:
    static const double m_radius;
    static const double m_width;
};

}  // end namespace generic
}  // end namespace vehicle
}  // end namespace chrono

#endif
