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
// Authors: Radu Serban, Justin Madsen, Daniel Melanz
// =============================================================================
//
// Front and Rear HMMWV suspension subsystems (double A-arm)
//
// These concrete suspension subsystems are defined with respect to right-handed
// frames with X pointing towards the front, Y to the left, and Z up (as imposed
// by the base class ChDoubleWishbone) and origins at the midpoint between the
// lower control arms' connection points to the chassis.
//
// All point locations are provided for the left half of the supspension.
//
// =============================================================================

#ifndef HMMWV_DOUBLEWISHBONE_H
#define HMMWV_DOUBLEWISHBONE_H

#include "chrono_vehicle/wheeled_vehicle/suspension/ChDoubleWishbone.h"

#include "chrono_models/ChApiModels.h"

namespace chrono {
namespace vehicle {
namespace hmmwv {

class CH_MODELS_API HMMWV_DoubleWishboneFront : public ChDoubleWishbone {
  public:
    HMMWV_DoubleWishboneFront(const std::string& name);
    ~HMMWV_DoubleWishboneFront();

    virtual double getSpindleMass() const override { return m_spindleMass; }
    virtual double getUCAMass() const override { return m_UCAMass; }
    virtual double getLCAMass() const override { return m_LCAMass; }
    virtual double getUprightMass() const override { return m_uprightMass; }

    virtual double getSpindleRadius() const override { return m_spindleRadius; }
    virtual double getSpindleWidth() const override { return m_spindleWidth; }
    virtual double getUCARadius() const override { return m_UCARadius; }
    virtual double getLCARadius() const override { return m_LCARadius; }
    virtual double getUprightRadius() const override { return m_uprightRadius; }

    virtual const ChVector<>& getSpindleInertia() const override { return m_spindleInertia; }
    virtual const ChVector<>& getUCAInertia() const override { return m_UCAInertia; }
    virtual const ChVector<>& getLCAInertia() const override { return m_LCAInertia; }
    virtual const ChVector<>& getUprightInertia() const override { return m_uprightInertia; }

    virtual double getAxleInertia() const override { return m_axleInertia; }

    virtual double getSpringRestLength() const override { return m_springRestLength; }
    virtual ChSpringForceCallback* getSpringForceCallback() const override { return m_springForceCB; }
    virtual ChSpringForceCallback* getShockForceCallback() const override { return m_shockForceCB; }

  private:
    virtual const ChVector<> getLocation(PointId which) override;

    ChSpringForceCallback* m_springForceCB;
    ChSpringForceCallback* m_shockForceCB;

    static const double m_spindleMass;
    static const double m_UCAMass;
    static const double m_LCAMass;
    static const double m_uprightMass;

    static const double m_spindleRadius;
    static const double m_spindleWidth;
    static const double m_UCARadius;
    static const double m_LCARadius;
    static const double m_uprightRadius;

    static const ChVector<> m_spindleInertia;
    static const ChVector<> m_UCAInertia;
    static const ChVector<> m_LCAInertia;
    static const ChVector<> m_uprightInertia;

    static const double m_axleInertia;

    static const double m_springCoefficient;
    static const double m_springRestLength;
};

// -----------------------------------------------------------------------------

class CH_MODELS_API HMMWV_DoubleWishboneRear : public ChDoubleWishbone {
  public:
    HMMWV_DoubleWishboneRear(const std::string& name);
    ~HMMWV_DoubleWishboneRear();

    virtual double getSpindleMass() const override { return m_spindleMass; }
    virtual double getUCAMass() const override { return m_UCAMass; }
    virtual double getLCAMass() const override { return m_LCAMass; }
    virtual double getUprightMass() const override { return m_uprightMass; }

    virtual double getSpindleRadius() const override { return m_spindleRadius; }
    virtual double getSpindleWidth() const override { return m_spindleWidth; }
    virtual double getUCARadius() const override { return m_UCARadius; }
    virtual double getLCARadius() const override { return m_LCARadius; }
    virtual double getUprightRadius() const override { return m_uprightRadius; }

    virtual const ChVector<>& getSpindleInertia() const override { return m_spindleInertia; }
    virtual const ChVector<>& getUCAInertia() const override { return m_UCAInertia; }
    virtual const ChVector<>& getLCAInertia() const override { return m_LCAInertia; }
    virtual const ChVector<>& getUprightInertia() const override { return m_uprightInertia; }

    virtual double getAxleInertia() const override { return m_axleInertia; }

    virtual double getSpringRestLength() const override { return m_springRestLength; }
    virtual ChSpringForceCallback* getSpringForceCallback() const override { return m_springForceCB; }
    virtual ChSpringForceCallback* getShockForceCallback() const override { return m_shockForceCB; }

  private:
    virtual const ChVector<> getLocation(PointId which) override;

    ChSpringForceCallback* m_springForceCB;
    ChSpringForceCallback* m_shockForceCB;

    static const double m_spindleMass;
    static const double m_UCAMass;
    static const double m_LCAMass;
    static const double m_uprightMass;

    static const double m_spindleRadius;
    static const double m_spindleWidth;
    static const double m_UCARadius;
    static const double m_LCARadius;
    static const double m_uprightRadius;

    static const ChVector<> m_spindleInertia;
    static const ChVector<> m_UCAInertia;
    static const ChVector<> m_LCAInertia;
    static const ChVector<> m_uprightInertia;

    static const double m_axleInertia;

    static const double m_springCoefficient;
    static const double m_springRestLength;
};

}  // end namespace hmmwv
}  // end namespace vehicle
}  // end namespace chrono

#endif
