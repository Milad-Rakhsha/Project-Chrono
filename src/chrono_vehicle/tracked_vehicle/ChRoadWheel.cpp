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
// Base class for a road wheel.
//
// The reference frame for a vehicle follows the ISO standard: Z-axis up, X-axis
// pointing forward, and Y-axis towards the left of the vehicle.
//
// =============================================================================

#include "chrono_vehicle/tracked_vehicle/ChRoadWheel.h"

namespace chrono {
namespace vehicle {

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
ChRoadWheel::ChRoadWheel(const std::string& name)
    : ChPart(name),
      m_friction(0.7f),
      m_restitution(0.1f),
      m_young_modulus(1e8f),
      m_poisson_ratio(0.3f),
      m_kn(2e5),
      m_kt(2e5),
      m_gn(40),
      m_gt(20) {}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void ChRoadWheel::SetContactMaterialProperties(float young_modulus, float poisson_ratio) {
    m_young_modulus = young_modulus;
    m_poisson_ratio = poisson_ratio;
}

void ChRoadWheel::SetContactMaterialCoefficients(float kn, float gn, float kt, float gt) {
    m_kn = kn;
    m_gn = gn;
    m_kt = kt;
    m_gt = gt;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void ChRoadWheel::Initialize(std::shared_ptr<ChBodyAuxRef> chassis,
                             std::shared_ptr<ChBody> carrier,
                             const ChVector<>& location) {
    // Express the road wheel reference frame in the absolute coordinate system.
    ChFrame<> wheel_to_abs(location);
    wheel_to_abs.ConcatenatePreTransformation(chassis->GetFrame_REF_to_abs());

    // Create and initialize the wheel body.
    m_wheel = std::shared_ptr<ChBody>(chassis->GetSystem()->NewBody());
    m_wheel->SetNameString(m_name + "_wheel");
    m_wheel->SetPos(wheel_to_abs.GetPos());
    m_wheel->SetRot(wheel_to_abs.GetRot());
    m_wheel->SetMass(GetWheelMass());
    m_wheel->SetInertiaXX(GetWheelInertia());
    chassis->GetSystem()->AddBody(m_wheel);

    // Set wheel contact material properties.
    switch (m_wheel->GetContactMethod()) {
        case ChMaterialSurfaceBase::DVI:
            m_wheel->GetMaterialSurface()->SetFriction(m_friction);
            m_wheel->GetMaterialSurface()->SetRestitution(m_restitution);
            break;
        case ChMaterialSurfaceBase::DEM:
            m_wheel->GetMaterialSurfaceDEM()->SetFriction(m_friction);
            m_wheel->GetMaterialSurfaceDEM()->SetRestitution(m_restitution);
            m_wheel->GetMaterialSurfaceDEM()->SetYoungModulus(m_young_modulus);
            m_wheel->GetMaterialSurfaceDEM()->SetPoissonRatio(m_poisson_ratio);
            m_wheel->GetMaterialSurfaceDEM()->SetKn(m_kn);
            m_wheel->GetMaterialSurfaceDEM()->SetGn(m_gn);
            m_wheel->GetMaterialSurfaceDEM()->SetKt(m_kt);
            m_wheel->GetMaterialSurfaceDEM()->SetGt(m_gt);
            break;
    }

    // Create and initialize the revolute joint between wheel and carrier.
    // The axis of rotation is the y axis of the road wheel reference frame.
    m_revolute = std::make_shared<ChLinkLockRevolute>();
    m_revolute->SetNameString(m_name + "_revolute");
    m_revolute->Initialize(carrier, m_wheel,
                           ChCoordsys<>(wheel_to_abs.GetPos(), wheel_to_abs.GetRot() * Q_from_AngX(CH_C_PI_2)));
    chassis->GetSystem()->AddLink(m_revolute);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void ChRoadWheel::LogConstraintViolations() {
    ChMatrix<>* C = m_revolute->GetC();
    GetLog() << "  Road-wheel revolute\n";
    GetLog() << "  " << C->GetElement(0, 0) << "  ";
    GetLog() << "  " << C->GetElement(1, 0) << "  ";
    GetLog() << "  " << C->GetElement(2, 0) << "  ";
    GetLog() << "  " << C->GetElement(3, 0) << "  ";
    GetLog() << "  " << C->GetElement(4, 0) << "\n";
}

}  // end namespace vehicle
}  // end namespace chrono
