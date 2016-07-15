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
// Base class for a vehicle model.
//
// =============================================================================

#include <algorithm>

#include "chrono_vehicle/ChVehicle.h"

namespace chrono {
namespace vehicle {

// -----------------------------------------------------------------------------
// Constructor for a ChVehicle using a default Chrono Chsystem.
// Specify default step size and solver parameters.
// -----------------------------------------------------------------------------
ChVehicle::ChVehicle(ChMaterialSurfaceBase::ContactMethod contact_method) : m_ownsSystem(true), m_stepsize(1e-3) {
    m_system = (contact_method == ChMaterialSurfaceBase::DVI) ? new ChSystem : new ChSystemDEM;

    m_system->Set_G_acc(ChVector<>(0, 0, -9.81));

    // Integration and Solver settings
    m_system->SetSolverType(ChSystem::SOLVER_SOR);
    m_system->SetMaxItersSolverSpeed(150);
    m_system->SetMaxItersSolverStab(150);
    m_system->SetMaxPenetrationRecoverySpeed(4.0);
}

// -----------------------------------------------------------------------------
// Constructor for a ChVehicle using the specified Chrono ChSystem.
// -----------------------------------------------------------------------------
ChVehicle::ChVehicle(ChSystem* system) : m_system(system), m_ownsSystem(false), m_stepsize(1e-3) {
}

// -----------------------------------------------------------------------------
// Destructor for ChVehicle
// -----------------------------------------------------------------------------
ChVehicle::~ChVehicle() {
    if (m_ownsSystem)
        delete m_system;
}

// -----------------------------------------------------------------------------
// Advance the state of the system, taking as many steps as needed to exactly
// reach the specified value 'step'.
// ---------------------------------------------------------------------------- -
void ChVehicle::Advance(double step) {
    double t = 0;
    while (t < step) {
        double h = std::min<>(m_stepsize, step - t);
        m_system->DoStepDynamics(h);
        t += h;
    }
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
ChVector<> ChVehicle::GetVehicleAcceleration(const ChVector<>& locpos) const {
    ChVector<> acc_abs = m_chassis->GetFrame_REF_to_abs().PointAccelerationLocalToParent(locpos);
    return m_chassis->GetFrame_REF_to_abs().TransformDirectionParentToLocal(acc_abs);
}

// -----------------------------------------------------------------------------
// Return the global driver position
// -----------------------------------------------------------------------------
ChVector<> ChVehicle::GetDriverPos() const {
    return m_chassis->GetCoord().TransformPointLocalToParent(GetLocalDriverCoordsys().pos);
}

}  // end namespace vehicle
}  // end namespace chrono
