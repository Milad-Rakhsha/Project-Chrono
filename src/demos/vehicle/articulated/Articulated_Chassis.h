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
// Chassis subsystem for the articulated vehicle.
//
// =============================================================================

#ifndef ARTICULATED_CHASSIS_H
#define ARTICULATED_CHASSIS_H

#include <string>

#include "chrono_vehicle/ChChassis.h"

class Articulated_Chassis : public chrono::vehicle::ChChassis {
  public:
    Articulated_Chassis(const std::string& name);
    ~Articulated_Chassis() {}

    /// Return the mass of the chassis body.
    virtual double GetMass() const override { return m_mass; }

    /// Return the moments of inertia of the chassis body.
    virtual const chrono::ChVector<>& GetInertia() const override { return m_inertia; }

    /// Get the location of the center of mass in the chassis frame.
    virtual const chrono::ChVector<>& GetLocalPosCOM() const override { return m_COM_loc; }

    /// Get the local driver position and orientation.
    /// This is a coordinate system relative to the chassis reference frame.
    virtual chrono::ChCoordsys<> GetLocalDriverCoordsys() const override { return m_driverCsys; }

    /// Add visualization of the road wheel.
    virtual void AddVisualizationAssets(chrono::vehicle::VisualizationType vis) override;

  protected:
    static const double m_mass;
    static const chrono::ChVector<> m_inertia;
    static const chrono::ChVector<> m_COM_loc;
    static const chrono::ChCoordsys<> m_driverCsys;
};

#endif
