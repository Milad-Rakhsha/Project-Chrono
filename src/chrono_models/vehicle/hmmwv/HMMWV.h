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
// Wrapper classes for modeling an entire HMMWV vehicle assembly
// (including the vehicle itself, the powertrain, and the tires).
//
// =============================================================================

#ifndef HMMWV_H
#define HMMWV_H

#include <array>
#include <string>

#include "chrono_vehicle/wheeled_vehicle/tire/ChPacejkaTire.h"

#include "chrono_models/ChApiModels.h"
#include "chrono_models/vehicle/hmmwv/HMMWV_VehicleFull.h"
#include "chrono_models/vehicle/hmmwv/HMMWV_VehicleReduced.h"
#include "chrono_models/vehicle/hmmwv/HMMWV_Powertrain.h"
#include "chrono_models/vehicle/hmmwv/HMMWV_SimplePowertrain.h"
#include "chrono_models/vehicle/hmmwv/HMMWV_FialaTire.h"
#include "chrono_models/vehicle/hmmwv/HMMWV_LugreTire.h"
#include "chrono_models/vehicle/hmmwv/HMMWV_RigidTire.h"

#ifdef CHRONO_FEA
#include "chrono_models/vehicle/hmmwv/HMMWV_ANCFTire.h"
#include "chrono_models/vehicle/hmmwv/HMMWV_ReissnerTire.h"
#endif

namespace chrono {
namespace vehicle {
namespace hmmwv {

class CH_MODELS_API HMMWV {
  public:
    virtual ~HMMWV();

    void SetContactMethod(ChMaterialSurfaceBase::ContactMethod val) { m_contactMethod = val; }

    void SetChassisFixed(bool val) { m_fixed = val; }

    void SetDriveType(DrivelineType val) { m_driveType = val; }
    void SetPowertrainType(PowertrainModelType val) { m_powertrainType = val; }
    void SetTireType(TireModelType val) { m_tireType = val; }

    void SetInitPosition(const ChCoordsys<>& pos) { m_initPos = pos; }
    void SetInitFwdVel(double fwdVel) { m_initFwdVel = fwdVel; }
    void SetInitWheelAngVel(const std::vector<double>& omega) { m_initOmega = omega; }

    void SetTireStepSize(double step_size) { m_tire_step_size = step_size; }
    void SetPacejkaParamfile(const std::string& filename) { m_pacejkaParamFile = filename; }

    ChSystem* GetSystem() const { return m_vehicle->GetSystem(); }
    ChWheeledVehicle& GetVehicle() const { return *m_vehicle; }
    std::shared_ptr<ChChassis> GetChassis() const { return m_vehicle->GetChassis(); }
    std::shared_ptr<ChBodyAuxRef> GetChassisBody() const { return m_vehicle->GetChassisBody(); }
    ChPowertrain& GetPowertrain() const { return *m_powertrain; }
    ChTire* GetTire(WheelID which) const { return m_tires[which.id()]; }

    void Initialize();

    void SetChassisVisualizationType(VisualizationType vis) { m_vehicle->SetChassisVisualizationType(vis); }
    void SetSuspensionVisualizationType(VisualizationType vis) { m_vehicle->SetSuspensionVisualizationType(vis); }
    void SetSteeringVisualizationType(VisualizationType vis) { m_vehicle->SetSteeringVisualizationType(vis); }
    void SetWheelVisualizationType(VisualizationType vis) { m_vehicle->SetWheelVisualizationType(vis); }
    void SetTireVisualizationType(VisualizationType vis);

    void Synchronize(double time,
                     double steering_input,
                     double braking_input,
                     double throttle_input,
                     const ChTerrain& terrain);

    void Advance(double step);

  protected:
    // Protected constructors -- this class cannot be instantiated by itself.
    HMMWV();
    HMMWV(ChSystem* system);

    virtual HMMWV_Vehicle* CreateVehicle() = 0;

    ChMaterialSurfaceBase::ContactMethod m_contactMethod;
    bool m_fixed;

    DrivelineType m_driveType;
    PowertrainModelType m_powertrainType;
    TireModelType m_tireType;

    double m_tire_step_size;
    std::string m_pacejkaParamFile;

    ChCoordsys<> m_initPos;
    double m_initFwdVel;
    std::vector<double> m_initOmega;

    ChSystem* m_system;
    HMMWV_Vehicle* m_vehicle;
    ChPowertrain* m_powertrain;
    std::array<ChTire*, 4> m_tires;
};

class CH_MODELS_API HMMWV_Full : public HMMWV {
  public:
    HMMWV_Full() {}
    HMMWV_Full(ChSystem* system) : HMMWV(system) {}

    void LogHardpointLocations() { ((HMMWV_VehicleFull*)m_vehicle)->LogHardpointLocations(); }
    void DebugLog(int what) { ((HMMWV_VehicleFull*)m_vehicle)->DebugLog(what); }

  private:
    virtual HMMWV_Vehicle* CreateVehicle() override {
        return m_system ? new HMMWV_VehicleFull(m_system, m_fixed, m_driveType)
                        : new HMMWV_VehicleFull(m_fixed, m_driveType, m_contactMethod);
    }
};

class CH_MODELS_API HMMWV_Reduced : public HMMWV {
  public:
    HMMWV_Reduced() {}
    HMMWV_Reduced(ChSystem* system) : HMMWV(system) {}

  private:
    virtual HMMWV_Vehicle* CreateVehicle() override {
        return m_system ? new HMMWV_VehicleReduced(m_system, m_fixed, m_driveType)
                        : new HMMWV_VehicleReduced(m_fixed, m_driveType, m_contactMethod);
    }
};

}  // end namespace hmmwv
}  // end namespace vehicle
}  // end namespace chrono

#endif
