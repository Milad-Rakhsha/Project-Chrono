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
// Base class for a tire.
// A tire system is a force element. It is passed position and velocity
// information of the wheel body and it produces ground reaction forces and
// moments to be applied to the wheel body.
//
// =============================================================================

#ifndef CH_TIRE_H
#define CH_TIRE_H

#include "chrono/core/ChVector.h"
#include "chrono/core/ChQuaternion.h"
#include "chrono/core/ChCoordsys.h"

#include "chrono_vehicle/ChApiVehicle.h"
#include "chrono_vehicle/ChSubsysDefs.h"
#include "chrono_vehicle/ChTerrain.h"

/**
    @addtogroup vehicle_wheeled
    @{
        @defgroup vehicle_wheeled_tire Tire subsystem
    @}
*/

namespace chrono {
namespace vehicle {

/// @addtogroup vehicle_wheeled_tire
/// @{

/// Base class for a tire system.
/// A tire subsystem is a force element. It is passed position and velocity
/// information of the wheel body and it produces ground reaction forces and
/// moments to be applied to the wheel body.
class CH_VEHICLE_API ChTire {
  public:
    ChTire(const std::string& name  ///< [in] name of this tire system
           );
    virtual ~ChTire() {}

    /// Get the name of this tire.
    const std::string& GetName() const { return m_name; }

    /// Set the name for this tire.
    void SetName(const std::string& name) { m_name = name; }

    /// Initialize this tire subsystem.
    /// A derived class must call this base implementation (which simply caches the
    /// associated wheel body and vehicle side flag).
    virtual void Initialize(std::shared_ptr<ChBody> wheel,  ///< [in] associated wheel body
                            VehicleSide side                ///< [in] left/right vehicle side
                            );

    /// Update the state of this tire system at the current time.
    /// The tire system is provided the current state of its associated wheel and
    /// a handle to the terrain system.
    virtual void Synchronize(double time,                    ///< [in] current time
                             const WheelState& wheel_state,  ///< [in] current state of associated wheel body
                             const ChTerrain& terrain        ///< [in] reference to the terrain system
                             ) {
        CalculateKinematics(time, wheel_state, terrain);
    }

    /// Advance the state of this tire by the specified time step.
    virtual void Advance(double step) {}

    /// Get the tire radius.
    virtual double GetRadius() const = 0;

    /// Get the tire force and moment.
    /// This represents the output from this tire system that is passed to the
    /// vehicle system.  Typically, the vehicle subsystem will pass the tire force
    /// to the appropriate suspension subsystem which applies it as an external
    /// force one the wheel body.
    virtual TireForce GetTireForce(bool cosim = false  ///< [in] indicate if the tire is co-simulated
                                   ) const = 0;

    /// Get the tire slip angle.
    /// Return the slip angle calculated based on the current state of the associated
    /// wheel body. A derived class may override this function with a more appropriate
    /// calculation based on its specific tire model.
    virtual double GetSlipAngle() const { return m_slip_angle; }

    /// Get the tire longitudinal slip.
    /// Return the longitudinal slip calculated based on the current state of the associated
    /// wheel body. A derived class may override this function with a more appropriate
    /// calculation based on its specific tire model.
    virtual double GetLongitudinalSlip() const { return m_longitudinal_slip; }

    /// Get the tire camber angle.
    /// Return the camber angle calculated based on the current state of the associated
    /// wheel body. A derived class may override this function with a more appropriate
    /// calculation based on its specific tire model.
    virtual double GetCamberAngle() const { return m_camber_angle; }

  protected:
    /// Perform disc-terrain collision detection.
    /// This utility function checks for contact between a disc of specified
    /// radius with given position and orientation (specified as the location of
    /// its center and a unit vector normal to the disc plane) and the terrain
    /// system associated with this tire. It returns true if the disc contacts the
    /// terrain and false otherwise.  If contact occurrs, it returns a coordinate
    /// system with the Z axis along the contact normal and the X axis along the
    /// "rolling" direction, as well as a positive penetration depth (i.e. the
    /// height below the terrain of the lowest point on the disc).
    static bool disc_terrain_contact(
        const ChTerrain& terrain,       ///< [in] reference to terrain system
        const ChVector<>& disc_center,  ///< [in] global location of the disc center
        const ChVector<>& disc_normal,  ///< [in] disc normal, expressed in the global frame
        double disc_radius,             ///< [in] disc radius
        ChCoordsys<>& contact,          ///< [out] contact coordinate system (relative to the global frame)
        double& depth                   ///< [out] penetration depth (positive if contact occurred)
        );

    std::string m_name;               ///< name of this tire subsystem
    VehicleSide m_side;               ///< tire mounted on left/right side
    std::shared_ptr<ChBody> m_wheel;  ///< associated wheel body

  private:
    /// Calculate kinematics quantities based on the current state of the associated
    /// wheel body.
    void CalculateKinematics(double time,                    ///< [in] current time
                             const WheelState& wheel_state,  ///< [in] current state of associated wheel body
                             const ChTerrain& terrain        ///< [in] reference to the terrain system
                             );

    double m_slip_angle;
    double m_longitudinal_slip;
    double m_camber_angle;
};

/// @} vehicle_wheeled_tire

}  // end namespace vehicle
}  // end namespace chrono

#endif
