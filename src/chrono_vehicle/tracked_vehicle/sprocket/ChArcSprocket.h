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
// Base class for a sprocket template with gear profile composed of circular
// arcs, suitable for interaction with single-pin track shoes.
//
// =============================================================================

#ifndef CH_ARC_SPROCKET_H
#define CH_ARC_SPROCKET_H

#include "chrono_vehicle/ChApiVehicle.h"

#include "chrono_vehicle/tracked_vehicle/ChSprocket.h"

namespace chrono {
namespace vehicle {

/// @addtogroup vehicle_tracked_sprocket
/// @{

/// Base class for a sprocket template with gear profile composed of circular arcs.
/// This sprocket type is suitable for interaction with single-pin track shoes.
class CH_VEHICLE_API ChArcSprocket : public ChSprocket {
  public:
    ChArcSprocket(const std::string& name  ///< [in] name of the subsystem
                  );

    virtual ~ChArcSprocket() {}

    /// Return the 2D gear profile.
    /// The gear profile, a ChLinePath geometric object, is made up of an arbitrary number
    /// of sub-paths of type ChLineArc or ChLineSegment sub-lines. These must be added in
    /// clockwise order, and the end of sub-path i must be coincident with beginning of
    /// sub-path i+1.
    virtual std::shared_ptr<geometry::ChLinePath> GetProfile() override;

    /// Return the custom collision callback object.
    virtual ChSystem::ChCustomComputeCollisionCallback* GetCollisionCallback(
        ChTrackAssembly* track  ///< [in] pointer to containing track assembly
        ) override;

  protected:
    /// Return the radius of the addendum circle.
    virtual double GetOuterRadius() const = 0;

    /// Return the radius of the (concave) tooth circular arc.
    virtual double GetArcRadius() const = 0;

    /// Return the radius of the tooth arc centers.
    virtual double GetArcCentersRadius() const = 0;
};

/// @} vehicle_tracked_sprocket

}  // end namespace vehicle
}  // end namespace chrono

#endif
