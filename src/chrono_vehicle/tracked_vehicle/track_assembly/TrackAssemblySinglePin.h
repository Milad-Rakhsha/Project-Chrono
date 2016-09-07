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
// Track assembly (single-pin) model constructed from a JSON specification file
//
// =============================================================================

#ifndef TRACK_ASSEMBLY_SINGLE_PIN_H
#define TRACK_ASSEMBLY_SINGLE_PIN_H

#include <vector>

#include "chrono_vehicle/ChApiVehicle.h"
#include "chrono_vehicle/tracked_vehicle/track_assembly/ChTrackAssemblySinglePin.h"

#include "chrono_thirdparty/rapidjson/document.h"

namespace chrono {
namespace vehicle {

/// @addtogroup vehicle_tracked
/// @{

/// Wheeled vehicle model constructed from a JSON specification file
class CH_VEHICLE_API TrackAssemblySinglePin : public ChTrackAssemblySinglePin {
  public:
    TrackAssemblySinglePin(const std::string& filename);
    TrackAssemblySinglePin(const rapidjson::Document& d);
    ~TrackAssemblySinglePin() {}

    virtual const ChVector<>& GetSprocketLocation() const override { return m_sprocket_loc; }
    virtual const ChVector<>& GetIdlerLocation() const override { return m_idler_loc; }
    virtual const ChVector<>& GetRoadWhelAssemblyLocation(int which) const override { return m_susp_locs[which]; }

  private:
    void Create(const rapidjson::Document& d);

    void LoadSprocket(const std::string& filename);
    void LoadBrake(const std::string& filename);
    void LoadIdler(const std::string& filename);
    void LoadSuspension(const std::string& filename, int which, bool has_shock);
    void LoadTrackShoes(const std::string& filename, int num_shoes);

  private:
    int m_num_susp;
    int m_num_track_shoes;

    ChVector<> m_sprocket_loc;
    ChVector<> m_idler_loc;
    std::vector<ChVector<>> m_susp_locs;
};

/// @} vehicle_tracked

}  // end namespace vehicle
}  // end namespace chrono

#endif
