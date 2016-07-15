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
// M113 sprocket subsystem (double pin).
//
// =============================================================================

#include "chrono/assets/ChColor.h"
#include "chrono/assets/ChTriangleMeshShape.h"
#include "chrono/utils/ChUtilsInputOutput.h"

#include "chrono_vehicle/ChVehicleModelData.h"

#include "models/vehicle/m113/M113_SprocketDoublePin.h"

namespace chrono {
namespace vehicle {
namespace m113 {

// -----------------------------------------------------------------------------
// Static variables
// -----------------------------------------------------------------------------
const int M113_SprocketDoublePin::m_num_teeth = 10;

const double M113_SprocketDoublePin::m_gear_mass = 436.7;
const ChVector<> M113_SprocketDoublePin::m_gear_inertia(12.22, 13.87, 12.22);
const double M113_SprocketDoublePin::m_axle_inertia = 1;
const double M113_SprocketDoublePin::m_separation = 0.278;

const double M113_SprocketDoublePin::m_gear_RT = 0.2715;  // 10.69''
const double M113_SprocketDoublePin::m_gear_R = 0.0223;   // 0.88''
const double M113_SprocketDoublePin::m_gear_RA = 0.242;   // 9.53''

const double M113_SprocketDoublePin::m_gear_C = 0.2371;  // 9.334''
const double M113_SprocketDoublePin::m_gear_W = 0.0464;  // 1.825''

const std::string M113_SprocketDoublePinLeft::m_meshName = "Sprocket2_L_POV_geom";
const std::string M113_SprocketDoublePinLeft::m_meshFile = "M113/Sprocket2_L.obj";

const std::string M113_SprocketDoublePinRight::m_meshName = "Sprocket2_R_POV_geom";
const std::string M113_SprocketDoublePinRight::m_meshFile = "M113/Sprocket2_R.obj";

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
M113_SprocketDoublePin::M113_SprocketDoublePin(const std::string& name) : ChSprocketDoublePin(name), m_vis_type(PRIMITIVES) {
    SetContactMaterial(0.4f, 0.1f, 1e7f, 0.3f);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void M113_SprocketDoublePin::AddGearVisualization() {
    switch (m_vis_type) {
        case PRIMITIVES:
            ChSprocket::AddGearVisualization();
            break;
        case MESH: {
            ////geometry::ChTriangleMeshConnected trimesh;
            ////trimesh.LoadWavefrontMesh(GetMeshFile(), false, false);
            ////auto trimesh_shape = std::make_shared<ChTriangleMeshShape>();
            ////trimesh_shape->SetMesh(trimesh);
            ////trimesh_shape->SetName(GetMeshName());
            ////m_gear->AddAsset(trimesh_shape);
            break;
        }
    }
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void M113_SprocketDoublePin::ExportMeshPovray(const std::string& out_dir) {
    ////utils::WriteMeshPovray(GetMeshFile(), GetMeshName(), out_dir, ChColor(0.15f, 0.15f, 0.15f));
}

}  // end namespace m113
}  // end namespace vehicle
}  // end namespace chrono
