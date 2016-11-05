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
// Rigid tire constructed with data from file (JSON format).
//
// =============================================================================

#include <algorithm>

#include "chrono_vehicle/wheeled_vehicle/tire/RigidTire.h"
#include "chrono_vehicle/ChVehicleModelData.h"

#include "chrono_thirdparty/rapidjson/filereadstream.h"

using namespace rapidjson;

namespace chrono {
namespace vehicle {

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
RigidTire::RigidTire(const std::string& filename) : ChRigidTire(""), m_has_mesh(false) {
    FILE* fp = fopen(filename.c_str(), "r");

    char readBuffer[65536];
    FileReadStream is(fp, readBuffer, sizeof(readBuffer));

    fclose(fp);

    Document d;
    d.ParseStream(is);

    Create(d);

    GetLog() << "Loaded JSON: " << filename.c_str() << "\n";
}

RigidTire::RigidTire(const rapidjson::Document& d) : ChRigidTire(""), m_has_mesh(false) {
    Create(d);
}

void RigidTire::Create(const rapidjson::Document& d) {
    // Read top-level data
    assert(d.HasMember("Type"));
    assert(d.HasMember("Template"));
    assert(d.HasMember("Name"));

    SetName(d["Name"].GetString());

    m_radius = d["Radius"].GetDouble();
    m_width = d["Width"].GetDouble();

    // Read contact material data
    assert(d.HasMember("Contact Material"));

    float mu = d["Contact Material"]["Coefficient of Friction"].GetDouble();
    float cr = d["Contact Material"]["Coefficient of Restitution"].GetDouble();

    SetContactFrictionCoefficient(mu);
    SetContactRestitutionCoefficient(cr);

    if (d["Contact Material"].HasMember("Properties")) {
        float ym = d["Contact Material"]["Properties"]["Young Modulus"].GetDouble();
        float pr = d["Contact Material"]["Properties"]["Poisson Ratio"].GetDouble();
        SetContactMaterialProperties(ym, pr);
    }
    if (d["Contact Material"].HasMember("Coefficients")) {
        float kn = d["Contact Material"]["Coefficients"]["Normal Stiffness"].GetDouble();
        float gn = d["Contact Material"]["Coefficients"]["Normal Damping"].GetDouble();
        float kt = d["Contact Material"]["Coefficients"]["Tangential Stiffness"].GetDouble();
        float gt = d["Contact Material"]["Coefficients"]["Tangential Damping"].GetDouble();
        SetContactMaterialCoefficients(kn, gn, kt, gt);
    }

    // Check if using contact mesh.
    if (d.HasMember("Contact Mesh")) {
        std::string mesh_file = d["Contact Mesh"]["Mesh Filename"].GetString();
        double sweep_radius = d["Contact Mesh"]["Sweep Sphere Radius"].GetDouble();
        SetMeshFilename(vehicle::GetDataFile(mesh_file), sweep_radius);
    }

    // Check how to visualize this tire.
    if (d.HasMember("Visualization")) {
        if (d["Visualization"].HasMember("Mesh Filename")) {
            m_meshFile = d["Visualization"]["Mesh Filename"].GetString();
            m_meshName = d["Visualization"]["Mesh Name"].GetString();
            m_has_mesh = true;
        }
    }
}

// -----------------------------------------------------------------------------
void RigidTire::AddVisualizationAssets(VisualizationType vis) {
    if (vis == VisualizationType::MESH && m_has_mesh) {
        geometry::ChTriangleMeshConnected trimesh;
        trimesh.LoadWavefrontMesh(vehicle::GetDataFile(m_meshFile), false, false);
        m_trimesh_shape = std::make_shared<ChTriangleMeshShape>();
        m_trimesh_shape->SetMesh(trimesh);
        m_trimesh_shape->SetName(m_meshName);
        m_wheel->AddAsset(m_trimesh_shape);
    } else {
        ChRigidTire::AddVisualizationAssets(vis);
    }
}

void RigidTire::RemoveVisualizationAssets() {
    ChRigidTire::RemoveVisualizationAssets();

    // Make sure we only remove the assets added by RigidTire::AddVisualizationAssets.
    // This is important for the ChTire object because a wheel may add its own assets
    // to the same body (the spindle/wheel).
    auto it = std::find(m_wheel->GetAssets().begin(), m_wheel->GetAssets().end(), m_trimesh_shape);
    if (it != m_wheel->GetAssets().end())
        m_wheel->GetAssets().erase(it);
}

}  // end namespace vehicle
}  // end namespace chrono
