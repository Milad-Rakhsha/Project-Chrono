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
// Torsion-bar suspension system using linear dampers constructed with data from
// file (JSON format)
//
// =============================================================================

#include "chrono_vehicle/tracked_vehicle/suspension/LinearDamperRWAssembly.h"
#include "chrono_vehicle/tracked_vehicle/road_wheel/SingleRoadWheel.h"
#include "chrono_vehicle/tracked_vehicle/road_wheel/DoubleRoadWheel.h"

#include "chrono_vehicle/ChVehicleModelData.h"

#include "chrono_thirdparty/rapidjson/document.h"
#include "chrono_thirdparty/rapidjson/filereadstream.h"

using namespace rapidjson;

namespace chrono {
namespace vehicle {

// -----------------------------------------------------------------------------
// This utility function returns a ChVector from the specified JSON array
// -----------------------------------------------------------------------------
static ChVector<> loadVector(const Value& a) {
    assert(a.IsArray());
    assert(a.Size() == 3);

    return ChVector<>(a[0u].GetDouble(), a[1u].GetDouble(), a[2u].GetDouble());
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void LinearDamperRWAssembly::LoadRoadWheel(const std::string& filename) {
    FILE* fp = fopen(filename.c_str(), "r");

    char readBuffer[65536];
    FileReadStream is(fp, readBuffer, sizeof(readBuffer));

    fclose(fp);

    Document d;
    d.ParseStream(is);

    // Check that the given file is a road-wheel specification file.
    assert(d.HasMember("Type"));
    std::string type = d["Type"].GetString();
    assert(type.compare("RoadWheel") == 0);

    // Extract the road-wheel type
    assert(d.HasMember("Template"));
    std::string subtype = d["Template"].GetString();

    // Create the road-wheel using the appropriate template.
    if (subtype.compare("SingleRoadWheel") == 0) {
        m_road_wheel = std::make_shared<SingleRoadWheel>(d);
    } else if (subtype.compare("DoubleRoadWheel") == 0) {
        m_road_wheel = std::make_shared<DoubleRoadWheel>(d);
    }

    GetLog() << "  Loaded JSON: " << filename.c_str() << "\n";
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
LinearDamperRWAssembly::LinearDamperRWAssembly(const std::string& filename, bool has_shock)
    : ChLinearDamperRWAssembly("", has_shock) {
    FILE* fp = fopen(filename.c_str(), "r");

    char readBuffer[65536];
    FileReadStream is(fp, readBuffer, sizeof(readBuffer));

    fclose(fp);

    Document d;
    d.ParseStream(is);

    Create(d);

    GetLog() << "Loaded JSON: " << filename.c_str() << "\n";
}

LinearDamperRWAssembly::LinearDamperRWAssembly(const rapidjson::Document& d, bool has_shock)
    : ChLinearDamperRWAssembly("", has_shock) {
    Create(d);
}

void LinearDamperRWAssembly::Create(const rapidjson::Document& d) {
    // Read top-level data
    assert(d.HasMember("Type"));
    assert(d.HasMember("Template"));
    assert(d.HasMember("Name"));

    SetName(d["Name"].GetString());

    // Read suspension arm data
    assert(d.HasMember("Suspension Arm"));
    assert(d["Suspension Arm"].IsObject());

    m_arm_mass = d["Suspension Arm"]["Mass"].GetDouble();
    m_points[ARM] = loadVector(d["Suspension Arm"]["COM"]);
    m_arm_inertia = loadVector(d["Suspension Arm"]["Inertia"]);
    m_points[ARM_CHASSIS] = loadVector(d["Suspension Arm"]["Location Chassis"]);
    m_points[ARM_WHEEL] = loadVector(d["Suspension Arm"]["Location Wheel"]);
    m_arm_radius = d["Suspension Arm"]["Radius"].GetDouble();

    // Read data for torsional spring
    assert(d.HasMember("Torsional Spring"));
    assert(d["Torsional Spring"].IsObject());

    ////double torsion_a0 = d["Torsional Spring"]["Free Angle"].GetDouble();
    double torsion_k = d["Torsional Spring"]["Spring Constant"].GetDouble();
    double torsion_c = d["Torsional Spring"]["Damping Coefficient"].GetDouble();
    double torsion_t = d["Torsional Spring"]["Preload"].GetDouble();
    m_torsion_force = new ChLinkForce;
    m_torsion_force->Set_active(1);
    m_torsion_force->Set_K(torsion_k);
    m_torsion_force->Set_R(torsion_c);
    m_torsion_force->Set_iforce(torsion_t);

    // Read linear shock data
    assert(d.HasMember("Damper"));

    m_points[SHOCK_C] = loadVector(d["Damper"]["Location Chassis"]);
    m_points[SHOCK_A] = loadVector(d["Damper"]["Location Arm"]);
    double shock_c = d["Damper"]["Damping Coefficient"].GetDouble();
    m_shock_forceCB = new LinearDamperForce(shock_c);

    // Create the associated road-wheel
    assert(d.HasMember("Road Wheel Input File"));

    std::string file_name = d["Road Wheel Input File"].GetString();
    LoadRoadWheel(vehicle::GetDataFile(file_name));
}

}  // end namespace vehicle
}  // end namespace chrono
