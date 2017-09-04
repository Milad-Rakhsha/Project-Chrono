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
// Authors: Milad Rakhsha, Antonio Recuero
// =============================================================================
//
// This is a more involved model of Interaction of tiba-femoral contact.
// This model the gait cycle and evaluate tibia-femoral contact.
// =============================================================================

#include "chrono/lcp/ChLcpIterativeMINRES.h"
#include "chrono_mkl/ChLcpMklSolver.h"
#include "chrono/physics/ChBodyEasy.h"
#include "chrono/physics/ChSystem.h"
#include "chrono/core/ChFileutils.h"
#include "chrono/ChConfig.h"

#include "thirdparty/rapidjson/document.h"
#include "thirdparty/rapidjson/filereadstream.h"

#include "chrono/utils/ChUtilsInputOutput.h"
#include "chrono_fea/ChElementShellANCF.h"
#include "chrono_fea/ChLinkDirFrame.h"
#include "chrono_fea/ChLinkPointFrame.h"
#include "chrono_fea/ChVisualizationFEAmesh.h"
#include "chrono_fea/ChMesh.h"
#include "chrono_fea/ChMeshFileLoader.h"
#include "chrono_fea/ChLinkPointFrame.h"
#include "chrono/physics/ChLoaderUV.h"

#include <iostream>
#include <math.h>
#include <sstream>
#include <fstream>
#include <string>
#include <algorithm>
#include <functional>
#include <stdio.h>

using namespace chrono;
using namespace chrono::fea;
using namespace std;
using namespace rapidjson;

bool outputData = true;
bool addGravity = false;
const int scaleFactor = 1;

int num_threads;
double time_step;
double dz;
double rho, E, nu;
double Input1, Input2;

void SetParamFromJSON(const std::string& filename, double& Input1, double& Input2);

int main(int argc, char* argv[]) {
    ChSystem my_system(false, true);

    SetParamFromJSON(GetChronoDataFile("fea/HeedsInput.json").c_str(), Input1, Input2);
    std::ofstream output_femur;
    output_femur.open("output.txt");
    output_femur << sin(Input1) + sin(Input2) << "\n";
    output_femur.close();
    cout << "The Output is: " << sin(Input1) + sin(Input2) << "\n\n\n";
// --------------------------
// Set number of threads
// --------------------------
#ifdef CHRONO_OPENMP_ENABLED
    int max_threads = CHOMPfunctions::GetNumProcs();

    if (num_threads > max_threads)
        num_threads = max_threads;

    CHOMPfunctions::SetNumThreads(num_threads);
#else
    GetLog() << "No OpenMP\n";
#endif

    std::vector<std::vector<double>> motionInfo;
    double D2R = 3.1415 / 180;
    ChVector<> RotAng(0, 0, 0);
    ChVector<> Origin(0, 0, 0);

    ChMatrix33<> MeshRotate;

    std::vector<double> NODE_AVE_AREA_f;
    std::vector<int> BC_NODES;

    auto material = std::make_shared<ChMaterialShellANCF>(rho, E, nu);
    auto my_mesh = std::make_shared<ChMesh>();
    ChMatrix33<> rot_transform(1);

    try {
        ChMeshFileLoader::ANCFShellFromGMFFile(my_mesh, GetChronoDataFile("fea/FemurFine.mesh").c_str(), material,
                                               NODE_AVE_AREA_f, BC_NODES, Origin, rot_transform, 1, false, false);
    } catch (ChException myerr) {
        GetLog() << myerr.what();
        return 0;
    }
    for (int node = 0; node < BC_NODES.size(); node++) {
        auto FixedNode = std::make_shared<ChNodeFEAxyzD>();
        FixedNode = std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(BC_NODES[node]));
        FixedNode->SetFixed(true);
    }

    double TotalNumNodes_Femur = my_mesh->GetNnodes();
    double TotalNumElements_Femur = my_mesh->GetNelements();
    for (int ele = 0; ele < TotalNumElements_Femur; ele++) {
        auto element = std::make_shared<ChElementShellANCF>();
        element = std::dynamic_pointer_cast<ChElementShellANCF>(my_mesh->GetElement(ele));
        // Add a single layers with a fiber angle of 0 degrees.
        element->AddLayer(dz, 0 * CH_C_DEG_TO_RAD, material);
        // Set other element properties
        element->SetAlphaDamp(0.01);   // Structural damping for this element
        element->SetGravityOn(false);  // gravitational forces
    }

    // Switch off mesh class gravity
    my_mesh->SetAutomaticGravity(addGravity);
    my_mesh->SetAutomaticGravity(addGravity);
    if (addGravity) {
        my_system.Set_G_acc(ChVector<>(0, -9.8, 0));
    } else {
        my_system.Set_G_acc(ChVector<>(0, 0, 0));
    }

    my_system.Add(my_mesh);
    my_system.SetupInitial();

    // ---------------
    // Simulation loop
    // ---------------
    ChLcpMklSolver* mkl_solver_stab = new ChLcpMklSolver;
    ChLcpMklSolver* mkl_solver_speed = new ChLcpMklSolver;
    my_system.ChangeLcpSolverStab(mkl_solver_stab);
    my_system.ChangeLcpSolverSpeed(mkl_solver_speed);
    mkl_solver_stab->SetSparsityPatternLock(true);
    mkl_solver_speed->SetSparsityPatternLock(true);

    return 0;
}
//======================================================================================================================================
//======================================================================================================================================
//======================================================================================================================================
//======================================================================================================================================
//======================================================================================================================================

void SetParamFromJSON(const std::string& filename, double& Input1, double& Input2) {
    // -------------------------------------------
    // Open and parse the input file
    // -------------------------------------------
    FILE* fp = fopen(filename.c_str(), "r");

    char readBuffer[65536];
    FileReadStream is(fp, readBuffer, sizeof(readBuffer));

    fclose(fp);

    Document d;
    d.ParseStream(is);

    // Read top-level data
    assert(d.HasMember("Chrono Inputs"));
    Input1 = d["Chrono Inputs"]["x"].GetDouble();
    Input2 = d["Chrono Inputs"]["y"].GetDouble();
    GetLog() << "Loaded JSON: " << filename.c_str() << "\n";
}
