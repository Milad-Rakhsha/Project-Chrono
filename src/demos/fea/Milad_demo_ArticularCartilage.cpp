//
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2013 Project Chrono
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file at the top level of the distribution
// and at http://projectchrono.org/license-chrono.txt.
//

#include "chrono/lcp/ChLcpIterativeMINRES.h"
#include "chrono_mkl/ChLcpMklSolver.h"
#include "chrono/physics/ChBodyEasy.h"
#include "chrono/physics/ChSystem.h"
#include "chrono/physics/ChSystemDEM.h"
#include "chrono/physics/ChLoadBodyMesh.h"

#include "chrono/utils/ChUtilsInputOutput.h"
#include "chrono_fea/ChElementShellANCF.h"
#include "chrono_fea/ChLinkDirFrame.h"
#include "chrono_fea/ChLinkPointFrame.h"
#include "chrono_fea/ChVisualizationFEAmesh.h"
#include "chrono_fea/ChMesh.h"
#include "chrono_fea/ChMeshFileLoader.h"
#include "chrono_fea/ChLinkPointFrame.h"
#include "chrono_irrlicht/ChIrrApp.h"
#include "chrono/physics/ChLoaderUV.h"
#include "chrono/physics/ChLoadContainer.h"
#include "chrono_fea/ChVisualizationFEAmesh.h"
#include "chrono/geometry/ChCTriangleMeshConnected.h"
#include "chrono_fea/ChContactSurfaceMesh.h"
#include "chrono_fea/ChLoadContactSurfaceMesh.h"
#include "chrono_fea/ChContactSurfaceNodeCloud.h"

#include "chrono_irrlicht/ChBodySceneNode.h"
#include "chrono_irrlicht/ChBodySceneNodeTools.h"
#include "chrono_irrlicht/ChIrrAppInterface.h"
#include "chrono_irrlicht/ChIrrTools.h"
#include "chrono_irrlicht/ChIrrWizard.h"

#include <iostream>
#include <math.h>
#include <sstream>
#include <fstream>
#include <string>
#include <algorithm>
#include <functional>

using namespace chrono;
using namespace chrono::geometry;
using namespace chrono::fea;
using namespace chrono::irrlicht;
using namespace irr;
using namespace std;

// bool addConstrain = true;
// bool addForce = true;
bool addGravity = true;
bool addPressure = false;
bool showTibia = false;
bool showFemur = true;
// bool addFixed = false;
double time_step = 0.00004;
int scaleFactor = 1;
double dz = 0.0005;
double MeterToInch = 0.02539998628;

int main(int argc, char* argv[]) {
    ChSystemDEM my_system;

    // Create the Irrlicht visualization (open the Irrlicht device,
    // bind a simple user interface, etc. etc.)
    ChIrrApp application(&my_system, L"ANCF Shells", core::dimension2d<u32>(800, 600), false, true);

    // Easy shortcuts to add camera, lights, logo and sky in Irrlicht scene:
    application.AddTypicalLogo();
    application.AddTypicalSky();
    application.AddTypicalLights();
    application.AddTypicalCamera(core::vector3df(-0.08f, -0.01f, 0.005f),  // camera location
                                 core::vector3df(0.0f, 0.f, 0.f));         // "look at" location
    application.SetContactsDrawMode(ChIrrTools::CONTACT_DISTANCES);

    // collision::ChCollisionModel::SetDefaultSuggestedEnvelope(0.0); // not needed, already 0 when using
    collision::ChCollisionModel::SetDefaultSuggestedMargin(1.5);  // max inside penetration - if not enough stiffness in
                                                                 // material: troubles
    // Use this value for an outward additional layer around meshes, that can improve
    // robustness of mesh-mesh collision detection (at the cost of having unnatural inflate effect)
    double sphere_swept_thickness = dz*0.02;

    double rho = 1000;  ///< material density
    double E = 1e5;     ///< Young's modulus
    double nu = 0.3;    ///< Poisson ratio
    // Create the surface material, containing information
    // about friction etc.
    // It is a DEM-p (penalty) material that we will assign to
    // all surfaces that might generate contacts.

    auto mysurfmaterial = std::make_shared<ChMaterialSurfaceDEM>();
    mysurfmaterial->SetYoungModulus(1e2);
    mysurfmaterial->SetFriction(0.3f);
    mysurfmaterial->SetRestitution(0.5f);
    mysurfmaterial->SetAdhesion(0);

    GetLog() << "-----------------------------------------------------------\n";
    GetLog() << "-----------------------------------------------------------\n";
    GetLog() << "   		Articular Cartilage Modeling   \n";
    GetLog() << "-----------------------------------------------------------\n\n";
    // Creating Rigid body
    GetLog() << "	Adding the Tibia as a Rigid Body ...\n";
    auto Tibia = std::make_shared<ChBody>();
    Tibia->SetPos(ChVector<>(0, 0, 0));
    Tibia->SetBodyFixed(true);
    Tibia->SetMaterialSurface(mysurfmaterial);
    //    my_system.Add(Tibia);
    Tibia->SetMass(0.1);
    Tibia->SetInertiaXX(ChVector<>(0.004, 0.8e-4, 0.004));
    auto mobjmesh1 = std::make_shared<ChObjShapeFile>();
    mobjmesh1->SetFilename(GetChronoDataFile("fea/tibia.obj"));
    if (showTibia) {
        Tibia->AddAsset(mobjmesh1);
    }

    GetLog() << "	Adding the Femur as a Rigid Body ...\n";
    ChVector<> Center_Femur(0, 0.002, 0);
    auto Femur = std::make_shared<ChBody>();
    Femur->SetPos(Center_Femur);
    Femur->SetBodyFixed(true);
    Femur->SetMaterialSurface(mysurfmaterial);
    // my_system.Add(Femur);
    Femur->SetMass(0.2);
    Femur->SetInertiaXX(ChVector<>(0.004, 0.8e-4, 0.004));
    auto mobjmesh2 = std::make_shared<ChObjShapeFile>();
    mobjmesh2->SetFilename(GetChronoDataFile("fea/femur.obj"));
    if (showFemur) {
        Femur->AddAsset(mobjmesh2);
    }
    // Constraining the motion of the Fumer to y direction for now
    //    if (true) {
    //        // Prismatic joint between hub and suspended mass
    //        auto primsJoint = std::make_shared<ChLinkLockPrismatic>();
    //        my_system.AddLink(primsJoint);
    //        primsJoint->Initialize(Femur, Tibia, ChCoordsys<>(ChVector<>(0, 0, 0), Q_from_AngX(-CH_C_PI_2)));
    //    }
    GetLog() << "-----------------------------------------------------------\n\n";

    int TotalNumNodes, TotalNumElements, TottalNumBEdges;
    std::vector<int> BC_NODES;
    std::vector<int> BC_NODES1;
    std::vector<int> BC_NODES2;

    GetLog() << "	Adding the Membrane Using ANCF Shell Elements...  \n\n";
    // Creating the membrane shell
    auto material = std::make_shared<ChMaterialShellANCF>(rho, E, nu);
    auto my_mesh = std::make_shared<ChMesh>();
    //    ChMatrix33<> rot_transform(0);
    //    rot_transform.SetElement(0, 0, 1);
    //    rot_transform.SetElement(1, 1, -1);
    //    rot_transform.SetElement(2, 2, 1);

    ChMatrix33<> rot_transform(1);
    // Import the Torus
    try {
        ChMeshFileLoader::ANCFShellFromGMFFile(my_mesh, GetChronoDataFile("fea/FemurFine.mesh").c_str(), material,
                                               BC_NODES, Center_Femur, rot_transform, MeterToInch, false, false);
    } catch (ChException myerr) {
        GetLog() << myerr.what();
        return 0;
    }
    //
    if (false) {
        for (int node = 0; node < BC_NODES.size(); node++) {
            auto NodePosBone = std::make_shared<ChLinkPointFrame>();
            auto NodeDirBone = std::make_shared<ChLinkDirFrame>();
            auto ConstrainedNode = std::make_shared<ChNodeFEAxyzD>();

            ConstrainedNode = std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(BC_NODES[node]));
            NodePosBone->Initialize(ConstrainedNode, Femur);
            my_system.Add(NodePosBone);

            NodeDirBone->Initialize(ConstrainedNode, Femur);
            NodeDirBone->SetDirectionInAbsoluteCoords(ConstrainedNode->D);
            my_system.Add(NodeDirBone);
        }
    }

    //    ChMatrix33<> rot_transform_2(1);
    //    rot_transform_2.SetElement(0, 0, 1);
    //    rot_transform_2.SetElement(1, 1, -1);
    //    rot_transform_2.SetElement(2, 2, -1);
    //    Center = ChVector<>(0, -0.2, 0.4);
    ChVector<> Center(0, 0.0, 0);
    // Import the mesh
    try {
        ChMeshFileLoader::ANCFShellFromGMFFile(my_mesh, GetChronoDataFile("fea/Tibia-1Low.mesh").c_str(), material,
                                               BC_NODES1, Center, rot_transform, MeterToInch, false, false);
    } catch (ChException myerr) {
        GetLog() << myerr.what();
        return 0;
    }
    for (int node = 0; node < BC_NODES1.size(); node++) {
        auto FixedNode = std::make_shared<ChNodeFEAxyzD>();
        FixedNode = std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(BC_NODES1[node]));
        FixedNode->SetFixed(true);
    }

    // Import the mesh
    try {
        ChMeshFileLoader::ANCFShellFromGMFFile(my_mesh, GetChronoDataFile("fea/Tibia-2Low.mesh").c_str(), material,
                                               BC_NODES2, Center, rot_transform, MeterToInch, false, false);
    } catch (ChException myerr) {
        GetLog() << myerr.what();
        return 0;
    }
    for (int node = 0; node < BC_NODES2.size(); node++) {
        auto FixedNode = std::make_shared<ChNodeFEAxyzD>();
        FixedNode = std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(BC_NODES2[node]));
        FixedNode->SetFixed(true);
    }

    // Create the contact surface(s).
    // In this case it is a ChContactSurfaceMesh, that allows mesh-mesh collsions.
    auto mcontactsurf = std::make_shared<ChContactSurfaceMesh>();
    my_mesh->AddContactSurface(mcontactsurf);
    mcontactsurf->AddFacesFromBoundary(sphere_swept_thickness);  // do this after my_mesh->AddContactSurface
    mcontactsurf->SetMaterialSurface(mysurfmaterial);            // use the DEM penalty contacts

    TotalNumNodes = my_mesh->GetNnodes();
    TotalNumElements = my_mesh->GetNelements();

    for (int ele = 0; ele < TotalNumElements; ele++) {
        auto element = std::make_shared<ChElementShellANCF>();
        element = std::dynamic_pointer_cast<ChElementShellANCF>(my_mesh->GetElement(ele));
        // Add a single layers with a fiber angle of 0 degrees.
        element->AddLayer(dz, 0 * CH_C_DEG_TO_RAD, material);
        // Set other element properties
        element->SetAlphaDamp(0.04);   // Structural damping for this element
        element->SetGravityOn(false);  // gravitational forces
    }

    //    if (addFixed) {
    //        for (int node = 0; node < BC_NODES.size(); node++) {
    //            ChSharedPtr<ChNodeFEAxyzD>
    //            FixedNode(my_mesh->GetNode(BC_NODES[node]).DynamicCastTo<ChNodeFEAxyzD>());
    //            FixedNode->SetFixed(true);
    //
    //            //            if (FixedNode->GetPos().x < -0.65)
    //            //                FixedNode->SetFixed(true);
    //            //            if (FixedNode->GetPos().x > 0.68)
    //            //                FixedNode->SetForce(ChVector<>(10, 0, 0));
    //        }
    //    }

    if (addPressure) {
        // First: loads must be added to "load containers",
        // and load containers must be added to your ChSystem
        auto Mloadcontainer = std::make_shared<ChLoadContainer>();
        // Add constant pressure using ChLoaderPressure (preferred for simple, constant pressure)
        for (int NoElmPre = 0; NoElmPre < TotalNumElements; NoElmPre++) {
            auto faceload = std::make_shared<ChLoad<ChLoaderPressure>>(
                std::static_pointer_cast<ChElementShellANCF>(my_mesh->GetElement(NoElmPre)));
            faceload->loader.SetPressure(-10);
            faceload->loader.SetStiff(false);
            faceload->loader.SetIntegrationPoints(2);
            Mloadcontainer->Add(faceload);
        }
        my_system.Add(Mloadcontainer);
    }

    // Switch off mesh class gravity
    my_mesh->SetAutomaticGravity(addGravity);
    if (addGravity) {
        my_system.Set_G_acc(ChVector<>(0, -9.8, 0));
    } else {
        my_system.Set_G_acc(ChVector<>(0, 0, 0));
    }
    // Add the mesh to the system
    my_system.Add(my_mesh);

    // Mark completion of system construction

    //    ////////////////////////////////////////
    //    // Options for visualization in irrlicht
    //    ////////////////////////////////////////
    auto mvisualizemesh = std::make_shared<ChVisualizationFEAmesh>(*(my_mesh.get()));
    mvisualizemesh->SetFEMdataType(ChVisualizationFEAmesh::E_PLOT_NODE_SPEED_NORM);
    mvisualizemesh->SetColorscaleMinMax(0.0, 5.50);
    mvisualizemesh->SetSmoothFaces(true);
    my_mesh->AddAsset(mvisualizemesh);

    auto mvisualizemeshcoll = std::make_shared<ChVisualizationFEAmesh>(*(my_mesh.get()));
    mvisualizemeshcoll->SetFEMdataType(ChVisualizationFEAmesh::E_PLOT_CONTACTSURFACES);
    mvisualizemeshcoll->SetWireframe(true);
    mvisualizemeshcoll->SetDefaultMeshColor(ChColor(1, 0.5, 0));
    my_mesh->AddAsset(mvisualizemeshcoll);

    auto mvisualizemeshbeam = std::make_shared<ChVisualizationFEAmesh>(*(my_mesh.get()));
    mvisualizemeshbeam->SetFEMdataType(ChVisualizationFEAmesh::E_PLOT_NODE_SPEED_NORM);
    mvisualizemeshbeam->SetColorscaleMinMax(0.0, 5.50);
    mvisualizemeshbeam->SetSmoothFaces(true);
    my_mesh->AddAsset(mvisualizemeshbeam);

    auto mvisualizemeshbeamnodes = std::make_shared<ChVisualizationFEAmesh>(*(my_mesh.get()));
    mvisualizemeshbeamnodes->SetFEMglyphType(ChVisualizationFEAmesh::E_GLYPH_NODE_DOT_POS);
    mvisualizemeshbeamnodes->SetFEMdataType(ChVisualizationFEAmesh::E_PLOT_NONE);
    mvisualizemeshbeamnodes->SetSymbolsThickness(0.0002);
    my_mesh->AddAsset(mvisualizemeshbeamnodes);

    application.AssetBindAll();
    application.AssetUpdateAll();
    application.AddShadowAll();
    my_system.SetupInitial();

    //    GetLog() << "\n\nREADME\n\n"
    //             << " - Press SPACE to start dynamic simulation \n - Press F10 for nonlinear statics - Press F11
    //             for "
    //                "linear statics. \n";

    // at beginning, no analysis is running..
    //    application.SetPaused(true);
    //    int AccuNoIterations = 0;
    //    application.SetStepManage(true);
    //    application.SetTimestep(time_step);
    //    application.SetTryRealtime(true);

    // ---------------
    // Simulation loop
    // ---------------
    ChLcpMklSolver* mkl_solver_stab = new ChLcpMklSolver;
    ChLcpMklSolver* mkl_solver_speed = new ChLcpMklSolver;
    my_system.ChangeLcpSolverStab(mkl_solver_stab);
    my_system.ChangeLcpSolverSpeed(mkl_solver_speed);
    mkl_solver_stab->SetSparsityPatternLock(true);
    mkl_solver_speed->SetSparsityPatternLock(true);
    application.GetSystem()->Update();

    // Setup solver
    //    my_system.SetLcpSolverType(ChSystem::LCP_ITERATIVE_MINRES);
    //    ChLcpIterativeMINRES* msolver = (ChLcpIterativeMINRES*)my_system.GetLcpSolverSpeed();
    //    msolver->SetDiagonalPreconditioning(true);
    //    my_system.SetIterLCPwarmStarting(true);  // this helps a lot to speedup convergence in this class of
    //    my_system.SetIterLCPmaxItersSpeed(40);
    //    my_system.SetTolForce(1e-6);
    //    msolver->SetVerbose(false);
    //
    // INT_HHT or INT_EULER_IMPLICIT
    //my_system.SetIntegrationType(ChSystem::INT_EULER_IMPLICIT_LINEARIZED);  // fast, less precise

    /*my_system.SetIntegrationType(ChSystem::INT_HHT);
    auto mystepper = std::dynamic_pointer_cast<ChTimestepperHHT>(my_system.GetTimestepper());
    mystepper->SetAlpha(-0.0);
    mystepper->SetMaxiters(200);
    mystepper->SetAbsTolerances(1e-06, 1e-03);
    mystepper->SetMode(ChTimestepperHHT::POSITION);
    mystepper->SetScaling(true);
    mystepper->SetVerbose(true);
    mystepper->SetMaxiters(20);*/

    application.SetTimestep(time_step);
    while (application.GetDevice()->run()) {
        application.BeginScene();
        application.DrawAll();
        std::cout << "Time t = " << my_system.GetChTime() << "\n";
        application.DoStep();
        application.EndScene();
    }

    return 0;
}
