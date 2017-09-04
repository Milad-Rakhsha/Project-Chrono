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
#include "chrono/utils/ChUtilsInputOutput.h"
#include "chrono_fea/ChElementShellANCF.h"
#include "chrono_fea/ChLinkDirFrame.h"
#include "chrono_fea/ChLinkPointFrame.h"
#include "chrono_fea/ChVisualizationFEAmesh.h"
#include "chrono_fea/ChMesh.h"
#include "chrono_fea/utils/ChMeshImport.h"
#include "chrono_fea/ChLinkPointFrame.h"
#include "chrono_irrlicht/ChIrrApp.h"
#include "chrono/physics/ChLoaderUV.h"
#include "chrono/physics/ChLoadContainer.h"
#include "chrono_fea/ChVisualizationFEAmesh.h"
#include "chrono/geometry/ChCTriangleMeshConnected.h"
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
using namespace geometry;
using namespace fea;
using namespace irr;
using namespace std;
using namespace scene;

bool addConstrain = false;
bool addForce = false;
bool addGravity = true;
bool addPressure = false;
bool showBone = true;
bool addFixed = false;
double time_step = 0.004;
int scaleFactor = 35;
double dz = 0.01;

int main(int argc, char* argv[]) {
    ChSystemDEM my_system;

    // Create the Irrlicht visualization (open the Irrlicht device,
    // bind a simple user interface, etc. etc.)
    ChIrrApp application(&my_system, L"ANCF Shells", core::dimension2d<u32>(800, 600), false, true);

    // Easy shortcuts to add camera, lights, logo and sky in Irrlicht scene:
    application.AddTypicalLogo();
    application.AddTypicalSky();
    application.AddTypicalLights();
    application.AddTypicalCamera(core::vector3df(-1.f, 0.f, -0.5f),  // camera location
                                 core::vector3df(0.0f, 0.f, 0.f));   // "look at" location
    application.SetContactsDrawMode(irr::ChIrrTools::CONTACT_DISTANCES);

    // collision::ChCollisionModel::SetDefaultSuggestedEnvelope(0.0); // not needed, already 0 when using
    collision::ChCollisionModel::SetDefaultSuggestedMargin(
        0.0005);  // max inside penetration - if not enough stiffness in material: troubles
    // Use this value for an outward additional layer around meshes, that can improve
    // robustness of mesh-mesh collision detection (at the cost of having unnatural inflate effect)
    double sphere_swept_thickness = 0.0008;

    // Create the surface material, containing information
    // about friction etc.
    // It is a DEM-p (penalty) material that we will assign to
    // all surfaces that might generate contacts.
    ChSharedPtr<ChMaterialSurfaceDEM> mysurfmaterial(new ChMaterialSurfaceDEM);
    mysurfmaterial->SetYoungModulus(6e4);
    mysurfmaterial->SetFriction(0.3);
    mysurfmaterial->SetRestitution(0.2);
    mysurfmaterial->SetAdhesion(0);

    GetLog() << "-----------------------------------------------------------\n";
    GetLog() << "-----------------------------------------------------------\n";
    GetLog() << "   		Articular Cartilage Modeling   \n";
    GetLog() << "-----------------------------------------------------------\n\n";

    // Creating Rigid body
    GetLog() << "	Adding the Bone as a Rigid Body ...\n";
    ChSharedPtr<ChBody> Body_Bone;  // Bone
    Body_Bone = ChSharedPtr<ChBody>(new ChBody);
    Body_Bone->SetPos(ChVector<>(0, 0, 0));
    Body_Bone->SetBodyFixed(true);
    Body_Bone->SetMaterialSurface(mysurfmaterial);
    my_system.Add(Body_Bone);
    Body_Bone->SetIdentifier(1);
    Body_Bone->SetMass(1);
    Body_Bone->SetInertiaXX(ChVector<>(1, 0.2, 1));

    ChSharedPtr<ChObjShapeFile> mobjmesh(new ChObjShapeFile);
    mobjmesh->SetFilename(GetChronoDataFile("fea/bone10.obj"));
    if (showBone) {
        Body_Bone->AddAsset(mobjmesh);
    }
    //    ChSharedPtr<ChTriangleMeshShape> masset_meshbox(new ChTriangleMeshShape());
    //    Body_Bone->AddAsset(masset_meshbox);
    //
    //    ChSharedPtr<ChTexture> masset_texture(new ChTexture());
    //    masset_texture->SetTextureFilename(GetChronoDataFile("concrete.jpg"));
    //    Body_Bone->AddAsset(masset_texture);
    //
    //    Body_Bone->GetCollisionModel()->ClearModel();
    //    Body_Bone->GetCollisionModel()->AddTriangleMesh(masset_meshbox, false, false, VNULL, ChMatrix33<>(1),
    //                                                    sphere_swept_thickness);
    //    Body_Bone->GetCollisionModel()->BuildModel();
    //    Body_Bone->SetCollide(true);

    ///////To do collision with bone as well.
    //    ChMatrix33<> rot_transform_bone(1);
    //    ChBodySceneNode* poly = (ChBodySceneNode*)addChBodySceneNode_easyGenericMesh(
    //        &my_system, application.GetSceneManager(), 1, ChVector<>(0, 0, 0), chrono::ChQuaternion<>(1, 0, 0, 0),
    //        GetChronoDataFile("fea/bone35.obj").c_str(),
    //        false,  // not static
    //        true);  // true=convex; false=concave(do convex decomposition of concave mesh
    //    poly->GetBody()->SetBodyFixed(true);
    //    poly->GetBody()->SetCollide(true);
    //    poly->GetBody()->SetMaterialSurface(mysurfmaterial);

    GetLog() << "-----------------------------------------------------------\n\n";

    // Adding the ground
    if (false) {
        ChSharedPtr<ChBodyEasyBox> mfloor(new ChBodyEasyBox(2, 2, 0.1, 2700, true));
        mfloor->SetBodyFixed(true);
        mfloor->SetMaterialSurface(mysurfmaterial);
        my_system.Add(mfloor);

        ChSharedPtr<ChTexture> masset_texture(new ChTexture());
        masset_texture->SetTextureFilename(GetChronoDataFile("concrete.jpg"));
        mfloor->AddAsset(masset_texture);
    }

    int TotalNumNodes, TotalNumElements, TottalNumBEdges;
    std::vector<int> BC_NODES1;
    std::vector<int> BC_NODES2;

    GetLog() << "	Adding the Membrane Using ANCF Shell Elements...  \n\n";
    // Creating the membrane shell
    ChSharedPtr<ChMaterialShellANCF> material(new ChMaterialShellANCF(200, 5e7, 0.3));
    ChSharedPtr<ChMesh> my_mesh(new ChMesh);
    //    ChMatrix33<> rot_transform(0);
    //    rot_transform.SetElement(0, 0, 1);
    //    rot_transform.SetElement(1, 2, -1);
    //    rot_transform.SetElement(2, 1, 1);
    //    ChVector<> Center(-1, 0, 0.2);

    ChMatrix33<> rot_transform(1);
    ChVector<> Center(0, 0, 0);

    // Import the Torus
    try {
        my_mesh->LoadANCFShellFromGMFFile(GetChronoDataFile("fea/Mesh_3.mesh").c_str(), material, BC_NODES1, Center,
                                          rot_transform, 10, true, true, true);
    } catch (ChException myerr) {
        GetLog() << myerr.what();
        return 0;
    }

    for (int node = 0; node < BC_NODES1.size(); node++) {
        ChSharedPtr<ChNodeFEAxyzD> FixedNode(my_mesh->GetNode(BC_NODES1[node]).DynamicCastTo<ChNodeFEAxyzD>());
        FixedNode->SetFixed(true);
    }

    ChMatrix33<> rot_transform_2(0);
    rot_transform_2.SetElement(0, 0, 1);
    rot_transform_2.SetElement(1, 1, -1);
    rot_transform_2.SetElement(2, 2, -1);
    Center = ChVector<>(0, -0.2, 0.4);

    // Import the mesh
    try {
        my_mesh->LoadANCFShellFromGMFFile(GetChronoDataFile("fea/Mesh_3.mesh").c_str(), material, BC_NODES2, Center,
                                          rot_transform_2, 10, true, true, true);
    } catch (ChException myerr) {
        GetLog() << myerr.what();
        return 0;
    }
    //    for (int node = 0; node < BC_NODES2.size(); node++) {
    //        ChSharedPtr<ChNodeFEAxyzD> FixedNode(my_mesh->GetNode(BC_NODES2[node]).DynamicCastTo<ChNodeFEAxyzD>());
    //        if (FixedNode->GetPos().x < 0.01 * scaleFactor)
    //            FixedNode->SetFixed(true);
    //    }

    ChSharedPtr<ChContactSurfaceGeneric> mcontactsurf(new ChContactSurfaceGeneric);
    my_mesh->AddContactSurface(mcontactsurf);
    mcontactsurf->AddFacesFromBoundary(sphere_swept_thickness);  // do this after my_mesh->AddContactSurface
    mcontactsurf->SetMaterialSurface(mysurfmaterial);            // use the DEM penalty contacts

    TotalNumNodes = my_mesh->GetNnodes();
    TotalNumElements = my_mesh->GetNelements();

    for (int ele = 0; ele < TotalNumElements; ele++) {
        ChSharedPtr<ChElementShellANCF> element(my_mesh->GetElement(ele).DynamicCastTo<ChElementShellANCF>());

        // Add a single layers with a fiber angle of 0 degrees.
        element->AddLayer(dz, 0 * CH_C_DEG_TO_RAD, material);
        // Set other element properties
        element->SetAlphaDamp(0.08);   // Structural damping for this element
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

    //    if (addConstrain) {
    //        for (int node = 0; node < BC_NODES.size(); node++) {
    //            ChSharedPtr<ChLinkPointFrame> NodePosBone;
    //            ChSharedPtr<ChLinkDirFrame> NodeDirBone;
    //            ChSharedPtr<ChNodeFEAxyzD> ConstrainedNode;
    //            ConstrainedNode =
    //                ChSharedPtr<ChNodeFEAxyzD>(my_mesh->GetNode(BC_NODES[node]).DynamicCastTo<ChNodeFEAxyzD>());
    //            NodePosBone = ChSharedPtr<ChLinkPointFrame>(new ChLinkPointFrame);
    //            NodePosBone->Initialize(ConstrainedNode, Body_Bone);
    //            my_system.Add(NodePosBone);
    //
    //            NodeDirBone = ChSharedPtr<ChLinkDirFrame>(new ChLinkDirFrame);
    //            NodeDirBone->Initialize(ConstrainedNode, Body_Bone);
    //            NodeDirBone->SetDirectionInAbsoluteCoords(ConstrainedNode->D);
    //            my_system.Add(NodeDirBone);
    //        }
    //}

    //    if (addForce) {
    //        ChSharedPtr<ChNodeFEAxyzD> forceNode(my_mesh->GetNode(0).DynamicCastTo<ChNodeFEAxyzD>());
    //        forceNode->SetForce(ChVector<>(0.0, 1, 0.0));
    //    }
    //    if (addPressure) {
    //        // First: loads must be added to "load containers",
    //        // and load containers must be added to your ChSystem
    //        ChSharedPtr<ChLoadContainer> Mloadcontainer(new ChLoadContainer);
    //        // Add constant pressure using ChLoaderPressure (preferred for simple, constant pressure)
    //        for (int NoElmPre = 0; NoElmPre < TotalNumElements; NoElmPre++) {
    //            ChSharedPtr<ChLoad<ChLoaderPressure>> faceload(
    //                new
    //                ChLoad<ChLoaderPressure>(my_mesh->GetElement(NoElmPre).StaticCastTo<ChElementShellANCF>()));
    //            faceload->loader.SetPressure(350);
    //            faceload->loader.SetStiff(false);
    //            faceload->loader.SetIntegrationPoints(2);
    //            Mloadcontainer->Add(faceload);
    //        }
    //        my_system.Add(Mloadcontainer);
    //    }

    // Switch off mesh class gravity
    my_mesh->SetAutomaticGravity(addGravity);
    my_system.Set_G_acc(ChVector<>(0, -9.8, 0));

    // Add the mesh to the system
    my_system.Add(my_mesh);

    // Mark completion of system construction

    //    ////////////////////////////////////////
    //    // Options for visualization in irrlicht
    //    ////////////////////////////////////////
    ChSharedPtr<ChVisualizationFEAmesh> mvisualizemesh(new ChVisualizationFEAmesh(*(my_mesh.get_ptr())));
    mvisualizemesh->SetFEMdataType(ChVisualizationFEAmesh::E_PLOT_NODE_SPEED_NORM);
    mvisualizemesh->SetColorscaleMinMax(0.0, 5.50);
    mvisualizemesh->SetShrinkElements(true, 0.85);
    mvisualizemesh->SetSmoothFaces(true);
    my_mesh->AddAsset(mvisualizemesh);

    ChSharedPtr<ChVisualizationFEAmesh> mvisualizemeshref(new ChVisualizationFEAmesh(*(my_mesh.get_ptr())));
    mvisualizemeshref->SetFEMdataType(ChVisualizationFEAmesh::E_PLOT_SURFACE);
    mvisualizemeshref->SetWireframe(true);
    mvisualizemeshref->SetDrawInUndeformedReference(true);
    my_mesh->AddAsset(mvisualizemeshref);

    ChSharedPtr<ChVisualizationFEAmesh> mvisualizemeshC(new ChVisualizationFEAmesh(*(my_mesh.get_ptr())));
    mvisualizemeshC->SetFEMglyphType(ChVisualizationFEAmesh::E_GLYPH_NODE_DOT_POS);
    mvisualizemeshC->SetFEMdataType(ChVisualizationFEAmesh::E_PLOT_NONE);
    mvisualizemeshC->SetSymbolsThickness(0.0004 * scaleFactor);
    my_mesh->AddAsset(mvisualizemeshC);

    ChSharedPtr<ChVisualizationFEAmesh> mvisualizemeshD(new ChVisualizationFEAmesh(*(my_mesh.get_ptr())));
    // mvisualizemeshD->SetFEMglyphType(ChVisualizationFEAmesh::E_GLYPH_NODE_VECT_SPEED);
    mvisualizemeshD->SetFEMglyphType(ChVisualizationFEAmesh::E_GLYPH_ELEM_TENS_STRAIN);
    mvisualizemeshD->SetFEMdataType(ChVisualizationFEAmesh::E_PLOT_NONE);
    mvisualizemeshD->SetSymbolsScale(1);
    mvisualizemeshD->SetColorscaleMinMax(-0.5, 5);
    mvisualizemeshD->SetZbufferHide(false);
    my_mesh->AddAsset(mvisualizemeshD);

    ChSharedPtr<ChVisualizationFEAmesh> mvisualizemeshcoll(new ChVisualizationFEAmesh(*(my_mesh.get_ptr())));
    mvisualizemeshcoll->SetFEMdataType(ChVisualizationFEAmesh::E_PLOT_CONTACTSURFACES);
    mvisualizemeshcoll->SetWireframe(true);
    mvisualizemeshcoll->SetDefaultMeshColor(ChColor(1, 0.5, 0));
    my_mesh->AddAsset(mvisualizemeshcoll);

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

    // Setup solver
    //    my_system.SetLcpSolverType(ChSystem::LCP_ITERATIVE_MINRES);
    //    ChLcpIterativeMINRES* msolver = (ChLcpIterativeMINRES*)my_system.GetLcpSolverSpeed();
    //    msolver->SetDiagonalPreconditioning(true);
    //    my_system.SetIterLCPwarmStarting(true);  // this helps a lot to speedup convergence in this class of
    //    my_system.SetIterLCPmaxItersSpeed(40);
    //    //    my_system.SetIterLCPmaxItersStab(1000);
    //    my_system.SetTolForce(1e-10);
    //    msolver->SetVerbose(false);
    //
    //    //    ChLcpMklSolver* mkl_solver_stab = new ChLcpMklSolver;  // MKL Solver option
    //    //    ChLcpMklSolver* mkl_solver_speed = new ChLcpMklSolver;
    //    //    my_system.ChangeLcpSolverStab(mkl_solver_stab);
    //    //    my_system.ChangeLcpSolverSpeed(mkl_solver_speed);
    //    //    mkl_solver_speed->SetSparsityPatternLock(true);
    //    //    mkl_solver_stab->SetSparsityPatternLock(true);
    //
    //    my_system.SetIntegrationType(ChSystem::INT_HHT);
    //    ChSharedPtr<ChTimestepperHHT> mystepper = my_system.GetTimestepper().DynamicCastTo<ChTimestepperHHT>();
    //    mystepper->SetAlpha(-0.2);
    //    mystepper->SetMaxiters(200);
    //    mystepper->SetTolerance(1e-06);
    //    mystepper->SetMode(ChTimestepperHHT::POSITION);
    //    mystepper->SetScaling(true);
    //    mystepper->SetVerbose(false);

    my_system.SetLcpSolverType(ChSystem::LCP_ITERATIVE_MINRES);
    my_system.SetIterLCPwarmStarting(true);  // this helps a lot to speedup convergence in this class of problems
    my_system.SetIterLCPmaxItersSpeed(40);
    my_system.SetTolForce(1e-10);
    my_system.SetIntegrationType(ChSystem::INT_EULER_IMPLICIT_LINEARIZED);

    application.SetTimestep(time_step);
    //    ChSharedPtr<ChNodeFEAxyzD> sampleNode(my_mesh->GetNode(89).DynamicCastTo<ChNodeFEAxyzD>());
    //    double y0 = sampleNode->pos.y;
    while (application.GetDevice()->run()) {
        application.BeginScene();
        application.DrawAll();
        //        std::cout << "Time t = " << my_system.GetChTime() << "s \t";
        //        std::cout << "pos.y = " << sampleNode->pos.y - y0 << "vs. " << -0.5 * 9.8 *
        //        pow(my_system.GetChTime(),
        //        2)
        //                  << "\n";
        double t_s = my_system.GetChTime();

        application.DoStep();
        application.EndScene();
    }

    return 0;
}
