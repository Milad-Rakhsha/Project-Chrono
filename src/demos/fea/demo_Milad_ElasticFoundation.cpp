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
bool addForce = true;
bool showTibia = false;
bool showFemur = true;
// bool addFixed = false;
double time_step = 0.0005;
int scaleFactor = 1;
double dz = 0.001;
// Stiffness of the foundation
double K_S = 1;     // Stiffness Constant
double C_S = 0.05;  // Damper Constant
double L0 = 1.0;    // Initial length

double MeterToInch = 0.02539998628;

class MyLoadSpringDamper : public ChLoadCustomMultiple {
  public:
    MyLoadSpringDamper(std::vector<std::shared_ptr<ChLoadable>>& mloadables, std::shared_ptr<ChBody> AttachBodyInput)
        : ChLoadCustomMultiple(mloadables),
          AttachBody(AttachBodyInput),
          K_sp(mloadables.size(), 1e4),
          C_dp(mloadables.size(), 1e2),
          LocalBodyAtt(mloadables.size(), AttachBody->GetPos()) {
        assert(std::dynamic_pointer_cast<ChNodeFEAxyzD>(mloadables[0]));
        loadables.push_back(AttachBody);
        load_Q.Reset(this->LoadGet_ndof_w());
        for (size_t i = 0; i < mloadables.size(); i++) {
            auto Node = std::static_pointer_cast<ChNodeFEAxyzD>(mloadables[i]);
            l0.push_back((Node->GetPos() - AttachBody->GetPos()).Length());
        }
    }

    std::shared_ptr<ChBody> AttachBody;    // Body to attach springs and dampers
    std::vector<double> K_sp;              // Stiffness of springs
    std::vector<double> C_dp;              // Damping coefficient of dampers
    std::vector<double> l0;                // Initial, undeformed spring-damper length
    std::vector<ChVector<>> LocalBodyAtt;  // Local Body Attachment

    virtual void ComputeQ(ChState* state_x,      ///< state position to evaluate Q
                          ChStateDelta* state_w  ///< state speed to evaluate Q
                          ) {
        ChVector<> Node_Pos;
        ChVector<> Node_Vel;
        ChVector<> Node_Grad;

        ChVector<> dij;   // Vector describing current spring-damper relative location
        ChVector<> ddij;  // Vector describing current spring-damper relative velocity

        ChVector<> NodeAttachment;
        if (state_x && state_w) {
            // GetLog() << " We are reaching this" << loadables.size();
            for (int iii = 0; iii < loadables.size() - 1; iii++) {
                Node_Pos = state_x->ClipVector(iii * 6, 0);
                Node_Grad = state_x->ClipVector(iii * 6 + 3, 0);
                Node_Vel = state_w->ClipVector(iii * 6, 0);

                ChVector<> BodyAttachWorld = AttachBody->Point_Body2World(LocalBodyAtt[iii]);
                dij = Node_Pos - BodyAttachWorld;  // Current relative vector between attachment points
                double c_length = dij.Length();    // l
                ddij = Node_Vel - AttachBody->PointSpeedLocalToParent(LocalBodyAtt[iii]);  // Time derivative of dij
                double dc_length = 1 / c_length * (dij.x * ddij.x + dij.y * ddij.y + dij.z * ddij.z);  // ldot
                double for_spdp =
                    K_sp[iii] * (c_length - l0[iii]) + C_dp[iii] * dc_length;  // Absolute value of spring-damper force

                // Apply generalized force to node
                //    GetLog() << " Vertical component and absolute value of force: " << dij.GetNormalized().y << "  "
                //    << for_spdp;
                this->load_Q(iii * 6 + 0) = -for_spdp * dij.GetNormalized().x;
                this->load_Q(iii * 6 + 1) = -for_spdp * dij.GetNormalized().y;
                this->load_Q(iii * 6 + 2) = -for_spdp * dij.GetNormalized().z;

                ChVectorDynamic<> Qi(6);  // Vector of generalized forces from spring and damper
                ChVectorDynamic<> Fi(6);  // Vector of applied forces and torques (6 components)
                double detJi = 0;         // Determinant of transformation (Not used)

                Fi(0) = for_spdp * dij.GetNormalized().x;
                Fi(1) = for_spdp * dij.GetNormalized().y;
                Fi(2) = for_spdp * dij.GetNormalized().z;
                Fi(3) = 0.0;
                Fi(4) = 0.0;
                Fi(5) = 0.0;

                ChState stateBody_x(7, NULL);
                ChStateDelta stateBody_w(6, NULL);

                stateBody_x(0) = (*state_x)((loadables.size() - 1) * 6);
                stateBody_x(1) = (*state_x)((loadables.size() - 1) * 6 + 1);
                stateBody_x(2) = (*state_x)((loadables.size() - 1) * 6 + 2);
                stateBody_x(3) = (*state_x)((loadables.size() - 1) * 6 + 3);
                stateBody_x(4) = (*state_x)((loadables.size() - 1) * 6 + 4);
                stateBody_x(5) = (*state_x)((loadables.size() - 1) * 6 + 5);
                stateBody_x(6) = (*state_x)((loadables.size() - 1) * 6 + 6);

                stateBody_w(0) = (*state_w)((loadables.size() - 1) * 6);
                stateBody_w(1) = (*state_w)((loadables.size() - 1) * 6 + 1);
                stateBody_w(2) = (*state_w)((loadables.size() - 1) * 6 + 2);
                stateBody_w(3) = (*state_w)((loadables.size() - 1) * 6 + 3);
                stateBody_w(4) = (*state_w)((loadables.size() - 1) * 6 + 4);
                stateBody_w(5) = (*state_w)((loadables.size() - 1) * 6 + 5);

                // Apply generalized force to rigid body (opposite sign)
                AttachBody->ComputeNF(BodyAttachWorld.x, BodyAttachWorld.y, BodyAttachWorld.z, Qi, detJi, Fi,
                                      &stateBody_x, &stateBody_w);

                // Apply forces to body (If body fixed, we should set those to zero not Qi(coordinate))
                this->load_Q((loadables.size() - 1) * 6) = 0.0;
                this->load_Q((loadables.size() - 1) * 6 + 1) = 0.0;
                this->load_Q((loadables.size() - 1) * 6 + 2) = 0.0;
                this->load_Q((loadables.size() - 1) * 6 + 3) = 0.0;
                this->load_Q((loadables.size() - 1) * 6 + 4) = 0.0;
                this->load_Q((loadables.size() - 1) * 6 + 5) = 0.0;
            }
        } else {
            // explicit integrators might call ComputeQ(0,0), null pointers mean
            // that we assume current state, without passing state_x for efficiency
        }
    }
    // Remember to set this as stiff, to enable the jacobians
    virtual bool IsStiff() { return false; }
};

int main(int argc, char* argv[]) {
    ChSystemDEM my_system;

    // Create the Irrlicht visualization (open the Irrlicht device,
    // bind a simple user interface, etc. etc.)
    ChIrrApp application(&my_system, L"Articular Cartilage", core::dimension2d<u32>(1200, 800), false, true);

    // Easy shortcuts to add camera, lights, logo and sky in Irrlicht scene:
    application.AddTypicalLogo();
    application.AddTypicalSky();
    application.AddTypicalLights();
    application.AddTypicalCamera(core::vector3df(-.5f, 0.f, 1.f),   // camera location
                                 core::vector3df(0.0f, 0.f, 0.f));  // "look at" location
    application.SetContactsDrawMode(ChIrrTools::CONTACT_DISTANCES);

    // collision::ChCollisionModel::SetDefaultSuggestedEnvelope(0.0); // not needed, already 0 when using
    collision::ChCollisionModel::SetDefaultSuggestedMargin(1.5);  // max inside penetration - if not enough stiffness in
                                                                  // material: troubles
    // Use this value for an outward additional layer around meshes, that can improve
    // robustness of mesh-mesh collision detection (at the cost of having unnatural inflate effect)
    double sphere_swept_thickness = dz * 0.01;

    double rho = 1000;  ///< material density
    double E = 7e09;    ///< Young's modulus
    double nu = 0.3;    ///< Poisson ratio
    // Create the surface material, containing information
    // about friction etc.
    // It is a DEM-p (penalty) material that we will assign to
    // all surfaces that might generate contacts.

    auto mysurfmaterial = std::make_shared<ChMaterialSurfaceDEM>();
    mysurfmaterial->SetYoungModulus(1.3e3);
    mysurfmaterial->SetFriction(0.3f);
    mysurfmaterial->SetRestitution(0.5f);
    mysurfmaterial->SetAdhesion(0);

    GetLog() << "-----------------------------------------------------------\n\n";
    // Adding the ground
    //    if (true) {
    auto mfloor = std::make_shared<ChBodyEasyBox>(2, 0.1, 2, 8000, true);
    mfloor->SetBodyFixed(true);
    mfloor->SetMaterialSurface(mysurfmaterial);
    my_system.Add(mfloor);
    auto masset_texture = std::make_shared<ChTexture>();
    masset_texture->SetTextureFilename(GetChronoDataFile("concrete.jpg"));
    mfloor->AddAsset(masset_texture);
    //    }
    int TotalNumNodes, TotalNumElements, TottalNumBEdges;
    std::vector<int> BC_NODES;

    GetLog() << "	Adding the Membrane Using ANCF Shell Elements...  \n\n";
    // Creating the membrane shell
    auto material = std::make_shared<ChMaterialShellANCF>(rho, E, nu);
    auto my_mesh = std::make_shared<ChMesh>();
    ChMatrix33<> rot_transform(0);
    rot_transform.SetElement(0, 0, 1);
    rot_transform.SetElement(1, 2, -1);
    rot_transform.SetElement(2, 1, 1);
    ChVector<> Center(0, 0.50, 0);
    //    ChMatrix33<> rot_transform(1);
    // Import the Torus
    try {
        ChMeshFileLoader::ANCFShellFromGMFFile(my_mesh, GetChronoDataFile("fea/Plate.mesh").c_str(), material, BC_NODES,
                                               Center, rot_transform, 1, false, false);
    } catch (ChException myerr) {
        GetLog() << myerr.what();
        return 0;
    }

    /// TEST TO INCLUDE SPRING-DAMPER GENERALIZED FORCES
    auto mloadcontainerGround = std::make_shared<ChLoadContainer>();

    if (true) {
        // Select on which nodes we are going to apply a load
        std::vector<std::shared_ptr<ChLoadable>> NodeList;
        for (int iNode = 0; iNode < my_mesh->GetNnodes(); iNode++) {
            auto NodeLoad = std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(iNode));
            NodeList.push_back(NodeLoad);
        }
        auto OneLoadSpringDamper = std::make_shared<MyLoadSpringDamper>(NodeList, mfloor);
        for (int iNode = 0; iNode < my_mesh->GetNnodes(); iNode++) {
            auto Node = std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(iNode));
            ChVector<> AttachBodyGlobal = Node->GetPos() - L0 * Node->GetD();  // Locate first the
            // attachment point in the body in global coordiantes
            OneLoadSpringDamper->C_dp[iNode] = C_S;
            OneLoadSpringDamper->K_sp[iNode] = K_S;
            OneLoadSpringDamper->l0[iNode] = L0;
            OneLoadSpringDamper->LocalBodyAtt[iNode] = mfloor->Point_World2Body(AttachBodyGlobal);
        }
        mloadcontainerGround->Add(OneLoadSpringDamper);
    }  // End loop over tires
    my_system.Add(mloadcontainerGround);

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

    if (addPressure) {
        // First: loads must be added to "load containers",
        // and load containers must be added to your ChSystem
        auto Mloadcontainer = std::make_shared<ChLoadContainer>();
        // Add constant pressure using ChLoaderPressure (preferred for simple, constant pressure)
        for (int NoElmPre = 0; NoElmPre < TotalNumElements; NoElmPre++) {
            auto faceload = std::make_shared<ChLoad<ChLoaderPressure>>(
                std::static_pointer_cast<ChElementShellANCF>(my_mesh->GetElement(NoElmPre)));
            faceload->loader.SetPressure(-200);
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
    mvisualizemeshcoll->SetDefaultMeshColor(ChColor(1, 1.0, 1));
    my_mesh->AddAsset(mvisualizemeshcoll);

    auto mvisualizemeshbeam = std::make_shared<ChVisualizationFEAmesh>(*(my_mesh.get()));
    mvisualizemeshbeam->SetFEMdataType(ChVisualizationFEAmesh::E_PLOT_ELEM_STRAIN_VONMISES);
    mvisualizemeshbeam->SetColorscaleMinMax(-0.0, 0.010);
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
    //    -------------- -
    //    ChLcpMklSolver* mkl_solver_stab = new ChLcpMklSolver;
    //    ChLcpMklSolver* mkl_solver_speed = new ChLcpMklSolver;
    //    my_system.ChangeLcpSolverStab(mkl_solver_stab);
    //    my_system.ChangeLcpSolverSpeed(mkl_solver_speed);
    //    mkl_solver_stab->SetSparsityPatternLock(true);
    //    mkl_solver_speed->SetSparsityPatternLock(true);
    //    application.GetSystem()->Update();

    // Setup solver
    my_system.SetLcpSolverType(ChSystem::LCP_ITERATIVE_MINRES);
    ChLcpIterativeMINRES* msolver = (ChLcpIterativeMINRES*)my_system.GetLcpSolverSpeed();
    msolver->SetDiagonalPreconditioning(true);
    my_system.SetIterLCPwarmStarting(true);  // this helps a lot to speedup convergence in this class of
    my_system.SetIterLCPmaxItersSpeed(40000000);
    my_system.SetTolForce(1e-6);
    msolver->SetVerbose(false);
    //
    // INT_HHT or INT_EULER_IMPLICIT
    my_system.SetIntegrationType(ChSystem::INT_EULER_IMPLICIT_LINEARIZED);  // fast, less precise
    auto NodeTip = std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(5));
    //
    //    my_system.SetIntegrationType(ChSystem::INT_HHT);
    //    auto mystepper = std::dynamic_pointer_cast<ChTimestepperHHT>(my_system.GetTimestepper());
    //    mystepper->SetAlpha(-0.2);
    //    mystepper->SetMaxiters(200);
    //    mystepper->SetAbsTolerances(1e-05, 1e-2);
    //    mystepper->SetMaxiters(16);
    //    mystepper->SetMaxItersSuccess(5);
    //    mystepper->SetMode(ChTimestepperHHT::POSITION);
    //    mystepper->SetScaling(true);
    //    mystepper->SetVerbose(false);
    //    application.SetPaused(false);
    application.SetTimestep(time_step);
    while (application.GetDevice()->run()) {
        application.BeginScene();
        application.DrawAll();
        std::cout << "Time t = " << my_system.GetChTime() << ", Node Pos = " << NodeTip->GetPos().y << "\n";

        application.DoStep();
        application.EndScene();
    }

    return 0;
}

