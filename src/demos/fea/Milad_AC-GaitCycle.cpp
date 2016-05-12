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
#include "chrono/physics/ChSystemDEM.h"
#include "chrono/physics/ChLoadBodyMesh.h"
#include "chrono/core/ChFileutils.h"
#include "chrono/ChConfig.h"

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
#include <stdio.h>

using namespace chrono;
using namespace chrono::geometry;
using namespace chrono::fea;
using namespace chrono::irrlicht;
using namespace irr;
using namespace std;
int num_threads = 4;
//#define USE_IRR ;
enum ROT_SYS { XYZ, ZXY };  // Only these are supported for now ...

bool outputData = true;
bool addGravity = false;
bool addPressure = false;
bool addForce = false;
bool showTibia = true;
bool showFemur = true;
bool tibiaCartilage = true;
bool femurCartilage = true;

ROT_SYS myRot = XYZ;
double time_step = 0.0001;
int scaleFactor = 1;
double dz = 0.001;
const double K_SPRINGS = 40e8;
const double C_DAMPERS = 40e3;
double MeterToInch = 0.02539998628;
double L0 = 0.01;  // Initial length
double L0_t = 0.01;
int write_interval = 0.005 / time_step;  // write every 0.01s

void ReadOBJConnectivity(const char* filename,
                         string SaveAs,
                         std::vector<std::vector<double>>& vCoor,
                         std::vector<std::vector<int>>& faces);
void WriteRigidBodyVTK(std::shared_ptr<ChBody> Body,
                       std::vector<std::vector<double>>& vCoor,
                       char SaveAsBuffer[64],
                       char ConnectivityFileBuffer[32]);
void GetDataFile(const char* filename, std::vector<std::vector<double>>& DATA);
void impose_TF_motion(std::vector<std::vector<double>> motionInfo,
                      int angleset,
                      std::shared_ptr<ChLinkLockLock> my_link_TF);
void writeMesh(std::shared_ptr<ChMesh> my_mesh, string SaveAs, std::vector<std::vector<int>>& NodeNeighborElement);
void writeFrame(std::shared_ptr<ChMesh> my_mesh,
                char SaveAsBuffer[64],
                char MeshFileBuffer[32],
                std::vector<std::vector<int>> NodeNeighborElement,
                std::vector<ChVector<>> NodeFrc);
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
                //    GetLog() << " Vertical component and absolute value of force: " << dij.GetNormalized().y << "
                //    "
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
                if (!AttachBody->GetBodyFixed()) {
                    this->load_Q((loadables.size() - 1) * 6) = -for_spdp * dij.GetNormalized().x;
                    this->load_Q((loadables.size() - 1) * 6 + 1) = -for_spdp * dij.GetNormalized().y;
                    this->load_Q((loadables.size() - 1) * 6 + 2) = -for_spdp * dij.GetNormalized().z;
                    this->load_Q((loadables.size() - 1) * 6 + 3) = 0.0;
                    this->load_Q((loadables.size() - 1) * 6 + 4) = 0.0;
                    this->load_Q((loadables.size() - 1) * 6 + 5) = 0.0;
                }
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
    ChSystemDEM my_system(false, true, 16000, 500);

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

#pragma omp parallel
    { printf("OMP NUM THREADS=%d ", omp_get_num_threads()); }

#ifdef USE_IRR

    ChIrrApp application(&my_system, L"Articular Cartilage", core::dimension2d<u32>(1200, 800), false, false);

    // Easy shortcuts to add camera, lights, logo and sky in Irrlicht scene:
    application.AddTypicalLogo();
    application.AddTypicalSky();
    application.AddTypicalLights();
    application.AddTypicalLights(core::vector3df(0, -5, 0), core::vector3df(0, 5, 0), 90, 90,
                                 irr::video::SColorf(0.5, 0.5, 0.5));

    application.AddTypicalCamera(core::vector3df(-0.1f, -0.000f, -0.0f),  // camera location
                                 core::vector3df(0.1f, 0.000f, -0.0f));   // "look at" location
    application.SetContactsDrawMode(ChIrrTools::CONTACT_DISTANCES);
#endif

    // collision::ChCollisionModel::SetDefaultSuggestedEnvelope(0.0); // not needed, already 0 when using
    collision::ChCollisionModel::SetDefaultSuggestedMargin(1.5);  // max inside penetration - if not enough stiffness in
                                                                  // material: troubles
    // Use this value for an outward additional layer around meshes, that can improve
    // robustness of mesh-mesh collision detection (at the cost of having unnatural inflate effect)
    double sphere_swept_thickness = dz * 0.2;

    double rho = 1000 * 0.005 / dz;  ///< material density
    double E = 40e7;                 ///< Young's modulus
    double nu = 0.3;                 ///< Poisson ratio
    // Create the surface material, containing information
    // about friction etc.
    // It is a DEM-p (penalty) material that we will assign to
    // all surfaces that might generate contacts.
    my_system.SetContactForceModel(ChSystemDEM::Hooke);
    auto mysurfmaterial = std::make_shared<ChMaterialSurfaceDEM>();

    mysurfmaterial->SetKn(2e5);
    mysurfmaterial->SetKt(0);
    mysurfmaterial->SetGn(5);
    mysurfmaterial->SetGt(0);

    GetLog() << "-----------------------------------------------------------\n";
    GetLog() << "-----------------------------------------------------------\n";
    GetLog() << "   		Articular Cartilage Modeling   \n";
    GetLog() << "-----------------------------------------------------------\n\n";
    ////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////
    // Creating Rigid body
    GetLog() << "	Adding the Tibia as a Rigid Body ...\n";
    auto Tibia = std::make_shared<ChBody>();
    Tibia->SetBodyFixed(false);
    Tibia->SetMaterialSurface(mysurfmaterial);
    my_system.Add(Tibia);
    Tibia->SetMass(200);
    Tibia->SetInertiaXX(ChVector<>(0.004, 0.8e-4, 0.004));
    auto mobjmesh1 = std::make_shared<ChObjShapeFile>();
    mobjmesh1->SetFilename(GetChronoDataFile("fea/tibia.obj"));
    if (showTibia) {
        Tibia->AddAsset(mobjmesh1);
    }

    //    GetLog() << "	Adding the Femur as a Rigid Body ...\n";
    ChVector<> Center_Femur(0, 0.0, 0);
    auto Femur = std::make_shared<ChBody>();
    Femur->SetPos(Center_Femur);
    Femur->SetBodyFixed(true);
    Femur->SetMaterialSurface(mysurfmaterial);
    Femur->SetMass(0.2);
    Femur->SetInertiaXX(ChVector<>(0.004, 0.8e-4, 0.004));
    auto mobjmesh2 = std::make_shared<ChObjShapeFile>();
    mobjmesh2->SetFilename(GetChronoDataFile("fea/femur.obj"));
    if (showFemur) {
        Femur->AddAsset(mobjmesh2);
    }

    auto MarkerFemur = std::make_shared<ChMarker>();
    auto MarkerTibia = std::make_shared<ChMarker>();
    Femur->AddMarker(MarkerFemur);
    Tibia->AddMarker(MarkerTibia);
    my_system.Add(Femur);

    auto my_link_TF = std::make_shared<ChLinkLockLock>();
    //    my_link_TF->Initialize(MarkerFemur, MarkerTibia);
    my_link_TF->Initialize(MarkerTibia, MarkerFemur);
    //    my_link_TF->Initialize(Tibia, Femur, ChCoordsys<>());

    my_system.AddLink(my_link_TF);

    std::vector<std::vector<double>> motionInfo;
    double D2R = 3.1415 / 180;
    ChVector<> RotAng(0, 0, 0);
    ChMatrix33<> MeshRotate;

    switch (myRot) {
        case XYZ: {
            GetDataFile(GetChronoDataFile("fea/XYZ.csv").c_str(), motionInfo);
            // motionInfo: t,TF_tx,TF_ty,TF_tz,TF_rx,TF_ry,TF_rz,PF_tx,PF_ty,PF_tz,PF_rx,PF_ry,PF_rz
            Tibia->SetPos(ChVector<>(motionInfo[0][1], motionInfo[0][2], motionInfo[0][3]));
            RotAng = ChVector<>(-motionInfo[0][4], -motionInfo[0][5], -motionInfo[0][6]) * D2R;
            // why negative sign is needed here?
            GetLog() << "\n RotAng is =" << RotAng << "\n ";
            MeshRotate = Angle_to_Quat(4, RotAng);

            Tibia->SetRot(MeshRotate);
            impose_TF_motion(motionInfo, 4, my_link_TF);
            break;
        }
        case ZXY: {
            GetDataFile(GetChronoDataFile("fea/ZXY.csv").c_str(), motionInfo);
            // motionInfo: t,TF_tx,TF_ty,TF_tz,TF_rx,TF_ry,TF_rz,PF_tx,PF_ty,PF_tz,PF_rx,PF_ry,PF_rz
            RotAng = ChVector<>(-motionInfo[0][4], -motionInfo[0][5], -motionInfo[0][6]) * D2R;
            MeshRotate = Angle_to_Quat(7, RotAng);
            Tibia->SetPos(ChVector<>(motionInfo[0][1], motionInfo[0][2], motionInfo[0][3]));
            Tibia->SetRot(MeshRotate);
            impose_TF_motion(motionInfo, 7, my_link_TF);
            break;
        }
    }

    //    ChQuaternion<> RotationZ;
    //    RotationZ = Q_from_AngZ(toRad * motionInfo[0][6]);
    //    ChQuaternion<> RotationY;
    //    RotationY = Q_from_AngY(-toRad * motionInfo[0][5]);
    //    ChQuaternion<> RotationX;
    //    RotationX = Q_from_AngX(toRad * motionInfo[0][4]);
    //

    //    ////Connect Fumer to Tibia through a plane-plane joint.
    //    ////The normal to the common plane is along the z global axis.
    //    auto plane_plane = std::make_shared<ChLinkLockPlanePlane>();
    //    my_system.AddLink(plane_plane);
    //    plane_plane->SetName("plane_plane");
    //    plane_plane->Initialize(Femur, Tibia, ChCoordsys<>(ChVector<>(0, 0, 0)));

    GetLog() << "-----------------------------------------------------------\n\n";

    std::vector<double> NODE_AVE_AREA_t, NODE_AVE_AREA_f;
    std::vector<int> BC_NODES, BC_NODES1, BC_NODES2;

    GetLog() << "	Adding the Membrane Using ANCF Shell Elements...  \n\n";
    // Creating the membrane shell
    auto material = std::make_shared<ChMaterialShellANCF>(rho, E, nu);
    auto my_mesh_femur = std::make_shared<ChMesh>();
    auto my_mesh_tibia = std::make_shared<ChMesh>();

    ChMatrix33<> rot_transform(1);
    double Tottal_stiff = 0;
    double Tottal_damp = 0;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////// Femur /////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    //    Import the Femur

    if (femurCartilage) {
        try {
            ChMeshFileLoader::ANCFShellFromGMFFile(my_mesh_femur, GetChronoDataFile("fea/Femur.mesh").c_str(), material,
                                                   NODE_AVE_AREA_f, BC_NODES, Center_Femur, rot_transform, MeterToInch,
                                                   false, false);
        } catch (ChException myerr) {
            GetLog() << myerr.what();
            return 0;
        }
        for (int node = 0; node < BC_NODES.size(); node++) {
            auto FixedNode = std::make_shared<ChNodeFEAxyzD>();
            FixedNode = std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh_femur->GetNode(BC_NODES[node]));
            FixedNode->SetFixed(true);
        }

        //
        auto mloadcontainerFemur = std::make_shared<ChLoadContainer>();
        if (true) {
            // Select on which nodes we are going to apply a load
            std::vector<std::shared_ptr<ChLoadable>> NodeList;
            for (int iNode = 0; iNode < my_mesh_femur->GetNnodes(); iNode++) {
                auto NodeLoad = std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh_femur->GetNode(iNode));
                NodeList.push_back(NodeLoad);
            }
            auto OneLoadSpringDamperFemur = std::make_shared<MyLoadSpringDamper>(NodeList, Femur);
            for (int iNode = 0; iNode < my_mesh_femur->GetNnodes(); iNode++) {
                auto Node = std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh_femur->GetNode(iNode));
                ChVector<> AttachBodyGlobal = Node->GetPos() - L0_t * Node->GetD();  // Locate first the
                // attachment point in the body in global coordiantes
                // Stiffness of the Elastic Foundation
                double K_S = K_SPRINGS * NODE_AVE_AREA_f[iNode];  // Stiffness Constant
                double C_S = C_DAMPERS * NODE_AVE_AREA_f[iNode];  // Damper Constant
                Tottal_stiff += K_S;
                Tottal_damp += C_S;
                // Initial length
                OneLoadSpringDamperFemur->C_dp[iNode] = C_S;
                OneLoadSpringDamperFemur->K_sp[iNode] = K_S;
                OneLoadSpringDamperFemur->l0[iNode] = L0_t;
                // Calculate the damping ratio zeta
                double zeta, m_ele;
                m_ele = rho * dz * NODE_AVE_AREA_f[iNode];
                zeta = C_S / (2 * sqrt(K_S * m_ele));
                //            GetLog() << "Zeta of node # " << iNode << " is set to : " << zeta << "\n";
                OneLoadSpringDamperFemur->LocalBodyAtt[iNode] = Femur->Point_World2Body(AttachBodyGlobal);
            }
            mloadcontainerFemur->Add(OneLoadSpringDamperFemur);
            GetLog() << "Total Stiffness (N/mm)= " << Tottal_stiff / 1e3 << " Total Damping = " << Tottal_damp
                     << " Average zeta= " << Tottal_damp / (2 * sqrt(Tottal_stiff * (rho * dz * 1e-3))) << "\n";
        }
        my_system.Add(mloadcontainerFemur);
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////// Tibia /////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    if (tibiaCartilage) {
        try {
            ChMeshFileLoader::ANCFShellFromGMFFile(my_mesh_tibia, GetChronoDataFile("fea/Tibia-1.mesh").c_str(),
                                                   material, NODE_AVE_AREA_t, BC_NODES1, Tibia->GetPos(), MeshRotate,
                                                   MeterToInch, false, false);
        } catch (ChException myerr) {
            GetLog() << myerr.what();
            return 0;
        }

        for (int node = 0; node < BC_NODES1.size(); node++) {
            auto NodePosBone = std::make_shared<ChLinkPointFrame>();
            auto NodeDirBone = std::make_shared<ChLinkDirFrame>();
            auto ConstrainedNode = std::make_shared<ChNodeFEAxyzD>();

            ConstrainedNode = std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh_tibia->GetNode(BC_NODES1[node]));
            NodePosBone->Initialize(ConstrainedNode, Tibia);
            my_system.Add(NodePosBone);

            NodeDirBone->Initialize(ConstrainedNode, Tibia);
            NodeDirBone->SetDirectionInAbsoluteCoords(ConstrainedNode->D);
            my_system.Add(NodeDirBone);
        }

        //   Import the Tibia
        try {
            ChMeshFileLoader::ANCFShellFromGMFFile(my_mesh_tibia, GetChronoDataFile("fea/Tibia-2.mesh").c_str(),
                                                   material, NODE_AVE_AREA_t, BC_NODES2, Tibia->GetPos(), MeshRotate,
                                                   MeterToInch, false, false);
        } catch (ChException myerr) {
            GetLog() << myerr.what();
            return 0;
        }

        for (int node = 0; node < BC_NODES2.size(); node++) {
            auto NodePosBone = std::make_shared<ChLinkPointFrame>();
            auto NodeDirBone = std::make_shared<ChLinkDirFrame>();
            auto ConstrainedNode = std::make_shared<ChNodeFEAxyzD>();

            ConstrainedNode = std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh_tibia->GetNode(BC_NODES2[node]));
            NodePosBone->Initialize(ConstrainedNode, Tibia);
            my_system.Add(NodePosBone);

            NodeDirBone->Initialize(ConstrainedNode, Tibia);
            NodeDirBone->SetDirectionInAbsoluteCoords(ConstrainedNode->D);
            my_system.Add(NodeDirBone);
        }

        /// TEST TO INCLUDE SPRING-DAMPER GENERALIZED FORCES
        auto mloadcontainerTibia = std::make_shared<ChLoadContainer>();
        Tottal_stiff = 0;
        Tottal_damp = 0;
        if (true) {
            // Select on which nodes we are going to apply a load
            std::vector<std::shared_ptr<ChLoadable>> NodeList;
            for (int iNode = 0; iNode < my_mesh_tibia->GetNnodes(); iNode++) {
                auto NodeLoad = std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh_tibia->GetNode(iNode));
                NodeList.push_back(NodeLoad);
            }
            auto OneLoadSpringDamperTibia = std::make_shared<MyLoadSpringDamper>(NodeList, Tibia);
            for (int iNode = 0; iNode < my_mesh_tibia->GetNnodes(); iNode++) {
                auto Node = std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh_tibia->GetNode(iNode));
                ChVector<> AttachBodyGlobal = Node->GetPos() - L0_t * Node->GetD();  // Locate first the
                // attachment point in the body in global coordiantes
                // Stiffness of the Elastic Foundation
                double K_S = K_SPRINGS * NODE_AVE_AREA_t[iNode];  // Stiffness Constant
                double C_S = C_DAMPERS * NODE_AVE_AREA_t[iNode];  // Damper Constant
                Tottal_stiff += K_S;
                Tottal_damp += C_S;
                // Initial length
                OneLoadSpringDamperTibia->C_dp[iNode] = C_S;
                OneLoadSpringDamperTibia->K_sp[iNode] = K_S;
                OneLoadSpringDamperTibia->l0[iNode] = L0_t;
                // Calculate the damping ratio zeta
                double zeta, m_ele;
                m_ele = rho * dz * NODE_AVE_AREA_t[iNode];
                zeta = C_S / (2 * sqrt(K_S * m_ele));
                //            GetLog() << "Zeta of node # " << iNode << " is set to : " << zeta << "\n";
                OneLoadSpringDamperTibia->LocalBodyAtt[iNode] = Tibia->Point_World2Body(AttachBodyGlobal);
            }
            GetLog() << "Total Stiffness (N/mm)= " << Tottal_stiff / 1e3 << " Total Damping = " << Tottal_damp
                     << " Average zeta= " << Tottal_damp / (2 * sqrt(Tottal_stiff * (rho * dz * 1e-3))) << "\n";
            mloadcontainerTibia->Add(OneLoadSpringDamperTibia);
        }  // End loop over tires
        my_system.Add(mloadcontainerTibia);
    }
    double TotalNumNodes_Femur = my_mesh_femur->GetNnodes();
    double TotalNumElements_Femur = my_mesh_femur->GetNelements();
    for (int ele = 0; ele < TotalNumElements_Femur; ele++) {
        auto element = std::make_shared<ChElementShellANCF>();
        element = std::dynamic_pointer_cast<ChElementShellANCF>(my_mesh_femur->GetElement(ele));
        // Add a single layers with a fiber angle of 0 degrees.
        element->AddLayer(dz, 0 * CH_C_DEG_TO_RAD, material);
        // Set other element properties
        element->SetAlphaDamp(0.05);   // Structural damping for this element
        element->SetGravityOn(false);  // gravitational forces
    }
    double TotalNumNodes_tibia = my_mesh_tibia->GetNnodes();
    double TotalNumElements_tibia = my_mesh_tibia->GetNelements();
    for (int ele = 0; ele < TotalNumElements_tibia; ele++) {
        auto element = std::make_shared<ChElementShellANCF>();
        element = std::dynamic_pointer_cast<ChElementShellANCF>(my_mesh_tibia->GetElement(ele));
        // Add a single layers with a fiber angle of 0 degrees.
        element->AddLayer(dz, 0 * CH_C_DEG_TO_RAD, material);
        // Set other element properties
        element->SetAlphaDamp(0.05);   // Structural damping for this element
        element->SetGravityOn(false);  // gravitational forces
    }

    if (addPressure) {
        // First: loads must be added to "load containers",
        // and load containers must be added to your ChSystem
        auto Mloadcontainer_Femur = std::make_shared<ChLoadContainer>();
        // Add constant pressure using ChLoaderPressure (preferred for simple, constant pressure)
        for (int NoElmPre = 0; NoElmPre < TotalNumElements_Femur; NoElmPre++) {
            auto faceload = std::make_shared<ChLoad<ChLoaderPressure>>(
                std::static_pointer_cast<ChElementShellANCF>(my_mesh_femur->GetElement(NoElmPre)));
            faceload->loader.SetPressure(1e6);
            faceload->loader.SetStiff(true);
            faceload->loader.SetIntegrationPoints(2);
            Mloadcontainer_Femur->Add(faceload);
        }
        my_system.Add(Mloadcontainer_Femur);
        auto Mloadcontainer_Tibia = std::make_shared<ChLoadContainer>();
        for (int NoElmPre = 0; NoElmPre < TotalNumElements_tibia; NoElmPre++) {
            auto faceload = std::make_shared<ChLoad<ChLoaderPressure>>(
                std::static_pointer_cast<ChElementShellANCF>(my_mesh_tibia->GetElement(NoElmPre)));
            faceload->loader.SetPressure(1e6);
            faceload->loader.SetStiff(true);
            faceload->loader.SetIntegrationPoints(2);
            Mloadcontainer_Tibia->Add(faceload);
        }
        my_system.Add(Mloadcontainer_Tibia);
    }

    // Switch off mesh class gravity
    my_mesh_femur->SetAutomaticGravity(addGravity);
    my_mesh_tibia->SetAutomaticGravity(addGravity);
    if (addGravity) {
        my_system.Set_G_acc(ChVector<>(0, -9.8, 0));
    } else {
        my_system.Set_G_acc(ChVector<>(0, 0, 0));
    }

    // Create the contact surface(s).
    // In this case it is a ChContactSurfaceMesh, that allows mesh-mesh collsions.
    auto mcontactsurf_tibia = std::make_shared<ChContactSurfaceMesh>();
    auto mcontactsurf_femur = std::make_shared<ChContactSurfaceMesh>();
    my_mesh_tibia->AddContactSurface(mcontactsurf_tibia);
    my_mesh_femur->AddContactSurface(mcontactsurf_femur);
    mcontactsurf_tibia->AddFacesFromBoundary(sphere_swept_thickness);  // do this after my_mesh->AddContactSurface
    mcontactsurf_tibia->SetMaterialSurface(mysurfmaterial);            // use the DEM penalty contacts
    mcontactsurf_femur->AddFacesFromBoundary(sphere_swept_thickness);  // do this after my_mesh->AddContactSurface
    mcontactsurf_femur->SetMaterialSurface(mysurfmaterial);            // use the DEM penalty contacts

    my_system.Add(my_mesh_tibia);
    my_system.Add(my_mesh_femur);

#ifdef USE_IRR

    //    ////////////////////////////////////////
    //    // Options for visualization in irrlicht
    //    ////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////
    //////////////Tibia
    //////////////////////////////////////////////////////////////////
    auto mvisualizemesh_tibia = std::make_shared<ChVisualizationFEAmesh>(*(my_mesh_tibia.get()));
    mvisualizemesh_tibia->SetFEMdataType(ChVisualizationFEAmesh::E_PLOT_NODE_SPEED_NORM);
    mvisualizemesh_tibia->SetColorscaleMinMax(0.0, 5.50);
    mvisualizemesh_tibia->SetSmoothFaces(true);
    my_mesh_tibia->AddAsset(mvisualizemesh_tibia);

    //    auto mvisualizemeshcoll_tibia = std::make_shared<ChVisualizationFEAmesh>(*(my_mesh_tibia.get()));
    //    mvisualizemeshcoll_tibia->SetFEMdataType(ChVisualizationFEAmesh::E_PLOT_CONTACTSURFACES);
    //    mvisualizemeshcoll_tibia->SetWireframe(true);
    //    mvisualizemeshcoll_tibia->SetDefaultMeshColor(ChColor(0.0, 0.0, 0.0));
    //    my_mesh_tibia->AddAsset(mvisualizemeshcoll_tibia);

    //    auto mvisualizemeshbeam_tibia = std::make_shared<ChVisualizationFEAmesh>(*(my_mesh_tibia.get()));
    //    mvisualizemeshbeam_tibia->SetFEMdataType(ChVisualizationFEAmesh::E_PLOT_ELEM_STRAIN_VONMISES);
    //    mvisualizemeshbeam_tibia->SetColorscaleMinMax(0.00, 0.20);
    //    mvisualizemeshbeam_tibia->SetSmoothFaces(true);
    //    my_mesh_tibia->AddAsset(mvisualizemeshbeam_tibia);

    auto mvisualizemeshDef_tibia = std::make_shared<ChVisualizationFEAmesh>(*(my_mesh_tibia.get()));
    mvisualizemeshDef_tibia->SetFEMdataType(ChVisualizationFEAmesh::E_PLOT_ANCF_SECTION_DISPLACEMENT);
    mvisualizemeshDef_tibia->SetColorscaleMinMax(0, 0.001);
    mvisualizemeshDef_tibia->SetSmoothFaces(true);
    my_mesh_tibia->AddAsset(mvisualizemeshDef_tibia);

    auto mvisualizemeshbeamnodes_tibia = std::make_shared<ChVisualizationFEAmesh>(*(my_mesh_tibia.get()));
    mvisualizemeshbeamnodes_tibia->SetFEMglyphType(ChVisualizationFEAmesh::E_GLYPH_NODE_DOT_POS);
    mvisualizemeshbeamnodes_tibia->SetFEMdataType(ChVisualizationFEAmesh::E_PLOT_NONE);
    mvisualizemeshbeamnodes_tibia->SetSymbolsThickness(0.0002);
    my_mesh_tibia->AddAsset(mvisualizemeshbeamnodes_tibia);

    ///////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////
    //////////////Femur
    //////////////////////////////////////////////////////////////////
    auto mvisualizemesh = std::make_shared<ChVisualizationFEAmesh>(*(my_mesh_femur.get()));
    mvisualizemesh->SetFEMdataType(ChVisualizationFEAmesh::E_PLOT_NODE_SPEED_NORM);
    mvisualizemesh->SetColorscaleMinMax(0, 0.5);
    mvisualizemesh->SetSmoothFaces(true);
    my_mesh_femur->AddAsset(mvisualizemesh);

    //    auto mvisualizemeshcoll = std::make_shared<ChVisualizationFEAmesh>(*(my_mesh_femur.get()));
    //    mvisualizemeshcoll->SetFEMdataType(ChVisualizationFEAmesh::E_PLOT_CONTACTSURFACES);
    //    mvisualizemeshcoll->SetWireframe(true);
    //    mvisualizemeshcoll->SetDefaultMeshColor(ChColor(0.0, 0.0, 0.0));
    //    my_mesh_femur->AddAsset(mvisualizemeshcoll);

    auto mvisualizemeshbeam = std::make_shared<ChVisualizationFEAmesh>(*(my_mesh_femur.get()));
    mvisualizemeshbeam->SetFEMdataType(ChVisualizationFEAmesh::E_PLOT_ELEM_STRAIN_VONMISES);
    mvisualizemeshbeam->SetColorscaleMinMax(0.00, 0.200);
    mvisualizemeshbeam->SetSmoothFaces(true);
    my_mesh_femur->AddAsset(mvisualizemeshbeam);

    //    auto mvisualizemeshDef = std::make_shared<ChVisualizationFEAmesh>(*(my_mesh_femur.get()));
    //    mvisualizemeshDef->SetFEMdataType(ChVisualizationFEAmesh::E_PLOT_ANCF_SECTION_DISPLACEMENT);
    //    mvisualizemeshDef->SetColorscaleMinMax(0, 0.001);
    //    mvisualizemeshDef->SetSmoothFaces(true);
    //    my_mesh_femur->AddAsset(mvisualizemeshDef);

    auto mvisualizemeshbeamnodes = std::make_shared<ChVisualizationFEAmesh>(*(my_mesh_femur.get()));
    mvisualizemeshbeamnodes->SetFEMglyphType(ChVisualizationFEAmesh::E_GLYPH_NODE_DOT_POS);
    mvisualizemeshbeamnodes->SetFEMdataType(ChVisualizationFEAmesh::E_PLOT_NONE);
    mvisualizemeshbeamnodes->SetSymbolsThickness(0.0004);
    my_mesh_femur->AddAsset(mvisualizemeshbeamnodes);

    application.AssetBindAll();
    application.AssetUpdateAll();
    application.AddShadowAll();
    application.SetTimestep(time_step);
#endif
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

#ifdef USE_IRR
    application.GetSystem()->Update();
    //    application.SetPaused(true);
    int AccuNoIterations = 0;
    application.SetStepManage(true);
#endif
    //    // Setup solver
    //    my_system.SetLcpSolverType(ChSystem::LCP_ITERATIVE_MINRES);
    //    ChLcpIterativeMINRES* msolver = (ChLcpIterativeMINRES*)my_system.GetLcpSolverSpeed();
    //    msolver->SetDiagonalPreconditioning(true);
    //    my_system.SetIterLCPwarmStarting(true);  // this helps a lot to speedup convergence in this class of
    //    my_system.SetIterLCPmaxItersSpeed(4000000);
    //    my_system.SetTolForce(1e-6);
    //    msolver->SetVerbose(false);
    //
    // INT_HHT or INT_EULER_IMPLICIT
    //    my_system.SetIntegrationType(ChSystem::INT_EULER_IMPLICIT_LINEARIZED);  // fast, less precise

    my_system.SetIntegrationType(ChSystem::INT_HHT);
    auto mystepper = std::dynamic_pointer_cast<ChTimestepperHHT>(my_system.GetTimestepper());
    mystepper->SetAlpha(-0.2);
    mystepper->SetMaxiters(20);
    //    mystepper->SetAbsTolerances(1e-04, 1e-3);  // For ACC
    mystepper->SetAbsTolerances(1e-05, 10);          // For Pos
    mystepper->SetMode(ChTimestepperHHT::POSITION);  // POSITION //ACCELERATION
    mystepper->SetScaling(true);
    mystepper->SetVerbose(true);
    mystepper->SetMaxItersSuccess(3);

#ifndef USE_IRR
    std::vector<std::vector<int>> NodeNeighborElementFemur;
    std::vector<std::vector<int>> NodeNeighborElementTibia;
    string saveAsFemur = "Femur";
    string saveAsTibia = "Tibia";
    writeMesh(my_mesh_femur, saveAsFemur, NodeNeighborElementFemur);
    writeMesh(my_mesh_tibia, saveAsTibia, NodeNeighborElementTibia);
    int step_count = 0;
    // Read Tibia Obj connectivity and positions
    // Also save the connectivity somewhere to use later in the simulation loop
    string saveAsTibiaObj = "TibiaConectivity";
    std::vector<std::vector<double>> vCoor;
    std::vector<std::vector<int>> faces;
    ReadOBJConnectivity(GetChronoDataFile("fea/tibia.obj").c_str(), saveAsTibiaObj, vCoor, faces);

#endif
/////////////////////////////////////////////////////////////////
//    ChVector<> Initial_Pos = Femur->GetPos();
//    MarkerFemur->Impose_Rel_Coord(ChCoordsys<>(-Initial_Pos));

#ifdef USE_IRR
    ChQuaternion<> myRot = Tibia->GetRot();
    ChVector<> myPos = Tibia->GetPos();
    printf("Tibia pos= %f  %f  %f  \n", myPos.x, myPos.y, myPos.z);
    printf("Tibia Rot= %f  %f  %f  %f \n", myRot.e1, myRot.e2, myRot.e3, myRot.e0);

    while (application.GetDevice()->run()) {
        application.BeginScene();
        application.DrawAll();
        application.DoStep();
        ChQuaternion<> myRot = Tibia->GetRot();
        ChVector<> myPos = Tibia->GetPos();

        if (my_system.GetChTime() > 0.01)
            throw ChException("--------------------\n");

        printf("Tibia pos= %f  %f  %f  \n", myPos.x, myPos.y, myPos.z);
        printf("Tibia Rot= %f  %f  %f  %f \n", myRot.e1, myRot.e2, myRot.e3, myRot.e0);

        //        throw ChException("ERROR opening Mesh file: \n");

        //        ManipulateFemur(my_system, Femur, Tibia, Initial_Pos, MarkerFemurIn);

        application.EndScene();
    }
#else
    ChQuaternion<> myRot = Tibia->GetRot();

    printf("Tibia Rot= %f  %f  %f  %f \n", myRot.e1, myRot.e2, myRot.e3, myRot.e0);

    while (my_system.GetChTime() < 2) {
        ChQuaternion<> myRot = Tibia->GetRot();
        printf("Tibia Rot= %f  %f  %f  %f \n", myRot.e1, myRot.e2, myRot.e3, myRot.e0);
        my_system.DoStepDynamics(time_step);
        step_count++;
        std::ofstream output_femur;
        std::ofstream output_tibia;
        std::ofstream output_femur_Rigid;
        std::ofstream output_tibia_Rigid;
        output_femur.open("AC-Data/femur.txt", std::ios::app);
        output_tibia.open("AC-Data/tibia.txt", std::ios::app);
        output_femur_Rigid.open("TimeVPlots/femur_rigid.txt");
        output_tibia_Rigid.open("TimeVPlots/tibia_rigid.txt");
        ////////////////////////////////////////
        ///////////Write to VTK/////////////////
        ////////////////////////////////////////
        if (step_count % write_interval == 0) {
            ////////////////////////////////////////////////////////////////
            // To compute pressure forces
            my_system.GetContactContainer()->ComputeContactForces();
            std::vector<ChVector<>> TibiaNodeFrc;
            std::vector<ChVector<>> FemurNodeFrc;
            TibiaNodeFrc.empty();
            FemurNodeFrc.empty();
            ChVector<> contact_force_total_tibia;
            for (int i = 0; i < my_mesh_tibia->GetNnodes(); i++) {
                auto nodetibia = std::dynamic_pointer_cast<ChNodeFEAxyz>(my_mesh_tibia->GetNode(i));
                ChVector<> contact_force = mcontactsurf_tibia->GetContactForce(&my_system, nodetibia.get());
                TibiaNodeFrc.push_back(contact_force);
                contact_force_total_tibia += contact_force;
            }
            output_tibia << my_system.GetChTime() << " " << contact_force_total_tibia.x << " "
                         << contact_force_total_tibia.y << " " << contact_force_total_tibia.z << "\n";
            output_tibia.close();

            ChVector<> contact_force_total_femur;
            for (int i = 0; i < my_mesh_femur->GetNnodes(); i++) {
                auto nodefemur = std::dynamic_pointer_cast<ChNodeFEAxyz>(my_mesh_femur->GetNode(i));
                ChVector<> contact_force = mcontactsurf_femur->GetContactForce(&my_system, nodefemur.get());
                FemurNodeFrc.push_back(contact_force);
                contact_force_total_femur += contact_force;
            }
            output_femur << my_system.GetChTime() << " " << contact_force_total_femur.x << " "
                         << contact_force_total_femur.y << " " << contact_force_total_femur.z << "\n";
            output_femur.close();
            // End compute pressure forces
            char SaveAsBufferFemur[64];        // The filename buffer.
            char SaveAsBufferTibia[64];        // The filename buffer.
            char SaveAsBufferTibiaObj[64];     // The filename buffer.
            char SaveAsBufferTibiaObjVTK[64];  // The filename buffer.

            snprintf(SaveAsBufferFemur, sizeof(char) * 64, "VTK_Animations/femur.%d.vtk", step_count / write_interval);
            snprintf(SaveAsBufferTibia, sizeof(char) * 64, "VTK_Animations/tibia.%d.vtk", step_count / write_interval);
            snprintf(SaveAsBufferTibiaObjVTK, sizeof(char) * 64, "VTK_Animations/ObjTibia.%d.vtk",
                     step_count / write_interval);

            char MeshFileBufferFemur[32];     // The filename buffer.
            char MeshFileBufferTibia[32];     // The filename buffer.
            char MeshFileBufferTibiaObj[64];  // The filename buffer.

            snprintf(MeshFileBufferFemur, sizeof(char) * 32, "VTK_Animations/%s.vtk", saveAsFemur.c_str());
            snprintf(MeshFileBufferTibia, sizeof(char) * 32, "VTK_Animations/%s.vtk", saveAsTibia.c_str());
            snprintf(MeshFileBufferTibiaObj, sizeof(char) * 64, "VTK_Animations/%s.vtk", saveAsTibiaObj.c_str());

            printf("%s from here\n", MeshFileBufferTibiaObj);
            WriteRigidBodyVTK(Tibia, vCoor, SaveAsBufferTibiaObjVTK, MeshFileBufferTibiaObj);
            writeFrame(my_mesh_femur, SaveAsBufferFemur, MeshFileBufferFemur, NodeNeighborElementFemur, FemurNodeFrc);
            writeFrame(my_mesh_tibia, SaveAsBufferTibia, MeshFileBufferTibia, NodeNeighborElementTibia, TibiaNodeFrc);
        }
    }
#endif

    return 0;
}

void GetDataFile(const char* filename, std::vector<std::vector<double>>& DATA) {
    ifstream inputFile;
    inputFile.open(filename);
    int steps = 0;
    string mline;
    while (!inputFile.eof()) {
        getline(inputFile, mline);
        steps++;
    }
    steps--;
    printf(" lines =%d\n", steps);
    DATA.resize(steps);

    std::fstream fin(filename);
    if (!fin.good())
        throw ChException("ERROR opening Mesh file: " + std::string(filename) + "\n");

    std::string line;
    for (int num_data = 0; num_data < steps; num_data++) {
        getline(fin, line);
        DATA[num_data].resize(14);
        int ntoken = 0;
        string token;
        std::istringstream ss(line);
        while (std::getline(ss, token, ',') && ntoken < 20) {
            std::istringstream stoken(token);
            stoken >> DATA[num_data][ntoken];
            ++ntoken;
        }
    }
    printf("%f %f %f %f", DATA[0][0], DATA[0][1], DATA[0][2], DATA[0][3]);
}

void ReadOBJConnectivity(const char* filename,
                         string SaveAs,
                         std::vector<std::vector<double>>& vCoor,
                         std::vector<std::vector<int>>& faces) {
    ifstream inputFile;
    inputFile.open(filename);
    int numLines = 0;
    string mline;
    while (!inputFile.eof()) {
        getline(inputFile, mline);
        numLines++;
    }
    numLines--;
    printf(" lines =%d\n", numLines);

    std::fstream fin(filename);
    if (!fin.good())
        throw ChException("ERROR opening Mesh file: " + std::string(filename) + "\n");

    std::string line;
    int numF = 0;
    int numV = 0;
    for (int num_data = 0; num_data < numLines; num_data++) {
        getline(fin, line);
        if (line.find("v ") == 0) {
            std::vector<double> ThisLineV;

            ThisLineV.resize(4);
            int ntoken = 0;
            string token;
            std::istringstream ss(line);
            while (std::getline(ss, token, ' ') && ntoken < 4) {
                std::istringstream stoken(token);
                stoken >> ThisLineV[ntoken];
                ++ntoken;
            }
            vCoor.push_back(ThisLineV);
            numV++;
            //            printf("%f %f %f %f\n", vCoor[numV - 1][0], vCoor[numV - 1][1], vCoor[numV - 1][2], vCoor[numV
            //            - 1][3]);
        }
        if (line.find("f ") == 0) {
            std::vector<int> ThisLineF;

            ThisLineF.resize(4);
            int ntoken = 0;
            string token;
            std::istringstream ss(line);
            while (std::getline(ss, token, ' ') && ntoken < 4) {
                std::istringstream stoken(token);
                stoken >> ThisLineF[ntoken];
                ++ntoken;
            }
            faces.push_back(ThisLineF);
            numF++;
        }
    }
    utils::CSV_writer VTK(" ");
    VTK.stream().setf(std::ios::scientific | std::ios::showpos);
    VTK.stream().precision(6);

    VTK << "CELLS " << (unsigned int)numF << (unsigned int)4 * numF << "\n";

    for (int iele = 0; iele < numF; iele++) {
        VTK << "3 " << (unsigned int)faces[iele][1] - 1 << " " << (unsigned int)faces[iele][2] - 1 << " "
            << (unsigned int)faces[iele][3] - 1 << "\n";
    }
    VTK << "\nCELL_TYPES " << numF << "\n";

    for (int iele = 0; iele < numF; iele++) {
        VTK << "5\n";
    }

    if (ChFileutils::MakeDirectory("VTK_Animations") < 0) {
        GetLog() << "Error creating directory VTK_Animations\n";
    }

    char bufferVTK[64];  // The filename buffer.
    snprintf(bufferVTK, sizeof(char) * 32, "VTK_Animations/");
    sprintf(bufferVTK + strlen(bufferVTK), SaveAs.c_str());
    sprintf(bufferVTK + strlen(bufferVTK), ".vtk");
    VTK.write_to_file(bufferVTK);

    int step_count = 0;
}
void WriteRigidBodyVTK(std::shared_ptr<ChBody> Body,
                       std::vector<std::vector<double>>& vCoor,
                       char SaveAsBuffer[64],
                       char ConnectivityFileBuffer[32]) {
    std::ofstream output;
    output.open(SaveAsBuffer, std::ios::app);

    ChVector<> position = Body->GetPos();
    ChMatrix33<> Rotation = Body->GetRot();

    std::vector<ChVector<double>> writeV;
    writeV.resize(vCoor.size());
    output << "# vtk DataFile Version 1.0\nUnstructured Grid Example\nASCII\n\n" << std::endl;
    output << "DATASET UNSTRUCTURED_GRID\nPOINTS " << vCoor.size() << " float\n";
    for (int i = 0; i < vCoor.size(); i++) {
        ChVector<double> thisNode;
        thisNode.x = vCoor[i][1];
        thisNode.y = vCoor[i][2];
        thisNode.z = vCoor[i][3];
        writeV[i] = Rotation * thisNode + position;  // rotate/scale, if needed
        output << writeV[i].x << " " << writeV[i].y << " " << writeV[i].z << "\n";
    }

    std::ifstream CopyFrom(ConnectivityFileBuffer);
    output << CopyFrom.rdbuf();
    printf("%s\n", ConnectivityFileBuffer);
    output.close();
}
void writeMesh(std::shared_ptr<ChMesh> my_mesh, string SaveAs, std::vector<std::vector<int>>& NodeNeighborElement) {
    utils::CSV_writer MESH(" ");
    NodeNeighborElement.resize(my_mesh->GetNnodes());
    MESH.stream().setf(std::ios::scientific | std::ios::showpos);
    MESH.stream().precision(6);
    //    out << my_system.GetChTime() << nodetip->GetPos() << std::endl;
    std::vector<std::shared_ptr<ChNodeFEAbase>> myvector;
    myvector.resize(my_mesh->GetNnodes());
    for (int i = 0; i < my_mesh->GetNnodes(); i++) {
        myvector[i] = std::dynamic_pointer_cast<ChNodeFEAbase>(my_mesh->GetNode(i));
    }
    MESH << "\nCELLS " << my_mesh->GetNelements() << 5 * my_mesh->GetNelements() << "\n";

    for (int iele = 0; iele < my_mesh->GetNelements(); iele++) {
        auto element = (my_mesh->GetElement(iele));
        MESH << "4 ";
        int nodeOrder[] = {0, 1, 2, 3};
        for (int myNodeN = 0; myNodeN < 4; myNodeN++) {
            auto nodeA = (element->GetNodeN(nodeOrder[myNodeN]));
            std::vector<std::shared_ptr<ChNodeFEAbase>>::iterator it;
            it = find(myvector.begin(), myvector.end(), nodeA);
            if (it == myvector.end()) {
                // name not in vector
            } else {
                auto index = std::distance(myvector.begin(), it);
                MESH << (unsigned int)index << " ";
                NodeNeighborElement[index].push_back(iele);
            }
        }
        MESH << "\n";
    }
    MESH << "\nCELL_TYPES " << my_mesh->GetNelements() << "\n";

    for (int iele = 0; iele < my_mesh->GetNelements(); iele++) {
        MESH << "9\n";
    }
    // Create output directory (if it does not already exist).
    if (ChFileutils::MakeDirectory("VTK_Animations") < 0) {
        GetLog() << "Error creating directory VTK_Animations\n";
    }
    char buffer[32];  // The filename buffer.
    snprintf(buffer, sizeof(char) * 32, "VTK_Animations/");
    sprintf(buffer + strlen(buffer), SaveAs.c_str());
    sprintf(buffer + strlen(buffer), ".vtk");
    MESH.write_to_file(buffer);
    int step_count = 0;
}

////////////////////////////////////////
///////////Write to VTK/////////////////
////////////////////////////////////////
void writeFrame(std::shared_ptr<ChMesh> my_mesh,
                char SaveAsBuffer[64],
                char MeshFileBuffer[32],
                std::vector<std::vector<int>> NodeNeighborElement,
                std::vector<ChVector<>> NodeFrc) {
    std::ofstream output;
    output.open(SaveAsBuffer, std::ios::app);

    output << "# vtk DataFile Version 1.0\nUnstructured Grid Example\nASCII\n\n" << std::endl;
    output << "DATASET UNSTRUCTURED_GRID\nPOINTS " << my_mesh->GetNnodes() << " float\n";
    for (int i = 0; i < my_mesh->GetNnodes(); i++) {
        auto node = std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(i));
        output << node->GetPos().x << " " << node->GetPos().y << " " << node->GetPos().z << "\n ";
    }
    std::ifstream CopyFrom(MeshFileBuffer);
    output << CopyFrom.rdbuf();
    output << "\nPOINT_DATA " << my_mesh->GetNnodes() << "\n ";
    output << "SCALARS Deflection float\n";
    output << "LOOKUP_TABLE default\n";
    for (int i = 0; i < my_mesh->GetNnodes(); i++) {
        double areaAve = 0;
        double scalar = 0;
        double myarea = 0;
        double dx, dy;
        for (int j = 0; j < NodeNeighborElement[i].size(); j++) {
            int myelemInx = NodeNeighborElement[i][j];
            std::dynamic_pointer_cast<ChElementShellANCF>(my_mesh->GetElement(myelemInx))->EvaluateDeflection(scalar);
            dx = std::dynamic_pointer_cast<ChElementShellANCF>(my_mesh->GetElement(myelemInx))->GetLengthX();
            dy = std::dynamic_pointer_cast<ChElementShellANCF>(my_mesh->GetElement(myelemInx))->GetLengthY();
            myarea += dx * dy / 4;
            areaAve += scalar * dx * dy / 4;
        }

        output << areaAve / myarea << "\n";
    }
    output << "\nVECTORS Strains float\n";
    for (unsigned int i = 0; i < my_mesh->GetNnodes(); i++) {
        double areaAve1 = 0, areaAve2 = 0, areaAve3 = 0;
        double myarea = 0;
        double dx, dy;
        for (int j = 0; j < NodeNeighborElement[i].size(); j++) {
            int myelemInx = NodeNeighborElement[i][j];
            ChVector<> StrainVector(0);
            StrainVector = std::dynamic_pointer_cast<fea::ChElementShellANCF>(my_mesh->GetElement(myelemInx))
                               ->EvaluateSectionStrains();
            dx = std::dynamic_pointer_cast<fea::ChElementShellANCF>(my_mesh->GetElement(myelemInx))->GetLengthX();
            dy = std::dynamic_pointer_cast<fea::ChElementShellANCF>(my_mesh->GetElement(myelemInx))->GetLengthY();
            myarea += dx * dy / 4;
            areaAve1 += StrainVector.x * dx * dy / 4;
            areaAve2 += StrainVector.y * dx * dy / 4;
            areaAve3 += StrainVector.z * dx * dy / 4;
        }
        output << areaAve1 / myarea << " " << areaAve2 / myarea << " " << areaAve3 / myarea << "\n";
    }
    std::vector<ChVector<>> MyResult;
    output << "\nVECTORS ep12_ratio float\n";
    for (unsigned int i = 0; i < my_mesh->GetNnodes(); i++) {
        double areaAve1 = 0, areaAve2 = 0, areaAve3 = 0;
        double myarea = 0;
        double dx, dy;
        for (int j = 0; j < NodeNeighborElement[i].size(); j++) {
            int myelemInx = NodeNeighborElement[i][j];
            MyResult = std::dynamic_pointer_cast<fea::ChElementShellANCF>(my_mesh->GetElement(myelemInx))
                           ->GetPrincipalStrains();
            dx = std::dynamic_pointer_cast<fea::ChElementShellANCF>(my_mesh->GetElement(myelemInx))->GetLengthX();
            dy = std::dynamic_pointer_cast<fea::ChElementShellANCF>(my_mesh->GetElement(myelemInx))->GetLengthY();
            myarea += dx * dy / 4;
            areaAve1 += MyResult[0].x * dx * dy / 4;
            areaAve2 += MyResult[0].y * dx * dy / 4;
            if (abs(MyResult[0].x) > 1e-3 && abs(MyResult[0].y) > 1e-3) {
                double ratio = abs(areaAve1 / areaAve2);
                if (ratio > 10)
                    ratio = 10;
                if (ratio < 0.1)
                    ratio = 0.1;
                areaAve3 += log10(ratio) * dx * dy / 4;
            } else {
                areaAve3 += 0.0 * dx * dy / 4;
            }
        }
        output << areaAve1 / myarea << " " << areaAve2 / myarea << " " << areaAve3 / myarea << "\n";
    }
    output << "\nVECTORS E_Princ_Dir1 float\n";
    for (unsigned int i = 0; i < my_mesh->GetNnodes(); i++) {
        double areaAve1 = 0, areaAve2 = 0, areaAve3 = 0;
        double myarea = 0;
        double dx, dy;
        for (int j = 0; j < NodeNeighborElement[i].size(); j++) {
            int myelemInx = NodeNeighborElement[i][j];
            MyResult = std::dynamic_pointer_cast<fea::ChElementShellANCF>(my_mesh->GetElement(myelemInx))
                           ->GetPrincipalStrains();
            dx = std::dynamic_pointer_cast<fea::ChElementShellANCF>(my_mesh->GetElement(myelemInx))->GetLengthX();
            dy = std::dynamic_pointer_cast<fea::ChElementShellANCF>(my_mesh->GetElement(myelemInx))->GetLengthY();
            myarea += dx * dy / 4;
            areaAve1 += MyResult[1].x * dx * dy / 4;
            areaAve2 += MyResult[1].y * dx * dy / 4;
            areaAve3 += MyResult[1].z * dx * dy / 4;
        }
        output << areaAve1 / myarea << " " << areaAve2 / myarea << " " << areaAve3 / myarea << "\n";
    }
    output << "\nVECTORS E_Princ_Dir2 float\n";
    for (unsigned int i = 0; i < my_mesh->GetNnodes(); i++) {
        double areaAve1 = 0, areaAve2 = 0, areaAve3 = 0;
        double myarea = 0;
        double dx, dy;
        for (int j = 0; j < NodeNeighborElement[i].size(); j++) {
            int myelemInx = NodeNeighborElement[i][j];
            MyResult = std::dynamic_pointer_cast<fea::ChElementShellANCF>(my_mesh->GetElement(myelemInx))
                           ->GetPrincipalStrains();
            dx = std::dynamic_pointer_cast<fea::ChElementShellANCF>(my_mesh->GetElement(myelemInx))->GetLengthX();
            dy = std::dynamic_pointer_cast<fea::ChElementShellANCF>(my_mesh->GetElement(myelemInx))->GetLengthY();
            myarea += dx * dy / 4;
            areaAve1 += MyResult[2].x * dx * dy / 4;
            areaAve2 += MyResult[2].y * dx * dy / 4;
            areaAve3 += MyResult[2].z * dx * dy / 4;
        }
        output << areaAve1 / myarea << " " << areaAve2 / myarea << " " << areaAve3 / myarea << "\n";
    }
    output << "\nVECTORS WAve_T_dir float\n";
    for (unsigned int i = 0; i < my_mesh->GetNnodes(); i++) {
        double areaAve1 = 0, areaAve2 = 0, areaAve3 = 0;
        double myarea = 0;
        double dx, dy;
        for (int j = 0; j < NodeNeighborElement[i].size(); j++) {
            int myelemInx = NodeNeighborElement[i][j];
            MyResult = std::dynamic_pointer_cast<fea::ChElementShellANCF>(my_mesh->GetElement(myelemInx))
                           ->GetPrincipalStrains();

            std::vector<ChVector<>> MyResult_mag =
                std::dynamic_pointer_cast<fea::ChElementShellANCF>(my_mesh->GetElement(myelemInx))
                    ->GetPrincipalStrains();
            dx = std::dynamic_pointer_cast<fea::ChElementShellANCF>(my_mesh->GetElement(myelemInx))->GetLengthX();
            dy = std::dynamic_pointer_cast<fea::ChElementShellANCF>(my_mesh->GetElement(myelemInx))->GetLengthY();
            myarea += dx * dy / 4;

            if (MyResult_mag[0].y < 0) {
                MyResult_mag[0].y = 0;
            }
            areaAve1 += (MyResult_mag[0].x * MyResult[1].x + MyResult_mag[0].y * MyResult[2].x) * dx * dy / 4;
            areaAve2 += (MyResult_mag[0].x * MyResult[1].y + MyResult_mag[0].y * MyResult[2].y) * dx * dy / 4;
            areaAve3 += (MyResult_mag[0].x * MyResult[1].z + MyResult_mag[0].y * MyResult[2].z) * dx * dy / 4;
        }
        output << areaAve1 / myarea << " " << areaAve2 / myarea << " " << areaAve3 / myarea << "\n";
    }
    output << "\nVECTORS WAve_dir float\n";
    for (unsigned int i = 0; i < my_mesh->GetNnodes(); i++) {
        double areaAve1 = 0, areaAve2 = 0, areaAve3 = 0;
        double myarea = 0;
        double dx, dy;
        for (int j = 0; j < NodeNeighborElement[i].size(); j++) {
            int myelemInx = NodeNeighborElement[i][j];
            MyResult = std::dynamic_pointer_cast<fea::ChElementShellANCF>(my_mesh->GetElement(myelemInx))
                           ->GetPrincipalStrains();

            std::vector<ChVector<>> MyResult_mag =
                std::dynamic_pointer_cast<fea::ChElementShellANCF>(my_mesh->GetElement(myelemInx))
                    ->GetPrincipalStrains();
            dx = std::dynamic_pointer_cast<fea::ChElementShellANCF>(my_mesh->GetElement(myelemInx))->GetLengthX();
            dy = std::dynamic_pointer_cast<fea::ChElementShellANCF>(my_mesh->GetElement(myelemInx))->GetLengthY();
            myarea += dx * dy / 4;

            areaAve1 += (MyResult_mag[0].x * MyResult[1].x + MyResult_mag[0].y * MyResult[2].x) * dx * dy / 4;
            areaAve2 += (MyResult_mag[0].x * MyResult[1].y + MyResult_mag[0].y * MyResult[2].y) * dx * dy / 4;
            areaAve3 += (MyResult_mag[0].x * MyResult[1].z + MyResult_mag[0].y * MyResult[2].z) * dx * dy / 4;
        }
        output << areaAve1 / myarea << " " << areaAve2 / myarea << " " << areaAve3 / myarea << "\n";
    }
    output << "\nVECTORS sigma12_theta float\n";
    for (unsigned int i = 0; i < my_mesh->GetNnodes(); i++) {
        double areaAve1 = 0, areaAve2 = 0, areaAve3 = 0;
        double myarea = 0;
        double dx, dy;
        for (int j = 0; j < NodeNeighborElement[i].size(); j++) {
            int myelemInx = NodeNeighborElement[i][j];
            MyResult = std::dynamic_pointer_cast<fea::ChElementShellANCF>(my_mesh->GetElement(myelemInx))
                           ->GetPrincipalStresses();
            dx = std::dynamic_pointer_cast<fea::ChElementShellANCF>(my_mesh->GetElement(myelemInx))->GetLengthX();
            dy = std::dynamic_pointer_cast<fea::ChElementShellANCF>(my_mesh->GetElement(myelemInx))->GetLengthY();
            myarea += dx * dy / 4;
            areaAve1 += MyResult[0].x * dx * dy / 4;
            areaAve2 += MyResult[0].y * dx * dy / 4;
            areaAve3 += MyResult[0].z * dx * dy / 4;
        }
        output << areaAve1 / myarea << " " << areaAve2 / myarea << " " << areaAve3 / myarea << "\n";
    }
    output << "\nVECTORS S_Princ_Dir1 float\n";
    for (unsigned int i = 0; i < my_mesh->GetNnodes(); i++) {
        double areaAve1 = 0, areaAve2 = 0, areaAve3 = 0;
        double myarea = 0;
        double dx, dy;
        for (int j = 0; j < NodeNeighborElement[i].size(); j++) {
            int myelemInx = NodeNeighborElement[i][j];
            MyResult = std::dynamic_pointer_cast<fea::ChElementShellANCF>(my_mesh->GetElement(myelemInx))
                           ->GetPrincipalStresses();
            dx = std::dynamic_pointer_cast<fea::ChElementShellANCF>(my_mesh->GetElement(myelemInx))->GetLengthX();
            dy = std::dynamic_pointer_cast<fea::ChElementShellANCF>(my_mesh->GetElement(myelemInx))->GetLengthY();
            myarea += dx * dy / 4;
            areaAve1 += MyResult[1].x * dx * dy / 4;
            areaAve2 += MyResult[1].y * dx * dy / 4;
            areaAve3 += MyResult[1].z * dx * dy / 4;
        }
        output << areaAve1 / myarea << " " << areaAve2 / myarea << " " << areaAve3 / myarea << "\n";
    }

    output << "\nVECTORS S_Princ_Dir2 float\n";
    for (unsigned int i = 0; i < my_mesh->GetNnodes(); i++) {
        double areaAve1 = 0, areaAve2 = 0, areaAve3 = 0;
        double myarea = 0;
        double dx, dy;
        for (int j = 0; j < NodeNeighborElement[i].size(); j++) {
            int myelemInx = NodeNeighborElement[i][j];
            MyResult = std::dynamic_pointer_cast<fea::ChElementShellANCF>(my_mesh->GetElement(myelemInx))
                           ->GetPrincipalStresses();
            dx = std::dynamic_pointer_cast<fea::ChElementShellANCF>(my_mesh->GetElement(myelemInx))->GetLengthX();
            dy = std::dynamic_pointer_cast<fea::ChElementShellANCF>(my_mesh->GetElement(myelemInx))->GetLengthY();
            myarea += dx * dy / 4;
            areaAve1 += MyResult[2].x * dx * dy / 4;
            areaAve2 += MyResult[2].y * dx * dy / 4;
            areaAve3 += MyResult[2].z * dx * dy / 4;
        }
        output << areaAve1 / myarea << " " << areaAve2 / myarea << " " << areaAve3 / myarea << "\n";
    }
    output << "\nVECTORS Force float\n";
    for (unsigned int i = 0; i < my_mesh->GetNnodes(); i++) {
        double areaAve1 = 0, areaAve2 = 0, areaAve3 = 0;
        double myarea = 0;
        double dx, dy;
        for (int j = 0; j < NodeNeighborElement[i].size(); j++) {
            int myelemInx = NodeNeighborElement[i][j];
            dx = std::dynamic_pointer_cast<fea::ChElementShellANCF>(my_mesh->GetElement(myelemInx))->GetLengthX();
            dy = std::dynamic_pointer_cast<fea::ChElementShellANCF>(my_mesh->GetElement(myelemInx))->GetLengthY();
            myarea += dx * dy / 4;
            areaAve1 += NodeFrc[i].x * dx * dy / 4;
            areaAve2 += NodeFrc[i].y * dx * dy / 4;
            areaAve3 += NodeFrc[i].z * dx * dy / 4;
        }
        output << areaAve1 / (myarea * myarea) << " " << areaAve2 / (myarea * myarea) << " "
               << areaAve3 / (myarea * myarea) << "\n";
    }

    output.close();
}

void impose_TF_motion(std::vector<std::vector<double>> motionInfo,
                      int angleset,
                      std::shared_ptr<ChLinkLockLock> my_link_TF) {
    class TF_tx : public ChFunction {
      public:
        std::vector<std::vector<double>> myMotion;
        TF_tx(std::vector<std::vector<double>> info) { myMotion = info; }
        ChFunction* new_Duplicate() { return new TF_tx(myMotion); }
        virtual double Get_y(double x) {
            int i = 0;
            int pos = 0;
            while (i < myMotion.size()) {
                if (x > myMotion[i][0] && x > myMotion[i + 1][0]) {
                    i++;
                    continue;
                }
                if (x <= myMotion[i + 1][0])
                    break;
            }

            double t1 = myMotion[i][0];
            double t2 = myMotion[i + 1][0];
            double y1 = myMotion[i][1];
            double y2 = myMotion[i + 1][1];
            double y = y1 + (y2 - y1) / (t2 - t1) * (x - t1);

            return (y);
        }
    };

    class TF_ty : public ChFunction {
      public:
        std::vector<std::vector<double>> myMotion;
        TF_ty(std::vector<std::vector<double>> info) { myMotion = info; }
        ChFunction* new_Duplicate() { return new TF_ty(myMotion); }
        virtual double Get_y(double x) {
            int i = 0;
            int pos = 0;
            while (i < myMotion.size()) {
                if (x > myMotion[i][0] && x > myMotion[i + 1][0]) {
                    i++;
                    continue;
                }
                if (x <= myMotion[i + 1][0])
                    break;
            }

            double t1 = myMotion[i][0];
            double t2 = myMotion[i + 1][0];
            double y1 = myMotion[i][2];
            double y2 = myMotion[i + 1][2];
            double y = y1 + (y2 - y1) / (t2 - t1) * (x - t1);

            //            printf(" outputting %f \n", y);
            return (y);
        }
    };

    class TF_tz : public ChFunction {
      public:
        std::vector<std::vector<double>> myMotion;
        TF_tz(std::vector<std::vector<double>> info) { myMotion = info; }
        ChFunction* new_Duplicate() { return new TF_tz(myMotion); }
        virtual double Get_y(double x) {
            int i = 0;
            int pos = 0;
            while (i < myMotion.size()) {
                if (x > myMotion[i][0] && x > myMotion[i + 1][0]) {
                    i++;
                    continue;
                }
                if (x <= myMotion[i + 1][0])
                    break;
            }

            double t1 = myMotion[i][0];
            double t2 = myMotion[i + 1][0];
            double y1 = myMotion[i][3];
            double y2 = myMotion[i + 1][3];
            double y = y1 + (y2 - y1) / (t2 - t1) * (x - t1);

            return (y);
        }
    };

    class TF_rx : public ChFunction {
      public:
        std::vector<std::vector<double>> myMotion;
        TF_rx(std::vector<std::vector<double>> info) { myMotion = info; }
        ChFunction* new_Duplicate() { return new TF_rx(myMotion); }
        virtual double Get_y(double x) {
            int i = 0;
            int pos = 0;
            while (i < myMotion.size()) {
                if (x > myMotion[i][0] && x > myMotion[i + 1][0]) {
                    i++;
                    continue;
                }
                if (x <= myMotion[i + 1][0])
                    break;
            }

            double t1 = myMotion[i][0];
            double t2 = myMotion[i + 1][0];
            double y1 = myMotion[i][4];
            double y2 = myMotion[i + 1][4];
            double y = y1 + (y2 - y1) / (t2 - t1) * (x - t1);

            return (-y * CH_C_PI / 180);
        }
    };

    class TF_ry : public ChFunction {
      public:
        std::vector<std::vector<double>> myMotion;
        TF_ry(std::vector<std::vector<double>> info) { myMotion = info; }
        ChFunction* new_Duplicate() { return new TF_ry(myMotion); }
        virtual double Get_y(double x) {
            int i = 0;
            int pos = 0;
            while (i < myMotion.size()) {
                if (x > myMotion[i][0] && x > myMotion[i + 1][0]) {
                    i++;
                    continue;
                }
                if (x <= myMotion[i + 1][0])
                    break;
            }

            double t1 = myMotion[i][0];
            double t2 = myMotion[i + 1][0];
            double y1 = myMotion[i][5];
            double y2 = myMotion[i + 1][5];
            double y = y1 + (y2 - y1) / (t2 - t1) * (x - t1);

            return (-y * CH_C_PI / 180);
        }
    };

    class TF_rz : public ChFunction {
      public:
        std::vector<std::vector<double>> myMotion;
        TF_rz(std::vector<std::vector<double>> info) { myMotion = info; }
        ChFunction* new_Duplicate() { return new TF_rz(myMotion); }
        virtual double Get_y(double x) {
            int i = 0;
            int pos = 0;
            while (i < myMotion.size()) {
                if (x > myMotion[i][0] && x > myMotion[i + 1][0]) {
                    i++;
                    continue;
                }
                if (x <= myMotion[i + 1][0])
                    break;
            }

            double t1 = myMotion[i][0];
            double t2 = myMotion[i + 1][0];
            double y1 = myMotion[i][6];
            double y2 = myMotion[i + 1][6];
            double y = y1 + (y2 - y1) / (t2 - t1) * (x - t1);
            return (-y * CH_C_PI / 180);
        }
    };

    TF_tx* TFtx = new TF_tx(motionInfo);
    TF_ty* TFty = new TF_ty(motionInfo);
    TF_tz* TFtz = new TF_tz(motionInfo);
    TF_rx* TFrx = new TF_rx(motionInfo);
    TF_ry* TFry = new TF_ry(motionInfo);
    TF_rz* TFrz = new TF_rz(motionInfo);
    my_link_TF->Set_angleset(angleset);
    my_link_TF->SetMotion_X(TFtx);
    my_link_TF->SetMotion_Y(TFty);
    my_link_TF->SetMotion_Z(TFtz);
    my_link_TF->SetMotion_ang(TFrx);
    my_link_TF->SetMotion_ang2(TFry);
    my_link_TF->SetMotion_ang3(TFrz);

    //    class A_COSX : public ChFunction {
    //      public:
    //        ChFunction* new_Duplicate() { return new A_COSX; }
    //        double A = -0.5;
    //        double w = 2 * 3.1415 / 0.1;
    //        virtual double Get_y(double x) {
    //            double y;
    //            if (x < 0.05)  // This means time not position
    //                y = 0;
    //            else
    //                y = 0.5 * A * (1 - cos(w * (x - 0.05)));
    //            return (y);
    //        }
    //    };
    //
    //    class AXB : public ChFunction {
    //      public:
    //        ChFunction* new_Duplicate() { return new AXB; }
    //        double Final_pos = +0.001;
    //        virtual double Get_y(double x) {
    //            double y;
    //            if (x < 0.05)  // This means time not position
    //                y = (0.24) * x - 0.01;
    //            else
    //                y = Final_pos;
    //            return (y);
    //        }
    //    };
    //
    //    A_COSX* phi_motion = new A_COSX;
    //    AXB* y_motion = new AXB;
    //    my_link_TF->SetMotion_ang(phi_motion);
    //    my_link_TF->SetMotion_Y(y_motion);

    //            ChMatrix33<> RotationXX(0);
    //            RotationXX.SetElement(0, 0, 1);
    //            RotationXX.SetElement(1, 1, cos(D2R * motionInfo[0][4]));
    //            RotationXX.SetElement(1, 2, +sin(D2R * motionInfo[0][4]));
    //            RotationXX.SetElement(2, 1, -sin(D2R * motionInfo[0][4]));
    //            RotationXX.SetElement(2, 2, cos(D2R * motionInfo[0][4]));
    //            ChMatrix33<> RotationYY(0);
    //            RotationYY.SetElement(1, 1, 1);
    //            RotationYY.SetElement(0, 0, cos(D2R * motionInfo[0][5]));
    //            RotationYY.SetElement(0, 2, -sin(D2R * motionInfo[0][5]));
    //            RotationYY.SetElement(2, 0, +sin(D2R * motionInfo[0][5]));
    //            RotationYY.SetElement(2, 2, cos(D2R * motionInfo[0][5]));
    //            ChMatrix33<> RotationZZ(0);
    //            RotationZZ.SetElement(2, 2, 1);
    //            RotationZZ.SetElement(0, 0, cos(D2R * motionInfo[0][6]));
    //            RotationZZ.SetElement(0, 1, +sin(D2R * motionInfo[0][6]));
    //            RotationZZ.SetElement(1, 0, -sin(D2R * motionInfo[0][6]));
    //            RotationZZ.SetElement(1, 1, cos(D2R * motionInfo[0][6]));
    //            ChMatrix33<> MeshRotate2;
    //            MeshRotate2.MatrMultiply(RotationXX, (RotationYY * RotationZZ));
}
//////////////////////////////////////////
/////////////Write to OBJ/////////////////
//////////////////////////////////////////
// void WriteOBJ(std::shared_ptr<ChBody> Body,
//              std::vector<std::vector<double>>& vCoor,
//              char SaveAsBuffer[64],
//              char ConnectivityFileBuffer[32]) {
//    std::ofstream output;
//    output.open(SaveAsBuffer, std::ios::app);
//
//    ChVector<> position = Body->GetPos();
//    ChMatrix33<> Rotation = Body->GetRot();
//
//    std::vector<ChVector<double>> writeV;
//    writeV.resize(vCoor.size());
//
//    for (int i = 0; i < vCoor.size(); i++) {
//        ChVector<double> thisNode;
//        thisNode.x = vCoor[i][1];
//        thisNode.y = vCoor[i][2];
//        thisNode.z = vCoor[i][3];
//
//        writeV[i] = Rotation * thisNode + position;  // rotate/scale, if needed
//        output << "v " << writeV[i].x << " " << writeV[i].y << " " << writeV[i].z << "\n";
//    }
//    std::ifstream CopyFrom(ConnectivityFileBuffer);
//    output << CopyFrom.rdbuf();
//    output.close();
//}
