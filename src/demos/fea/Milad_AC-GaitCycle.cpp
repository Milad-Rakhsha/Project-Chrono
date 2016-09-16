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
#include "chrono/solver/ChSolverMINRES.h"
#include "chrono_mkl/ChSolverMKL.h"
#include "chrono/physics/ChBodyEasy.h"
#include "chrono/physics/ChSystem.h"
#include "chrono/physics/ChSystemDEM.h"
#include "chrono/physics/ChLoadBodyMesh.h"
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
#include "chrono_irrlicht/ChIrrApp.h"
#include "chrono/physics/ChLoaderUV.h"
#include "chrono/physics/ChLoadContainer.h"
#include "chrono_fea/ChVisualizationFEAmesh.h"
#include "chrono_fea/ChContactSurfaceMesh.h"
#include "chrono_fea/ChLoadContactSurfaceMesh.h"
#include "chrono_fea/ChContactSurfaceNodeCloud.h"
#include "chrono_irrlicht/ChBodySceneNode.h"
#include "chrono_irrlicht/ChBodySceneNodeTools.h"
#include "chrono_irrlicht/ChIrrAppInterface.h"
#include "chrono_irrlicht/ChIrrTools.h"
#include "chrono_irrlicht/ChIrrWizard.h"
#include "chrono/motion_functions/ChFunction.h"

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
using namespace rapidjson;

enum ROT_SYS { XYZ, ZXY };
// Only these are supported for now ...
#define NormalSP  // Defines whether spring and dampers will always remain normal to the surface
//#define USE_IRR ;
#define addPressure
#define CalcRegions

bool imposeMotion = true;
bool outputData = true;
bool addGravity = false;
bool addForce = false;
bool showTibia = true;
bool showFemur = true;
bool tibiaCartilage = true;
bool femurCartilage = true;
const double MeterToInch = 0.02539998628;
const int scaleFactor = 1;
const ROT_SYS myRot = ZXY;

int num_threads;
double time_step;
int write_interval;                                           // write every 0.01s
double dz, AlphaDamp;                                         // ANCF SHELL
double K_SPRINGS, C_DAMPERS, L0, L0_t, init_def, p0_f, p0_t;  // Elastic Foundation
double rho, E, nu;                                            // Material Properties
double sphere_swept_thickness, Kn, Kt, Gn, Gt;                // Contact
ChSystemDEM::ContactForceModel ContactModel;
double TibiaMass, femurMass;                                  // Rigid Bodies
ChVector<> TibiaInertia, femurInertia;                        // Rigid Bodies
double AlphaHHT, AbsToleranceHHT, AbsToleranceHHTConstraint;  // HHT Solver
int MaxitersHHT, MaxItersSuccessHHT;                          // HHT Solver
std::string RootFolder, simulationFolder;

void SetParamFromJSON(const std::string& filename,
                      std::string& RootFolder,
                      std::string& simulationFolder,
                      int& num_threads,
                      double& time_step,
                      int& write_interval_time,
                      double& dz,
                      double& AlphaDamp,
                      double& K_SPRINGS,
                      double& C_DAMPERS,
                      double& init_def,
                      double& p0_f,
                      double& p0_t,
                      double& L0,
                      double& L0_t,
                      double& TibiaMass,
                      double& femurMass,
                      ChVector<>& TibiaInertia,
                      ChVector<>& femurInertia,
                      double& rho,
                      double& E,
                      double& nu,
                      ChSystemDEM::ContactForceModel& ContactModel,
                      double& sphere_swept_thickness,
                      double& Kn,
                      double& Kt,
                      double& Gn,
                      double& Gt,
                      double& AlphaHHT,
                      int& MaxitersHHT,
                      double& AbsToleranceHHT,
                      double& AbsToleranceHHTConstraint,
                      int& MaxItersSuccessHHT);

double calcPressureInside(std::shared_ptr<ChMesh> my_mesh, double p_0, double v_0);
double findRegion(ChVector<double> pos);
void ReadOBJConnectivity(const char* filename,
                         string SaveAs,
                         std::vector<std::vector<double>>& vCoor,
                         std::vector<std::vector<int>>& faces);
void WriteRigidBodyVTK(std::shared_ptr<ChBody> Body,
                       std::vector<std::vector<double>>& vCoor,
                       char SaveAsBuffer[256],
                       char ConnectivityFileBuffer[256]);
void GetDataFile(const char* filename, std::vector<std::vector<double>>& DATA);
void impose_TF_motion(std::vector<std::vector<double>> motionInfo,
                      int angleset,
                      std::shared_ptr<ChLinkLockLock> my_link_TF);
void writeMesh(std::shared_ptr<ChMesh> my_mesh, string SaveAs, std::vector<std::vector<int>>& NodeNeighborElement);
void writeFrame(std::shared_ptr<ChMesh> my_mesh,
                char SaveAsBuffer[256],
                char MeshFileBuffer[256],
                std::vector<std::vector<int>> NodeNeighborElement,
                std::vector<ChVector<>> NodeFrc);
class MyLoadSpringDamper : public ChLoadCustomMultiple {
  public:
    MyLoadSpringDamper(std::vector<std::shared_ptr<ChLoadable>>& mloadables,
                       std::shared_ptr<ChBody> AttachBodyInput,
                       double init_sp_def)
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
        init_spring_def = init_sp_def;
    }

    std::shared_ptr<ChBody> AttachBody;    // Body to attach springs and dampers
    std::vector<double> K_sp;              // Stiffness of springs
    std::vector<double> C_dp;              // Damping coefficient of dampers
    std::vector<double> l0;                // Initial, undeformed spring-damper length
    std::vector<ChVector<>> LocalBodyAtt;  // Local Body Attachment
    double init_spring_def;

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
                double for_spdp = K_sp[iii] * (c_length - l0[iii] - init_spring_def) +
                                  C_dp[iii] * dc_length;  // Absolute value of spring-damper force

#ifdef NormalSP

                ChVector<> UnitNormal = Node_Grad.GetNormalized();
                this->load_Q(iii * 6 + 0) = -for_spdp * UnitNormal.x;
                this->load_Q(iii * 6 + 1) = -for_spdp * UnitNormal.y;
                this->load_Q(iii * 6 + 2) = -for_spdp * UnitNormal.z;

                ChVectorDynamic<> Qi(6);  // Vector of generalized forces from spring and damper
                ChVectorDynamic<> Fi(6);  // Vector of applied forces and torques (6 components)
                double detJi = 0;         // Determinant of transformation (Not used)

                Fi(0) = for_spdp * UnitNormal.x;
                Fi(1) = for_spdp * UnitNormal.y;
                Fi(2) = for_spdp * UnitNormal.z;
                Fi(3) = 0.0;
                Fi(4) = 0.0;
                Fi(5) = 0.0;
#else

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
#endif
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
    ChSystemDEM my_system(false, 16000, 500);

    //    std::string inputParameters = "Milad_AC-GaitCycle.json";
    //    pass this to the below function -> inputParameters.c_str()
    SetParamFromJSON(argv[1], RootFolder, simulationFolder, num_threads, time_step, write_interval, dz, AlphaDamp,
                     K_SPRINGS, C_DAMPERS, init_def, p0_f, p0_t, L0, L0_t, TibiaMass, femurMass, TibiaInertia,
                     femurInertia, rho, E, nu, ContactModel, sphere_swept_thickness, Kn, Kt, Gn, Gt, AlphaHHT,
                     MaxitersHHT, AbsToleranceHHT, AbsToleranceHHTConstraint, MaxItersSuccessHHT);

    printf("time_step= %f, write_interval_time=%d, num_threads=%d\n", time_step, write_interval, num_threads);
    printf("AlphaDamp= %f, dz=%f\n", AlphaDamp, dz);
    printf("E= %f, rho=%f, nu=%f\n", E, rho, nu);
    printf("K_SPRINGS= %f, C_DAMPERS=%f, L0=%f\n", K_SPRINGS, C_DAMPERS, L0);
    printf("Kn= %f, Kt=%f, Gn=%f, Gt=%f, sphere_swept_thickness=%f\n", Kn, Kt, Gn, Gt, sphere_swept_thickness);
    printf("AlphaHHT= %f, AbsToleranceHHT=%f, AbsToleranceHHTConstraint=%f, MaxItersSuccessHHT=%d, MaxitersHHT=%d\n",
           AlphaHHT, AbsToleranceHHT, AbsToleranceHHTConstraint, MaxItersSuccessHHT, MaxitersHHT);

    if (ChFileutils::MakeDirectory(RootFolder.c_str()) < 0) {
        std::cout << "Error creating directory " << RootFolder << std::endl;
        return 1;
    }

    //    remove((RootFolder + simulationFolder).c_str());
    //    system("rm -r  (RootFolder + simulationFolder).c_str()");

    if (ChFileutils::MakeDirectory((RootFolder + simulationFolder).c_str()) < 0) {
        std::cout << "Error creating directory " << RootFolder + simulationFolder << std::endl;
        return 1;
    }

    std::ofstream SaveJASON;
    cout << RootFolder + simulationFolder + "/" + simulationFolder + ".json" << endl;
    SaveJASON.open((RootFolder + simulationFolder + "/" + simulationFolder + ".json").c_str());
    std::ifstream CopyFrom(argv[1]);
    SaveJASON << CopyFrom.rdbuf();
    SaveJASON.close();

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
    //    double sphere_swept_thickness = dz * 0.1;
    //
    //    double rho = 1000 * 0.005 / dz;  ///< material density
    //    double E = 40e7;                 ///< Young's modulus
    //    double nu = 0.3;                 ///< Poisson ratio
    // Create the surface material, containing information
    // about friction etc.
    // It is a DEM-p (penalty) material that we will assign to
    // all surfaces that might generate contacts.
    my_system.SetContactForceModel(ChSystemDEM::Hooke);
    auto mysurfmaterial = std::make_shared<ChMaterialSurfaceDEM>();

    mysurfmaterial->SetKn(Kn);
    mysurfmaterial->SetKt(Kt);
    mysurfmaterial->SetGn(Gn);
    mysurfmaterial->SetGt(Gt);

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
    if (!imposeMotion) {
        Tibia->SetBodyFixed(true);
    }

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

    //    my_system.AddLink(my_link_TF);

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
            if (imposeMotion)
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
            if (imposeMotion)
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
            ChMeshFileLoader::ANCFShellFromGMFFile(my_mesh_femur, GetChronoDataFile("fea/FemurVeryFine.mesh").c_str(),
                                                   material, NODE_AVE_AREA_f, BC_NODES, Center_Femur, rot_transform,
                                                   MeterToInch, false, false);
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
            auto OneLoadSpringDamperFemur = std::make_shared<MyLoadSpringDamper>(NodeList, Femur, init_def);
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
    int tibia1NumNodes;
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

        tibia1NumNodes = my_mesh_tibia->GetNnodes();
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
            auto OneLoadSpringDamperTibia = std::make_shared<MyLoadSpringDamper>(NodeList, Tibia, init_def);
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
        element->SetAlphaDamp(AlphaDamp);  // Structural damping for this element
        element->SetGravityOn(false);      // gravitational forces
    }
    double TotalNumNodes_tibia = my_mesh_tibia->GetNnodes();
    double TotalNumElements_tibia = my_mesh_tibia->GetNelements();
    for (int ele = 0; ele < TotalNumElements_tibia; ele++) {
        auto element = std::make_shared<ChElementShellANCF>();
        element = std::dynamic_pointer_cast<ChElementShellANCF>(my_mesh_tibia->GetElement(ele));
        // Add a single layers with a fiber angle of 0 degrees.
        element->AddLayer(dz, 0 * CH_C_DEG_TO_RAD, material);
        // Set other element properties
        element->SetAlphaDamp(AlphaDamp);  // Structural damping for this element
        element->SetGravityOn(false);      // gravitational forces
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

#ifdef addPressure
    std::vector<std::shared_ptr<ChLoad<ChLoaderPressure>>> faceload_fumer;
    std::vector<std::shared_ptr<ChLoad<ChLoaderPressure>>> faceload_tibia;

    // First: loads must be added to "load containers",
    // and load containers must be added to your ChSystem
    auto Mloadcontainer_Femur = std::make_shared<ChLoadContainer>();
    // Add constant pressure using ChLoaderPressure (preferred for simple, constant pressure)
    for (int NoElmPre = 0; NoElmPre < TotalNumElements_Femur; NoElmPre++) {
        auto faceload = std::make_shared<ChLoad<ChLoaderPressure>>(
            std::static_pointer_cast<ChElementShellANCF>(my_mesh_femur->GetElement(NoElmPre)));
        faceload->loader.SetPressure(p0_f);
        faceload->loader.SetStiff(true);
        faceload->loader.SetIntegrationPoints(2);
        faceload_fumer.push_back(faceload);
        Mloadcontainer_Femur->Add(faceload);
    }
    my_system.Add(Mloadcontainer_Femur);
    auto Mloadcontainer_Tibia = std::make_shared<ChLoadContainer>();
    for (int NoElmPre = 0; NoElmPre < TotalNumElements_tibia; NoElmPre++) {
        auto faceload = std::make_shared<ChLoad<ChLoaderPressure>>(
            std::static_pointer_cast<ChElementShellANCF>(my_mesh_tibia->GetElement(NoElmPre)));
        faceload->loader.SetPressure(p0_t);
        faceload->loader.SetStiff(true);
        faceload->loader.SetIntegrationPoints(2);
        faceload_tibia.push_back(faceload);
        Mloadcontainer_Tibia->Add(faceload);
    }
    my_system.Add(Mloadcontainer_Tibia);
#endif

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

    ChSolverMKL* mkl_solver_stab = nullptr;
    ChSolverMKL* mkl_solver_speed = nullptr;
    mkl_solver_stab = new ChSolverMKL;
    mkl_solver_speed = new ChSolverMKL;
    my_system.ChangeSolverStab(mkl_solver_stab);
    my_system.ChangeSolverSpeed(mkl_solver_speed);
    mkl_solver_speed->SetSparsityPatternLock(true);
    mkl_solver_stab->SetSparsityPatternLock(true);

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

    // Set up integrator
    my_system.SetIntegrationType(ChSystem::INT_HHT);
    auto mystepper = std::static_pointer_cast<ChTimestepperHHT>(my_system.GetTimestepper());
    mystepper->SetAlpha(AlphaHHT);
    mystepper->SetMaxiters(MaxitersHHT);
    //    mystepper->SetAbsTolerances(1e-04, 1e-3);  // For ACC
    mystepper->SetAbsTolerances(AbsToleranceHHT, AbsToleranceHHTConstraint);  // For Pos
    mystepper->SetMode(ChTimestepperHHT::POSITION);
    mystepper->SetStepControl(true);
    mystepper->SetModifiedNewton(true);
    mystepper->SetScaling(true);
    mystepper->SetVerbose(true);
    mystepper->SetMaxItersSuccess(MaxItersSuccessHHT);

#ifndef USE_IRR

    std::string VTKFolder, plots2D_Folder;
    VTKFolder = RootFolder + simulationFolder + "/VTK/";
    plots2D_Folder = RootFolder + simulationFolder + "/2DPlots/";

    printf(" VTKFolder IS %s \n\n\n", VTKFolder.c_str());

    if (ChFileutils::MakeDirectory(VTKFolder.c_str()) < 0) {
        std::cout << "Error creating directory " << VTKFolder << std::endl;
        return 1;
    }
    if (ChFileutils::MakeDirectory(plots2D_Folder.c_str()) < 0) {
        std::cout << "Error creating directory " << plots2D_Folder << std::endl;
        return 1;
    }

    std::ofstream plotForces;
    plotForces.open((plots2D_Folder + "PlotForces.p").c_str());
    std::ifstream CopyFromGnuplot(GetChronoDataFile("fea/ShortCodes/PlotForces-GaitCycle.p").c_str());
    plotForces << CopyFromGnuplot.rdbuf();
    plotForces.close();

    std::vector<std::vector<int>> NodeNeighborElementFemur;
    std::vector<std::vector<int>> NodeNeighborElementTibia;
    string saveAsFemur = VTKFolder + "Femur.vtk";
    string saveAsTibia = VTKFolder + "Tibia.vtk";
    writeMesh(my_mesh_femur, saveAsFemur, NodeNeighborElementFemur);
    writeMesh(my_mesh_tibia, saveAsTibia, NodeNeighborElementTibia);
    int step_count = 0;
    // Read Tibia Obj connectivity and positions
    // Also save the connectivity somewhere to use later in the simulation loop
    string saveAsTibiaObj = VTKFolder + "TibiaConectivity.vtk";
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

        if (my_system.GetChTime() > 2)
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
#ifdef addPressure
        double h_0 = 0.003;
        double p_femur = calcPressureInside(my_mesh_femur, p0_f, h_0);
        printf("The pressure ratio is %f\n\n", p_femur / p0_f);
        for (int i = 0; i < faceload_fumer.size(); i++) {
            faceload_fumer[i]->loader.SetPressure(p_femur);
        }

        for (int i = 0; i < faceload_tibia.size(); i++) {
            faceload_tibia[i]->loader.SetPressure(p0_t);
        }

#endif
        my_system.DoStepDynamics(time_step);
        step_count++;
        std::ofstream output_femur;
        std::ofstream output_tibia_1;
        std::ofstream output_tibia_2;
        std::ofstream output_femur_Rigid;
        std::ofstream output_tibia_Rigid;
        output_femur.open(plots2D_Folder + "/femur.txt", std::ios::app);
        output_tibia_1.open(plots2D_Folder + "/tibia1.txt", std::ios::app);
        output_tibia_2.open(plots2D_Folder + "/tibia2.txt", std::ios::app);

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
            ChVector<> contact_force_total_tibia1;
            // Only store the lateral forces for tibia;
            // The medial force is the femur force + this force
            for (int i = 0; i < tibia1NumNodes; i++) {
                auto nodetibia = std::dynamic_pointer_cast<ChNodeFEAxyz>(my_mesh_tibia->GetNode(i));
                ChVector<> contact_force = mcontactsurf_tibia->GetContactForce(&my_system, nodetibia.get());
                TibiaNodeFrc.push_back(contact_force);
                contact_force_total_tibia1 += contact_force;
            }
            output_tibia_1 << my_system.GetChTime() << " " << contact_force_total_tibia1.x << " "
                           << contact_force_total_tibia1.y << " " << contact_force_total_tibia1.z << "\n";

            output_tibia_1.close();

            ChVector<> contact_force_total_tibia2;
            // Only store the lateral forces for tibia;
            // The medial force is the femur force + this force
            for (int i = tibia1NumNodes; i < my_mesh_tibia->GetNnodes(); i++) {
                auto nodetibia = std::dynamic_pointer_cast<ChNodeFEAxyz>(my_mesh_tibia->GetNode(i));
                ChVector<> contact_force = mcontactsurf_tibia->GetContactForce(&my_system, nodetibia.get());
                TibiaNodeFrc.push_back(contact_force);
                contact_force_total_tibia2 += contact_force;
            }
            output_tibia_2 << my_system.GetChTime() << " " << contact_force_total_tibia2.x << " "
                           << contact_force_total_tibia2.y << " " << contact_force_total_tibia2.z << "\n";

            output_tibia_2.close();

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
            char SaveAsBufferFemur[256];        // The filename buffer.
            char SaveAsBufferTibia[256];        // The filename buffer.
            char SaveAsBufferTibiaObj[256];     // The filename buffer.
            char SaveAsBufferTibiaObjVTK[256];  // The filename buffer.

            snprintf(SaveAsBufferFemur, sizeof(char) * 256, (VTKFolder + "/femur.%d.vtk").c_str(),
                     step_count / write_interval);
            snprintf(SaveAsBufferTibia, sizeof(char) * 256, (VTKFolder + "/tibia.%d.vtk").c_str(),
                     step_count / write_interval);
            snprintf(SaveAsBufferTibiaObjVTK, sizeof(char) * 256, (VTKFolder + "/ObjTibia.%d.vtk").c_str(),
                     step_count / write_interval);

            char MeshFileBufferFemur[256];     // The filename buffer.
            char MeshFileBufferTibia[256];     // The filename buffer.
            char MeshFileBufferTibiaObj[256];  // The filename buffer.

            snprintf(MeshFileBufferFemur, sizeof(char) * 256, ("%s"), saveAsFemur.c_str());
            snprintf(MeshFileBufferTibia, sizeof(char) * 256, ("%s"), saveAsTibia.c_str());
            snprintf(MeshFileBufferTibiaObj, sizeof(char) * 256, ("%s"), saveAsTibiaObj.c_str());

            printf("%s from here\n", MeshFileBufferFemur);
            WriteRigidBodyVTK(Tibia, vCoor, SaveAsBufferTibiaObjVTK, MeshFileBufferTibiaObj);
            writeFrame(my_mesh_femur, SaveAsBufferFemur, MeshFileBufferFemur, NodeNeighborElementFemur, FemurNodeFrc);
            writeFrame(my_mesh_tibia, SaveAsBufferTibia, MeshFileBufferTibia, NodeNeighborElementTibia, TibiaNodeFrc);
        }
    }
#endif

    return 0;
}
//======================================================================================================================================
//======================================================================================================================================
//======================================================================================================================================
//======================================================================================================================================
//======================================================================================================================================
//======================================================================================================================================
//======================================================================================================================================
//======================================================================================================================================
//======================================================================================================================================
//======================================================================================================================================
//======================================================================================================================================
//======================================================================================================================================

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

    VTK.write_to_file(SaveAs);

    int step_count = 0;
}

void WriteRigidBodyVTK(std::shared_ptr<ChBody> Body,
                       std::vector<std::vector<double>>& vCoor,
                       char SaveAsBuffer[256],
                       char ConnectivityFileBuffer[256]) {
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
    MESH << "\nPOINT_DATA " << my_mesh->GetNnodes() << "\n ";
    MESH << "SCALARS Region float\n";
    MESH << "LOOKUP_TABLE default\n";
    for (int i = 0; i < my_mesh->GetNnodes(); i++) {
        auto node = std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(i));
        MESH << (uint)(findRegion(node->GetPos())) << "\n";
    }

    MESH.write_to_file(SaveAs);
    int step_count = 0;
}

double calcPressureInside(std::shared_ptr<ChMesh> my_mesh, double p_1, double h_0) {
    double Total_dV = 0;
    double v1 = 0;
    for (int i = 0; i < my_mesh->GetNelements(); i++) {
        double areaAve = 0;
        double dz = 0;
        double myarea = 0;
        double dx, dy;
        std::dynamic_pointer_cast<ChElementShellANCF>(my_mesh->GetElement(i))->EvaluateDeflection(dz);
        dx = std::dynamic_pointer_cast<ChElementShellANCF>(my_mesh->GetElement(i))->GetLengthX();
        dy = std::dynamic_pointer_cast<ChElementShellANCF>(my_mesh->GetElement(i))->GetLengthY();
        Total_dV += dx * dy * dz;
        v1 += dx * dy * h_0;
    }

    double v2 = -Total_dV + v1;
    return (p_1 * std::pow(v1 / v2, 5));
}

////////////////////////////////////////
///////////Write to VTK/////////////////
////////////////////////////////////////
void writeFrame(std::shared_ptr<ChMesh> my_mesh,
                char SaveAsBuffer[256],
                char MeshFileBuffer[256],
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
    //    output << "\nPOINT_DATA " << my_mesh->GetNnodes() << "\n ";
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
            myarea += dx * dy / NodeNeighborElement[i].size();
            areaAve += scalar * dx * dy / NodeNeighborElement[i].size();
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
            myarea += dx * dy / NodeNeighborElement[i].size();
            areaAve1 += StrainVector.x * dx * dy / NodeNeighborElement[i].size();
            areaAve2 += StrainVector.y * dx * dy / NodeNeighborElement[i].size();
            areaAve3 += StrainVector.z * dx * dy / NodeNeighborElement[i].size();
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
            myarea += dx * dy / NodeNeighborElement[i].size();
            areaAve1 += MyResult[0].x * dx * dy / NodeNeighborElement[i].size();
            areaAve2 += MyResult[0].y * dx * dy / NodeNeighborElement[i].size();
            if (abs(MyResult[0].x) > 1e-3 && abs(MyResult[0].y) > 1e-3) {
                double ratio = abs(areaAve1 / areaAve2);
                if (ratio > 10)
                    ratio = 10;
                if (ratio < 0.1)
                    ratio = 0.1;
                areaAve3 += log10(ratio) * dx * dy / NodeNeighborElement[i].size();
            } else {
                areaAve3 += 0.0 * dx * dy / NodeNeighborElement[i].size();
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
            myarea += dx * dy / NodeNeighborElement[i].size();
            MyResult[1] = MyResult[1] / MyResult[1].Length();
            if (MyResult[1].x * areaAve1, MyResult[1].y * areaAve2, MyResult[1].z * areaAve3 < 0) {
                MyResult[1] = MyResult[1] * -1;
            }
            areaAve1 += MyResult[1].x * dx * dy / NodeNeighborElement[i].size();
            areaAve2 += MyResult[1].y * dx * dy / NodeNeighborElement[i].size();
            areaAve3 += MyResult[1].z * dx * dy / NodeNeighborElement[i].size();
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
            myarea += dx * dy / NodeNeighborElement[i].size();
            MyResult[2] = MyResult[2] / MyResult[2].Length();
            if (MyResult[2].x * areaAve1, MyResult[2].y * areaAve2, MyResult[2].z * areaAve3 < 0) {
                MyResult[2] = MyResult[2] * -1;
            }
            areaAve1 += MyResult[2].x * dx * dy / NodeNeighborElement[i].size();
            areaAve2 += MyResult[2].y * dx * dy / NodeNeighborElement[i].size();
            areaAve3 += MyResult[2].z * dx * dy / NodeNeighborElement[i].size();
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
            myarea += dx * dy / NodeNeighborElement[i].size();

            if (MyResult_mag[0].y < 0) {
                MyResult_mag[0].y = 0;
            }
            areaAve1 += (MyResult_mag[0].x * MyResult[1].x + MyResult_mag[0].y * MyResult[2].x) * dx * dy /
                        NodeNeighborElement[i].size();
            areaAve2 += (MyResult_mag[0].x * MyResult[1].y + MyResult_mag[0].y * MyResult[2].y) * dx * dy /
                        NodeNeighborElement[i].size();
            areaAve3 += (MyResult_mag[0].x * MyResult[1].z + MyResult_mag[0].y * MyResult[2].z) * dx * dy /
                        NodeNeighborElement[i].size();
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
            myarea += dx * dy / NodeNeighborElement[i].size();

            areaAve1 += (MyResult_mag[0].x * MyResult[1].x + MyResult_mag[0].y * MyResult[2].x) * dx * dy /
                        NodeNeighborElement[i].size();
            areaAve2 += (MyResult_mag[0].x * MyResult[1].y + MyResult_mag[0].y * MyResult[2].y) * dx * dy /
                        NodeNeighborElement[i].size();
            areaAve3 += (MyResult_mag[0].x * MyResult[1].z + MyResult_mag[0].y * MyResult[2].z) * dx * dy /
                        NodeNeighborElement[i].size();
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
            myarea += dx * dy / NodeNeighborElement[i].size();
            areaAve1 += MyResult[0].x * dx * dy / NodeNeighborElement[i].size();
            areaAve2 += MyResult[0].y * dx * dy / NodeNeighborElement[i].size();
            areaAve3 += MyResult[0].z * dx * dy / NodeNeighborElement[i].size();
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
            myarea += dx * dy / NodeNeighborElement[i].size();
            MyResult[1] = MyResult[1] / MyResult[1].Length();

            if (MyResult[1].x * areaAve1, MyResult[1].y * areaAve2, MyResult[1].z * areaAve3 < 0) {
                MyResult[1] = MyResult[1] * -1;
            }
            areaAve1 += MyResult[1].x * dx * dy / NodeNeighborElement[i].size();
            areaAve2 += MyResult[1].y * dx * dy / NodeNeighborElement[i].size();
            areaAve3 += MyResult[1].z * dx * dy / NodeNeighborElement[i].size();
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
            myarea += dx * dy / NodeNeighborElement[i].size();
            MyResult[2] = MyResult[2] / MyResult[2].Length();

            if (MyResult[2].x * areaAve1, MyResult[2].y * areaAve2, MyResult[2].z * areaAve3 < 0) {
                MyResult[2] = MyResult[2] * -1;
            }
            areaAve1 += MyResult[2].x * dx * dy / NodeNeighborElement[i].size();
            areaAve2 += MyResult[2].y * dx * dy / NodeNeighborElement[i].size();
            areaAve3 += MyResult[2].z * dx * dy / NodeNeighborElement[i].size();
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
            myarea += dx * dy / NodeNeighborElement[i].size();
            areaAve1 += NodeFrc[i].x * dx * dy / NodeNeighborElement[i].size();
            areaAve2 += NodeFrc[i].y * dx * dy / NodeNeighborElement[i].size();
            areaAve3 += NodeFrc[i].z * dx * dy / NodeNeighborElement[i].size();
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
        TF_tx(std::vector<std::vector<double>> motion) {
            myMotion = motion;
            printf("it is called\n");
        }
        virtual TF_tx* Clone() const override { return new TF_tx(myMotion); }
        virtual double Get_y(double x) const override {
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
        TF_ty(std::vector<std::vector<double>> motion) { myMotion = motion; }
        virtual TF_ty* Clone() const override { return new TF_ty(myMotion); }
        virtual double Get_y(double x) const override {
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
            return (y);
        }
    };

    class TF_tz : public ChFunction {
      public:
        std::vector<std::vector<double>> myMotion;
        TF_tz(std::vector<std::vector<double>> motion) { myMotion = motion; }

        virtual TF_tz* Clone() const override { return new TF_tz(myMotion); }
        virtual double Get_y(double x) const override {
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
        TF_rx(std::vector<std::vector<double>> motion) { myMotion = motion; }

        virtual TF_rx* Clone() const override { return new TF_rx(myMotion); }

        virtual double Get_y(double x) const override {
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
        TF_ry(std::vector<std::vector<double>> motion) { myMotion = motion; }

        virtual TF_ry* Clone() const override { return new TF_ry(myMotion); }

        virtual double Get_y(double x) const override {
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
        TF_rz(std::vector<std::vector<double>> motion) { myMotion = motion; }

        virtual TF_rz* Clone() const override { return new TF_rz(myMotion); }

        virtual double Get_y(double x) const override {
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

// double time_step = 0.0001;
// int scaleFactor = 1;
// double dz = 0.001;
// const double K_SPRINGS = 70e8;
// const double C_DAMPERS = 70e3;
// double MeterToInch = 0.02539998628;
// double L0 = 0.01;  // Initial length
// double L0_t = 0.01;
// int write_interval = 0.005 / time_step;  // write every 0.01s
double findRegion(ChVector<double> pos) {
    ChVector<> BoxSize;
    ChVector<> Boxcenter;
    for (int i = 1; i <= 36; i++) {
        switch (i) {
            case 1:
                BoxSize = ChVector<>(0.015, 0.023, 0.01);
                Boxcenter = ChVector<>(-0.016, 0.008, -0.028);
                break;
            case 2:
                BoxSize = ChVector<>(0.015, 0.023, 0.01);
                Boxcenter = ChVector<>(-0.016, 0.008, -0.018);
                break;
            case 3:
                BoxSize = ChVector<>(0.015, 0.023, 0.01);
                Boxcenter = ChVector<>(-0.016, 0.008, -0.008);
                break;

            case 4:
                BoxSize = ChVector<>(0.018, 0.023, 0.01);
                Boxcenter = ChVector<>(-0.016, -0.015, -0.028);
                break;
            case 5:
                BoxSize = ChVector<>(0.018, 0.023, 0.01);
                Boxcenter = ChVector<>(-0.016, -0.015, -0.018);
                break;
            case 6:
                BoxSize = ChVector<>(0.018, 0.023, 0.01);
                Boxcenter = ChVector<>(-0.016, -0.015, -0.008);
                break;

            case 7:
                BoxSize = ChVector<>(0.018, 0.023, 0.01);
                Boxcenter = ChVector<>(0.002, -0.015, -0.028);
                break;
            case 8:
                BoxSize = ChVector<>(0.018, 0.023, 0.01);
                Boxcenter = ChVector<>(0.002, -0.015, -0.018);
                break;
            case 9:
                BoxSize = ChVector<>(0.018, 0.023, 0.01);
                Boxcenter = ChVector<>(0.002, -0.015, -0.008);
                break;

            case 10:
                BoxSize = ChVector<>(0.016, 0.023, 0.012);
                Boxcenter = ChVector<>(0.019, -0.023, -0.025);
                break;
            case 11:
                BoxSize = ChVector<>(0.016, 0.023, 0.010);
                Boxcenter = ChVector<>(0.019, -0.015, -0.014);
                break;
            case 12:
                BoxSize = ChVector<>(0.016, 0.023, 0.010);
                Boxcenter = ChVector<>(0.019, -0.015, -0.004);
                break;

            case 13:
                BoxSize = ChVector<>(0.016, 0.023, 0.010);
                Boxcenter = ChVector<>(0.035, -0.01, -0.017);
                break;
            case 14:
                BoxSize = ChVector<>(0.016, 0.023, 0.008);
                Boxcenter = ChVector<>(0.035, -0.01, -0.008);
                break;
            case 15:
                BoxSize = ChVector<>(0.016, 0.023, 0.008);
                Boxcenter = ChVector<>(0.035, -0.01, -0.001);
                break;

            case 16:
                BoxSize = ChVector<>(0.016, 0.023, 0.010);
                Boxcenter = ChVector<>(0.035, 0.013, -0.017);
                break;
            case 17:
                BoxSize = ChVector<>(0.016, 0.023, 0.008);
                Boxcenter = ChVector<>(0.035, 0.013, -0.008);
                break;
            case 18:
                BoxSize = ChVector<>(0.016, 0.025, 0.006);
                Boxcenter = ChVector<>(0.035, 0.014, -0.001);
                break;
            case 19:
                BoxSize = ChVector<>(0.015, 0.023, 0.01);
                Boxcenter = ChVector<>(-0.016, 0.008, 0.03);
                break;
            case 20:
                BoxSize = ChVector<>(0.015, 0.023, 0.01);
                Boxcenter = ChVector<>(-0.016, 0.008, 0.02);
                break;
            case 21:
                BoxSize = ChVector<>(0.015, 0.023, 0.01);
                Boxcenter = ChVector<>(-0.016, 0.008, 0.01);
                break;

            case 22:
                BoxSize = ChVector<>(0.018, 0.023, 0.01);
                Boxcenter = ChVector<>(-0.016, -0.015, 0.03);
                break;
            case 23:
                BoxSize = ChVector<>(0.018, 0.023, 0.01);
                Boxcenter = ChVector<>(-0.016, -0.015, 0.02);
                break;
            case 24:
                BoxSize = ChVector<>(0.018, 0.023, 0.01);
                Boxcenter = ChVector<>(-0.016, -0.015, 0.01);
                break;

            case 25:
                BoxSize = ChVector<>(0.018, 0.023, 0.01);
                Boxcenter = ChVector<>(0.002, -0.015, 0.03);
                break;
            case 26:
                BoxSize = ChVector<>(0.018, 0.023, 0.01);
                Boxcenter = ChVector<>(0.002, -0.015, 0.02);
                break;
            case 27:
                BoxSize = ChVector<>(0.018, 0.023, 0.012);
                Boxcenter = ChVector<>(0.002, -0.015, 0.009);
                break;

            case 28:
                BoxSize = ChVector<>(0.016, 0.020, 0.012);
                Boxcenter = ChVector<>(0.019, -0.016, 0.028);
                break;
            case 29:
                BoxSize = ChVector<>(0.016, 0.023, 0.010);
                Boxcenter = ChVector<>(0.019, -0.015, 0.017);
                break;
            case 30:
                BoxSize = ChVector<>(0.016, 0.023, 0.011);
                Boxcenter = ChVector<>(0.019, -0.015, 0.0065);
                break;

            case 31:
                BoxSize = ChVector<>(0.02, 0.023, 0.010);
                Boxcenter = ChVector<>(0.037, -0.01, 0.026);
                break;
            case 32:
                BoxSize = ChVector<>(0.02, 0.023, 0.01);
                Boxcenter = ChVector<>(0.037, -0.01, 0.0155);
                break;
            case 33:
                BoxSize = ChVector<>(0.016, 0.023, 0.0085);
                Boxcenter = ChVector<>(0.035, -0.01, 0.006);
                break;
            case 34:
                BoxSize = ChVector<>(0.016, 0.023, 0.012);
                Boxcenter = ChVector<>(0.038, 0.013, 0.026);
                break;
            case 35:
                BoxSize = ChVector<>(0.016, 0.023, 0.01);
                Boxcenter = ChVector<>(0.038, 0.013, 0.015);
                break;
            case 36:
                BoxSize = ChVector<>(0.016, 0.025, 0.008);
                Boxcenter = ChVector<>(0.038, 0.014, 0.006);
                break;
        }
        ChVector<> BoxMin = Boxcenter - BoxSize / 2;
        ChVector<> BoxMax = Boxcenter + BoxSize / 2;

        if (pos >= BoxMin && pos <= BoxMax)
            return i;
    }

    return 0;
}
void SetParamFromJSON(const std::string& filename,
                      std::string& RootFolder,
                      std::string& simulationFolder,
                      int& num_threads,
                      double& time_step,
                      int& write_interval_time,
                      double& dz,
                      double& AlphaDamp,
                      double& K_SPRINGS,
                      double& C_DAMPERS,
                      double& init_def,
                      double& p0_f,
                      double& p0_t,
                      double& L0,
                      double& L0_t,
                      double& TibiaMass,
                      double& femurMass,
                      ChVector<>& TibiaInertia,
                      ChVector<>& femurInertia,
                      double& rho,
                      double& E,
                      double& nu,
                      ChSystemDEM::ContactForceModel& ContactModel,
                      double& sphere_swept_thickness,
                      double& Kn,
                      double& Kt,
                      double& Gn,
                      double& Gt,
                      double& AlphaHHT,
                      int& MaxitersHHT,
                      double& AbsToleranceHHT,
                      double& AbsToleranceHHTConstraint,
                      int& MaxItersSuccessHHT) {
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
    assert(d.HasMember("RootFolder"));
    RootFolder = d["RootFolder"].GetString();
    simulationFolder = filename.c_str();

    unsigned first = simulationFolder.find_last_of("/") + 1;
    bool DidFind = (simulationFolder.find_last_of("/") != std::string::npos);
    unsigned last = simulationFolder.find_last_of(".");

    string strNew;
    if (DidFind) {
        strNew = simulationFolder.substr(first, last - first);
        cout << "simulation folder is SUCC " << strNew << "\n\n\n";
    } else {
        strNew = simulationFolder.substr(0, last);
        cout << "simulation folder is NOT " << strNew << "\n\n\n";
    }
    simulationFolder = strNew;

    assert(d.HasMember("Solution Control"));
    time_step = d["Solution Control"]["Time Step"].GetDouble();
    write_interval_time = d["Solution Control"]["Write Interval Step"].GetDouble() / time_step;
    num_threads = d["Solution Control"]["OMP Threads"].GetDouble();

    assert(d.HasMember("ANCF Shell"));
    AlphaDamp = d["ANCF Shell"]["Alpha Damp"].GetDouble();
    dz = d["ANCF Shell"]["Thickness"].GetDouble();

    assert(d.HasMember("Material Properties"));
    E = d["Material Properties"]["Elastisity"].GetDouble();
    rho = d["Material Properties"]["Density"].GetDouble() * 0.005 / dz;
    nu = d["Material Properties"]["poisson"].GetDouble();

    assert(d.HasMember("Elastic Foundation"));
    K_SPRINGS = d["Elastic Foundation"]["Stiffness"].GetDouble();
    C_DAMPERS = d["Elastic Foundation"]["Damping"].GetDouble();
    init_def = d["Elastic Foundation"]["Initial Springs displacement"].GetDouble();
    p0_f = d["Elastic Foundation"]["Internal Pressure Femur"].GetDouble();
    p0_t = d["Elastic Foundation"]["Internal Pressure Tibia"].GetDouble();

    L0 = d["Elastic Foundation"]["Springs_L0"].GetDouble();
    L0_t = d["Elastic Foundation"]["Dampers_L0"].GetDouble();

    assert(d.HasMember("Contact Properties"));
    ChSystemDEM::ContactForceModel* c_model = new ChSystemDEM::ContactForceModel;
    c_model = (ChSystemDEM::ContactForceModel*)d["Contact Properties"]["Contact Model"].GetString();
    ContactModel = *c_model;
    sphere_swept_thickness = d["Contact Properties"]["Contact thickness"].GetDouble() * dz;
    Kn = d["Contact Properties"]["Kn"].GetDouble();
    Kt = d["Contact Properties"]["Kt"].GetDouble();
    Gn = d["Contact Properties"]["Gn"].GetDouble();
    Gt = d["Contact Properties"]["Gt"].GetDouble();

    assert(d.HasMember("HHT Control"));
    AlphaHHT = d["HHT Control"]["Alpha"].GetDouble();
    MaxitersHHT = d["HHT Control"]["Maxiters"].GetInt();
    AbsToleranceHHT = d["HHT Control"]["AbsTolerance"].GetDouble();
    AbsToleranceHHTConstraint = d["HHT Control"]["AbsTolerance Constraints"].GetDouble();
    MaxItersSuccessHHT = d["HHT Control"]["MaxItersSuccessHHT"].GetInt();

    GetLog() << "Loaded JSON: " << filename.c_str() << "\n";
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
