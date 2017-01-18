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
// Author: Milad Rakhsha
// =============================================================================

// =============================================================================

// General Includes
#include <assert.h>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits.h>
#include <stdlib.h>  // system
#include <string>
#include <vector>

// Chrono Parallel Includes
#include "chrono/physics/ChSystemDEM.h"
// Solver
#include "chrono/ChConfig.h"

//#include "chrono_utils/ChUtilsVehicle.h"
#include "chrono/utils/ChUtilsCreators.h"
#include "chrono/utils/ChUtilsGenerators.h"
#include "chrono/utils/ChUtilsGeometry.h"
#include "chrono/utils/ChUtilsInputOutput.h"
#include "chrono/solver/ChSolverMINRES.h"
#include "chrono_mkl/ChSolverMKL.h"

// Chrono general utils

#include "chrono/core/ChFileutils.h"
#include "chrono/core/ChTransform.h"  //transform acc from GF to LF for post process
// Chrono fsi includes
#include "chrono_fsi/ChDeviceUtils.cuh"
#include "chrono_fsi/ChFsiTypeConvert.h"
#include "chrono_fsi/ChSystemFsi.h"

#include "chrono_fsi/utils/ChUtilsGeneratorFsi.h"
#include "chrono_fsi/utils/ChUtilsPrintSph.h"
#include "chrono_fsi/custom_math.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// Chrono fea includes
#include "chrono/physics/ChLoadContainer.h"
#include "chrono/physics/ChLoaderUV.h"
#include "chrono_fea/ChElementShellANCF.h"
#include "chrono_fea/ChLinkDirFrame.h"
#include "chrono_fea/ChLinkPointFrame.h"
#include "chrono_fea/ChNodeFEAxyzD.h"
#include "chrono_fea/ChMesh.h"

// FSI Interface Includes
#include "demos/fsi/demo_AC_model.h"  //SetupParamsH()

#define haveFluid 1
#define addPressure
#define NormalSP  // Defines whether spring and dampers will always remain normal to the surface

// Chrono namespaces
using namespace chrono;
using namespace chrono::fea;
using namespace chrono::collision;

using std::cout;
using std::endl;
std::ofstream simParams;
// =============================================================================

//----------------------------
// output directories and settings
//----------------------------
const std::string h_file = "/home/milad/CHRONO/Project-Chrono-Milad-IISPH/src/demos/fsi/demo_AC_model.h";
const std::string cpp_file = "/home/milad/CHRONO/Project-Chrono-Milad-IISPH/src/demos/fsi/demo_AC_model.cpp";

const std::string out_dir = "FSI_OUTPUT";  //"../FSI_OUTPUT";
const std::string data_folder = out_dir + "/AC_model/";
std::string MESH_CONNECTIVITY = data_folder + "Flex_MESH.vtk";

std::vector<std::vector<int>> NodeNeighborElementMesh;

bool povray_output = true;
int out_fps = 1000;

typedef fsi::Real Real;
Real contact_recovery_speed = 1;  ///< recovery speed for MBD

Real bxDim = 0.032;
Real byDim = 0.032;
Real bzDim = 0.005;
Real p0 = 0;
double p_max = 1000;
double t_ramp = 0.05;

Real fxDim = bxDim;
Real fyDim = byDim;
Real fzDim = bzDim;

double init_def = 0;
double K_SPRINGS = 1000;
double C_DAMPERS = 0.;
double L0_t = 0.005;
bool addSprings = true;

void writeMesh(std::shared_ptr<ChMesh> my_mesh, std::string SaveAs, std::vector<std::vector<int>>& NodeNeighborElement);
void writeFrame(std::shared_ptr<ChMesh> my_mesh,
                char SaveAsBuffer[256],
                char MeshFileBuffer[256],
                std::vector<std::vector<int>>& NodeNeighborElement);
void saveInputFile(std::string inputFile, std::string outAddress);
double applyRampPressure(double t, double t_ramp, double p_max);
void applySpringForce(std::shared_ptr<fea::ChMesh>& my_mesh,
                      double K_tot,
                      std::vector<ChVector<double>>& x0,
                      bool saveInitials);
void Calculator(fsi::ChSystemFsi& myFsiSystem,
                std::shared_ptr<fea::ChMesh> my_mesh,
                std::vector<ChVector<double>>& x0,
                double time,
                bool saveInitials);
void SaveParaViewFilesMBD(fsi::ChSystemFsi& myFsiSystem,
                          ChSystemDEM& mphysicalSystem,
                          std::shared_ptr<fea::ChMesh> my_mesh,
                          std::vector<std::vector<int>> NodeNeighborElementMesh,
                          chrono::fsi::SimParams* paramsH,
                          int next_frame,
                          double mTime);

void Create_MB_FE(ChSystemDEM& mphysicalSystem, fsi::ChSystemFsi& myFsiSystem, chrono::fsi::SimParams* paramsH);
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

// =============================================================================

#ifdef addPressure
std::vector<std::shared_ptr<ChLoad<ChLoaderPressure>>> faceload_mesh;
#endif

int main(int argc, char* argv[]) {
    time_t rawtime;
    struct tm* timeinfo;

    time(&rawtime);
    timeinfo = localtime(&rawtime);

    if (ChFileutils::MakeDirectory(out_dir.c_str()) < 0) {
        cout << "Error creating directory " << out_dir << endl;
        return 1;
    }

    if (ChFileutils::MakeDirectory(data_folder.c_str()) < 0) {
        cout << "Error creating directory " << data_folder << endl;
        return 1;
    }

    const std::string rmCmd = (std::string("rm ") + data_folder + std::string("/*"));
    system(rmCmd.c_str());
    saveInputFile(h_file, data_folder + "/hfile.h");
    saveInputFile(cpp_file, data_folder + "/cppfile.cpp");
    //****************************************************************************************
    const std::string simulationParams = out_dir + "/simulation_specific_parameters.txt";
    simParams.open(simulationParams);
    simParams << " Job was submitted at date/time: " << asctime(timeinfo) << endl;
    simParams.close();
    //****************************************************************************************
    bool mHaveFluid = false;
#if haveFluid
    mHaveFluid = true;
#endif

    // ************* Create Fluid *************************
    ChSystemDEM mphysicalSystem;
    chrono::fsi::ChSystemFsi myFsiSystem(&mphysicalSystem, mHaveFluid);
    chrono::ChVector<> CameraLocation = chrono::ChVector<>(0, -10, 0);
    chrono::ChVector<> CameraLookAt = chrono::ChVector<>(0, 0, 0);

    chrono::fsi::SimParams* paramsH = myFsiSystem.GetSimParams();

    SetupParamsH(paramsH, bxDim, byDim, bzDim, fxDim, fyDim, fzDim);
#if haveFluid
    Real initSpace0 = paramsH->MULT_INITSPACE * paramsH->HSML;
    utils::GridSampler<> sampler(initSpace0);

    chrono::fsi::Real3 boxCenter = chrono::fsi::mR3(-bxDim / 2 + fxDim / 2, 0 * initSpace0, fzDim / 2 + 2 * initSpace0);

    chrono::fsi::Real3 boxHalfDim = chrono::fsi::mR3(fxDim / 2, fyDim / 2, fzDim / 2);
    utils::Generator::PointVector points = sampler.SampleBox(fsi::ChFsiTypeConvert::Real3ToChVector(boxCenter),
                                                             fsi::ChFsiTypeConvert::Real3ToChVector(boxHalfDim));
    int numPart = points.size();
    for (int i = 0; i < numPart; i++) {
        myFsiSystem.GetDataManager()->AddSphMarker(
            chrono::fsi::mR3(points[i].x, points[i].y, points[i].z), chrono::fsi::mR3(0),
            chrono::fsi::mR4(paramsH->rho0, paramsH->BASEPRES, paramsH->mu0, -1));
    }

    int numPhases = myFsiSystem.GetDataManager()->fsiGeneralData.referenceArray.size();  // Arman TODO: either rely on

    if (numPhases != 0) {
        std::cout << "Error! numPhases is wrong, thrown from main\n" << std::endl;
        std::cin.get();
        return -1;
    } else {
        myFsiSystem.GetDataManager()->fsiGeneralData.referenceArray.push_back(
            mI4(0, numPart, -1, -1));  // map fluid -1, Arman : this will later be
                                       // removed, relying on finalize function and
                                       // automatic sorting
        myFsiSystem.GetDataManager()->fsiGeneralData.referenceArray.push_back(
            mI4(numPart, numPart, 0, 0));  // Arman : delete later
    }
#endif

    // ********************** Create Rigid ******************************

    ChVector<> gravity = ChVector<>(paramsH->gravity.x, paramsH->gravity.y, paramsH->gravity.z);

    Create_MB_FE(mphysicalSystem, myFsiSystem, paramsH);
    myFsiSystem.Finalize();
    auto my_fsi_mesh = myFsiSystem.GetFsiMesh();

    if (myFsiSystem.GetDataManager()->sphMarkersH.posRadH.size() !=
        myFsiSystem.GetDataManager()->numObjects.numAllMarkers) {
        printf("\n\n\n\n Error! (2) numObjects is not set correctly \n %d, %d \n\n\n",
               myFsiSystem.GetDataManager()->sphMarkersH.posRadH.size(),
               myFsiSystem.GetDataManager()->numObjects.numAllMarkers);
        return -1;
    }

    myFsiSystem.InitializeChronoGraphics(CameraLocation, CameraLookAt);

    int step_count = 0;
    double mTime = 0;
#ifdef CHRONO_FSI_USE_DOUBLE
    printf("Double Precision\n");
#else
    printf("Single Precision\n");
#endif

#ifdef USE_IRR
    application.GetSystem()->Update();
    //    application.SetPaused(true);
    int AccuNoIterations = 0;
    application.SetStepManage(true);
#endif

    mphysicalSystem.SetupInitial();

    //    mphysicalSystem.SetSolverType(ChSystem::SOLVER_MINRES);
    //    ChSolverMINRES* msolver = (ChSolverMINRES*)mphysicalSystem.GetSolverSpeed();
    //    msolver->SetDiagonalPreconditioning(true);
    //    mphysicalSystem.SetSolverWarmStarting(true);
    //    mphysicalSystem.SetMaxItersSolverSpeed(100);
    //    mphysicalSystem.SetMaxItersSolverStab(100);
    //    mphysicalSystem.SetTolForce(1e-6);

    ChSolverMKL<>* mkl_solver_stab = new ChSolverMKL<>;
    ChSolverMKL<>* mkl_solver_speed = new ChSolverMKL<>;
    mphysicalSystem.ChangeSolverStab(mkl_solver_stab);
    mphysicalSystem.ChangeSolverSpeed(mkl_solver_speed);
    mkl_solver_speed->SetSparsityPatternLock(true);
    mkl_solver_stab->SetSparsityPatternLock(true);
    //    mkl_solver_speed->SetVerbose(true);

    // Set up integrator
    mphysicalSystem.SetIntegrationType(ChSystem::INT_HHT);
    auto mystepper = std::static_pointer_cast<ChTimestepperHHT>(mphysicalSystem.GetTimestepper());
    mystepper->SetAlpha(-0.2);
    mystepper->SetMaxiters(100);
    mystepper->SetAbsTolerances(1e-5);
    mystepper->SetMode(ChTimestepperHHT::POSITION);
    mystepper->SetScaling(true);
    mystepper->SetVerbose(true);

    int stepEnd = int(paramsH->tFinal / paramsH->dT);
    stepEnd = 1000000;
    int TotalNumNodes = my_fsi_mesh->GetNnodes();

    SaveParaViewFilesMBD(myFsiSystem, mphysicalSystem, my_fsi_mesh, NodeNeighborElementMesh, paramsH, 0, mTime);
    std::vector<ChVector<double>> x0;  // displacement of the nodes
    Real time = 0;
    Real Global_max_dT = paramsH->dT_Max;
    for (int tStep = 0; tStep < stepEnd + 1; tStep++) {
        printf("step : %d, time= : %f (s) \n", tStep, time);
        double frame_time = 1.0 / out_fps;
        int next_frame = std::floor((time + 1e-6) / frame_time) + 1;
        double next_frame_time = next_frame * frame_time;
        double max_allowable_dt = next_frame_time - time;
        if (max_allowable_dt > 1e-6)
            paramsH->dT_Max = std::min(Global_max_dT, max_allowable_dt);
        else
            paramsH->dT_Max = Global_max_dT;

#if haveFluid

#ifdef addPressure
        double pressure = applyRampPressure(time, t_ramp, p_max);
        printf("The applied pressure is %f kPa. p_Max= %f, total Force= %f\n", pressure / 1000, p_max / 1000,
               pressure * bxDim * byDim);
        for (int i = 0; i < faceload_mesh.size(); i++) {
            faceload_mesh[i]->loader.SetPressure(pressure);
        }
#endif
        myFsiSystem.DoStepDynamics_FSI_Implicit();
        Calculator(myFsiSystem, my_fsi_mesh, x0, time, tStep == 0);

#else
        myFsiSystem.DoStepDynamics_ChronoRK2();
#endif
        time += paramsH->dT;
        SaveParaViewFilesMBD(myFsiSystem, mphysicalSystem, my_fsi_mesh, NodeNeighborElementMesh, paramsH, next_frame,
                             time);

        if (time > 0.4)
            break;
    }

    return 0;
}

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------
// Create the objects of the MBD system. Rigid bodies, and if fsi, their
// bce representation are created and added to the systems
//------------------------------------------------------------------
void Create_MB_FE(ChSystemDEM& mphysicalSystem, fsi::ChSystemFsi& myFsiSystem, chrono::fsi::SimParams* paramsH) {
    mphysicalSystem.Set_G_acc(ChVector<>(0, 0, 0));
    std::shared_ptr<chrono::ChMaterialSurface> mat_g(new chrono::ChMaterialSurface);
    // Set common material Properties
    mat_g->SetFriction(0.8);
    mat_g->SetCohesion(0);
    mat_g->SetCompliance(0.0);
    mat_g->SetComplianceT(0.0);
    mat_g->SetDampingF(0.2);

    auto ground = std::make_shared<ChBody>();
    ground->SetIdentifier(-1);

    ground->SetBodyFixed(true);
    ground->SetCollide(true);
    ground->SetMaterialSurface(mat_g);

    ground->GetCollisionModel()->ClearModel();
    Real initSpace0 = paramsH->MULT_INITSPACE * paramsH->HSML;

    // Bottom wall
    //    ChVector<> sizeBottom(bxDim / 2 + 3 * initSpace0, byDim / 2 + 3 * initSpace0, 2 * initSpace0);
    ChVector<> sizeBottom(bxDim / 2, byDim / 2, 3 * initSpace0);
    ChVector<> posBottom(0, 0, -2 * initSpace0);
    ChVector<> posTop(0, 0, bzDim + 10 * initSpace0);

    // left and right Wall
    ChVector<> size_YZ(2 * initSpace0, byDim / 2 + 3 * initSpace0, bzDim / 2);
    ChVector<> pos_xp(bxDim / 2 + initSpace0, 0.0, bzDim / 2 + 1 * initSpace0);
    ChVector<> pos_xn(-bxDim / 2 - 3 * initSpace0, 0.0, bzDim / 2 + 1 * initSpace0);

    // Front and back Wall
    ChVector<> size_XZ(bxDim / 2, 2 * initSpace0, bzDim / 2);
    ChVector<> pos_yp(0, byDim / 2 + initSpace0, bzDim / 2 + 1 * initSpace0);
    ChVector<> pos_yn(0, -byDim / 2 - 3 * initSpace0, bzDim / 2 + 1 * initSpace0);

    chrono::utils::AddBoxGeometry(ground.get(), sizeBottom, posBottom, chrono::QUNIT, true);
    //    chrono::utils::AddBoxGeometry(ground.get(), size_YZ, pos_xp, chrono::QUNIT, true);
    //    chrono::utils::AddBoxGeometry(ground.get(), size_YZ, pos_xn, chrono::QUNIT, true);
    //    chrono::utils::AddBoxGeometry(ground.get(), size_XZ, pos_yp, chrono::QUNIT, true);
    //    chrono::utils::AddBoxGeometry(ground.get(), size_XZ, pos_yn, chrono::QUNIT, true);
    mphysicalSystem.AddBody(ground);

#if haveFluid

    /*================== Walls =================*/

    chrono::fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, ground, posBottom, chrono::QUNIT, sizeBottom);
    chrono::fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, ground, posTop, chrono::QUNIT, sizeBottom);

    //    chrono::fsi::utils::AddBoxBceYZ(myFsiSystem.GetDataManager(), paramsH, ground, pos_xp, chrono::QUNIT,
    //    size_YZ);
    //
    //    chrono::fsi::utils::AddBoxBceYZ(myFsiSystem.GetDataManager(), paramsH, ground, pos_xn, chrono::QUNIT,
    //    size_YZ);
    //
    //    chrono::fsi::utils::AddBoxBceXZ(myFsiSystem.GetDataManager(), paramsH, ground, pos_yp, chrono::QUNIT,
    //    size_XZ);
    //
    //    chrono::fsi::utils::AddBoxBceXZ(myFsiSystem.GetDataManager(), paramsH, ground, pos_yn, chrono::QUNIT,
    //    size_XZ);

    /*================== Flexible-Body =================*/

    GetLog() << "-----------------------------------------------------------\n";
    GetLog() << "-----------------------------------------------------------\n";
    GetLog() << "     ANCF Shell Elements demo with implicit integration \n";
    GetLog() << "-----------------------------------------------------------\n";

    // Create a mesh, that is a container for groups of elements and their referenced nodes.
    auto my_mesh = std::make_shared<fea::ChMesh>();

    // Geometry of the plate
    double plate_lenght_x = bxDim + 0 * initSpace0;
    double plate_lenght_y = byDim + 0 * initSpace0;
    double plate_lenght_z = 0.001;
    // Specification of the mesh
    int numDiv_x = 8;
    int numDiv_y = 8;
    int numDiv_z = 1;
    int N_x = numDiv_x + 1;
    int N_y = numDiv_y + 1;
    int N_z = numDiv_z + 1;
    // Number of elements in the z direction is considered as 1
    int TotalNumElements = numDiv_x * numDiv_y;
    int TotalNumNodes = (numDiv_x + 1) * (numDiv_y + 1);
    // For uniform mesh
    double dx = plate_lenght_x / numDiv_x;
    double dy = plate_lenght_y / numDiv_y;
    double dz = plate_lenght_z / numDiv_z;
    std::vector<std::vector<int>> elementsNodes_mesh;
    std::vector<std::vector<int>> NodeNeighborElement_mesh;
    elementsNodes_mesh.resize(TotalNumElements);
    NodeNeighborElement_mesh.resize(TotalNumNodes);
    // Create and add the nodes
    for (int i = 0; i < TotalNumNodes; i++) {
        // Node location
        double loc_x = (i % (numDiv_x + 1)) * dx - bxDim / 2 - 0 * initSpace0;
        double loc_y = (i / (numDiv_x + 1)) % (numDiv_y + 1) * dy - byDim / 2 - 0 * initSpace0;
        double loc_z = (i) / ((numDiv_x + 1) * (numDiv_y + 1)) * dz + bzDim + 3 * initSpace0;

        // Node direction
        double dir_x = 0;
        double dir_y = 0;
        double dir_z = 1;

        // Create the node
        auto node = std::make_shared<ChNodeFEAxyzD>(ChVector<>(loc_x, loc_y, loc_z), ChVector<>(dir_x, dir_y, dir_z));

        node->SetMass(0);

        //        if (i == 0 || i == numDiv_x || i == (TotalNumNodes - 1) || i == (TotalNumNodes - 1) - numDiv_x) {
        //            node->SetFixed(true);
        //        }

        //        if (i == 0 || i == numDiv_x) {
        //            node->SetFixed(true);
        //        }

        if (i % (numDiv_x + 1) == 0 || i % (numDiv_x) == 0 || i < (numDiv_x + 1) ||
            i >= (TotalNumNodes - numDiv_x - 1)) {
            auto NodeDir = std::make_shared<ChLinkDirFrame>();
            NodeDir->Initialize(node, ground);
            NodeDir->SetDirectionInAbsoluteCoords(node->D);
            mphysicalSystem.Add(NodeDir);
        }

        //        if (abs(loc_x) < bxDim / 4 && abs(loc_y) < byDim / 4) {
        //            node->SetForce(ChVector<>(0, 0, 1));
        //        }

        // Add node to mesh
        my_mesh->AddNode(node);
    }

    ChVector<> mforce;
    for (int iNode = 0; iNode < TotalNumNodes; iNode++) {
        auto Node = std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(iNode));
        //            Node->SetForce(ChVector<>(0, 0, 0.1));
        mforce += Node->GetForce();
    }
    printf("GET.FORCES INITIAL=%f\n", mforce.Length());

    // Create an orthotropic material.
    // All layers for all elements share the same material.
    double rho = 1000;
    double E = 40e7;
    double nu = 0.3;
    auto mat = std::make_shared<ChMaterialShellANCF>(rho, E, nu);
    // Create the elements
    for (int i = 0; i < TotalNumElements; i++) {
        // Adjacent nodes
        int node0 = (i / (numDiv_x)) * (N_x) + i % numDiv_x;
        int node1 = (i / (numDiv_x)) * (N_x) + i % numDiv_x + 1;
        int node2 = (i / (numDiv_x)) * (N_x) + i % numDiv_x + 1 + N_x;
        int node3 = (i / (numDiv_x)) * (N_x) + i % numDiv_x + N_x;
        elementsNodes_mesh[i].push_back(node0 + 1);
        elementsNodes_mesh[i].push_back(node1 + 1);
        elementsNodes_mesh[i].push_back(node2 + 1);
        elementsNodes_mesh[i].push_back(node3 + 1);
        NodeNeighborElement_mesh[node0].push_back(i);
        NodeNeighborElement_mesh[node1].push_back(i);
        NodeNeighborElement_mesh[node2].push_back(i);
        NodeNeighborElement_mesh[node3].push_back(i);
        // Create the element and set its nodes.
        auto element = std::make_shared<ChElementShellANCF>();
        element->SetNodes(std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(node0)),
                          std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(node1)),
                          std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(node2)),
                          std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(node3)));
        // Set element dimensions
        element->SetDimensions(dx, dy);
        // Add a single layers with a fiber angle of 0 degrees.
        element->AddLayer(dz, 0 * CH_C_DEG_TO_RAD, mat);
        // Set other element properties
        element->SetAlphaDamp(0.05);   // Structural damping for this element
        element->SetGravityOn(false);  // turn internal gravitational force calculation off
        // Add element to mesh
        my_mesh->AddElement(element);
    }

    double NODE_AVE_AREA = dx * dy;
    double Tottal_stiff = 0;
    double Tottal_damp = 0;

    auto Springsloadcontainer = std::make_shared<ChLoadContainer>();

    if (addSprings) {
        // Select on which nodes we are going to apply a load
        std::vector<std::shared_ptr<ChLoadable>> NodeList;
        for (int iNode = 0; iNode < my_mesh->GetNnodes(); iNode++) {
            auto NodeLoad = std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(iNode));
            NodeList.push_back(NodeLoad);
        }
        auto OneLoadSpringDamper = std::make_shared<MyLoadSpringDamper>(NodeList, ground, init_def);
        for (int iNode = 0; iNode < my_mesh->GetNnodes(); iNode++) {
            auto Node = std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(iNode));
            ChVector<> AttachBodyGlobal = Node->GetPos() - L0_t * Node->GetD();  // Locate first the
            // attachment point in the body in global coordiantes
            // Stiffness of the Elastic Foundation
            double K_S = K_SPRINGS / TotalNumNodes;  // Stiffness Constant
            double C_S = C_DAMPERS / TotalNumNodes;  // Damper Constant
            Tottal_stiff += K_S;
            Tottal_damp += C_S;
            // Initial length
            OneLoadSpringDamper->C_dp[iNode] = C_S;
            OneLoadSpringDamper->K_sp[iNode] = K_S;
            OneLoadSpringDamper->l0[iNode] = L0_t;

            // Calculate the damping ratio zeta
            double zeta, m_ele;
            m_ele = rho * dz * NODE_AVE_AREA;
            zeta = C_S / (2 * sqrt(K_S * m_ele));
            //            GetLog() << "Zeta of node # " << iNode << " is set to : " << zeta << "\n";
            OneLoadSpringDamper->LocalBodyAtt[iNode] = ground->Point_World2Body(AttachBodyGlobal);
        }
        Springsloadcontainer->Add(OneLoadSpringDamper);
        GetLog() << "Total Stiffness (N/mm)= " << Tottal_stiff / 1e3 << " Total Damping = " << Tottal_damp
                 << " Average zeta= " << Tottal_damp / (2 * sqrt(Tottal_stiff * (rho * dz * 1e-3))) << "\n";

        mphysicalSystem.Add(Springsloadcontainer);
    }
    my_mesh->SetAutomaticGravity(false);

#ifdef addPressure
    // First: loads must be added to "load containers",
    // and load containers must be added to your ChSystem
    auto Pressureloadcontainer = std::make_shared<ChLoadContainer>();
    // Add constant pressure using ChLoaderPressure (preferred for simple, constant pressure)
    for (int NoElmPre = 0; NoElmPre < TotalNumElements; NoElmPre++) {
        auto faceload = std::make_shared<ChLoad<ChLoaderPressure>>(
            std::static_pointer_cast<ChElementShellANCF>(my_mesh->GetElement(NoElmPre)));
        faceload->loader.SetPressure(p0);
        faceload->loader.SetStiff(true);
        faceload->loader.SetIntegrationPoints(2);
        faceload_mesh.push_back(faceload);
        Pressureloadcontainer->Add(faceload);
    }
    mphysicalSystem.Add(Pressureloadcontainer);
#endif

    // Add the mesh to the system

    std::vector<std::shared_ptr<chrono::fea::ChElementShellANCF>>* FSI_Shells = myFsiSystem.GetFsiShellsPtr();
    std::vector<std::shared_ptr<chrono::fea::ChNodeFEAxyzD>>* FSI_Nodes = myFsiSystem.GetFsiNodesPtr();

    bool multilayer = true;
    bool removeMiddleLayer = false;

    chrono::fsi::utils::AddBCE_ShellFromMesh(myFsiSystem.GetDataManager(), paramsH, FSI_Shells, FSI_Nodes, my_mesh,
                                             elementsNodes_mesh, NodeNeighborElement_mesh, multilayer,
                                             removeMiddleLayer, +2);

    mphysicalSystem.Add(my_mesh);
    myFsiSystem.SetFsiMesh(my_mesh);
    myFsiSystem.SetShellelementsNodes(elementsNodes_mesh);

    writeMesh(my_mesh, MESH_CONNECTIVITY, NodeNeighborElementMesh);

#endif
}

double applyRampPressure(double t, double t_ramp, double p_max) {
    if (t > t_ramp)
        return (p_max);
    else
        return (p_max * t / t_ramp);
}

void applySpringForce(std::shared_ptr<fea::ChMesh>& my_mesh,
                      double K_tot,
                      std::vector<ChVector<double>>& x0,
                      bool saveInitials) {
    int TotalNumNodes = my_mesh->GetNnodes();
    double k_i = K_tot / TotalNumNodes;
    ChVector<> Dir;
    double delta_Ave;
    for (int iNode = 0; iNode < TotalNumNodes; iNode++) {
        auto Node = std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(iNode));
        if (saveInitials) {
            x0.push_back(Node->GetPos());
        }
        double delta = Vdot(Node->GetPos() - x0[iNode], Node->GetD().GetNormalized());
        ChVector<> myForce = -1 * delta * k_i * Node->GetD().GetNormalized();
        delta_Ave += delta;
        Node->SetForce(myForce + Node->GetForce());
        Dir += Node->GetD().GetNormalized();
    }
    delta_Ave /= TotalNumNodes;
    printf("The spring forces=%f, delta_ave= %f, Total Stiffness= %f\n", delta_Ave * K_tot, delta_Ave, K_tot);
}

void Calculator(fsi::ChSystemFsi& myFsiSystem,
                std::shared_ptr<fea::ChMesh> my_mesh,
                std::vector<ChVector<double>>& x0,
                double time,
                bool saveInitials) {
    std::ofstream output;
    output.open((data_folder + "/Analysis.txt").c_str(), std::ios::app);

    //    thrust::host_vector<Real3> posRadH = myFsiSystem.GetDataManager()->sphMarkersD2.posRadD;
    //    thrust::host_vector<Real3> velMasH = myFsiSystem.GetDataManager()->sphMarkersD2.velMasD;
    thrust::host_vector<chrono::fsi::Real4> rhoPresMuH = myFsiSystem.GetDataManager()->sphMarkersD2.rhoPresMuD;
    thrust::host_vector<int4> referenceArray = myFsiSystem.GetDataManager()->fsiGeneralData.referenceArray;
    Real p_Ave = 0;
    for (int i = referenceArray[0].x; i < referenceArray[0].y; i++) {
        //    Real3 pos = posRadH[i];
        //    Real3 vel = velMasH[i];
        p_Ave += rhoPresMuH[i].y;
    }
    p_Ave /= (referenceArray[0].y - referenceArray[0].x);
    //    posRadH.clear();
    //    velMasH.clear();
    rhoPresMuH.clear();

    int TotalNumNodes = my_mesh->GetNnodes();
    double delta_Ave;
    for (int iNode = 0; iNode < TotalNumNodes; iNode++) {
        auto Node = std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(iNode));
        if (saveInitials) {
            x0.push_back(Node->GetPos());
        }
        ChVector<> x_i = x0[iNode];
        double delta = Vdot(Node->GetPos() - x_i, Node->GetD().GetNormalized());
        delta_Ave += delta;
    }
    delta_Ave /= TotalNumNodes;
    double AppliedPressure = applyRampPressure(time, t_ramp, p_max);
    printf("delta(mm)=%f, applied_Pressure=%f, sigma_s=%f, sigma_f=%f\n", delta_Ave * 1000, AppliedPressure,
           K_SPRINGS * delta_Ave / bxDim / byDim, p_Ave);
    output << time << " " << delta_Ave << " " << AppliedPressure << " " << K_SPRINGS * delta_Ave / bxDim / byDim << " "
           << p_Ave << std::endl;
    output.close();
}

//------------------------------------------------------------------
// Function to save the povray files of the MBD
//------------------------------------------------------------------
void SaveParaViewFilesMBD(fsi::ChSystemFsi& myFsiSystem,
                          ChSystemDEM& mphysicalSystem,
                          std::shared_ptr<fea::ChMesh> my_mesh,
                          std::vector<std::vector<int>> NodeNeighborElementMesh,
                          chrono::fsi::SimParams* paramsH,
                          int next_frame,
                          double mTime) {
    static double exec_time;
    int out_steps = std::ceil((1.0 / paramsH->dT) / out_fps);
    exec_time += mphysicalSystem.GetTimerStep();
    int num_contacts = mphysicalSystem.GetNcontacts();
    double frame_time = 1.0 / out_fps;
    static int out_frame = 0;

    // If enabled, output data for PovRay postprocessing.
    //    printf("mTime= %f\n", mTime - (next_frame)*frame_time);

    if (povray_output && std::abs(mTime - (next_frame)*frame_time) < 0.00001) {
        // **** out fluid
        chrono::fsi::utils::PrintToParaViewFile(
            myFsiSystem.GetDataManager()->sphMarkersD2.posRadD, myFsiSystem.GetDataManager()->sphMarkersD2.velMasD,
            myFsiSystem.GetDataManager()->sphMarkersD2.rhoPresMuD,
            myFsiSystem.GetDataManager()->fsiGeneralData.referenceArray,
            myFsiSystem.GetDataManager()->fsiGeneralData.referenceArray_FEA, data_folder);
#ifdef AddCylinder
        char SaveAsRigidObjVTK[256];  // The filename buffer.
        static int RigidCounter = 0;

        snprintf(SaveAsRigidObjVTK, sizeof(char) * 256, (data_folder + "/Cylinder.%d.vtk").c_str(), RigidCounter);
        WriteCylinderVTK(Cylinder, cyl_radius, cyl_length, 100, SaveAsRigidObjVTK);

        RigidCounter++;
#endif

        cout << "\n------------ Output frame:   " << next_frame << endl;
        cout << "             Sim frame:      " << next_frame << endl;
        cout << "             Time:           " << mTime << endl;
        cout << "             Avg. contacts:  " << num_contacts / out_steps << endl;
        cout << "             Execution time: " << exec_time << endl << endl;
        cout << "\n----------------------------\n" << endl;

        char SaveAsBuffer[256];  // The filename buffer.
        snprintf(SaveAsBuffer, sizeof(char) * 256, (data_folder + "/flex_body.%d.vtk").c_str(), next_frame);
        char MeshFileBuffer[256];  // The filename buffer.
        snprintf(MeshFileBuffer, sizeof(char) * 256, ("%s"), MESH_CONNECTIVITY.c_str());
        //        printf("%s from here\n", MeshFileBuffer);
        writeFrame(my_mesh, SaveAsBuffer, MeshFileBuffer, NodeNeighborElementMesh);

        out_frame++;
    }
}
//////////////////////////////////////////////////////
///////////Write to MESH Cennectivity/////////////////
/////////////////////////////////////////////////////
void writeMesh(std::shared_ptr<ChMesh> my_mesh,
               std::string SaveAs,
               std::vector<std::vector<int>>& NodeNeighborElement) {
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

    MESH.write_to_file(SaveAs);
}
////////////////////////////////////////
///////////Write to VTK/////////////////
////////////////////////////////////////
void writeFrame(std::shared_ptr<ChMesh> my_mesh,
                char SaveAsBuffer[256],
                char MeshFileBuffer[256],
                std::vector<std::vector<int>>& NodeNeighborElement) {
    std::ofstream output;
    output.open(SaveAsBuffer, std::ios::app);

    output << "# vtk DataFile Version 1.0\nUnstructured Grid Example\nASCII\n\n" << std::endl;
    output << "DATASET UNSTRUCTURED_GRID\nPOINTS " << my_mesh->GetNnodes() << " float\n";
    for (int i = 0; i < my_mesh->GetNnodes(); i++) {
        auto node = std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(i));
        output << node->GetPos().x << " " << node->GetPos().y << " " << node->GetPos().z << "\n";
    }
    std::ifstream CopyFrom(MeshFileBuffer);
    output << CopyFrom.rdbuf();
    output << "\nPOINT_DATA " << my_mesh->GetNnodes() << "\n";
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
            areaAve1 += MyResult[2].x * dx * dy / NodeNeighborElement[i].size();
            areaAve2 += MyResult[2].y * dx * dy / NodeNeighborElement[i].size();
            areaAve3 += MyResult[2].z * dx * dy / NodeNeighborElement[i].size();
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
            areaAve1 += MyResult[2].x * dx * dy / NodeNeighborElement[i].size();
            areaAve2 += MyResult[2].y * dx * dy / NodeNeighborElement[i].size();
            areaAve3 += MyResult[2].z * dx * dy / NodeNeighborElement[i].size();
        }
        output << areaAve1 / myarea << " " << areaAve2 / myarea << " " << areaAve3 / myarea << "\n";
    }

    output.close();
}

//------------------------------------------------------------------
// function to set the solver setting for the
//------------------------------------------------------------------

void saveInputFile(std::string inputFile, std::string outAddress) {
    std::ifstream inFile;
    inFile.open(inputFile);
    std::ofstream outFile;
    outFile.open(outAddress);
    outFile << inFile.rdbuf();
    inFile.close();
    outFile.close();
    std::cout << inputFile << "	" << outAddress << "\n";
}
