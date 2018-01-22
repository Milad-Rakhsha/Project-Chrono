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
#include <limits.h>
#include <stdlib.h>  // system
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// Chrono Parallel Includes
#include "chrono/physics/ChSystemSMC.h"
// Solver
#include "chrono/ChConfig.h"

//#include "chrono_utils/ChUtilsVehicle.h"
#include "chrono/motion_functions/ChFunction.h"
#include "chrono/solver/ChSolverMINRES.h"
#include "chrono/utils/ChUtilsCreators.h"
#include "chrono/utils/ChUtilsGenerators.h"
#include "chrono/utils/ChUtilsGeometry.h"
#include "chrono/utils/ChUtilsInputOutput.h"
#include "chrono_mkl/ChSolverMKL.h"

// Chrono general utils

#include "chrono/core/ChFileutils.h"
#include "chrono/core/ChTransform.h"  //transform acc from GF to LF for post process
// Chrono fsi includes
#include "chrono_fsi/ChDeviceUtils.cuh"
#include "chrono_fsi/ChFsiTypeConvert.h"
#include "chrono_fsi/ChSystemFsi.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "chrono_fsi/custom_math.h"
#include "chrono_fsi/utils/ChUtilsGeneratorFsi.h"
#include "chrono_fsi/utils/ChUtilsPrintSph.cuh"

// Chrono fea includes

#include "chrono/physics/ChBodyEasy.h"
#include "chrono/physics/ChForce.h"
#include "chrono/physics/ChLoadContainer.h"
#include "chrono/physics/ChLoaderUV.h"

#include "chrono_fea/ChBuilderBeam.h"
#include "chrono_fea/ChElementCableANCF.h"
#include "chrono_fea/ChElementShellANCF.h"
#include "chrono_fea/ChLinkDirFrame.h"

#include "chrono_fea/ChContactSurfaceMesh.h"
#include "chrono_fea/ChLinkPointFrame.h"
#include "chrono_fea/ChLinkPointFrameGeneral.h"
#include "chrono_fea/ChLinkPointPoint.h"
#include "chrono_fea/ChMesh.h"
#include "chrono_fea/ChNodeFEAxyzD.h"

// FSI Interface Includes
#include "demos/fsi/demo_indentation.h"  //SetupParamsH()

#define haveFluid 1
//#define addPressure
#define addIndentor
#define NormalSP  // Defines whether spring and dampers will always remain normal to the surface

// Chrono namespaces
using namespace chrono;
using namespace chrono::fea;
using namespace chrono::collision;

using std::cout;
using std::endl;
std::ofstream simParams;
// =============================================================================
const std::string out_dir = GetChronoOutputPath() + "Indentation_test";
const std::string demo_dir = out_dir + "/Indentation_test";
std::string MESH_CONNECTIVITY = demo_dir + "Flex_MESH.vtk";

bool save_output = true;

std::vector<std::vector<int>> NodeNeighborElementMesh;

bool povray_output = true;
int out_fps = 1000;

typedef fsi::Real Real;
Real contact_recovery_speed = 1;  ///< recovery speed for MBD

Real bxDim = 0.01;
Real byDim = 0.01;
Real bzDim = 0.0045;
Real p0 = 0;

Real fxDim = 0.01;
Real fyDim = 0.01;
Real fzDim = 0.0035;

// For displacement driven method
double Indentor_R = 0.0032;
double Indentaiton_rate = -500.0 * 1e-6;
double x0 = bzDim;

// For force-driven method
double t_ramp = 100;
double p_max = 1;

double init_def = 0;
double K_SPRINGS = 200;  // 1000;
double C_DAMPERS = 0.;
double L0_t = 0.005;
bool addSprings = false;
bool addCable = true;
bool refresh_FlexBodiesInsideFluid = true;

int numCableNodes = 0;

std::vector<std::shared_ptr<ChLinkPointFrame>> rigidLinks;

void writeMesh(std::shared_ptr<ChMesh> my_mesh,
               std::string SaveAs,
               std::vector<std::vector<int>>& NodeNeighborElement,
               std::vector<std::vector<int>> _1D_elementsNodes_mesh,
               std::vector<std::vector<int>> _2D_elementsNodes_mesh);
void writeFrame(std::shared_ptr<ChMesh> my_mesh,
                char SaveAsBuffer[256],
                char MeshFileBuffer[256],
                std::vector<std::vector<int>>& NodeNeighborElement);
double applyRampPressure(double t, double t_ramp, double p_max);
void impose_motion(std::shared_ptr<ChLinkLockLock> my_link);
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
                          ChSystemSMC& mphysicalSystem,
                          std::shared_ptr<fea::ChMesh> my_mesh,
                          std::vector<std::vector<int>> NodeNeighborElementMesh,
                          chrono::fsi::SimParams* paramsH,
                          int next_frame,
                          double mTime);

void Create_MB_FE(ChSystemSMC& mphysicalSystem, fsi::ChSystemFsi& myFsiSystem, chrono::fsi::SimParams* paramsH);
void calcSomeParams(const int numDiv_x,
                    const int numDiv_y,
                    const int numDiv_z,
                    const double plate_lenght_x,
                    const double plate_lenght_y,
                    const double plate_lenght_z,
                    int& N_x,
                    int& N_y,
                    int& N_z,
                    int& TotalNumElements,
                    int& TotalNumNodes_onSurface,
                    double& dx,
                    double& dy,
                    double& dz);
// =============================================================================

#ifdef addPressure
std::vector<std::shared_ptr<ChLoad<ChLoaderPressure>>> faceload_mesh;
#endif

int main(int argc, char* argv[]) {
    time_t rawtime;
    struct tm* timeinfo;

    time(&rawtime);
    timeinfo = localtime(&rawtime);
    std::cout << "Starting device 1 on" << std::endl;
    cudaSetDevice(0);

    if (ChFileutils::MakeDirectory(out_dir.c_str()) < 0) {
        cout << "Error creating directory " << out_dir << endl;
        return 1;
    }

    if (ChFileutils::MakeDirectory(demo_dir.c_str()) < 0) {
        cout << "Error creating directory " << demo_dir << endl;
        return 1;
    }

    const std::string rmCmd = (std::string("rm ") + demo_dir + std::string("/*"));
    system(rmCmd.c_str());

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
    ChSystemSMC mphysicalSystem;
    chrono::fsi::ChSystemFsi myFsiSystem(&mphysicalSystem, mHaveFluid, fsi::ChFluidDynamics::Integrator::IISPH);
    chrono::ChVector<> CameraLocation = chrono::ChVector<>(0, -10, 0);
    chrono::ChVector<> CameraLookAt = chrono::ChVector<>(0, 0, 0);

    chrono::fsi::SimParams* paramsH = myFsiSystem.GetSimParams();

    SetupParamsH(paramsH, bxDim, byDim, bzDim, fxDim, fyDim, fzDim);
#if haveFluid
    Real initSpace0 = paramsH->MULT_INITSPACE * paramsH->HSML;
    utils::GridSampler<> sampler(initSpace0);

    chrono::fsi::Real3 boxCenter = chrono::fsi::mR3(-bxDim / 2 + fxDim / 2, 0 * initSpace0, fzDim / 2 + 1 * initSpace0);

    chrono::fsi::Real3 boxHalfDim = chrono::fsi::mR3(fxDim / 2, fyDim / 2, fzDim / 2);

    utils::Generator::PointVector points = sampler.SampleBox(fsi::ChFsiTypeConvert::Real3ToChVector(boxCenter),
                                                             fsi::ChFsiTypeConvert::Real3ToChVector(boxHalfDim));
    int numPart = points.size();

    // This section is important to remove the fluid particles that overlap the BCE particles.
    // In order to use this future, two things needs to be done;
    // 1- run the simulation as usual, e.g. ./bin/demo_indentation, which will create the BCE_Flex0.csv
    // This could be the actual simulation if there is no overlap between the fluid and BCE markers, if there is:
    // 2- run the actual simulation with a command line argument such as ./bin/demo_indentation 2, which will go over
    // the BCE_Flex0.csv file and use those information to remove overlap of fluid and BCE markers

    std::cout << "Initial fluid size:" << points.size() << std::endl;
    std::vector<fsi::Real3> particles_position;
    if (argc > 1) {
        fsi::Real3 this_particle;

        std::fstream fin("BCE_Flex0.csv");
        if (!fin.good())
            throw ChException("ERROR opening Mesh file: BCE_Flex0.csv \n");

        std::string line;
        getline(fin, line);

        while (getline(fin, line)) {
            int ntoken = 0;
            std::string token;
            std::istringstream ss(line);
            while (std::getline(ss, token, ' ') && ntoken < 4) {
                std::istringstream stoken(token);
                if (ntoken == 0)
                    stoken >> this_particle.x;
                if (ntoken == 1)
                    stoken >> this_particle.y;
                if (ntoken == 2)
                    stoken >> this_particle.z;
                ++ntoken;
            }
            particles_position.push_back(this_particle);
        }
    }
    std::cout << "Set Removal points from: BCE_Flex0.csv" << std::endl;
    std::cout << "Searching among " << particles_position.size() << "flex points" << std::endl;

    int numremove = 0;
    for (int i = 0; i < numPart; i++) {
        bool removeThis = false;
        fsi::Real4 p = fsi::mR4(points[i].x(), points[i].y(), points[i].z(), initSpace0);
        for (int remove = 0; remove < particles_position.size(); remove++) {
            double dist = length(particles_position[remove] - mR3(p));
            if (dist < initSpace0 * 0.95) {
                removeThis = true;
                break;
            }
        }
        if (!removeThis) {
            myFsiSystem.GetDataManager()->AddSphMarker(
                p, chrono::fsi::mR3(0.0), chrono::fsi::mR4(paramsH->rho0, paramsH->BASEPRES, paramsH->mu0, -1));
        } else
            numremove++;
    }

    std::cout << "Removed " << numremove << " Fluid particles from the simulation" << std::endl;
    std::cout << "Final fluid size:" << myFsiSystem.GetDataManager()->sphMarkersH.rhoPresMuH.size() << std::endl;
    particles_position.clear();

    // This is important because later on we
    int numFluidPart = myFsiSystem.GetDataManager()->sphMarkersH.posRadH.size();

    int numPhases = myFsiSystem.GetDataManager()->fsiGeneralData.referenceArray.size();  // Arman TODO: either rely on
    if (numPhases != 0) {
        std::cout << "Error! numPhases is wrong, thrown from main\n" << std::endl;
        std::cin.get();
        return -1;
    } else {
        myFsiSystem.GetDataManager()->fsiGeneralData.referenceArray.push_back(
            mI4(0, numFluidPart, -1, -1));  // map fluid -1, Arman : this will later be
                                            // removed, relying on finalize function and
                                            // automatic sorting
        myFsiSystem.GetDataManager()->fsiGeneralData.referenceArray.push_back(
            mI4(numFluidPart, numFluidPart, 0, 0));  // Arman : delete later
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

#ifdef CHRONO_MKL
    auto mkl_solver = std::make_shared<ChSolverMKL<>>();
    mkl_solver->SetSparsityPatternLock(true);
    mphysicalSystem.SetSolver(mkl_solver);
#else
    mphysicalSystem.SetSolverType(ChSolver::Type::MINRES);
    mphysicalSystem.SetSolverWarmStarting(true);
    mphysicalSystem.SetMaxItersSolverSpeed(10000);
    mphysicalSystem.SetTolForce(1e-10);
#endif

    // Set up integrator
    mphysicalSystem.SetTimestepperType(ChTimestepper::Type::EULER_IMPLICIT);

    //    auto mystepper = std::static_pointer_cast<ChTimestepperHHT>(mphysicalSystem.GetTimestepper());
    //    mystepper->SetAlpha(-0.2);
    //    mystepper->SetMaxiters(100);
    //    mystepper->SetAbsTolerances(1e-5);
    //    mystepper->SetMode(ChTimestepperHHT::POSITION);
    //    mystepper->SetScaling(true);
    //    mystepper->SetVerbose(false);

    int stepEnd = int(paramsH->tFinal / paramsH->dT);
    stepEnd = 1000000;
    int TotalNumNodes = my_fsi_mesh->GetNnodes();

    SaveParaViewFilesMBD(myFsiSystem, mphysicalSystem, my_fsi_mesh, NodeNeighborElementMesh, paramsH, 0, mTime);
    const std::string copyInitials =
        (std::string("cp ") + demo_dir + std::string("/BCE_Flex0.csv") + std::string(" ./ "));
    system(copyInitials.c_str());
    if (argc <= 1) {
        printf("now please run with an input argument\n");
        return 0;
    }
    std::vector<ChVector<double>> x0;  // displacement of the nodes
    Real time = 0;
    Real Global_max_dT = paramsH->dT_Max;
    for (int tStep = 0; tStep < stepEnd + 1; tStep++) {
        ChVector<> ground_Force(0);
        ChVector<> ground_Xforce(0);

        //        auto mBody = (std::shared_ptr<ChBody>)mphysicalSystem.Get_bodylist()->at(0);
        //        ground_Xforce = mBody->Get_Xforce() * 1000;
        //        std::vector<std::shared_ptr<ChForce>> body_forces = mBody->GetForceList();
        //        chrono::ChForce::ForceType mode = ChForce::FORCE;
        //        for (int i = 0; i < body_forces.size(); i++) {
        //            if (body_forces[i]->GetMode() == mode)
        //                ground_Force += body_forces[i]->GetForce() * 1000;
        //        }
        //
        //        printf("Force to the ground is %f,%f,%f \n", ground_Force.x, ground_Force.y, ground_Force.z);
        //        printf("ground_Xforce is %f,%f,%f \n", ground_Xforce.x, ground_Xforce.y, ground_Xforce.z);

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

        if (time > 100)
            break;
    }

    return 0;
}

//--------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------
// Create the objects of the MBD system. Rigid bodies, and if fsi, their
// bce representation are created and added to the systems
//------------------------------------------------------------------
void Create_MB_FE(ChSystemSMC& mphysicalSystem, fsi::ChSystemFsi& myFsiSystem, chrono::fsi::SimParams* paramsH) {
    mphysicalSystem.Set_G_acc(ChVector<>(0, 0, 0));

    Real initSpace0 = paramsH->MULT_INITSPACE * paramsH->HSML;

    auto mysurfmaterial = std::make_shared<ChMaterialSurfaceSMC>();
    // Set common material Properties
    mysurfmaterial->SetKn(2e6);
    mysurfmaterial->SetKt(0);
    mysurfmaterial->SetGn(5.0);
    mysurfmaterial->SetGt(0);
    mphysicalSystem.SetContactForceModel(ChSystemSMC::Hooke);

    auto ground = std::make_shared<ChBody>();
    ground->SetIdentifier(-1);

    ground->SetBodyFixed(true);
    ground->SetCollide(true);
    ground->SetMaterialSurface(mysurfmaterial);

    ground->GetCollisionModel()->ClearModel();

    // Bottom wall
    ChVector<> sizeBottom(bxDim / 2 + 3 * initSpace0, byDim / 2 + 3 * initSpace0, 2 * initSpace0);
    //    ChVector<> sizeBottom(bxDim / 2, byDim / 2, 3 * initSpace0);
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
    mphysicalSystem.Add(ground);

    // auto Indentor = std::make_shared<ChBody>();
    // Indentor->SetBodyFixed(false);
    // Indentor->SetMaterialSurface(mysurfmaterial);

    collision::ChCollisionModel::SetDefaultSuggestedMargin(1.5);  // max inside penetration - if not enough stiffness in
                                                                  // material: troubles

    x0 = fzDim + 2 * initSpace0 + Indentor_R;

    auto Indentor = std::make_shared<ChBodyEasySphere>(0.0032, 7400, false);
    Indentor->SetPos(ChVector<>(0.0, 0.0, x0));
    Indentor->SetMaterialSurface(mysurfmaterial);
    mphysicalSystem.Add(Indentor);

    auto MarkerGround = std::make_shared<ChMarker>();
    auto MarkerIndentor = std::make_shared<ChMarker>();
    ground->AddMarker(MarkerGround);
    Indentor->AddMarker(MarkerIndentor);

    auto my_link = std::make_shared<ChLinkLockLock>();
    my_link->Initialize(MarkerIndentor, MarkerGround);
    mphysicalSystem.AddLink(my_link);

    impose_motion(my_link);

#if haveFluid

    /*================== Walls =================*/
    chrono::fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, ground, posBottom, chrono::QUNIT, sizeBottom);
    //    chrono::fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, ground, posTop, chrono::QUNIT,
    //    sizeBottom);

    chrono::fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, ground, pos_xp, chrono::QUNIT, size_YZ, 23);

    chrono::fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, ground, pos_xn, chrono::QUNIT, size_YZ, 23);

    chrono::fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, ground, pos_yp, chrono::QUNIT, size_XZ, 13);

    chrono::fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, ground, pos_yn, chrono::QUNIT, size_XZ, 13);

    /*================== Flexible-Body =================*/

    GetLog() << "-----------------------------------------------------------\n";
    GetLog() << "-----------------------------------------------------------\n";
    GetLog() << "     Multi-Physics model of articular cartilage    \n";
    GetLog() << "-----------------------------------------------------------\n";

    // Create a mesh, that is a container for groups of elements and their referenced nodes.
    /*================== Flexible-Bodies =================*/
    auto my_mesh = std::make_shared<fea::ChMesh>();

    std::vector<std::vector<int>> NodeNeighborElement;
    std::vector<std::vector<int>> _1D_elementsNodes_mesh;
    std::vector<std::vector<int>> _2D_elementsNodes_mesh;
    int N_x, N_y, N_z, TotalNumElements, TotalNumNodes_onSurface;
    double dx, dy, dz;

    // Geometry of the surface
    double plate_lenght_x = bxDim + 0 * initSpace0;
    double plate_lenght_y = byDim + 0 * initSpace0;
    double plate_lenght_z = 0.0005;
    // Specification of the mesh
    int numDiv_x = 5;
    int numDiv_y = 5;
    int numDiv_z = 1;

    calcSomeParams(numDiv_x, numDiv_y, numDiv_z, plate_lenght_x, plate_lenght_y, plate_lenght_z, N_x, N_y, N_z,
                   TotalNumElements, TotalNumNodes_onSurface, dx, dy, dz);

    std::ofstream Nodal_Constraint;
    Nodal_Constraint.open(demo_dir + "Nodal_Constraint.csv");
    std::string delim = ",";
    Nodal_Constraint << "idx,type,x,y,z\n";
    Nodal_Constraint.precision(4);
    std::vector<std::shared_ptr<ChNodeFEAxyzD>> Constraint_nodes_Shell;
    std::vector<std::shared_ptr<ChNodeFEAxyzD>> Constraint_nodes_fibers;

    if (addCable) {
        int num_Fibers = TotalNumNodes_onSurface;
        double K_Fiber = K_SPRINGS / num_Fibers;
        int numCableElems_Per_Fiber = 8;
        double rho = 1000;
        double Fiber_Diameter = initSpace0 * 2;
        double Fiber_Length = bzDim;

        double Area = 3.1415 * std::pow(Fiber_Diameter, 2) / 4;
        double E = 50e7;
        double nu = 0.3;
        auto mat = std::make_shared<ChMaterialShellANCF>(rho, E, nu);
        /*================== Cable Elements =================*/
        auto msection_cable = std::make_shared<ChBeamSectionCable>();
        msection_cable->SetDiameter(Fiber_Diameter);
        msection_cable->SetYoungModulus(E);
        msection_cable->SetBeamRaleyghDamping(0.05);
        // Create material.
        //        type: 0:Fixed node- 1:Position Constrain- 2:Position+Dir Constraint
        //        std::ofstream fileName_1D_Fixed_nodes;

        for (int Fiber = 0; Fiber < num_Fibers; Fiber++) {
            ChBuilderBeamANCF builder;
            double loc_x = (Fiber % (numDiv_x + 1)) * dx - bxDim / 2 - 0 * initSpace0;
            double loc_y = (Fiber / (numDiv_x + 1)) % (numDiv_y + 1) * dy - byDim / 2 - 0 * initSpace0;
            double loc_z = fzDim + 2 * initSpace0;

            if (std::abs(loc_x) < bxDim / 2 && std::abs(loc_y) < byDim / 2) {
                // Now, simply use BuildBeam to create a beam from a point to another:
                builder.BuildBeam_FSI(
                    my_mesh,                      // the mesh where to put the created nodes and elements
                    msection_cable,               // the ChBeamSectionCable to use for the ChElementBeamANCF elements
                    numCableElems_Per_Fiber,      // the number of ChElementBeamANCF to create
                    ChVector<>(loc_x, loc_y, 0),  // the 'A' point in space (beginning of beam)
                    ChVector<>(loc_x, loc_y, loc_z),  // the 'B' point in space (end of beam) _1D_elementsNodes_mesh,
                    _1D_elementsNodes_mesh, NodeNeighborElementMesh);

                auto Node = std::dynamic_pointer_cast<ChNodeFEAxyzD>(builder.GetLastBeamNodes().back());
                Constraint_nodes_fibers.push_back(Node);
                Nodal_Constraint << Node->GetIndex() << delim << 2 << delim << Node->GetPos().x() << delim
                                 << Node->GetPos().y() << delim << Node->GetPos().z() << std::endl;

                //        // After having used BuildBeam(), you can retrieve the nodes used for the beam,
                //        // For example say you want to fix both pos and dir of A end and apply a force to the B end:
                //        builder.GetLastBeamNodes().back()->SetFixed(true);

                //        auto constraint_hinge = std::make_shared<ChLinkPointFrame>();
                //        constraint_hinge->Initialize(builder.GetLastBeamNodes().back(), mtruss);
                //        mphysicalSystem.Add(constraint_hinge);
            }
        }

        for (int i = 0; i < my_mesh->GetNnodes(); i++) {
            auto node = std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(i));
            if (node->GetPos().z() <= initSpace0 + 1e-6) {
                node->SetFixed(true);
                Nodal_Constraint << node->GetIndex() << delim << 0 << delim << node->GetPos().x() << delim
                                 << node->GetPos().y() << delim << node->GetPos().z() << std::endl;
            }
        }
    }
    printf("after adding fibers to chrono::fea, _1D_elementsNodes_mesh.size()=%d\n", _1D_elementsNodes_mesh.size());

    int currentNodesize = NodeNeighborElementMesh.size();
    int currentElemsize = _1D_elementsNodes_mesh.size();
    numCableNodes = currentNodesize;

    numDiv_x = 20;
    numDiv_y = 20;
    numDiv_z = 1;

    calcSomeParams(numDiv_x, numDiv_y, numDiv_z, plate_lenght_x, plate_lenght_y, plate_lenght_z, N_x, N_y, N_z,
                   TotalNumElements, TotalNumNodes_onSurface, dx, dy, dz);

    printf("currentElemsize is %d\n", currentElemsize);
    _2D_elementsNodes_mesh.resize(TotalNumElements);
    NodeNeighborElementMesh.resize(currentNodesize + TotalNumNodes_onSurface);
    // Create and add the nodes
    for (int i = 0; i < TotalNumNodes_onSurface; i++) {
        // Node location
        double loc_x = (i % (numDiv_x + 1)) * dx - bxDim / 2 - 0 * initSpace0;
        double loc_y = (i / (numDiv_x + 1)) % (numDiv_y + 1) * dy - byDim / 2 - 0 * initSpace0;
        double loc_z = (i) / ((numDiv_x + 1) * (numDiv_y + 1)) * dz + fzDim + 2 * initSpace0;

        // Node direction
        double dir_x = 0;
        double dir_y = 0;
        double dir_z = 1;

        // Create the node
        auto node = std::make_shared<ChNodeFEAxyzD>(ChVector<>(loc_x, loc_y, loc_z), ChVector<>(dir_x, dir_y, dir_z));

        if (std::abs(loc_x) < bxDim / 2 && std::abs(loc_y) < byDim / 2) {
            // look into the the fibers' nodes that need to be constraint;
            // if this node is very close! to one of those nodes constraint it
            for (int fibernodes = 0; fibernodes < Constraint_nodes_fibers.size(); fibernodes++) {
                if ((node->GetPos() - Constraint_nodes_fibers[fibernodes]->GetPos()).Length() < 1e-6)
                    Constraint_nodes_Shell.push_back(node);
            }
        }

        node->SetMass(0);

        //        if (i == 0 || i == numDiv_x || i == (TotalNumNodes - 1) || i == (TotalNumNodes - 1) - numDiv_x) {
        //            node->SetFixed(true);
        //        }

        //        if (i == 0 || i == numDiv_x) {
        //            node->SetFixed(true);
        //        }

        //        if ((i + 1) % (numDiv_x + 1) == 0 || i % (numDiv_x + 1) == 0 || i < (numDiv_x + 1) ||
        //            i >= (TotalNumNodes_onSurface - numDiv_x - 1)) {
        //            auto NodePos = std::make_shared<ChLinkPointFrameGeneral>(ChVector<>(0, 0, 1));
        //            NodePos->Initialize(node, ground);
        //            mphysicalSystem.Add(NodePos);
        //            printf("general constraint node %d is set at %f, %f, %f\n", i, node->GetPos().x, node->GetPos().y,
        //                   node->GetPos().z);
        //        }

        if ((i + 1) % (numDiv_x + 1) == 0 || i % (numDiv_x + 1) == 0 || i < (numDiv_x + 1) ||
            i >= (TotalNumNodes_onSurface - numDiv_x - 1)) {
            auto NodePos = std::make_shared<ChLinkPointFrame>();
            NodePos->Initialize(node, ground);
            rigidLinks.push_back(NodePos);
            mphysicalSystem.Add(NodePos);
            printf("position constraint node %d is set at %f, %f, %f\n", i, node->GetPos().x(), node->GetPos().y(),
                   node->GetPos().z());
            Nodal_Constraint << node->GetIndex() << delim << 1 << delim << node->GetPos().x() << delim
                             << node->GetPos().y() << delim << node->GetPos().z() << std::endl;
        }

#ifndef addPressure
#ifdef addIndentor

        ChVector<> nodePos = node->GetPos();
        if (pow(std::abs(nodePos.x()), 2) + pow(std::abs(nodePos.y()), 2) < pow(Indentor_R / 2, 2) + 1e-6) {
            printf("indentation is applied to node %d pos=(%f, %f, %f)\n", i, nodePos.x(), nodePos.y(), nodePos.z());
            auto NodePos = std::make_shared<ChLinkPointFrame>();
            auto NodeDir = std::make_shared<ChLinkDirFrame>();

            NodePos->Initialize(node, Indentor);
            mphysicalSystem.Add(NodePos);

            NodeDir->Initialize(node, Indentor);
            NodeDir->SetDirectionInAbsoluteCoords(node->D);
            mphysicalSystem.Add(NodeDir);

            Nodal_Constraint << node->GetIndex() << delim << 2 << delim << node->GetPos().x() << delim
                             << node->GetPos().y() << delim << node->GetPos().z() << std::endl;
        }

#endif
#endif

        //        if (abs(loc_x) < bxDim / 4 && abs(loc_y) < byDim / 4) {
        //            node->SetForce(ChVector<>(0, 0, 1));
        //        }

        // Add node to mesh
        my_mesh->AddNode(node);
    }

    ChVector<> mforce;
    for (int iNode = 0; iNode < TotalNumNodes_onSurface; iNode++) {
        auto Node = std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(iNode));
        //            Node->SetForce(ChVector<>(0, 0, 0.1));
        mforce += Node->GetForce();
    }
    printf("GET.FORCES INITIAL=%f\n", mforce.Length());

    if (Constraint_nodes_Shell.size() == Constraint_nodes_fibers.size()) {
        for (int iNode = 0; iNode < Constraint_nodes_fibers.size(); iNode++) {
            auto constr = std::make_shared<ChLinkPointPoint>();
            constr->Initialize(Constraint_nodes_fibers[iNode], Constraint_nodes_Shell[iNode]);
            mphysicalSystem.Add(constr);
            Nodal_Constraint << Constraint_nodes_Shell[iNode]->GetIndex() << delim << 1 << delim
                             << Constraint_nodes_Shell[iNode]->GetPos().x() << delim
                             << Constraint_nodes_Shell[iNode]->GetPos().y() << delim
                             << Constraint_nodes_Shell[iNode]->GetPos().z() << std::endl;
        }
    } else if (addCable) {
        std::cout << "Error! Constraints are not applied correctly\n" << std::endl;
        std::cin.get();
    }

    // Create an orthotropic material.
    // All layers for all elements share the same material.

    double rho = 1000;
    double E = 1.0e7;
    double nu = 0.3;
    auto mat = std::make_shared<ChMaterialShellANCF>(rho, E, nu);

    //    ChVector<> G(8.0769231e6, 8.0769231e6, 8.0769231e6);
    //    auto mat = std::make_shared<ChMaterialShellANCF>(rho, E, nu, G);

    // Create the elements
    for (int i = 0; i < TotalNumElements; i++) {
        // Adjacent nodes
        int node0 = (i / (numDiv_x)) * (N_x) + i % numDiv_x + currentNodesize;
        int node1 = (i / (numDiv_x)) * (N_x) + i % numDiv_x + 1 + currentNodesize;
        int node2 = (i / (numDiv_x)) * (N_x) + i % numDiv_x + 1 + N_x + currentNodesize;
        int node3 = (i / (numDiv_x)) * (N_x) + i % numDiv_x + N_x + currentNodesize;
        _2D_elementsNodes_mesh[i].push_back(node0);
        _2D_elementsNodes_mesh[i].push_back(node1);
        _2D_elementsNodes_mesh[i].push_back(node2);
        _2D_elementsNodes_mesh[i].push_back(node3);
        printf("Adding nodes %d,%d,%d,%d to the shell element %i\n ", node0, node1, node2, node3, i);

        NodeNeighborElementMesh[node0].push_back(i + currentElemsize);
        NodeNeighborElementMesh[node1].push_back(i + currentElemsize);
        NodeNeighborElementMesh[node2].push_back(i + currentElemsize);
        NodeNeighborElementMesh[node3].push_back(i + currentElemsize);
        printf("Adding element %d to the nodes %d,%d,%d,%d\n ", i + currentElemsize, node0, node1, node2, node3);

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
        printf("mat->Get_E(); %f\n ", element->GetLayer(0).GetMaterial()->Get_E());
    }

    // auto mcontactsurf = std::make_shared<ChContactSurfaceMesh>();
    // my_mesh->AddContactSurface(mcontactsurf);
    // mcontactsurf->AddFacesFromBoundary(0.001); // do this after my_mesh->AddContactSurface
    // mcontactsurf->SetMaterialSurface(mysurfmaterial); // use the DEM penalty contacts

    double NODE_AVE_AREA = dx * dy;
    double Tottal_stiff = 0;
    double Tottal_damp = 0;

    my_mesh->SetAutomaticGravity(false);

#ifdef addPressure
    // First: loads must be added to "load containers",
    // and load containers must be added to your ChSystem
    auto Pressureloadcontainer = std::make_shared<ChLoadContainer>();
    // Add constant pressure using ChLoaderPressure (preferred for simple, constant pressure)
    for (int NoElmPre = 0; NoElmPre < my_mesh->GetNelements(); NoElmPre++) {
        if (auto element = std::dynamic_pointer_cast<ChElementShellANCF>(my_mesh->GetElement(NoElmPre))) {
            ChVector<> center = 0.25 * (element->GetNodeA()->GetPos() + element->GetNodeB()->GetPos() +
                                        element->GetNodeC()->GetPos() + element->GetNodeD()->GetPos());

            if ((std::abs(center.x) < bxDim / 5) && (std::abs(center.y) < byDim / 5)) {
                printf("Applying Pressure at the %f, %f, %f\n", center.x, center.y, center.z);

                auto faceload = std::make_shared<ChLoad<ChLoaderPressure>>(
                    std::static_pointer_cast<ChElementShellANCF>(my_mesh->GetElement(NoElmPre)));
                faceload->loader.SetPressure(p0);
                faceload->loader.SetStiff(true);
                faceload->loader.SetIntegrationPoints(2);
                faceload_mesh.push_back(faceload);
                Pressureloadcontainer->Add(faceload);
            }
        }
    }
    mphysicalSystem.Add(Pressureloadcontainer);
#endif
    Nodal_Constraint.close();
    // Add the mesh to the system

    std::vector<std::shared_ptr<chrono::fea::ChElementCableANCF>>* FSI_Cables = myFsiSystem.GetFsiCablesPtr();
    std::vector<std::shared_ptr<chrono::fea::ChElementShellANCF>>* FSI_Shells = myFsiSystem.GetFsiShellsPtr();
    std::vector<std::shared_ptr<chrono::fea::ChNodeFEAxyzD>>* FSI_Nodes = myFsiSystem.GetFsiNodesPtr();

    bool multilayer = true;
    bool removeMiddleLayer = false;
    bool add1DElem = false;
    bool add2DElem = true;
    chrono::fsi::utils::AddBCE_FromMesh(
        myFsiSystem.GetDataManager(), paramsH, my_mesh, FSI_Nodes, FSI_Cables, FSI_Shells, NodeNeighborElementMesh,
        _1D_elementsNodes_mesh, _2D_elementsNodes_mesh, add1DElem, add2DElem, multilayer, removeMiddleLayer, +1, +2);
    myFsiSystem.SetCableElementsNodes(_1D_elementsNodes_mesh);
    myFsiSystem.SetShellElementsNodes(_2D_elementsNodes_mesh);
    myFsiSystem.SetFsiMesh(my_mesh);
    mphysicalSystem.Add(my_mesh);

    writeMesh(my_mesh, MESH_CONNECTIVITY, NodeNeighborElementMesh, _1D_elementsNodes_mesh, _2D_elementsNodes_mesh);

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
    output.open((demo_dir + "/Analysis.txt").c_str(), std::ios::app);

    //    //    thrust::host_vector<Real3> posRadH = myFsiSystem.GetDataManager()->sphMarkersD2.posRadD;
    //    //    thrust::host_vector<Real3> velMasH = myFsiSystem.GetDataManager()->sphMarkersD2.velMasD;

    //    thrust::host_vector<chrono::fsi::Real4> rhoPresMuH = myFsiSystem.GetDataManager()->sphMarkersD2.rhoPresMuD;
    thrust::host_vector<int4> referenceArray = myFsiSystem.GetDataManager()->fsiGeneralData.referenceArray;
    chrono::fsi::ChDeviceUtils fsiUtils;
    fsiUtils.CopyD2H(myFsiSystem.GetDataManager()->sphMarkersD2.rhoPresMuD,
                     myFsiSystem.GetDataManager()->sphMarkersH.rhoPresMuH);
    thrust::host_vector<chrono::fsi::Real4> rhoPresMu = myFsiSystem.GetDataManager()->sphMarkersH.rhoPresMuH;

    Real p_Ave = 0;
    Real Rho_ave = 0;
    Real rho_max = 0;

    for (int i = referenceArray[0].x; i < referenceArray[0].y; i++) {
        //    Real3 pos = posRadH[i];
        if (rhoPresMu[i].x > rho_max)
            rho_max = rhoPresMu[i].x;
        Rho_ave += rhoPresMu[i].x;
        p_Ave += rhoPresMu[i].y;
    }
    p_Ave /= (referenceArray[0].y - referenceArray[0].x);
    Rho_ave /= (referenceArray[0].y - referenceArray[0].x);
    ChVector<> mforces = fxDim * fxDim * p_Ave * 1000;

    //    posRadH.clear();
    //    velMasH.clear();
    rhoPresMu.clear();

    int TotalNumNodes = my_mesh->GetNnodes();
    double delta_Ave;
    double delta_s = 0;
    // x0.resize(TotalNumNodes-numCableNodes);
    for (int iNode = numCableNodes; iNode < TotalNumNodes; iNode++) {
        auto Node = std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(iNode));
        if (saveInitials) {
            x0.push_back(Node->GetPos());
        }
        ChVector<> x_i = x0[iNode - numCableNodes];

        // Only add the forces exerted to nodes at the surface
        //        if (std::abs(Node->GetPos().z) > fzDim)
        //        mforces += Node->GetForce() * 1000;

        if (std::abs(Node->GetPos().x()) < 1e-6 && std::abs(Node->GetPos().y()) < 1e-6)
            delta_s = Node->GetPos().z() - x_i.z();

        double delta = Vdot(Node->GetPos() - x_i, Node->GetD().GetNormalized());
        delta_Ave += delta;
    }
    delta_Ave /= TotalNumNodes;

#ifdef addPressure
    double AppliedPressure = applyRampPressure(time, t_ramp, p_max);

    printf("delta(mm)=%f, applied_Pressure=%f, sigma_s=%f, sigma_f=%f\n", delta_Ave * 1000, AppliedPressure,
           K_SPRINGS * delta_Ave / bxDim / byDim, p_Ave);
    output << time << " " << delta_Ave << " " << AppliedPressure << " " << K_SPRINGS * delta_Ave / bxDim / byDim << " "
           << p_Ave << std::endl;
#endif

#ifndef addPressure
#ifdef addIndentor

    ChVector<> indentor_force(0);

    for (int i = 0; i < rigidLinks.size(); i++) {
        auto frame = rigidLinks[i]->GetLinkAbsoluteCoords();
        indentor_force += rigidLinks[i]->Get_react_force() * 1000 >> frame;
    }

    printf("delta_s(micro m)=%f, ave_compression%= %f, fN_fluid(mN)=(%f,%f,%f), fN_Indentor=(%f,%f,%f)\n",
           delta_s * 1e6, (Rho_ave - 1000) / 10, mforces.x(), mforces.y(), mforces.z(), indentor_force.x(),
           indentor_force.y(), indentor_force.z());
    output << time << " " << delta_s << " " << mforces.z() / 1000 << " " << indentor_force.z() / 1000 << " " << Rho_ave
           << std::endl;
#endif
#endif

    output.close();
}

//------------------------------------------------------------------
// Function to save the povray files of the MBD
//------------------------------------------------------------------
void SaveParaViewFilesMBD(fsi::ChSystemFsi& myFsiSystem,
                          ChSystemSMC& mphysicalSystem,
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
        chrono::fsi::utils::PrintToFile(myFsiSystem.GetDataManager()->sphMarkersD2.posRadD,
                                        myFsiSystem.GetDataManager()->fsiGeneralData.vis_vel_SPH_D,
                                        myFsiSystem.GetDataManager()->sphMarkersD2.rhoPresMuD,
                                        myFsiSystem.GetDataManager()->fsiGeneralData.referenceArray,
                                        myFsiSystem.GetDataManager()->fsiGeneralData.referenceArray_FEA, demo_dir,
                                        true);
#ifdef AddCylinder
        char SaveAsRigidObjVTK[256];  // The filename buffer.
        static int RigidCounter = 0;

        snprintf(SaveAsRigidObjVTK, sizeof(char) * 256, (demo_dir + "/Cylinder.%d.vtk").c_str(), RigidCounter);
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
        snprintf(SaveAsBuffer, sizeof(char) * 256, (demo_dir + "/flex_body.%d.vtk").c_str(), next_frame);
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
               std::vector<std::vector<int>>& NodeNeighborElement,
               std::vector<std::vector<int>> _1D_elementsNodes_mesh,
               std::vector<std::vector<int>> _2D_elementsNodes_mesh) {
    NodeNeighborElement.resize(my_mesh->GetNnodes());

    std::vector<std::shared_ptr<ChNodeFEAbase>> myvector;
    myvector.resize(my_mesh->GetNnodes());
    for (int i = 0; i < my_mesh->GetNnodes(); i++) {
        myvector[i] = std::dynamic_pointer_cast<ChNodeFEAbase>(my_mesh->GetNode(i));
    }
    printf("myvector[i].size()= %d\n\n\n", myvector.size());

    int NUM_1D_NODES = 0;
    //    std::sort(nodeList1D.begin(), nodeList1D.end());
    if (addCable) {
        utils::CSV_writer MESH(" ");
        MESH.stream().setf(std::ios::scientific | std::ios::showpos);
        MESH.stream().precision(6);
        MESH << "\nLINES " << _1D_elementsNodes_mesh.size() << 3 * _1D_elementsNodes_mesh.size() << "\n";

        for (int iele = 0; iele < my_mesh->GetNelements(); iele++) {
            if (auto element = std::dynamic_pointer_cast<ChElementCableANCF>(my_mesh->GetElement(iele))) {
                MESH << "2 ";
                int nodeOrder[] = {0, 1};
                for (int myNodeN = 0; myNodeN < 2; myNodeN++) {
                    auto nodeA = (element->GetNodeN(nodeOrder[myNodeN]));
                    std::vector<std::shared_ptr<ChNodeFEAbase>>::iterator it;
                    it = find(myvector.begin(), myvector.end(), nodeA);
                    if (it == myvector.end()) {
                        // name not in vector
                    } else {
                        auto index = std::distance(myvector.begin(), it);
                        // This is because the index starts from 0
                        if (index > NUM_1D_NODES)
                            NUM_1D_NODES = index + 1;
                        MESH << (unsigned int)index << " ";
                        NodeNeighborElement[index].push_back(iele);
                    }
                }
                MESH << "\n";
            }
        }
        MESH << "\nCELL_DATA " << _1D_elementsNodes_mesh.size() << "\n";

        MESH.write_to_file(SaveAs + "_1D");
    }

    printf("NUM_1D_NODES=%d", NUM_1D_NODES);
    ///////////////////////////////////////////////////////

    utils::CSV_writer MESH(" ");
    MESH.stream().setf(std::ios::scientific | std::ios::showpos);
    MESH.stream().precision(6);
    MESH << "\nCELLS " << _2D_elementsNodes_mesh.size() << 5 * _2D_elementsNodes_mesh.size() << "\n";

    for (int iele = 0; iele < my_mesh->GetNelements(); iele++) {
        if (auto element = std::dynamic_pointer_cast<ChElementShellANCF>(my_mesh->GetElement(iele))) {
            MESH << "4 ";
            int nodeOrder[] = {0, 1, 2, 3};
            for (int myNodeN = 0; myNodeN < 4; myNodeN++) {
                auto nodeA = (element->GetNodeN(nodeOrder[myNodeN]));
                std::vector<std::shared_ptr<ChNodeFEAbase>>::iterator it;
                it = find(myvector.begin(), myvector.end(), nodeA);
                if (it > myvector.end()) {
                    // name not in vector
                } else {
                    auto index = std::distance(myvector.begin(), it);
                    MESH << (unsigned int)(index - NUM_1D_NODES) << " ";
                    NodeNeighborElement[index].push_back(iele);
                }
            }
            MESH << "\n";
        }
    }
    MESH << "\nCELL_TYPES " << _2D_elementsNodes_mesh.size() << "\n";

    for (int iele = 0; iele < _2D_elementsNodes_mesh.size(); iele++) {
        MESH << "9\n";
    }

    MESH.write_to_file(SaveAs + "_2D");
}
////////////////////////////////////////
///////////Write to VTK/////////////////
////////////////////////////////////////
void writeFrame(std::shared_ptr<ChMesh> my_mesh,
                char SaveAsBuffer[256],
                char MeshFileBuffer[256],
                std::vector<std::vector<int>>& NodeNeighborElement) {
    ///////////
    if (addCable) {
        std::vector<int> nodeList;
        /////////First I will find the nodes that are used by a cable element
        for (int i = 0; i < my_mesh->GetNelements(); i++) {
            if (auto element = std::dynamic_pointer_cast<ChElementCableANCF>(my_mesh->GetElement(i))) {
                if (std::find(nodeList.begin(), nodeList.end(), element->GetNodeA()->GetIndex()) == nodeList.end())
                    nodeList.push_back(element->GetNodeA()->GetIndex());
                if (std::find(nodeList.begin(), nodeList.end(), element->GetNodeB()->GetIndex()) == nodeList.end())
                    nodeList.push_back(element->GetNodeB()->GetIndex());
            }
        }
        std::sort(nodeList.begin(), nodeList.end());

        std::string MeshFileBuffer_string(MeshFileBuffer);
        std::ofstream output;
        char MeshFileBuffer_1D[256];
        snprintf(MeshFileBuffer_1D, sizeof(char) * 256, ("%s"), (MeshFileBuffer_string + "_1D").c_str());

        std::string SaveAsBuffer_string(SaveAsBuffer);
        char SaveAsBuffer_1D[256];
        SaveAsBuffer_string.erase(SaveAsBuffer_string.length() - 4, 4);
        cout << SaveAsBuffer_string << endl;
        snprintf(SaveAsBuffer_1D, sizeof(char) * 256, ("%s"), (SaveAsBuffer_string + ".1D.vtk").c_str());
        output.open(SaveAsBuffer_1D, std::ios::app);

        output << "# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET POLYDATA\n" << std::endl;
        output << "POINTS " << nodeList.size() << " float\n";
        for (int i = 0; i < nodeList.size(); i++) {
            auto node = std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(nodeList[i]));
            output << node->GetPos().x() << " " << node->GetPos().y() << " " << node->GetPos().z() << "\n";
        }

        // Later on : if you want to connect to 1D and 2D elements:
        // you have to interpolate between all elements not just 1D
        std::ifstream CopyFrom(MeshFileBuffer_1D);
        output << CopyFrom.rdbuf();
        output << "\nPOINT_DATA " << nodeList.size() << "\n";
        output << "scalars strain float\n";
        output << "LOOKUP_TABLE default\n";
        for (int i = 0; i < nodeList.size(); i++) {
            double areaAve = 0;
            ChVector<> StrainV;
            ChMatrix<> disp;
            double myarea = 0;
            double dx;
            for (int j = 0; j < NodeNeighborElement[nodeList[i]].size(); j++) {
                int myelemInx = NodeNeighborElement[nodeList[i]][j];
                std::dynamic_pointer_cast<ChElementCableANCF>(my_mesh->GetElement(myelemInx))
                    ->EvaluateSectionStrain(0.0, disp, StrainV);
                dx = std::dynamic_pointer_cast<ChElementCableANCF>(my_mesh->GetElement(myelemInx))->GetCurrLength();
                myarea += dx / NodeNeighborElement[nodeList[i]].size();
                areaAve += StrainV.x() * dx / NodeNeighborElement[nodeList[i]].size();
            }

            output << areaAve / myarea << "\n";
        }
        output.close();
    }

    if (true) {
        std::vector<int> nodeList;
        /////////First I will find the nodes that are used by a cable element
        for (int i = 0; i < my_mesh->GetNelements(); i++) {
            if (auto element = std::dynamic_pointer_cast<ChElementShellANCF>(my_mesh->GetElement(i))) {
                if (std::find(nodeList.begin(), nodeList.end(), element->GetNodeA()->GetIndex()) == nodeList.end())
                    nodeList.push_back(element->GetNodeA()->GetIndex());
                if (std::find(nodeList.begin(), nodeList.end(), element->GetNodeB()->GetIndex()) == nodeList.end())
                    nodeList.push_back(element->GetNodeB()->GetIndex());
                if (std::find(nodeList.begin(), nodeList.end(), element->GetNodeC()->GetIndex()) == nodeList.end())
                    nodeList.push_back(element->GetNodeC()->GetIndex());
                if (std::find(nodeList.begin(), nodeList.end(), element->GetNodeD()->GetIndex()) == nodeList.end())
                    nodeList.push_back(element->GetNodeD()->GetIndex());
            }
        }
        std::sort(nodeList.begin(), nodeList.end());

        std::string MeshFileBuffer_string(MeshFileBuffer);
        std::ofstream output;
        char MeshFileBuffer_2D[256];
        snprintf(MeshFileBuffer_2D, sizeof(char) * 256, ("%s"), (MeshFileBuffer_string + "_2D").c_str());

        std::string SaveAsBuffer_string(SaveAsBuffer);
        char SaveAsBuffer_2D[256];
        SaveAsBuffer_string.erase(SaveAsBuffer_string.length() - 4, 4);
        cout << SaveAsBuffer_string << endl;
        snprintf(SaveAsBuffer_2D, sizeof(char) * 256, ("%s"), (SaveAsBuffer_string + ".2D.vtk").c_str());
        output.open(SaveAsBuffer_2D, std::ios::app);

        output << "# vtk DataFile Version 1.0\nUnstructured Grid Example\nASCII\n\n" << std::endl;
        output << "DATASET UNSTRUCTURED_GRID\nPOINTS " << nodeList.size() << " float\n";
        for (int i = 0; i < nodeList.size(); i++) {
            auto node = std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(nodeList[i]));
            output << node->GetPos().x() << " " << node->GetPos().y() << " " << node->GetPos().z() << "\n";
        }

        std::ifstream CopyFrom(MeshFileBuffer_2D);
        output << CopyFrom.rdbuf();

        printf("NodeNeighborElement.size() in writeFrame = %d\n", NodeNeighborElement.size());

        output << "\nPOINT_DATA " << nodeList.size() << "\n";
        output << "SCALARS Deflection float\n";
        output << "LOOKUP_TABLE default\n";
        for (int i = 0; i < nodeList.size(); i++) {
            double areaAve = 0;
            double scalar = 0;
            double myarea = 0;
            double dx, dy;
            for (int j = 0; j < NodeNeighborElement[nodeList[i]].size(); j++) {
                int myelemInx = NodeNeighborElement[nodeList[i]][j];
                std::dynamic_pointer_cast<ChElementShellANCF>(my_mesh->GetElement(myelemInx))
                    ->EvaluateDeflection(scalar);
                dx = std::dynamic_pointer_cast<ChElementShellANCF>(my_mesh->GetElement(myelemInx))->GetLengthX();
                dy = std::dynamic_pointer_cast<ChElementShellANCF>(my_mesh->GetElement(myelemInx))->GetLengthY();
                myarea += dx * dy / NodeNeighborElement[nodeList[i]].size();
                areaAve += scalar * dx * dy / NodeNeighborElement[nodeList[i]].size();
            }

            output << areaAve / myarea << "\n";
        }
        std::vector<ChVector<>> MyResult;
        output << "\nVECTORS ep12_ratio float\n";
        for (unsigned int i = 0; i < nodeList.size(); i++) {
            double areaAve1 = 0, areaAve2 = 0, areaAve3 = 0;
            double myarea = 0;
            double dx, dy;
            for (int j = 0; j < NodeNeighborElement[nodeList[i]].size(); j++) {
                int myelemInx = NodeNeighborElement[nodeList[i]][j];
                MyResult = std::dynamic_pointer_cast<fea::ChElementShellANCF>(my_mesh->GetElement(myelemInx))
                               ->GetPrincipalStrains();
                dx = std::dynamic_pointer_cast<fea::ChElementShellANCF>(my_mesh->GetElement(myelemInx))->GetLengthX();
                dy = std::dynamic_pointer_cast<fea::ChElementShellANCF>(my_mesh->GetElement(myelemInx))->GetLengthY();
                myarea += dx * dy / NodeNeighborElement[nodeList[i]].size();
                areaAve1 += MyResult[0].x() * dx * dy / NodeNeighborElement[nodeList[i]].size();
                areaAve2 += MyResult[0].y() * dx * dy / NodeNeighborElement[nodeList[i]].size();
                if (abs(MyResult[0].x()) > 1e-3 && abs(MyResult[0].y()) > 1e-3) {
                    double ratio = abs(areaAve1 / areaAve2);
                    if (ratio > 10)
                        ratio = 10;
                    if (ratio < 0.1)
                        ratio = 0.1;
                    areaAve3 += log10(ratio) * dx * dy / NodeNeighborElement[nodeList[i]].size();
                } else {
                    areaAve3 += 0.0 * dx * dy / NodeNeighborElement[nodeList[i]].size();
                }
            }
            output << areaAve1 / myarea << " " << areaAve2 / myarea << " " << areaAve3 / myarea << "\n";
        }

        output << "\nVECTORS Position float\n";

        for (int j = 0; j < nodeList.size(); j++) {
            auto node = std::dynamic_pointer_cast<ChNodeFEAxyz>(my_mesh->GetNode(nodeList[j]));
            ChVector<double> pos = node->GetPos();
            output << pos.x() << " " << pos.y() << " " << pos.z() << "\n";
        }

        output << "\nVECTORS E_Princ_Dir1 float\n";
        for (unsigned int i = 0; i < nodeList.size(); i++) {
            double areaAve1 = 0, areaAve2 = 0, areaAve3 = 0;
            double myarea = 0;
            double dx, dy;
            for (int j = 0; j < NodeNeighborElement[nodeList[i]].size(); j++) {
                int myelemInx = NodeNeighborElement[nodeList[i]][j];
                MyResult = std::dynamic_pointer_cast<fea::ChElementShellANCF>(my_mesh->GetElement(myelemInx))
                               ->GetPrincipalStrains();
                dx = std::dynamic_pointer_cast<fea::ChElementShellANCF>(my_mesh->GetElement(myelemInx))->GetLengthX();
                dy = std::dynamic_pointer_cast<fea::ChElementShellANCF>(my_mesh->GetElement(myelemInx))->GetLengthY();
                myarea += dx * dy / NodeNeighborElement[nodeList[i]].size();
                areaAve1 += MyResult[1].x() * dx * dy / NodeNeighborElement[nodeList[i]].size();
                areaAve2 += MyResult[1].y() * dx * dy / NodeNeighborElement[nodeList[i]].size();
                areaAve3 += MyResult[1].z() * dx * dy / NodeNeighborElement[nodeList[i]].size();
            }
            output << areaAve1 / myarea << " " << areaAve2 / myarea << " " << areaAve3 / myarea << "\n";
        }
        output << "\nVECTORS E_Princ_Dir2 float\n";
        for (unsigned int i = 0; i < nodeList.size(); i++) {
            double areaAve1 = 0, areaAve2 = 0, areaAve3 = 0;
            double myarea = 0;
            double dx, dy;
            for (int j = 0; j < NodeNeighborElement[nodeList[i]].size(); j++) {
                int myelemInx = NodeNeighborElement[nodeList[i]][j];
                MyResult = std::dynamic_pointer_cast<fea::ChElementShellANCF>(my_mesh->GetElement(myelemInx))
                               ->GetPrincipalStrains();
                dx = std::dynamic_pointer_cast<fea::ChElementShellANCF>(my_mesh->GetElement(myelemInx))->GetLengthX();
                dy = std::dynamic_pointer_cast<fea::ChElementShellANCF>(my_mesh->GetElement(myelemInx))->GetLengthY();
                myarea += dx * dy / NodeNeighborElement[nodeList[i]].size();
                areaAve1 += MyResult[2].x() * dx * dy / NodeNeighborElement[nodeList[i]].size();
                areaAve2 += MyResult[2].y() * dx * dy / NodeNeighborElement[nodeList[i]].size();
                areaAve3 += MyResult[2].z() * dx * dy / NodeNeighborElement[nodeList[i]].size();
            }
            output << areaAve1 / myarea << " " << areaAve2 / myarea << " " << areaAve3 / myarea << "\n";
        }

        MyResult.clear();

        output << "\nVECTORS sigma12_theta float\n";

        for (unsigned int i = 0; i < nodeList.size(); i++) {
            double areaAve1 = 0, areaAve2 = 0, areaAve3 = 0;
            double myarea = 0;
            double dx, dy;
            for (int j = 0; j < NodeNeighborElement[nodeList[i]].size(); j++) {
                int myelemInx = NodeNeighborElement[nodeList[i]][j];
                ChVector<> test;
                MyResult = std::dynamic_pointer_cast<fea::ChElementShellANCF>(my_mesh->GetElement(myelemInx))
                               ->GetPrincipalStresses();

                dx = std::dynamic_pointer_cast<fea::ChElementShellANCF>(my_mesh->GetElement(myelemInx))->GetLengthX();
                dy = std::dynamic_pointer_cast<fea::ChElementShellANCF>(my_mesh->GetElement(myelemInx))->GetLengthY();
                myarea += dx * dy / NodeNeighborElement[nodeList[i]].size();
                areaAve1 += MyResult[0].x() * dx * dy / NodeNeighborElement[nodeList[i]].size();
                areaAve2 += MyResult[0].y() * dx * dy / NodeNeighborElement[nodeList[i]].size();
                areaAve3 += MyResult[0].z() * dx * dy / NodeNeighborElement[nodeList[i]].size();
            }
            output << areaAve1 / myarea << " " << areaAve2 / myarea << " " << areaAve3 / myarea << "\n";
        }

        output << "\nVECTORS S_Princ_Dir1 float\n";
        for (unsigned int i = 0; i < nodeList.size(); i++) {
            double areaAve1 = 0, areaAve2 = 0, areaAve3 = 0;
            double myarea = 0;
            double dx, dy;
            for (int j = 0; j < NodeNeighborElement[nodeList[i]].size(); j++) {
                int myelemInx = NodeNeighborElement[nodeList[i]][j];
                MyResult = std::dynamic_pointer_cast<fea::ChElementShellANCF>(my_mesh->GetElement(myelemInx))
                               ->GetPrincipalStresses();
                dx = std::dynamic_pointer_cast<fea::ChElementShellANCF>(my_mesh->GetElement(myelemInx))->GetLengthX();
                dy = std::dynamic_pointer_cast<fea::ChElementShellANCF>(my_mesh->GetElement(myelemInx))->GetLengthY();
                myarea += dx * dy / NodeNeighborElement[nodeList[i]].size();
                areaAve1 += MyResult[1].x() * dx * dy / NodeNeighborElement[nodeList[i]].size();
                areaAve2 += MyResult[1].y() * dx * dy / NodeNeighborElement[nodeList[i]].size();
                areaAve3 += MyResult[1].z() * dx * dy / NodeNeighborElement[nodeList[i]].size();
            }
            output << areaAve1 / myarea << " " << areaAve2 / myarea << " " << areaAve3 / myarea << "\n";
        }

        output << "\nVECTORS S_Princ_Dir2 float\n";
        for (unsigned int i = 0; i < nodeList.size(); i++) {
            double areaAve1 = 0, areaAve2 = 0, areaAve3 = 0;
            double myarea = 0;
            double dx, dy;
            for (int j = 0; j < NodeNeighborElement[nodeList[i]].size(); j++) {
                int myelemInx = NodeNeighborElement[nodeList[i]][j];
                MyResult = std::dynamic_pointer_cast<fea::ChElementShellANCF>(my_mesh->GetElement(myelemInx))
                               ->GetPrincipalStresses();
                dx = std::dynamic_pointer_cast<fea::ChElementShellANCF>(my_mesh->GetElement(myelemInx))->GetLengthX();
                dy = std::dynamic_pointer_cast<fea::ChElementShellANCF>(my_mesh->GetElement(myelemInx))->GetLengthY();
                myarea += dx * dy / NodeNeighborElement[nodeList[i]].size();
                areaAve1 += MyResult[2].x() * dx * dy / NodeNeighborElement[nodeList[i]].size();
                areaAve2 += MyResult[2].y() * dx * dy / NodeNeighborElement[nodeList[i]].size();
                areaAve3 += MyResult[2].z() * dx * dy / NodeNeighborElement[nodeList[i]].size();
            }
            output << areaAve1 / myarea << " " << areaAve2 / myarea << " " << areaAve3 / myarea << "\n";
        }

        output.close();
    }
}

void calcSomeParams(const int numDiv_x,
                    const int numDiv_y,
                    const int numDiv_z,
                    const double plate_lenght_x,
                    const double plate_lenght_y,
                    const double plate_lenght_z,
                    int& N_x,
                    int& N_y,
                    int& N_z,
                    int& TotalNumElements,
                    int& TotalNumNodes_onSurface,
                    double& dx,
                    double& dy,
                    double& dz) {
    N_x = numDiv_x + 1;
    N_y = numDiv_y + 1;
    N_z = numDiv_z + 1;
    // Number of elements in the z direction is considered as 1
    TotalNumElements = numDiv_x * numDiv_y;
    TotalNumNodes_onSurface = (numDiv_x + 1) * (numDiv_y + 1);
    // For uniform mesh
    dx = plate_lenght_x / numDiv_x;
    dy = plate_lenght_y / numDiv_y;
    dz = plate_lenght_z / numDiv_z;
}

void impose_motion(std::shared_ptr<ChLinkLockLock> my_link) {
    class z_motion : public ChFunction {
      public:
        z_motion() {}

        virtual z_motion* Clone() const override { return new z_motion(); }

        virtual double Get_y(double x) const override {
            double y;
            y = Indentaiton_rate * x + x0;

            if (Indentaiton_rate * x < -200e-6)
                Indentaiton_rate *= -1;

            if (Indentaiton_rate * x > 0)
                Indentaiton_rate *= -1;

            return (y);
        }
    };

    // printf("\n\n\n\n\n Applying motion_Z with rate: %f\n\n", Indentaiton_rate);
    // printf("test function(1)= %f", mymotion->Get_y(1));

    std::shared_ptr<z_motion> mymotion = std::make_shared<z_motion>();

    my_link->SetMotion_X(std::make_shared<ChFunction_Const>(0));
    my_link->SetMotion_Y(std::make_shared<ChFunction_Const>(0));
    my_link->SetMotion_Z(mymotion);
}
