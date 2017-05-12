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
#include "chrono/motion_functions/ChFunction.h"

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

#include "chrono/physics/ChForce.h"
#include "chrono/physics/ChLoadContainer.h"
#include "chrono/physics/ChLoaderUV.h"
#include "chrono/physics/ChBodyEasy.h"

#include "chrono_fea/ChElementCableANCF.h"
#include "chrono_fea/ChBuilderBeam.h"
#include "chrono_fea/ChElementShellANCF.h"
#include "chrono_fea/ChLinkDirFrame.h"

#include "chrono_fea/ChLinkPointFrameGeneral.h"
#include "chrono_fea/ChLinkPointFrame.h"
#include "chrono_fea/ChLinkPointPoint.h"
#include "chrono_fea/ChNodeFEAxyzD.h"
#include "chrono_fea/ChMesh.h"
#include "chrono_fea/ChContactSurfaceMesh.h"

// FSI Interface Includes
#include "demos/fsi/test_FSI_ValidationBeam.h"  //SetupParamsH()

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
const std::string h_file = "/home/milad/CHRONO/Project-Chrono-Milad-IISPH/src/demos/fsi/test_FSI_ValidationBeam.cpp.h";
const std::string cpp_file = "/home/milad/CHRONO/Project-Chrono-Milad-IISPH/src/demos/fsi/test_FSI_indentation.cpp";

const std::string out_dir = "FSI_OUTPUT";  //"../FSI_OUTPUT";
const std::string data_folder = out_dir + "/test_ValidationBeam/";
std::string MESH_CONNECTIVITY = data_folder + "Flex_MESH.vtk";

std::vector<std::vector<int>> NodeNeighborElementMesh;

bool povray_output = true;
int out_fps = 100;

typedef fsi::Real Real;
Real contact_recovery_speed = 1;  ///< recovery speed for MBD

Real bxDim = 3;
Real byDim = 0.3;
Real bzDim = 3;

Real fxDim = bxDim;
Real fyDim = byDim;
Real fzDim = bzDim;

#define haveFluid 1
double x0 = bzDim;

bool addCable = true;
bool refresh_FlexBodiesInsideFluid = true;

int numCableNodes = 0;
void writeMesh(std::shared_ptr<ChMesh> my_mesh,
               std::string SaveAs,
               std::vector<std::vector<int>>& NodeNeighborElement,
               std::vector<std::vector<int>> _1D_elementsNodes_mesh,
               std::vector<std::vector<int>> _2D_elementsNodes_mesh);
void writeFrame(std::shared_ptr<ChMesh> my_mesh,
                char SaveAsBuffer[256],
                char MeshFileBuffer[256],
                std::vector<std::vector<int>>& NodeNeighborElement);
void saveInputFile(std::string inputFile, std::string outAddress);
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
                          ChSystemDEM& mphysicalSystem,
                          std::shared_ptr<fea::ChMesh> my_mesh,
                          std::vector<std::vector<int>> NodeNeighborElementMesh,
                          chrono::fsi::SimParams* paramsH,
                          int next_frame,
                          double mTime);

void Create_MB_FE(ChSystemDEM& mphysicalSystem, fsi::ChSystemFsi& myFsiSystem, chrono::fsi::SimParams* paramsH);
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

    Real initSpace0 = paramsH->MULT_INITSPACE * paramsH->HSML;
    utils::GridSampler<> sampler(initSpace0);

    chrono::fsi::Real3 boxCenter = chrono::fsi::mR3(0, 0, 0);

    chrono::fsi::Real3 boxHalfDim = chrono::fsi::mR3(fxDim / 2, fyDim / 2, fzDim / 2 - initSpace0);

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
        fsi::Real3 p = fsi::mR3(points[i].x, points[i].y, points[i].z);
        for (int remove = 0; remove < particles_position.size(); remove++) {
            double dist = length(particles_position[remove] - p);
            if (dist < initSpace0 * 0.95) {
                removeThis = true;
                break;
            }
        }
        if (!removeThis) {
            myFsiSystem.GetDataManager()->AddSphMarker(
                p, chrono::fsi::mR3(0.1, 0, 0), chrono::fsi::mR4(paramsH->rho0, paramsH->BASEPRES, paramsH->mu0, -1));
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
    mystepper->SetMaxiters(1000);
    mystepper->SetAbsTolerances(1e-5);
    mystepper->SetMode(ChTimestepperHHT::POSITION);
    mystepper->SetScaling(true);
    mystepper->SetVerbose(false);

    int stepEnd = int(paramsH->tFinal / paramsH->dT);
    stepEnd = 1000000;
    int TotalNumNodes = my_fsi_mesh->GetNnodes();

    SaveParaViewFilesMBD(myFsiSystem, mphysicalSystem, my_fsi_mesh, NodeNeighborElementMesh, paramsH, 0, mTime);
    const std::string copyInitials =
        (std::string("cp ") + data_folder + std::string("/BCE_Flex0.csv") + std::string(" ./ "));
    system(copyInitials.c_str());
    std::vector<ChVector<double>> x0;  // displacement of the nodes
    Real time = 0;
    Real Global_max_dT = paramsH->dT_Max;
    for (int tStep = 0; tStep < stepEnd + 1; tStep++) {
        ChVector<> ground_Force(0);

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

    Real initSpace0 = paramsH->MULT_INITSPACE * paramsH->HSML;

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

    // Bottom wall
    ChVector<> sizeBottom(bxDim / 2, byDim / 2, 2 * initSpace0);
    //    ChVector<> sizeBottom(bxDim / 2, byDim / 2, 3 * initSpace0);
    ChVector<> posBottom(0, 0, -bzDim / 2 - 2 * initSpace0);
    ChVector<> posTop(0, 0, bzDim / 2);

    // left and right Wall
    ChVector<> size_YZ(2 * initSpace0, byDim / 2 + 3 * initSpace0, bzDim / 2);
    ChVector<> pos_xp(bxDim / 2 + initSpace0, 0.0, bzDim / 2 + 1 * initSpace0);
    ChVector<> pos_xn(-bxDim / 2 - 3 * initSpace0, 0.0, bzDim / 2 + 1 * initSpace0);

    // Front and back Wall
    ChVector<> size_XZ(bxDim / 2, 2 * initSpace0, bzDim / 2);
    ChVector<> pos_yp(0, byDim / 2 + initSpace0, bzDim / 2 + 1 * initSpace0);
    ChVector<> pos_yn(0, -byDim / 2 - 3 * initSpace0, bzDim / 2 + 1 * initSpace0);

    chrono::utils::AddBoxGeometry(ground.get(), sizeBottom, posBottom, chrono::QUNIT, true);
    chrono::utils::AddBoxGeometry(ground.get(), sizeBottom, posTop, chrono::QUNIT, true);

    chrono::utils::AddBoxGeometry(ground.get(), size_YZ, pos_xp, chrono::QUNIT, true);
    chrono::utils::AddBoxGeometry(ground.get(), size_YZ, pos_xn, chrono::QUNIT, true);
    chrono::utils::AddBoxGeometry(ground.get(), size_XZ, pos_yp, chrono::QUNIT, true);
    chrono::utils::AddBoxGeometry(ground.get(), size_XZ, pos_yn, chrono::QUNIT, true);
    mphysicalSystem.Add(ground);

    /*================== Walls =================*/
    chrono::fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, ground, posBottom, chrono::QUNIT, sizeBottom);
    chrono::fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, ground, posTop, chrono::QUNIT, sizeBottom);

    //    chrono::fsi::utils::AddBoxBceYZ(myFsiSystem.GetDataManager(), paramsH, ground, pos_xp, chrono::QUNIT,
    //    size_YZ);
    //
    //    chrono::fsi::utils::AddBoxBceYZ(myFsiSystem.GetDataManager(), paramsH, ground, pos_xn, chrono::QUNIT,
    //    size_YZ);

    //    chrono::fsi::utils::AddBoxBceXZ(myFsiSystem.GetDataManager(), paramsH, ground, pos_yp, chrono::QUNIT,
    //    size_XZ);
    //
    //    chrono::fsi::utils::AddBoxBceXZ(myFsiSystem.GetDataManager(), paramsH, ground, pos_yn, chrono::QUNIT,
    //    size_XZ);

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
    int numDiv_x = 1;
    int numDiv_y = 1;
    int numDiv_z = 1;

    calcSomeParams(numDiv_x, numDiv_y, numDiv_z, plate_lenght_x, plate_lenght_y, plate_lenght_z, N_x, N_y, N_z,
                   TotalNumElements, TotalNumNodes_onSurface, dx, dy, dz);
    if (addCable) {
        int num_Fibers = 1;
        int numCableElems_Per_Fiber = 4;
        double rho = 1000;
        double Fiber_Diameter = 0.02;
        double Fiber_Length = bzDim;

        double Area = 3.1415 * std::pow(Fiber_Diameter, 2) / 4;
        double E = 1e5;
        double nu = 0.3;
        auto mat = std::make_shared<ChMaterialShellANCF>(rho, E, nu);
        /*================== Cable Elements =================*/
        auto msection_cable = std::make_shared<ChBeamSectionCable>();
        msection_cable->SetDiameter(Fiber_Diameter);
        msection_cable->SetYoungModulus(E);
        msection_cable->SetBeamRaleyghDamping(0.02);
        // Create material.
        ChBuilderBeamANCF builder;
        double x_pos = -fxDim / 2 + initSpace0 * 30;
        double L = 1.6;
        builder.BuildBeam_FSI(
            my_mesh,                       // the mesh where to put the created nodes and elements
            msection_cable,                // the ChBeamSectionCable to use for the ChElementBeamANCF elements
            numCableElems_Per_Fiber,       // the number of ChElementBeamANCF to create
            ChVector<>(x_pos, 0, -L / 2),  // the 'A' point in space (beginning of beam)
            ChVector<>(x_pos, 0, L / 2),   // the 'B' point in space (end of beam) _1D_elementsNodes_mesh,
            _1D_elementsNodes_mesh, NodeNeighborElementMesh);

        for (int i = 0; i < my_mesh->GetNnodes(); i++) {
            auto node = std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(i));
            if (node->GetPos().z <= -L / 2) {
                node->SetFixed(true);
                printf("nodes %d is set to be fixed %f, %f, %f\n", i, node->GetPos().x, node->GetPos().y,
                       node->GetPos().z);
            }
        }
    }

    auto nodeA = std::make_shared<ChNodeFEAxyzD>(ChVector<>(0, 0, 0), ChVector<>(1, 0, 0));
    my_mesh->AddNode(nodeA);

    printf("after adding fibers to chrono::fea, _1D_elementsNodes_mesh.size()=%d\n", _1D_elementsNodes_mesh.size());

    int currentNodesize = NodeNeighborElementMesh.size();
    int currentElemsize = _1D_elementsNodes_mesh.size();
    numCableNodes = currentNodesize;

    numDiv_x = 10;
    numDiv_y = 10;
    numDiv_z = 1;

    calcSomeParams(numDiv_x, numDiv_y, numDiv_z, plate_lenght_x, plate_lenght_y, plate_lenght_z, N_x, N_y, N_z,
                   TotalNumElements, TotalNumNodes_onSurface, dx, dy, dz);

    printf("currentElemsize is %d\n", currentElemsize);

    // Add the mesh to the system

    std::vector<std::shared_ptr<chrono::fea::ChElementCableANCF>>* FSI_Cables = myFsiSystem.GetFsiCablesPtr();
    std::vector<std::shared_ptr<chrono::fea::ChElementShellANCF>>* FSI_Shells = myFsiSystem.GetFsiShellsPtr();
    std::vector<std::shared_ptr<chrono::fea::ChNodeFEAxyzD>>* FSI_Nodes = myFsiSystem.GetFsiNodesPtr();

    bool multilayer = true;
    bool removeMiddleLayer = false;
    bool add1DElem = true;
    bool add2DElem = true;
    chrono::fsi::utils::AddBCE_FromMesh(
        myFsiSystem.GetDataManager(), paramsH, my_mesh, FSI_Nodes, FSI_Cables, FSI_Shells, NodeNeighborElementMesh,
        _1D_elementsNodes_mesh, _2D_elementsNodes_mesh, add1DElem, add2DElem, multilayer, removeMiddleLayer, +1, +2);
    myFsiSystem.SetCableElementsNodes(_1D_elementsNodes_mesh);
    myFsiSystem.SetShellElementsNodes(_2D_elementsNodes_mesh);
    myFsiSystem.SetFsiMesh(my_mesh);
    mphysicalSystem.Add(my_mesh);

    writeMesh(my_mesh, MESH_CONNECTIVITY, NodeNeighborElementMesh, _1D_elementsNodes_mesh, _2D_elementsNodes_mesh);
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
    Real Rho_ave = 0;
    Real rho_max = 0;

    for (int i = referenceArray[0].x; i < referenceArray[0].y; i++) {
        //    Real3 pos = posRadH[i];
        if (rhoPresMuH[i].x > rho_max)
            rho_max = rhoPresMuH[i].x;
        Rho_ave += rhoPresMuH[i].x;
        p_Ave += rhoPresMuH[i].y;
    }
    p_Ave /= (referenceArray[0].y - referenceArray[0].x);
    Rho_ave /= (referenceArray[0].y - referenceArray[0].x);
    //    posRadH.clear();
    //    velMasH.clear();
    rhoPresMuH.clear();

    int TotalNumNodes = my_mesh->GetNnodes();
    double delta_Ave;
    double delta_s = 0;
    ChVector<> mforces;
    // x0.resize(TotalNumNodes-numCableNodes);
    for (int iNode = numCableNodes; iNode < TotalNumNodes; iNode++) {
        auto Node = std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(iNode));
        if (saveInitials) {
            x0.push_back(Node->GetPos());
        }
        ChVector<> x_i = x0[iNode - numCableNodes];

        // Only add the forces exerted to nodes at the surface
        if (std::abs(Node->GetPos().z) > fzDim)
            mforces += Node->GetForce();

        if (std::abs(Node->GetPos().x) < 1e-6 && std::abs(Node->GetPos().y) < 1e-6)
            delta_s = Node->GetPos().z - x_i.z;

        double delta = Vdot(Node->GetPos() - x_i, Node->GetD().GetNormalized());
        delta_Ave += delta;
    }
    delta_Ave /= TotalNumNodes;

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
            output << node->GetPos().x << " " << node->GetPos().y << " " << node->GetPos().z << "\n";
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
                areaAve += StrainV.x * dx / NodeNeighborElement[nodeList[i]].size();
            }

            output << areaAve / myarea << "\n";
        }
        output.close();
    }

    if (false) {
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
            output << node->GetPos().x << " " << node->GetPos().y << " " << node->GetPos().z << "\n";
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
                areaAve1 += MyResult[0].x * dx * dy / NodeNeighborElement[nodeList[i]].size();
                areaAve2 += MyResult[0].y * dx * dy / NodeNeighborElement[nodeList[i]].size();
                if (abs(MyResult[0].x) > 1e-3 && abs(MyResult[0].y) > 1e-3) {
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
            output << pos.x << " " << pos.y << " " << pos.z << "\n";
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
                areaAve1 += MyResult[1].x * dx * dy / NodeNeighborElement[nodeList[i]].size();
                areaAve2 += MyResult[1].y * dx * dy / NodeNeighborElement[nodeList[i]].size();
                areaAve3 += MyResult[1].z * dx * dy / NodeNeighborElement[nodeList[i]].size();
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
                areaAve1 += MyResult[2].x * dx * dy / NodeNeighborElement[nodeList[i]].size();
                areaAve2 += MyResult[2].y * dx * dy / NodeNeighborElement[nodeList[i]].size();
                areaAve3 += MyResult[2].z * dx * dy / NodeNeighborElement[nodeList[i]].size();
            }
            output << areaAve1 / myarea << " " << areaAve2 / myarea << " " << areaAve3 / myarea << "\n";
        }

        output << "\nVECTORS sigma12_theta float\n";
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
                areaAve1 += MyResult[0].x * dx * dy / NodeNeighborElement[nodeList[i]].size();
                areaAve2 += MyResult[0].y * dx * dy / NodeNeighborElement[nodeList[i]].size();
                areaAve3 += MyResult[0].z * dx * dy / NodeNeighborElement[nodeList[i]].size();
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
                areaAve1 += MyResult[1].x * dx * dy / NodeNeighborElement[nodeList[i]].size();
                areaAve2 += MyResult[1].y * dx * dy / NodeNeighborElement[nodeList[i]].size();
                areaAve3 += MyResult[1].z * dx * dy / NodeNeighborElement[nodeList[i]].size();
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
                areaAve1 += MyResult[2].x * dx * dy / NodeNeighborElement[nodeList[i]].size();
                areaAve2 += MyResult[2].y * dx * dy / NodeNeighborElement[nodeList[i]].size();
                areaAve3 += MyResult[2].z * dx * dy / NodeNeighborElement[nodeList[i]].size();
            }
            output << areaAve1 / myarea << " " << areaAve2 / myarea << " " << areaAve3 / myarea << "\n";
        }

        output.close();
    }
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
