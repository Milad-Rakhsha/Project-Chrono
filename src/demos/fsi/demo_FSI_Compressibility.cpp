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
#include "chrono_parallel/physics/ChSystemParallel.h"

//#include "chrono_utils/ChUtilsVehicle.h"
#include "chrono/utils/ChUtilsCreators.h"
#include "chrono/utils/ChUtilsGenerators.h"
#include "chrono/utils/ChUtilsGeometry.h"
#include "chrono/utils/ChUtilsInputOutput.h"

// Chrono general utils
#include "chrono/core/ChFileutils.h"
#include "chrono/core/ChTransform.h"  //transform acc from GF to LF for post process
#include <thrust/reduce.h>

// Chrono fsi includes
#include "chrono_fsi/ChDeviceUtils.cuh"
#include "chrono_fsi/custom_math.h"

#include "chrono_fsi/ChFsiTypeConvert.h"
#include "chrono_fsi/ChSystemFsi.h"
#include "chrono_fsi/utils/ChUtilsGeneratorFsi.h"
#include "chrono_fsi/utils/ChUtilsPrintSph.h"

// FSI Interface Includes
#include "demos/fsi/demo_FSI_Compressibility.h"  //SetupParamsH()

#define haveFluid 1

// Chrono namespaces
using namespace chrono;
using namespace chrono::collision;
using namespace chrono::fsi;

using std::cout;
using std::endl;
std::ofstream simParams;
// =============================================================================

//----------------------------
// output directories and settings
//----------------------------
const std::string h_file = "/home/milad/CHRONO/Project-Chrono-Milad-IISPH/src/demos/fsi/demo_FSI_Compressibility.h";
const std::string cpp_file = "/home/milad/CHRONO/Project-Chrono-Milad-IISPH/src/demos/fsi/demo_FSI_Compressibility.cpp";

const std::string out_dir = "FSI_OUTPUT";  //"../FSI_OUTPUT";
const std::string pov_dir_fluid = out_dir + "/Compressibility";
const std::string pov_dir_mbd = out_dir + "/povFilesHmmwv";
bool povray_output = true;
int out_fps = 25;

typedef fsi::Real Real;
Real contact_recovery_speed = 1;  ///< recovery speed for MBD

Real hdimX = 14;  // 5.5;
Real hdimY = 0.0;

Real hthick = 1;
Real basinDepth = 2.5;

Real bxDim = 1.0;
Real byDim = 1.0;
Real bzDim = 1.2;

Real fxDim = bxDim;
Real fyDim = byDim;
Real fzDim = 1;

void WriteCylinderVTK(std::shared_ptr<ChBody> Body, double radius, double length, int res, char SaveAsBuffer[256]);
void saveInputFile(std::string inputFile, std::string outAddress);
void SetArgumentsForMbdFromInput(int argc,
                                 char* argv[],
                                 int& threads,
                                 int& max_iteration_sliding,
                                 int& max_iteration_bilateral,
                                 int& max_iteration_normal,
                                 int& max_iteration_spinning);

void InitializeMbdPhysicalSystem(ChSystemParallelDVI& mphysicalSystem, ChVector<> gravity, int argc, char* argv[]);

void SaveParaViewFilesMBD(fsi::ChSystemFsi& myFsiSystem,
                          ChSystemParallelDVI& mphysicalSystem,
                          chrono::fsi::SimParams* paramsH,
                          int tStep,
                          double mTime);

//------------------------------------------------------------------
// Create the objects of the MBD system. Rigid bodies, and if fsi, their
// bce representation are created and added to the systems
//------------------------------------------------------------------

void CreateMbdPhysicalSystemObjects(ChSystemParallelDVI& mphysicalSystem,
                                    fsi::ChSystemFsi& myFsiSystem,
                                    chrono::fsi::SimParams* paramsH) {
    std::shared_ptr<chrono::ChMaterialSurface> mat_g(new chrono::ChMaterialSurface);
    // Set common material Properties
    mat_g->SetFriction(0.8);
    mat_g->SetCohesion(0);
    mat_g->SetCompliance(0.0);
    mat_g->SetComplianceT(0.0);
    mat_g->SetDampingF(0.2);

    // Ground body
    auto ground = std::make_shared<ChBody>(new collision::ChCollisionModelParallel);
    ground->SetIdentifier(-1);

    ground->SetBodyFixed(true);
    ground->SetCollide(true);

    ground->SetMaterialSurface(mat_g);

    ground->GetCollisionModel()->ClearModel();
    Real initSpace0 = paramsH->MULT_INITSPACE * paramsH->HSML;

    ChVector<> sizeBottom(bxDim / 2 + 3 * initSpace0, byDim / 2 + 3 * initSpace0, 2 * initSpace0);
    //    ChVector<> sizeBottom(bxDim / 2, byDim / 2, 3 * initSpace0);
    ChVector<> posBottom(0, 0, -2 * initSpace0);
    ChVector<> posTop(0, 0, bzDim + 2 * initSpace0);

    // left and right Wall
    ChVector<> size_YZ(2 * initSpace0, byDim / 2 + 3 * initSpace0, bzDim / 2);
    ChVector<> pos_xp(bxDim / 2 + initSpace0, 0.0, bzDim / 2 + 1 * initSpace0);
    ChVector<> pos_xn(-bxDim / 2 - 3 * initSpace0, 0.0, bzDim / 2 + 1 * initSpace0);

    // Front and back Wall
    ChVector<> size_XZ(bxDim / 2, 2 * initSpace0, bzDim / 2);
    ChVector<> pos_yp(0, byDim / 2 + initSpace0, bzDim / 2 + 1 * initSpace0);
    ChVector<> pos_yn(0, -byDim / 2 - 3 * initSpace0, bzDim / 2 + 1 * initSpace0);

    chrono::utils::AddBoxGeometry(ground.get(), sizeBottom, posBottom, chrono::QUNIT, true);
    chrono::utils::AddBoxGeometry(ground.get(), size_YZ, pos_xp, chrono::QUNIT, true);
    chrono::utils::AddBoxGeometry(ground.get(), size_YZ, pos_xn, chrono::QUNIT, true);
    chrono::utils::AddBoxGeometry(ground.get(), size_XZ, pos_yp, chrono::QUNIT, true);
    chrono::utils::AddBoxGeometry(ground.get(), size_XZ, pos_yn, chrono::QUNIT, true);
    ground->GetCollisionModel()->BuildModel();
    mphysicalSystem.AddBody(ground);

#if haveFluid

    chrono::fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, ground, posBottom, chrono::QUNIT, sizeBottom);
    // chrono::fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, ground, posTop, chrono::QUNIT, sizeBottom);

    chrono::fsi::utils::AddBoxBceYZ(myFsiSystem.GetDataManager(), paramsH, ground, pos_xp, chrono::QUNIT, size_YZ);

    chrono::fsi::utils::AddBoxBceYZ(myFsiSystem.GetDataManager(), paramsH, ground, pos_xn, chrono::QUNIT, size_YZ);

    chrono::fsi::utils::AddBoxBceXZ(myFsiSystem.GetDataManager(), paramsH, ground, pos_yp, chrono::QUNIT, size_XZ);

    chrono::fsi::utils::AddBoxBceXZ(myFsiSystem.GetDataManager(), paramsH, ground, pos_yn, chrono::QUNIT, size_XZ);

#endif

    // version 0, create one cylinder // note: rigid body initialization should
    // come after boundary initialization

    // **************************
    // **** Test angular velocity
    // **************************

    auto body = std::make_shared<ChBody>(new collision::ChCollisionModelParallel);
    // body->SetIdentifier(-1);
    body->SetBodyFixed(false);
    body->SetCollide(true);
    body->SetMaterialSurface(mat_g);
    body->SetPos(ChVector<>(5, 0, 2));
    body->SetRot(chrono::Q_from_AngAxis(CH_C_PI / 3, VECT_Y) * chrono::Q_from_AngAxis(CH_C_PI / 6, VECT_X) *
                 chrono::Q_from_AngAxis(CH_C_PI / 6, VECT_Z));
    //    body->SetWvel_par(ChVector<>(0, 10, 0));  // Arman : note, SetW should
    //    come after SetRot
    //
    double sphereRad = 0.3;
    double volume = chrono::utils::CalcSphereVolume(sphereRad);
    ChVector<> gyration = chrono::utils::CalcSphereGyration(sphereRad).Get_Diag();
    double density = paramsH->rho0;
    double mass = density * volume;
    body->SetMass(mass);
    body->SetInertiaXX(mass * gyration);
    //
    body->GetCollisionModel()->ClearModel();
    chrono::utils::AddSphereGeometry(body.get(), sphereRad);
    body->GetCollisionModel()->BuildModel();
    //    // *** keep this: how to calculate the velocity of a marker lying on a
    //    rigid body
    //    //
    //    ChVector<> pointRel = ChVector<>(0, 0, 1);
    //    ChVector<> pointPar = pointRel + body->GetPos();
    //    // method 1
    //    ChVector<> l_point = body->Point_World2Body(pointPar);
    //    ChVector<> velvel1 = body->RelPoint_AbsSpeed(l_point);
    //    printf("\n\n\n\n\n\n\n\n\n ***********   velocity1  %f %f %f
    //    \n\n\n\n\n\n\n ", velvel1.x, velvel1.y,
    //    velvel1.z);
    //
    //    // method 2
    //    ChVector<> posLoc = ChTransform<>::TransformParentToLocal(pointPar,
    //    body->GetPos(), body->GetRot());
    //    ChVector<> velvel2 = body->PointSpeedLocalToParent(posLoc);
    //    printf("\n\n\n\n\n\n\n\n\n ***********   velocity 2 %f %f %f
    //    \n\n\n\n\n\n\n ", velvel2.x, velvel2.y,
    //    velvel2.z);
    //
    //    // method 3
    //    ChVector<> velvel3 = body->GetPos_dt() + body->GetWvel_par() % pointRel;
    //    printf("\n\n\n\n\n\n\n\n\n ***********   velocity3  %f %f %f
    //    \n\n\n\n\n\n\n ", velvel3.x, velvel3.y,
    //    velvel3.z);
    //    //

    int numRigidObjects = mphysicalSystem.Get_bodylist()->size();
    mphysicalSystem.AddBody(body);
}

//------------------------------------------------------------------
// Print the simulation parameters: those pre-set and those set from
// command line
//------------------------------------------------------------------

void printSimulationParameters(chrono::fsi::SimParams* paramsH) {
    simParams << " time_pause_fluid_external_force: " << paramsH->timePause << endl
              << " contact_recovery_speed: " << contact_recovery_speed << endl
              << " maxFlowVelocity " << paramsH->v_Max << endl
              << " time_step (paramsH->dT): " << paramsH->dT << endl
              << " time_end: " << paramsH->tFinal << endl;
}

// =============================================================================

int main(int argc, char* argv[]) {
    time_t rawtime;
    struct tm* timeinfo;

    //(void) cudaSetDevice(0);

    time(&rawtime);
    timeinfo = localtime(&rawtime);

    //****************************************************************************************
    // Arman take care of this block.
    // Set path to ChronoVehicle data files
    //  vehicle::SetDataPath(CHRONO_VEHICLE_DATA_DIR);
    //  vehicle::SetDataPath("/home/arman/Repos/GitBeta/chrono/src/demos/data/");
    //  SetChronoDataPath(CHRONO_DATA_DIR);

    // --------------------------
    // Create output directories.
    // --------------------------

    if (ChFileutils::MakeDirectory(out_dir.c_str()) < 0) {
        cout << "Error creating directory " << out_dir << endl;
        return 1;
    }

    if (povray_output) {
        if (ChFileutils::MakeDirectory(pov_dir_mbd.c_str()) < 0) {
            cout << "Error creating directory " << pov_dir_mbd << endl;
            return 1;
        }
    }

    if (ChFileutils::MakeDirectory(pov_dir_fluid.c_str()) < 0) {
        cout << "Error creating directory " << pov_dir_fluid << endl;
        return 1;
    }

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
    ChSystemParallelDVI mphysicalSystem;
    fsi::ChSystemFsi myFsiSystem(&mphysicalSystem, mHaveFluid);
    chrono::ChVector<> CameraLocation = chrono::ChVector<>(0, -10, 0);
    chrono::ChVector<> CameraLookAt = chrono::ChVector<>(0, 0, 0);

    chrono::fsi::SimParams* paramsH = myFsiSystem.GetSimParams();

    SetupParamsH(paramsH, bxDim, byDim, bzDim, fxDim, fyDim, fzDim);
    printSimulationParameters(paramsH);
#if haveFluid
    Real initSpace0 = paramsH->MULT_INITSPACE * paramsH->HSML;
    chrono::utils::GridSampler<> sampler(initSpace0);
    cout << " \n\n\n\nbasinDepth: " << basinDepth << "cMin.z:" << paramsH->cMin.z << endl;

    chrono::fsi::Real3 boxCenter = chrono::fsi::mR3(-bxDim / 2 + fxDim / 2, 0 * initSpace0, fzDim / 2 + 1 * initSpace0);

    chrono::fsi::Real3 boxHalfDim = chrono::fsi::mR3(fxDim / 2, fyDim / 2, fzDim / 2);
    chrono::utils::Generator::PointVector points = sampler.SampleBox(
        fsi::ChFsiTypeConvert::Real3ToChVector(boxCenter), fsi::ChFsiTypeConvert::Real3ToChVector(boxHalfDim));

    int numPart = points.size();

    for (int i = 0; i < numPart; i++) {
        myFsiSystem.GetDataManager()->AddSphMarker(
            chrono::fsi::mR3(points[i].x, points[i].y, points[i].z), chrono::fsi::mR3(0),
            chrono::fsi::mR4(paramsH->rho0, paramsH->BASEPRES, paramsH->mu0, -1));
    }

    int numPhases = myFsiSystem.GetDataManager()->fsiGeneralData.referenceArray.size();  // Arman TODO: either rely on
                                                                                         // pointers, or stack
                                                                                         // entirely, combination of

    if (numPhases != 0) {
        std::cout << "Error! numPhases is wrong, thrown from main\n" << std::endl;
        std::cin.get();
        return -1;
    } else {
        myFsiSystem.GetDataManager()->fsiGeneralData.referenceArray.push_back(
            chrono::fsi::mI4(0, numPart, -1, -1));  // map fluid -1, Arman : this will later be
                                                    // removed, relying on finalize function and
                                                    // automatic sorting
        myFsiSystem.GetDataManager()->fsiGeneralData.referenceArray.push_back(
            chrono::fsi::mI4(numPart, numPart, 0, 0));  // Arman : delete later
    }
#endif

    // ********************** Create Rigid ******************************

    ChVector<> gravity = ChVector<>(paramsH->gravity.x, paramsH->gravity.y, paramsH->gravity.z);

    InitializeMbdPhysicalSystem(mphysicalSystem, gravity, argc, argv);

    // This needs to be called after fluid initialization because I am using
    // "numObjects.numBoundaryMarkers" inside it

    CreateMbdPhysicalSystemObjects(mphysicalSystem, myFsiSystem, paramsH);

    // ******************* Create Interface  *****************

    myFsiSystem.Finalize();

    printf("\n\n sphMarkersH end%d, numAllMarkers is %d \n\n\n",
           myFsiSystem.GetDataManager()->sphMarkersH.posRadH.size(),
           myFsiSystem.GetDataManager()->numObjects.numAllMarkers);

    if (myFsiSystem.GetDataManager()->sphMarkersH.posRadH.size() !=
        myFsiSystem.GetDataManager()->numObjects.numAllMarkers) {
        printf("\n\n\n\n Error! (2) numObjects is not set correctly \n %d, %d \n\n\n",
               myFsiSystem.GetDataManager()->sphMarkersH.posRadH.size(),
               myFsiSystem.GetDataManager()->numObjects.numAllMarkers);
        return -1;
    }
    //*** Add sph data to the physics system

    cout << " -- ChSystem size : " << mphysicalSystem.Get_bodylist()->size() << endl;

    // ***************************** System Initialize
    // ********************************************
    myFsiSystem.InitializeChronoGraphics(CameraLocation, CameraLookAt);

    double mTime = 0;

#ifdef CHRONO_FSI_USE_DOUBLE
    printf("Double Precision\n");
#else
    printf("Single Precision\n");
#endif
    int stepEnd = int(paramsH->tFinal / paramsH->dT);
    stepEnd = 5000000;
    std::vector<std::vector<double>> vCoor;
    std::vector<std::vector<int>> faces;
    std::string RigidConectivity = pov_dir_fluid + "RigidConectivity.vtk";

#ifdef AddCylinder
    std::vector<std::shared_ptr<ChBody>>* FSI_Bodies = (myFsiSystem.GetFsiBodiesPtr());
    auto Cylinder = ((*FSI_Bodies)[0]);
#else
    std::shared_ptr<ChBody> Cylinder;
#endif

    cout << " \n\n\n myFsiSystem.GetDataManager()->SphMarkerDataH.posRadH: "
         << myFsiSystem.GetDataManager()->sphMarkersH.rhoPresMuH[1].w << endl;

    SaveParaViewFilesMBD(myFsiSystem, mphysicalSystem, paramsH, 0, 0);
    const std::string rmCmd = (std::string("rm ") + pov_dir_fluid + std::string("/*"));
    system(rmCmd.c_str());
    saveInputFile(h_file, pov_dir_fluid + "/hfile.h");
    saveInputFile(cpp_file, pov_dir_fluid + "/cppfile.cpp");

    Real time = 0;
    Real Global_max_dT = paramsH->dT_Max;
    for (int tStep = 0; tStep < stepEnd + 1; tStep++) {
        printf("\nstep : %d, time= : %f (s) \n", tStep, time);
        double frame_time = 1.0 / out_fps;
        int next_frame = std::floor((time + 1e-6) / frame_time) + 1;
        double next_frame_time = next_frame * frame_time;
        double max_allowable_dt = next_frame_time - time;
        if (max_allowable_dt > 1e-7)
            paramsH->dT_Max = std::min(Global_max_dT, max_allowable_dt);
        else
            paramsH->dT_Max = Global_max_dT;

#if haveFluid
        myFsiSystem.DoStepDynamics_FSI_Implicit();
#else
        myFsiSystem.DoStepDynamics_ChronoRK2();
#endif

        thrust::host_vector<Real3> velMasH = myFsiSystem.GetDataManager()->sphMarkersD2.velMasD;
        thrust::host_vector<Real4> rhoPresMuH = myFsiSystem.GetDataManager()->sphMarkersD2.rhoPresMuD;
        std::ofstream output;
        output.open((pov_dir_fluid + "/Analysis.txt"), std::ios::app);

        Real KE = 0;
        Real Rho = 0;
        int numFluidMarkers = myFsiSystem.GetDataManager()->fsiGeneralData.referenceArray[0].y;
        for (int i = 0; i < numFluidMarkers; i++) {
            KE += 0.5 * (length(velMasH[i]));
            Rho += rhoPresMuH[i].x;
        }
        output << time << " " << Rho / numFluidMarkers << " " << paramsH->markerMass * KE / numFluidMarkers << endl;
        output.close();
        time += paramsH->dT;

        SaveParaViewFilesMBD(myFsiSystem, mphysicalSystem, paramsH, next_frame, time);

        if (time > paramsH->tFinal)
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
//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

// void WriteCylinder(std::shared_ptr<ChBody> Body, double radius, double length,
//		int res, char SaveAsBuffer[256]) {
//
//	std::ofstream output;
//	output.open(SaveAsBuffer, std::ios::app);
//
//	ChVector<> center = Body->GetPos();
//	ChMatrix33<> Rotation = Body->GetRot();
//
//    ChVector<double> vertex;
//	for (int i = 0; i <  res; i++) {
//		ChVector<double> thisNode;
//		thisNode.x = radius*cos(2*i*3.1415/res);
//		thisNode.y = -1*length/2;
//		thisNode.z = radius*sin(2*i*3.1415/res);
//		vertex = Rotation * thisNode + center;  // rotate/scale, if needed
//		output << "v " << vertex.x << " " << vertex.y << " " << vertex.z
//				<< "\n";
//	}
//
//	for (int i = 0; i <  res; i++) {
//		ChVector<double> thisNode;
//		thisNode.x = radius*cos(2*i*3.1415/res);
//		thisNode.y = +1*length/2;
//		thisNode.z = radius*sin(2*i*3.1415/res);
//		vertex = Rotation * thisNode + center;  // rotate/scale, if needed
//		output << "v " << vertex.x << " " << vertex.y << " " << vertex.z
//				<< "\n";
//	}
//
//	for (int i = 1; i <=res; i++) {
//		output << "f " << i << " " << i+1 << " " << i+res << " " << i+res-1
//				<< "\n";
//	}
//
//	output << "f ";
//	for (int i = 1; i <= res; i++) {
//		output << i << " " ;
//	}
//
//	output << "\nf ";
//	for (int i = res+1; i <= 2*res; i++) {
//		output << i << " " ;
//	}
//
//
//
//	output.close();
//}

void WriteCylinderVTK(std::shared_ptr<ChBody> Body, double radius, double length, int res, char SaveAsBuffer[256]) {
    std::ofstream output;
    output.open(SaveAsBuffer, std::ios::app);
    output << "# vtk DataFile Version 1.0\nUnstructured Grid Example\nASCII\n\n" << std::endl;
    output << "DATASET UNSTRUCTURED_GRID\nPOINTS " << 2 * res << " float\n";

    ChVector<> center = Body->GetPos();
    ChMatrix33<> Rotation = Body->GetRot();
    ChVector<double> vertex;
    for (int i = 0; i < res; i++) {
        ChVector<double> thisNode;
        thisNode.x = radius * cos(2 * i * 3.1415 / res);
        thisNode.y = -1 * length / 2;
        thisNode.z = radius * sin(2 * i * 3.1415 / res);
        vertex = Rotation * thisNode + center;  // rotate/scale, if needed
        output << vertex.x << " " << vertex.y << " " << vertex.z << "\n";
    }

    for (int i = 0; i < res; i++) {
        ChVector<double> thisNode;
        thisNode.x = radius * cos(2 * i * 3.1415 / res);
        thisNode.y = +1 * length / 2;
        thisNode.z = radius * sin(2 * i * 3.1415 / res);
        vertex = Rotation * thisNode + center;  // rotate/scale, if needed
        output << vertex.x << " " << vertex.y << " " << vertex.z << "\n";
    }

    output << "\n\nCELLS " << (unsigned int)res + res << "\t" << (unsigned int)5 * (res + res) << "\n";

    for (int i = 0; i < res - 1; i++) {
        output << "4 " << i << " " << i + 1 << " " << i + res + 1 << " " << i + res << "\n";
    }
    output << "4 " << res - 1 << " " << 0 << " " << res << " " << 2 * res - 1 << "\n";

    for (int i = 0; i < res / 4; i++) {
        output << "4 " << i << " " << i + 1 << " " << +res / 2 - i - 1 << " " << +res / 2 - i << "\n";
    }

    for (int i = 0; i < res / 4; i++) {
        output << "4 " << i + res << " " << i + 1 + res << " " << +res / 2 - i - 1 + res << " " << +res / 2 - i + res
               << "\n";
    }

    output << "4 " << +res / 2 << " " << 1 + res / 2 << " " << +res - 1 << " " << 0 << "\n";

    for (int i = 1; i < res / 4; i++) {
        output << "4 " << i + res / 2 << " " << i + 1 + res / 2 << " " << +res / 2 - i - 1 + res / 2 << " "
               << +res / 2 - i + res / 2 << "\n";
    }

    output << "4 " << 3 * res / 2 << " " << 1 + 3 * res / 2 << " " << +2 * res - 1 << " " << +res << "\n";

    for (int i = 1; i < res / 4; i++) {
        output << "4 " << i + 3 * res / 2 << " " << i + 1 + 3 * res / 2 << " " << +2 * res - i - 1 << " "
               << +2 * res - i << "\n";
    }

    output << "\nCELL_TYPES " << res + res << "\n";

    for (int iele = 0; iele < (res + res); iele++) {
        output << "9\n";
    }
}

//------------------------------------------------------------------
// Function to save the povray files of the MBD
//------------------------------------------------------------------

void SaveParaViewFilesMBD(fsi::ChSystemFsi& myFsiSystem,
                          ChSystemParallelDVI& mphysicalSystem,
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

    if (povray_output && std::abs(mTime - (next_frame)*frame_time) < 0.0001) {
        // **** out fluid

        chrono::fsi::utils::PrintToParaViewFile(
            myFsiSystem.GetDataManager()->sphMarkersD2.posRadD, myFsiSystem.GetDataManager()->sphMarkersD2.velMasD,
            myFsiSystem.GetDataManager()->sphMarkersD2.rhoPresMuD,
            myFsiSystem.GetDataManager()->fsiGeneralData.referenceArray,
            myFsiSystem.GetDataManager()->fsiGeneralData.referenceArray_FEA, pov_dir_fluid);

#ifdef AddCylinder
        char SaveAsRigidObjVTK[256];  // The filename buffer.
        static int RigidCounter = 0;

        snprintf(SaveAsRigidObjVTK, sizeof(char) * 256, (pov_dir_fluid + "/Cylinder.%d.vtk").c_str(), RigidCounter);
        WriteCylinderVTK(Cylinder, cyl_radius, cyl_length, 100, SaveAsRigidObjVTK);

        RigidCounter++;
#endif

        // **** out mbd
        if (next_frame / out_steps == 0) {
            const std::string rmCmd = std::string("rm ") + pov_dir_mbd + std::string("/*.dat");
            system(rmCmd.c_str());
        }

        char filename[100];
        sprintf(filename, "%s/data_%03d.dat", pov_dir_mbd.c_str(), out_frame + 1);
        chrono::utils::WriteShapesPovray(&mphysicalSystem, filename);
        cout << "-------------------------------------\n" << endl;
        cout << "             Output frame:   " << next_frame << endl;
        cout << "             Time:           " << mTime << endl;
        cout << "-------------------------------------\n" << endl;

        out_frame++;
    }
}
//------------------------------------------------------------------
// function to set some simulation settings from command line
//------------------------------------------------------------------
void SetArgumentsForMbdFromInput(int argc,
                                 char* argv[],
                                 int& threads,
                                 int& max_iteration_sliding,
                                 int& max_iteration_bilateral,
                                 int& max_iteration_normal,
                                 int& max_iteration_spinning) {
    if (argc > 1) {
        const char* text = argv[1];
        threads = atoi(text);
    }
    if (argc > 2) {
        const char* text = argv[2];
        max_iteration_sliding = atoi(text);
    }
    if (argc > 3) {
        const char* text = argv[3];
        max_iteration_bilateral = atoi(text);
    }
    if (argc > 4) {
        const char* text = argv[4];
        max_iteration_normal = atoi(text);
    }
    if (argc > 5) {
        const char* text = argv[5];
        max_iteration_spinning = atoi(text);
    }
}

//------------------------------------------------------------------
// function to set the solver setting for the
//------------------------------------------------------------------

void InitializeMbdPhysicalSystem(ChSystemParallelDVI& mphysicalSystem, ChVector<> gravity, int argc, char* argv[]) {
    // Desired number of OpenMP threads (will be clamped to maximum available)
    int threads = 1;
    // Perform dynamic tuning of number of threads?
    bool thread_tuning = true;

    //	uint max_iteration = 20;//10000;
    int max_iteration_normal = 0;
    int max_iteration_sliding = 200;
    int max_iteration_spinning = 0;
    int max_iteration_bilateral = 100;

    // ----------------------
    // Set params from input
    // ----------------------

    SetArgumentsForMbdFromInput(argc, argv, threads, max_iteration_sliding, max_iteration_bilateral,
                                max_iteration_normal, max_iteration_spinning);

    // ----------------------
    // Set number of threads.
    // ----------------------

    //  omp_get_num_procs();
    int max_threads = omp_get_num_procs();
    if (threads > max_threads)
        threads = max_threads;
    mphysicalSystem.SetParallelThreadNumber(threads);
    omp_set_num_threads(threads);
    cout << "Using " << threads << " threads" << endl;

    mphysicalSystem.GetSettings()->perform_thread_tuning = thread_tuning;
    mphysicalSystem.GetSettings()->min_threads = std::max(1, threads / 2);
    mphysicalSystem.GetSettings()->max_threads = int(3.0 * threads / 2);

    // ---------------------
    // Print the rest of parameters
    // ---------------------

    simParams << endl
              << " number of threads: " << threads << endl
              << " max_iteration_normal: " << max_iteration_normal << endl
              << " max_iteration_sliding: " << max_iteration_sliding << endl
              << " max_iteration_spinning: " << max_iteration_spinning << endl
              << " max_iteration_bilateral: " << max_iteration_bilateral << endl
              << endl;

    // ---------------------
    // Edit mphysicalSystem settings.
    // ---------------------

    double tolerance = 0.1;  // 1e-3;  // Arman, move it to paramsH
    // double collisionEnvelop = 0.04 * paramsH->HSML;
    mphysicalSystem.Set_G_acc(gravity);
    printf("g(m/s):%f inside\n", mphysicalSystem.Get_G_acc().z);

    mphysicalSystem.GetSettings()->solver.solver_mode = SLIDING;                              // NORMAL, SPINNING
    mphysicalSystem.GetSettings()->solver.max_iteration_normal = max_iteration_normal;        // max_iteration / 3
    mphysicalSystem.GetSettings()->solver.max_iteration_sliding = max_iteration_sliding;      // max_iteration / 3
    mphysicalSystem.GetSettings()->solver.max_iteration_spinning = max_iteration_spinning;    // 0
    mphysicalSystem.GetSettings()->solver.max_iteration_bilateral = max_iteration_bilateral;  // max_iteration / 3
    mphysicalSystem.GetSettings()->solver.use_full_inertia_tensor = true;
    mphysicalSystem.GetSettings()->solver.tolerance = tolerance;
    mphysicalSystem.GetSettings()->solver.alpha = 0;  // Arman, find out what is this
    mphysicalSystem.GetSettings()->solver.contact_recovery_speed = contact_recovery_speed;
    mphysicalSystem.ChangeSolverType(APGD);  // Arman check this APGD APGDBLAZE
    //  mphysicalSystem.GetSettings()->collision.narrowphase_algorithm =
    //  NARROWPHASE_HYBRID_MPR;

    //    mphysicalSystem.GetSettings()->collision.collision_envelope =
    //    collisionEnvelop;   // global collisionEnvelop
    //    does not work. Maybe due to sph-tire size mismatch
    mphysicalSystem.GetSettings()->collision.bins_per_axis = vec3(40, 40, 40);  // Arman check
}

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