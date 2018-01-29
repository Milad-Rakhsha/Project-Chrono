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
#include <stdlib.h>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "chrono/physics/ChSystemSMC.h"

#include "chrono/core/ChFileutils.h"
#include "chrono/utils/ChUtilsCreators.h"
#include "chrono/utils/ChUtilsGenerators.h"
#include "chrono/utils/ChUtilsGeometry.h"
#include "chrono/utils/ChUtilsInputOutput.h"

// Chrono fsi includes
#include "chrono_fsi/ChDeviceUtils.cuh"
#include "chrono_fsi/ChFsiTypeConvert.h"
#include "chrono_fsi/ChSystemFsi.h"
#include "chrono_fsi/utils/ChUtilsGeneratorFsi.h"
#include "chrono_fsi/utils/ChUtilsPrintSph.cuh"

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
const std::string out_dir = GetChronoOutputPath() + "FSI_COMPRESSIBILITY";
const std::string demo_dir = out_dir + "/Compressibility";
// Save data as csv files, turn it on to be able to see the results off-line using paraview
bool save_output = true;

int out_fps = 20;

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

void SaveParaViewFilesMBD(fsi::ChSystemFsi& myFsiSystem,
                          ChSystemSMC& mphysicalSystem,
                          chrono::fsi::SimParams* paramsH,
                          int tStep,
                          double mTime);

//------------------------------------------------------------------
// Create the objects of the MBD system. Rigid bodies, and if fsi, their
// bce representation are created and added to the systems
//------------------------------------------------------------------

void CreateMbdPhysicalSystemObjects(ChSystemSMC& mphysicalSystem,
                                    fsi::ChSystemFsi& myFsiSystem,
                                    chrono::fsi::SimParams* paramsH) {
    std::shared_ptr<ChMaterialSurfaceNSC> mat_g(new ChMaterialSurfaceNSC);
    auto mysurfmaterial = std::make_shared<ChMaterialSurfaceSMC>();

    // Set common material Properties
    mysurfmaterial->SetYoungModulus(6e4);
    mysurfmaterial->SetFriction(0.3f);
    mysurfmaterial->SetRestitution(0.2f);
    mysurfmaterial->SetAdhesion(0);
    // Ground body
    auto ground = std::make_shared<ChBody>();
    ground->SetIdentifier(-1);
    ground->SetBodyFixed(true);
    ground->SetCollide(true);
    ground->SetMaterialSurface(mysurfmaterial);
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

    chrono::fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, ground, pos_xp, chrono::QUNIT, size_YZ, 23);

    chrono::fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, ground, pos_xn, chrono::QUNIT, size_YZ, 23);

    chrono::fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, ground, pos_yp, chrono::QUNIT, size_XZ, 13);

    chrono::fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, ground, pos_yn, chrono::QUNIT, size_XZ, 13);

#endif
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

    if (ChFileutils::MakeDirectory(demo_dir.c_str()) < 0) {
        cout << "Error creating directory " << demo_dir << endl;
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
    ChSystemSMC mphysicalSystem;
    fsi::ChSystemFsi myFsiSystem(&mphysicalSystem, mHaveFluid, fsi::ChFluidDynamics::Integrator::I2SPH);
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
            chrono::fsi::mR4(points[i].x(), points[i].y(), points[i].z(), paramsH->HSML), chrono::fsi::mR3(1e-10),
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
    const std::string rmCmd = (std::string("rm ") + demo_dir + std::string("/*"));
    system(rmCmd.c_str());
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

    cout << " \n\n\n myFsiSystem.GetDataManager()->SphMarkerDataH.posRadH: "
         << myFsiSystem.GetDataManager()->sphMarkersH.rhoPresMuH[1].w << endl;

    SaveParaViewFilesMBD(myFsiSystem, mphysicalSystem, paramsH, 0, 0);

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

        thrust::host_vector<Real3> velMasH = myFsiSystem.GetDataManager()->sphMarkersH.velMasH;
        thrust::host_vector<Real4> rhoPresMuH = myFsiSystem.GetDataManager()->sphMarkersH.rhoPresMuH;
        std::ofstream output;
        output.open((out_dir + "/Analysis.txt"), std::ios::app);

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

//------------------------------------------------------------------
// Function to save the povray files of the MBD
//------------------------------------------------------------------

void SaveParaViewFilesMBD(fsi::ChSystemFsi& myFsiSystem,
                          ChSystemSMC& mphysicalSystem,
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

    if (save_output && std::abs(mTime - (next_frame)*frame_time) < 0.00001) {
        chrono::fsi::utils::PrintToFile(myFsiSystem.GetDataManager()->sphMarkersD2.posRadD,
                                        myFsiSystem.GetDataManager()->fsiGeneralData.vis_vel_SPH_D,
                                        myFsiSystem.GetDataManager()->sphMarkersD2.rhoPresMuD,
                                        myFsiSystem.GetDataManager()->fsiGeneralData.referenceArray,
                                        thrust::host_vector<int4>(), demo_dir, true);

#ifdef AddCylinder
        char SaveAsRigidObjVTK[256];  // The filename buffer.
        static int RigidCounter = 0;

        snprintf(SaveAsRigidObjVTK, sizeof(char) * 256, (pov_dir_fluid + "/Cylinder.%d.vtk").c_str(), RigidCounter);
        WriteCylinderVTK(Cylinder, cyl_radius, cyl_length, 100, SaveAsRigidObjVTK);

        RigidCounter++;
#endif

        // **** out mbd
        if (next_frame / out_steps == 0) {
            const std::string rmCmd = std::string("rm ") + out_dir + std::string("/*.dat");
            system(rmCmd.c_str());
        }

        char filename[100];
        sprintf(filename, "%s/data_%03d.dat", out_dir.c_str(), out_frame + 1);
        chrono::utils::WriteShapesPovray(&mphysicalSystem, filename);
        cout << "-------------------------------------\n" << endl;
        cout << "             Output frame:   " << next_frame << endl;
        cout << "             Time:           " << mTime << endl;
        cout << "-------------------------------------\n" << endl;

        out_frame++;
    }
}
