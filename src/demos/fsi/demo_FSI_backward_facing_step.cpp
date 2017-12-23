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
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "chrono/physics/ChSystemSMC.h"

//#include "chrono_utils/ChUtilsVehicle.h"
#include "chrono/utils/ChUtilsCreators.h"
#include "chrono/utils/ChUtilsGenerators.h"
#include "chrono/utils/ChUtilsGeometry.h"
#include "chrono/utils/ChUtilsInputOutput.h"

// Chrono general utils
#include "chrono/core/ChFileutils.h"
#include "chrono/core/ChTransform.h"  //transform acc from GF to LF for post process

// Chrono fsi includes
#include "chrono_fsi/ChFsiTypeConvert.h"
#include "chrono_fsi/ChSystemFsi.h"
#include "chrono_fsi/utils/ChUtilsGeneratorFsi.h"
#include "chrono_fsi/utils/ChUtilsPrintSph.cuh"

// FSI Interface Includes
#include "demos/fsi/demo_FSI_backward_facing_step.h"  //SetupParamsH()

#define haveFluid 1
#define AddCylinder 1
// Chrono namespaces
using namespace chrono;
using namespace chrono::collision;

using std::cout;
using std::endl;
std::ofstream simParams;
// =============================================================================
//----------------------------
const std::string out_dir = GetChronoOutputPath() + "backward_facing_step";
const std::string demo_dir = out_dir + "/backward_facing_step";
bool save_output = true;

int out_fps = 10;

typedef fsi::Real Real;
Real contact_recovery_speed = 1;  ///< recovery speed for MBD

Real SCALE = 1.0;
Real bxDim = 2 * SCALE * 8;
Real byDim = 0.3;
Real bzDim = SCALE;
Real step_length = 2 * SCALE * 2;
Real step_ehight = SCALE / 2;

Real fxDim = bxDim;
Real fyDim = byDim;
Real fzDim = bzDim;

void SaveParaViewFilesMBD(fsi::ChSystemFsi& myFsiSystem,
                          ChSystemSMC& mphysicalSystem,
                          chrono::fsi::SimParams* paramsH,
                          int next_frame,
                          double mTime);

//------------------------------------------------------------------
// Create the objects of the MBD system. Rigid bodies, and if fsi, their
// bce representation are created and added to the systems
//------------------------------------------------------------------

void CreateMbdPhysicalSystemObjects(ChSystemSMC& mphysicalSystem,
                                    fsi::ChSystemFsi& myFsiSystem,
                                    chrono::fsi::SimParams* paramsH) {
    mphysicalSystem.Set_G_acc(ChVector<>(0));

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

    // Step Geometry
    ChVector<> size_middle(step_length / 2, byDim / 2, bzDim / 4 - initSpace0);
    ChVector<> pos_middle(0, 0, 7 * initSpace0);
    // Bottom wall
    ChVector<> sizeBottom(bxDim / 2, byDim / 2 - 0 * initSpace0, 2 * initSpace0);
    ChVector<> posBottom(0, 0, 0.0);
    ChVector<> posTop(0, 0, bzDim + 1 * initSpace0);

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
    ground->GetCollisionModel()->BuildModel();
    mphysicalSystem.AddBody(ground);

#if haveFluid
    chrono::fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, ground, posBottom, chrono::QUNIT, sizeBottom);
    chrono::fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, ground, posTop, chrono::QUNIT, sizeBottom);
    chrono::fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, ground, pos_middle, chrono::QUNIT, size_middle,
                                  123);
//    chrono::fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, ground, pos_yp, chrono::QUNIT, size_XZ, 13);
//    chrono::fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, ground, pos_yn, chrono::QUNIT, size_XZ, 13);
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
    cudaSetDevice(1);

    time(&rawtime);
    timeinfo = localtime(&rawtime);

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
    Real initSpace1 = initSpace0;
    utils::GridSampler<> sampler1(initSpace1);

    chrono::fsi::Real3 boxCenter1 = chrono::fsi::mR3(-bxDim / 2 + fxDim / 2, 0, fzDim * 0.5);
    chrono::fsi::Real3 boxHalfDim1 = chrono::fsi::mR3(fxDim / 2, fyDim / 2 + 0.00001, fzDim / 2);

    int numhelperMarkers = myFsiSystem.GetDataManager()->sphMarkersH.posRadH.size();

    // Ghost particles for Variable Resolution SPH
    int numGhostMarkers = 0;

    // Fluid markers
    utils::Generator::PointVector points1 = sampler1.SampleBox(fsi::ChFsiTypeConvert::Real3ToChVector(boxCenter1),
                                                               fsi::ChFsiTypeConvert::Real3ToChVector(boxHalfDim1));

    int numPart1 = points1.size();

    // This section is important to remove the fluid particles that overlap the BCE particles.
    // In order to use this future, two things needs to be done;
    // 1- run the simulation as usual, e.g. ./bin/demo_, which will create the BCE.csv
    // This could be the actual simulation if there is no overlap between the fluid and BCE markers, if there is:
    // 2- run the actual simulation with a command line argument such as ./bin/demo_ 2, which will go over
    // the BCE.csv file and use those information to remove overlap of fluid and BCE markers

    std::vector<fsi::Real3> particles_position;
    if (argc > 1) {
        fsi::Real3 this_particle;

        std::fstream fin("BCE.csv");
        if (!fin.good())
            throw ChException("ERROR opening Mesh file: BCE.csv \n");

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

    int BCE_RIGID_SIZE = particles_position.size();
    for (int i = 0; i < numPart1; i++) {
        particles_position.push_back(fsi::ChFsiTypeConvert::ChVectorToReal3(points1[i]));
    }

    std::cout << "Set Removal points from: BCE.csv" << std::endl;
    std::cout << "Searching among " << particles_position.size() << "BCE points" << std::endl;

    int numremove = 0;
    for (int i = 0; i < numPart1; i++) {
        bool removeThis = false;
        Real h = paramsH->HSML;
        fsi::Real4 p = fsi::mR4(points1[i].x(), points1[i].y(), points1[i].z(), h);
        for (int remove = 0; remove < BCE_RIGID_SIZE; remove++) {
            double dist = length(particles_position[remove] - mR3(p));
            if (dist < 0.9 * initSpace1) {
                removeThis = true;
                break;
            }
        }
        if (!removeThis) {
            myFsiSystem.GetDataManager()->AddSphMarker(p, paramsH->V_in,
                                                       chrono::fsi::mR4(paramsH->rho0, 1e-10, paramsH->mu0, -1.0));
        } else
            numremove++;
    }

    std::cout << "Removed " << numremove << " Fluid particles from the simulation" << std::endl;

    particles_position.clear();

    int numPhases = myFsiSystem.GetDataManager()->fsiGeneralData.referenceArray.size();

    int numFluidPart = myFsiSystem.GetDataManager()->sphMarkersH.posRadH.size();
    std::cout << "Final fluid size:" << numFluidPart << std::endl;

    myFsiSystem.GetDataManager()->fsiGeneralData.referenceArray.push_back(
        mI4(numhelperMarkers + numGhostMarkers, numFluidPart, -1, -1));
    //        myFsiSystem.GetDataManager()->fsiGeneralData.referenceArray.push_back(mI4(numFluidPart, numFluidPart,
    //        0, 0));
    printf("Ref size=%d\n", myFsiSystem.GetDataManager()->fsiGeneralData.referenceArray.size());

#endif

    // ********************** Create Rigid ******************************

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

    // **************** System Initialize ***************************

    double mTime = 0;
    int stepEnd = int(paramsH->tFinal / paramsH->dT);
    stepEnd = 100000;
    std::vector<std::vector<double>> vCoor;
    std::vector<std::vector<int>> faces;
    const std::string rmCmd = (std::string("rm ") + demo_dir + std::string("/*"));
    system(rmCmd.c_str());
    SaveParaViewFilesMBD(myFsiSystem, mphysicalSystem, paramsH, 0, mTime);
    const std::string copyInitials =
        (std::string("cp ") + demo_dir + std::string("/boundary0.csv") + std::string(" ./BCE.csv "));
    system(copyInitials.c_str());
    if (argc <= 1) {
        printf("now please run with an input argument\n");
        return 0;
    }

    Real time = 0;
    Real Global_max_dT = paramsH->dT_Max;
    for (int tStep = 0; tStep < stepEnd + 1; tStep++) {
        //        if (tStep % 2 == 0)
        //            paramsH->dT = paramsH->dT / 2;
        //        else
        //            paramsH->dT = paramsH->dT * 2;

        printf("\nstep : %d, time= : %f (s) \n", tStep, time);
        double frame_time = 1.0 / out_fps;
        int next_frame = std::floor((time + 1e-6) / frame_time) + 1;
        double next_frame_time = next_frame * frame_time;
        double max_allowable_dt = next_frame_time - time;
        if (max_allowable_dt > 1e-6)
            paramsH->dT_Max = std::min(Global_max_dT, max_allowable_dt);
        else
            paramsH->dT_Max = Global_max_dT;

        printf("next_frame is:%d,  max dt is set to %f\n", next_frame, paramsH->dT_Max);
#if haveFluid
        myFsiSystem.DoStepDynamics_FSI_Implicit();
#else
        myFsiSystem.DoStepDynamics_ChronoRK2();
#endif
        time += paramsH->dT;
        SaveParaViewFilesMBD(myFsiSystem, mphysicalSystem, paramsH, next_frame, time);
    }

    return 0;
}

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
        fsi::utils::PrintToFile(myFsiSystem.GetDataManager()->sphMarkersD2.posRadD,
                                myFsiSystem.GetDataManager()->fsiGeneralData.vis_vel_SPH_D,
                                myFsiSystem.GetDataManager()->sphMarkersD2.rhoPresMuD,
                                myFsiSystem.GetDataManager()->fsiGeneralData.referenceArray,
                                myFsiSystem.GetDataManager()->fsiGeneralData.referenceArray_FEA, demo_dir, true);

#ifdef AddCylinder
        char SaveAsRigidObjVTK[256];  // The filename buffer.
        static int RigidCounter = 0;

        snprintf(SaveAsRigidObjVTK, sizeof(char) * 256, (demo_dir + "/Cylinder.%d.vtk").c_str(), RigidCounter);
        RigidCounter++;
#endif

        char filename[100];
        cout << "\n------------ Output frame:   " << next_frame << endl;
        cout << "\n----------------------------\n" << endl;

        out_frame++;
    }
}
