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
#include "chrono_fsi/ChDeviceUtils.cuh"
#include "chrono_fsi/ChFluidDynamics.cuh"

#include "chrono_fsi/ChFsiTypeConvert.h"
#include "chrono_fsi/ChSystemFsi.h"
#include "chrono_fsi/utils/ChUtilsGeneratorFsi.h"
#include "chrono_fsi/utils/ChUtilsPrintSph.h"

// FSI Interface Includes
#include "demos/fsi/demo_FSI_flow_around_cylinder.h"  //SetupParamsH()

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
const std::string out_dir = GetChronoOutputPath() + "FSI_FLOW_AROUND_CYLINDER";
const std::string demo_dir = out_dir + "/FlowAroundCylinder";
bool save_output = true;

int out_fps = 200;

typedef fsi::Real Real;
Real contact_recovery_speed = 1;  ///< recovery speed for MBD

Real bxDim = 1.0;
Real byDim = 0.08;
Real bzDim = 0.40;

Real fxDim = bxDim;
Real fyDim = byDim;
Real fzDim = bzDim;
double cyl_length = 0.11;
double cyl_radius = .05;
ChVector<> cyl_pos = ChVector<>(0.0);

void WriteCylinderVTK(std::shared_ptr<ChBody> Body, double radius, double length, int res, char SaveAsBuffer[256]);
void InitializeMbdPhysicalSystem(ChSystemSMC& mphysicalSystem, ChVector<> gravity, int argc, char* argv[]);
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

    // Bottom wall
    ChVector<> sizeBottom(bxDim / 2, byDim / 2 + 3 * paramsH->HSML, 2 * paramsH->HSML);
    ChVector<> posBottom(0, 0, -2 * paramsH->HSML);
    ChVector<> posTop(0, 0, bzDim + 2 * paramsH->HSML);

    // left and right Wall
    ChVector<> size_YZ(2 * paramsH->HSML, byDim / 2 + 3 * paramsH->HSML, bzDim / 2);
    ChVector<> pos_xp(bxDim / 2 + paramsH->HSML, 0.0, bzDim / 2 + 1 * paramsH->HSML);
    ChVector<> pos_xn(-bxDim / 2 - 3 * paramsH->HSML, 0.0, bzDim / 2 + 1 * paramsH->HSML);

    // Front and back Wall
    ChVector<> size_XZ(bxDim / 2, 2 * paramsH->HSML, bzDim / 2);
    ChVector<> pos_yp(0, byDim / 2 + paramsH->HSML, bzDim / 2 + 1 * paramsH->HSML);
    ChVector<> pos_yn(0, -byDim / 2 - 3 * paramsH->HSML, bzDim / 2 + 1 * paramsH->HSML);

    chrono::utils::AddBoxGeometry(ground.get(), sizeBottom, posBottom, chrono::QUNIT, true);
    //    chrono::utils::AddBoxGeometry(ground.get(), size_YZ, pos_xp, chrono::QUNIT, true);
    //    chrono::utils::AddBoxGeometry(ground.get(), size_YZ, pos_xn, chrono::QUNIT, true);
    //    chrono::utils::AddBoxGeometry(ground.get(), size_XZ, pos_yp, chrono::QUNIT, true);
    //    chrono::utils::AddBoxGeometry(ground.get(), size_XZ, pos_yn, chrono::QUNIT, true);
    ground->GetCollisionModel()->BuildModel();
    mphysicalSystem.AddBody(ground);

#if haveFluid
    printf("Ref size=%d\n", myFsiSystem.GetDataManager()->fsiGeneralData.referenceArray.size());

    chrono::fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, ground, posBottom, chrono::QUNIT, sizeBottom);
    chrono::fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, ground, posTop, chrono::QUNIT, sizeBottom);
    chrono::fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, ground, pos_yp, chrono::QUNIT, size_XZ, 13);
    chrono::fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, ground, pos_yn, chrono::QUNIT, size_XZ, 13);
#endif

    // Add falling cylinder
    ChQuaternion<> cyl_rot = chrono::QUNIT;
    std::vector<std::shared_ptr<ChBody>>* FSI_Bodies = myFsiSystem.GetFsiBodiesPtr();

    auto body = std::make_shared<ChBody>();
    // body->SetIdentifier(-1);
    body->SetBodyFixed(true);
    body->SetCollide(true);
    body->SetMaterialSurface(mysurfmaterial);

    body->SetPos(cyl_pos);

    double sphereRad = 0.3;
    double volume = utils::CalcSphereVolume(sphereRad);
    ChVector<> gyration = utils::CalcSphereGyration(sphereRad).Get_Diag();
    double density = paramsH->rho0;
    double mass = density * volume;
    body->SetMass(mass);
    body->SetInertiaXX(mass * gyration);
    body->GetCollisionModel()->ClearModel();
    utils::AddSphereGeometry(body.get(), sphereRad);
    body->GetCollisionModel()->BuildModel();
    int numRigidObjects = mphysicalSystem.Get_bodylist()->size();
    mphysicalSystem.AddBody(body);

    std::vector<std::shared_ptr<ChBody>>* fsiBodeisPtr = myFsiSystem.GetFsiBodiesPtr();
    fsiBodeisPtr->push_back(body);
    chrono::fsi::utils::AddCylinderBce(myFsiSystem.GetDataManager(), paramsH, body, ChVector<>(0, 0, 0),
                                       ChQuaternion<>(1, 0, 0, 0), cyl_radius, cyl_length, paramsH->HSML / 2, false);
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
    fsi::ChSystemFsi myFsiSystem(&mphysicalSystem, mHaveFluid);
    chrono::ChVector<> CameraLocation = chrono::ChVector<>(0, -10, 0);
    chrono::ChVector<> CameraLookAt = chrono::ChVector<>(0, 0, 0);

    chrono::fsi::SimParams* paramsH = myFsiSystem.GetSimParams();

    SetupParamsH(paramsH, bxDim, byDim, bzDim, fxDim, fyDim, fzDim);
    printSimulationParameters(paramsH);
#if haveFluid

    Real initSpace0 = paramsH->MULT_INITSPACE * paramsH->HSML;
    Real initSpace1 = initSpace0 / 2;
    Real initSpace2 = initSpace0;
    utils::GridSampler<> sampler1(initSpace1);
    utils::GridSampler<> sampler2(initSpace2);

    chrono::fsi::Real3 boxCenter1 = chrono::fsi::mR3(-bxDim / 2 + fxDim / 2 - paramsH->HSML / 3, +paramsH->HSML / 12,
                                                     fzDim * 0.5 - paramsH->HSML / 3);

    chrono::fsi::Real3 boxCenter2 =
        chrono::fsi::mR3(-bxDim / 2 + fxDim / 2, 0 * paramsH->HSML, fzDim * 0.5 + 1 * paramsH->HSML);

    chrono::fsi::Real3 boxHalfDim1 = chrono::fsi::mR3(fxDim / 6, fyDim / 3, fzDim / 6);
    chrono::fsi::Real3 boxHalfDim2 = chrono::fsi::mR3(fxDim / 2, fyDim / 2, fzDim / 2);
    auto mbody = std::make_shared<ChBody>();
    mbody->SetBodyFixed(true);

    cyl_pos = ChVector<>(-paramsH->HSML / 2, paramsH->HSML, 0.2 - paramsH->HSML / 2);
    mbody->SetPos(cyl_pos);

    // Here we add a particle to the surface of the cylinder for merging purposes
    chrono::fsi::utils::AddCylinderSurfaceBce(myFsiSystem.GetDataManager(), paramsH, mbody, ChVector<>(0, 0, 0),
                                              ChQuaternion<>(1, 0, 0, 0), 0.12, 0.20, paramsH->HSML * sqrt(6));
    //    chrono::fsi::utils::AddSphereSurfaceBce(myFsiSystem.GetDataManager(), paramsH, mbody, ChVector<>(0, 0, 0),
    //                                            ChQuaternion<>(1, 0, 0, 0), 0.07, paramsH->HSML);

    cyl_pos = ChVector<>(-paramsH->HSML / 2, 0, 0.2 - paramsH->HSML / 2);
    mbody->SetPos(cyl_pos);
    int numhelperMarkers = myFsiSystem.GetDataManager()->sphMarkersH.posRadH.size();

    // Ghost particles for Variable Resolution SPH
    int numGhostMarkers = 10000;
    for (int i = 0; i < numGhostMarkers; i++) {
        myFsiSystem.GetDataManager()->AddSphMarker(chrono::fsi::mR4(0.0, 0.0, -0.4, 0.0),
                                                   chrono::fsi::mR3(1e-10, 0.0, 0.0),
                                                   chrono::fsi::mR4(paramsH->rho0, 1e-10, paramsH->mu0, -2.0));
    }

    utils::Generator::PointVector points1 =
        sampler1.SampleCylinderY(fsi::ChFsiTypeConvert::Real3ToChVector(boxCenter1), 0.12, 0.04);

    utils::Generator::PointVector points2 = sampler2.SampleBox(fsi::ChFsiTypeConvert::Real3ToChVector(boxCenter2),
                                                               fsi::ChFsiTypeConvert::Real3ToChVector(boxHalfDim2));
    int numPart1 = points1.size();
    int numPart2 = points2.size();

    // This section is important to remove the fluid particles that overlap the BCE particles.
    // In order to use this future, two things needs to be done;
    // 1- run the simulation as usual, e.g. ./bin/demo_, which will create the BCE.csv
    // This could be the actual simulation if there is no overlap between the fluid and BCE markers, if there is:
    // 2- run the actual simulation with a command line argument such as ./bin/demo_ 2, which will go over
    // the BCE.csv file and use those information to remove overlap of fluid and BCE markers

    std::cout << "Initial fluid size:" << numPart1 + numPart2 << std::endl;
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
        Real h = initSpace1;
        fsi::Real4 p = fsi::mR4(points1[i].x(), points1[i].y(), points1[i].z(), h);
        for (int remove = 0; remove < BCE_RIGID_SIZE; remove++) {
            double dist = length(particles_position[remove] - mR3(p));
            if (dist < 0.98 * initSpace1) {
                removeThis = true;
                break;
            }
        }
        if (!removeThis) {
            myFsiSystem.GetDataManager()->AddSphMarker(p, chrono::fsi::mR3(paramsH->V_in, 0.0, 0.0),
                                                       chrono::fsi::mR4(paramsH->rho0, 1e-10, paramsH->mu0, -1.0));
        } else
            numremove++;
    }

    for (int i = 0; i < numPart2; i++) {
        bool removeThis = false;
        Real h = initSpace2;
        fsi::Real4 p = fsi::mR4(points2[i].x(), points2[i].y(), points2[i].z(), h);
        for (int remove = 0; remove < particles_position.size(); remove++) {
            double dist = length(particles_position[remove] - mR3(p));
            if (dist < 0.98 * initSpace1) {
                removeThis = true;
                break;
            }
        }
        if (!removeThis) {
            myFsiSystem.GetDataManager()->AddSphMarker(p, chrono::fsi::mR3(paramsH->V_in, 0.0, 0.0),
                                                       chrono::fsi::mR4(paramsH->rho0, 1e-10, paramsH->mu0, -1));
        } else
            numremove++;
    }

    std::cout << "Removed " << numremove << " Fluid particles from the simulation" << std::endl;

    particles_position.clear();

    //    int numPart = points.size();
    //    for (int i = 0; i < numPart; i++) {
    //        myFsiSystem.GetDataManager()->AddSphMarker(fsi::mR3(points[i].x(), points[i].y(), points[i].z()),
    //                                                   fsi::mR3(1e-10),
    //                                                   fsi::mR4(paramsH->rho0, paramsH->BASEPRES, paramsH->mu0, -1));
    //    }

    int numPhases = myFsiSystem.GetDataManager()->fsiGeneralData.referenceArray.size();

    int numFluidPart = myFsiSystem.GetDataManager()->sphMarkersH.posRadH.size();
    std::cout << "Final fluid size:" << numFluidPart << std::endl;

    myFsiSystem.GetDataManager()->fsiGeneralData.referenceArray.push_back(mI4(0, numGhostMarkers, -2, -1));
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

    // ***************************** System Initialize
    // ********************************************

    double mTime = 0;
    int stepEnd = int(paramsH->tFinal / paramsH->dT);
    stepEnd = 100000;
    std::vector<std::vector<double>> vCoor;
    std::vector<std::vector<int>> faces;
    const std::string rmCmd = (std::string("rm ") + demo_dir + std::string("/*"));
    system(rmCmd.c_str());
    SaveParaViewFilesMBD(myFsiSystem, mphysicalSystem, paramsH, 0, mTime);
    const std::string copyInitials =
        (std::string("cp ") + demo_dir + std::string("/BCE0.csv") + std::string(" ./BCE.csv "));
    system(copyInitials.c_str());
    myFsiSystem.SetFluidIntegratorType(fsi::ChFluidDynamics::Integrator::IISPH);

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
        chrono::fsi::utils::PrintToFile(
            myFsiSystem.GetDataManager()->sphMarkersD2.posRadD, myFsiSystem.GetDataManager()->sphMarkersD2.velMasD,
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
        thisNode.x() = radius * cos(2 * i * 3.1415 / res);
        thisNode.y() = -1 * length / 2;
        thisNode.z() = radius * sin(2 * i * 3.1415 / res);
        vertex = Rotation * thisNode + center;  // rotate/scale, if needed
        output << vertex.x() << " " << vertex.y() << " " << vertex.z() << "\n";
    }

    for (int i = 0; i < res; i++) {
        ChVector<double> thisNode;
        thisNode.x() = radius * cos(2 * i * 3.1415 / res);
        thisNode.y() = +1 * length / 2;
        thisNode.z() = radius * sin(2 * i * 3.1415 / res);
        vertex = Rotation * thisNode + center;  // rotate/scale, if needed
        output << vertex.x() << " " << vertex.y() << " " << vertex.z() << "\n";
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
