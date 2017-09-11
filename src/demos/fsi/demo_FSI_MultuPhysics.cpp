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

//#include "chrono_utils/ChUtilsVehicle.h"
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
#include "chrono_fsi/utils/ChUtilsGeneratorFsi.h"
#include "chrono_fsi/utils/ChUtilsPrintSph.h"

// Chrono fea includes
#include "chrono_fea/ChElementCableANCF.h"
#include "chrono_fea/ChBuilderBeam.h"
#include "chrono_fea/ChElementShellANCF.h"
#include "chrono_fea/ChLinkDirFrame.h"
#include "chrono_fea/ChLinkPointFrame.h"
#include "chrono_fea/ChMesh.h"

// FSI Interface Includes
#include "demos/fsi/demo_FSI_MultiPhysics.h"  //SetupParamsH()

#define haveFluid 1

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
const std::string h_file =
    "/home/milad/CHRONO/Project-Chrono-Milad-IISPH/src/demos/fsi/demo_FSI_DamBreak_Flexible_Shell.h";
const std::string cpp_file =
    "/home/milad/CHRONO/Project-Chrono-Milad-IISPH/src/demos/fsi/demo_FSI_DamBreak_Flexible_Shell.cpp";

const std::string out_dir = "FSI_OUTPUT";  //"../FSI_OUTPUT";
const std::string data_folder = out_dir + "/MultiPhysics/";
std::string MESH_CONNECTIVITY = data_folder + "Flex_MESH.vtk";

std::vector<std::vector<int>> NodeNeighborElementMesh;

bool povray_output = true;
int out_fps = 100;

typedef fsi::Real Real;
Real contact_recovery_speed = 1;  ///< recovery speed for MBD

Real bxDim = 3;
Real byDim = 0.2;
Real bzDim = 2;

Real fxDim = 1;
Real fyDim = byDim;
Real fzDim = 1;

bool addCable = true;
bool addPlate = false;

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
void SetArgumentsForMbdFromInput(int argc,
                                 char* argv[],
                                 int& threads,
                                 int& max_iteration_sliding,
                                 int& max_iteration_bilateral,
                                 int& max_iteration_normal,
                                 int& max_iteration_spinning);

void InitializeMbdPhysicalSystem(ChSystemDEM& mphysicalSystem, ChVector<> gravity, int argc, char* argv[]);

void SaveParaViewFilesMBD(fsi::ChSystemFsi& myFsiSystem,
                          ChSystemDEM& mphysicalSystem,
                          std::shared_ptr<fea::ChMesh> my_mesh,
                          std::vector<std::vector<int>> NodeNeighborElementMesh,
                          chrono::fsi::SimParams* paramsH,
                          int next_frame,
                          double mTime);

void Create_MB_FE(ChSystemDEM& mphysicalSystem, fsi::ChSystemFsi& myFsiSystem, chrono::fsi::SimParams* paramsH);

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
    fsi::ChSystemFsi myFsiSystem(&mphysicalSystem, mHaveFluid);
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
    auto my_mesh = std::make_shared<fea::ChMesh>();

    my_mesh = myFsiSystem.GetFsiMesh();
    myFsiSystem.Finalize();

    if (myFsiSystem.GetDataManager()->sphMarkersH.posRadH.size() !=
        myFsiSystem.GetDataManager()->numObjects.numAllMarkers) {
        printf("\n\n\n\n Error! (2) numObjects is not set correctly \n %d, %d \n\n\n",
               myFsiSystem.GetDataManager()->sphMarkersH.posRadH.size(),
               myFsiSystem.GetDataManager()->numObjects.numAllMarkers);
        return -1;
    }
    //*** Add sph data to the physics system

    cout << " -- ChSystem size : " << mphysicalSystem.Get_bodylist()->size() << endl;

    // ******************** System Initialize ************************

    int step_count = 0;

    double mTime = 0;
#ifdef CHRONO_FSI_USE_DOUBLE
    printf("Double Precision\n");
#else
    printf("Single Precision\n");
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
    mystepper->SetMaxiters(100);
    mystepper->SetAbsTolerances(1e-5);
    mystepper->SetMode(ChTimestepperHHT::POSITION);
    mystepper->SetScaling(true);

    int stepEnd = int(paramsH->tFinal / paramsH->dT);
    stepEnd = 1000000;

    SaveParaViewFilesMBD(myFsiSystem, mphysicalSystem, my_mesh, NodeNeighborElementMesh, paramsH, 0, mTime);

    Real time = 0;
    Real Global_max_dT = paramsH->dT_Max;
    for (int tStep = 0; tStep < stepEnd + 1; tStep++) {
        printf("\nstep : %d, time= : %f (s) \n", tStep, time);
        double frame_time = 1.0 / out_fps;
        int next_frame = std::floor((time + 1e-6) / frame_time) + 1;
        double next_frame_time = next_frame * frame_time;
        double max_allowable_dt = next_frame_time - time;
        if (max_allowable_dt > 1e-6)
            paramsH->dT_Max = std::min(Global_max_dT, max_allowable_dt);
        else
            paramsH->dT_Max = Global_max_dT;

        //        std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(my_mesh->GetNnodes() - 1))
        //            ->SetForce(ChVector<>(+5, 0, 0));
        //        std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(40))->SetForce(ChVector<>(+100, 0, 0));

        printf("next_frame is:%d,  max dt is set to %f\n", next_frame, paramsH->dT_Max);

#if haveFluid
        myFsiSystem.DoStepDynamics_FSI_Implicit();
#else
        myFsiSystem.DoStepDynamics_ChronoRK2();
#endif
        time += paramsH->dT;
        SaveParaViewFilesMBD(myFsiSystem, mphysicalSystem, my_mesh, NodeNeighborElementMesh, paramsH, next_frame, time);
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
    ChVector<> sizeBottom(bxDim / 2 + 3 * initSpace0, byDim / 2 + 3 * initSpace0, 2 * initSpace0);
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
    mphysicalSystem.AddBody(ground);

#if haveFluid

    /*================== Walls =================*/

    chrono::fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, ground, posBottom, chrono::QUNIT, sizeBottom);
    chrono::fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, ground, posTop, chrono::QUNIT, sizeBottom);

    chrono::fsi::utils::AddBoxBceYZ(myFsiSystem.GetDataManager(), paramsH, ground, pos_xp, chrono::QUNIT, size_YZ);

    chrono::fsi::utils::AddBoxBceYZ(myFsiSystem.GetDataManager(), paramsH, ground, pos_xn, chrono::QUNIT, size_YZ);

    chrono::fsi::utils::AddBoxBceXZ(myFsiSystem.GetDataManager(), paramsH, ground, pos_yp, chrono::QUNIT, size_XZ);

    chrono::fsi::utils::AddBoxBceXZ(myFsiSystem.GetDataManager(), paramsH, ground, pos_yn, chrono::QUNIT, size_XZ);

    /*================== Flexible-Bodies =================*/
    auto my_mesh = std::make_shared<fea::ChMesh>();
    double rho = 1000;
    double E = 1e6;
    double nu = 0.3;
    auto mat = std::make_shared<ChMaterialShellANCF>(rho, E, nu);
    std::vector<std::vector<int>> NodeNeighborElement;
    std::vector<std::vector<int>> _1D_elementsNodes_mesh;
    std::vector<std::vector<int>> _2D_elementsNodes_mesh;
    if (addCable) {
        int numCableElems = 4;
        /*================== Cable Elements =================*/
        auto msection_cable = std::make_shared<ChBeamSectionCable>();
        msection_cable->SetDiameter(initSpace0 * 2);
        msection_cable->SetYoungModulus(E);
        msection_cable->SetBeamRaleyghDamping(0.01);
        // Create material.

        // Shortcut!
        // This ChBuilderBeamANCF helper object is very useful because it will
        // subdivide 'beams' into sequences of finite elements of beam type, ex.
        // one 'beam' could be made of 5 FEM elements of ChElementBeamANCF class.
        // If new nodes are needed, it will create them for you.
        ChBuilderBeamANCF builder;

        // Now, simply use BuildBeam to create a beam from a point to another:
        builder.BuildBeam_FSI(
            my_mesh,                           // the mesh where to put the created nodes and elements
            msection_cable,                    // the ChBeamSectionCable to use for the ChElementBeamANCF elements
            numCableElems,                     // the number of ChElementBeamANCF to create
            ChVector<>(0, 0, initSpace0),      // the 'A' point in space (beginning of beam)
            ChVector<>(0, 0, 9 * initSpace0),  // the 'B' point in space (end of beam) _1D_elementsNodes_mesh,
            _1D_elementsNodes_mesh, NodeNeighborElementMesh);

        //        // After having used BuildBeam(), you can retrieve the nodes used for the beam,
        //        // For example say you want to fix both pos and dir of A end and apply a force to the B end:
        //        builder.GetLastBeamNodes().back()->SetFixed(true);

        for (int i = 0; i < my_mesh->GetNnodes(); i++) {
            auto node = std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(i));
            if (node->GetPos().z >= bzDim / 4)
                node->SetFixed(true);
        }

        // For instance, now retrieve the A end and add a constraint to
        // block the position only of that node:
        auto mtruss = std::make_shared<ChBody>();
        mtruss->SetBodyFixed(true);

        //        auto constraint_hinge = std::make_shared<ChLinkPointFrame>();
        //        constraint_hinge->Initialize(builder.GetLastBeamNodes().back(), mtruss);
        //        mphysicalSystem.Add(constraint_hinge);
    }

    if (addPlate) {
        GetLog() << "-----------------------------------------------------------\n";
        GetLog() << "-----------------------------------------------------------\n";
        GetLog() << "     ANCF Shell Elements demo with implicit integration \n";
        GetLog() << "-----------------------------------------------------------\n";
        double rho = 1000;
        double E = 1e6;
        double nu = 0.3;
        auto mat = std::make_shared<ChMaterialShellANCF>(rho, E, nu);
        // Create a mesh, that is a container for groups of elements and their referenced nodes.
        int numFlexBody = 1;
        // Geometry of the plate
        double plate_lenght_x = bzDim / 4;
        double plate_lenght_y = byDim;
        double plate_lenght_z = 0.02;
        // Specification of the mesh
        int numDiv_x = 4;
        int numDiv_y = 4;
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
        int currentsize = NodeNeighborElement.size();
        _2D_elementsNodes_mesh.resize(TotalNumElements);
        NodeNeighborElementMesh.resize(currentsize + TotalNumNodes);
        // Create and add the nodes
        for (int i = 0; i < TotalNumNodes; i++) {
            // Node location
            double loc_x = (i % (numDiv_x + 1)) * dx + initSpace0;
            double loc_y = (i / (numDiv_x + 1)) % (numDiv_y + 1) * dy - byDim / 2;
            double loc_z = (i) / ((numDiv_x + 1) * (numDiv_y + 1)) * dz + bxDim / 4 + 2 * initSpace0;

            // Node direction
            double dir_x = 0;
            double dir_y = 0;
            double dir_z = 1;

            // Create the node
            auto node =
                std::make_shared<ChNodeFEAxyzD>(ChVector<>(loc_z, loc_y, loc_x), ChVector<>(dir_z, dir_y, dir_x));

            node->SetMass(0);

            // Fix all nodes along the axis X=0
            if (i % (numDiv_x + 1) == 0)
                node->SetFixed(true);

            // Add node to mesh
            my_mesh->AddNode(node);
        }

        // Get a handle to the tip node.
        auto nodetip = std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(TotalNumNodes - 1));

        // Create the elements
        for (int i = 0; i < TotalNumElements; i++) {
            // Adjacent nodes
            int node0 = (i / (numDiv_x)) * (N_x) + i % numDiv_x + currentsize;
            int node1 = (i / (numDiv_x)) * (N_x) + i % numDiv_x + 1 + currentsize;
            int node2 = (i / (numDiv_x)) * (N_x) + i % numDiv_x + 1 + N_x + currentsize;
            int node3 = (i / (numDiv_x)) * (N_x) + i % numDiv_x + N_x + currentsize;
            _2D_elementsNodes_mesh[i].push_back(node0 + 1);
            _2D_elementsNodes_mesh[i].push_back(node1 + 1);
            _2D_elementsNodes_mesh[i].push_back(node2 + 1);
            _2D_elementsNodes_mesh[i].push_back(node3 + 1);
            NodeNeighborElementMesh[node0].push_back(i);
            NodeNeighborElementMesh[node1].push_back(i);
            NodeNeighborElementMesh[node2].push_back(i);
            NodeNeighborElementMesh[node3].push_back(i);
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
            ChVector<> center = 0.25 * (element->GetNodeA()->GetPos() + element->GetNodeB()->GetPos() +
                                        element->GetNodeC()->GetPos() + element->GetNodeD()->GetPos());
            cout << "Adding element" << i << "  with center:  " << center.x << " " << center.y << " " << center.z
                 << endl;
        }
    }
    // Add the mesh to the system
    mphysicalSystem.Add(my_mesh);

    std::vector<std::shared_ptr<chrono::fea::ChElementCableANCF>>* FSI_Cables = myFsiSystem.GetFsiCablesPtr();
    std::vector<std::shared_ptr<chrono::fea::ChElementShellANCF>>* FSI_Shells = myFsiSystem.GetFsiShellsPtr();
    std::vector<std::shared_ptr<chrono::fea::ChNodeFEAxyzD>>* FSI_Nodes = myFsiSystem.GetFsiNodesPtr();

    bool multilayer = true;
    bool removeMiddleLayer = false;
    bool add1DElem = true;
    bool add2DElem = true;
    chrono::fsi::utils::AddBCE_FromMesh(myFsiSystem.GetDataManager(), paramsH, my_mesh, FSI_Nodes, FSI_Cables,
                                        FSI_Shells, NodeNeighborElementMesh, _1D_elementsNodes_mesh,
                                        _2D_elementsNodes_mesh, add1DElem, add2DElem, multilayer, removeMiddleLayer, 1);

    printf("_1D_elementsNodes_mesh.size() = %d\n", _1D_elementsNodes_mesh.size());

    myFsiSystem.SetCableElementsNodes(_1D_elementsNodes_mesh);
    myFsiSystem.SetShellElementsNodes(_2D_elementsNodes_mesh);
    myFsiSystem.SetFsiMesh(my_mesh);

    writeMesh(my_mesh, MESH_CONNECTIVITY, NodeNeighborElementMesh, _1D_elementsNodes_mesh, _2D_elementsNodes_mesh);
//    NodeNeighborElementMesh = NodeNeighborElement;

#endif
}

//------------------------------------------------------------------
// Function to save the ParaView files of the MBD
//------------------------------------------------------------------
void ReadPreviousData(fsi::ChSystemFsi& myFsiSystem,
                      ChSystemDEM& mphysicalSystem,
                      std::shared_ptr<fea::ChMesh> my_mesh,
                      std::vector<std::vector<int>> NodeNeighborElementMesh,
                      chrono::fsi::SimParams* paramsH,
                      int next_frame,
                      double mTime) {
    // **** out fluid
    chrono::fsi::utils::PrintToParaViewFile(
        myFsiSystem.GetDataManager()->sphMarkersD2.posRadD, myFsiSystem.GetDataManager()->sphMarkersD2.velMasD,
        myFsiSystem.GetDataManager()->sphMarkersD2.rhoPresMuD,
        myFsiSystem.GetDataManager()->fsiGeneralData.referenceArray,
        myFsiSystem.GetDataManager()->fsiGeneralData.referenceArray_FEA, data_folder);

    char SaveAsBuffer[256];  // The filename buffer.
    snprintf(SaveAsBuffer, sizeof(char) * 256, (data_folder + "/flex_body.%d.vtk").c_str(), next_frame);
    char MeshFileBuffer[256];  // The filename buffer.
    snprintf(MeshFileBuffer, sizeof(char) * 256, ("%s"), MESH_CONNECTIVITY.c_str());
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

        cout << "\n------------ Output frame:   " << next_frame << endl;
        cout << "             Sim frame:      " << next_frame << endl;
        cout << "             Time:           " << mTime << endl;
        cout << "             Avg. contacts:  " << num_contacts / out_steps << endl;
        cout << "             Execution time: " << exec_time << endl;
        cout << "\n------------------------------------------------\n" << endl;

        char SaveAsBuffer[256];  // The filename buffer.
        snprintf(SaveAsBuffer, sizeof(char) * 256, (data_folder + "/flex_body.%d.vtk").c_str(), next_frame);
        char MeshFileBuffer[256];  // The filename buffer.
        snprintf(MeshFileBuffer, sizeof(char) * 256, ("%s"), MESH_CONNECTIVITY.c_str());
        printf("%s from here\n", MeshFileBuffer);

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

    if (addPlate) {
        utils::CSV_writer MESH(" ");
        MESH.stream().setf(std::ios::scientific | std::ios::showpos);
        MESH.stream().precision(6);
        MESH << "\nCELLS " << _2D_elementsNodes_mesh.size() << 5 * _2D_elementsNodes_mesh.size() << "\n";

        for (int iele = 0; iele < _2D_elementsNodes_mesh.size(); iele++) {
            if (auto element = std::dynamic_pointer_cast<ChElementShellANCF>(my_mesh->GetElement(iele))) {
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
        }
        MESH << "\nCELL_TYPES " << _2D_elementsNodes_mesh.size() << "\n";

        for (int iele = 0; iele < _2D_elementsNodes_mesh.size(); iele++) {
            MESH << "9\n";
        }

        MESH.write_to_file(SaveAs + "_2D");
    }
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

    if (addPlate) {
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