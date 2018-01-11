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

#ifdef CHRONO_MKL
#include "chrono_mkl/ChSolverMKL.h"
#endif

#include "chrono/solver/ChSolverMINRES.h"
#include "chrono/utils/ChUtilsCreators.h"
#include "chrono/utils/ChUtilsGenerators.h"
#include "chrono/utils/ChUtilsGeometry.h"
#include "chrono/utils/ChUtilsInputOutput.h"

// Chrono general utils
#include "chrono/core/ChFileutils.h"
#include "chrono/core/ChTransform.h"  //transform acc from GF to LF for post process

// Chrono fsi includes
#include "chrono_fsi/ChDeviceUtils.cuh"
#include "chrono_fsi/ChFsiTypeConvert.h"
#include "chrono_fsi/ChSystemFsi.h"
#include "chrono_fsi/utils/ChUtilsGeneratorFsi.h"
#include "chrono_fsi/utils/ChUtilsPrintSph.cuh"

// Chrono fea includes

#include "chrono_fea/ChElementShellANCF.h"
#include "chrono_fea/ChLinkDirFrame.h"
#include "chrono_fea/ChLinkPointFrame.h"
#include "chrono_fea/ChMesh.h"

// FSI Interface Includes
#include "demos/fsi/demo_FSI_Flexible_Shell.h"

#define haveFluid 1

// Chrono namespaces
using namespace chrono;
using namespace chrono::fea;
using namespace chrono::collision;

using std::cout;
using std::endl;
std::ofstream simParams;

//****************************************************************************************
const std::string out_dir = GetChronoOutputPath() + "FSI_FLEXIBLE_SHELL";
const std::string data_folder = out_dir + "/Paraview/";
std::string MESH_CONNECTIVITY = data_folder + "Flex_MESH.vtk";
// Save data as csv files, turn it on to be able to see the results off-line using paraview
bool pv_output = true;

std::vector<std::vector<int>> NodeNeighborElementMesh;

int out_fps = 20;

typedef fsi::Real Real;
Real contact_recovery_speed = 1;  ///< recovery speed for MBD

// Dimension of the domain
Real bxDim = 3;
Real byDim = 0.2;
Real bzDim = 1.5;

// Dimension of the fluid domain
Real fxDim = 1;
Real fyDim = byDim;
Real fzDim = 1;

// Forward declarations
void writeMesh(std::shared_ptr<ChMesh> my_mesh, std::string SaveAs, std::vector<std::vector<int>>& NodeNeighborElement);
void writeFrame(std::shared_ptr<ChMesh> my_mesh,
                char SaveAsBuffer[256],
                char MeshFileBuffer[256],
                std::vector<std::vector<int>>& NodeNeighborElement);
void SaveParaViewFiles(fsi::ChSystemFsi& myFsiSystem,
                       ChSystemSMC& mphysicalSystem,
                       std::shared_ptr<fea::ChMesh> my_mesh,
                       std::vector<std::vector<int>> NodeNeighborElementMesh,
                       chrono::fsi::SimParams* paramsH,
                       int next_frame,
                       double mTime);
void Create_MB_FE(ChSystemSMC& mphysicalSystem, fsi::ChSystemFsi& myFsiSystem, chrono::fsi::SimParams* paramsH);
//****************************************************************************************

int main(int argc, char* argv[]) {
    time_t rawtime;
    struct tm* timeinfo;

    time(&rawtime);
    timeinfo = localtime(&rawtime);
    // You can set your cuda device, if you have multiple of them
    // If not this should be set to 0 or not specified at all
    cudaSetDevice(1);

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
    // ******************************* Create Fluid region ***********************************
    ChSystemSMC mphysicalSystem;
    fsi::ChSystemFsi myFsiSystem(&mphysicalSystem, mHaveFluid, fsi::ChFluidDynamics::Integrator::IISPH);

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
            chrono::fsi::mR4(points[i].x(), points[i].y(), points[i].z(), paramsH->HSML), chrono::fsi::mR3(1e-10),
            chrono::fsi::mR4(paramsH->rho0, paramsH->BASEPRES, paramsH->mu0, -1.0));
    }

    int numPhases = myFsiSystem.GetDataManager()->fsiGeneralData.referenceArray.size();

    if (numPhases != 0) {
        std::cout << "Error! numPhases is wrong, thrown from main\n" << std::endl;
        std::cin.get();
        return -1;
    } else {
        myFsiSystem.GetDataManager()->fsiGeneralData.referenceArray.push_back(mI4(0, numPart, -1, -1));
        myFsiSystem.GetDataManager()->fsiGeneralData.referenceArray.push_back(mI4(numPart, numPart, 0, 0));
    }
#endif

    // ******************************* Create MBD or FE model ******************************
    ChVector<> gravity = ChVector<>(paramsH->gravity.x, paramsH->gravity.y, paramsH->gravity.z);

    Create_MB_FE(mphysicalSystem, myFsiSystem, paramsH);
    auto my_mesh = std::make_shared<fea::ChMesh>();
    if (mphysicalSystem.Get_otherphysicslist()->size()) {
        my_mesh = std::dynamic_pointer_cast<fea::ChMesh>(mphysicalSystem.Get_otherphysicslist()->at(0));
    }
    myFsiSystem.Finalize();
    if (myFsiSystem.GetDataManager()->sphMarkersH.posRadH.size() !=
        myFsiSystem.GetDataManager()->numObjects.numAllMarkers) {
        printf("\n\n\n\n Error! (2) numObjects is not set correctly \n %d, %d \n\n\n",
               myFsiSystem.GetDataManager()->sphMarkersH.posRadH.size(),
               myFsiSystem.GetDataManager()->numObjects.numAllMarkers);
        return -1;
    }
    // ******************************* System Initialize ***********************************
    int step_count = 0;
    double mTime = 0;

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

    mphysicalSystem.SetTimestepperType(ChTimestepper::Type::HHT);
    auto mystepper = std::static_pointer_cast<ChTimestepperHHT>(mphysicalSystem.GetTimestepper());
    mystepper->SetAlpha(-0.2);
    mystepper->SetMaxiters(1000);
    mystepper->SetAbsTolerances(1e-5);
    mystepper->SetMode(ChTimestepperHHT::POSITION);
    mystepper->SetScaling(true);

    //
    //    // Set up integrator
    //    mphysicalSystem.SetTimestepperType(ChTimestepper::Type::EULER_IMPLICIT);

    int stepEnd = int(paramsH->tFinal / paramsH->dT);
    stepEnd = 1000000;
    SaveParaViewFiles(myFsiSystem, mphysicalSystem, my_mesh, NodeNeighborElementMesh, paramsH, 0, mTime);

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

        printf("next_frame is:%d,  max dt is set to %f\n", next_frame, paramsH->dT_Max);

#if haveFluid
        myFsiSystem.DoStepDynamics_FSI_Implicit();
#else
        myFsiSystem.DoStepDynamics_ChronoRK2();
#endif
        time += paramsH->dT;
        SaveParaViewFiles(myFsiSystem, mphysicalSystem, my_mesh, NodeNeighborElementMesh, paramsH, next_frame, time);
    }

    return 0;
}

//------------------------------------------------------------------
// Create the objects of the MBD system. Rigid/flexible bodies, and if fsi, their
// bce representation are created and added to the systems
//------------------------------------------------------------------
void Create_MB_FE(ChSystemSMC& mphysicalSystem, fsi::ChSystemFsi& myFsiSystem, chrono::fsi::SimParams* paramsH) {
    mphysicalSystem.Set_G_acc(ChVector<>(0, 0, 0));
    auto mysurfmaterial = std::make_shared<ChMaterialSurfaceSMC>();

    // Set common material Properties
    mysurfmaterial->SetYoungModulus(6e4);
    mysurfmaterial->SetFriction(0.3f);
    mysurfmaterial->SetRestitution(0.2f);
    mysurfmaterial->SetAdhesion(0);

    auto ground = std::make_shared<ChBody>();
    ground->SetIdentifier(-1);
    ground->SetBodyFixed(true);
    ground->SetCollide(true);
    ground->SetMaterialSurface(mysurfmaterial);

    ground->GetCollisionModel()->ClearModel();
    Real initSpace0 = paramsH->MULT_INITSPACE * paramsH->HSML;

    // Bottom and top wall
    ChVector<> sizeBottom(bxDim / 2 + 3 * initSpace0, byDim / 2 + 3 * initSpace0, 2 * initSpace0);
    ChVector<> posBot(0, 0, -2 * initSpace0);
    ChVector<> posTop(0, 0, bzDim + 2 * initSpace0);

    // left and right Wall
    ChVector<> size_YZ(2 * initSpace0, byDim / 2 + 3 * initSpace0, bzDim / 2);
    ChVector<> pos_xp(bxDim / 2 + initSpace0, 0.0, bzDim / 2 + 1 * initSpace0);
    ChVector<> pos_xn(-bxDim / 2 - 3 * initSpace0, 0.0, bzDim / 2 + 1 * initSpace0);

    // Front and back Wall
    ChVector<> size_XZ(bxDim / 2, 2 * initSpace0, bzDim / 2);
    ChVector<> pos_yp(0, byDim / 2 + initSpace0, bzDim / 2 + 1 * initSpace0);
    ChVector<> pos_yn(0, -byDim / 2 - 3 * initSpace0, bzDim / 2 + 1 * initSpace0);

    // MBD representation of walls
    chrono::utils::AddBoxGeometry(ground.get(), sizeBottom, posBot, chrono::QUNIT, true);
    chrono::utils::AddBoxGeometry(ground.get(), size_YZ, pos_xp, chrono::QUNIT, true);
    chrono::utils::AddBoxGeometry(ground.get(), size_YZ, pos_xn, chrono::QUNIT, true);
    chrono::utils::AddBoxGeometry(ground.get(), size_XZ, pos_yp, chrono::QUNIT, true);
    chrono::utils::AddBoxGeometry(ground.get(), size_XZ, pos_yn, chrono::QUNIT, true);
    mphysicalSystem.AddBody(ground);

#if haveFluid
    // Fluid representation of walls
    chrono::fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, ground, posBot, chrono::QUNIT, sizeBottom, 12);
    chrono::fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, ground, posTop, chrono::QUNIT, sizeBottom, 12);
    chrono::fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, ground, pos_xp, chrono::QUNIT, size_YZ, 23);
    chrono::fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, ground, pos_xn, chrono::QUNIT, size_YZ, 23);
    chrono::fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, ground, pos_yp, chrono::QUNIT, size_XZ, 13);
    chrono::fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, ground, pos_yn, chrono::QUNIT, size_XZ, 13);
#endif
    // ******************************* Flexible bodies ***********************************
    // Create a mesh, that is a container for groups of elements and their referenced nodes.
    auto my_mesh = std::make_shared<fea::ChMesh>();
    int numFlexBody = 1;
    // Geometry of the plate
    double plate_lenght_x = initSpace0 * 16;
    double plate_lenght_y = byDim;
    double plate_lenght_z = 0.02;
    // Specification of the mesh
    int numDiv_x = 8;
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
    std::vector<std::vector<int>> elementsNodes_mesh;
    std::vector<std::vector<int>> NodeNeighborElement_mesh;
    elementsNodes_mesh.resize(TotalNumElements);
    NodeNeighborElement_mesh.resize(TotalNumNodes);
    // Create and add the nodes
    for (int i = 0; i < TotalNumNodes; i++) {
        // Node location
        double loc_x = (i % (numDiv_x + 1)) * dx + initSpace0;
        double loc_y = (i / (numDiv_x + 1)) % (numDiv_y + 1) * dy - byDim / 2;
        double loc_z = (i) / ((numDiv_x + 1) * (numDiv_y + 1)) * dz + bxDim / 8 + 3 * initSpace0;

        // Node direction
        double dir_x = 0;
        double dir_y = 0;
        double dir_z = 1;

        // Create the node
        auto node = std::make_shared<ChNodeFEAxyzD>(ChVector<>(loc_z, loc_y, loc_x), ChVector<>(dir_z, dir_y, dir_x));

        node->SetMass(0);
        // Fix all nodes along the axis X=0
        if (i % (numDiv_x + 1) == 0)
            node->SetFixed(true);

        // Add node to mesh
        my_mesh->AddNode(node);
    }

    // Get a handle to the tip node.
    auto nodetip = std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(TotalNumNodes - 1));

    // Create an isotropic material.
    // All layers for all elements share the same material.
    double rho = 1000;
    double E = 5e6;
    double nu = 0.3;
    auto mat = std::make_shared<ChMaterialShellANCF>(rho, E, nu);
    // Create the elements
    for (int i = 0; i < TotalNumElements; i++) {
        // Adjacent nodes
        int node0 = (i / (numDiv_x)) * (N_x) + i % numDiv_x;
        int node1 = (i / (numDiv_x)) * (N_x) + i % numDiv_x + 1;
        int node2 = (i / (numDiv_x)) * (N_x) + i % numDiv_x + 1 + N_x;
        int node3 = (i / (numDiv_x)) * (N_x) + i % numDiv_x + N_x;
        elementsNodes_mesh[i].push_back(node0);
        elementsNodes_mesh[i].push_back(node1);
        elementsNodes_mesh[i].push_back(node2);
        elementsNodes_mesh[i].push_back(node3);
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
        element->SetAlphaDamp(0.1);    // Structural damping for this element
        element->SetGravityOn(false);  // turn internal gravitational force calculation off

        // Add element to mesh
        my_mesh->AddElement(element);
        ChVector<> center = 0.25 * (element->GetNodeA()->GetPos() + element->GetNodeB()->GetPos() +
                                    element->GetNodeC()->GetPos() + element->GetNodeD()->GetPos());
        cout << "Adding element" << i << "  with center:  " << center.x() << " " << center.y() << " " << center.z()
             << endl;
    }

    // Add the mesh to the system
    mphysicalSystem.Add(my_mesh);

    // fluid representation of flexible bodies
    std::vector<std::shared_ptr<chrono::fea::ChElementCableANCF>>* FSI_Cables = myFsiSystem.GetFsiCablesPtr();
    std::vector<std::shared_ptr<chrono::fea::ChElementShellANCF>>* FSI_Shells = myFsiSystem.GetFsiShellsPtr();
    std::vector<std::shared_ptr<chrono::fea::ChNodeFEAxyzD>>* FSI_Nodes = myFsiSystem.GetFsiNodesPtr();
    bool multilayer = true;
    bool removeMiddleLayer = false;
    bool add1DElem = false;
    bool add2DElem = true;
    std::vector<std::vector<int>> _1D_elementsNodes_mesh;
    chrono::fsi::utils::AddBCE_FromMesh(myFsiSystem.GetDataManager(), paramsH, my_mesh, FSI_Nodes, FSI_Cables,
                                        FSI_Shells, NodeNeighborElement_mesh, _1D_elementsNodes_mesh,
                                        elementsNodes_mesh, add1DElem, add2DElem, multilayer, removeMiddleLayer, 0, 0);

    myFsiSystem.SetShellElementsNodes(elementsNodes_mesh);
    myFsiSystem.SetFsiMesh(my_mesh);
    writeMesh(my_mesh, MESH_CONNECTIVITY, NodeNeighborElementMesh);
}

//------------------------------------------------------------------
// Function to save the paraview files
//------------------------------------------------------------------
void SaveParaViewFiles(fsi::ChSystemFsi& myFsiSystem,
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

    if (pv_output && std::abs(mTime - (next_frame)*frame_time) < 0.00001) {
        chrono::fsi::utils::PrintToFile(
            myFsiSystem.GetDataManager()->sphMarkersD2.posRadD, myFsiSystem.GetDataManager()->sphMarkersD2.velMasD,
            myFsiSystem.GetDataManager()->sphMarkersD2.rhoPresMuD,
            myFsiSystem.GetDataManager()->fsiGeneralData.referenceArray,
            myFsiSystem.GetDataManager()->fsiGeneralData.referenceArray_FEA, data_folder, true);

        cout << "             Sim frame:      " << next_frame << endl;
        cout << "             Time:           " << mTime << endl;
        cout << "---------------------------------------------" << endl;

        char SaveAsBuffer[256];  // The filename buffer.
        snprintf(SaveAsBuffer, sizeof(char) * 256, (data_folder + "/flex_body.%d.vtk").c_str(), next_frame);
        char MeshFileBuffer[256];  // The filename buffer.
        snprintf(MeshFileBuffer, sizeof(char) * 256, ("%s"), MESH_CONNECTIVITY.c_str());
        writeFrame(my_mesh, SaveAsBuffer, MeshFileBuffer, NodeNeighborElementMesh);

        out_frame++;
    }
}
//------------------------------------------------------------------
// Write  MESH Cennectivity
//------------------------------------------------------------------
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
//------------------------------------------------------------------
// Write to VTK
//------------------------------------------------------------------
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
        output << node->GetPos().x() << " " << node->GetPos().y() << " " << node->GetPos().z() << "\n";
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

        output << areaAve / myarea + 1e-20 << "\n";
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
            areaAve1 += MyResult[0].x() * dx * dy / NodeNeighborElement[i].size();
            areaAve2 += MyResult[0].y() * dx * dy / NodeNeighborElement[i].size();
            if (abs(MyResult[0].x()) > 1e-3 && abs(MyResult[0].y()) > 1e-3) {
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
            areaAve1 += MyResult[1].x() * dx * dy / NodeNeighborElement[i].size();
            areaAve2 += MyResult[1].y() * dx * dy / NodeNeighborElement[i].size();
            areaAve3 += MyResult[1].z() * dx * dy / NodeNeighborElement[i].size();
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
            areaAve1 += MyResult[2].x() * dx * dy / NodeNeighborElement[i].size();
            areaAve2 += MyResult[2].y() * dx * dy / NodeNeighborElement[i].size();
            areaAve3 += MyResult[2].z() * dx * dy / NodeNeighborElement[i].size();
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
            areaAve1 += MyResult[0].x() * dx * dy / NodeNeighborElement[i].size();
            areaAve2 += MyResult[0].y() * dx * dy / NodeNeighborElement[i].size();
            areaAve3 += MyResult[0].z() * dx * dy / NodeNeighborElement[i].size();
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
            areaAve1 += MyResult[1].x() * dx * dy / NodeNeighborElement[i].size();
            areaAve2 += MyResult[1].y() * dx * dy / NodeNeighborElement[i].size();
            areaAve3 += MyResult[1].z() * dx * dy / NodeNeighborElement[i].size();
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
            areaAve1 += MyResult[2].x() * dx * dy / NodeNeighborElement[i].size();
            areaAve2 += MyResult[2].y() * dx * dy / NodeNeighborElement[i].size();
            areaAve3 += MyResult[2].z() * dx * dy / NodeNeighborElement[i].size();
        }
        output << areaAve1 / myarea << " " << areaAve2 / myarea << " " << areaAve3 / myarea << "\n";
    }

    output.close();
}
