

// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2013 Project Chrono
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file at the top level of the distribution
// and at http://projectchrono.org/license-chrono.txt.
//

//   Demos code about
//
//     - FEA using ANCF (introduction to dynamics)
//
//// Include some headers used by this tutorial...
#include "chrono/core/ChFileutils.h"
#include "chrono/physics/ChSystem.h"
#include "chrono/lcp/ChLcpIterativeMINRES.h"
#include "chrono/utils/ChUtilsInputOutput.h"
#include "chrono_fea/ChElementSpring.h"
#include "chrono_fea/ChElementBrick.h"
#include "chrono_fea/ChElementBar.h"
#include "chrono_fea/ChLinkPointFrame.h"
#include "chrono_fea/ChLinkDirFrame.h"
#include "chrono_fea/ChVisualizationFEAmesh.h"
#include <iostream>
#include "core/ChTimer.h"
// Remember to use the namespace 'chrono' because all classes
// of Chrono::Engine belong to this namespace and its children...

using namespace chrono;
using namespace fea;
double step_size = 1e-3;
double sim_time = 0.5;
int main(int argc, char* argv[]) {
    // If no command line arguments, run in "performance" mode and only report run time.
    // Otherwise, generate output files to verify correctness.
    bool output = (argc > 1);
    output = true;

    if (output) {
        GetLog() << "Output file: ../TEST/tip_position_brick.txt\n";
    } else {
        GetLog() << "Running in performance test mode.\n";
    }

    // --------------------------
    // Create the physical system
    // --------------------------
    ChSystem my_system;

    GetLog() << "-----------------------------------------------------------\n";
    GetLog() << "-----------------------------------------------------------\n";
    GetLog() << "     Brick Elements demo with implicit integration \n";
    GetLog() << "-----------------------------------------------------------\n";

    // The physical system: it contains all physical objects.
    // Create a mesh, that is a container for groups
    // of elements and their referenced nodes.
    ChSharedPtr<ChMesh> my_mesh(new ChMesh);
    int numFlexBody = 1;
    // Geometry of the plate
    double plate_lenght_x = 1;
    double plate_lenght_y = 1;
    double plate_lenght_z = 0.01;  // small thickness
    // Specification of the mesh
    int numDiv_x = 20;
    int numDiv_y = 20;
    int numDiv_z = 1;
    int N_x = numDiv_x + 1;
    int N_y = numDiv_y + 1;
    int N_z = numDiv_z + 1;
    // Number of elements in the z direction is considered as 1
    int TotalNumElements = numDiv_x * numDiv_y;
    int TotalNumNodes = (numDiv_x + 1) * (numDiv_y + 1) * (numDiv_z + 1);
    // For uniform mesh
    double dx = plate_lenght_x / numDiv_x;
    double dy = plate_lenght_y / numDiv_y;
    double dz = plate_lenght_z / numDiv_z;
    int MaxMNUM = 1;
    int MTYPE = 1;
    int MaxLayNum = 1;

    ChMatrixDynamic<double> COORDFlex(TotalNumNodes, 3);
    ChMatrixDynamic<double> VELCYFlex(TotalNumNodes, 3);
    ChMatrixDynamic<int> NumNodes(TotalNumElements, 8);
    ChMatrixDynamic<int> LayNum(TotalNumElements, 1);
    ChMatrixDynamic<int> NDR(TotalNumNodes, 3);
    ChMatrixDynamic<double> ElemLengthXY(TotalNumElements, 3);
    ChMatrixNM<double, 10, 12> MPROP;

    //!------------------------------------------------!
    //!------------ Read Material Data-----------------!
    //!------------------------------------------------!

    for (int i = 0; i < MaxMNUM; i++) {
        MPROP(i, 0) = 500;      // Density [kg/m3]
        MPROP(i, 1) = 2.1E+08;  // H(m)
        MPROP(i, 2) = 0.3;      // nu
    }

    ChSharedPtr<ChContinuumElastic> mmaterial(new ChContinuumElastic);
    mmaterial->Set_RayleighDampingK(0.0);
    mmaterial->Set_RayleighDampingM(0.0);
    mmaterial->Set_density(MPROP(0, 0));
    mmaterial->Set_E(MPROP(0, 1));
    mmaterial->Set_G(MPROP(0, 1) / (2 + 2 * MPROP(0, 2)));
    mmaterial->Set_v(MPROP(0, 2));
    //!------------------------------------------------!
    //!--------------- Element data--------------------!
    //!------------------------------------------------!
    for (int i = 0; i < TotalNumElements; i++) {
        // All the elements belong to the same layer, e.g layer number 1.
        LayNum(i, 0) = 1;
        // Node number of the 4 nodes which creates element i.
        // The nodes are distributed this way. First in the x direction for constant
        // y when max x is reached go to the
        // next level for y by doing the same   distribution but for y+1 and keep
        // doing until y limit is reached. Node
        // number start from 1.

        NumNodes(i, 0) = (i / (numDiv_x)) * (N_x) + i % numDiv_x;
        NumNodes(i, 1) = (i / (numDiv_x)) * (N_x) + i % numDiv_x + 1;
        NumNodes(i, 2) = (i / (numDiv_x)) * (N_x) + i % numDiv_x + N_x;
        NumNodes(i, 3) = (i / (numDiv_x)) * (N_x) + i % numDiv_x + 1 + N_x;
        NumNodes(i, 4) = (numDiv_x + 1) * (numDiv_y + 1) + NumNodes(i, 0);
        NumNodes(i, 5) = (numDiv_x + 1) * (numDiv_y + 1) + NumNodes(i, 1);
        NumNodes(i, 6) = (numDiv_x + 1) * (numDiv_y + 1) + NumNodes(i, 2);
        NumNodes(i, 7) = (numDiv_x + 1) * (numDiv_y + 1) + NumNodes(i, 3);

        // Let's keep the element length a fixed number in both direction. (uniform
        // distribution of nodes in both direction)

        ElemLengthXY(i, 0) = dx;
        ElemLengthXY(i, 1) = dy;
        ElemLengthXY(i, 2) = dz;

        if (MaxLayNum < LayNum(i, 0)) {
            MaxLayNum = LayNum(i, 0);
        }
    }
    //!----------------------------------------------!
    //!--------- NDR,COORDFlex,VELCYFlex-------------!
    //!----------------------------------------------!

    for (int i = 0; i < TotalNumNodes; i++) {
        // If the node is the first node from the left side fix the x,y,z degree of
        // freedom. 1 for constrained 0 for ...
        //-The NDR array is used to define the degree of freedoms that are
        // constrained in the 6 variable explained above.
        NDR(i, 0) = (i % (numDiv_x + 1) == 0) ? 1 : 0;
        NDR(i, 1) = (i % (numDiv_x + 1) == 0) ? 1 : 0;
        NDR(i, 2) = (i % (numDiv_x + 1) == 0) ? 1 : 0;

        //-COORDFlex are the initial coordinates for each node,
        // the first three are the position
        COORDFlex(i, 0) = (i % (numDiv_x + 1)) * dx;
        COORDFlex(i, 1) = (i / (numDiv_x + 1)) % (numDiv_y + 1) * dy;
        COORDFlex(i, 2) = (i) / ((numDiv_x + 1) * (numDiv_y + 1)) * dz;

        //-VELCYFlex is essentially the same as COORDFlex, but for the initial
        // velocity instead of position.
        // let's assume zero initial velocity for nodes
        VELCYFlex(i, 0) = 0;
        VELCYFlex(i, 1) = 0;
        VELCYFlex(i, 2) = 0;
    }

    // Adding the nodes to the mesh
    int i = 0;
    while (i < TotalNumNodes) {
        ChSharedPtr<ChNodeFEAxyz> node(new ChNodeFEAxyz(ChVector<>(COORDFlex(i, 0), COORDFlex(i, 1), COORDFlex(i, 2))));
        node->SetMass(0.0);

        my_mesh->AddNode(node);
        if (NDR(i, 0) == 1 && NDR(i, 1) == 1 && NDR(i, 2) == 1) {
            node->SetFixed(true);
        }
        i++;
    }

    ChSharedPtr<ChNodeFEAxyz> nodetip(my_mesh->GetNode(TotalNumNodes - 1).DynamicCastTo<ChNodeFEAxyz>());

    int elemcount = 0;
    while (elemcount < TotalNumElements) {
        ChSharedPtr<ChElementBrick> element(new ChElementBrick);
        ChMatrixNM<double, 3, 1> InertFlexVec;  // read element length, used in ChElementBrick
        InertFlexVec.Reset();
        InertFlexVec(0, 0) = ElemLengthXY(elemcount, 0);
        InertFlexVec(1, 0) = ElemLengthXY(elemcount, 1);
        InertFlexVec(2, 0) = ElemLengthXY(elemcount, 2);
        element->SetInertFlexVec(InertFlexVec);

        element->SetNodes(my_mesh->GetNode(NumNodes(elemcount, 0)).DynamicCastTo<ChNodeFEAxyz>(),
                          my_mesh->GetNode(NumNodes(elemcount, 1)).DynamicCastTo<ChNodeFEAxyz>(),
                          my_mesh->GetNode(NumNodes(elemcount, 2)).DynamicCastTo<ChNodeFEAxyz>(),
                          my_mesh->GetNode(NumNodes(elemcount, 3)).DynamicCastTo<ChNodeFEAxyz>(),
                          my_mesh->GetNode(NumNodes(elemcount, 4)).DynamicCastTo<ChNodeFEAxyz>(),
                          my_mesh->GetNode(NumNodes(elemcount, 5)).DynamicCastTo<ChNodeFEAxyz>(),
                          my_mesh->GetNode(NumNodes(elemcount, 6)).DynamicCastTo<ChNodeFEAxyz>(),
                          my_mesh->GetNode(NumNodes(elemcount, 7)).DynamicCastTo<ChNodeFEAxyz>());

        element->SetMaterial(mmaterial);
        element->SetElemNum(elemcount);            // for EAS
        ChMatrixNM<double, 9, 1> stock_alpha_EAS;  //
        stock_alpha_EAS.Reset();
        element->SetStockAlpha(stock_alpha_EAS(0, 0), stock_alpha_EAS(1, 0), stock_alpha_EAS(2, 0),
                               stock_alpha_EAS(3, 0), stock_alpha_EAS(4, 0), stock_alpha_EAS(5, 0),
                               stock_alpha_EAS(6, 0), stock_alpha_EAS(7, 0), stock_alpha_EAS(8, 0));
        my_mesh->AddElement(element);
        elemcount++;
    }

    // This is mandatory
    my_system.SetupInitial();

    // Remember to add the mesh to the system!
    my_system.Add(my_mesh);

    // Perform a dynamic time integration:
    my_system.SetLcpSolverType(ChSystem::LCP_ITERATIVE_MINRES);  // <- NEEDED because other solvers can't
    // handle stiffness matrices
    chrono::ChLcpIterativeMINRES* msolver = (chrono::ChLcpIterativeMINRES*)my_system.GetLcpSolverSpeed();
    msolver->SetDiagonalPreconditioning(true);
    my_system.SetIterLCPwarmStarting(true);  // this helps a lot to speedup convergence in this class of problems
    my_system.SetIterLCPmaxItersSpeed(10000);
    my_system.SetIterLCPmaxItersStab(10000);
    my_system.SetTolForce(1e-6);

    // INT_HHT or INT_EULER_IMPLICIT
    my_system.SetIntegrationType(ChSystem::INT_HHT);

    ChSharedPtr<ChTimestepperHHT> mystepper = my_system.GetTimestepper().DynamicCastTo<ChTimestepperHHT>();
    mystepper->SetAlpha(0.0);
    mystepper->SetMaxiters(10000);
    mystepper->SetTolerance(1e-06);
    mystepper->SetMode(ChTimestepperHHT::POSITION);
    mystepper->SetScaling(true);
    ChTimer<double> timer;
    // ---------------
    // Simulation loop
    // ---------------

    if (output) {
        // Create output directory (if it does not already exist).
        if (ChFileutils::MakeDirectory("../TEST") < 0) {
            GetLog() << "Error creating directory ../TEST\n";
            return 1;
        }

        // Initialize the output stream and set precision.
        utils::CSV_writer out("\t");

        out.stream().setf(std::ios::scientific | std::ios::showpos);
        out.stream().precision(6);
        double T_F = 6;
        // Simulate to final time, while saving position of tip node.
        while (my_system.GetChTime() < sim_time) {
            my_system.DoStepDynamics(step_size);
            double t_sim = my_system.GetChTime();
            if (t_sim < T_F)
                nodetip->SetForce(ChVector<>(0, 0, -50 / 2 * (1 - cos((t_sim / T_F) * 3.1415))));
            else {
                nodetip->SetForce(ChVector<>(0, 0, -50));
            }
            out << my_system.GetChTime() << nodetip->GetPos() << nodetip->GetForce().z << std::endl;
            //            GetLog() << "time = " << my_system.GetChTime() << "\t" << nodetip->GetPos().z << "\t"
            //                     << nodetip->GetForce().z << "\n";
        }

        // Write results to output file.
        out.write_to_file("../TEST/tip_position_brick.txt");
    } else {
        //        double startTime = omp_get_wtime();
        // Initialize total number of iterations and timer.
        int Iterations = 0;
        double start = std::clock();
        timer.start();
        // Simulate to final time, while accumulating number of iterations.
        while (my_system.GetChTime() < sim_time) {
            my_system.DoStepDynamics(step_size);
            Iterations += mystepper->GetNumIterations();
        }
        timer.stop();
        //        double endTime = omp_get_wtime();
        //        double seq_time = endTime - startTime;  // Report run time and total number of iterations.
        double duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
        GetLog() << "Simulation CPU Time: " << duration << "\n";
        GetLog() << "Wall clock time : " << timer() << "\n ";
        GetLog() << "Number of iterations: " << Iterations << "\n";
    }

    return 0;
}
