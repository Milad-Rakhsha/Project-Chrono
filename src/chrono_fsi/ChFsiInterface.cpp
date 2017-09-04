// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2014 projectchrono.org
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Author: Arman Pazouki
// =============================================================================
//
// Base class for processing the interface between chrono and fsi modules
// =============================================================================

#include "chrono_fsi/ChFsiInterface.h"
#include "chrono_fsi/ChFsiTypeConvert.h"
#include "chrono_fea/ChMesh.h"
#include "chrono_fea/ChNodeFEAxyzD.h"
#include "chrono_fea/ChElementCableANCF.h"
#include "chrono_fea/ChElementShellANCF.h"

namespace chrono {
namespace fsi {
//------------------------------------------------------------------------------------
ChFsiInterface::ChFsiInterface(FsiBodiesDataH* other_fsiBodiesH,
                               FsiMeshDataH* other_fsiMeshH,
                               chrono::ChSystem* other_mphysicalSystem,
                               std::vector<std::shared_ptr<chrono::ChBody>>* other_fsiBodeisPtr,
                               std::vector<std::shared_ptr<chrono::fea::ChNodeFEAxyzD>>* other_fsiNodesPtr,
                               std::vector<std::shared_ptr<chrono::fea::ChElementCableANCF>>* other_fsiCablesPtr,
                               std::vector<std::shared_ptr<chrono::fea::ChElementShellANCF>>* other_fsiShellsPtr,
                               std::shared_ptr<chrono::fea::ChMesh> other_fsiMesh,
                               thrust::host_vector<int2>* other_CableElementsNodesH,
                               thrust::device_vector<int2>* other_CableElementsNodes,
                               thrust::host_vector<int4>* other_ShellElementsNodesH,
                               thrust::device_vector<int4>* other_ShellElementsNodes,
                               thrust::device_vector<Real3>* other_rigid_FSI_ForcesD,
                               thrust::device_vector<Real3>* other_rigid_FSI_TorquesD,
                               thrust::device_vector<Real3>* other_Flex_FSI_ForcesD)
    : mphysicalSystem(other_mphysicalSystem),
      fsiBodeisPtr(other_fsiBodeisPtr),
      fsiBodiesH(other_fsiBodiesH),
      fsiMeshH(other_fsiMeshH),
      fsi_mesh(other_fsiMesh),
      fsiNodesPtr(other_fsiNodesPtr),
      fsiCablesPtr(other_fsiCablesPtr),
      fsiShellsPtr(other_fsiShellsPtr),
      rigid_FSI_ForcesD(other_rigid_FSI_ForcesD),
      rigid_FSI_TorquesD(other_rigid_FSI_TorquesD),
      CableElementsNodesH(other_CableElementsNodesH),
      CableElementsNodes(other_CableElementsNodes),
      ShellElementsNodesH(other_ShellElementsNodesH),
      ShellElementsNodes(other_ShellElementsNodes),
      Flex_FSI_ForcesD(other_Flex_FSI_ForcesD) {
    int numBodies = mphysicalSystem->Get_bodylist()->size();
    int numShells = 0;
    int numCables = 0;

    int numNodes = 0;

    if (mphysicalSystem->Get_otherphysicslist()->size())
        numShells =
            std::dynamic_pointer_cast<fea::ChMesh>(mphysicalSystem->Get_otherphysicslist()->at(0))->GetNelements();

    if (mphysicalSystem->Get_otherphysicslist()->size())
        numNodes = std::dynamic_pointer_cast<fea::ChMesh>(mphysicalSystem->Get_otherphysicslist()->at(0))->GetNnodes();

    chronoRigidBackup = new ChronoBodiesDataH(numBodies);
    chronoFlexMeshBackup = new ChronoMeshDataH(numNodes);

    printf("** size other_fsiBodiesH %d \n ", other_fsiBodiesH->posRigid_fsiBodies_H.size());
    printf("** size chronoRigidBackup %d \n ", chronoRigidBackup->pos_ChSystemH.size());
}

//------------------------------------------------------------------------------------
ChFsiInterface::~ChFsiInterface() {}
//------------------------------------------------------------------------------------
// FSI_Bodies_Index_H[i] is the the index of the i_th sph represented rigid body
// in ChSystem
void ChFsiInterface::Add_Rigid_ForceTorques_To_ChSystem() {
    int numRigids = fsiBodeisPtr->size();
    //#pragma omp parallel for // Arman: you can bring it back later, when you
    // have a lot of bodies
    for (int i = 0; i < numRigids; i++) {
        auto bodyPtr = (*fsiBodeisPtr)[i];

        //		// --------------------------------
        //		// Add forces to bodies: Version 1
        //		// --------------------------------
        //
        //		bodyPtr->Empty_forces_accumulators();
        //		Real3 mforce = (*rigid_FSI_ForcesD)[i];
        //
        //		printf("\n\n\n\n\n\n\n rigid forces %e %e %e \n", mforce.x,
        // mforce.y,
        //				mforce.z);
        //		std::cout << "body name: " << bodyPtr->GetName() <<
        //"\n\n\n\n\n";
        //		bodyPtr->Empty_forces_accumulators();
        //
        //		bodyPtr->Accumulate_force(ChFsiTypeConvert::Real3ToChVector(mforce),
        //				bodyPtr->GetPos(), false);
        //
        //		Real3 mtorque = (*rigid_FSI_TorquesD)[i];
        //		bodyPtr->Accumulate_torque(ChFsiTypeConvert::Real3ToChVector(mtorque),
        // false);

        // --------------------------------
        // Add forces to bodies: Version 2
        // --------------------------------

        //	string forceTag("hydrodynamics_force");
        char forceTag[] = "fsi_force";
        char torqueTag[] = "fsi_torque";
        auto hydroForce = bodyPtr->SearchForce(forceTag);
        auto hydroTorque = bodyPtr->SearchForce(torqueTag);

        if (!hydroForce) {
            hydroForce = std::make_shared<chrono::ChForce>();
            hydroTorque = std::make_shared<chrono::ChForce>();

            hydroForce->SetMode(ChForce::FORCE);
            hydroTorque->SetMode(ChForce::TORQUE);

            hydroForce->SetName(forceTag);
            hydroTorque->SetName(torqueTag);

            bodyPtr->AddForce(hydroForce);
            bodyPtr->AddForce(hydroTorque);
        }

        chrono::ChVector<> mforce = ChFsiTypeConvert::Real3ToChVector((*rigid_FSI_ForcesD)[i]);
        chrono::ChVector<> mtorque = ChFsiTypeConvert::Real3ToChVector((*rigid_FSI_TorquesD)[i]);

        hydroForce->SetVpoint(bodyPtr->GetPos());
        hydroForce->SetMforce(mforce.Length());
        mforce.Normalize();
        hydroForce->SetDir(mforce);

        hydroTorque->SetMforce(mtorque.Length());
        mtorque.Normalize();
        hydroTorque->SetDir(mtorque);
    }
}

void ChFsiInterface::Add_Flex_Forces_To_ChSystem() {
    int numShells = 0;
    int numNodes_ChSystem = 0;

    int numNodes = fsiNodesPtr->size();
    chrono::ChVector<> total_force(0, 0, 0);
    for (int i = 0; i < numNodes; i++) {
        chrono::ChVector<> mforce = ChFsiTypeConvert::Real3ToChVector((*Flex_FSI_ForcesD)[i]);
        // if (mforce.Length() != 0.0)
        //      printf("Added the force of %f,%f,%f to the flex nodes %d\n", mforce.x, mforce.y, mforce.z, i);
        auto node = std::dynamic_pointer_cast<fea::ChNodeFEAxyzD>(fsi_mesh->GetNode(i));
        //    ChVector<> OldForce = node->GetForce();
        node->SetForce(-mforce);

        total_force += mforce;
    }

    printf("Total Force from the fluid to the solid = (%f,%f,%f)\n", total_force.x(), total_force.y(), total_force.z());
}

//------------------------------------------------------------------------------------
// FSI_Bodies_Index_H[i] is the the index of the i_th sph represented rigid body
// in ChSystem
void ChFsiInterface::Copy_External_To_ChSystem() {
    int numBodies = mphysicalSystem->Get_bodylist()->size();
    if (chronoRigidBackup->pos_ChSystemH.size() != numBodies) {
        throw std::runtime_error(
            "Size of the external data does not match the "
            "ChSystem; thrown from Copy_External_To_ChSystem "
            "!\n");
    }
    //#pragma omp parallel for // Arman: you can bring it back later, when you
    // have a lot of bodies
    for (int i = 0; i < numBodies; i++) {
        auto mBody = mphysicalSystem->Get_bodylist()->at(i);
        mBody->SetPos(ChFsiTypeConvert::Real3ToChVector(chronoRigidBackup->pos_ChSystemH[i]));
        mBody->SetPos_dt(ChFsiTypeConvert::Real3ToChVector(chronoRigidBackup->vel_ChSystemH[i]));
        mBody->SetPos_dtdt(ChFsiTypeConvert::Real3ToChVector(chronoRigidBackup->acc_ChSystemH[i]));

        mBody->SetRot(ChFsiTypeConvert::Real4ToChQuaternion(chronoRigidBackup->quat_ChSystemH[i]));
        mBody->SetWvel_par(ChFsiTypeConvert::Real3ToChVector(chronoRigidBackup->omegaVelGRF_ChSystemH[i]));
        chrono::ChVector<> acc = ChFsiTypeConvert::Real3ToChVector(chronoRigidBackup->omegaAccGRF_ChSystemH[i]);
        mBody->SetWacc_par(acc);
    }
}
//------------------------------------------------------------------------------------
void ChFsiInterface::Copy_ChSystem_to_External() {
    //	// Arman, assume no change in chrono num bodies. the resize is done in
    // initializaiton.

    int numBodies = mphysicalSystem->Get_bodylist()->size();
    auto bodyList = mphysicalSystem->Get_bodylist();

    if (chronoRigidBackup->pos_ChSystemH.size() != numBodies) {
        throw std::runtime_error(
            "Size of the external data does not match the "
            "ChSystem; thrown from Copy_ChSystem_to_External "
            "!\n");
    }

    chronoRigidBackup->resize(numBodies);

    //#pragma omp parallel for // Arman: you can bring it back later, when you
    // have a lot of bodies
    for (int i = 0; i < numBodies; i++) {
        auto mBody = mphysicalSystem->Get_bodylist()->at(i);
        chronoRigidBackup->pos_ChSystemH[i] = ChFsiTypeConvert::ChVectorToReal3(mBody->GetPos());
        chronoRigidBackup->vel_ChSystemH[i] = ChFsiTypeConvert::ChVectorToReal3(mBody->GetPos_dt());
        chronoRigidBackup->acc_ChSystemH[i] = ChFsiTypeConvert::ChVectorToReal3(mBody->GetPos_dtdt());

        chronoRigidBackup->quat_ChSystemH[i] = ChFsiTypeConvert::ChQuaternionToReal4(mBody->GetRot());
        chronoRigidBackup->omegaVelGRF_ChSystemH[i] = ChFsiTypeConvert::ChVectorToReal3(mBody->GetWvel_par());
        chronoRigidBackup->omegaAccGRF_ChSystemH[i] = ChFsiTypeConvert::ChVectorToReal3(mBody->GetWacc_par());
    }

    int numNodes = fsi_mesh->GetNnodes();

    //  int numCables = 0;
    //  for (int i = 0; i < fsi_mesh->GetNelements(); i++) {
    //    if (std::dynamic_pointer_cast<fea::ChElementCableANCF>(fsi_mesh->GetElement(i)))
    //      numCables++;
    //  }
    //
    //  int numShells = 0;
    //  for (int i = 0; i < fsi_mesh->GetNelements(); i++) {
    //    if (std::dynamic_pointer_cast<fea::ChElementShellANCF>(fsi_mesh->GetElement(i)))
    //      numShells++;
    //  }

    for (int i = 0; i < numNodes; i++) {
        auto node = std::dynamic_pointer_cast<fea::ChNodeFEAxyz>(fsi_mesh->GetNode(i));
        chronoFlexMeshBackup->posFlex_ChSystemH_H[i] = ChFsiTypeConvert::ChVectorToReal3(node->GetPos());
        chronoFlexMeshBackup->velFlex_ChSystemH_H[i] = ChFsiTypeConvert::ChVectorToReal3(node->GetPos_dt());
        chronoFlexMeshBackup->accFlex_ChSystemH_H[i] = ChFsiTypeConvert::ChVectorToReal3(node->GetPos_dtdt());
    }
}
//------------------------------------------------------------------------------------
// FSI_Bodies_Index_H[i] is the the index of the i_th sph represented rigid body
// in ChSystem
void ChFsiInterface::Copy_fsiBodies_ChSystem_to_FluidSystem(FsiBodiesDataD* fsiBodiesD) {
    //#pragma omp parallel for // Arman: you can bring it back later, when you
    // have a lot of bodies
    int num_fsiBodies_Rigids = fsiBodeisPtr->size();
    //  printf("\n\n fsiBodiesH->posRigid_fsiBodies_H.size() %d\n", fsiBodiesH->posRigid_fsiBodies_H.size());

    for (int i = 0; i < num_fsiBodies_Rigids; i++) {
        auto bodyPtr = (*fsiBodeisPtr)[i];
        fsiBodiesH->posRigid_fsiBodies_H[i] = ChFsiTypeConvert::ChVectorToReal3(bodyPtr->GetPos());
        fsiBodiesH->velMassRigid_fsiBodies_H[i] =
            ChFsiTypeConvert::ChVectorRToReal4(bodyPtr->GetPos_dt(), bodyPtr->GetMass());
        fsiBodiesH->accRigid_fsiBodies_H[i] = ChFsiTypeConvert::ChVectorToReal3(bodyPtr->GetPos_dtdt());

        fsiBodiesH->q_fsiBodies_H[i] = ChFsiTypeConvert::ChQuaternionToReal4(bodyPtr->GetRot());
        fsiBodiesH->omegaVelLRF_fsiBodies_H[i] = ChFsiTypeConvert::ChVectorToReal3(bodyPtr->GetWvel_loc());
        fsiBodiesH->omegaAccLRF_fsiBodies_H[i] = ChFsiTypeConvert::ChVectorToReal3(bodyPtr->GetWacc_loc());
    }
    //  printf("\n\n CopyFromH Copy_fsiBodies_ChSystem_to_FluidSystem...\n");

    fsiBodiesD->CopyFromH(*fsiBodiesH);
}

void ChFsiInterface::Copy_fsiNodes_ChSystem_to_FluidSystem(FsiMeshDataD* FsiMeshD) {
    int num_fsiNodes_Felx = fsiNodesPtr->size();

    for (int i = 0; i < num_fsiNodes_Felx; i++) {
        auto NodePtr = (*fsiNodesPtr)[i];
        fsiMeshH->pos_fsi_fea_H[i] = ChFsiTypeConvert::ChVectorToReal3(NodePtr->GetPos());
        fsiMeshH->vel_fsi_fea_H[i] = ChFsiTypeConvert::ChVectorToReal3(NodePtr->GetPos_dt());
        fsiMeshH->acc_fsi_fea_H[i] = ChFsiTypeConvert::ChVectorToReal3(NodePtr->GetPos_dtdt());
    }
    FsiMeshD->CopyFromH(*fsiMeshH);
}

//------------------------------------------------------------------------------------
void ChFsiInterface::ResizeChronoBodiesData() {
    int numBodies = mphysicalSystem->Get_bodylist()->size();
    chronoRigidBackup->resize(numBodies);
}

void ChFsiInterface::ResizeChronoNodesData() {
    int numNodes = 0;
    auto my_mesh = std::make_shared<fea::ChMesh>();
    if (mphysicalSystem->Get_otherphysicslist()->size()) {
        printf("fsi_mesh.size in ResizeChronNodesData  %d\n", numNodes);
    }
    numNodes = fsi_mesh->GetNnodes();
    printf("numNodes in ResizeChronNodesData  %d\n", numNodes);
    chronoFlexMeshBackup->resize(numNodes);
}
//------------------------------------------------------------------------------------

void ChFsiInterface::ResizeChronoFEANodesData() {
    int numNodes = 0;
    auto my_mesh = std::make_shared<fea::ChMesh>();
    if (mphysicalSystem->Get_otherphysicslist()->size()) {
        my_mesh = std::dynamic_pointer_cast<fea::ChMesh>(mphysicalSystem->Get_otherphysicslist()->at(0));
    }
    numNodes = fsi_mesh->GetNnodes();
    printf("fsi_mesh.size in ResizeChronoFEANodesData  %d\n", numNodes);

    printf("numNodes in ResizeChronoFEANodeData  %d\n", numNodes);

    chronoFlexMeshBackup->resize(numNodes);
}
//------------------------------------------------------------------------------------

void ChFsiInterface::ResizeChronoCablesData(std::vector<std::vector<int>> CableElementsNodesSTDVector,
                                            thrust::host_vector<int2>* CableElementsNodesH) {
    auto my_mesh = std::make_shared<fea::ChMesh>();
    if (mphysicalSystem->Get_otherphysicslist()->size()) {
        my_mesh = std::dynamic_pointer_cast<fea::ChMesh>(mphysicalSystem->Get_otherphysicslist()->at(0));
    }

    int numCables = 0;
    for (int i = 0; i < fsi_mesh->GetNelements(); i++) {
        if (std::dynamic_pointer_cast<fea::ChElementCableANCF>(fsi_mesh->GetElement(i)))
            numCables++;
    }
    printf("fsi_mesh.size.shell in ResizeChronoCablesData  %d\n", numCables);
    printf("numCables in ResizeChronoCablesData  %d\n", numCables);
    printf("CableElementsNodesSTDVector.size() in ResizeChronoCablesData  %d\n", CableElementsNodesSTDVector.size());

    if (CableElementsNodesSTDVector.size() != numCables) {
        throw std::runtime_error(
            "Size of the external data does not match the "
            "ChSystem; thrown from ChFsiInterface::ResizeChronoCableData "
            "!\n");
    }

    //  ShellElementsNodesH is the elements connectivity
    // Important: in CableElementsNodesH[i][j] j index starts from 1 not zero
    // This is because of how the GMF files are read in chrono
    CableElementsNodesH->resize(numCables);
    for (int i = 0; i < numCables; i++) {
        (*CableElementsNodesH)[i].x = CableElementsNodesSTDVector[i][0];
        (*CableElementsNodesH)[i].y = CableElementsNodesSTDVector[i][1];
    }
}
//------------------------------------------------------------------------------------

void ChFsiInterface::ResizeChronoShellsData(std::vector<std::vector<int>> ShellElementsNodesSTDVector,
                                            thrust::host_vector<int4>* ShellElementsNodesH) {
    auto my_mesh = std::make_shared<fea::ChMesh>();
    if (mphysicalSystem->Get_otherphysicslist()->size()) {
        my_mesh = std::dynamic_pointer_cast<fea::ChMesh>(mphysicalSystem->Get_otherphysicslist()->at(0));
    }

    int numShells = 0;
    for (int i = 0; i < fsi_mesh->GetNelements(); i++) {
        if (std::dynamic_pointer_cast<fea::ChElementShellANCF>(fsi_mesh->GetElement(i)))
            numShells++;
    }

    printf("\n\n numShells in ResizeChronoShellsData  %d\n", numShells);
    printf("ShellElementsNodesSTDVector.size() in ResizeChronoShellsData  %d\n", ShellElementsNodesSTDVector.size());

    if (ShellElementsNodesSTDVector.size() != numShells) {
        throw std::runtime_error(
            "Size of the external data does not match the "
            "ChSystem; thrown from ChFsiInterface::ResizeChronoShellsData "
            "!\n");
    }

    //  ShellElementsNodesH is the elements connectivity
    // Important: in ShellElementsNodesH[i][j] j index starts from 1 not zero
    // This is because of how the GMF files are read in chrono
    ShellElementsNodesH->resize(numShells);
    for (int i = 0; i < numShells; i++) {
        (*ShellElementsNodesH)[i].x = ShellElementsNodesSTDVector[i][0];
        (*ShellElementsNodesH)[i].y = ShellElementsNodesSTDVector[i][1];
        (*ShellElementsNodesH)[i].z = ShellElementsNodesSTDVector[i][2];
        (*ShellElementsNodesH)[i].w = ShellElementsNodesSTDVector[i][3];
    }

    //  (*ShellelementsNodes).resize(numShells);
    //  thrust::copy(ShellelementsNodes_H.begin(), ShellelementsNodes_H.end(), (*ShellElementsNodes).begin());
}

//------------------------------------------------------------------------------------

}  // end namespace fsi
}  // end namespace chrono
