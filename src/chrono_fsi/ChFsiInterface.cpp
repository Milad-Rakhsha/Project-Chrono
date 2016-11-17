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
// Author: Arman Pazouki
// =============================================================================
//
// Base class for processing the interface between chrono and fsi modules
// =============================================================================

#include "chrono_fsi/ChFsiInterface.h"
#include "chrono_fsi/ChFsiTypeConvert.h"
#include "chrono_fea/ChMesh.h"
#include "chrono_fea/ChElementShellANCF.h"

namespace chrono {
namespace fsi {
//------------------------------------------------------------------------------------
ChFsiInterface::ChFsiInterface(FsiBodiesDataH* other_fsiBodiesH,
                               FsiShellsDataH* other_fsiShellsH,
                               chrono::ChSystem* other_mphysicalSystem,
                               std::vector<std::shared_ptr<chrono::ChBody>>* other_fsiBodeisPtr,
                               std::vector<std::shared_ptr<chrono::fea::ChElementShellANCF>>* other_fsiShellsPtr,
                               thrust::device_vector<Real3>* other_rigid_FSI_ForcesD,
                               thrust::device_vector<Real3>* other_rigid_FSI_TorquesD,
                               thrust::device_vector<Real3>* other_Flex_FSI_ForcesD_nA,
                               thrust::device_vector<Real3>* other_Flex_FSI_ForcesD_nB,
                               thrust::device_vector<Real3>* other_Flex_FSI_ForcesD_nC,
                               thrust::device_vector<Real3>* other_Flex_FSI_ForcesD_nD)
    : fsiBodiesH(other_fsiBodiesH),
      fsiShellsH(other_fsiShellsH),
      mphysicalSystem(other_mphysicalSystem),
      fsiBodeisPtr(other_fsiBodeisPtr),
      fsiShellsPtr(other_fsiShellsPtr),
      rigid_FSI_ForcesD(other_rigid_FSI_ForcesD),
      rigid_FSI_TorquesD(other_rigid_FSI_TorquesD),
      Flex_FSI_ForcesD_nA(other_Flex_FSI_ForcesD_nA),
      Flex_FSI_ForcesD_nB(other_Flex_FSI_ForcesD_nB),
      Flex_FSI_ForcesD_nC(other_Flex_FSI_ForcesD_nC),
      Flex_FSI_ForcesD_nD(other_Flex_FSI_ForcesD_nD) {
  int numBodies = mphysicalSystem->Get_bodylist()->size();
  int numShells = 0;
  if (mphysicalSystem->Get_otherphysicslist()->size())
    numShells = std::dynamic_pointer_cast<fea::ChMesh>(mphysicalSystem->Get_otherphysicslist()->at(0))->GetNelements();

  chronoRigidBackup = new ChronoBodiesDataH(numBodies);
  chronoFlexBackup = new ChronoShellsDataH(numShells);
  printf("** size other_fsiBodiesH %d \n ", other_fsiBodiesH->posRigid_fsiBodies_H.size());
  printf("** size other_fsiShellsH %d \n ", other_fsiShellsH->posFlex_fsiBodies_nA_H.size());
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

  int numShells = 0;
  auto my_mesh = std::make_shared<fea::ChMesh>();
  if (mphysicalSystem->Get_otherphysicslist()->size()) {
    my_mesh = std::dynamic_pointer_cast<fea::ChMesh>(mphysicalSystem->Get_otherphysicslist()->at(0));
    numShells = my_mesh->GetNelements();
  }

  for (int i = 0; i < numShells; i++) {
  }
}

void ChFsiInterface::Add_Flex_Forces_To_ChSystem() {
  int numShells = 0;
  auto my_mesh = std::make_shared<fea::ChMesh>();
  if (mphysicalSystem->Get_otherphysicslist()->size()) {
    my_mesh = std::dynamic_pointer_cast<fea::ChMesh>(mphysicalSystem->Get_otherphysicslist()->at(0));
    numShells = my_mesh->GetNelements();
  }
  int numFlex = fsiShellsPtr->size();

  if (numFlex != numShells)
    throw std::runtime_error(
        "Size of the external data does not match the "
        "ChSystem; thrown from Add_Flex_Forces_To_ChSystem "
        "!\n");

  for (int i = 0; i < numFlex; i++) {
    auto ShellPtr = (*fsiShellsPtr)[i];
    chrono::ChVector<> mforceA = ChFsiTypeConvert::Real3ToChVector((*Flex_FSI_ForcesD_nA)[i]);
    chrono::ChVector<> mforceB = ChFsiTypeConvert::Real3ToChVector((*Flex_FSI_ForcesD_nB)[i]);
    chrono::ChVector<> mforceC = ChFsiTypeConvert::Real3ToChVector((*Flex_FSI_ForcesD_nC)[i]);
    chrono::ChVector<> mforceD = ChFsiTypeConvert::Real3ToChVector((*Flex_FSI_ForcesD_nD)[i]);
    auto element = std::dynamic_pointer_cast<fea::ChElementShellANCF>(my_mesh->GetElement(i));
    element->GetNodeA()->SetForce(-mforceA);
    element->GetNodeB()->SetForce(-mforceB);
    element->GetNodeC()->SetForce(-mforceC);
    element->GetNodeD()->SetForce(-mforceD);
    chrono::ChVector<> TotalForce = (mforceA + mforceB + mforceC + mforceD);
    printf("Added the force of %f,%f,%f to the flex body %d\n", TotalForce.x, TotalForce.y, TotalForce.z, i);
  }
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

  printf("\n\n bodyList.size()=%d \n", bodyList->size());

  if (chronoRigidBackup->pos_ChSystemH.size() != numBodies) {
    throw std::runtime_error(
        "Size of the external data does not match the "
        "ChSystem; thrown from Copy_ChSystem_to_External "
        "!\n");
  }
  //	chronoRigidBackup->resize(numBodies);
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

  int numShells = 0;
  auto my_mesh = std::make_shared<fea::ChMesh>();
  if (mphysicalSystem->Get_otherphysicslist()->size()) {
    my_mesh = std::dynamic_pointer_cast<fea::ChMesh>(mphysicalSystem->Get_otherphysicslist()->at(0));
    numShells = my_mesh->GetNelements();
  }

  for (int i = 0; i < numShells; i++) {
    auto element = std::dynamic_pointer_cast<fea::ChElementShellANCF>(my_mesh->GetElement(i));

    chronoFlexBackup->posFlex_ChSystemH_nA_H[i] = ChFsiTypeConvert::ChVectorToReal3(element->GetNodeA()->GetPos());
    chronoFlexBackup->posFlex_ChSystemH_nB_H[i] = ChFsiTypeConvert::ChVectorToReal3(element->GetNodeB()->GetPos());
    chronoFlexBackup->posFlex_ChSystemH_nC_H[i] = ChFsiTypeConvert::ChVectorToReal3(element->GetNodeC()->GetPos());
    chronoFlexBackup->posFlex_ChSystemH_nD_H[i] = ChFsiTypeConvert::ChVectorToReal3(element->GetNodeD()->GetPos());

    chronoFlexBackup->velFlex_ChSystemH_nA_H[i] = ChFsiTypeConvert::ChVectorToReal3(element->GetNodeA()->GetPos_dt());
    chronoFlexBackup->velFlex_ChSystemH_nB_H[i] = ChFsiTypeConvert::ChVectorToReal3(element->GetNodeB()->GetPos_dt());
    chronoFlexBackup->velFlex_ChSystemH_nC_H[i] = ChFsiTypeConvert::ChVectorToReal3(element->GetNodeC()->GetPos_dt());
    chronoFlexBackup->velFlex_ChSystemH_nD_H[i] = ChFsiTypeConvert::ChVectorToReal3(element->GetNodeD()->GetPos_dt());

    chronoFlexBackup->accFlex_ChSystemH_nA_H[i] = ChFsiTypeConvert::ChVectorToReal3(element->GetNodeA()->GetPos_dtdt());
    chronoFlexBackup->accFlex_ChSystemH_nB_H[i] = ChFsiTypeConvert::ChVectorToReal3(element->GetNodeB()->GetPos_dtdt());
    chronoFlexBackup->accFlex_ChSystemH_nC_H[i] = ChFsiTypeConvert::ChVectorToReal3(element->GetNodeC()->GetPos_dtdt());
    chronoFlexBackup->accFlex_ChSystemH_nD_H[i] = ChFsiTypeConvert::ChVectorToReal3(element->GetNodeD()->GetPos_dtdt());
  }
}
//------------------------------------------------------------------------------------
// FSI_Bodies_Index_H[i] is the the index of the i_th sph represented rigid body
// in ChSystem
void ChFsiInterface::Copy_fsiBodies_ChSystem_to_FluidSystem(FsiBodiesDataD* fsiBodiesD) {
  //#pragma omp parallel for // Arman: you can bring it back later, when you
  // have a lot of bodies
  int num_fsiBodies_Rigids = fsiBodeisPtr->size();
  printf("\n\n fsiBodiesH->posRigid_fsiBodies_H.size() %d\n", fsiBodiesH->posRigid_fsiBodies_H.size());

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
  printf("\n\n CopyFromH Copy_fsiBodies_ChSystem_to_FluidSystem...\n");

  fsiBodiesD->CopyFromH(*fsiBodiesH);
}

void ChFsiInterface::Copy_fsiShells_ChSystem_to_FluidSystem(FsiShellsDataD* fsiShellsD) {
  int num_fsiBodies_Felx = fsiShellsPtr->size();
  ChVector<> centerFEA;
  for (int i = 0; i < num_fsiBodies_Felx; i++) {
    auto ShellPtr = (*fsiShellsPtr)[i];
    fsiShellsH->posFlex_fsiBodies_nA_H[i] = ChFsiTypeConvert::ChVectorToReal3(ShellPtr->GetNodeA()->GetPos());
    fsiShellsH->posFlex_fsiBodies_nB_H[i] = ChFsiTypeConvert::ChVectorToReal3(ShellPtr->GetNodeB()->GetPos());
    fsiShellsH->posFlex_fsiBodies_nC_H[i] = ChFsiTypeConvert::ChVectorToReal3(ShellPtr->GetNodeC()->GetPos());
    fsiShellsH->posFlex_fsiBodies_nD_H[i] = ChFsiTypeConvert::ChVectorToReal3(ShellPtr->GetNodeD()->GetPos());
    centerFEA = 0.25 * (ShellPtr->GetNodeA()->GetPos() + ShellPtr->GetNodeB()->GetPos() +
                        ShellPtr->GetNodeC()->GetPos() + ShellPtr->GetNodeD()->GetPos());
    fsiShellsH->velFlex_fsiBodies_nA_H[i] = ChFsiTypeConvert::ChVectorToReal3(ShellPtr->GetNodeA()->GetPos_dt());
    fsiShellsH->velFlex_fsiBodies_nB_H[i] = ChFsiTypeConvert::ChVectorToReal3(ShellPtr->GetNodeB()->GetPos_dt());
    fsiShellsH->velFlex_fsiBodies_nC_H[i] = ChFsiTypeConvert::ChVectorToReal3(ShellPtr->GetNodeC()->GetPos_dt());
    fsiShellsH->velFlex_fsiBodies_nD_H[i] = ChFsiTypeConvert::ChVectorToReal3(ShellPtr->GetNodeD()->GetPos_dt());

    fsiShellsH->accFlex_fsiBodies_nA_H[i] = ChFsiTypeConvert::ChVectorToReal3(ShellPtr->GetNodeA()->GetPos_dtdt());
    fsiShellsH->accFlex_fsiBodies_nB_H[i] = ChFsiTypeConvert::ChVectorToReal3(ShellPtr->GetNodeB()->GetPos_dtdt());
    fsiShellsH->accFlex_fsiBodies_nC_H[i] = ChFsiTypeConvert::ChVectorToReal3(ShellPtr->GetNodeC()->GetPos_dtdt());
    fsiShellsH->accFlex_fsiBodies_nD_H[i] = ChFsiTypeConvert::ChVectorToReal3(ShellPtr->GetNodeD()->GetPos_dtdt());
  }
  fsiShellsD->CopyFromH(*fsiShellsH);

  Real3 centerFSI = 0.25 * (fsiShellsD->posFlex_fsiBodies_nA_D[0] + fsiShellsD->posFlex_fsiBodies_nB_D[0] +
                            fsiShellsD->posFlex_fsiBodies_nC_D[0] + fsiShellsD->posFlex_fsiBodies_nD_D[0]);
  //
  //  printf("FlexBody[] centerFSI= %f,%f,%f,  centerFEA= %f,%f,%f\n", centerFSI.x, centerFSI.y, centerFSI.z,
  //  centerFEA.x,
  //         centerFEA.y, centerFEA.z);
}
//------------------------------------------------------------------------------------
void ChFsiInterface::ResizeChronoBodiesData() {
  int numBodies = mphysicalSystem->Get_bodylist()->size();
  chronoRigidBackup->resize(numBodies);
}

void ChFsiInterface::ResizeChronoShellsData() {
  int numShells = 0;
  auto my_mesh = std::make_shared<fea::ChMesh>();
  if (mphysicalSystem->Get_otherphysicslist()->size()) {
    my_mesh = std::dynamic_pointer_cast<fea::ChMesh>(mphysicalSystem->Get_otherphysicslist()->at(0));
    numShells = my_mesh->GetNelements();
  }

  printf("\n\n numShells in ResizeChronoShellsData  %d\n", numShells);

  chronoFlexBackup->resize(numShells);
}

}  // end namespace fsi
}  // end namespace chrono
