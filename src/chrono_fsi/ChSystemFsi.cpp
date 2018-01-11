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
// Author: Arman Pazouki, Milad Rakhsha
// =============================================================================
//
// Implementation of fsi system that includes all subclasses for proximity and
// force calculation, and time integration
// =============================================================================

#include "chrono_fsi/ChSystemFsi.h"

#ifdef CHRONO_FEA
#include "chrono_fea/ChElementCableANCF.h"
#include "chrono_fea/ChElementShellANCF.h"
#include "chrono_fea/ChMesh.h"
#include "chrono_fea/ChNodeFEAxyzD.h"
#endif

namespace chrono {
namespace fsi {

//--------------------------------------------------------------------------------------------------------------------------------

ChSystemFsi::ChSystemFsi(ChSystem* other_physicalSystem, bool other_haveFluid, ChFluidDynamics::Integrator type)
    : mphysicalSystem(other_physicalSystem), haveFluid(other_haveFluid), mTime(0) {
    fsiData = new ChFsiDataManager();
    paramsH = new SimParams;
    numObjectsH = &(fsiData->numObjects);
    fluidIntegrator = type;
    bceWorker = new ChBce(&(fsiData->sortedSphMarkersD), &(fsiData->markersProximityD), &(fsiData->fsiGeneralData),
                          paramsH, numObjectsH);
    fluidDynamics = new ChFluidDynamics(bceWorker, fsiData, paramsH, numObjectsH, fluidIntegrator);

#ifdef CHRONO_FEA
    fsi_mesh = std::make_shared<fea::ChMesh>();
    printf("fsi_mesh.sizeElements()=%d\n", fsi_mesh->GetNelements());
    printf("fsi_mesh.sizeNodes()..\n", fsi_mesh->GetNnodes());
    fsiBodeisPtr.resize(0);
    fsiShellsPtr.resize(0);
    fsiCablesPtr.resize(0);
    fsiNodesPtr.resize(0);
    fsiInterface = new ChFsiInterface(
        &(fsiData->fsiBodiesH), &(fsiData->fsiMeshH), mphysicalSystem, &fsiBodeisPtr, &fsiNodesPtr, &fsiCablesPtr,
        &fsiShellsPtr, fsi_mesh, &(fsiData->fsiGeneralData.CableElementsNodesH),
        &(fsiData->fsiGeneralData.CableElementsNodes), &(fsiData->fsiGeneralData.ShellElementsNodesH),
        &(fsiData->fsiGeneralData.ShellElementsNodes), &(fsiData->fsiGeneralData.rigid_FSI_ForcesD),
        &(fsiData->fsiGeneralData.rigid_FSI_TorquesD), &(fsiData->fsiGeneralData.Flex_FSI_ForcesD));
#else
    fsiInterface = new ChFsiInterface(
        &(fsiData->fsiBodiesH), mphysicalSystem, &fsiBodeisPtr, &(fsiData->fsiGeneralData.CableElementsNodesH),
        &(fsiData->fsiGeneralData.CableElementsNodes), &(fsiData->fsiGeneralData.ShellElementsNodesH),
        &(fsiData->fsiGeneralData.ShellElementsNodes), &(fsiData->fsiGeneralData.rigid_FSI_ForcesD),
        &(fsiData->fsiGeneralData.rigid_FSI_TorquesD), &(fsiData->fsiGeneralData.Flex_FSI_ForcesD));
#endif
}

//--------------------------------------------------------------------------------------------------------------------------------

void ChSystemFsi::Finalize() {
    printf("\n\n ChSystemFsi::Finalize 1-FinalizeData..\n");
    FinalizeData();

    if (haveFluid) {
        printf("\n\n ChSystemFsi::Finalize 2-bceWorker->Finalize..\n");
        bceWorker->Finalize(&(fsiData->sphMarkersD1), &(fsiData->fsiBodiesD1), &(fsiData->fsiMeshD));
        printf("\n\n ChSystemFsi::Finalize 3-fluidDynamics->Finalize..\n");
        fluidDynamics->Finalize();
        std::cout << "referenceArraySize in 3-fluidDynamics->Finalize.. "
                  << GetDataManager()->fsiGeneralData.referenceArray.size() << "\n";
    }
}
//--------------------------------------------------------------------------------------------------------------------------------

ChSystemFsi::~ChSystemFsi() {
    delete fsiData;
    delete paramsH;
    delete bceWorker;
    delete fluidDynamics;
    delete fsiInterface;
}
//--------------------------------------------------------------------------------------------------------------------------------
void ChSystemFsi::CopyDeviceDataToHalfStep() {
    thrust::copy(fsiData->sphMarkersD1.posRadD.begin(), fsiData->sphMarkersD1.posRadD.end(),
                 fsiData->sphMarkersD2.posRadD.begin());
    thrust::copy(fsiData->sphMarkersD1.velMasD.begin(), fsiData->sphMarkersD1.velMasD.end(),
                 fsiData->sphMarkersD2.velMasD.begin());
    thrust::copy(fsiData->sphMarkersD1.rhoPresMuD.begin(), fsiData->sphMarkersD1.rhoPresMuD.end(),
                 fsiData->sphMarkersD2.rhoPresMuD.begin());
}
//--------------------------------------------------------------------------------------------------------------------------------

void ChSystemFsi::DoStepDynamics_FSI() {
    fsiInterface->Copy_ChSystem_to_External();
    this->CopyDeviceDataToHalfStep();
    ChDeviceUtils::FillMyThrust4(fsiData->fsiGeneralData.derivVelRhoD, mR4(0));
    fluidDynamics->IntegrateSPH(&(fsiData->sphMarkersD2), &(fsiData->sphMarkersD1), &(fsiData->fsiBodiesD1),
                                0.5 * paramsH->dT);

    bceWorker->Rigid_Forces_Torques(&(fsiData->sphMarkersD1), &(fsiData->fsiBodiesD1));
    fsiInterface->Add_Rigid_ForceTorques_To_ChSystem();
    mTime += 0.5 * paramsH->dT;

    mphysicalSystem->DoStepDynamics(0.5 * paramsH->dT);

    fsiInterface->Copy_fsiBodies_ChSystem_to_FluidSystem(&(fsiData->fsiBodiesD2));
    bceWorker->UpdateRigidMarkersPositionVelocity(&(fsiData->sphMarkersD2), &(fsiData->fsiBodiesD2));

    fluidDynamics->IntegrateSPH(&(fsiData->sphMarkersD1), &(fsiData->sphMarkersD2), &(fsiData->fsiBodiesD2),
                                0.5 * paramsH->dT);

    bceWorker->Rigid_Forces_Torques(&(fsiData->sphMarkersD2), &(fsiData->fsiBodiesD2));

    fsiInterface->Add_Rigid_ForceTorques_To_ChSystem();

    mTime -= 0.5 * paramsH->dT;
    fsiInterface->Copy_External_To_ChSystem();
    mTime += paramsH->dT;

    mphysicalSystem->DoStepDynamics(0.5 * paramsH->dT);
    //
    fsiInterface->Copy_fsiBodies_ChSystem_to_FluidSystem(&(fsiData->fsiBodiesD1));
    bceWorker->UpdateRigidMarkersPositionVelocity(&(fsiData->sphMarkersD1), &(fsiData->fsiBodiesD1));

    // Density re-initialization
    int tStep = mTime / paramsH->dT;
    if ((tStep % 10 == 0) && (paramsH->densityReinit != 0)) {
        fluidDynamics->DensityReinitialization();
    }
}

void ChSystemFsi::DoStepDynamics_FSI_Implicit() {
    printf("Copy_ChSystem_to_External\n");
    fsiInterface->Copy_ChSystem_to_External();
    printf("IntegrateIISPH\n");
    fluidDynamics->IntegrateIISPH(&(fsiData->sphMarkersD2), &(fsiData->fsiBodiesD2), &(fsiData->fsiMeshD));
    printf("Calc nodal forces\n");
    bceWorker->Rigid_Forces_Torques(&(fsiData->sphMarkersD2), &(fsiData->fsiBodiesD2));
#ifdef CHRONO_FEA
    bceWorker->Flex_Forces(&(fsiData->sphMarkersD2), &(fsiData->fsiMeshD));
    printf("DataTransfer...(Nodal force from device to host)\n");
    // Note that because of applying forces to the nodal coordinates using SetForce() no other external forces can be
    // applied, or if any thing has been applied will be rewritten by Add_Flex_Forces_To_ChSystem();
    fsiInterface->Add_Flex_Forces_To_ChSystem();
#endif
    fsiInterface->Add_Rigid_ForceTorques_To_ChSystem();

    mTime += 1 * paramsH->dT;
    if (paramsH->dT_Flex == 0)
        paramsH->dT_Flex = paramsH->dT;
    int sync = (paramsH->dT / paramsH->dT_Flex);
    if (sync < 1)
        sync = 1;
    printf("%d * DoStepChronoSystem with dt= %f\n", sync, paramsH->dT_Flex);
    for (int t = 0; t < sync; t++) {
        mphysicalSystem->DoStepDynamics(paramsH->dT / sync);
    }
#ifdef CHRONO_FEA
    printf("DataTransfer...(Flexible pos-vel-acc from host to device)\n");
    fsiInterface->Copy_fsiNodes_ChSystem_to_FluidSystem(&(fsiData->fsiMeshD));
#endif
    fsiInterface->Copy_fsiBodies_ChSystem_to_FluidSystem(&(fsiData->fsiBodiesD2));
    printf("Update Marker\n");

    bceWorker->UpdateRigidMarkersPositionVelocity(&(fsiData->sphMarkersD2), &(fsiData->fsiBodiesD2));

#ifdef CHRONO_FEA
    bceWorker->UpdateFlexMarkersPositionVelocity(&(fsiData->sphMarkersD2), &(fsiData->fsiMeshD));
#endif

    printf("=================================================================================================\n");
}
//--------------------------------------------------------------------------------------------------------------------------------
void ChSystemFsi::DoStepDynamics_ChronoRK2() {
    fsiInterface->Copy_ChSystem_to_External();
    mTime += 0.5 * paramsH->dT;

    mphysicalSystem->DoStepDynamics(0.5 * paramsH->dT);
    mTime -= 0.5 * paramsH->dT;
    fsiInterface->Copy_External_To_ChSystem();
    mTime += paramsH->dT;
    mphysicalSystem->DoStepDynamics(1.0 * paramsH->dT);
}

void SetIntegratorType(ChFluidDynamics::Integrator type) {}

//--------------------------------------------------------------------------------------------------------------------------------
void ChSystemFsi::FinalizeData() {
    printf("\n\nfsiInterface->ResizeChronoBodiesData()\n");
    fsiInterface->ResizeChronoBodiesData();
    int fea_node = 0;
#ifdef CHRONO_FEA
    fsiInterface->ResizeChronoCablesData(CableElementsNodes, &(fsiData->fsiGeneralData.CableElementsNodesH));
    fsiInterface->ResizeChronoShellsData(ShellElementsNodes, &(fsiData->fsiGeneralData.ShellElementsNodesH));
    fsiInterface->ResizeChronoFEANodesData();
    printf("\nfsiData->ResizeDataManager...\n");
    fea_node = fsi_mesh->GetNnodes();

#endif
    fsiData->ResizeDataManager(fea_node);

    printf("\n\nfsiInterface->Copy_fsiBodies_ChSystem_to_FluidSystem()\n");
    fsiInterface->Copy_fsiBodies_ChSystem_to_FluidSystem(&(fsiData->fsiBodiesD1));  //(1)

#ifdef CHRONO_FEA
    fsiInterface->Copy_fsiNodes_ChSystem_to_FluidSystem(&(fsiData->fsiMeshD));
#endif

    std::cout << "referenceArraySize in FinalizeData " << GetDataManager()->fsiGeneralData.referenceArray.size()
              << "\n";
    fsiData->fsiBodiesD2 = fsiData->fsiBodiesD1;  //(2) construct midpoint rigid data
}
//--------------------------------------------------------------------------------------------------------------------------------

}  // end namespace fsi
}  // end namespace chrono
