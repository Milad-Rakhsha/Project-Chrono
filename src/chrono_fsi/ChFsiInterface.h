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
#ifndef CH_FSIINTERFACE_H_
#define CH_FSIINTERFACE_H_

#include "chrono/physics/ChSystem.h"
#include "chrono_fsi/ChApiFsi.h"
#include "chrono_fsi/ChFsiDataManager.cuh"
#include "chrono_fsi/ChFsiGeneral.cuh"

namespace chrono {
namespace fsi {

class CH_FSI_API ChFsiInterface : public ChFsiGeneral {
 public:
  ChFsiInterface(FsiBodiesDataH* other_fsiBodiesH,
                 FsiShellsDataH* other_fsiShellsH,
                 FsiMeshDataH* other_fsiMeshH,
                 chrono::ChSystem* other_mphysicalSystem,
                 std::vector<std::shared_ptr<chrono::ChBody>>* other_fsiBodeisPtr,
                 std::vector<std::shared_ptr<chrono::fea::ChNodeFEAxyzD>>* other_fsiNodesPtr,
                 std::vector<std::shared_ptr<chrono::fea::ChElementShellANCF>>* other_fsiShellsPtr,
                 thrust::host_vector<int4>* other_ShellelementsNodesH,
                 thrust::device_vector<int4>* other_ShellelementsNodes,
                 thrust::device_vector<Real3>* other_rigid_FSI_ForcesD,
                 thrust::device_vector<Real3>* other_rigid_FSI_TorquesD,
                 thrust::device_vector<Real3>* other_Flex_FSI_ForcesD);

  virtual ~ChFsiInterface();  // TODO

  virtual void Add_Rigid_ForceTorques_To_ChSystem();
  virtual void Add_Flex_Forces_To_ChSystem();

  virtual void Copy_External_To_ChSystem();
  virtual void Copy_ChSystem_to_External();
  virtual void GetFsiMesh(std::shared_ptr<chrono::fea::ChMesh> other_fsi_mesh) { fsi_mesh = other_fsi_mesh; };
  virtual void Copy_fsiBodies_ChSystem_to_FluidSystem(FsiBodiesDataD* fsiBodiesD);
  virtual void Copy_fsiShells_ChSystem_to_FluidSystem(FsiShellsDataD* fsiShellsD);
  virtual void Copy_fsiNodes_ChSystem_to_FluidSystem(FsiMeshDataD* FsiMeshD);
  virtual void ResizeChronoBodiesData();
  virtual void ResizeChronoNodesData();
  virtual void ResizeChronoShellsData(std::vector<std::vector<int>> ShellelementsNodesSTDVector,
                                      thrust::host_vector<int4>* ShellelementsNodesH);
  virtual void ResizeChronoFEANodesData();

 private:
  FsiBodiesDataH* fsiBodiesH;
  ChronoBodiesDataH* chronoRigidBackup;

  FsiShellsDataH* fsiShellsH;
  ChronoShellsDataH* chronoFlexBackup;

  FsiMeshDataH* fsiMeshH;
  ChronoMeshDataH* chronoFlexMeshBackup;

  chrono::ChSystem* mphysicalSystem;

  std::vector<std::shared_ptr<chrono::ChBody>>* fsiBodeisPtr;
  std::vector<std::shared_ptr<chrono::fea::ChElementShellANCF>>* fsiShellsPtr;
  std::vector<std::shared_ptr<chrono::fea::ChNodeFEAxyzD>>* fsiNodesPtr;  // These are all the FE nodes available to fsi
  std::shared_ptr<chrono::fea::ChMesh> fsi_mesh;                          // These are all the nodes available to fsi

  thrust::host_vector<int4>* ShellelementsNodesH;   // These are the indices of nodes of each Element
  thrust::device_vector<int4>* ShellelementsNodes;  // These are the indices of nodes of each Element

  thrust::device_vector<Real3>* rigid_FSI_ForcesD;
  thrust::device_vector<Real3>* rigid_FSI_TorquesD;

  thrust::device_vector<Real3>* Flex_FSI_ForcesD;
};
}  // end namespace fsi
}  // end namespace chrono
#endif
