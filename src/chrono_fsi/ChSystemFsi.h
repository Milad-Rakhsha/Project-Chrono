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
// Implementation of FSI system that includes all subclasses for proximity and
// force calculation, and time integration.
//
// =============================================================================

#ifndef CH_SYSTEMFSI_H_
#define CH_SYSTEMFSI_H_

#include "chrono/ChConfig.h"
#include "chrono/physics/ChSystem.h"
#include "chrono_fea/ChElementCableANCF.h"
#include "chrono_fea/ChElementShellANCF.h"
#include "chrono_fea/ChMesh.h"
#include "chrono_fea/ChNodeFEAxyzD.h"
#include "chrono_fsi/ChBce.cuh"
//#include "chrono_fsi/ChDeviceUtils.cuh"
#include "chrono_fsi/ChFluidDynamics.cuh"
#include "chrono_fsi/ChFsiDataManager.cuh"
//#include "chrono_fsi/ChFsiGeneral.cuh"
#include "chrono_fsi/ChFsiInterface.h"

namespace chrono {
namespace fsi {

/// @addtogroup fsi_physics
/// @{

/// @brief Physical system for fluid-solid interaction problem.
///
/// This class is used to represent a fluid-solid interaction problem consist of
/// fluid dynamics and multibody system. Each of the two underlying physics are
/// independent objects owned and instantiated by this class. Additionally,
/// the fsi system owns other objects to handle the interface between the two
/// systems, boundary condition enforcing markers, and data.
class CH_FSI_API ChSystemFsi : public ChFsiGeneral {
  public:
    /// Constructor for FSI system.
    /// This class constructor instantiates all the member objects. Wherever relevant, the
    /// instantiation is handled by sending a pointer to other objects or data.
    /// Therefore, the sub-classes have pointers to the same data.
    ChSystemFsi(ChSystem* other_physicalSystem, bool other_haveFluid);

    /// Destructor for the FSI system.
    ~ChSystemFsi();

    /// Function to integrate the fsi system in time.
    /// It uses a Runge-Kutta 2nd order algorithm to update both the fluid and multibody
    /// system dynamics. The midpoint data of MBS is needed for fluid dynamics update.
    virtual void DoStepDynamics_FSI();
    virtual void DoStepDynamics_FSI_Implicit();
    void SetFluidIntegratorType(ChFluidDynamics::Integrator type) { fluidIntegrator = type; }

    /// Function to integrate the multibody system dynamics based on Runge-Kutta
    /// 2nd-order integration scheme.
    virtual void DoStepDynamics_ChronoRK2();

    /// Function to initialize the midpoint device data of the fluid system by
    /// copying from the full step
    virtual void CopyDeviceDataToHalfStep();

    /// Fill out the dependent data based on the independent one. For instance,
    /// it copies the position of the rigid bodies from the multibody dynamics system
    /// to arrays in fsi system since they are needed for internal use.
    virtual void FinalizeData();

    /// Get a pointer to the data manager.
    ChFsiDataManager* GetDataManager() { return fsiData; }

    /// Get a pointer to the parameters used to set up the simulation.
    SimParams* GetSimParams() { return paramsH; }

    /// Get a pointer to the fsi bodies.
    /// fsi bodies are the ones seen by the fluid dynamics system.
    /// These ChBodies are stored in a std::vector using their shared pointers.
    std::vector<std::shared_ptr<ChBody>>* GetFsiBodiesPtr() { return &fsiBodeisPtr; }
    std::vector<std::shared_ptr<chrono::fea::ChElementCableANCF>>* GetFsiCablesPtr() { return &fsiCablesPtr; }
    std::vector<std::shared_ptr<chrono::fea::ChElementShellANCF>>* GetFsiShellsPtr() { return &fsiShellsPtr; }
    std::vector<std::shared_ptr<chrono::fea::ChNodeFEAxyzD>>* GetFsiNodesPtr() { return &fsiNodesPtr; }

    /// Finalize the construction of the fsi system.
    void SetCableElementsNodes(std::vector<std::vector<int>> elementsNodes) {
        CableElementsNodes = elementsNodes;
        int test = fsiData->fsiGeneralData.CableElementsNodes.size();
        std::cout << "numObjects.numFlexNodes" << test << std::endl;
    }

    void SetShellElementsNodes(std::vector<std::vector<int>> elementsNodes) {
        ShellElementsNodes = elementsNodes;
        int test = fsiData->fsiGeneralData.ShellElementsNodes.size();

        std::cout << "numObjects.numFlexNodes" << test << std::endl;
    }
    void SetFsiMesh(std::shared_ptr<chrono::fea::ChMesh> other_fsi_mesh) {
        fsi_mesh = other_fsi_mesh;
        fsiInterface->SetFsiMesh(other_fsi_mesh);
    }

    std::shared_ptr<chrono::fea::ChMesh> GetFsiMesh() { return fsi_mesh; }

    /// Finzalize data by calling FinalizeData function and finalize fluid and bce
    /// and also finalizes the fluid and bce objects if the system have fluid.
    virtual void Finalize();

  private:
    /// Integrate the chrono system based on an explicit Euler scheme.
    int DoStepChronoSystem(Real dT, double mTime);

    ChFsiDataManager* fsiData;                          ///< pointer to data manager which holds all the data
    std::vector<std::shared_ptr<ChBody>> fsiBodeisPtr;  ///< vector of a pointers to fsi bodies. fsi bodies
                                                        /// are those that interact with fluid

    std::vector<std::vector<int>> ShellElementsNodes;  ///< These are the indices of nodes of each Element
    std::vector<std::vector<int>> CableElementsNodes;  ///< These are the indices of nodes of each Element

    std::vector<std::shared_ptr<chrono::fea::ChElementCableANCF>>
        fsiCablesPtr;  ///< vector of ChElementShellANCF pointers
    std::vector<std::shared_ptr<chrono::fea::ChElementShellANCF>>
        fsiShellsPtr;                                                      ///< vector of ChElementShellANCF pointers
    std::vector<std::shared_ptr<chrono::fea::ChNodeFEAxyzD>> fsiNodesPtr;  ///< vector of ChNodeFEAxyzD nodes
    std::shared_ptr<chrono::fea::ChMesh> fsi_mesh;                         ///< ChMesh Pointer

    ChFluidDynamics* fluidDynamics;                                                    ///< pointer to the fluid system
    ChFluidDynamics::Integrator fluidIntegrator = ChFluidDynamics::Integrator::IISPH;  ///< IISPH by default
    ChFsiInterface* fsiInterface;  ///< pointer to the fsi interface system
    ChBce* bceWorker;              ///< pointer to the bce workers

    chrono::ChSystem* mphysicalSystem;  ///< pointer to the multibody system

    SimParams* paramsH;            ///< pointer to the simulation parameters
    NumberOfObjects* numObjectsH;  ///< pointer to the number of objects, fluid, bce, and boundary markers

    double mTime;    ///< current real time of the simulation
    bool haveFluid;  ///< boolean to check if the system have fluid or not
};

/// @} fsi_physics

}  // end namespace fsi
}  // end namespace chrono

#endif
