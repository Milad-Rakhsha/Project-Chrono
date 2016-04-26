//
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2013 Project Chrono
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file at the top level of the distribution
// and at http://projectchrono.org/license-chrono.txt.
//
// File authors: Andrea Favali, Alessandro Tasora

#ifndef CHMESH_H
#define CHMESH_H

#include <stdlib.h>
#include <math.h>

#include "chrono/core/ChTimer.h"
#include "chrono/physics/ChContinuumMaterial.h"
#include "chrono/physics/ChIndexedNodes.h"
#include "chrono/physics/ChMaterialSurface.h"
#include "chrono_fea/ChContactSurface.h"
#include "chrono_fea/ChElementBase.h"
#include "chrono_fea/ChMeshSurface.h"
#include "chrono_fea/ChNodeFEAbase.h"


namespace chrono {

namespace fea {

/// @addtogroup fea_module
/// @{

/// Class which defines a mesh of finite elements of class ChFelem,
/// between nodes of class  ChFnode.
class ChApiFea ChMesh : public ChIndexedNodes {
    // Chrono simulation of RTTI, needed for serialization
    CH_RTTI(ChMesh, ChIndexedNodes);

  private:
    std::vector<std::shared_ptr<ChNodeFEAbase> > vnodes;     ///<  nodes
    std::vector<std::shared_ptr<ChElementBase> > velements;  ///<  elements

    unsigned int n_dofs;    ///< total degrees of freedom
    unsigned int n_dofs_w;  ///< total degrees of freedom, derivative (Lie algebra)

    std::vector<std::shared_ptr<ChContactSurface> > vcontactsurfaces;  ///<  contact surfaces
    std::vector<std::shared_ptr<ChMeshSurface> >    vmeshsurfaces;     ///<  mesh surfaces, ex.for loads
    
    bool automatic_gravity_load;
	int num_points_gravity;

    ChTimer<> timer_internal_forces;
    ChTimer<> timer_KRMload;
    int ncalls_internal_forces;
    int ncalls_KRMload;

  public:
    ChMesh()
        : n_dofs(0),
          n_dofs_w(0),
          automatic_gravity_load(true),
          num_points_gravity(1),
          ncalls_internal_forces(0),
          ncalls_KRMload(0) {}

    ~ChMesh() {}

    void AddNode(std::shared_ptr<ChNodeFEAbase> m_node);
    void AddElement(std::shared_ptr<ChElementBase> m_elem);
    void ClearNodes();
    void ClearElements();

    /// Access the N-th node
    virtual std::shared_ptr<ChNodeBase> GetNode(unsigned int n) { return vnodes[n]; };
    /// Access the N-th element
    virtual std::shared_ptr<ChElementBase> GetElement(unsigned int n) { return velements[n]; };

    unsigned int GetNnodes() { return (unsigned int)vnodes.size(); }
    unsigned int GetNelements() { return (unsigned int)velements.size(); }
    virtual int GetDOF() { return n_dofs; }
    virtual int GetDOF_w() { return n_dofs_w; }

    /// Override default in ChPhysicsItem
    virtual bool GetCollide() { return true; }

    /// Get number of calls to internal forces evaluation.
    int GetNumCallsInternalForces() { return ncalls_internal_forces; }
    /// Get number of calls to load Jacobian information.
    int GetNumCallsJacobianLoad() { return ncalls_KRMload; }

    /// Get cummulative timing for internal force evaluation.
    double GetTimingInternalForces() { return timer_internal_forces(); }
    /// Get cummulative timing for Jacobian load calls.
    double GetTimingJacobianLoad() { return timer_KRMload(); }

    /// Add a contact surface
    void AddContactSurface(std::shared_ptr<ChContactSurface> m_surf);

    /// Access the N-th contact surface
    virtual std::shared_ptr<ChContactSurface> GetContactSurface(unsigned int n) { return vcontactsurfaces[n]; };

    /// Get number of added contact surfaces
    unsigned int GetNcontactSurfaces() { return (unsigned int)vcontactsurfaces.size(); }

    /// Remove all contact surfaces
    void ClearContactSurfaces();


    /// Add a mesh surface (set of ChLoadableUV items that support area loads such as pressure, etc.)
    void AddMeshSurface(std::shared_ptr<ChMeshSurface> m_surf);

    /// Access the N-th mesh surface
    virtual std::shared_ptr<ChMeshSurface> GetMeshSurface(unsigned int n) { return vmeshsurfaces[n]; };

    /// Get number of added mesh surfaces
    unsigned int GetNmeshSurfaces() { return (unsigned int)vmeshsurfaces.size(); }

    /// Remove all mesh surfaces
    void ClearMeshSurfaces() { vmeshsurfaces.clear(); }


    /// Set reference position of nodes as current position, for all nodes.
    void Relax();

    /// Set no speed and no accelerations in nodes (but does not change reference positions)
    void SetNoSpeedNoAcceleration();

    /// This recomputes the number of DOFs, constraints,
    /// as well as state offsets of contained items
    virtual void Setup();

    /// Update time dependent data, for all elements.
    /// Updates all [A] coord.systems for all (corotational) elements.
    virtual void Update(double m_time, bool update_assets = true);

    	// Functions to interface this with ChPhysicsItem container 
	virtual void SyncCollisionModels();
	virtual void AddCollisionModelsToSystem();
	virtual void RemoveCollisionModelsFromSystem();

    /// If true, as by default, this mesh will add automatically a gravity load 
    /// to all contained elements (that support gravity) using the G value from the ChSystem.
    /// So this saves you from adding many ChLoad<ChLoaderGravity> to all elements.
	void SetAutomaticGravity(bool mg, int num_points = 1) { automatic_gravity_load = mg; num_points_gravity = num_points; }
    /// Tell if this mesh will add automatically a gravity load to all contained elements
    bool GetAutomaticGravity() { return automatic_gravity_load; }

    /// Get ChMesh mass properties
    void ComputeMassProperties(double& mass,          ///< ChMesh object mass
                               ChVector<>& com,       ///< ChMesh center of gravity
                               ChMatrix33<>& inertia  ///< ChMesh inertia tensor
                               );

    //
    // STATE FUNCTIONS
    //

        // (override/implement interfaces for global state vectors, see ChPhysicsItem for comments.)
        virtual void IntStateGather(const unsigned int off_x,
                                    ChState& x,
                                    const unsigned int off_v,
                                    ChStateDelta& v,
                                    double& T);
    virtual void IntStateScatter(const unsigned int off_x,
                                 const ChState& x,
                                 const unsigned int off_v,
                                 const ChStateDelta& v,
                                 const double T);
    virtual void IntStateGatherAcceleration(const unsigned int off_a, ChStateDelta& a);
    virtual void IntStateScatterAcceleration(const unsigned int off_a, const ChStateDelta& a);
    virtual void IntStateIncrement(const unsigned int off_x,
                                   ChState& x_new,
                                   const ChState& x,
                                   const unsigned int off_v,
                                   const ChStateDelta& Dv);
    virtual void IntLoadResidual_F(const unsigned int off, ChVectorDynamic<>& R, const double c);
    virtual void IntLoadResidual_Mv(const unsigned int off,
                                    ChVectorDynamic<>& R,
                                    const ChVectorDynamic<>& w,
                                    const double c);
    virtual void IntToLCP(const unsigned int off_v,
                          const ChStateDelta& v,
                          const ChVectorDynamic<>& R,
                          const unsigned int off_L,
                          const ChVectorDynamic<>& L,
                          const ChVectorDynamic<>& Qc);
    virtual void IntFromLCP(const unsigned int off_v, ChStateDelta& v, const unsigned int off_L, ChVectorDynamic<>& L);

    //
    // LCP SYSTEM FUNCTIONS        for interfacing all elements with LCP solver
    //

    /// Tell to a system descriptor that there are items of type
    /// ChLcpKblock in this object (for further passing it to a LCP solver)
    /// Basically does nothing, but maybe that inherited classes may specialize this.
    virtual void InjectKRMmatrices(ChLcpSystemDescriptor& mdescriptor);

    /// Adds the current stiffness K and damping R and mass M matrices in encapsulated
    /// ChLcpKblock item(s), if any. The K, R, M matrices are added with scaling
    /// values Kfactor, Rfactor, Mfactor.
    virtual void KRMmatricesLoad(double Kfactor, double Rfactor, double Mfactor);

    /// Sets the 'fb' part (the known term) of the encapsulated ChLcpVariables to zero.
    virtual void VariablesFbReset();

    /// Adds the current forces (applied to item) into the
    /// encapsulated ChLcpVariables, in the 'fb' part: qf+=forces*factor
    virtual void VariablesFbLoadForces(double factor = 1.);

    /// Initialize the 'qb' part of the ChLcpVariables with the
    /// current value of speeds. Note: since 'qb' is the unknown of the LCP, this
    /// function seems unuseful, unless used before VariablesFbIncrementMq()
    virtual void VariablesQbLoadSpeed();

    /// Adds M*q (masses multiplied current 'qb') to Fb, ex. if qb is initialized
    /// with v_old using VariablesQbLoadSpeed, this method can be used in
    /// timestepping schemes that do: M*v_new = M*v_old + forces*dt
    virtual void VariablesFbIncrementMq();

    /// Fetches the item speed (ex. linear and angular vel.in rigid bodies) from the
    /// 'qb' part of the ChLcpVariables and sets it as the current item speed.
    /// If 'step' is not 0, also should compute the approximate acceleration of
    /// the item using backward differences, that is  accel=(new_speed-old_speed)/step.
    /// Mostly used after the LCP provided the solution in ChLcpVariables.
    virtual void VariablesQbSetSpeed(double step = 0.);

    /// Increment item positions by the 'qb' part of the ChLcpVariables,
    /// multiplied by a 'step' factor.
    ///     pos+=qb*step
    /// If qb is a speed, this behaves like a single step of 1-st order
    /// numerical integration (Eulero integration).
    virtual void VariablesQbIncrementPosition(double step);

    /// Tell to a system descriptor that there are variables of type
    /// ChLcpVariables in this object (for further passing it to a LCP solver)
    /// Basically does nothing, but maybe that inherited classes may specialize this.
    virtual void InjectVariables(ChLcpSystemDescriptor& mdescriptor);

  private:
    /// Initial setup (before analysis).
    /// This function is called from ChSystem::SetupInitial, marking a point where system
    /// construction is completed.
    /// - Computes the total number of degrees of freedom
    /// - Precompute auxiliary data, such as (local) stiffness matrices Kl, if any, for each element.
    virtual void SetupInitial() override;
};

/// @} fea_module

}  // END_OF_NAMESPACE____
}  // END_OF_NAMESPACE____

#endif
