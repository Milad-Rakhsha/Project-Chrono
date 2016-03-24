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
// File authors: Alessandro Tasora

#ifndef CHCONTACTSURFACENODECLOUD_H
#define CHCONTACTSURFACENODECLOUD_H

#include "chrono/collision/ChCCollisionModel.h"
#include "chrono_fea/ChContactSurface.h"
#include "chrono_fea/ChNodeFEAxyz.h"

namespace chrono {

namespace fea {

/// Proxy to FEA nodes, to grant them the features
/// needed for collision detection.
class ChApiFea ChContactNodeXYZ : public ChContactable_1vars<3> {
    // Chrono simulation of RTTI, needed for serialization
    CH_RTTI_ROOT(ChContactNodeXYZ);

  public:
    ChContactNodeXYZ(ChNodeFEAxyz* anode = 0, ChContactSurface* acontainer = 0) {
        mnode = anode;
        container = acontainer;
    }

    //
    // FUNCTIONS
    //

    /// Access the FEA node to whom this is is a proxy
    ChNodeFEAxyz* GetNode() { return mnode; }
    /// Set the FEA node to whom this is a proxy
    void SetNode(ChNodeFEAxyz* mn) { mnode = mn; }

    /// Get the contact surface container
    ChContactSurface* GetContactSurface() const { return container; }
    /// Set the contact surface container
    void GetContactSurface(ChContactSurface* mc) { container = mc; }

    //
    // INTERFACE TO ChContactable
    //

    /// Access variables
    virtual ChLcpVariables* GetVariables1() override { return &mnode->Variables(); }

    /// Tell if the object must be considered in collision detection
    virtual bool IsContactActive() override { return true; }

    /// Get the number of DOFs affected by this object (position part)
    virtual int ContactableGet_ndof_x() override { return 3; }

    /// Get the number of DOFs affected by this object (speed part)
    virtual int ContactableGet_ndof_w() override { return 3; }

    /// Get all the DOFs packed in a single vector (position part)
    virtual void ContactableGetStateBlock_x(ChState& x) override { x.PasteVector(this->mnode->pos, 0, 0); }

    /// Get all the DOFs packed in a single vector (speed part)
    virtual void ContactableGetStateBlock_w(ChStateDelta& w) override { w.PasteVector(this->mnode->pos_dt, 0, 0); }

    /// Increment the provided state of this object by the given state-delta increment.
    /// Compute: x_new = x + dw.
    virtual void ContactableIncrementState(const ChState& x, const ChStateDelta& dw, ChState& x_new) override {
        this->mnode->NodeIntStateIncrement(0, x_new, x, 0, dw);
    }

    /// Express the local point in absolute frame, for the given state position.
    virtual ChVector<> GetContactPoint(const ChVector<>& loc_point, const ChState& state_x) override {
        return state_x.ClipVector(0, 0);
    }

    /// Get the absolute speed of a local point attached to the contactable.
    /// The given point is assumed to be expressed in the local frame of this object.
    /// This function must use the provided states.
    virtual ChVector<> GetContactPointSpeed(const ChVector<>& loc_point,
                                            const ChState& state_x,
                                            const ChStateDelta& state_w) override {
        return state_w.ClipVector(0, 0);
    }

    /// Get the absolute speed of point abs_point if attached to the
    /// surface. Easy in this case because there are no roations..
    virtual ChVector<> GetContactPointSpeed(const ChVector<>& abs_point) override { return this->mnode->pos_dt; }

    /// Return the coordinate system for the associated collision model.
    /// ChCollisionModel might call this to get the position of the
    /// contact model (when rigid) and sync it.
    virtual ChCoordsys<> GetCsysForCollisionModel() override { return ChCoordsys<>(this->mnode->pos, QNULL); }

    /// Apply the force, expressed in absolute reference, applied in pos, to the
    /// coordinates of the variables. Force for example could come from a penalty model.
    virtual void ContactForceLoadResidual_F(const ChVector<>& F,
                                            const ChVector<>& abs_point,
                                            ChVectorDynamic<>& R) override;

    /// Apply the given force at the given point and load the generalized force array.
    /// The force and its application point are specified in the gloabl frame.
    /// Each object must set the entries in Q corresponding to its variables, starting at the specified offset.
    /// If needed, the object states must be extracted from the provided state position.
    virtual void ContactForceLoadQ(const ChVector<>& F,
                                   const ChVector<>& point,
                                   const ChState& state_x,
                                   ChVectorDynamic<>& Q,
                                   int offset) override {
        Q.PasteVector(F, offset, 0);
    }

    /// Compute the jacobian(s) part(s) for this contactable item. For example,
    /// if the contactable is a ChBody, this should update the corresponding 1x6 jacobian.
    virtual void ComputeJacobianForContactPart(const ChVector<>& abs_point,
                                               ChMatrix33<>& contact_plane,
                                               type_constraint_tuple& jacobian_tuple_N,
                                               type_constraint_tuple& jacobian_tuple_U,
                                               type_constraint_tuple& jacobian_tuple_V,
                                               bool second) override;

    virtual double GetContactableMass() override {
        //***TODO***!!!!!!!!!!!!!!!!!!!!
        return 1;
        // return this->mnode->GetMass(); // no!! could be zero in nodes of non-lumped-masses meshes!
    }

    /// Return the pointer to the surface material.
    virtual std::shared_ptr<ChMaterialSurfaceBase>& GetMaterialSurfaceBase() override;

    /// This is only for backward compatibility
    virtual ChPhysicsItem* GetPhysicsItem() override;

  private:
    ChNodeFEAxyz* mnode;

    ChContactSurface* container;
};

/// Proxy to FEA nodes for collisions, with spheres associated to nodes, for point-cloud 
/// type of collisions.
class ChApiFea ChContactNodeXYZsphere : public ChContactNodeXYZ {

    // Chrono simulation of RTTI, needed for serialization
    CH_RTTI(ChContactNodeXYZsphere, ChContactNodeXYZ);

public:
    ChContactNodeXYZsphere(ChNodeFEAxyz* anode = 0, ChContactSurface* acontainer = 0);

    virtual ~ChContactNodeXYZsphere(){ delete collision_model;}

    collision::ChCollisionModel* GetCollisionModel() {return  collision_model;}

private:
    collision::ChCollisionModel* collision_model;
};


/// Class which defines a contact surface for FEA elements, where only xyz nodes
/// in the FEA model are used as contact items for the collision detection.
/// Might be an efficient option in case of dense tesselations (but misses the FEAnodes-vs-FEAfaces
/// cases, and misses FEAedge-vs-edges)
class ChApiFea ChContactSurfaceNodeCloud : public ChContactSurface {

    // Chrono simulation of RTTI, needed for serialization
    CH_RTTI(ChContactSurfaceNodeCloud, ChContactSurface);


  public:
    ChContactSurfaceNodeCloud(ChMesh* parentmesh = 0) : 
        ChContactSurface(parentmesh) {};

    virtual ~ChContactSurfaceNodeCloud(){};

    //
    // FUNCTIONS
    //

    /// Add a specific node to this collision cloud
    void AddNode(std::shared_ptr<ChNodeFEAxyz> mnode, const double point_radius = 0.001);

    /// Add all nodes of the mesh to this collision cloud
    void AddAllNodes(const double point_radius = 0.001);

    /// Add nodes of the mesh, belonging to the node_set, to this collision cloud
    void AddFacesFromNodeSet(std::vector<std::shared_ptr<ChNodeFEAbase> >& node_set, const double point_radius = 0.001);

    /// Get the number of nodes.
    unsigned int GetNnodes() const { return (unsigned int)vnodes.size(); }

    /// Access the N-th node
    std::shared_ptr<ChContactNodeXYZsphere> GetNode(unsigned int n) { return vnodes[n]; };

    // Functions to interface this with ChPhysicsItem container
    virtual void SurfaceSyncCollisionModels();
    virtual void SurfaceAddCollisionModelsToSystem(ChSystem* msys);
    virtual void SurfaceRemoveCollisionModelsFromSystem(ChSystem* msys);

  private:
    std::vector<std::shared_ptr<ChContactNodeXYZsphere> > vnodes;  //  nodes
};

}  // END_OF_NAMESPACE____
}  // END_OF_NAMESPACE____

#endif
