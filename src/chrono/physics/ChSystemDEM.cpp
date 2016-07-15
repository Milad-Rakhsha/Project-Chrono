//
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2010-2012 Alessandro Tasora
// Copyright (c) 2013 Project Chrono
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file at the top level of the distribution
// and at http://projectchrono.org/license-chrono.txt.
//

#include "chrono/physics/ChSystemDEM.h"
#include "chrono/physics/ChContactContainerDEM.h"

#include "chrono/solver/ChSystemDescriptor.h"
#include "chrono/solver/ChSolverDEM.h"

#include "chrono/collision/ChCCollisionSystemBullet.h"

namespace chrono {

// Register into the object factory, to enable run-time
// dynamic creation and persistence
ChClassRegister<ChSystemDEM> a_registration_ChSystemDEM;

ChSystemDEM::ChSystemDEM(bool use_material_properties, unsigned int max_objects, double scene_size)
    : ChSystem(max_objects, scene_size, false),
      m_use_mat_props(use_material_properties),
      m_contact_model(Hertz),
      m_adhesion_model(Constant),
      m_tdispl_model(OneStep),
      m_stiff_contact(false) {
    descriptor = new ChSystemDescriptor;
    descriptor->SetNumThreads(parallel_thread_number);

    solver_type = ChSystem::SOLVER_DEM;

    solver_speed = new ChSolverDEM();
    solver_stab = new ChSolverDEM();

    collision_system = new collision::ChCollisionSystemBullet(max_objects, scene_size);

    // For default DEM there is no need to create contacts 'in advance' 
    // when models are closer than the safety envelope, so set default envelope to 0
    collision::ChCollisionModel::SetDefaultSuggestedEnvelope(0);

    contact_container = std::make_shared<ChContactContainerDEM>();
    contact_container->SetSystem(this);

    m_minSlipVelocity = 1e-4; 
    m_characteristicVelocity = 1; 
}

void ChSystemDEM::SetSolverType(eCh_solverType mval) {

    ChSystem::SetSolverType(mval);

    contact_container = std::make_shared<ChContactContainerDEM>();
    contact_container->SetSystem(this);
}

/*
void ChSystemDEM::ChangeSolverSpeed(ChSolver* newsolver) {
    if (dynamic_cast<ChSolverDEM*>(newsolver))
        ChSystem::ChangeSolverSpeed(newsolver);
}
*/

void ChSystemDEM::ChangeContactContainer(std::shared_ptr<ChContactContainerBase>  newcontainer) {
    if (std::dynamic_pointer_cast<ChContactContainerDEM>(newcontainer))
        ChSystem::ChangeContactContainer(newcontainer);
}


////////
////////  STREAMING - FILE HANDLING
////////

// Trick to avoid putting the following mapper macro inside the class definition in .h file:
// enclose macros in local 'my_enum_mappers', just to avoid avoiding cluttering of the parent class.
class my_enum_mappers : public ChSystemDEM {
public:
    CH_ENUM_MAPPER_BEGIN(ContactForceModel);
      CH_ENUM_VAL(Hooke);
      CH_ENUM_VAL(Hertz);
    CH_ENUM_MAPPER_END(ContactForceModel);

    CH_ENUM_MAPPER_BEGIN(AdhesionForceModel);
      CH_ENUM_VAL(Constant);
      CH_ENUM_VAL(DMT);
    CH_ENUM_MAPPER_END(AdhesionForceModel);

    CH_ENUM_MAPPER_BEGIN(TangentialDisplacementModel);
      CH_ENUM_VAL(None);
      CH_ENUM_VAL(OneStep);
      CH_ENUM_VAL(MultiStep);
    CH_ENUM_MAPPER_END(TangentialDisplacementModel);
};


void ChSystemDEM::ArchiveOUT(ChArchiveOut& marchive)
{
    // version number
    marchive.VersionWrite(1);

    // serialize parent class
    ChSystem::ArchiveOUT(marchive);

    // serialize all member data:
    marchive << CHNVP(m_use_mat_props); 
    marchive << CHNVP(m_minSlipVelocity);
    marchive << CHNVP(m_characteristicVelocity); 
    my_enum_mappers::ContactForceModel_mapper mmodel_mapper;
    marchive << CHNVP(mmodel_mapper(m_contact_model),"contact_model");
    my_enum_mappers::AdhesionForceModel_mapper madhesion_mapper;
    marchive << CHNVP(madhesion_mapper(m_adhesion_model),"adhesion_model");
    my_enum_mappers::TangentialDisplacementModel_mapper mtangential_mapper;
    marchive << CHNVP(mtangential_mapper(m_tdispl_model), "tangential_model");
    //***TODO*** complete...
}

/// Method to allow de serialization of transient data from archives.
void ChSystemDEM::ArchiveIN(ChArchiveIn& marchive) 
{
    // version number
    int version = marchive.VersionRead();

    // deserialize parent class
    ChSystem::ArchiveIN(marchive);

    // stream in all member data:
    marchive >> CHNVP(m_use_mat_props); 
    marchive >> CHNVP(m_minSlipVelocity);
    marchive >> CHNVP(m_characteristicVelocity); 
    my_enum_mappers::ContactForceModel_mapper mmodel_mapper;
    marchive >> CHNVP(mmodel_mapper(m_contact_model),"contact_model");
    my_enum_mappers::AdhesionForceModel_mapper madhesion_mapper;
    marchive >> CHNVP(madhesion_mapper(m_adhesion_model),"adhesion_model");
    my_enum_mappers::TangentialDisplacementModel_mapper mtangential_mapper;
    marchive >> CHNVP(mtangential_mapper(m_tdispl_model), "tangential_model");
    //***TODO*** complete...

    // Recompute statistics, offsets, etc.
    this->Setup();
}

}  // end namespace chrono
