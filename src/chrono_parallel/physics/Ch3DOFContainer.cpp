
#include <stdlib.h>
#include <algorithm>
#include <math.h>
#include "chrono_parallel/physics/ChSystemParallel.h"
#include <chrono_parallel/physics/Ch3DOFContainer.h>
#include "chrono_parallel/ChDataManager.h"

namespace chrono {

using namespace collision;
using namespace geometry;

//////////////////////////////////////
//////////////////////////////////////

/// CLASS FOR A 3DOF FLUID NODE

Ch3DOFContainer::Ch3DOFContainer() {
    kernel_radius = .04;
    collision_envelope = 0;
    contact_recovery_speed = 10;
    contact_cohesion = 0;
    contact_compliance = 0;
    contact_mu = 0;
    max_velocity = 20;
    data_manager = 0;

    num_fluid_contacts = 0;
    num_fluid_bodies = 0;
    num_rigid_bodies = 0;
    num_rigid_fluid_contacts = 0;
    num_unilaterals = 0;
    num_bilaterals = 0;
    num_shafts = 0;
    num_fea_tets = 0;
    num_fea_nodes = 0;
    alpha = 0;
}

Ch3DOFContainer::~Ch3DOFContainer() {}

Ch3DOFContainer::Ch3DOFContainer(const Ch3DOFContainer& other) : ChPhysicsItem(other) {
    this->data_manager = other.data_manager;
}

Ch3DOFContainer& Ch3DOFContainer::operator=(const Ch3DOFContainer& other) {
    if (&other == this)
        return *this;

    ChPhysicsItem::operator=(other);
    return *this;
}

real3 Ch3DOFContainer::GetPos(int i) {
    return data_manager->host_data.pos_3dof[i];
}
void Ch3DOFContainer::SetPos(const int& i, const real3& mpos) {
    data_manager->host_data.pos_3dof[i] = mpos;
}

real3 Ch3DOFContainer::GetPos_dt(int i) {
    return data_manager->host_data.vel_3dof[i];
}
void Ch3DOFContainer::SetPos_dt(const int& i, const real3& mposdt) {
    data_manager->host_data.vel_3dof[i] = mposdt;
}

void Ch3DOFContainer::Setup(int start_constraint) {
    start_row = start_constraint;
    if (data_manager) {
        num_fluid_contacts = data_manager->num_fluid_contacts;
        num_fluid_bodies = data_manager->num_fluid_bodies;
        num_rigid_bodies = data_manager->num_rigid_bodies;
        num_rigid_fluid_contacts = data_manager->num_rigid_fluid_contacts;
        num_unilaterals = data_manager->num_unilaterals;
        num_bilaterals = data_manager->num_bilaterals;
        num_shafts = data_manager->num_shafts;
        num_fea_tets = data_manager->num_fea_tets;
        num_fea_nodes = data_manager->num_fea_nodes;
    }
}

void Ch3DOFContainer::SetFamily(short mfamily, short mask_no_collision) {
    family.x = (1 << mfamily);
    family.y &= ~(1 << mask_no_collision);
}
}  // END_OF_NAMESPACE____

/////////////////////
