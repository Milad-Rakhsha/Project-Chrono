///////////////////////////////////////////////////////////////////////////////
//	collideSphereSphere.cuh
//	header file implements kernels and functions for fluid force calculation and update, rigids, and bce
//
//	Created by Arman Pazouki, Milad Rakhsha
#ifndef INCOMPRESSIBLE_COLLIDESPHERESPHERE_CUH
#define INCOMPRESSIBLE_COLLIDESPHERESPHERE_CUH

#include <thrust/device_vector.h>

#include "chrono_fsi/MyStructs.cuh"  //just for SimParams
/**
 * @brief InitSystem
 * @details
 * 			Initializes paramsD and numObjectsD in collideSphereSphere.cu and SDKCollisionSystem.cu
 * 			These two are exactly the same struct as paramsH and numObjects but they are stored in
 * 			the device memory.
 *
 * @param paramsH Parameters that will be used in the system
 * @param numObjects [description]
 */
void InitSystem(SimParams paramsH, NumberOfObjects numObjects);


void MakeRigidIdentifier(thrust::device_vector<uint>& rigidIdentifierD,
		int numRigidBodies, int startRigidMarkers,
		const thrust::host_vector<int4>& referenceArray);

void Populate_RigidSPH_MeshPos_LRF(
		thrust::device_vector<uint>& rigidIdentifierD,
		thrust::device_vector<Real3>& rigidSPH_MeshPos_LRF_D,
		const thrust::device_vector<Real3>& posRadD,
		const thrust::device_vector<Real3>& posRigidD,
		const thrust::device_vector<Real4>& qD,
		const thrust::host_vector<int4>& referenceArray,
		const NumberOfObjects& numObjects);

void Rigid_Forces_Torques(thrust::device_vector<Real3>& rigid_FSI_ForcesD,
		thrust::device_vector<Real3>& rigid_FSI_TorquesD,

		const thrust::device_vector<Real3>& posRadD,
		const thrust::device_vector<Real3>& posRigidD,

		const thrust::device_vector<Real4>& derivVelRhoD,
		const thrust::device_vector<uint>& rigidIdentifierD,

		const NumberOfObjects& numObjects);

void UpdateRigidMarkersPositionVelocity(thrust::device_vector<Real3>& posRadD,
		thrust::device_vector<Real3>& velMasD,
		const thrust::device_vector<Real3>& rigidSPH_MeshPos_LRF_D,
		const thrust::device_vector<uint>& rigidIdentifierD,
		const thrust::device_vector<Real3>& posRigidD,
		const thrust::device_vector<Real4>& qD,
		const thrust::device_vector<Real4>& velMassRigidD,
		const thrust::device_vector<Real3>& omegaLRF_D,
		NumberOfObjects numObjects);

void UpdateRigidMarkersPositionVelocity(thrust::device_vector<Real3>& posRadD,
		thrust::device_vector<Real3>& velMasD,
		const thrust::device_vector<Real3>& rigidSPH_MeshPos_LRF_D,
		const thrust::device_vector<uint>& rigidIdentifierD,
		const thrust::device_vector<Real3>& posRigidD,
		const thrust::device_vector<Real4>& qD,
		const thrust::device_vector<Real4>& velMassRigidD,
		const thrust::device_vector<Real3>& omegaLRF_D,
		NumberOfObjects numObjects,
		SimParams paramsH);

/**
 * @brief Calculates the force on each particles
 * @details
 * 			Algorithm:
 *          1. Build neighbor list of each particle. These are the particles that are within the
 *          interaction radius (HSML).
 *          2. Calculate interaction force between:
 *          	- fluid-fluid
 *          	- fluid-solid
 *          	- solid-fluid
 *          3. Calculates forces from other SPH or solid particles as well as boundaries.
 *
 * @param &posRadD
 * @param &velMasD
 * @param &vel_XSPH_D
 * @param &rhoPresMuD
 * @param &bodyIndexD
 * @param &derivVelRhoD
 * @param &referenceArray
 * @param &numObjects
 * @param paramsH These are the simulation parameters which were set in the initialization part.
 * @param dT Time step
 */
void ForceSPH_implicit(thrust::device_vector<Real3>& posRadD,
		thrust::device_vector<Real3>& velMasD,
		thrust::device_vector<Real4>& rhoPresMuD,
		thrust::device_vector<uint>& bodyIndexD,
		thrust::device_vector<Real4>& derivVelRhoD,
		const thrust::host_vector<int4>& referenceArray,

		const thrust::device_vector<Real4>& q_fsiBodies_D,
		const thrust::device_vector<Real3>& accRigid_fsiBodies_D,
		const thrust::device_vector<Real3>& omegaVelLRF_fsiBodies_D,
		const thrust::device_vector<Real3>& omegaAccLRF_fsiBodies_D,
		const thrust::device_vector<Real3>& rigidSPH_MeshPos_LRF_D,
		const thrust::device_vector<uint>& rigidIdentifierD,

		const NumberOfObjects& numObjects, SimParams paramsH,
		Real dT);

void IntegrateSPH_implicit(thrust::device_vector<Real4>& derivVelRhoD,
		thrust::device_vector<Real3>& posRadD2,
		thrust::device_vector<Real3>& velMasD2,
		thrust::device_vector<Real4>& rhoPresMuD2,

		thrust::device_vector<Real3>& posRadD,
		thrust::device_vector<Real3>& velMasD,
		thrust::device_vector<Real4>& rhoPresMuD,

		thrust::device_vector<uint>& bodyIndexD,
		const thrust::host_vector<int4>& referenceArray,

		const thrust::device_vector<Real4>& q_fsiBodies_D,
		const thrust::device_vector<Real3>& accRigid_fsiBodies_D,
		const thrust::device_vector<Real3>& omegaVelLRF_fsiBodies_D,
		const thrust::device_vector<Real3>& omegaAccLRF_fsiBodies_D,
		const thrust::device_vector<Real3>& rigidSPH_MeshPos_LRF_D,
		const thrust::device_vector<uint>& rigidIdentifierD,

		const NumberOfObjects& numObjects, SimParams currentParamsH, Real dT);
#endif
