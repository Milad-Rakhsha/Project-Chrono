/*
 * incompressible_integrate.h
 *
 *  Created on: Feb 29, 2016
 *      Author: Arman Pazouki, Milad Rakhsha
 */

#ifndef INCOMPRESSIBLE_INTEGRATE_H_
#define INCOMPRESSIBLE_INTEGRATE_H_

void DoStepFluid_implicit(
		thrust::device_vector<Real3>& posRadD,
		thrust::device_vector<Real3>& velMasD,
		thrust::device_vector<Real4>& rhoPresMuD,

		thrust::device_vector<Real3>& posRadD2,
		thrust::device_vector<Real3>& velMasD2,
		thrust::device_vector<Real4>& rhoPresMuD2,

		thrust::device_vector<Real4>& derivVelRhoD,
		thrust::device_vector<uint>& rigidIdentifierD,
		const thrust::device_vector<Real3>& rigidSPH_MeshPos_LRF_D,

		thrust::device_vector<Real3>& posRigid_fsiBodies_D,
		thrust::device_vector<Real4>& velMassRigid_fsiBodies_D,
		thrust::device_vector<Real3>& accRigid_fsiBodies_D,
		thrust::device_vector<Real4>& q_fsiBodies_D,
		thrust::device_vector<Real3>& omegaVelLRF_fsiBodies_D,
		thrust::device_vector<Real3>& omegaAccLRF_fsiBodies_D,


		thrust::device_vector<Real3>& posRigid_fsiBodies_D2,
		thrust::device_vector<Real4>& velMassRigid_fsiBodies_D2,
		thrust::device_vector<Real3>& accRigid_fsiBodies_D2,
		thrust::device_vector<Real4>& q_fsiBodies_D2,
		thrust::device_vector<Real3>& omegaVelLRF_fsiBodies_D2,
		thrust::device_vector<Real3>& omegaAccLRF_fsiBodies_D2,

		thrust::host_vector<Real3>& pos_ChSystemBackupH,
		thrust::host_vector<Real3>& vel_ChSystemBackupH,
		thrust::host_vector<Real3>& acc_ChSystemBackupH,
		thrust::host_vector<Real4>& quat_ChSystemBackupH,
		thrust::host_vector<Real3>& omegaVelGRF_ChSystemBackupH,
		thrust::host_vector<Real3>& omegaAccGRF_ChSystemBackupH,

		thrust::host_vector<Real3>& posRigid_fsiBodies_dummyH,
		thrust::host_vector<Real4>& velMassRigid_fsiBodies_dummyH,
		thrust::host_vector<Real3>& accRigid_fsiBodies_dummyH,
		thrust::host_vector<Real4>& q_fsiBodies_dummyH,
		thrust::host_vector<Real3>& omegaVelLRF_fsiBodies_dummyH,
		thrust::host_vector<Real3>& omegaAccLRF_fsiBodies_dummyH,

		thrust::device_vector<Real3>& rigid_FSI_ForcesD,
		thrust::device_vector<Real3>& rigid_FSI_TorquesD,

		thrust::device_vector<uint>& bodyIndexD,
		const thrust::host_vector<int4>& referenceArray,
		const NumberOfObjects& numObjects, const SimParams& paramsH);



#endif /* INCOMPRESSIBLE_INTEGRATE_H_ */
