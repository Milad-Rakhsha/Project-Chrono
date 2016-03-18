/*
 * incompressible_integrate.cpp
 *
 *  Created on: Feb 29, 2016
 *      Author: Arman Pazouki, Milad Rakhsha
 */

#include "chrono_fsi/incompressible_integrate.h"
#include "chrono_fsi/incompressible_collideSphereSphere.cuh"
#include "chrono_fsi/SPHCudaUtils.h"
#include "chrono_fsi/UtilsDeviceOperations.cuh"

//--------------------------------------------------------------------------------------------------------------------------------

void DoStepFluid_implicit(thrust::device_vector<Real3>& posRadD,
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
                          const NumberOfObjects& numObjects,
                          const SimParams& paramsH) {
  //**********************************
  //----------------------------
  //--------- start fluid ------
  //----------------------------
  // ** initialize host mid step data
  thrust::copy(posRadD.begin(), posRadD.end(), posRadD2.begin());
  thrust::copy(velMasD.begin(), velMasD.end(), velMasD2.begin());
  thrust::copy(rhoPresMuD.begin(), rhoPresMuD.end(), rhoPresMuD2.begin());

  FillMyThrust4(derivVelRhoD, mR4(0));

  //**********************************
  // ******************
  // ******************
  // ******************
  // ******************
  // ****************** RK2: 1/2

  //  IntegrateSPH_implicit(derivVelRhoD, posRadD2, velMasD2, rhoPresMuD2, posRadD, velMasD, rhoPresMuD, bodyIndexD,
  //                        referenceArray, q_fsiBodies_D, accRigid_fsiBodies_D, omegaVelLRF_fsiBodies_D,
  //                        omegaAccLRF_fsiBodies_D, rigidSPH_MeshPos_LRF_D, rigidIdentifierD, numObjects, paramsH,
  //                        0.5 * paramsH.dT);

  IntegrateIISPH(derivVelRhoD, posRadD2, velMasD2, rhoPresMuD2, posRadD, velMasD, rhoPresMuD, bodyIndexD,
                 referenceArray, q_fsiBodies_D, accRigid_fsiBodies_D, omegaVelLRF_fsiBodies_D, omegaAccLRF_fsiBodies_D,
                 rigidSPH_MeshPos_LRF_D, rigidIdentifierD, numObjects, paramsH, 0.5 * paramsH.dT);

  //	Rigid_Forces_Torques(rigid_FSI_ForcesD, rigid_FSI_TorquesD, posRadD,
  //			posRigid_fsiBodies_D, derivVelRhoD, rigidIdentifierD, numObjects);
  //	// TODO
  //	// integrate rigid bodies
  //	UpdateRigidMarkersPositionVelocity(posRadD2, velMasD2, rigidSPH_MeshPos_LRF_D,
  //			rigidIdentifierD, posRigid_fsiBodies_D2, q_fsiBodies_D2,
  //			velMassRigid_fsiBodies_D2, omegaVelLRF_fsiBodies_D2, numObjects);
  // ******************
  // ******************
  // ******************
  // ******************
  // ****************** RK2: 2/2
  //  FillMyThrust4(derivVelRhoD, mR4(0));

  // //assumes ...D2 is a copy of ...D
  //  IntegrateSPH_implicit(derivVelRhoD, posRadD, velMasD, rhoPresMuD, posRadD2, velMasD2, rhoPresMuD2, bodyIndexD,
  //                        referenceArray, q_fsiBodies_D2, accRigid_fsiBodies_D2, omegaVelLRF_fsiBodies_D2,
  //                        omegaAccLRF_fsiBodies_D2, rigidSPH_MeshPos_LRF_D, rigidIdentifierD, numObjects, paramsH,
  //                        paramsH.dT);

  //	Rigid_Forces_Torques(rigid_FSI_ForcesD, rigid_FSI_TorquesD, posRadD2,
  //			posRigid_fsiBodies_D2, derivVelRhoD, rigidIdentifierD, numObjects);
  //	// TODO
  //	// integrate rigid bodies
  //	UpdateRigidMarkersPositionVelocity(posRadD, velMasD, rigidSPH_MeshPos_LRF_D,
  //			rigidIdentifierD, posRigid_fsiBodies_D, q_fsiBodies_D,
  //			velMassRigid_fsiBodies_D, omegaVelLRF_fsiBodies_D, numObjects);
  // ****************** End RK2
}
