/*
 * UtilsDeviceOperations.cuh
 *
 *  Created on: Feb 29, 2016
 *      Author: Arman Pazouki
 */

#ifndef UTILSDEVICEOPERATIONS_CUH_
#define UTILSDEVICEOPERATIONS_CUH_


#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "chrono_fsi/custom_cutil_math.h"

/**
 * @brief FillMyThrust, FillMyThrust4
 * @details
 * 			Same explanation as ResizeMyThrust.
 * @param mThrustVec Vector to resize
 * @param v Value to fill thrust vector with.
 */
void FillMyThrust4(thrust::device_vector<Real4>& mThrustVec, Real4 v);

/**
 * @brief ClearMyThrust, ClearMyThrust3, ClearMyThrust4, ClearMyThrustU
 * @details
 * 			Same explanation as ResizeMyThrust.
 * @param mThrustVec Vector to clear
 */
void ClearMyThrustR3(thrust::device_vector<Real3>& mThrustVec);
void ClearMyThrustR4(thrust::device_vector<Real4>& mThrustVec);
void ClearMyThrustU1(thrust::device_vector<uint>& mThrustVec);
void PushBackR3(thrust::device_vector<Real3>& mThrustVec, Real3 a3);
void PushBackR4(thrust::device_vector<Real4>& mThrustVec, Real4 a4);
void ResizeR3(thrust::device_vector<Real3>& mThrustVec, int size);
void ResizeR4(thrust::device_vector<Real4>& mThrustVec, int size);
void ResizeU1(thrust::device_vector<uint>& mThrustVec, int size);

void CopyFluidDataH2D(
		thrust::device_vector<Real3>& posRigid_fsiBodies_D,
		thrust::device_vector<Real4>& velMassRigid_fsiBodies_D,
		thrust::device_vector<Real3>& accRigid_fsiBodies_D,
		thrust::device_vector<Real4>& q_fsiBodies_D,
		thrust::device_vector<Real3>& rigidOmegaLRF_fsiBodies_D,
		thrust::device_vector<Real3>& omegaAccLRF_fsiBodies_D,

		thrust::host_vector<Real3>& posRigid_fsiBodies_H,
		thrust::host_vector<Real4>& velMassRigid_fsiBodies_H,
		thrust::host_vector<Real3>& accRigid_fsiBodies_H,
		thrust::host_vector<Real4>& q_fsiBodies_H,
		thrust::host_vector<Real3>& rigidOmegaLRF_fsiBodies_H,
		thrust::host_vector<Real3>& omegaAccLRF_fsiBodies_H);



#endif /* UTILSDEVICEOPERATIONS_CUH_ */
