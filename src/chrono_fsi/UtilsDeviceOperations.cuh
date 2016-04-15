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
void uintCopyH2D(thrust::host_vector<uint>& mHostVec, thrust::device_vector<uint>& mDevVec);
void RealCopyH2D(thrust::host_vector<Real>& mHostVec, thrust::device_vector<Real>& mDevVec);
void Real3CopyH2D(thrust::host_vector<Real3>& mHostVec, thrust::device_vector<Real3>& mDevVec);
void Real4CopyH2D(thrust::host_vector<Real4>& mHostVec, thrust::device_vector<Real4>& mDevVec);
void uintCopyD2D(thrust::device_vector<uint>& mDFrom, thrust::device_vector<uint>& mDTo);
void RealCopyD2D(thrust::device_vector<Real>& mDFrom, thrust::device_vector<Real>& mDTo);
void Real3CopyD2D(thrust::device_vector<Real3>& mDFrom, thrust::device_vector<Real3>& mDTo);
void Real4CopyD2D(thrust::device_vector<Real4>& mDFrom, thrust::device_vector<Real4>& mDTo);
void CopyFluidDataH2D(thrust::device_vector<Real3>& posRigid_fsiBodies_D,
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
