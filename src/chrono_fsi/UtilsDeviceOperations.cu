/*
 * UtilsDeviceOperations.cu
 *
 *  Created on: Feb 29, 2016
 *      Author: Arman Pazouki
 */

#include "chrono_fsi/UtilsDeviceOperations.cuh"
#include "chrono_fsi/SPHCudaUtils.h"

/**
 * @brief See collideSphereSphere.cuh for documentation.
 */
void FillMyThrust4(thrust::device_vector<Real4>& mThrustVec, Real4 v) {
  thrust::fill(mThrustVec.begin(), mThrustVec.end(), v);
}

/**
 * @brief See collideSphereSphere.cuh for documentation.
 */
void ClearMyThrustR3(thrust::device_vector<Real3>& mThrustVec) {
  mThrustVec.clear();
}
void ClearMyThrustR4(thrust::device_vector<Real4>& mThrustVec) {
  mThrustVec.clear();
}
void ClearMyThrustU1(thrust::device_vector<uint>& mThrustVec) {
  mThrustVec.clear();
}
void PushBackR3(thrust::device_vector<Real3>& mThrustVec, Real3 a3) {
  mThrustVec.push_back(a3);
}
void PushBackR4(thrust::device_vector<Real4>& mThrustVec, Real4 a4) {
  mThrustVec.push_back(a4);
}
void ResizeR3(thrust::device_vector<Real3>& mThrustVec, int size) {
  mThrustVec.resize(size);
}
void ResizeR4(thrust::device_vector<Real4>& mThrustVec, int size) {
  mThrustVec.resize(size);
}
void ResizeU1(thrust::device_vector<uint>& mThrustVec, int size) {
  mThrustVec.resize(size);
}
void uintCopyH2D(thrust::host_vector<uint>& mHostVec, thrust::device_vector<uint>& mDevVec) {
  thrust::copy(mHostVec.begin(), mHostVec.end(), mDevVec.begin());
}
void RealCopyH2D(thrust::host_vector<Real>& mHostVec, thrust::device_vector<Real>& mDevVec) {
  thrust::copy(mHostVec.begin(), mHostVec.end(), mDevVec.begin());
}
void Real3CopyH2D(thrust::host_vector<Real3>& mHostVec, thrust::device_vector<Real3>& mDevVec) {
  thrust::copy(mHostVec.begin(), mHostVec.end(), mDevVec.begin());
}
void Real4CopyH2D(thrust::host_vector<Real4>& mHostVec, thrust::device_vector<Real4>& mDevVec) {
  thrust::copy(mHostVec.begin(), mHostVec.end(), mDevVec.begin());
}
void uintCopyD2D(thrust::device_vector<uint>& mDFrom, thrust::device_vector<uint>& mDTo) {
  thrust::copy(mDFrom.begin(), mDFrom.end(), mDTo.begin());
}
void RealCopyD2D(thrust::device_vector<Real>& mDFrom, thrust::device_vector<Real>& mDTo) {
  thrust::copy(mDFrom.begin(), mDFrom.end(), mDTo.begin());
}
void Real3CopyD2D(thrust::device_vector<Real3>& mDFrom, thrust::device_vector<Real3>& mDTo) {
  thrust::copy(mDFrom.begin(), mDFrom.end(), mDTo.begin());
}
void Real4CopyD2D(thrust::device_vector<Real4>& mDFrom, thrust::device_vector<Real4>& mDTo) {
  thrust::copy(mDFrom.begin(), mDFrom.end(), mDTo.begin());
}

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
                      thrust::host_vector<Real3>& omegaAccLRF_fsiBodies_H) {
  thrust::copy(posRigid_fsiBodies_H.begin(), posRigid_fsiBodies_H.end(), posRigid_fsiBodies_D.begin());
  thrust::copy(velMassRigid_fsiBodies_H.begin(), velMassRigid_fsiBodies_H.end(), velMassRigid_fsiBodies_D.begin());
  thrust::copy(accRigid_fsiBodies_H.begin(), accRigid_fsiBodies_H.end(), accRigid_fsiBodies_D.begin());
  thrust::copy(q_fsiBodies_H.begin(), q_fsiBodies_H.end(), q_fsiBodies_D.begin());
  thrust::copy(rigidOmegaLRF_fsiBodies_H.begin(), rigidOmegaLRF_fsiBodies_H.end(), rigidOmegaLRF_fsiBodies_D.begin());
  thrust::copy(omegaAccLRF_fsiBodies_H.begin(), omegaAccLRF_fsiBodies_H.end(), omegaAccLRF_fsiBodies_D.begin());
}
