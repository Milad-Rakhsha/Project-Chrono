/*
 * SPHCudaUtils.h
 *
 *  Created on: Mar 2, 2015
 *      Author: Arman Pazouki
 */
// ****************************************************************************
// This file contains miscellaneous macros and utilities used in the SPH code.
// ****************************************************************************
#ifndef CH_SPH_GENERAL_CUH
#define CH_SPH_GENERAL_CUH

// ----------------------------------------------------------------------------
// CUDA headers
// ----------------------------------------------------------------------------
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include "chrono_fsi/ChDeviceUtils.cuh"
#include "chrono_fsi/ChFsiDataManager.cuh"
#include "chrono_fsi/ChParams.cuh"
#include "chrono_fsi/custom_math.h"
#include "chrono_fsi/solver6x6.cuh"

// #include <cstdio>

namespace chrono {
namespace fsi {

__constant__ SimParams paramsD;
__constant__ NumberOfObjects numObjectsD;

inline void CopyParams_NumberOfObjects(SimParams* paramsH, NumberOfObjects* numObjectsH);
//--------------------------------------------------------------------------------------------------------------------------------
// 3D SPH kernel function, W3_SplineA
__device__ inline Real W3_Spline(Real d) {  // d is positive. h is the sph particle radius (i.e. h in
                                            // the document) d is the distance of 2 particles
    Real h = paramsD.HSML;
    Real q = fabs(d) / h;
    if (q < 1) {
        return (0.25f / (PI * h * h * h) * (pow(2 - q, Real(3)) - 4 * pow(1 - q, Real(3))));
    }
    if (q < 2) {
        return (0.25f / (PI * h * h * h) * pow(2 - q, Real(3)));
    }
    return 0;
}
//--------------------------------------------------------------------------------------------------------------------------------
// 3D SPH kernel function, W3_SplineA
__device__ inline Real W3h_Spline(Real d, Real h) {  // d is positive. h is the sph particle radius (i.e. h in
                                                     // the document) d is the distance of 2 particles
    Real q = fabs(d) / h;
    if (q < 1) {
        return (0.25f / (PI * h * h * h) * (pow(2 - q, Real(3)) - 4 * pow(1 - q, Real(3))));
    }
    if (q < 2) {
        return (0.25f / (PI * h * h * h) * pow(2 - q, Real(3)));
    }
    return 0;
}
////--------------------------------------------------------------------------------------------------------------------------------
////2D SPH kernel function, W2_SplineA
//__device__ inline Real W2_Spline(Real d) { // d is positive. h is the sph
// particle radius (i.e. h in the document) d
// is the distance of 2 particles
//	Real h = paramsD.HSML;
//	Real q = fabs(d) / h;
//	if (q < 1) {
//		return (5 / (14 * PI * h * h) * (pow(2 - q, Real(3)) - 4 * pow(1 -
// q, Real(3))));
//	}
//	if (q < 2) {
//		return (5 / (14 * PI * h * h) * pow(2 - q, Real(3)));
//	}
//	return 0;
//}
////--------------------------------------------------------------------------------------------------------------------------------
////3D SPH kernel function, W3_QuadraticA
//__device__ inline Real W3_Quadratic(Real d, Real h) { // d is positive. h is
// the sph particle radius (i.e. h in the
// document) d is the distance of 2 particles
//	Real q = fabs(d) / h;
//	if (q < 2) {
//		return (1.25f / (PI * h * h * h) * .75f * (pow(.5f * q, Real(2)) -
// q + 1));
//	}
//	return 0;
//}
////--------------------------------------------------------------------------------------------------------------------------------
////2D SPH kernel function, W2_QuadraticA
//__device__ inline Real W2_Quadratic(Real d, Real h) { // d is positive. h is
// the sph particle radius (i.e. h in the
// document) d is the distance of 2 particles
//	Real q = fabs(d) / h;
//	if (q < 2) {
//		return (2.0f / (PI * h * h) * .75f * (pow(.5f * q, Real(2)) - q +
// 1));
//	}
//	return 0;
//}
//--------------------------------------------------------------------------------------------------------------------------------
// Gradient of the kernel function
// d: magnitude of the distance of the two particles
// dW * dist3 gives the gradiant of W3_Quadratic, where dist3 is the distance
// vector of the two particles, (dist3)a =
// pos_a - pos_b
__device__ inline Real3 GradW_Spline(Real3 d) {  // d is positive. r is the sph particle radius (i.e. h
                                                 // in the document) d is the distance of 2 particles
    Real h = paramsD.HSML;
    Real q = length(d) / h;
    bool less1 = (q < 1);
    bool less2 = (q < 2);
    return (less1 * (3 * q - 4) + less2 * (!less1) * (-q + 4.0f - 4.0f / q)) * .75f * (INVPI)*pow(h, Real(-5)) * d;
    //	if (q < 1) {
    //		return .75f * (INVPI) *pow(h, Real(-5))* (3 * q - 4) * d;
    //	}
    //	if (q < 2) {
    //		return .75f * (INVPI) *pow(h, Real(-5))* (-q + 4.0f - 4.0f / q) *
    // d;
    //	}
    //	return mR3(0);
}

__device__ inline Real3 GradWh_Spline(Real3 d, Real h) {  // d is positive. r is the sph particle radius (i.e. h
                                                          // in the document) d is the distance of 2 particles
    Real q = length(d) / h;
    bool less1 = (q < 1);
    bool less2 = (q < 2);
    return (less1 * (3 * q - 4) + less2 * (!less1) * (-q + 4.0f - 4.0f / q)) * .75f * (INVPI)*pow(h, Real(-5)) * d;
}
////--------------------------------------------------------------------------------------------------------------------------------
////Gradient of the kernel function
//// d: magnitude of the distance of the two particles
//// dW * dist3 gives the gradiant of W3_Quadratic, where dist3 is the distance
/// vector of the two particles, (dist3)a =
/// pos_a - pos_b
//__device__ inline Real3 GradW_Quadratic(Real3 d, Real h) { // d is positive. r
// is the sph particle radius (i.e. h in
// the document) d is the distance of 2 particles
//	Real q = length(d) / h;
//	if (q < 2) {
//		return 1.25f / (PI * pow(h, Real(5))) * .75f * (.5f - 1.0f / q) *
// d;
//	}
//	return mR3(0);
//}
//--------------------------------------------------------------------------------------------------------------------------------
#define W3 W3_Spline
#define W3h W3h_Spline

//#define W2 W2_Spline
#define GradW GradW_Spline
#define GradWh GradWh_Spline

//--------------------------------------------------------------------------------------------------------------------------------
// Eos is also defined in SDKCollisionSystem.cu
// fluid equation of state
__device__ inline Real Eos(Real rho, Real type) {
    ////******************************
    // Real gama = 1;
    // if (type < -.1) {
    //	return 1 * (100000 * (pow(rho / paramsD.rho0, gama) - 1) +
    // paramsD.BASEPRES);
    //	//return 100 * rho;
    //}
    //////else {
    //////	return 1e9;
    //////}

    //******************************
    Real gama = 7;
    Real B = 100 * paramsD.rho0 * paramsD.v_Max * paramsD.v_Max /
             gama;  // 200;//314e6; //c^2 * paramsD.rho0 / gama where c = 1484 m/s
                    // for water
    return B * (pow(rho / paramsD.rho0, gama) - 1) +
           paramsD.BASEPRES;  // 1 * (B * (pow(rho / paramsD.rho0, gama) - 1) +
                              // paramsD.BASEPRES);
                              //	if (type < +.1f) {
                              //		return B * (pow(rho / paramsD.rho0, gama) - 1) + paramsD.BASEPRES;
                              //// 1 * (B * (pow(rho / paramsD.rho0,
                              // gama) - 1) + paramsD.BASEPRES);
                              //	} else
                              //		return paramsD.BASEPRES;
}
//--------------------------------------------------------------------------------------------------------------------------------
__device__ inline Real InvEos(Real pw) {
    Real gama = 7;
    Real B = 100 * paramsD.rho0 * paramsD.v_Max * paramsD.v_Max /
             gama;  // 200;//314e6; //c^2 * paramsD.rho0 / gama where c = 1484 m/s
                    // for water
    Real powerComp = (pw - paramsD.BASEPRES) / B + 1.0;
    Real rho = (powerComp > 0) ? paramsD.rho0 * pow(powerComp, 1.0 / gama)
                               : -paramsD.rho0 * pow(fabs(powerComp),
                                                     1.0 / gama);  // did this since CUDA is
                                                                   // stupid and freaks out by
                                                                   // negative^(1/gama)
    return rho;
}
//--------------------------------------------------------------------------------------------------------------------------------
// ferrariCi
__device__ inline Real FerrariCi(Real rho) {
    int gama = 7;
    Real B = 100 * paramsD.rho0 * paramsD.v_Max * paramsD.v_Max /
             gama;  // 200;//314e6; //c^2 * paramsD.rho0 / gama where c = 1484 m/s
                    // for water
    return sqrt(gama * B / paramsD.rho0) * pow(rho / paramsD.rho0, 0.5 * (gama - 1));
}
//--------------------------------------------------------------------------------------------------------------------------------

__device__ inline Real3 Modify_Local_PosB(Real3& b, Real3 a) {
    Real3 dist3 = a - b;
    b.x += ((dist3.x > 0.5f * paramsD.boxDims.x) ? paramsD.boxDims.x : 0);
    b.x -= ((dist3.x < -0.5f * paramsD.boxDims.x) ? paramsD.boxDims.x : 0);

    b.y += ((dist3.y > 0.5f * paramsD.boxDims.y) ? paramsD.boxDims.y : 0);
    b.y -= ((dist3.y < -0.5f * paramsD.boxDims.y) ? paramsD.boxDims.y : 0);

    b.z += ((dist3.z > 0.5f * paramsD.boxDims.z) ? paramsD.boxDims.z : 0);
    b.z -= ((dist3.z < -0.5f * paramsD.boxDims.z) ? paramsD.boxDims.z : 0);

    dist3 = a - b;
    // modifying the markers perfect overlap
    Real d = length(dist3);
    if (d < paramsD.epsMinMarkersDis * paramsD.HSML) {
        dist3 = mR3(paramsD.epsMinMarkersDis * paramsD.HSML, 0, 0);
    }
    b = a - dist3;
    return (dist3);
}

/**
 * @brief Distance
 * @details
 *          Distance between two particles, considering the periodic boundary
 * condition
 *
 * @param a Position of Particle A
 * @param b Position of Particle B
 *
 * @return Distance vector (distance in x, distance in y, distance in z)
 */
__device__ inline Real3 Distance(Real3 a, Real3 b) {
    //	Real3 dist3 = a - b;
    //	dist3.x -= ((dist3.x > 0.5f * paramsD.boxDims.x) ? paramsD.boxDims.x :
    // 0);
    //	dist3.x += ((dist3.x < -0.5f * paramsD.boxDims.x) ? paramsD.boxDims.x :
    // 0);
    //
    //	dist3.y -= ((dist3.y > 0.5f * paramsD.boxDims.y) ? paramsD.boxDims.y :
    // 0);
    //	dist3.y += ((dist3.y < -0.5f * paramsD.boxDims.y) ? paramsD.boxDims.y :
    // 0);
    //
    //	dist3.z -= ((dist3.z > 0.5f * paramsD.boxDims.z) ? paramsD.boxDims.z :
    // 0);
    //	dist3.z += ((dist3.z < -0.5f * paramsD.boxDims.z) ? paramsD.boxDims.z :
    // 0);
    return Modify_Local_PosB(b, a);
}

//--------------------------------------------------------------------------------------------------------------------------------
// first comp of q is rotation, last 3 components are axis of rot

__device__ inline void RotationMatirixFromQuaternion(Real3& AD1, Real3& AD2, Real3& AD3, const Real4& q) {
    AD1 = 2 * mR3(0.5f - q.z * q.z - q.w * q.w, q.y * q.z - q.x * q.w, q.y * q.w + q.x * q.z);
    AD2 = 2 * mR3(q.y * q.z + q.x * q.w, 0.5f - q.y * q.y - q.w * q.w, q.z * q.w - q.x * q.y);
    AD3 = 2 * mR3(q.y * q.w - q.x * q.z, q.z * q.w + q.x * q.y, 0.5f - q.y * q.y - q.z * q.z);
}
//--------------------------------------------------------------------------------------------------------------------------------

__device__ inline Real3 InverseRotate_By_RotationMatrix_DeviceHost(const Real3& A1,
                                                                   const Real3& A2,
                                                                   const Real3& A3,
                                                                   const Real3& r3) {
    return mR3(A1.x * r3.x + A2.x * r3.y + A3.x * r3.z, A1.y * r3.x + A2.y * r3.y + A3.y * r3.z,
               A1.z * r3.x + A2.z * r3.y + A3.z * r3.z);
}
//--------------------------------------------------------------------------------------------------------------------------------

/**
 * @brief calcGridHash
 * @details  See SDKCollisionSystem.cuh
 */
__device__ inline int3 calcGridPos(Real3 p) {
    int3 gridPos;
    if (paramsD.cellSize.x * paramsD.cellSize.y * paramsD.cellSize.z == 0)
        printf("calcGridPos=%f,%f,%f\n", paramsD.cellSize.x, paramsD.cellSize.y, paramsD.cellSize.z);

    gridPos.x = floor((p.x - paramsD.worldOrigin.x) / paramsD.cellSize.x);
    gridPos.y = floor((p.y - paramsD.worldOrigin.y) / paramsD.cellSize.y);
    gridPos.z = floor((p.z - paramsD.worldOrigin.z) / paramsD.cellSize.z);
    return gridPos;
}

/**
 * @brief calcGridHash
 * @details  See SDKCollisionSystem.cuh
 */

__device__ inline uint calcGridHash(int3 gridPos) {
    gridPos.x -= ((gridPos.x >= paramsD.gridSize.x) ? paramsD.gridSize.x : 0);
    gridPos.y -= ((gridPos.y >= paramsD.gridSize.y) ? paramsD.gridSize.y : 0);
    gridPos.z -= ((gridPos.z >= paramsD.gridSize.z) ? paramsD.gridSize.z : 0);

    gridPos.x += ((gridPos.x < 0) ? paramsD.gridSize.x : 0);
    gridPos.y += ((gridPos.y < 0) ? paramsD.gridSize.y : 0);
    gridPos.z += ((gridPos.z < 0) ? paramsD.gridSize.z : 0);

    return gridPos.z * paramsD.gridSize.y * paramsD.gridSize.x + gridPos.y * paramsD.gridSize.x + gridPos.x;
}
////--------------------------------------------------------------------------------------------------------------------------------
inline __device__ void BCE_Vel_Acc(int i_idx,
                                   Real3& myAcc,
                                   Real3& V_prescribed,

                                   Real4* sortedPosRad,
                                   int4 updatePortion,
                                   uint* gridMarkerIndexD,

                                   Real4* velMassRigid_fsiBodies_D,
                                   Real3* accRigid_fsiBodies_D,
                                   uint* rigidIdentifierD,

                                   Real3* pos_fsi_fea_D,
                                   Real3* vel_fsi_fea_D,
                                   Real3* acc_fsi_fea_D,
                                   uint* FlexIdentifierD,
                                   const int numFlex1D,
                                   uint2* CableElementsNodes,
                                   uint4* ShellelementsNodes);
//--------------------------------------------------------------------------------------------------------------------------------
inline __device__ void grad_scalar(int i_idx,
                                   Real4* sortedPosRad,  // input: sorted positions
                                   Real4* sortedRhoPreMu,
                                   Real* sumWij_inv,
                                   Real* G_i,
                                   Real4* Scalar,
                                   Real3& myGrad,
                                   uint* cellStart,
                                   uint* cellEnd);

//--------------------------------------------------------------------------------------------------------------------------------
inline __device__ void grad_vector(int i_idx,
                                   Real4* sortedPosRad,  // input: sorted positions
                                   Real4* sortedRhoPreMu,
                                   Real* sumWij_inv,
                                   Real* G_i,
                                   Real3* Vector,
                                   Real3& myGradx,
                                   Real3& myGrady,
                                   Real3& myGradz,
                                   uint* cellStart,
                                   uint* cellEnd);
//--------------------------------------------------------------------------------------------------------------------------------
inline __global__ void calc_A_tensor(Real* A_tensor,
                                     Real* G_tensor,
                                     Real4* sortedPosRad,
                                     Real4* sortedRhoPreMu,
                                     Real* sumWij_inv,
                                     uint* cellStart,
                                     uint* cellEnd,
                                     const int numAllMarkers,
                                     volatile bool* isErrorD);
//--------------------------------------------------------------------------------------------------------------------------------
inline __global__ void calc_L_tensor(Real* A_tensor,
                                     Real* L_tensor,
                                     Real* G_tensor,
                                     Real4* sortedPosRad,
                                     Real4* sortedRhoPreMu,
                                     Real* sumWij_inv,
                                     uint* cellStart,
                                     uint* cellEnd,
                                     const int numAllMarkers,
                                     volatile bool* isErrorD);

//--------------------------------------------------------------------------------------------------------------------------------
inline __global__ void calcRho_kernel(Real4* sortedPosRad,  // input: sorted positionsmin(
                                      Real4* sortedRhoPreMu,
                                      Real* sumWij_inv,
                                      uint* cellStart,
                                      uint* cellEnd,
                                      uint* mynumContact,
                                      const int numAllMarkers,
                                      volatile bool* isErrorD);

//--------------------------------------------------------------------------------------------------------------------------------
inline __global__ void calcNormalizedRho_kernel(Real4* sortedPosRad,  // input: sorted positions
                                                Real3* sortedVelMas,
                                                Real4* sortedRhoPreMu,
                                                Real* sumWij_inv,
                                                Real* G_i,
                                                Real* dxi_over_Vi,
                                                Real* Color,
                                                uint* cellStart,
                                                uint* cellEnd,
                                                const int numAllMarkers,
                                                volatile bool* isErrorD);
//--------------------------------------------------------------------------------------------------------------------------------
inline __global__ void calcNormalizedRho_Gi_fillInMatrixIndices(Real4* sortedPosRad,  // input: sorted positions
                                                                Real3* sortedVelMas,
                                                                Real4* sortedRhoPreMu,
                                                                Real* sumWij_inv,
                                                                Real* G_i,
                                                                uint* csrColInd,
                                                                unsigned long int* GlobalcsrColInd,
                                                                uint* numContacts,
                                                                uint* cellStart,
                                                                uint* cellEnd,
                                                                const int numAllMarkers,
                                                                volatile bool* isErrorD);
//--------------------------------------------------------------------------------------------------------------------------------
inline __global__ void Gradient_Laplacian_Operator(Real4* sortedPosRad,  // input: sorted positions
                                                   Real3* sortedVelMas,
                                                   Real4* sortedRhoPreMu,
                                                   Real* sumWij_inv,
                                                   Real* G_tensor,
                                                   Real* L_tensor,
                                                   Real* A_L,   // velcotiy Laplacian matrix;
                                                   Real3* A_G,  // This is a matrix in a way that A*p gives the gradp
                                                   uint* csrColInd,
                                                   unsigned long int* GlobalcsrColInd,
                                                   uint* numContacts,
                                                   const int numAllMarkers,
                                                   volatile bool* isErrorD);
//--------------------------------------------------------------------------------------------------------------------------------
inline __global__ void Initialize_Variables(Real4* sortedRhoPreMu,
                                            Real* p_old,
                                            Real3* sortedVelMas,
                                            Real3* V_new,
                                            const int numAllMarkers,
                                            volatile bool* isErrorD);

//--------------------------------------------------------------------------------------------------------------------------------
inline __global__ void UpdateDensity(Real3* vis_vel,
                                     Real3* new_vel,       // Write
                                     Real4* sortedPosRad,  // Read
                                     Real4* sortedRhoPreMu,
                                     Real* sumWij_inv,
                                     uint* cellStart,
                                     uint* cellEnd,
                                     uint numAllMarkers,
                                     volatile bool* isErrorD);

//--------------------------------------------------------------------------------------------------------------------------------
inline __global__ void Calc_HelperMarkers_normals(Real4* sortedPosRad,
                                                  Real4* sortedRhoPreMu,
                                                  Real3* helpers_normal,
                                                  int* myType,
                                                  uint* cellStart,
                                                  uint* cellEnd,
                                                  uint* gridMarkerIndexD,
                                                  int numAllMarkers,
                                                  volatile bool* isErrorD);
//--------------------------------------------------------------------------------------------------------------------------------
inline __global__ void Calc_Splits_and_Merges(Real4* sortedPosRad,
                                              Real4* sortedRhoPreMu,
                                              Real3* sortedVelMas,
                                              Real3* helpers_normal,
                                              Real* sumWij_inv,
                                              Real* G_i,
                                              uint* splitMe,
                                              uint* MergeMe,
                                              int* myType,
                                              uint* cellStart,
                                              uint* cellEnd,
                                              uint* gridMarkerIndexD,
                                              Real fineResolution,
                                              Real coarseResolution,
                                              int numAllMarkers,
                                              int limit,
                                              volatile bool* isErrorD);

//--------------------------------------------------------------------------------------------------------------------------------
inline __global__ void Split(Real4* sortedPosRad,
                             Real4* sortedRhoPreMu,
                             Real3* sortedVelMas,
                             Real3* helpers_normal,
                             Real* sumWij_inv,
                             Real* G_i,
                             Real* L_i,
                             uint* splitMe,
                             uint* MergeMe,
                             int* myType,
                             uint* cellStart,
                             uint* cellEnd,
                             uint* gridMarkerIndexD,
                             Real fineResolution,
                             Real coarseResolution,
                             int numAllMarkers,
                             int limit,
                             volatile bool* isErrorD);
//--------------------------------------------------------------------------------------------------------------------------------

}  // namespace fsi
}  // namespace chrono
#endif
