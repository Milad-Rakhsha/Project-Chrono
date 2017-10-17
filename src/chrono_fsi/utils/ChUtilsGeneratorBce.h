// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2014 projectchrono.org
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Author: Arman Pazouki
// =============================================================================
//
// Utility class for generating fluid markers.//
// =============================================================================

#ifndef CH_UTILSGENERATORBCE__CUH
#define CH_UTILSGENERATORBCE__CUH

#include <thrust/host_vector.h>
#include <string>
#include "chrono_fea/ChElementCableANCF.h"
#include "chrono_fea/ChElementShellANCF.h"
#include "chrono_fsi/ChParams.cuh"
#include "chrono_fsi/custom_math.h"

namespace chrono {
namespace fsi {
namespace utils {
// =============================================================================
void CreateBCE_On_Sphere(thrust::host_vector<Real4>& posRadBCE, Real rad, SimParams* paramsH);
void CreateBCE_On_surface_of_Sphere(thrust::host_vector<Real4>& posRadBCE, Real rad, Real kernel_h);
void CreateBCE_On_Cylinder(thrust::host_vector<Real4>& posRadBCE,
                           Real cyl_rad,
                           Real cyl_h,
                           SimParams* paramsH,
                           Real kernel_h);

void CreateBCE_On_surface_of_Cylinder(thrust::host_vector<Real4>& posRadBCE,
                                      thrust::host_vector<Real3>& normals,
                                      Real cyl_rad,
                                      Real cyl_h,
                                      Real spacing);

void CreateBCE_On_Box(thrust::host_vector<Real4>& posRadBCE, const Real3& hsize, int face, SimParams* paramsH);

void CreateBCE_On_shell(thrust::host_vector<Real4>& posRadBCE,
                        SimParams* paramsH,
                        std::shared_ptr<chrono::fea::ChElementShellANCF> shell,
                        bool multiLayer = true,
                        bool removeMiddleLayer = false,
                        int SIDE = -2);

void CreateBCE_On_ChElementCableANCF(thrust::host_vector<Real4>& posRadBCE,
                                     SimParams* paramsH,
                                     std::shared_ptr<chrono::fea::ChElementCableANCF> cable,
                                     std::vector<int> remove,
                                     bool multiLayer = true,
                                     bool removeMiddleLayer = false,
                                     int SIDE = 1);

void CreateBCE_On_ChElementShellANCF(thrust::host_vector<Real4>& posRadBCE,
                                     SimParams* paramsH,
                                     std::shared_ptr<chrono::fea::ChElementShellANCF> shell,
                                     std::vector<int> remove,
                                     bool multiLayer = true,
                                     bool removeMiddleLayer = false,
                                     int SIDE = -2);

void LoadBCE_fromFile(thrust::host_vector<Real4>& posRadBCE,  // do not set the
                                                              // size here since
                                                              // you are using
                                                              // push back later
                      std::string fileName);

}  // end namespace utils
}  // end namespace fsi
}  // end namespace chrono
#endif
