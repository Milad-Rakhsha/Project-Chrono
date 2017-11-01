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
// Author:  Milad Rakhsha
// =============================================================================
//
// Base class for processing sph force in fsi system.//
// =============================================================================

#ifndef CH_SPHKERNELS_CU_
#define CH_SPHKERNELS_CU_
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>
#include "chrono_fsi/ChParams.cuh"
#include "chrono_fsi/ChSphGeneral.cuh"
#include "chrono_fsi/solver6x6.cuh"

namespace chrono {
namespace fsi {}  // namespace fsi
}  // namespace chrono
#endif
