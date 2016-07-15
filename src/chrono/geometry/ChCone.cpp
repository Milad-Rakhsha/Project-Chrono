// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2014 projectchrono.org
// All right reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Authors: Alessandro Tasora, Radu Serban
// =============================================================================

#include <stdio.h>

#include "chrono/geometry/ChCone.h"

namespace chrono {
namespace geometry {

// Register into the object factory, to enable run-time dynamic creation and persistence
ChClassRegister<ChCone> a_registration_ChCone;

ChCone::ChCone(const ChCone& source) {
    center = source.center;
    rad = source.rad;
}

}  // end namespace geometry
}  // end namespace chrono
