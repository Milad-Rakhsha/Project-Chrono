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

#include "chrono/geometry/ChRoundedCylinder.h"

namespace chrono {
namespace geometry {

// Register into the object factory, to enable run-time dynamic creation and persistence
ChClassRegister<ChRoundedCylinder> a_registration_ChRoundedCylinder;

ChRoundedCylinder::ChRoundedCylinder(const ChRoundedCylinder& source) {
    center = source.center;
    rad = source.rad;
    hlen = source.hlen;
    radsphere = source.radsphere;
}

void ChRoundedCylinder::CovarianceMatrix(ChMatrix33<>& C) const {
    C.Reset();
    C(0, 0) = center.x * center.x;
    C(1, 1) = center.y * center.y;
    C(2, 2) = center.z * center.z;
}

}  // end namespace geometry
}  // end namespace chrono
