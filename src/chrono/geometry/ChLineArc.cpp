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

#include "chrono/geometry/ChLineArc.h"

namespace chrono {
namespace geometry {

// Register into the object factory, to enable run-time dynamic creation and persistence
ChClassRegister<ChLineArc> a_registration_ChLineArc;

ChLineArc::ChLineArc(const ChCoordsys<> morigin,
                     const double mradius,
                     const double mangle1,
                     const double mangle2,
                     const bool mcounterclockwise)
    : origin(morigin), radius(mradius), angle1(mangle1), angle2(mangle2), counterclockwise(mcounterclockwise) {}

ChLineArc::ChLineArc(const ChLineArc& source) : ChLine(source) {
    origin = source.origin;
    radius = source.radius;
    angle1 = source.angle1;
    angle2 = source.angle2;
    counterclockwise = source.counterclockwise;
}

void ChLineArc::Evaluate(ChVector<>& pos, const double parU, const double parV, const double parW) const {
    double ang1 = this->angle1;
    double ang2 = this->angle2;
    if (this->counterclockwise) {
        if (ang2 < ang1)
            ang2 += CH_C_2PI;
    } else {
        if (ang2 > ang1)
            ang2 -= CH_C_2PI;
    }
    double mangle = ang1 * (1 - parU) + ang2 * (parU);
    ChVector<> localP(radius * cos(mangle), radius * sin(mangle), 0);
    pos = localP >> origin;  // translform to absolute coordinates
}

}  // end namespace geometry
}  // end namespace chrono
