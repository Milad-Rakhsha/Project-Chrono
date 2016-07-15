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

#include "chrono/geometry/ChTriangleMeshSoup.h"

namespace chrono {
namespace geometry {

// Register into the object factory, to enable run-time dynamic creation and persistence
ChClassRegister<ChTriangleMeshSoup> a_registration_ChTriangleMeshSoup;

ChTriangleMeshSoup::ChTriangleMeshSoup(const ChTriangleMeshSoup& source) {
    m_triangles = source.m_triangles;
}

void ChTriangleMeshSoup::Transform(const ChVector<> displ, const ChMatrix33<> rotscale) {
    for (int i = 0; i < this->m_triangles.size(); ++i) {
        m_triangles[i].p1 = rotscale * m_triangles[i].p1;
        m_triangles[i].p1 += displ;
        m_triangles[i].p2 = rotscale * m_triangles[i].p2;
        m_triangles[i].p2 += displ;
        m_triangles[i].p3 = rotscale * m_triangles[i].p3;
        m_triangles[i].p3 += displ;
    }
}

}  // end namespace geometry
}  // end namespace chrono
