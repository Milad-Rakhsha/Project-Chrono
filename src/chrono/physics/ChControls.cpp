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

#include "chrono/physics/ChControls.h"

namespace chrono {

// Register into the object factory, to enable run-time dynamic creation and persistence
ChClassRegisterABSTRACT<ChControls> a_registration_ChControls;

void ChControls::ArchiveOUT(ChArchiveOut& marchive) {
    // version number
    marchive.VersionWrite(1);

    // serialize parent class
    ChObj::ArchiveOUT(marchive);
}

/// Method to allow de serialization of transient data from archives.
void ChControls::ArchiveIN(ChArchiveIn& marchive) {
    // version number
    int version = marchive.VersionRead();

    // deserialize parent class
    ChObj::ArchiveIN(marchive);
}

}  // end namespace chrono
