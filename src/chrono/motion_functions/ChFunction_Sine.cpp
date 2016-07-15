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

#include "chrono/motion_functions/ChFunction_Sine.h"

namespace chrono {

// Register into the object factory, to enable run-time dynamic creation and persistence
ChClassRegister<ChFunction_Sine> a_registration_sine;

ChFunction_Sine::ChFunction_Sine(const ChFunction_Sine& other) {
    amp = other.amp;
    phase = other.phase;
    freq = other.freq;
    w = other.w;
}

double ChFunction_Sine::Get_y(double x) const {
    return amp * (sin(phase + w * x));
}

double ChFunction_Sine::Get_y_dx(double x) const {
    return amp * w * (cos(phase + w * x));
}

double ChFunction_Sine::Get_y_dxdx(double x) const {
    return amp * -w * w * (sin(phase + w * x));
}

}  // end namespace chrono
