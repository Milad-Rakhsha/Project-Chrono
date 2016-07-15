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

#include "chrono/motion_functions/ChFunction_Matlab.h"

namespace chrono {

// Register into the object factory, to enable run-time dynamic creation and persistence
ChClassRegister<ChFunction_Matlab> a_registration_matlab;

ChFunction_Matlab::ChFunction_Matlab() {
    strcpy(this->mat_command, "x*2+x^2");
}

ChFunction_Matlab::ChFunction_Matlab(const ChFunction_Matlab& other) {
    strcpy(this->mat_command, other.mat_command);
}

double ChFunction_Matlab::Get_y(double x) const {
    double ret = 0;

#ifdef CH_MATLAB
    char m_eval_command[CHF_MATLAB_STRING_LEN + 20];

    // no function: shortcut!
    if (*this->mat_command == NULL)
        return 0.0;

    // set string as "x=[x];ans=[mat_command]"
    sprintf(m_eval_command, "x=%g;ans=%s;", x, this->mat_command);

    // EVAL string, retrieving y = "ans"
    ret = CHGLOBALS().Mat_Eng_Eval(m_eval_command);
#endif

    return ret;
}

double ChFunction_Matlab::Get_y_dx(double x) const {
    return ((Get_y(x + BDF_STEP_HIGH) - Get_y(x)) / BDF_STEP_HIGH);
}

double ChFunction_Matlab::Get_y_dxdx(double x) const {
    return ((Get_y_dx(x + BDF_STEP_HIGH) - Get_y_dx(x)) / BDF_STEP_HIGH);
}

}  // end namespace chrono
