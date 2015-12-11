//
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2010 Alessandro Tasora
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file at the top level of the distribution
// and at http://projectchrono.org/license-chrono.txt.
//

///////////////////////////////////////////////////
//
//   Demo on how to use Chrono mathematical
//   functions (vector math, linear algebra, etc)
//
//	 CHRONO
//   ------
//   Multibody dinamics engine
//
// ------------------------------------------------
//             www.deltaknowledge.com
// ------------------------------------------------
///////////////////////////////////////////////////

#include "core/ChTransform.h"
#include "core/ChMatrix.h"
#include "core/ChLog.h"
#include "core/ChVector.h"
#include "core/ChQuadrature.h"
#include "core/ChException.h"
#include "core/ChTimer.h"

using namespace chrono;

int main(int argc, char* argv[]) {
    GetLog() << "CHRONO foundation classes test: math\n\n";
    ChTimer<double> timer;

    chrono::ChMatrixNM<double, 2, 8> A;
    chrono::ChMatrixNM<double, 8, 11> B;
    chrono::ChMatrixNM<double, 2, 11> C;
    chrono::ChMatrixNM<double, 2, 11> D;
    chrono::ChMatrixNM<double, 2, 11> C_REF;

    A.FillElem(1);  // Fill a matrix with an element
    B.FillElem(1);  // Fill a matrix with an element
    // For A and B defined above the C_REF is a matrix with A.GetColumns() for each element
    C_REF.FillElem(A.GetColumns());

    timer.start();

     C.MatrMultiply(A, B);

     //C = A * B;
    timer.stop();
    //	GetLog() << "The method result in " << timer() << " (s) \n";
    // Otherwise check the matrix C against the reference matrix C_REF
    //	C.StreamOUT(GetLog());  // Print a matrix to cout (ex. the console, if open)
    GetLog() << "Reference result : \n ";
    // Print the Reference Matrix
    //	C_REF.StreamOUT(GetLog());
    if (C == C_REF) {
        GetLog() << "Matrices are exactly equal \n";
    } else {
        GetLog() << "Not Goog! \n";
    }

    return 0;
}
