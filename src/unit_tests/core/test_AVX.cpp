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

    bool printMul = false;
    bool printScale = false;
    bool printAdd = false;

    int ITERATION = 100000;
    int A_row = 5;
    int A_col = 29;
    int B_col = 27;
    int B_row = A_col;

    ChMatrixDynamic<double> A(A_row, A_col);
    ChMatrixDynamic<double> B(B_row, B_col);  // For Multiplication
    ChMatrixDynamic<double> C(A_row, A_col);  // For add/sub
    ChMatrixDynamic<double> AmulB(A_row, B_col);
    ChMatrixDynamic<double> AmulB_ref(A_row, B_col);
    ChMatrixDynamic<double> AAddC(A_row, A_col);
    ChMatrixDynamic<double> AAddC_ref(A_row, A_col);

    A.FillRandom(10, -10);  // Fill a matrix with an element
    B.FillRandom(10, -10);  // Fill a matrix with an element

    GetLog() << "--------------------------------------- \n";
    timer.start();
    for (int j = 0; j < ITERATION; j++)
        AmulB_ref.MatrMultiply(A, B);
    timer.stop();
    double tempTime = timer();
    GetLog() << "The MatrMultiply results in " << timer() << " (s) \n";
    timer.reset();
    timer.start();
    for (int j = 0; j < ITERATION; j++)
        AmulB.MatrMultiplyAVX(A, B);
    timer.stop();
    double AVXTime = timer();
    GetLog() << "The AVX results in " << timer() << " (s) \n";
    GetLog() << "Speed up =  " << tempTime / AVXTime << "x \n";

    //    if (printMul) {
    //        GetLog() << "--------------------------------------- \n";
    //        GetLog() << "AVX result is : ";
    //        AmulB.StreamOUT(GetLog());  // Print a matrix to cout (ex. the console, if open)
    //        GetLog() << "Reference result is : ";
    //        AmulB_ref.StreamOUT(GetLog());
    //        GetLog() << "--------------------------------------- \n";
    //    }
    if (AmulB_ref == AmulB) {
        GetLog() << "MatrMultiplyAVX is Ok ... \n";
    } else {
        GetLog() << "MatrMultiplyAVX is not Good! \n";
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////

    GetLog() << "--------------------------------------- \n";
    timer.start();
    for (int j = 0; j < ITERATION; j++)
        AAddC_ref.MatrAdd(A, C);
    timer.stop();
    tempTime = timer();
    GetLog() << "The MatrAdd results in " << timer() << " (s) \n";
    timer.reset();
    timer.start();
    for (int j = 0; j < ITERATION; j++)
        AAddC.MatrAddAVX(A, C);
    timer.stop();
    AVXTime = timer();
    GetLog() << "The AVX results in " << timer() << " (s) \n";
    GetLog() << "Speed up =  " << tempTime / AVXTime << "x \n";
    //    if (printAdd) {
    //        GetLog() << "--------------------------------------- \n";
    //        GetLog() << "AVX result is : ";
    //        AAddC.StreamOUT(GetLog());  // Print a matrix to cout (ex. the console, if open)
    //        GetLog() << "Reference result is : ";
    //        AAddC_ref.StreamOUT(GetLog());
    //        GetLog() << "--------------------------------------- \n";
    //    }
    if (AAddC_ref == AAddC) {
        GetLog() << "MatrAddAVX is Ok ... \n";
    } else {
        GetLog() << "MatrAddAVX is not Good! \n";
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////
    GetLog() << "--------------------------------------- \n";
    timer.start();
    for (int j = 0; j < ITERATION; j++)
        AmulB_ref.MatrScale(0.021);
    timer.stop();
    tempTime = timer();
    GetLog() << "The MatrScale results in " << timer() << " (s) \n";
    timer.reset();
    timer.start();
    for (int j = 0; j < ITERATION; j++)
        AmulB.MatrScaleAVX(0.021);
    timer.stop();
    AVXTime = timer();
    GetLog() << "The AVX results in " << timer() << " (s) \n";
    GetLog() << "Speed up =  " << tempTime / AVXTime << "x \n";
    //    if (printScale) {
    //        GetLog() << "--------------------------------------- \n";
    //        GetLog() << "AVX result is : ";
    //        AmulB.StreamOUT(GetLog());  // Print a matrix to cout (ex. the console, if open)
    //        GetLog() << "Reference result is : ";
    //        AmulB_ref.StreamOUT(GetLog());
    //        GetLog() << "--------------------------------------- \n";
    //    }
    if (AAddC_ref == AAddC) {
        GetLog() << "MatrScaleAVX is Ok ... \n";
    } else {
        GetLog() << "MatrScaleAVX is not Good! \n";
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////

    GetLog() << "--------------------------------------- \n";
    timer.start();
    for (int j = 0; j < ITERATION; j++)
        AAddC_ref.MatrScale(C);
    timer.stop();
    tempTime = timer();
    GetLog() << "The MatrScale for matrices results in " << timer() << " (s) \n";
    timer.reset();
    timer.start();
    for (int j = 0; j < ITERATION; j++)
        AAddC.MatrScaleAVX(C);
    timer.stop();
    AVXTime = timer();
    GetLog() << "The AVX results in " << timer() << " (s) \n";
    GetLog() << "Speed up =  " << tempTime / AVXTime << "x \n";
    //    if (printScale) {
    //        GetLog() << "--------------------------------------- \n";
    //        GetLog() << "AVX result is : ";
    //        AAddC.StreamOUT(GetLog());  // Print a matrix to cout (ex. the console, if open)
    //        GetLog() << "Reference result is : ";
    //        AAddC_ref.StreamOUT(GetLog());
    //        GetLog() << "--------------------------------------- \n";
    //    }
    if (AAddC_ref == AAddC) {
        GetLog() << "MatrScaleAVX is Ok ... \n";
    } else {
        GetLog() << "MatrScaleAVX is not Good! \n";
    }
    return 0;
}
