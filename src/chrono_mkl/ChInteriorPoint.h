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


/// Class for Interior-Point methods
/// for QP convex programming


#ifndef CHIPENGINE_H
#define CHIPENGINE_H

///////////////////////////////////////////////////
//
//   ChMklEngine.h
//
//   Use this header if you want to exploit
//   Interior-Point methods
//   from Chrono::Engine programs.
//
//   HEADER file for CHRONO,
//  Multibody dynamics engine
//
// ------------------------------------------------
//             www.deltaknowledge.com
// ------------------------------------------------
///////////////////////////////////////////////////

#include <mkl.h>
#include "chrono_mkl/ChApiMkl.h"
//#include "core/ChMatrixDynamic.h"
#include "chrono_mkl/ChCSR3Matrix.h"
#include "lcp/ChLcpSystemDescriptor.h"
#include "lcp/ChLcpSolver.h"
#include "ChMklEngine.h"

// Interior point methdon based on Numerical Optimization by Nocedal, Wright
// minimize 0.5*xT*G*x + xT*x while Ax>=b (16.54 pag.480)
// WARNING: FOR THE MOMNET THE CONSTRAINTS MUST BE INEQUALITIES

// KKT conditions (16.55 pag.481)
// G*x-AT*lam+c = 0; (dual)
// A*x-y-b = 0; (primal)
// y.*lam = 0 (mixed)
// y>=0
// lam>=0

namespace chrono
{

	class ChApiMkl ChInteriorPoint : public ChLcpSolver
	{
	private:
		size_t m; // size of lam, A rows
		size_t n; // size of x, y, G
		size_t iteration_count_max;
		size_t solver_call;


		bool EQUAL_STEP_LENGTH;
		bool ADAPTIVE_ETA;
		bool ONLY_PREDICT;

		static size_t cycle_count;

		enum QP_SOLUTION_TECHNIQUE
		{
			STANDARD,
			AUGMENTED,
			NORMAL
		} qp_solve_type;

		// variables: (x,y) primal variables; lam dual
		ChMatrixDynamic<double> x;
		ChMatrixDynamic<double> y;
		ChMatrixDynamic<double> lam;

		ChMatrixDynamic<double> x_pred;
		ChMatrixDynamic<double> y_pred;
		ChMatrixDynamic<double> lam_pred;

		ChMatrixDynamic<double> x_corr;
		ChMatrixDynamic<double> y_corr;
		ChMatrixDynamic<double> lam_corr;

		ChMatrixDynamic<double> Dx;
		ChMatrixDynamic<double> Dy;
		ChMatrixDynamic<double> Dlam;

		// vectors
		ChMatrixDynamic<double> b; // rhs of constraints (is -b in chrono)
		ChMatrixDynamic<double> c; // part of minimization function (is -f in chrono)


		// residuals: TODO do not use this intermediate values, but operate directly on 'rhs'
		ChMatrixDynamic<double> rp; // primal constraint A*x - y - b = 0
		ChMatrixDynamic<double> rd; // dual constraint G*x - AT*lam + c = 0
		ChMatrixDynamic<double> rpd_pred; // primal-dual constraints used in the predictor phase
		ChMatrixDynamic<double> rpd_corr; // primal-dual constraints used in the corrector phase

		// problem matrices and vectors
		ChMatrixDynamic<double> rhs;
		ChMatrixDynamic<double> sol;
		ChCSR3Matrix BigMat;
		ChCSR3Matrix SmallMat;

		// MKL engine
		ChMklEngine mkl_engine;

		// temporary vectors
		ChMatrixDynamic<double> vectn; // temporary variable that has always size (n,1)
		ChMatrixDynamic<double> vectm; // temporary variable that has always size (m,1)


		void KKTsolve(double sigma_dot_mu = 0);
		double findNewtonStepLength(ChMatrix<double>& vect, ChMatrix<double>& Dvect, double eta = 1);
		void reset_dimensions();

		void dump_all(std::string suffix = "");

		void makePositiveDefinite(ChCSR3Matrix* mat);
		void fullupdate_residual();


	public:

		ChInteriorPoint(QP_SOLUTION_TECHNIQUE qp_solve_type_selection = AUGMENTED);

		virtual double Solve(ChLcpSystemDescriptor& sysd) override;
		void Initialize(ChLcpSystemDescriptor& sysd);
		void InteriorPointIterate();
		
	};



} // end of namespace chrono


#endif