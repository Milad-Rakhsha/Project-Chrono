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
// Further references: (all pages number refers to [1] if not otherwise specified)
// [1] Nocedal&Wright, Numerical Optimization 2nd edition
// [2] D'Apuzzo et al., Starting-point strategies for an infeasible potential reduction method

// Symbol conversion table from [1] to [2]
// [2] | [1]
//  z  |  y
//  y  | lam
//  Q  |  G
// lam |  -
//  b  |  b
//  s  |  -
//  u  |  -
//  v  |  -
//  d  |  -
//  t  |  -
//  G  |  -

// Symbol conversion table from [1] to Chrono
// Chr | [1]
//  M  |  G
//  q  |  x
// Cq  |  A
//  l  | lam
//  f  |  -c
//  E  |  E (+ o - ?)
//  c  |  y
//  b  |  -b



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
	public:
		enum KKT_SOLUTION_METHOD
		{
			STANDARD,
			AUGMENTED,
			NORMAL
		};

	private:
		size_t m; // size of lam, y, A rows
		size_t n; // size of x, G, A columns
		size_t solver_call;
		size_t iteration_count;
		size_t iteration_count_max;

		bool EQUAL_STEP_LENGTH;
		bool ADAPTIVE_ETA;
		bool ONLY_PREDICT;
		bool warm_start_broken;
		bool warm_start;

		struct IPresidual_t
		{
			double rp_nnorm;
			double rd_nnorm;

			IPresidual_t() : rp_nnorm(0), rd_nnorm(0) {};
			explicit IPresidual_t(double tol) : rp_nnorm(tol), rd_nnorm(tol) {};
		} IPtolerances, IPres;

		double mu_tolerance; // stop iterations if mu under this threshold

		KKT_SOLUTION_METHOD KKT_solve_method;

		// variables: x primal; lam dual, y slack
		ChMatrixDynamic<double> x; // ('q' in chrono)
		ChMatrixDynamic<double> y; // ('c' in chrono)
		ChMatrixDynamic<double> lam; // ('l' in chrono)

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
		ChMatrixDynamic<double> b; // rhs of constraints (is '-b' in chrono)
		ChMatrixDynamic<double> c; // part of minimization function (is '-f' in chrono)


		// residuals
		ChMatrixDynamic<double> rp; // primal constraint A*x - y - b = 0
		ChMatrixDynamic<double> rd; // dual constraint G*x - AT*lam + c = 0
		ChMatrixDynamic<double> rpd; // primal-dual constraints
		double mu; // complementarity measure


		// problem matrices and vectors
		ChMatrixDynamic<double> rhs;
		ChMatrixDynamic<double> sol;
		ChCSR3Matrix BigMat;
		ChCSR3Matrix SmallMat;
		ChCSR3Matrix E; // compliance matrix
		
		// MKL engine
		ChMklEngine mkl_engine;

		// temporary vectors
		ChMatrixDynamic<double> vectn; // temporary variable that has always size (n,1)
		ChMatrixDynamic<double> vectm; // temporary variable that has always size (m,1)

		// IP specific functions
		void KKTsolve(double sigma = 0.0);
		void initialize(ChLcpSystemDescriptor& sysd);
		void starting_point_STP1();
		void starting_point_STP2(int n_old);
		void starting_point_Nocedal_WS(int n_old, int m_old); // warm_start; try to reuse solution from previous cycles
		void starting_point_Nocedal(int n_old, int m_old);
		void iterate();
		double find_Newton_step_length(ChMatrix<double>& vect, ChMatrix<double>& Dvect, double eta = 1) const;
		double evaluate_objective_function();

		// Auxiliary
		void reset_dimensions(int n_old, int m_old);
		void fullupdate_residual();
		void make_positive_definite();
		void multiplyA(ChMatrix<double>& vect_in, ChMatrix<double>& vect_out) const;
		void multiplyNegAT(ChMatrix<double>& vect_in, ChMatrix<double>& vect_out) const;
		void multiplyG(ChMatrix<double>& vect_in, ChMatrix<double>& vect_out) const;
		void normalize_Arows();
		bool check_exit_conditions(bool only_mu = true);
		bool check_feasibility(double tolerance);
		
		// Test
		std::ofstream history_file;
		bool print_history;
		void update_history();
		ChMatrixDynamic<double> sol_chrono;
		void generate_solution();
		void LoadProblem();

	public:

		ChInteriorPoint();
		~ChInteriorPoint();
		virtual double Solve(ChLcpSystemDescriptor& sysd) override;
		
		// Auxiliary
		void VerifyKKTconditions(bool print = false);
		void SetKKTSolutionMethod(KKT_SOLUTION_METHOD qp_solve_type_selection) { KKT_solve_method = qp_solve_type_selection; }
		void SetMaxIterations(size_t max_iter){ iteration_count_max = max_iter; }
		void SetTolerances(double rp_tol, double rd_tol, double complementarity_tol)
		{
			IPtolerances.rp_nnorm = rp_tol;
			IPtolerances.rd_nnorm = rd_tol;
			mu_tolerance = complementarity_tol;
		}

		// Test
		void DumpProblem(std::string suffix = "");
		void DumpIPStatus(std::string suffix = "");
		void PrintHistory(bool on_off, std::string filepath = "history_file.txt");
	};

} // end of namespace chrono


#endif