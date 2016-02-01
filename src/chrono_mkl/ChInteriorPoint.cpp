#include "ChInteriorPoint.h"
#include <algorithm>

#define DEBUG_MODE true
#define SKIP_CONTACTS_UV true
#define ADD_COMPLIANCE true

namespace chrono
{
	template <class matrix>
	void ExportToFile(matrix mat, std::string filepath, int precision = 12)
	{
		std::ofstream ofile;
		ofile.open(filepath);
		ofile << std::scientific << std::setprecision(precision);

		for (size_t row_sel = 0; row_sel < mat.GetRows(); row_sel++)
		{
			for (size_t col_sel = 0; col_sel < mat.GetColumns(); col_sel++)
			{
				ofile << mat.GetElement(row_sel, col_sel) << " , ";
			}

			ofile << std::endl;
		}

		ofile.close();
	}

	template<class ChMatrixIN>
	void PrintMatrix(ChMatrixIN& matrice){
		for (size_t i = 0; i < matrice.GetRows(); i++){
			for (size_t j = 0; j < matrice.GetColumns(); j++){
				printf("%.1f ", matrice.GetElement(i, j));
			}
			printf("\n");
		}
	}

	// | M  -Cq'|*|q|- | f|= |0|
    // | Cq  -E | |l|  |-b|  |c| 

	double ChInteriorPoint::Solve(ChLcpSystemDescriptor& sysd)
	{
		initialize(sysd);

		SetTolerances(1e-9, 1e-10, 1e-10);

		size_t iteration_count = 0;
		for (; iteration_count < iteration_count_max; iteration_count++)
		{
			if (check_exit_conditions())
			{
				break;
			}
				
			iterate();
		}
 		solver_call++;

		if (verbose)
		{
			std::cout << "IP call: " << solver_call << "; Iter: " << iteration_count << "/" << iteration_count_max << std::endl;
			VerifyKKTconditions(true);
		}

		generate_solution();

		sysd.FromVectorToUnknowns(sol_chrono);

		return 0;
	}

	void ChInteriorPoint::VerifyKKTconditions(bool print)
	{
		IPresiduals.rp_nnorm = rp.NormTwo()/n;
		IPresiduals.rd_nnorm = rd.NormTwo()/m;
			
		if (print)
		{
			std::cout << std::scientific << std::setprecision(1);
			std::cout << "|rp|/n: " << IPresiduals.rp_nnorm
					  << "; |rd|/m: " << IPresiduals.rd_nnorm
					  << "; mu: " << mu << std::endl;

			bool neg_y = false;
			bool neg_lam = false;
			for (int cont = 0; cont < m; cont++)
			{
				if (y(cont, 0) < 0)
					neg_y = true;
				if (lam(cont, 0) < 0)
					neg_lam = true;
			}

			if (neg_y)
				std::cout << "y has negative elements" << std::endl;

			if (neg_lam)
				std::cout << "lam has negative elements" << std::endl;

		}
	}

	// The problem to be solved is loaded into the main matrix that will be used to solve the various step of the IP method
	// The initial guess is modified in order to be feasible
	// The residuals are computed
	void ChInteriorPoint::initialize(ChLcpSystemDescriptor& sysd)
	{
		verbose = true;

		/********** Load system **********/
		// dimensions change at every call; 'm' for sure, but maybe 'n' does not?
		int n_old = n;
		int m_old = m;
		n = sysd.CountActiveVariables();
		m = sysd.CountActiveConstraints(false, SKIP_CONTACTS_UV);
		reset_dimensions();

		// load system matrix in 'BigMat', 'rhs', 'b' and 'c'
		switch (KKT_solve_method)
		{
		case STANDARD:
			sysd.ConvertToMatrixForm(&BigMat, nullptr, false, SKIP_CONTACTS_UV, 2);
			make_positive_definite();
			break;
		case AUGMENTED:
			sysd.ConvertToMatrixForm(&BigMat, nullptr, false, SKIP_CONTACTS_UV, 1);
			make_positive_definite();
			break;
		case NORMAL:
			sysd.ConvertToMatrixForm(&SmallMat, &BigMat, nullptr, nullptr, nullptr, nullptr, false, true);
			break;
		}

		if (ADD_COMPLIANCE)
		{
			sysd.ConvertToMatrixForm(nullptr, nullptr, &E, nullptr, nullptr, nullptr, false, SKIP_CONTACTS_UV);
			E *= -1;
		}

		sysd.ConvertToMatrixForm(nullptr, nullptr, nullptr, &c, &b, nullptr, false, SKIP_CONTACTS_UV);
		c.MatrScale(-1); // adapt to InteriorPoint convention
		b.MatrScale(-1); // adapt to InteriorPoint convention

		//starting_point_Nocedal(n_old, m_old);
		//starting_point_STP1();
		starting_point_STP2();
		

	}

	void ChInteriorPoint::starting_point_STP1() // from [2]
	{
		double infeas_dual_ratio = 0.1; // TODO: dependant on n
		
		x.FillElem(1);
		y.FillElem(1);
		lam.FillElem(1);
		
		double duality_gap_calc = y.MatrDot(&y, &lam); // [2] pag. 132
		double duality_gap = m; // [2] pag. 132
		assert(duality_gap_calc == duality_gap);

		
		// norm of all residuals; [2] pag. 132
		fullupdate_residual();
		double res_norm = rp.MatrDot(&rp, &rp); 
		res_norm += rp.MatrDot(&rd, &rd);
		res_norm = sqrt(res_norm);

		if (res_norm / duality_gap > infeas_dual_ratio)
		{
			double coeff = res_norm / (duality_gap * infeas_dual_ratio);
			x.MatrScale(coeff);
			y.MatrScale(coeff);
			lam.MatrScale(coeff);

			fullupdate_residual();
		}

	}

	void ChInteriorPoint::starting_point_STP2()
	{
		double threshold = 1; // 'epsilon' in [2]
		x.FillElem(1);
		vectm.Resize(m, 1);
		multiplyA(x, vectm);
		vectm -= b;

		for (size_t cont = 0; cont < m; cont++)
		{
			if (vectm(cont, 0) > threshold)
				y(cont, 0) = vectm(cont, 0);
			else
				y(cont, 0) = threshold;
			
			lam(cont, 0) = 1 / y(cont, 0);
		}
		
		fullupdate_residual();

	}

	void ChInteriorPoint::starting_point_Nocedal(int n_old, int m_old)
	{

		/********** Initialize IP algorithm **********/
		// Initial guess
		if (n_old != n || solver_call == 0)
			x.FillElem(0); // TIP: every ChMatrix is initialized with zeros by default
		if (m_old != m || solver_call == 0)
			lam.FillElem(1); // each element of lam will be at the denominator; avoid zeros!

		// since A is generally changed between calls, also with warm_start,
		// all the residuals and feasibility check must be redone
		multiplyA(x, y);  // y = A*x
		y -= b;

		// Calculate the residual
		fullupdate_residual();

		// Feasible starting Point (pag.484-485)
		KKTsolve(); // to obtain Dx, Dy, Dlam called "affine"

		// x is accepted as it is
		y += Dy; // calculate y0
		lam += Dlam; // calculate lam0

		for (size_t row_sel = 0; row_sel < m; row_sel++)
			y(row_sel) = abs(y(row_sel)) < 1 ? 1 : abs(y(row_sel));

		for (size_t row_sel = 0; row_sel < m; row_sel++)
			lam(row_sel) = abs(lam(row_sel)) < 1 ? 1 : abs(lam(row_sel));

		// Update the residual considering the new values of 'y' and 'lam'
		fullupdate_residual();

	}

	// Iterating function
	// output: (x, y, lam) are computed
	// (rp, rd, mu) are updated based on most recent (x, y, lam)
	void ChInteriorPoint::iterate()
	{
		/********** Prediction phase **********/
		KKTsolve(); // to obtain Dx, Dy, Dlam called "affine" aka "prediction"

		// from 16.60 pag.482 from 14.32 pag.408 (remember that y>=0!)
		double alfa_pred_prim = find_Newton_step_length(y, Dy);
		double alfa_pred_dual = find_Newton_step_length(lam, Dlam);

		if (EQUAL_STEP_LENGTH)
		{
			double alfa_pred = std::min(alfa_pred_prim, alfa_pred_dual);
			alfa_pred_prim = alfa_pred;
			alfa_pred_dual = alfa_pred;
		} 

		  y_pred = Dy;     y_pred.MatrScale(alfa_pred_prim);   y_pred += y;
		lam_pred = Dlam; lam_pred.MatrScale(alfa_pred_dual); lam_pred += lam;

		double mu_pred = y_pred.MatrDot(&y_pred, &lam_pred) / m; // from 14.33 pag.408 //TODO: why MatrDot is a member?

		if (ONLY_PREDICT)
		{
			x_pred = Dx; x_pred.MatrScale(alfa_pred_prim); x_pred += x; x = x_pred;
			y = y_pred;
			lam = lam_pred;
			
			rp.MatrScale(1 - alfa_pred_prim);
			
			multiplyG(Dx, vectn); // vectn = G * Dx
			vectn.MatrScale(alfa_pred_prim - alfa_pred_dual); // vectn = (alfa_pred_prim - alfa_pred_dual) * (G * Dx)
			rd.MatrScale(1 - alfa_pred_dual);
			rd += vectn;

			mu = mu_pred;
		}
		

		/********** Correction phase **********/
		double sigma = (mu_pred / mu)*(mu_pred / mu)*(mu_pred / mu); // from 14.34 pag.408

		KKTsolve(sigma);

		double eta = ADAPTIVE_ETA ? exp(-mu*m)*0.1 + 0.9 : 0.95; // exponential descent of eta

		double alfa_corr_prim = find_Newton_step_length(y, Dy, eta);
		double alfa_corr_dual = find_Newton_step_length(lam, Dlam, eta);

		if (EQUAL_STEP_LENGTH)
		{
			double alfa_corr = std::min(alfa_corr_prim, alfa_corr_dual);
			alfa_corr_prim = alfa_corr;
			alfa_corr_dual = alfa_corr;
		}

		  x_corr = Dx;		  x_corr.MatrScale(alfa_corr_prim);		  x_corr += x;		  x = x_corr;
		  y_corr = Dy;		  y_corr.MatrScale(alfa_corr_prim);		  y_corr += y;		  y = y_corr;
		lam_corr = Dlam;	lam_corr.MatrScale(alfa_corr_dual);		lam_corr += lam;	lam = lam_corr;


		/********** Variable update **********/
		rp.MatrScale(1 - alfa_corr_prim);
		rd.MatrScale(1 - alfa_corr_dual);
		mu = y.MatrDot(&y, &lam) / m; // from 14.6 pag.395
		
		if (!EQUAL_STEP_LENGTH)
		{
			multiplyG(Dx, vectn); // vectn = G*Dx
			vectn.MatrScale(alfa_corr_prim - alfa_corr_dual); // vectn = (alfa_pred_prim - alfa_pred_dual) * (G * Dx)
			rd += vectn;
		}
	}

	void ChInteriorPoint::KKTsolve(double sigma)
	{
		// TODO: only for test purpose
		ChMatrixDynamic<double> res(BigMat.GetRows(), 1);
		double res_norm;

		switch (KKT_solve_method)
		{
		case STANDARD:
			// update lambda and y diagonal submatrices
			for (size_t diag_sel = 0; diag_sel < m; diag_sel++)
			{
				BigMat.SetElement(n + m + diag_sel, n + diag_sel, lam.GetElement(diag_sel, 0)); // write lambda diagonal submatrix
				BigMat.SetElement(n + m + diag_sel, n + m + diag_sel, y.GetElement(diag_sel, 0)); // write y diagonal submatrix
				BigMat.SetElement(n + diag_sel, n + diag_sel, -1); // write -identy_matrix diagonal submatrix
			}

			if (sigma != 0) // rpd_corr
			{
				// I'm supposing that in 'rpd', since the previous call should have been without perturbation,
				// there is already y°lam
				vectm = Dlam; // I could use Dlam directly, but it is not really clear
				vectm.MatrScale(Dy);
				vectm.MatrAdd(-sigma*mu);
				rpd += vectm;
			}
			else // rpd_pred as (16.57 pag.481 suggests)
			{
				rpd = y;
				rpd.MatrScale(lam);
			}

			// Fill 'rhs' with [-rd;-rp;-rpd]
			for (size_t row_sel = 0; row_sel < n; row_sel++)
				rhs.SetElement(row_sel, 0, -rd.GetElement(row_sel, 0));

			for (size_t row_sel = 0; row_sel < m; row_sel++)
			{
				rhs.SetElement(row_sel + n, 0, -rp.GetElement(row_sel, 0));
				rhs.SetElement(row_sel + n + m, 0, -rpd.GetElement(row_sel, 0));
			}


			// Solve the KKT system
			BigMat.Compress();
			mkl_engine.SetProblem(BigMat, rhs, sol);
			mkl_engine.PardisoCall(13, 0);

			// Extract 'Dx', 'Dy' and 'Dlam' from 'sol'
			for (size_t row_sel = 0; row_sel < n; row_sel++)
				Dx.SetElement(row_sel, 0, sol.GetElement(row_sel, 0));

			for (size_t row_sel = 0; row_sel < m; row_sel++)
			{
				Dy.SetElement(row_sel, 0, sol.GetElement(row_sel + n, 0));
				Dlam.SetElement(row_sel, 0, sol.GetElement(row_sel + n + m, 0));
			}

			break;
		case AUGMENTED:
			// update y/lambda diagonal submatrix
			if (ADD_COMPLIANCE)
				for (size_t diag_sel = 0; diag_sel < m; diag_sel++)
				{
					BigMat.SetElement(n + diag_sel, n + diag_sel, y.GetElement(diag_sel, 0) / lam.GetElement(diag_sel, 0) + E.GetElement(diag_sel, diag_sel));
				}
			else
				for (size_t diag_sel = 0; diag_sel < m; diag_sel++)
				{
					BigMat.SetElement(n + diag_sel, n + diag_sel, y.GetElement(diag_sel, 0) / lam.GetElement(diag_sel, 0));
				}


			// Fill 'rhs' with [-rd;-rp-y-sigma*mu/lam]
			for (size_t row_sel = 0; row_sel < n; row_sel++)
				rhs.SetElement(row_sel, 0, -rd.GetElement(row_sel, 0));

			if (sigma != 0)
			{
				for (size_t row_sel = 0; row_sel < m; row_sel++)
					rhs.SetElement(row_sel + n, 0, -rp(row_sel, 0) - y(row_sel, 0) + sigma*mu / lam(row_sel, 0));
			}
			else
			{
				for (size_t row_sel = 0; row_sel < m; row_sel++)
					rhs.SetElement(row_sel + n, 0, -rp(row_sel, 0) - y(row_sel, 0));
			}


			// Solve the KKT system
			BigMat.Compress();
			mkl_engine.SetProblem(BigMat, rhs, sol);
			mkl_engine.PardisoCall(13, 0);

			if (verbose)
			{
				// TODO: used for testing; delete
				res.Reset(BigMat.GetRows(), 1);
				mkl_engine.GetResidual(res);
				res_norm = mkl_engine.GetResidualNorm(res);
				std::cout << "Residual norm of MKL call: " << res_norm << std::endl;
			}
			

			// Extract 'Dx' and 'Dlam' from 'sol'
			for (size_t row_sel = 0; row_sel < n; row_sel++)
				Dx.SetElement(row_sel, 0, sol.GetElement(row_sel, 0));
			for (size_t row_sel = 0; row_sel < m; row_sel++)
				Dlam.SetElement(row_sel, 0, sol.GetElement(row_sel + n, 0));

			// Calc 'Dy' (it is also possible to evaluate Dy as Dy=(-lam°y+sigma*mu*e-y°Dlam)./lam )
			multiplyA(Dx, Dy);  // Dy = A*Dx
			Dy += rp;
			if (ADD_COMPLIANCE)
			{
				E.MatMultiply(Dlam, vectm);
				Dy += vectm;
			}
				

			break;
		case NORMAL:
			for (size_t row_sel = 0; row_sel < n; row_sel++)
			{
				for (size_t col_sel = 0; col_sel < n; col_sel++)
				{
					int temp = 0;
					for (size_t el_sel = 0; el_sel < m; el_sel++)
					{
						temp += lam(el_sel, 0) / y(el_sel, 0) * SmallMat.GetElement(el_sel, row_sel) * SmallMat.GetElement(el_sel, col_sel);
						if (temp != 0)
							BigMat.SetElement(row_sel, col_sel, temp, false);
					}
				}
			}

			break;
		}
	}

	// this function obtain the maximum length, along the direction defined by Dvect, so that vect has no negative components;
	// it will be applied to the two vectors that are forced to be non-negative i.e. 'lam' and 'y'
	double ChInteriorPoint::find_Newton_step_length(ChMatrix<double>& vect, ChMatrix<double>& Dvect, double eta ) const
	{
		double alpha = 1;
		for (size_t row_sel = 0; row_sel < vect.GetRows(); row_sel++)
		{
			if (Dvect(row_sel,0)<0)
			{
				double alfa_temp = -eta * vect(row_sel,0) / Dvect(row_sel,0);
				if (alfa_temp < alpha)
					alpha = alfa_temp;
			}
		}

		return (alpha>0) ? alpha : 0;
	}

	double ChInteriorPoint::evaluate_objective_function()
	{
		multiplyG(x, vectn);
		double obj_value = vectn.MatrDot(&x, &vectn);
		obj_value += c.MatrDot(&x, &c);

		return obj_value;
	}

	void ChInteriorPoint::reset_dimensions()
	{
		// variables: (x,y) primal variables; lam dual
		x.Resize(n, 1);
		y.Resize(m, 1);
		lam.Resize(m, 1);
        
		x_pred.Resize(n, 1);
		y_pred.Resize(m, 1);
		lam_pred.Resize(m, 1);
        
		x_corr.Resize(n, 1);
		y_corr.Resize(m, 1);
		lam_corr.Resize(m, 1);
        
		Dx.Resize(n, 1);
		Dy.Resize(m, 1);
		Dlam.Resize(m, 1);

		// vectors
		b.Resize(m, 1);
		c.Resize(n, 1);
        
		// residuals
		rp.Resize(m, 1);
		rd.Resize(n, 1);
		rpd.Resize(m, 1);

		// temporary vectors
		vectn.Resize(n, 1);
		SKIP_CONTACTS_UV ? sol_chrono.Resize(n + 3 * m, 1) : sol_chrono.Resize(n + m, 1);

		// BigMat and sol
		switch (KKT_solve_method)
		{
		case STANDARD:
			BigMat.Reset(2 * m + n, 2 * m + n, static_cast<int>(n*n*SPM_DEF_FULLNESS));
			sol.Resize(2 * m + n, 1);
			rhs.Resize(2 * m + n, 1);
			break;
		case AUGMENTED:
			BigMat.Reset(n + m, n + m, static_cast<int>(n*n*SPM_DEF_FULLNESS));
			sol.Resize(n + m, 1);
			rhs.Resize(n + m, 1);
			break;
		case NORMAL:
			std::cout << std::endl << "Perturbed KKT system cannot be stored with 'NORMAL' method yet.";
			break;
		}
	}


	void ChInteriorPoint::DumpProblem(std::string suffix)
	{
		ExportToFile(y, "dump/y" + suffix + ".txt");
		ExportToFile(x, "dump/x" + suffix + ".txt");
		ExportToFile(lam, "dump/lam" + suffix + ".txt");

		ExportToFile(b, "dump/b" + suffix + ".txt");
		ExportToFile(c, "dump/c" + suffix + ".txt");

		BigMat.Compress();
		BigMat.ExportToDatFile("dump/", 8);
	}

	void ChInteriorPoint::DumpIPStatus(std::string suffix)
	{
		ExportToFile(y, "dump/y" + suffix + ".txt");
		ExportToFile(x, "dump/x" + suffix + ".txt");
		ExportToFile(lam, "dump/lam" + suffix + ".txt");

		ExportToFile(Dx, "dump/Dx" + suffix + ".txt");
		ExportToFile(Dy, "dump/Dy" + suffix + ".txt");
		ExportToFile(Dlam, "dump/Dlam" + suffix + ".txt");

		ExportToFile(rhs, "dump/rhs" + suffix + ".txt");
		ExportToFile(sol, "dump/sol" + suffix + ".txt");
	}

	void ChInteriorPoint::make_positive_definite()
	{
		double* values = BigMat.GetValuesAddress();
		int* colIndex = BigMat.GetColIndexAddress();
		int* rowIndex = BigMat.GetRowIndexAddress();

		int offset_AT_col = n;
		if (KKT_solve_method == STANDARD)
			offset_AT_col = n + m;

		for (size_t row_sel = 0; row_sel < n; row_sel++)
		{
			for (size_t col_sel = rowIndex[row_sel]; col_sel < rowIndex[row_sel+1]; col_sel++)
			{
				if (colIndex[col_sel] >= offset_AT_col && colIndex[col_sel] < offset_AT_col + m)
					values[col_sel] *= -1;
			}
		}

	}

	// Perform moltiplication of A with vect_in: vect_out = A*vect_in
	void ChInteriorPoint::multiplyA(ChMatrix<double>& vect_in, ChMatrix<double>& vect_out) const
	{
		switch (KKT_solve_method)
		{
		case STANDARD:
			BigMat.MatMultiplyClipped(vect_in, vect_out, n, n + m - 1, 0, n - 1, 0, 0);
			break;
		case AUGMENTED:
			BigMat.MatMultiplyClipped(vect_in, vect_out, n, n + m - 1, 0, n - 1, 0, 0);
			break;
		case NORMAL:
			std::cout << std::endl << "A multiplication is not implemented in 'NORMAL' method yet.";
			break;
		}
	}

	// Perform moltiplication of -AT with vect_in: vect_out = -AT*vect_in
	void ChInteriorPoint::multiplyNegAT(ChMatrix<double>& vect_in, ChMatrix<double>& vect_out) const
	{
		switch (KKT_solve_method)
		{
		case STANDARD:
			BigMat.MatMultiplyClipped(vect_in, vect_out, 0, n - 1, n + m, n + 2*m - 1, 0, 0);
			break;
		case AUGMENTED:
			BigMat.MatMultiplyClipped(vect_in, vect_out, 0, n - 1, n, n + m - 1, 0, 0);
			break;
		case NORMAL:
			std::cout << std::endl << "AT multiplication is not implemented in 'NORMAL' method yet.";
			break;
		}
	}


	void ChInteriorPoint::multiplyG(ChMatrix<double>& vect_in, ChMatrix<double>& vect_out) const
	{
		switch (KKT_solve_method)
		{
		case STANDARD:
			BigMat.MatMultiplyClipped(vect_in, vect_out, 0, n - 1, 0, n - 1, 0, 0);
			break;
		case AUGMENTED:
			BigMat.MatMultiplyClipped(vect_in, vect_out, 0, n - 1, 0, n - 1, 0, 0);
			break;
		case NORMAL:
			std::cout << std::endl << "G multiplication is not implemented in 'NORMAL' method yet.";
			break;
		}
	}

	void ChInteriorPoint::normalize_Arows()
	{
		
		int offset_AT_cols = 0;
		switch (KKT_solve_method)
		{
		case STANDARD:
			offset_AT_cols = n;
			break;
		case AUGMENTED:
			offset_AT_cols = n + m;
			break;
		case NORMAL:
			std::cout << std::endl << "G multiplication is not implemented in 'NORMAL' method yet.";
			break;
		}

		double row_norm;

		double* a_mat = BigMat.GetValuesAddress();
		int* ia_mat = BigMat.GetRowIndexAddress();
		int* ja_mat = BigMat.GetColIndexAddress();

		for (size_t row_sel = n; row_sel < n + m; row_sel++)
		{
			// Calc row norm from A in BigMat
			row_norm = 0;
			for (size_t col_sel = ia_mat[row_sel]; ja_mat[col_sel] < n && col_sel < ia_mat[row_sel + 1]; col_sel++)
			{
				row_norm += a_mat[col_sel] * a_mat[col_sel];
			}
			row_norm = sqrt(row_norm);

			// Normalize A and rewrite -AT
			for (size_t col_sel = ia_mat[row_sel]; ja_mat[col_sel] < n && col_sel < ia_mat[row_sel + 1]; col_sel++)
			{
				a_mat[col_sel] = a_mat[col_sel] / row_norm;
				BigMat.SetElement( ja_mat[col_sel], row_sel, -a_mat[col_sel] );
			}
		}
	}

	bool ChInteriorPoint::check_exit_conditions(bool only_mu)
	{
		if (false)
		{
			if (mu < mu_tolerance)
			{
				if (only_mu)
					return true;

				VerifyKKTconditions();
				if (
					IPresiduals.rp_nnorm < IPtolerances.rp_nnorm &&
					IPresiduals.rd_nnorm < IPtolerances.rd_nnorm
					)
					return true;
			}
		}

		if (true)
		{
			double relative_duality_gap = y.MatrDot(&y, &lam); // TODO: you can recycle 'mu'
			
			VerifyKKTconditions();
			if (relative_duality_gap < mu_tolerance &&
				IPresiduals.rp_nnorm < IPtolerances.rp_nnorm &&
				IPresiduals.rd_nnorm < IPtolerances.rd_nnorm
				)
				return true;

		}
		

		return false;
	}

	bool ChInteriorPoint::check_feasibility(double tolerance)
	{
		VerifyKKTconditions();
		if (IPresiduals.rp_nnorm < tolerance &&
			IPresiduals.rd_nnorm < tolerance )
			return true;

		return false;
	}

	//void ChInteriorPoint::save_lam_mean()
	//{
	//	lam_mean = 0;
	//	for (size_t row_sel = 0; row_sel < m; row_sel++)
	//	{
	//		lam_mean += lam.GetElement(row_sel, 0);
	//	}
	//	lam_mean = lam_mean / m;
	//}

	void ChInteriorPoint::fullupdate_residual()
	{
			// Residual initialization (16.59 pag.482)
			// rp initialization 
			multiplyA(x, rp);  // rp = A*x
			rp -= y + b;
			if (ADD_COMPLIANCE)
			{
				E.MatMultiply(Dlam, vectm);
				rp += vectm;
			}
				

			// rd initialization
			multiplyG(x, rd); // rd = G*x
			rd += c; // rd = G*x + c
			multiplyNegAT(lam, vectn); // vectn = (-A^T)*lam
			rd += vectn; // rd = (G*x + c) + (-A^T*lam)

			mu = y.MatrDot(&y, &lam) / m;
	}

	void ChInteriorPoint::generate_solution()
	{

		for (size_t row_sel = 0; row_sel < n; row_sel++)
			sol_chrono(row_sel, 0) = x(row_sel,0);

		if (SKIP_CONTACTS_UV)
		{
			for (size_t row_sel = 0; row_sel < m; row_sel++)
			{
				sol_chrono(n + row_sel*3, 0) = -lam(row_sel,0); // there will be an inversion inside FromVectorToUnknowns()
				sol_chrono(n + row_sel*3 + 1, 0) = 0;
				sol_chrono(n + row_sel*3 + 2, 0) = 0;
			}
		}
		else
		{
			for (size_t row_sel = 0; row_sel < m; row_sel++)
				sol_chrono(row_sel + n, 0) = -lam(row_sel); // there will be an inversion inside FromVectorToUnknowns()
		}
	}

	ChInteriorPoint::ChInteriorPoint():
		m(0),
		n(0),
		solver_call(0),
		iteration_count_max(50),
		EQUAL_STEP_LENGTH(false),
		ADAPTIVE_ETA(false),
		ONLY_PREDICT(false),
		IPtolerances(1e-12),
		mu_tolerance(1e-12),
		KKT_solve_method(AUGMENTED),
		mu(0)
	{
		mkl_engine.SetProblem(BigMat, rhs, sol);
	}


}
