#include "ChInteriorPoint.h"
#include <algorithm>

#define TEST_MATRIX false
#define SKIP_CONTACTS_UV true

namespace chrono
{
	template <class matrix>
	void ExportToFile(matrix mat, std::string filepath, int precision = 12)
	{
		std::ofstream ofile;
		ofile.open(filepath);
		ofile << std::scientific << std::setprecision(precision);

		for (int row_sel = 0; row_sel < mat.GetRows(); row_sel++)
		{
			for (int col_sel = 0; col_sel < mat.GetColumns(); col_sel++)
			{
				ofile << mat.GetElement(row_sel, col_sel) << " , ";
			}

			ofile << std::endl;
		}

		ofile.close();
	}

	template<class ChMatrixIN>
	void PrintMatrix(ChMatrixIN& matrice){
		for (int i = 0; i < matrice.GetRows(); i++){
			for (int j = 0; j < matrice.GetColumns(); j++){
				printf("%.1f ", matrice.GetElement(i, j));
			}
			printf("\n");
		}
	}

	// | M  -Cq'|*|q|- | f|= |0|
    // | Cq  -E | |l|  |-b|  |c| 

	void ChInteriorPoint::Initialize(ChLcpSystemDescriptor& sysd)
	{
		verbose = true;

		// Initial resizing
		//if (solver_call == 0)
		//{
			// not mandatory, but it speeds up the first build of the matrix, guessing its sparsity; needs to stay BEFORE ConvertToMatrixForm()
			n = sysd.CountActiveVariables();
			m = sysd.CountActiveConstraints(false, SKIP_CONTACTS_UV);
	
			reset_dimensions();
		//}

		// load system matrix in 'BigMat', 'rhs', 'b' and 'c'
		switch (qp_solve_type)
		{
		case STANDARD:
			std::cout << std::endl << "Perturbed KKT system cannot be loaded with 'STANDARD' method yet.";
			break;
		case AUGMENTED:
			sysd.ConvertToMatrixForm(&BigMat, nullptr, false, SKIP_CONTACTS_UV);
			make_positive_definite(&BigMat);
			break;
		case NORMAL:
			std::cout << std::endl << "Perturbed KKT system cannot be loaded with 'NORMAL' method yet.";
			break;
		}

		sysd.ConvertToMatrixForm(nullptr, nullptr, nullptr, &c, &b, nullptr, false, SKIP_CONTACTS_UV);
		c.MatrScale(-1); // adapt to InteriorPoint convention
		b.MatrScale(-1); // adapt to InteriorPoint convention



		if (TEST_MATRIX)
			TestAugmentedMatrix();

		// Initial guess
		for (int n_temp = 0; n_temp < n; n_temp++)
			x(n_temp, 0) = 0;

		for (int m_temp = 0; m_temp < m; m_temp++)
			lam(m_temp, 0) = 0.1;

		BigMat.MatMultiplyClipped(x, y, n, n + m - 1, 0, n - 1, 0, 0);  // y = A*x
		y -= b;
		
		// Calculate the residual
		fullupdate_residual();
		
		// Feasible starting Point (pag.484-485)
		KKTsolve(0); // to obtain Dx, Dy, Dlam called "affine"

		y += Dy; // calculate y0
		lam += Dlam; // calculate lam0
		// x0 is equal to x

		for (int row_sel = 0; row_sel < m; row_sel++)
			y(row_sel) = abs(y(row_sel)) < 1 ? 1 : abs(y(row_sel));

		for (int row_sel = 0; row_sel < m; row_sel++)
			lam(row_sel) = abs(lam(row_sel)) < 1 ? 1 : abs(lam(row_sel));


		// Update the residual considering the new values of 'y' and 'lam'
		fullupdate_residual();
		
	}

	double ChInteriorPoint::Iterate()
	{
		// Prediction phase
		KKTsolve(0); // to obtain Dx, Dy, Dlam called "affine" aka "prediction"
		

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
			x_pred = Dx;		  x_pred.MatrScale(alfa_pred_prim);		x_pred += x;	  x = x_pred;
			y = y_pred;
			lam = lam_pred;
			
			rp.MatrScale(1 - alfa_pred_prim);
			
			BigMat.MatMultiplyClipped(Dx, vectn, 0, n-1, 0, n-1, 0, 0); // vectn = G * Dx
			vectn.MatrScale(alfa_pred_prim - alfa_pred_dual); // vectn = (alfa_pred_prim - alfa_pred_dual) * (G * Dx)
			rd.MatrScale(1 - alfa_pred_dual);
			rd += vectn;

			iterate_count++;
			return mu_pred;
		}
		

		// Corrector phase
		double mu = y.MatrDot(&y, &lam)/m; // from 14.6 pag.395
		double sigma = (mu_pred / mu)*(mu_pred / mu)*(mu_pred / mu); // from 14.34 pag.408

		KKTsolve(sigma*mu);

		double eta = (ADAPTIVE_ETA) ? exp(-mu*m)*0.1 + 0.9 : 0.95; // exponential descent of eta

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


		// Update for new cycle
		rp.MatrScale(1 - alfa_corr_prim);
		rd.MatrScale(1 - alfa_corr_dual);
		
		if (!EQUAL_STEP_LENGTH)
		{
			BigMat.MatMultiplyClipped(Dx, vectn, 0, n - 1, 0, n - 1, 0, 0); // vectn = G*Dx
			vectn.MatrScale(alfa_corr_prim - alfa_corr_dual); // vectn = (alfa_pred_prim - alfa_pred_dual) * (G * Dx)
			rd += vectn;
		}
		iterate_count++;
		return mu;
	}

	void ChInteriorPoint::KKTsolve(const double perturbation)
	{
		switch (qp_solve_type)
		{
		case STANDARD:
			std::cout << std::endl << "Perturbed KKT system has not been checked to solve with 'STANDARD' method yet.";
			// update lambda and y diagonal submatrices
			for (int diag_sel = 0; diag_sel < m; diag_sel++)
			{
				BigMat.SetElement(n + m + diag_sel, n + diag_sel, lam.GetElement(diag_sel, 0)); // write lambda diagonal submatrix
				BigMat.SetElement(n + m + diag_sel, n + m + diag_sel, y.GetElement(diag_sel, 0)); // write y diagonal submatrix
			}

			if (perturbation) // rpd_corr
			{
				// I'm supposing that in 'rpd', since the previous call should have been without perturbation,
				// there is already y°lam
				vectm = Dlam; // I could use Dlam directly, but it is not really clear
				vectm.MatrScale(Dy);
				vectm.MatrAdd(-perturbation);
				rpd += vectm;
			}
			else // rpd_pred as (16.57 pag.481 suggests)
			{
				rpd = y;
				rpd.MatrScale(lam); 
			}

			// Fill 'rhs' with [-rd;-rp;-rpd]
			for (int row_sel = 0; row_sel < n; row_sel++)
				rhs.SetElement(row_sel, 0, -rd.GetElement(row_sel, 0));

			for (int row_sel = 0; row_sel < m; row_sel++)
			{
				rhs.SetElement(row_sel + n,     0, -rp.GetElement(row_sel, 0));
				rhs.SetElement(row_sel + n + m, 0, -rpd.GetElement(row_sel, 0));
			}
			
			// Solve the KKT system
			mkl_engine.SetProblem(BigMat, rhs, sol);
			mkl_engine.PardisoCall(13, 0);

			// Extract 'Dx', 'Dy' and 'Dlam' from 'sol'
			for (int row_sel = 0; row_sel < n; row_sel++)
				Dx.SetElement(row_sel, 0, sol.GetElement(row_sel, 0));

			for (int row_sel = 0; row_sel < m; row_sel++)
			{
				Dy.SetElement(row_sel, 0, sol.GetElement(row_sel + n, 0));
				Dlam.SetElement(row_sel, 0, sol.GetElement(row_sel + n + m, 0));
			}		
			
			break;
		case AUGMENTED:
			// update lambda°y diagonal submatrix
			for (int diag_sel = 0; diag_sel < m; diag_sel++)
			{
				BigMat.SetElement(n + diag_sel, n + diag_sel, y.GetElement(diag_sel, 0) / lam.GetElement(diag_sel, 0) );
			}

			

			// Fill 'rhs' with [-rd;-rp-y-sigma*mu/lam]
			for (int row_sel = 0; row_sel < n; row_sel++)
				rhs.SetElement(row_sel, 0, -rd.GetElement(row_sel, 0));

			if (perturbation != 0)
			{
				for (int row_sel = 0; row_sel < m; row_sel++)
					rhs.SetElement(row_sel + n, 0, -rp(row_sel, 0) - y(row_sel, 0) + perturbation / lam(row_sel, 0));
			}
			else
			{
				for (int row_sel = 0; row_sel < m; row_sel++)
					rhs.SetElement(row_sel + n, 0, -rp(row_sel, 0) - y(row_sel, 0));
			}
			

			// Solve the KKT system
			BigMat.Compress();
			mkl_engine.SetProblem(BigMat, rhs, sol);
			mkl_engine.PardisoCall(13, 0);

			// Extract 'Dx' and 'Dlam' from 'sol'
			for (int row_sel = 0; row_sel < n; row_sel++)
				Dx.SetElement(row_sel, 0, sol.GetElement(row_sel, 0));
			for (int row_sel = 0; row_sel < m; row_sel++)
				Dlam.SetElement(row_sel, 0, sol.GetElement(row_sel + n, 0));

			// Calc 'Dy'
			BigMat.MatMultiplyClipped(Dx, Dy, n, n + m - 1, 0, n - 1, 0, 0);  // Dy = A*Dx
			Dy += rp;

			break;
		case NORMAL:
			std::cout << std::endl << "Perturbed KKT system cannot be solved with 'NORMAL' method yet.";
			break;
		}
	}

	double ChInteriorPoint::find_Newton_step_length(ChMatrix<double>& vect, ChMatrix<double>& Dvect, double eta )
	{
		double alpha = 1;
		for (int row_sel = 0; row_sel < vect.GetRows(); row_sel++)
		{
			//if (Dvect(row_sel,0)<0 && vect(row_sel,0)) // actually vect should be always >0 but it isn't..
			if (Dvect(row_sel,0)<0)
			{
				double alfa_temp = -eta * vect(row_sel,0) / Dvect(row_sel,0);
				if (alfa_temp < alpha)
					alpha = alfa_temp;
			}
		}

		return (alpha>0) ? alpha : 0;
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
		switch (qp_solve_type)
		{
		case STANDARD:
			BigMat.Reset(2 * m + n, 2 * m + n, static_cast<int>(n*n*SPM_DEF_FULLNESS));
			sol.Resize(2 * m + n, 1);
			rhs.Resize(2 * n + m, 1);
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


	void ChInteriorPoint::dump_all(std::string suffix)
	{
		ExportToFile(y, "dump/y" + suffix + ".txt");
		ExportToFile(x, "dump/x" + suffix + ".txt");
		ExportToFile(lam, "dump/lam" + suffix + ".txt");
		ExportToFile(Dx, "dump/Dx" + suffix + ".txt");
		ExportToFile(Dy, "dump/Dy" + suffix + ".txt");
		ExportToFile(Dlam, "dump/Dlam" + suffix + ".txt");
		ExportToFile(rp, "dump/rp" + suffix + ".txt");
		ExportToFile(rd, "dump/rd" + suffix + ".txt");
		ExportToFile(sol, "dump/sol" + suffix + ".txt");
		ExportToFile(rhs, "dump/rhs" + suffix + ".txt");
		ExportToFile(b, "dump/b" + suffix + ".txt");
		ExportToFile(c, "dump/c" + suffix + ".txt");

		ExportToFile(vectm, "dump/vectm" + suffix + ".txt");
		BigMat.ExportToDatFile("dump/", 8);
	}

	void ChInteriorPoint::make_positive_definite(ChCSR3Matrix* mat)
	{
		for (int col_sel = n; col_sel < n + m; col_sel++)
		{
			for (int row_sel = 0; row_sel < n; row_sel++)
			{
				mat->Element(row_sel, col_sel) *= -1;
			}
		}
	}

	void ChInteriorPoint::fullupdate_residual()
	{
		switch (qp_solve_type)
		{
		case STANDARD:
			std::cout << std::endl << "rp and rd cannot be updated in 'STANDARD' method yet.";
			break;
		case AUGMENTED:
			// Residual initialization (16.59 pag.482)
			// rp initialization 
			BigMat.MatMultiplyClipped(x, rp, n, n + m - 1, 0, n - 1, 0, 0);  // rp = A*x
			rp -= y + b;

			// rd initialization
			BigMat.MatMultiplyClipped(x, rd, 0, n - 1, 0, n - 1, 0, 0); // rd = G*x
			rd += c; // rd = G*x + c
			BigMat.MatMultiplyClipped(lam, vectn, 0, n - 1, n, n + m - 1, 0, 0); // vectn = (-A^T)*lam
			rd += vectn; // rd = (G*x + c) + (-A^T*lam)
			break;
		case NORMAL:
			std::cout << std::endl << "rp and rd cannot be updated in 'NORMAL' method yet.";
			break;
		}
	}


	void ChInteriorPoint::generate_solution()
	{

		for (int row_sel = 0; row_sel < n; row_sel++)
			sol_chrono(row_sel, 0) = x(row_sel,0);

		if (SKIP_CONTACTS_UV)
		{
			for (int row_sel = 0; row_sel < m; row_sel++)
			{
				sol_chrono(row_sel + n, 0) = -lam(row_sel,0); // there will be an inversion inside FromVectorToUnknowns()
				sol_chrono(row_sel + n + 1, 0) = 0;
				sol_chrono(row_sel + n + 2, 0) = 0;
			}
		}
		else
		{
			for (int row_sel = 0; row_sel < m; row_sel++)
				sol_chrono(row_sel + n, 0) = -lam(row_sel); // there will be an inversion inside FromVectorToUnknowns()
		}
	}

	ChInteriorPoint::ChInteriorPoint(QP_SOLUTION_TECHNIQUE qp_solve_type_selection):
		qp_solve_type(qp_solve_type_selection),
		solver_call(0),
		iterate_count(0),
		n(0),
		m(0),
		iteration_count_max(10),
		EQUAL_STEP_LENGTH(false),
		ADAPTIVE_ETA(true),
		ONLY_PREDICT(false)
	{

	}

	double ChInteriorPoint::Solve(ChLcpSystemDescriptor& sysd)
	{
		Initialize(sysd);

		for (int iteration_count = 0; iteration_count < iteration_count_max; iteration_count++)
		{
			if (abs(Iterate()) < 1e-12)
				break;
		}
		solver_call++;
		generate_solution();

		ExportToFile(sol_chrono, "dump/sol_ip.txt");

		sysd.FromVectorToUnknowns(sol_chrono);

		return 0;
	}


	void ChInteriorPoint::TestAugmentedMatrix()
	{

		n = 3;
		m = 3;

		reset_dimensions();

		// M matrix
		BigMat(0, 0) = 1;
		BigMat(1, 1) = 1;
		BigMat(2, 2) = 1;

		// A matrix
		BigMat(3, 0) = 1;
		BigMat(3, 1) = -1;
		BigMat(4, 1) = 1;
		BigMat(4, 2) = -1;
		BigMat(5, 2) = 1;

		// -A'
		BigMat(0, 3) = -1;
		BigMat(1, 3) = 1;
		BigMat(1, 4) = -1;
		BigMat(2, 4) = 1;
		BigMat(2, 5) = -1;


		
		// b
		b.Resize(m, 1);
		b(0,0) = 0;
		b(1,0) = 0;
		b(2,0) = 0;

		// c
		c.Resize(n, 1);
		c(0, 0) = -10;
		c(1, 0) = +10;
		c(2, 0) = 0;

		
	}

}
