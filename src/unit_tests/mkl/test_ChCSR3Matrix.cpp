//
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2010-2011 Alessandro Tasora
// Copyright (c) 2013 Project Chrono
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file at the top level of the distribution
// and at http://projectchrono.org/license-chrono.txt.
//

///////////////////////////////////////////////////
//
//   Demo code about
//
//     - how to call Mkl from Chrono::Engine
//
//	 CHRONO
//   ------
//   Multibody dinamics engine
//
// ------------------------------------------------
//             www.deltaknowledge.com
// ------------------------------------------------
///////////////////////////////////////////////////

#include "chrono_mkl/ChCSR3Matrix.h"
#include "chrono/core/ChMatrixDynamic.h"

#define PRINT_RESULTS false

using namespace chrono;

template <class matrix_t>
void PrintMatrix(matrix_t& mat)
{
	for (int ii = 0; ii < mat.GetRows(); ii++){
		for (int jj = 0; jj < mat.GetColumns(); jj++)
			std::cout << mat.GetElement(ii, jj) << "\t";
		std::cout << std::endl;
	}
}

bool are_arrays_equal(ChCSR3Matrix& mat1, ChCSR3Matrix& mat2, bool only_rowIndex = false)
{
	int rows = mat1.GetRows();
	{
		int rows_temp = mat2.GetRows();
		if (rows_temp != rows) return false;
	}

	for (int cont = 0; cont <= rows; cont++)
	{
		if (mat1.GetRowIndexAddress()[cont] != mat2.GetRowIndexAddress()[cont]) return false;
	}

	if (!only_rowIndex)
	for (int cont = 0; cont < mat1.GetRowIndexAddress()[rows]; cont++)
	{
		if (mat1.GetValuesAddress()[cont] != mat2.GetValuesAddress()[cont]) return false;
		if (mat1.GetColIndexAddress()[cont] != mat2.GetColIndexAddress()[cont]) return false;
	}

	return true;
}

template <class matrixIN, class matrixOUT>
bool is_equal(matrixOUT& mat_out, matrixIN& mat_in)
{
	if (mat_in.GetRows() != mat_out.GetRows() || mat_in.GetColumns() != mat_out.GetColumns()) return false;

	for (int m_sel = 0; m_sel < mat_in.GetRows(); m_sel++)
		for (int n_sel = 0; n_sel < mat_in.GetColumns(); n_sel++)
			if (mat_in.GetElement(m_sel, n_sel) != mat_out.GetElement(m_sel, n_sel)) return false;

	return true;
}

template <class matrixIN, class matrixOUT>
void CopyMatrix(matrixOUT& mat_out, matrixIN& mat_in)
{

	//assert(mat_out.GetRows() == mat_in.GetRows());
	//assert(mat_out.GetColumns() == mat_in.GetColumns());
	for (int m_sel = 0; m_sel < mat_in.GetRows(); m_sel++)
		for (int n_sel = 0; n_sel < mat_in.GetColumns(); n_sel++)
			if (mat_in.GetElement(m_sel, n_sel) != 0) mat_out.SetElement(m_sel, n_sel, mat_in.GetElement(m_sel, n_sel));
}


bool is_Initialize_with_nonzeros_vector_broken()
{

	return false;
}

bool is_Reset_function_broken()
{
	{
		ChCSR3Matrix mat(3, 3, 3);
		mat.SetElement(0, 0, 1.1);
		mat.SetElement(1, 1, 2.2);
		mat.SetElement(2, 2, 3.3);

		mat.SetRowIndexLock(true);
		mat.SetColIndexLock(true);

		mat.Compress();

		// both sparsity block must be kept after this command below
		mat.SetElement(2, 2, 4.4);
		if (mat.IsRowIndexLockBroken() || mat.IsColIndexLockBroken()) return true;

		// both sparsity block must be broken after this command below
		mat.SetElement(2, 1, 4.4);
		if (!mat.IsRowIndexLockBroken() || !mat.IsColIndexLockBroken()) return true;

		// sparsity locks must be restored after this command below
		mat.Reset(3, 3, 3);
		if (mat.IsRowIndexLockBroken() || mat.IsColIndexLockBroken()) return true;
	}

	{
		ChCSR3Matrix mat(3, 3, 3);
		mat.SetElement(0, 0, 1.1);
		mat.SetElement(1, 1, 2.2);
		mat.SetElement(2, 2, 3.3);

		mat.SetRowIndexLock(true);
		mat.SetColIndexLock(true);

		mat.Compress();

		std::vector<int> rowInd;
		std::vector<int> colInd;

		for (int cont = 0; cont < 3; cont++)
		{
			colInd[cont] = mat.GetColIndexAddress()[cont];
		}

		for (int cont = 0; cont < 3+1; cont++)
		{
			rowInd[cont] = mat.GetRowIndexAddress()[cont];
		}

		mat.Reset(mat.GetRows(), mat.GetColumns());

		for (int cont = 0; cont < 3; cont++)
		{
			if (colInd[cont] != mat.GetColIndexAddress()[cont]) return true;
		}

		for (int cont = 0; cont < 3 + 1; cont++)
		{
			if (rowInd[cont] != mat.GetRowIndexAddress()[cont]) return true;
		}

	}



	return false;
}

int main(){

	bool test1 = true;

	int* nonzeros_vector = nullptr;
	int m = 5;
	int n = 3;

	ChCSR3Matrix matCSR3(m,n);

	ChMatrixDynamic<double> mat_base(m,n);
	mat_base(0, 1) = 0.1;
	mat_base(1, 2) = 1.2;
	mat_base(2, 0) = 2.0;
	mat_base(3, 1) = 3.1;
	mat_base(4, 0) = 4.0;

	mat_base(4, 1) = 4.1;
	mat_base(2, 1) = 2.1;

	CopyMatrix(matCSR3, mat_base);

	nonzeros_vector = new int[matCSR3.GetRows()];
	matCSR3.GetNonZerosDistribution(nonzeros_vector);
	

	if (PRINT_RESULTS)
	{
		PrintMatrix(mat_base);
		std::cout << std::endl;
		PrintMatrix(matCSR3);
	}

	

	for (int m_sel = 0; m_sel < m; m_sel++)
		for (int n_sel = 0; n_sel < n; n_sel++)
			if (mat_base(m_sel, n_sel) != matCSR3.GetElement(m_sel, n_sel))
				test1 = false;

	//////////////////

	ChCSR3Matrix matCSR3_1(m, n, nonzeros_vector);
	CopyMatrix(matCSR3_1, mat_base);

	bool test2 = are_arrays_equal(matCSR3_1, matCSR3);

	matCSR3_1.Reset(matCSR3_1.GetRows(),matCSR3_1.GetColumns());

	bool test3 = are_arrays_equal(matCSR3_1, matCSR3, true);

	///////////////////

	delete nonzeros_vector;

	if (PRINT_RESULTS)
	{
		std::cout << std::endl;
		std::cout << "Test 1: " << (test1 ? "passed" : "NOT passed") << std::endl;
		std::cout << "Test 2: " << (test2 ? "passed" : "NOT passed") << std::endl;
		std::cout << "Test 3: " << (test3 ? "passed" : "NOT passed") << std::endl;
		getchar();
	}

	is_Reset_function_broken();
	

	return (0);

}