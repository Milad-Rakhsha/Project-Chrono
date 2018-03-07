/*
 * SetupFsiParams.h
 *
 *  Created on: Mar 2, 2015
 *      Author: Arman Pazouki
 */
#ifndef FSI_TEST_CYLINDERDROP_NEW_
#define FSI_TEST_CYLINDERDROP_NEW_

/* C/C++ standard library*/
/* Chrono::FSI Library*/
#include "chrono_fsi/ChParams.cuh"
#include "chrono_fsi/utils/ChUtilsPrintStruct.h"

namespace chrono {
namespace fsi {
// -----------------------------------------------------------------------------
// Simulation parameters FSI
// -----------------------------------------------------------------------------

// Duration of the "hold time" (vehicle chassis fixed and no driver inputs).
// This can be used to allow the granular material to settle.

// Dimensions
// -----------------------------------------------------------------------------
// Simulation parameters Fluid
// -----------------------------------------------------------------------------
//	when adding functionality using "useWallBce" and "haveFluid" macros, pay
// attention to  "initializeFluidFromFile"
// options.
//	for a double security, do your best to set "haveFluid" and "useWallBce"
// based on the data you have from
// checkpoint files
// very important, since this option will overwrite the BCE pressure and
// paramsH->LARGE_PRES is only used for the initialization of the BCE markers

/**
 * @brief
 *    Fills in paramsH with simulation parameters.
 * @details
 *    The description of each parameter set here can be found in MyStruct.h
 *
 * @param paramsH: struct defined in MyStructs.cuh
 */
void SetupParamsH(SimParams* paramsH, Real bxDim, Real byDim, Real bzDim, Real fxDim, Real fyDim, Real fzDim) {
    paramsH->sizeScale = 1;  // don't change it.
    paramsH->HSML = 0.01 / 0.95;
    paramsH->MULT_INITSPACE = 0.95;
    Real initSpace = paramsH->MULT_INITSPACE * paramsH->HSML;
    paramsH->epsMinMarkersDis = .001;
    paramsH->NUM_BOUNDARY_LAYERS = 3;
    paramsH->toleranceZone = paramsH->NUM_BOUNDARY_LAYERS * (paramsH->HSML * paramsH->MULT_INITSPACE);
    paramsH->BASEPRES = 0.0;
    paramsH->LARGE_PRES = 0;
    paramsH->deltaPress = mR3(0.0, 0, 0.0);
    paramsH->multViscosity_FSI = 1;
    paramsH->gravity = mR3(0.0, 0, 0.0);
    paramsH->bodyForce3 = mR3(0.001, 0, 0);

    paramsH->V_in = mR3(0.00, 0, 0.0);
    paramsH->x_in = -bxDim + 3 * initSpace;

    paramsH->Conservative_Form = false;
    paramsH->USE_NonIncrementalProjection = true;
    paramsH->Adaptive_time_stepping = true;  ///< This let you use large time steps when possible
    paramsH->dT = 1e-3;
    paramsH->dT_Max = 1.0;
    paramsH->Co_number = 0.2;  ///< 0.2 works well for most cases
    paramsH->EPS_XSPH = 0.0;   // Note that increasing this coefficient stabilizes the simulation but adds dissipation
    paramsH->beta_shifting = 0.2;  // increasing this factor decreases the Lagrangian nature of the model

    paramsH->rho0 = 1000;
    paramsH->markerMass = pow(paramsH->MULT_INITSPACE * paramsH->HSML, 3) * paramsH->rho0;
    paramsH->mu0 = 0.1;
    paramsH->kappa = 0.000;  ///< surface tension parameter, experimental
    paramsH->v_Max = 0.0;    // Arman, I changed it to 0.1 for vehicle. Check this

    paramsH->L_Characteristic = bzDim;
    paramsH->USE_LinearSolver = false;                ///< IISPH parameter: whether or not use linear solvers
    paramsH->LinearSolver = bicgstab;                 ///< IISPH parameter: gmres, cr, bicgstab, cg
    paramsH->Verbose_monitoring = false;              ///< IISPH parameter: showing iter/residual
    paramsH->PPE_Solution_type = FORM_SPARSE_MATRIX;  ///< MATRIX_FREE, FORM_SPARSE_MATRIX
    paramsH->LinearSolver_Rel_Tol = 1e-8;  ///< relative res, is used in the matrix free solver and linear solvers
    paramsH->LinearSolver_Abs_Tol = 1e-5;  ///< absolute error, applied when linear solvers are used
    paramsH->LinearSolver_Max_Iter = 100;  ///< max number of iteration for linear solvers
    paramsH->PPE_relaxation = 0.98;        ///< Increasing this to 0.5 causes instability, only used in MATRIX_FREE form

    paramsH->Apply_BC_U = false;   ///< You should go to custom_math.h all the way to end of file and set your function
    paramsH->bceType = mORIGINAL;  // ADAMI, mORIGINAL
    paramsH->ApplyInFlowOutFlow = false;
    paramsH->outflow = paramsH->cMax - mR3(initSpace * 25);
    paramsH->inflow = paramsH->cMin + mR3(initSpace * 25);
    paramsH->cMin = mR3(-bxDim / 2 - initSpace / 2, -byDim / 2 - initSpace / 2, 0.0 - 5.0 * initSpace);
    paramsH->cMax = mR3(bxDim / 2 + initSpace / 2, byDim / 2 + initSpace / 2, bzDim + 5.0 * initSpace);

    //****************************************************************************************
    // note that neighbor search should be performed via the largest characteristic length for now
    int3 side0 = mI3(floor((paramsH->cMax.x - paramsH->cMin.x) / (2 * paramsH->HSML)),
                     floor((paramsH->cMax.y - paramsH->cMin.y) / (2 * paramsH->HSML)),
                     floor((paramsH->cMax.z - paramsH->cMin.z) / (2 * paramsH->HSML)));

    Real3 binSize3 = mR3((paramsH->cMax.x - paramsH->cMin.x) / side0.x, (paramsH->cMax.y - paramsH->cMin.y) / side0.y,
                         (paramsH->cMax.z - paramsH->cMin.z) / side0.z);
    paramsH->binSize0 = (binSize3.x > binSize3.y) ? binSize3.x : binSize3.y;
    //	paramsH->binSize0 = (paramsH->binSize0 > binSize3.z) ? paramsH->binSize0
    //: binSize3.z;
    paramsH->binSize0 = binSize3.x;  // for effect of distance. Periodic BC in x
                                     // direction. we do not care about
                                     // paramsH->cMax y and z.
    paramsH->boxDims = paramsH->cMax - paramsH->cMin;
    //****************************************************************************************
    //	paramsH->cMinInit = mR3(-1.7, -1.55, -0.5); // 3D channel
    //	paramsH->cMaxInit = mR3(+1.7, +1.55, +0.5);
    //****************************************************************************************
    //*** initialize straight channel
    paramsH->straightChannelBoundaryMin = paramsH->cMin;  // mR3(0, 0, 0);  // 3D channel
    paramsH->straightChannelBoundaryMax = paramsH->cMax;  // SmR3(3, 2, 3) * paramsH->sizeScale;
    //************************** modify pressure ***************************
    //		paramsH->deltaPress = paramsH->rho0 * paramsH->boxDims *
    // paramsH->bodyForce3;  //did not work as I expected
    paramsH->deltaPress = mR3(0);  // Wrong: 0.9 * paramsH->boxDims *
                                   // paramsH->bodyForce3; // viscosity and
                                   // boundary shape should play a roll

    // modify bin size stuff
    //****************************** bin size adjustement and contact detection
    // stuff *****************************
    int3 SIDE = mI3(int((paramsH->cMax.x - paramsH->cMin.x) / paramsH->binSize0 + .1),
                    int((paramsH->cMax.y - paramsH->cMin.y) / paramsH->binSize0 + .1),
                    int((paramsH->cMax.z - paramsH->cMin.z) / paramsH->binSize0 + .1));
    Real mBinSize = paramsH->binSize0;  // Best solution in that case may be to
                                        // change cMax or cMin such that periodic
                                        // sides be a multiple of binSize
    //**********************************************************************************************************
    paramsH->gridSize = SIDE;
    // paramsH->numCells = SIDE.x * SIDE.y * SIDE.z;
    paramsH->worldOrigin = paramsH->cMin - 0 * mR3(paramsH->HSML);

    paramsH->cellSize = mR3(mBinSize, mBinSize, mBinSize);

    std::cout << "******************** paramsH Content" << std::endl;
    std::cout << "paramsH->sizeScale: " << paramsH->sizeScale << std::endl;
    std::cout << "paramsH->HSML: " << paramsH->HSML << std::endl;
    std::cout << "paramsH->bodyForce3: ";
    utils::printStruct(paramsH->bodyForce3);
    std::cout << "paramsH->gravity: ";
    utils::printStruct(paramsH->gravity);
    std::cout << "paramsH->rho0: " << paramsH->rho0 << std::endl;
    std::cout << "paramsH->mu0: " << paramsH->mu0 << std::endl;
    std::cout << "paramsH->v_Max: " << paramsH->v_Max << std::endl;
    std::cout << "paramsH->dT: " << paramsH->dT << std::endl;
    std::cout << "paramsH->tFinal: " << paramsH->tFinal << std::endl;
    std::cout << "paramsH->timePause: " << paramsH->timePause << std::endl;
    std::cout << "paramsH->timePauseRigidFlex: " << paramsH->timePauseRigidFlex << std::endl;
    std::cout << "paramsH->densityReinit: " << paramsH->densityReinit << std::endl;
    std::cout << "paramsH->cMin: ";
    utils::printStruct(paramsH->cMin);
    std::cout << "paramsH->cMax: ";
    utils::printStruct(paramsH->cMax);
    std::cout << "paramsH->MULT_INITSPACE: " << paramsH->MULT_INITSPACE << std::endl;
    std::cout << "paramsH->NUM_BOUNDARY_LAYERS: " << paramsH->NUM_BOUNDARY_LAYERS << std::endl;
    std::cout << "paramsH->toleranceZone: " << paramsH->toleranceZone << std::endl;
    std::cout << "paramsH->NUM_BCE_LAYERS: " << paramsH->NUM_BCE_LAYERS << std::endl;
    std::cout << "paramsH->solidSurfaceAdjust: " << paramsH->solidSurfaceAdjust << std::endl;
    std::cout << "paramsH->BASEPRES: " << paramsH->BASEPRES << std::endl;
    std::cout << "paramsH->LARGE_PRES: " << paramsH->LARGE_PRES << std::endl;
    std::cout << "paramsH->deltaPress: ";
    utils::printStruct(paramsH->deltaPress);
    std::cout << "paramsH->nPeriod: " << paramsH->nPeriod << std::endl;
    std::cout << "paramsH->EPS_XSPH: " << paramsH->EPS_XSPH << std::endl;
    std::cout << "paramsH->multViscosity_FSI: " << paramsH->multViscosity_FSI << std::endl;
    std::cout << "paramsH->rigidRadius: ";
    utils::printStruct(paramsH->rigidRadius);
    std::cout << "paramsH->binSize0: " << paramsH->binSize0 << std::endl;
    std::cout << "paramsH->boxDims: ";
    utils::printStruct(paramsH->boxDims);
    std::cout << "paramsH->gridSize: ";
    utils::printStruct(paramsH->gridSize);
    std::cout << "********************" << std::endl;
}

// -----------------------------------------------------------------------------
// Specification of the terrain
// -----------------------------------------------------------------------------
//
//// Control visibility of containing bin walls
// bool visible_walls = false;
//
// int Id_g = 100;
// Real r_g = 0.02;
// Real rho_g = 2500;
// Real vol_g = (4.0 / 3) * chrono::CH_C_PI * r_g * r_g * r_g; // Arman: PI or
// CH_C_PI
// Real mass_g = rho_g * vol_g;
// chrono::ChVector<> inertia_g = 0.4 * mass_g * r_g * r_g
//		* chrono::ChVector<>(1, 1, 1);
//
//
//
// int num_particles = 1000;

// -----------------------------------------------------------------------------
// Output parameters
// -----------------------------------------------------------------------------

// Real vertical_offset = 0;  // vehicle vertical offset
}  // end namespace fsi
}  // end namespace chrono
#endif  // end of FSI_HMMWV_PARAMS_H_
