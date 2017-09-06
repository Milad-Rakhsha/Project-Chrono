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
// Simulation parameters Fluid
// -----------------------------------------------------------------------------
// When adding functionality using "useWallBce" and "haveFluid" macros, pay
// attention to  "initializeFluidFromFile"
// options. For a double security, do your best to set "haveFluid" and "useWallBce"
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
    paramsH->HSML = 0.025;
    paramsH->MULT_INITSPACE = 1.0;
    paramsH->MULT_INITSPACE_Shells = 1.0;
    paramsH->epsMinMarkersDis = .01;
    paramsH->NUM_BOUNDARY_LAYERS = 3;
    paramsH->toleranceZone = paramsH->NUM_BOUNDARY_LAYERS * (paramsH->HSML * paramsH->MULT_INITSPACE);
    paramsH->BASEPRES = 1e-10;
    paramsH->LARGE_PRES = 0;
    paramsH->deltaPress;
    paramsH->multViscosity_FSI = 1;
    paramsH->gravity = mR3(0, 0, -1);
    paramsH->bodyForce3 = mR3(0, 0, 0);
    paramsH->rho0 = 1000;
    paramsH->markerMass = pow(paramsH->MULT_INITSPACE * paramsH->HSML, 3) * paramsH->rho0;
    paramsH->mu0 = 0.05;   ///< viscosity
    paramsH->kappa = 0.0;  ///<surface tension parameter
    paramsH->v_Max = 1;
    paramsH->EPS_XSPH = .5f;

    paramsH->USE_CUSP = true;                         ///< IISPH parameter: whether or not use cusp as the linear solver
    paramsH->Cusp_solver = bicgstab;                  ///< IISPH parameter: gmres, cr, bicgstab, cg
    paramsH->Verbose_monitoring = false;              ///< IISPH parameter: cusp linear solver setting
    paramsH->PPE_Solution_type = FORM_SPARSE_MATRIX;  ///< MATRIX_FREE, FORM_SPARSE_MATRIX
    paramsH->PPE_res = 1e-6;        ///< relative res, is used in the iterative solver and cusp solvers
    paramsH->PPE_Abs_res = 1e-5;    ///< absolute error, applied when cusp solvers are used
    paramsH->PPE_Max_Iter = 5;      ///< max number of iteration for cusp solvers
    paramsH->PPE_relaxation = 0.3;  ///< Increasing this to 0.5 causes instability, only used in MATRIX_FREE form
    paramsH->dT = 5e-3;
    paramsH->dT_Flex = 1e-3;

    /// Experimental parameters
    paramsH->Max_Pressure = 1e5;
    paramsH->IncompressibilityFactor = 1;     ///< Increasing this causes lager compressibility, but let for larger dt
    paramsH->ClampPressure = true;            ///< If the negative pressure should be clamped to zero or not
    paramsH->Adaptive_time_stepping = false;  ///< This let you use large time steps when possible
    paramsH->Co_number = 0.8;                 ///< 0.2 works well for most cases
    paramsH->dT_Max = 0.01;  ///< This is problem dependent should set by the user based on characteristic time step
    paramsH->Apply_BC_U = false;  ///< You should go to custom_math.h all the way to end of file and set your function

    paramsH->tFinal = 2;
    paramsH->timePause = 0;
    paramsH->kdT = 5;
    paramsH->gammaBB = 0.5;
    paramsH->binSize0;
    paramsH->rigidRadius;
    paramsH->densityReinit = 0;
    paramsH->enableTweak = 1;
    paramsH->enableAggressiveTweak = 0;
    paramsH->tweakMultV = 0.1;
    paramsH->tweakMultRho = .002;
    paramsH->bceType = ADAMI;  // ADAMI, mORIGINAL
    paramsH->cMin = mR3(-bxDim, -byDim, -bzDim) - mR3(paramsH->HSML * 5);
    paramsH->cMax = mR3(bxDim, byDim, 1.2 * bzDim) + mR3(paramsH->HSML * 5);

    //**************************************************************************************************
    int3 side0 = mI3(floor((paramsH->cMax.x - paramsH->cMin.x) / (2 * paramsH->HSML)),
                     floor((paramsH->cMax.y - paramsH->cMin.y) / (2 * paramsH->HSML)),
                     floor((paramsH->cMax.z - paramsH->cMin.z) / (2 * paramsH->HSML)));
    Real3 binSize3 = mR3((paramsH->cMax.x - paramsH->cMin.x) / side0.x, (paramsH->cMax.y - paramsH->cMin.y) / side0.y,
                         (paramsH->cMax.z - paramsH->cMin.z) / side0.z);
    paramsH->binSize0 = (binSize3.x > binSize3.y) ? binSize3.x : binSize3.y;
    paramsH->binSize0 = binSize3.x;
    paramsH->boxDims = paramsH->cMax - paramsH->cMin;
    //**************************************************************************************************
    //*** initialize straight channel
    paramsH->straightChannelBoundaryMin = paramsH->cMin;  // mR3(0, 0, 0);  // 3D channel
    paramsH->straightChannelBoundaryMax = paramsH->cMax;  // SmR3(3, 2, 3) * paramsH->sizeScale;
    //************************** modify pressure ***************************
    paramsH->deltaPress = mR3(0);
    // modify bin size stuff
    //******************* bin size adjustement and contact detection stuff *****************************
    int3 SIDE = mI3(int((paramsH->cMax.x - paramsH->cMin.x) / paramsH->binSize0 + .1),
                    int((paramsH->cMax.y - paramsH->cMin.y) / paramsH->binSize0 + .1),
                    int((paramsH->cMax.z - paramsH->cMin.z) / paramsH->binSize0 + .1));
    Real mBinSize = paramsH->binSize0;
    //**************************************************************************************************
    paramsH->gridSize = SIDE;
    paramsH->worldOrigin = paramsH->cMin;
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

}  // end namespace fsi
}  // end namespace chrono
#endif  // end of FSI_HMMWV_PARAMS_H_
