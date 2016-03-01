///////////////////////////////////////////////////////////////////////////////
//	main.cpp
//	Reads the initializes the particles, either from file or inside the code
//
//	Related Files: collideSphereSphere.cu, collideSphereSphere.cuh
//	Input File:		initializer.txt (optional: if initialize from file)
//					This file contains the sph particles specifications. The description
//					reads the number of particles first. The each line provides the
//					properties of one SPH particl:
//					position(x,y,z), radius, velocity(x,y,z), mass, \rho, pressure, mu,
// particle_type(rigid
// or fluid)
//
//	Created by Arman Pazouki
///////////////////////////////////////////////////////////////////////////////

// note: this is the original fsi_hmmwv model. uses RK2, an specific coupling, and density re_initializaiton.

// General Includes
#include <iostream>
#include <fstream>
#include <string>
#include <limits.h>
#include <vector>
#include <ctime>
#include <assert.h>
#include <stdlib.h>  // system

// SPH includes
#include "chrono_fsi/MyStructs.cuh"  //just for SimParams
#include "chrono_fsi/collideSphereSphere.cuh"
#include "chrono_fsi/printToFile.cuh"
#include "chrono_fsi/custom_cutil_math.h"
#include "chrono_fsi/SPHCudaUtils.h"
#include "chrono_fsi/checkPointReduced.h"
#include "chrono_fsi/UtilsDeviceOperations.cuh"


// Chrono Parallel Includes
#include "chrono_parallel/physics/ChSystemParallel.h"
#include "chrono_parallel/lcp/ChLcpSystemDescriptorParallel.h"

// Chrono Vehicle Include
#include "chrono_fsi/VehicleExtraProperties.h"
#include "chrono_vehicle/ChVehicleModelData.h"

//#include "chrono_utils/ChUtilsVehicle.h"
#include "utils/ChUtilsGeometry.h"
#include "utils/ChUtilsCreators.h"
#include "utils/ChUtilsGenerators.h"
#include "utils/ChUtilsInputOutput.h"

// Chrono general utils
#include "core/ChFileutils.h"
#include <core/ChTransform.h>  //transform acc from GF to LF for post process

//#include "BallDropParams.h"
//#include "chrono_fsi/SphInterface.h"
#include "chrono_fsi/incompressible_integrate.h"

#include "chrono_fsi/InitializeSphMarkers.h"
#include "chrono_fsi/FSI_Integrate.h"

// FSI Interface Includes
#include "params_test_fsi_implicit_damBreak.h"  //SetupParamsH()

#define haveFluid true
#define haveVehicle false

// Chrono namespaces
using namespace chrono;
using namespace chrono::collision;

using std::cout;
using std::endl;

// Define General variables
SimParams paramsH;


// =============================================================================

int main(int argc, char* argv[]) {
	//****************************************************************************************
	time_t rawtime;
	struct tm* timeinfo;

	//(void) cudaSetDevice(0);

	time(&rawtime);
	timeinfo = localtime(&rawtime);

	//****************************************************************************************
	// Arman take care of this block.
	// Set path to ChronoVehicle data files
	//  vehicle::SetDataPath(CHRONO_VEHICLE_DATA_DIR);
	//  vehicle::SetDataPath("/home/arman/Repos/GitBeta/chrono/src/demos/data/");
	//  SetChronoDataPath(CHRONO_DATA_DIR);

	// --------------------------
	// Create output directories.
	// --------------------------

	if (ChFileutils::MakeDirectory(out_dir.c_str()) < 0) {
		cout << "Error creating directory " << out_dir << endl;
		return 1;
	}

	if (povray_output) {
		if (ChFileutils::MakeDirectory(pov_dir_mbd.c_str()) < 0) {
			cout << "Error creating directory " << pov_dir_mbd << endl;
			return 1;
		}
	}

	if (ChFileutils::MakeDirectory(pov_dir_fluid.c_str()) < 0) {
		cout << "Error creating directory " << pov_dir_fluid << endl;
		return 1;
	}

	//****************************************************************************************
	const std::string simulationParams = out_dir
			+ "/simulation_specific_parameters.txt";
	simParams.open(simulationParams);
	simParams << " Job was submitted at date/time: " << asctime(timeinfo)
			<< endl;
	// ***************************** Create Fluid ********************************************
	thrust::host_vector<::int4> referenceArray;
	thrust::host_vector<Real3> posRadH; // do not set the size here since you are using push back later
	thrust::host_vector<Real3> velMasH;
	thrust::host_vector<Real4> rhoPresMuH;
	thrust::host_vector<uint> bodyIndex;

	thrust::host_vector<Real3> pos_ChSystemBackupH;
	thrust::host_vector<Real3> vel_ChSystemBackupH;
	thrust::host_vector<Real3> acc_ChSystemBackupH;
	thrust::host_vector<Real4> quat_ChSystemBackupH;
	thrust::host_vector<Real3> omegaVelGRF_ChSystemBackupH;
	thrust::host_vector<Real3> omegaAccGRF_ChSystemBackupH;

	Real sphMarkerMass = 0; // To be initialized in CreateFluidMarkers, and used in other places

	SetupParamsH(paramsH);

#if haveFluid
		//*** default num markers

		int numAllMarkers = 0;

		//*** initialize fluid particles
		::int2 num_fluidOrBoundaryMarkers = CreateFluidMarkers(posRadH, velMasH,
				rhoPresMuH, bodyIndex, paramsH);
		printf("num_fluidOrBoundaryMarkers %d %d \n",
				num_fluidOrBoundaryMarkers.x, num_fluidOrBoundaryMarkers.y);
		referenceArray.push_back(mI4(0, num_fluidOrBoundaryMarkers.x, -1, -1)); // map fluid -1
		numAllMarkers += num_fluidOrBoundaryMarkers.x;
		referenceArray.push_back(
		mI4(numAllMarkers, numAllMarkers + num_fluidOrBoundaryMarkers.y, 0, 0));
		numAllMarkers += num_fluidOrBoundaryMarkers.y;

		//*** set num objects

		SetNumObjects(numObjects, referenceArray, numAllMarkers);
		//        assert(posRadH.size() == numObjects.numAllMarkers && "(1) numObjects is not set correctly");
		if (posRadH.size() != numObjects.numAllMarkers) {
			printf(
					"\n\n\n\n Error! (1) numObjects is not set correctly \n\n\n\n");
			return -1;
		}
		if (numObjects.numAllMarkers == 0) {
			posRadH.clear();
			velMasH.clear();
			rhoPresMuH.clear();
			bodyIndex.clear();
			referenceArray.clear();
			std::cout << "No marker to begin with " << numObjects.numAllMarkers
					<< std::endl;
			return 0;
		}
#endif

	// ***************************** Create Interface ********************************************

	//    assert(posRadH.size() == numObjects.numAllMarkers && "(2) numObjects is not set correctly");
	if (posRadH.size() != numObjects.numAllMarkers) {
		printf("\n\n\n\n Error! (2) numObjects is not set correctly \n\n\n\n");
		return -1;
	}

	//*** Add sph data to the physics system

	int startIndexSph = 0;
#if haveFluid
	thrust::device_vector<Real3> posRadD = posRadH;
	thrust::device_vector<Real3> velMasD = velMasH;
	thrust::device_vector<Real4> rhoPresMuD = rhoPresMuH;
	thrust::device_vector<uint> bodyIndexD = bodyIndex;
	thrust::device_vector<Real4> derivVelRhoD;
	ResizeR4(derivVelRhoD, numObjects.numAllMarkers);

	int numFsiBodies = 0; //FSI_Bodies.size();
	thrust::device_vector<Real3> posRigid_fsiBodies_D;
	thrust::device_vector<Real4> velMassRigid_fsiBodies_D;
	thrust::device_vector<Real3> accRigid_fsiBodies_D;
	thrust::device_vector<Real4> q_fsiBodies_D;
	thrust::device_vector<Real3> omegaVelLRF_fsiBodies_D;
	thrust::device_vector<Real3> omegaAccLRF_fsiBodies_D;
	ResizeR3(posRigid_fsiBodies_D, numFsiBodies);
	ResizeR4(velMassRigid_fsiBodies_D, numFsiBodies);
	ResizeR3(accRigid_fsiBodies_D, numFsiBodies);
	ResizeR4(q_fsiBodies_D, numFsiBodies);
	ResizeR3(omegaVelLRF_fsiBodies_D, numFsiBodies);
	ResizeR3(omegaAccLRF_fsiBodies_D, numFsiBodies);

	thrust::host_vector<Real3> posRigid_fsiBodies_dummyH(numFsiBodies);
	thrust::host_vector<Real4> velMassRigid_fsiBodies_dummyH(numFsiBodies);
	thrust::host_vector<Real3> accRigid_fsiBodies_dummyH(numFsiBodies);
	thrust::host_vector<Real4> q_fsiBodies_dummyH(numFsiBodies);
	thrust::host_vector<Real3> omegaVelLRF_fsiBodies_dummyH(numFsiBodies);
	thrust::host_vector<Real3> omegaAccLRF_fsiBodies_dummyH(numFsiBodies);

	thrust::device_vector<Real3> posRigid_fsiBodies_D2 = posRigid_fsiBodies_D;
	thrust::device_vector<Real4> velMassRigid_fsiBodies_D2 = velMassRigid_fsiBodies_D;
	thrust::device_vector<Real3> accRigid_fsiBodies_D2 = accRigid_fsiBodies_D;

	thrust::device_vector<Real4> q_fsiBodies_D2 = q_fsiBodies_D;
	thrust::device_vector<Real3> omegaVelLRF_fsiBodies_D2 = omegaVelLRF_fsiBodies_D;
	thrust::device_vector<Real3> omegaAccLRF_fsiBodies_D2 = omegaAccLRF_fsiBodies_D;

	thrust::device_vector<Real3> rigid_FSI_ForcesD;
	thrust::device_vector<Real3> rigid_FSI_TorquesD;
	ResizeR3(rigid_FSI_ForcesD, numFsiBodies);
	ResizeR3(rigid_FSI_TorquesD, numFsiBodies);
	// assert
	if ((numObjects.numRigidBodies != numFsiBodies)
			|| (referenceArray.size() - 2 != numFsiBodies)) {
		printf(
				"\n\n\n\n Error! number of fsi bodies (%d) does not match numObjects.numRigidBodies (%d). Size of "
						"reference array: %d \n\n\n\n", numFsiBodies,
				numObjects.numRigidBodies, referenceArray.size());
		return -1;
	}
	ResizeR3(rigid_FSI_ForcesD, numObjects.numRigidBodies);
	ResizeR3(rigid_FSI_TorquesD, numObjects.numRigidBodies);

	thrust::device_vector<uint> rigidIdentifierD;
	ResizeU1(rigidIdentifierD, numObjects.numRigid_SphMarkers);
	thrust::device_vector<Real3> rigidSPH_MeshPos_LRF_D;
	ResizeR3(rigidSPH_MeshPos_LRF_D, numObjects.numRigid_SphMarkers);

	InitSystem(paramsH, numObjects);

	Populate_RigidSPH_MeshPos_LRF(rigidIdentifierD, rigidSPH_MeshPos_LRF_D,
			posRadD, posRigid_fsiBodies_D, q_fsiBodies_D, referenceArray,
			numObjects);

	// sync BCE velocity and position with rigid bodies kinematics
	UpdateRigidMarkersPositionVelocity(posRadD, velMasD, rigidSPH_MeshPos_LRF_D,
			rigidIdentifierD, posRigid_fsiBodies_D, q_fsiBodies_D,
			velMassRigid_fsiBodies_D, omegaVelLRF_fsiBodies_D, numObjects, paramsH);

	// ** initialize device mid step data
	thrust::device_vector<Real3> posRadD2 = posRadD;
	thrust::device_vector<Real3> velMasD2 = velMasD;
	thrust::device_vector<Real4> rhoPresMuD2 = rhoPresMuD;
	thrust::device_vector<Real3> vel_XSPH_D;
	ResizeR3(vel_XSPH_D, numObjects.numAllMarkers);
	//    assert(posRadD.size() == numObjects.numAllMarkers && "(3) numObjects is not set correctly");
	if (posRadD.size() != numObjects.numAllMarkers) {
		printf("\n\n\n\n Error! (3) numObjects is not set correctly \n\n\n\n");
		return -1;
	}
#endif

	// ***************************** System Initialize ********************************************

	double mTime = 0;

	DOUBLEPRECISION ?
			printf("Double Precision\n") : printf("Single Precision\n");

	int stepEnd = int(paramsH.tFinal / paramsH.dT); // 1.0e6;//2.4e6;//600000;//2.4e6 * (.02 * paramsH.sizeScale) /
													// currentParamsH.dT ; //1.4e6 * (.02 * paramsH.sizeScale) /
													// currentParamsH.dT ;//0.7e6 * (.02 * paramsH.sizeScale) /
													// currentParamsH.dT ;//0.7e6;//2.5e6;
													// //200000;//10000;//50000;//100000;
	printf("stepEnd %d\n", stepEnd);
	Real realTime = 0;

	SimParams paramsH_B = paramsH;
	paramsH_B.bodyForce3 = mR3(0);
	paramsH_B.dT = paramsH.dT;

	printf("\ntimePause %f, numPause %d\n", paramsH.timePause,
			int(paramsH.timePause / paramsH_B.dT));
	printf("paramsH.timePauseRigidFlex %f, numPauseRigidFlex %d\n\n",
			paramsH.timePauseRigidFlex,
			int(
					(paramsH.timePauseRigidFlex - paramsH.timePause)
							/ paramsH.dT + paramsH.timePause / paramsH_B.dT));
	//  InitSystem(paramsH, numObjects);
	SimParams currentParamsH = paramsH;

	simParams.close();

	// ******************************************************************************************
	// ******************************************************************************************
	// ******************************************************************************************
	// ******************************************************************************************
	// ***************************** Simulation loop ********************************************

	chrono::ChTimerParallel fsi_timer;
	fsi_timer.AddTimer("total_step_time");
	fsi_timer.AddTimer("fluid_initialization");
	fsi_timer.AddTimer("DoStepDynamics_FSI");
	fsi_timer.AddTimer("DoStepDynamics_ChronoRK2");

	for (int tStep = 0; tStep < stepEnd + 1; tStep++) {
		// -------------------
		// SPH Block
		// -------------------
		fsi_timer.Reset();

#if haveFluid
		CpuTimer mCpuTimer;
		mCpuTimer.Start();
		GpuTimer myGpuTimer;
		myGpuTimer.Start();

		if (realTime <= paramsH.timePause) {
			currentParamsH = paramsH_B;
		} else {
			currentParamsH = paramsH;
		}

		fsi_timer.start("total_step_time");

		fsi_timer.start("fluid_initialization");

		int out_steps = std::ceil((1.0 / paramsH.dT) / out_fps);
		PrintToFile(posRadD, velMasD, rhoPresMuD, referenceArray,
				currentParamsH, realTime, tStep, out_steps, pov_dir_fluid);

		// ******* slow down the sys.Check point the sys.
		CheckPointMarkers_Write(posRadH, velMasH, rhoPresMuH, bodyIndex,
				referenceArray, paramsH, numObjects, tStep, tStepsCheckPoint);

		//    // freeze sph. check it later
		//    if (fmod(realTime, 0.6) < paramsH.dT && realTime < 1.3) {
		//      FreezeSPH(velMasD, velMasH);
		//    }
		// *******

		fsi_timer.stop("fluid_initialization");
#endif
#if haveFluid
		fsi_timer.start("DoStepDynamics_FSI");

		DoStepFluid_implicit(posRadD, velMasD,
				rhoPresMuD,

				posRadD2, velMasD2, rhoPresMuD2,

				derivVelRhoD, rigidIdentifierD, rigidSPH_MeshPos_LRF_D,

				posRigid_fsiBodies_D, velMassRigid_fsiBodies_D, accRigid_fsiBodies_D,
				q_fsiBodies_D, omegaVelLRF_fsiBodies_D, omegaAccLRF_fsiBodies_D,

				posRigid_fsiBodies_D2, velMassRigid_fsiBodies_D2, accRigid_fsiBodies_D2,
				q_fsiBodies_D2, omegaVelLRF_fsiBodies_D2, omegaAccLRF_fsiBodies_D2,

				pos_ChSystemBackupH, vel_ChSystemBackupH, acc_ChSystemBackupH,
				quat_ChSystemBackupH, omegaVelGRF_ChSystemBackupH, omegaAccGRF_ChSystemBackupH,

				posRigid_fsiBodies_dummyH, velMassRigid_fsiBodies_dummyH, accRigid_fsiBodies_dummyH,
				q_fsiBodies_dummyH, omegaVelLRF_fsiBodies_dummyH, omegaAccLRF_fsiBodies_dummyH,

				rigid_FSI_ForcesD, rigid_FSI_TorquesD,

				bodyIndexD, referenceArray, numObjects, paramsH);
		fsi_timer.stop("DoStepDynamics_FSI");
#endif
		// -------------------
		// End SPH Block
		// -------------------

// -------------------
// SPH Block
// -------------------
#if haveFluid
		mCpuTimer.Stop();
		myGpuTimer.Stop();
		if (tStep % 2 == 0) {
			printf(
					"step: %d, realTime: %f, step Time (CUDA): %f, step Time (CPU): %f\n ",
					tStep, realTime, (Real) myGpuTimer.Elapsed(),
					1000 * mCpuTimer.Elapsed());
		}
#endif
		fsi_timer.stop("total_step_time");
		fsi_timer.PrintReport();
		// -------------------
		// End SPH Block
		// -------------------

		fflush(stdout);
		realTime += currentParamsH.dT;

	}
	posRadH.clear();
	velMasH.clear();
	rhoPresMuH.clear();
	bodyIndex.clear();
	referenceArray.clear();

	pos_ChSystemBackupH.clear();
	vel_ChSystemBackupH.clear();
	acc_ChSystemBackupH.clear();
	quat_ChSystemBackupH.clear();
	omegaVelGRF_ChSystemBackupH.clear();
	omegaAccGRF_ChSystemBackupH.clear();

// Arman LRF in omegaLRF may need change
#if haveFluid
	ClearMyThrustR3(posRadD);
	ClearMyThrustR3(velMasD);
	ClearMyThrustR4(rhoPresMuD);
	ClearMyThrustU1(bodyIndexD);
	ClearMyThrustR4(derivVelRhoD);
	ClearMyThrustU1(rigidIdentifierD);
	ClearMyThrustR3(rigidSPH_MeshPos_LRF_D);

	ClearMyThrustR3(posRadD2);
	ClearMyThrustR3(velMasD2);
	ClearMyThrustR4(rhoPresMuD2);
	ClearMyThrustR3(vel_XSPH_D);

	ClearMyThrustR3(posRigid_fsiBodies_D);
	ClearMyThrustR4(velMassRigid_fsiBodies_D);
	ClearMyThrustR3(accRigid_fsiBodies_D);
	ClearMyThrustR4(q_fsiBodies_D);
	ClearMyThrustR3(omegaVelLRF_fsiBodies_D);
	ClearMyThrustR3(omegaAccLRF_fsiBodies_D);

	ClearMyThrustR3(posRigid_fsiBodies_D2);
	ClearMyThrustR4(velMassRigid_fsiBodies_D2);
	ClearMyThrustR3(accRigid_fsiBodies_D2);
	ClearMyThrustR4(q_fsiBodies_D2);
	ClearMyThrustR3(omegaVelLRF_fsiBodies_D2);
	ClearMyThrustR3(omegaAccLRF_fsiBodies_D2);

	ClearMyThrustR3(rigid_FSI_ForcesD);
	ClearMyThrustR3(rigid_FSI_TorquesD);

	posRigid_fsiBodies_dummyH.clear();
	velMassRigid_fsiBodies_dummyH.clear();
	accRigid_fsiBodies_dummyH.clear();
	q_fsiBodies_dummyH.clear();
	omegaVelLRF_fsiBodies_dummyH.clear();
	omegaAccLRF_fsiBodies_dummyH.clear();
#endif
	return 0;
}
