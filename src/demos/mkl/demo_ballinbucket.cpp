//
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2010-2012 Alessandro Tasora
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
//     - collisions and contacts
//     - sharing a ChMaterialSurface property between bodies
//
//       (This is just a possible method of integration
//       of Chrono::Engine + Irrlicht: many others
//       are possible.)
//
//	 CHRONO
//   ------
//   Multibody dinamics engine
//
// ------------------------------------------------
//             www.deltaknowledge.com
// ------------------------------------------------
///////////////////////////////////////////////////

#include "chrono/physics/ChSystem.h"
#include "chrono/physics/ChBodyEasy.h"
#include "chrono/assets/ChTexture.h"
#include "chrono_irrlicht/ChIrrApp.h"
#include "chrono_mkl/ChLcpMklSolver.h"
#include <chrono_mkl/ChInteriorPoint.h>
#include <chrono_matlab/ChLcpMatlabSolver.h>

// Use the namespace of Chrono

using namespace chrono;

// Use the main namespaces of Irrlicht
using namespace irr;

using namespace core;
using namespace scene;
using namespace video;
using namespace io;
using namespace gui;

// Create a bunch of ChronoENGINE rigid bodies that
// represent bricks in a large wall.

void create_bucket(ChSystem& mphysicalSystem) {

    // Create a material that will be shared between bricks
    ChSharedPtr<ChMaterialSurface> mmaterial(new ChMaterialSurface);
    mmaterial->SetFriction(0.4f);
    mmaterial->SetCompliance(0.0000005f);
    mmaterial->SetComplianceT(0.0000005f);
    mmaterial->SetDampingF(0.2f);

    // Create bricks
	int edges = 6;
	double s = 0.01;
	double apothem = 0.15;
	double height = 0.25;
	double ball_radius = 0.05;

	
	double alfa = 2*PI/edges;
	double width = 2*apothem*tan(alfa/2);
	double ball_pos_radius = 0.5*apothem / cos(alfa / 2);


	// Create Ployhedron
	for (int edge_k = 0; edge_k < edges; edge_k++)
	{
		double alfa_k = alfa*edge_k;
		ChSharedPtr<ChBodyEasyBox> wall(new ChBodyEasyBox(width, height, s,  // x,y,z size
															1000,         // density
															true,         // collide enable?
															(alfa_k>=0 && alfa_k<PI) ? true : false)
															//true)
															);       // visualization?

		ChQuaternion<double> quat(cos(alfa_k/2), 0, sin(alfa_k / 2), 0);
		wall->SetRot(quat);
		wall->SetPos(ChVector<>((apothem + s / 2)*sin(alfa_k), height / 2, (apothem + s / 2)*cos(alfa_k)));
		wall->SetMaterialSurface(mmaterial);
		wall->SetBodyFixed(true);
		mphysicalSystem.Add(wall);

		// optional, attach a texture for better visualization
		ChSharedPtr<ChTexture> mtexture(new ChTexture());
		mtexture->SetTextureFilename(GetChronoDataFile("cubetexture_borders.png"));
		wall->AddAsset(mtexture);

		int ball_arrays = 1;
		for (int ball_set = 0; ball_set < ball_arrays; ball_set++)
		{
			// Create a ball that will collide with wall
			ChSharedPtr<ChBodyEasySphere> mrigidBall(new ChBodyEasySphere(ball_radius,       // radius
				8000,    // density
				true,    // collide enable?
				true));  // visualization?
			mrigidBall->SetMaterialSurface(mmaterial);
			mrigidBall->SetPos(ChVector<>(ball_pos_radius*sin(alfa_k + alfa / 2), (ball_set+1)*height / ball_arrays, ball_pos_radius*cos(alfa_k + alfa / 2)));
			mrigidBall->SetPos_dtdt(ChVector<>(0, -3, 0));          // set initial acceleration
			mrigidBall->GetMaterialSurface()->SetFriction(0.4f);  // use own (not shared) matrial properties
			mrigidBall->GetMaterialSurface()->SetCompliance(0.0);
			mrigidBall->GetMaterialSurface()->SetComplianceT(0.0);
			mrigidBall->GetMaterialSurface()->SetDampingF(0.2f);

			mphysicalSystem.Add(mrigidBall);

			// optional, attach a texture for better visualization
			ChSharedPtr<ChTexture> mtextureball(new ChTexture());
			mtextureball->SetTextureFilename(GetChronoDataFile("bluwhite.png"));
			mrigidBall->AddAsset(mtextureball);
		}
		

	}


    // Create the floor using
    // fixed rigid body of 'box' type:

    ChSharedPtr<ChBodyEasyBox> mrigidFloor(new ChBodyEasyBox(250, 4, 250,  // x,y,z size
                                                             1000,         // density
                                                             true,         // collide enable?
                                                             true));       // visualization?
    mrigidFloor->SetPos(ChVector<>(0, -2, 0));
    mrigidFloor->SetMaterialSurface(mmaterial);
    mrigidFloor->SetBodyFixed(true);

    mphysicalSystem.Add(mrigidFloor);

    
}


int main(int argc, char* argv[]) {
    // Create a ChronoENGINE physical system
    ChSystem mphysicalSystem;

    // Create the Irrlicht visualization (open the Irrlicht device,
    // bind a simple user interface, etc. etc.)
    ChIrrApp application(&mphysicalSystem, L"Balls in bucket", core::dimension2d<u32>(800, 600), false, true);

    // Easy shortcuts to add camera, lights, logo and sky in Irrlicht scene:
    ChIrrWizard::add_typical_Logo(application.GetDevice());
    ChIrrWizard::add_typical_Sky(application.GetDevice());
    ChIrrWizard::add_typical_Lights(application.GetDevice(), core::vector3df(70.f, 120.f, -90.f), core::vector3df(30.f, 80.f, 60.f), 290, 190);
    ChIrrWizard::add_typical_Camera(application.GetDevice(), core::vector3df(0, 1, 0), core::vector3df(0, 0, 0));
    //ChIrrWizard::add_typical_Camera(application.GetDevice(), core::vector3df(-15, 14, -30), core::vector3df(0, 5, 0));

    //
    // HERE YOU POPULATE THE MECHANICAL SYSTEM OF CHRONO...
    //

    // Create all the rigid bodies.
    create_bucket(mphysicalSystem);

    // Use this function for adding a ChIrrNodeAsset to all items
    // If you need a finer control on which item really needs a visualization proxy in
    // Irrlicht, just use application.AssetBind(myitem); on a per-item basis.
    application.AssetBindAll();

    // Use this function for 'converting' into Irrlicht meshes the assets
    // into Irrlicht-visualizable meshes
    application.AssetUpdateAll();

    // Prepare the physical system for the simulation

    mphysicalSystem.SetLcpSolverType(ChSystem::LCP_ITERATIVE_SOR_MULTITHREAD);

    //mphysicalSystem.SetUseSleeping(true);

    mphysicalSystem.SetMaxPenetrationRecoverySpeed(1.6);  // used by Anitescu stepper only
    mphysicalSystem.SetIterLCPmaxItersSpeed(40);
    mphysicalSystem.SetIterLCPmaxItersStab(20);  // unuseful for Anitescu, only Tasora uses this
    mphysicalSystem.SetIterLCPwarmStarting(true);
    mphysicalSystem.SetParallelThreadNumber(4);

	// Change solver to MKL
	ChInteriorPoint* ip_solver = new ChInteriorPoint;
	mphysicalSystem.ChangeLcpSolverStab(ip_solver);
	mphysicalSystem.ChangeLcpSolverSpeed(ip_solver);
	application.GetSystem()->Update();

	//// Change solver to Matlab external linear solver, for max precision in benchmarks
	//ChMatlabEngine matlab_engine;
	//ChLcpMatlabSolver* matlab_solver_stab = new ChLcpMatlabSolver(matlab_engine);
	//ChLcpMatlabSolver* matlab_solver_speed = new ChLcpMatlabSolver(matlab_engine);
	//mphysicalSystem.ChangeLcpSolverStab(matlab_solver_stab);
	//mphysicalSystem.ChangeLcpSolverSpeed(matlab_solver_speed);
	//application.GetSystem()->Update();


    //
    // THE SOFT-REAL-TIME CYCLE
    //

    application.SetStepManage(true);
    application.SetTimestep(0.02);

    while (application.GetDevice()->run()) {
        application.GetVideoDriver()->beginScene(true, true, SColor(255, 140, 161, 192));

        ChIrrTools::drawGrid(application.GetVideoDriver(), 5, 5, 20, 20,
                             ChCoordsys<>(ChVector<>(0, 0.04, 0), Q_from_AngAxis(CH_C_PI / 2, VECT_X)),
                             video::SColor(50, 90, 90, 150), true);

        application.DrawAll();

        application.DoStep();

        application.GetVideoDriver()->endScene();
    }

    return 0;
}
