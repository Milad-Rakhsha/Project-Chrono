// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2014 projectchrono.org
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Authors: Radu Serban, Hammad Mazhar
// =============================================================================
//
// ChronoParallel test program using penalty method for frictional contact.
//
// The model simulated here consists of a number of spherical objects falling
// onto a mixer blade attached through a revolute joint to the ground.
//
// The global reference frame has Z up.
//
// If available, OpenGL is used for run-time rendering. Otherwise, the
// simulation is carried out for a pre-defined duration and output files are
// generated for post-processing with POV-Ray.
// =============================================================================

#include <cmath>
#include <cstdio>
#include <vector>

#include "chrono/physics/ChSystemNSC.h"

#include "chrono/ChConfig.h"
#include "chrono/core/ChFileutils.h"
#include "chrono/utils/ChUtilsCreators.h"
#include "chrono/utils/ChUtilsInputOutput.h"

#include "chrono/core/ChLinearAlgebra.h"
#include "chrono/core/ChLinkedListMatrix.h"
#include "chrono/physics/ChContactContainer.h"
#include "chrono/physics/ChContactContainerNSC.h"
#include "chrono/solver/ChSystemDescriptor.h"

#ifdef CHRONO_OPENGL
#include "chrono_opengl/ChOpenGLWindow.h"
#endif
int checknumcontact = 0;

using namespace chrono;
using namespace chrono::collision;

// -----------------------------------------------------------------------------
// Create a bin consisting of five boxes attached to the ground and a mixer
// blade attached through a revolute joint to ground. The mixer is constrained
// to rotate at constant angular velocity.
// -----------------------------------------------------------------------------
void AddContainer(ChSystemNSC* sys) {
    // IDs for the two bodies
    int binId = -200;
    int mixerId = -201;

    // Create a common material
    auto mat = std::make_shared<ChMaterialSurfaceNSC>();
    mat->SetFriction(0.4f);

    // Create the containing bin (2 x 2 x 1)
    auto bin = std::make_shared<ChBody>(ChMaterialSurface::NSC);
    bin->SetMaterialSurface(mat);
    bin->SetIdentifier(binId);
    bin->SetMass(1);
    bin->SetPos(ChVector<>(0, 0, 0));
    bin->SetRot(ChQuaternion<>(1, 0, 0, 0));
    bin->SetCollide(true);
    bin->SetBodyFixed(true);

    ChVector<> hdim(1, 1, 0.5);
    double hthick = 0.1;

    bin->GetCollisionModel()->ClearModel();
    utils::AddBoxGeometry(bin.get(), ChVector<>(hdim.x(), hdim.y(), hthick), ChVector<>(0, 0, -hthick));
    utils::AddBoxGeometry(bin.get(), ChVector<>(hthick, hdim.y(), hdim.z()),
                          ChVector<>(-hdim.x() - hthick, 0, hdim.z()));
    utils::AddBoxGeometry(bin.get(), ChVector<>(hthick, hdim.y(), hdim.z()),
                          ChVector<>(hdim.x() + hthick, 0, hdim.z()));
    utils::AddBoxGeometry(bin.get(), ChVector<>(hdim.x(), hthick, hdim.z()),
                          ChVector<>(0, -hdim.y() - hthick, hdim.z()));
    utils::AddBoxGeometry(bin.get(), ChVector<>(hdim.x(), hthick, hdim.z()),
                          ChVector<>(0, hdim.y() + hthick, hdim.z()));
    bin->GetCollisionModel()->SetFamily(1);
    bin->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(2);
    bin->GetCollisionModel()->BuildModel();

    sys->AddBody(bin);

    // The rotating mixer body (1.6 x 0.2 x 0.4)
    auto mixer = std::make_shared<ChBody>(ChMaterialSurface::NSC);
    mixer->SetMaterialSurface(mat);
    mixer->SetIdentifier(mixerId);
    mixer->SetMass(10.0);
    mixer->SetInertiaXX(ChVector<>(50, 50, 50));
    mixer->SetPos(ChVector<>(0, 0, 0.205));
    mixer->SetBodyFixed(false);
    mixer->SetCollide(true);

    ChVector<> hsize(0.8, 0.1, 0.2);

    mixer->GetCollisionModel()->ClearModel();
    utils::AddBoxGeometry(mixer.get(), hsize);
    mixer->GetCollisionModel()->SetFamily(2);
    mixer->GetCollisionModel()->BuildModel();

    sys->AddBody(mixer);

    // Create an engine between the two bodies, constrained to rotate at 90 deg/s
    auto motor = std::make_shared<ChLinkEngine>();

    motor->Initialize(mixer, bin, ChCoordsys<>(ChVector<>(0, 0, 0), ChQuaternion<>(1, 0, 0, 0)));

    motor->Set_eng_mode(ChLinkEngine::ENG_MODE_ROTATION);
    motor->Set_rot_funct(std::make_shared<ChFunction_Ramp>(0, CH_C_PI / 2));

    sys->AddLink(motor);
}

// -----------------------------------------------------------------------------
// Create the falling spherical objects in a unfiorm rectangular grid.
// -----------------------------------------------------------------------------
void AddFallingBalls(ChSystemNSC* sys) {
    // Common material
    auto ballMat = std::make_shared<ChMaterialSurfaceNSC>();
    ballMat->SetFriction(0.4f);

    // Create the falling balls
    int ballId = 0;
    double mass = 1;
    double radius = 0.1;
    ChVector<> inertia = (2.0 / 5.0) * mass * radius * radius * ChVector<>(1, 1, 1);

    for (int ix = -2; ix < 3; ix++) {
        for (int iy = -2; iy < 3; iy++) {
            ChVector<> pos(0.4 * ix, 0.4 * iy, 1);

            auto ball = std::make_shared<ChBody>(ChMaterialSurface::NSC);
            ball->SetMaterialSurface(ballMat);

            ball->SetIdentifier(ballId++);
            ball->SetMass(mass);
            ball->SetInertiaXX(inertia);
            ball->SetPos(pos);
            ball->SetRot(ChQuaternion<>(1, 0, 0, 0));
            ball->SetBodyFixed(false);
            ball->SetCollide(true);

            ball->GetCollisionModel()->ClearModel();
            utils::AddSphereGeometry(ball.get(), radius);
            ball->GetCollisionModel()->BuildModel();

            sys->AddBody(ball);
        }
    }
}

// -----------------------------------------------------------------------------
// Create the system, specify simulation parameters, and run simulation loop.
// -----------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    GetLog() << "Copyright (c) 2017 projectchrono.org\nChrono version: " << CHRONO_VERSION << "\n\n";
    //    int nbt = atoi(argv[1]);
    //    double wd = strtod(argv[2], NULL);
    int threads = 8;

    // Simulation parameters
    // ---------------------

    double gravity = 9.81;
    double time_step = 1e-3;
    double time_end = 1;

    double out_fps = 50;

    uint max_iteration = 30;
    double tolerance = 1e-3;

    // Create system
    // -------------

    ChSystemNSC msystem;

    // Set number of threads.
    int max_threads = CHOMPfunctions::GetNumProcs();
    if (threads > max_threads)
        threads = max_threads;
    msystem.SetParallelThreadNumber(threads);
    CHOMPfunctions::SetNumThreads(threads);

    // Set gravitational acceleration
    msystem.Set_G_acc(ChVector<>(0, 0, -gravity));

    // Create the fixed and moving bodies
    // ----------------------------------

    AddContainer(&msystem);
    AddFallingBalls(&msystem);

    // Perform the simulation
    // ----------------------

    // Run simulation for specified time
    int num_steps = (int)std::ceil(time_end / time_step);
    double time = 0;
    std::string fname = "_" + std::to_string(0) + "_";

    std::vector<std::shared_ptr<ChBody> > blist = *(msystem.Get_bodylist());

    if (ChFileutils::MakeDirectory("dumpDir_SER") < 0) {
        std::cout << "Error creating directory dumpDir\n";
        return 1;
    }

    std::shared_ptr<ChSystemDescriptor> sysd = msystem.GetSystemDescriptor();
    std::shared_ptr<ChContactContainerNSC> iconcon =
        std::dynamic_pointer_cast<ChContactContainerNSC>(msystem.GetContactContainer());

    for (int i = 0; i < num_steps; i++) {
        msystem.DoStepDynamics(time_step);
        time += time_step;

        int numContact = iconcon->GetNcontacts();
        int numBodies = msystem.Get_bodylist()->size();

        std::cout << "numContact : " << numContact << "\n";
        std::cout << "numBodies : " << numBodies << "\n";
        std::cout << "checknumcontact : " << checknumcontact << "\n";

        std::cout << "\n Dumping Newton matrix components : " << i << "\n ";

        chrono::ChLinkedListMatrix mdM;
        chrono::ChLinkedListMatrix mdCq;
        chrono::ChLinkedListMatrix mdE;
        chrono::ChMatrixDynamic<double> mdf;
        chrono::ChMatrixDynamic<double> mdb;
        chrono::ChMatrixDynamic<double> mdfric;
        chrono::ChMatrixDynamic<double> mx;  // constraint data (q,l)
        sysd->ConvertToMatrixForm(&mdCq, &mdM, &mdE, &mdf, &mdb, &mdfric);
        sysd->FromVariablesToVector(mx);

        std::string fname = "_" + std::to_string(i) + "_";

        std::string nameM = "dumpDir_SER/_" + fname + "dump_M_.dat";
        chrono::ChStreamOutAsciiFile file_M(nameM.c_str());
        mdM.StreamOUTsparseMatlabFormat(file_M);

        std::string nameC = "dumpDir_SER/_" + fname + "dump_Cq_.dat";
        chrono::ChStreamOutAsciiFile file_Cq(nameC.c_str());
        mdCq.StreamOUTsparseMatlabFormat(file_Cq);

        std::string nameE = "dumpDir_SER/_" + fname + "dump_E_.dat";
        chrono::ChStreamOutAsciiFile file_E(nameE.c_str());
        mdE.StreamOUTsparseMatlabFormat(file_E);

        std::string namef = "dumpDir_SER/_" + fname + "dump_f_.dat";
        chrono::ChStreamOutAsciiFile file_f(namef.c_str());
        mdf.StreamOUTdenseMatlabFormat(file_f);

        std::string nameb = "dumpDir_SER/_" + fname + "dump_b_.dat";
        chrono::ChStreamOutAsciiFile file_b(nameb.c_str());
        mdb.StreamOUTdenseMatlabFormat(file_b);

        std::string namer = "dumpDir_SER/_" + fname + "dump_fric_.dat";
        chrono::ChStreamOutAsciiFile file_fric(namer.c_str());
        mdfric.StreamOUTdenseMatlabFormat(file_fric);

        std::string namex = "dumpDir_SER/_" + fname + "dump_x_.dat";
        chrono::ChStreamOutAsciiFile file_x(namex.c_str());
        mx.StreamOUTdenseMatlabFormat(file_x);
    }

    return 0;
}
