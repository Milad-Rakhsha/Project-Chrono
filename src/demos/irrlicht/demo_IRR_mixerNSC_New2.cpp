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

#include "chrono_parallel/physics/ChSystemParallel.h"

#include "chrono/ChConfig.h"
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

using namespace chrono;
using namespace chrono::collision;

int checknumcontact = 0;

// Custom contact container -- get access to the contact lists in the base
// class.
class MyContactContainer : public ChContactContainerNSC {
  public:
    std::vector<std::shared_ptr<ChBody>> body_list;
    MyContactContainer(std::vector<std::shared_ptr<ChBody>> bodyList) { body_list = bodyList; }
    MyContactContainer() {}
    // Traverse the list contactlist_6_6
    void DoStuffWithContainer(bool print, std::string fhead, int tspi) {
        int Nct = contactlist_6_6.size();
        chrono::ChMatrixDynamic<double> mCP(Nct, 6), mCF(Nct, 3), mCN(Nct, 3), mCdst(Nct, 1), mCrd(Nct, 7);
        chrono::ChMatrixDynamic<int> mIdC(Nct, 2);
        // Look at ChContactContainerNSC and there are other lists (than
        // contactlist_6_6) that you might want to go over
        auto iter = contactlist_6_6.begin();
        int num_contact = 0;
        while (iter != contactlist_6_6.end()) {
            ChContactable* objA = (*iter)->GetObjA();
            ChContactable* objB = (*iter)->GetObjB();
            ChVector<> p1 = (*iter)->GetContactP1();
            ChVector<> p2 = (*iter)->GetContactP2();
            ChVector<> F = (*iter)->GetContactForce();
            ChVector<> N = (*iter)->GetContactNormal();
            ChCoordsys<> CS = (*iter)->GetContactCoords();
            double CD = (*iter)->GetContactDistance();

            // Build rows of dynamic matrices
            mCP.SetElement(num_contact, 0, p1.x());
            mCP.SetElement(num_contact, 1, p1.y());
            mCP.SetElement(num_contact, 2, p1.z());
            mCP.SetElement(num_contact, 3, p2.x());
            mCP.SetElement(num_contact, 4, p2.y());
            mCP.SetElement(num_contact, 5, p2.z());
            mCF.SetElement(num_contact, 0, F.x());
            mCF.SetElement(num_contact, 1, F.y());
            mCF.SetElement(num_contact, 2, F.z());
            mCN.SetElement(num_contact, 0, N.x());
            mCN.SetElement(num_contact, 1, N.y());
            mCN.SetElement(num_contact, 2, N.z());
            mCdst.SetElement(num_contact, 0, CD);
            mCrd.SetElement(num_contact, 0, CS.pos.x());
            mCrd.SetElement(num_contact, 1, CS.pos.y());
            mCrd.SetElement(num_contact, 2, CS.pos.z());
            mCrd.SetElement(num_contact, 3, CS.rot.e0());
            mCrd.SetElement(num_contact, 4, CS.rot.e1());
            mCrd.SetElement(num_contact, 5, CS.rot.e2());
            mCrd.SetElement(num_contact, 6, CS.rot.e3());

            // Iterate rhrough the body list and figure out the indices of the bodies
            // for this contact
            if (print) {
                printf("===========\ncontact %d\n", num_contact);
            }
            for (int i = 0; i < body_list.size(); i++) {
                std::shared_ptr<ChBody> m_bd = body_list[i];
                if (objA == m_bd.get()) {
                    if (print) {
                        printf("body idx=%d\n", i);
                    }
                    mIdC.SetElement(num_contact, 0, i);
                } else if (objB == m_bd.get()) {
                    if (print) {
                        printf("body idx=%d\n", i);
                    }
                    mIdC.SetElement(num_contact, 1, i);
                }
            }

            // Here are some examples information about contact
            if (print) {
                printf("P1=[%f %f %f]\n", p1.x(), p1.y(), p1.z());
                printf("P2=[%f %f %f]\n", p2.x(), p2.y(), p2.z());
                printf("Contact Distance=%f\n\n", CD);
            }

            num_contact++;
            ++iter;
        }

        printf("num contacts in contactlist_6_6=%d", contactlist_6_6.size());
        // Dump files to dense matlab ascii format
        std::string nameCP = fhead + "dump_CP_" + std::to_string(tspi) + ".dat";
        chrono::ChStreamOutAsciiFile file_CP(nameCP.c_str());
        mCP.StreamOUTdenseMatlabFormat(file_CP);

        std::string nameCF = fhead + "dump_CF_" + std::to_string(tspi) + ".dat";
        chrono::ChStreamOutAsciiFile file_CF(nameCF.c_str());
        mCF.StreamOUTdenseMatlabFormat(file_CF);

        std::string nameCN = fhead + "dump_CN_" + std::to_string(tspi) + ".dat";
        chrono::ChStreamOutAsciiFile file_CN(nameCN.c_str());
        mCN.StreamOUTdenseMatlabFormat(file_CN);

        std::string nameCdst = fhead + "dump_Cdst_" + std::to_string(tspi) + ".dat";
        chrono::ChStreamOutAsciiFile file_Cdst(nameCdst.c_str());
        mCdst.StreamOUTdenseMatlabFormat(file_Cdst);

        std::string nameCrd = fhead + "dump_Crd_" + std::to_string(tspi) + ".dat";
        chrono::ChStreamOutAsciiFile file_Crd(nameCrd.c_str());
        mCrd.StreamOUTdenseMatlabFormat(file_Crd);

        std::string nameIdC = fhead + "dump_IdC_" + std::to_string(tspi) + ".dat";
        chrono::ChStreamOutAsciiFile file_IdC(nameIdC.c_str());
        mIdC.StreamOUTdenseMatlabFormat(file_IdC);
    }
};

// -----------------------------------------------------------------------------
// Callback class for contact reporting
// -----------------------------------------------------------------------------
class ContactReporter : public ChContactContainer::ReportContactCallback {
  public:
    std::vector<std::shared_ptr<ChBody>> body_list;
    ContactReporter(std::vector<std::shared_ptr<ChBody>> bodyList) { body_list = bodyList; }

  private:
    virtual bool OnReportContact(const ChVector<>& pA,
                                 const ChVector<>& pB,
                                 const ChMatrix33<>& plane_coord,
                                 const double& distance,
                                 const ChVector<>& cforce,
                                 const ChVector<>& ctorque,
                                 ChContactable* modA,
                                 ChContactable* modB) override {
        printf("Contact %d: \n", checknumcontact);
        for (int i = 0; i < body_list.size(); i++) {
            std::shared_ptr<ChBody> m_bd = body_list[i];
            if (modA == m_bd.get()) {
                printf("  %6.3f  %6.3f  %6.3f with body %d\n", pA.x(), pA.y(), pA.z(), i);
            } else if (modB == m_bd.get()) {
                printf("  %6.3f  %6.3f  %6.3f with body %d\n", pB.x(), pB.y(), pB.z(), i);
            }
        }
        checknumcontact++;
        printf("  %6.3f\n", (pA - pB).Length());

        return true;
    }
};

// -----------------------------------------------------------------------------
// Create a bin consisting of five boxes attached to the ground and a mixer
// blade attached through a revolute joint to ground. The mixer is constrained
// to rotate at constant angular velocity.
// -----------------------------------------------------------------------------
void AddContainer(ChSystemParallelNSC* sys, double wd) {
    // IDs for the two bodies
    int binId = -200;
    int mixerId = -201;

    // Create a common material
    auto mat = std::make_shared<ChMaterialSurfaceNSC>();
    mat->SetFriction(0.4f);

    // Create the containing bin (2 x 2 x 1)
    auto bin = std::make_shared<ChBody>(std::make_shared<ChCollisionModelParallel>());
    bin->SetMaterialSurface(mat);
    bin->SetIdentifier(binId);
    bin->SetMass(1);
    bin->SetPos(ChVector<>(0, 0, 0));
    bin->SetRot(ChQuaternion<>(1, 0, 0, 0));
    bin->SetCollide(true);
    bin->SetBodyFixed(true);

    ChVector<> hdim(wd, wd, wd);
    double hthick = 0.5;

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
    auto mixer = std::make_shared<ChBody>(std::make_shared<ChCollisionModelParallel>());
    mixer->SetMaterialSurface(mat);
    mixer->SetIdentifier(mixerId);
    mixer->SetMass(10.0);
    mixer->SetInertiaXX(ChVector<>(50, 50, 50));
    mixer->SetPos(ChVector<>(0, 0, 0.205));
    mixer->SetBodyFixed(false);
    mixer->SetCollide(true);

    ChVector<> hsize(0.8 * wd, 0.1, 0.2);

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
void AddFallingBalls(ChSystemParallel* sys, int nbt) {
    // Common material
    auto ballMat = std::make_shared<ChMaterialSurfaceNSC>();
    ballMat->SetFriction(0.4f);

    // Create the falling balls
    int ballId = 0;
    double mass = 1;
    double radius = 0.3;
    ChVector<> inertia = (2.0 / 5.0) * mass * radius * radius * ChVector<>(1, 1, 1);

    for (int ix = -nbt; ix < nbt + 1; ix++) {
        for (int iy = -nbt; iy < nbt + 1; iy++) {
            for (int iz = -nbt; iz < nbt + 1; iz++) {
                ChVector<> pos(0.4 * ix, 0.4 * iy, 0.4 * (iz + nbt));

                auto ball = std::make_shared<ChBody>(std::make_shared<ChCollisionModelParallel>());
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
}

// -----------------------------------------------------------------------------
// Create the system, specify simulation parameters, and run simulation loop.
// -----------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    GetLog() << "Copyright (c) 2017 projectchrono.org\nChrono version: " << CHRONO_VERSION << "\n\n";

    //(2) Arguments: int nbt (number of bodies of each type), double wd = 5.7; (width of box)
    int nbt = atoi(argv[1]);
    double wd = strtod(argv[2], NULL);

    std::cout << "Box sedimentation demo, Number of bodies = " << pow(nbt, 3) << ", width = " << wd << "\n";

    int threads = 1;

    // Simulation parameters
    // ---------------------

    double gravity = 9.81;
    double time_step = 1e-3;
    double time_end = 1;

    double out_fps = 50;

    uint max_iteration = 30;
    real tolerance = 1e-3;

    // Create system
    // -------------

    ChSystemParallelNSC msystem;

    // Set number of threads.
    int max_threads = CHOMPfunctions::GetNumProcs();
    if (threads > max_threads)
        threads = max_threads;
    msystem.SetParallelThreadNumber(threads);
    CHOMPfunctions::SetNumThreads(threads);

    std::cout << "max num threads" << threads << "\n";

    // Set gravitational acceleration
    msystem.Set_G_acc(ChVector<>(0, 0, -gravity));

    // Set solver parameters
    msystem.GetSettings()->solver.solver_mode = SolverMode::SLIDING;
    msystem.GetSettings()->solver.max_iteration_normal = max_iteration / 3;
    msystem.GetSettings()->solver.max_iteration_sliding = max_iteration / 3;
    msystem.GetSettings()->solver.max_iteration_spinning = 0;
    msystem.GetSettings()->solver.max_iteration_bilateral = max_iteration / 3;
    msystem.GetSettings()->solver.tolerance = tolerance;
    msystem.GetSettings()->solver.alpha = 0;
    msystem.GetSettings()->solver.contact_recovery_speed = 10000;
    msystem.ChangeSolverType(SolverType::APGD);
    msystem.GetSettings()->collision.narrowphase_algorithm = NarrowPhaseType::NARROWPHASE_HYBRID_MPR;

    msystem.GetSettings()->collision.collision_envelope = 0.01;
    msystem.GetSettings()->collision.bins_per_axis = vec3(10, 10, 10);

    // Create the fixed and moving bodies
    // ----------------------------------

    AddContainer(&msystem, wd);
    AddFallingBalls(&msystem, nbt);

    // Perform the simulation
    // ----------------------

#ifdef CHRONO_OPENGL
    opengl::ChOpenGLWindow& gl_window = opengl::ChOpenGLWindow::getInstance();
    gl_window.Initialize(1280, 720, "mixerNSC", &msystem);
    gl_window.SetCamera(ChVector<>(0, -3, 2), ChVector<>(0, 0, 0), ChVector<>(0, 0, 1));
    gl_window.SetRenderMode(opengl::WIREFRAME);

    // Uncomment the following two lines for the OpenGL manager to automatically
    // run the simulation in an infinite loop.
    // gl_window.StartDrawLoop(time_step);
    // return 0;

    while (true) {
        if (gl_window.Active()) {
            gl_window.DoStepDynamics(time_step);
            gl_window.Render();
        } else {
            break;
        }
    }
#else
    std::vector<std::shared_ptr<ChBody>> blist = *(msystem.Get_bodylist());
    auto container = std::make_shared<MyContactContainer>(blist);
    //    msystem.SetContactContainer(container);

    std::string fname = "sedimentationtest_sph_par_" + std::to_string(nbt) + "_";

    // Run simulation for specified time
    int num_steps = (int)std::ceil(time_end / time_step);
    double time = 0;
    for (int i = 0; i < num_steps; i++) {
        msystem.DoStepDynamics(time_step);
        time += time_step;

        std::shared_ptr<ChSystemDescriptor> idesc = msystem.GetSystemDescriptor();
        std::shared_ptr<ChContactContainerNSC> iconcon =
            std::dynamic_pointer_cast<ChContactContainerNSC>(msystem.GetContactContainer());

        std::cout << "simulation time" << time << ", iteration " << i << "\n";

#pragma omp master
        {
            try {
                int numContact = iconcon->GetNcontacts();
                int numBodies = msystem.Get_bodylist()->size();

                container->DoStuffWithContainer(false, fname, i);

                ContactReporter creporter(blist);
                printf("==== body %d info:\n", i);
                printf("contact points\n");
                msystem.GetContactContainer()->ReportAllContacts(&creporter);

                std::vector<ChVector<>> v_k(numBodies), w_k(numBodies);
                chrono::ChMatrixDynamic<double> mdV(numBodies, 6);
                // To save velocities:
                for (int i = 0; i < numBodies; i++) {
                    auto this_body = std::make_shared<ChBody>();
                    this_body = blist[i];

                    ChVector<> Vi = this_body->GetPos_dt();
                    v_k.push_back(Vi);
                    ChVector<> Wi = this_body->GetWvel_loc();
                    w_k.push_back(Wi);

                    // translational velocities
                    mdV.SetElement(i, 0, Vi.x());
                    mdV.SetElement(i, 1, Vi.y());
                    mdV.SetElement(i, 2, Vi.z());
                    mdV.SetElement(i, 3, Wi.x());
                    mdV.SetElement(i, 4, Wi.y());
                    mdV.SetElement(i, 5, Wi.z());
                    // printf("vel[%d]= %6.3f  %6.3f  %6.3f\n", i, Vi.x(), Vi.y(), Vi.z());
                }
                std::cout << "numContact : " << numContact << "\n";
                std::cout << "numBodies : " << numBodies << "\n";
                std::cout << "checknumcontact : " << checknumcontact << "\n";

                std::cout << "\n Dumping Newton matrix components \n";

                chrono::ChLinkedListMatrix mdM;
                chrono::ChLinkedListMatrix mdCq;
                chrono::ChLinkedListMatrix mdE;
                chrono::ChMatrixDynamic<double> mdf;
                chrono::ChMatrixDynamic<double> mdb;
                chrono::ChMatrixDynamic<double> mdfric;
                chrono::ChMatrixDynamic<double> mx;  // constraint data (q,l)
                printf("..........\n");

                idesc->ConvertToMatrixForm(&mdCq, &mdM, &mdE, &mdf, &mdb, &mdfric);
                idesc->FromVariablesToVector(mx);

                std::string nameM = fname + "dump_M_" + std::to_string(i) + ".dat";
                chrono::ChStreamOutAsciiFile file_M(nameM.c_str());
                mdM.StreamOUTsparseMatlabFormat(file_M);

                std::string nameC = fname + "dump_Cq_" + std::to_string(i) + ".dat";
                chrono::ChStreamOutAsciiFile file_Cq(nameC.c_str());
                mdCq.StreamOUTsparseMatlabFormat(file_Cq);

                std::string nameE = fname + "dump_E_" + std::to_string(i) + ".dat";
                chrono::ChStreamOutAsciiFile file_E(nameE.c_str());
                mdE.StreamOUTsparseMatlabFormat(file_E);

                std::string namef = fname + "dump_f_" + std::to_string(i) + ".dat";
                chrono::ChStreamOutAsciiFile file_f(namef.c_str());
                mdf.StreamOUTdenseMatlabFormat(file_f);

                std::string nameb = fname + "dump_b_" + std::to_string(i) + ".dat";
                chrono::ChStreamOutAsciiFile file_b(nameb.c_str());
                mdb.StreamOUTdenseMatlabFormat(file_b);

                std::string namer = fname + "dump_fric_" + std::to_string(i) + ".dat";
                chrono::ChStreamOutAsciiFile file_fric(namer.c_str());
                mdfric.StreamOUTdenseMatlabFormat(file_fric);

                std::string namex = fname + "dump_x_" + std::to_string(i) + ".dat";
                chrono::ChStreamOutAsciiFile file_x(namex.c_str());
                mx.StreamOUTdenseMatlabFormat(file_x);

                std::string nameV = fname + "dump_V_" + std::to_string(i) + ".dat";
                chrono::ChStreamOutAsciiFile file_V(nameV.c_str());
                mdV.StreamOUTdenseMatlabFormat(file_V);

            }

            catch (chrono::ChException myexc) {
                chrono::GetLog() << myexc.what();
            }
        }
    }
#endif

    return 0;
}
