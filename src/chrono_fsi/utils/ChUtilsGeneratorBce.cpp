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
// Author: Arman Pazouki
// =============================================================================
//
// Utility class for generating fluid markers.//
// =============================================================================
#include <fstream>  // std::ifstream
#include <sstream>  // std::stringstream

#include "chrono/core/ChMathematics.h"  // for CH_C_PI
#include "chrono_fea/ChElementCableANCF.h"
#include "chrono_fea/ChElementShellANCF.h"
#include "chrono_fea/ChNodeFEAxyzD.h"
#include "chrono_fsi/ChDeviceUtils.cuh"
#include "chrono_fsi/utils/ChUtilsGeneratorBce.h"

namespace chrono {
namespace fsi {
namespace utils {
// =============================================================================
void CreateBCE_On_Sphere(thrust::host_vector<Real4>& posRadBCE, Real rad, SimParams* paramsH) {
    Real spacing = paramsH->MULT_INITSPACE * paramsH->HSML;

    for (Real r = spacing; r < rad - paramsH->solidSurfaceAdjust; r += spacing) {
        Real deltaTeta = spacing / r;
        Real deltaPhi = deltaTeta;

        for (Real phi = .1 * deltaPhi; phi < chrono::CH_C_PI - .1 * deltaPhi; phi += deltaPhi) {
            for (Real teta = .1 * deltaTeta; teta < 2 * chrono::CH_C_PI - .1 * deltaTeta; teta += deltaTeta) {
                Real3 BCE_Pos_local = mR3(r * sin(phi) * cos(teta), r * sin(phi) * sin(teta), r * cos(phi));
                posRadBCE.push_back(mR4(BCE_Pos_local, spacing));
            }
        }
    }
}
// =============================================================================
void CreateBCE_On_surface_of_Sphere(thrust::host_vector<Real4>& posRadBCE, Real rad, Real kernel_h) {
    Real spacing = kernel_h;

    Real r = rad;
    Real deltaTeta = spacing / r;
    Real deltaPhi = deltaTeta;

    int numphi = std::floor(3.1415 * r / spacing);
    for (Real p = 0.0; p < numphi; p++) {
        Real phi = p * 3.1415 / numphi;
        int numTheta = std::floor(2 * 3.1415 * r * sin(phi) / spacing);
        for (Real t = 0.0; t < numTheta; t++) {
            Real teta = t * 2 * 3.1415 / numTheta;
            Real3 BCE_Pos_local = mR3(r * sin(phi) * cos(teta), r * sin(phi) * sin(teta), r * cos(phi));
            posRadBCE.push_back(mR4(BCE_Pos_local, spacing));
        }
    }
}
// =============================================================================

void CreateBCE_On_Cylinder(thrust::host_vector<Real4>& posRadBCE,
                           Real cyl_rad,
                           Real cyl_h,
                           SimParams* paramsH,
                           Real kernel_h) {
    int num_layers = floor(cyl_h / kernel_h);
    for (int si = 0; si < num_layers; si++) {
        Real s = -0.5 * cyl_h + (cyl_h / num_layers) * si;
        ///////////
        //        for (Real x = -cyl_rad; x <= cyl_rad; x += spacing) {
        //            for (Real y = -cyl_rad; y <= cyl_rad; y += spacing) {
        //                if (x * x + y * y <= cyl_rad * cyl_rad)
        //                    posRadBCE.push_back(mR4(x, s, y, spacing));
        //            }
        //        }
        Real3 centerPointLF = mR3(0, s, 0);
        printf("creating markers on the surface of the cylinder at layer at y=%f\n", s);
        Real numr = floor(cyl_rad / kernel_h);
        for (int ir = 0; ir < numr; ir++) {
            Real r = kernel_h + ir * cyl_rad / numr;
            //                Real deltaTeta = 2 * spacing / r;
            int numTheta = std::floor(2 * 3.1415 * r / kernel_h);
            for (Real t = 0.0; t < numTheta; t++) {
                Real teta = t * 2 * 3.1415 / numTheta;
                Real3 BCE_Pos_local = mR3(r * cos(teta), 0, r * sin(teta)) + centerPointLF;
                posRadBCE.push_back(mR4(BCE_Pos_local, kernel_h));
            }
        }
    }
}

// =============================================================================

void CreateBCE_On_surface_of_Cylinder(thrust::host_vector<Real4>& posRadBCE,
                                      thrust::host_vector<Real3>& normals,
                                      Real cyl_rad,
                                      Real cyl_h,
                                      Real spacing) {
    int num_layers = floor(cyl_h / spacing);
    for (int si = 0; si < num_layers; si++) {
        Real s = -0.5 * cyl_h + (cyl_h / num_layers) * si;
        ///////////
        //        for (Real x = -cyl_rad; x <= cyl_rad; x += spacing) {
        //            for (Real y = -cyl_rad; y <= cyl_rad; y += spacing) {
        //                if (x * x + y * y <= cyl_rad * cyl_rad)
        //                    if (abs(s + 0.5 * cyl_h) < spacing / 5 || (s - 0.5 * cyl_h) > 0 ||
        //                        abs(x * x + y * y - cyl_rad * cyl_rad) < spacing / 5) {
        //                        posRadBCE.push_back(mR4(x, s, y, spacing));
        //                    }
        //        }
        Real3 centerPointLF = mR3(0, s, 0);
        printf("creating markers on the surface of the cylinder at layer at y=%f\n", s);
        Real numr = floor(cyl_rad / spacing);
        for (int ir = 1; ir < numr; ir++) {
            Real r = spacing + ir * cyl_rad / numr;
            //                Real deltaTeta = 2 * spacing / r;
            int numTheta = std::floor(2 * 3.1415 * r / spacing);
            for (Real t = 0.0; t < numTheta; t++) {
                Real teta = t * 2 * 3.1415 / numTheta;
                Real3 BCE_Pos_local = mR3(r * cos(teta), 0, r * sin(teta)) + centerPointLF;
                if (si == 0 || si == num_layers - 1 || ir == numr - 1) {
                    posRadBCE.push_back(mR4(BCE_Pos_local, spacing));
                }
            }
        }
    }
}

// =============================================================================
// note, the function in the current implementation creates boundary BCE (zero
// velocity)
// x=1, y=2, z =3; therefore 12 means creating markers on the top surface
// parallel to xy plane,
// similarly -12 means bottom face paralel to xy. similarly 13, -13, 23, -23
void CreateBCE_On_Box(thrust::host_vector<Real4>& posRadBCE, const Real3& hsize, int face, SimParams* paramsH) {
    Real initSpace0 = paramsH->MULT_INITSPACE * paramsH->HSML;
    int nFX = ceil(hsize.x / (initSpace0));
    int nFY = ceil(hsize.y / (initSpace0));
    int nFZ = ceil(hsize.z / (initSpace0));

    Real initSpaceX = hsize.x / nFX;
    Real initSpaceY = hsize.y / nFY;
    Real initSpaceZ = hsize.z / nFZ;

    int2 iBound = mI2(-nFX, nFX);
    int2 jBound = mI2(-nFY, nFY);
    int2 kBound = mI2(-nFZ, nFZ);

    switch (face) {
        case 12:
            kBound = mI2(nFZ - paramsH->NUM_BOUNDARY_LAYERS + 1, nFZ);
            break;
        case -12:
            kBound = mI2(-nFZ, -nFZ + paramsH->NUM_BOUNDARY_LAYERS - 1);
            break;
        case 13:
            jBound = mI2(nFY - paramsH->NUM_BOUNDARY_LAYERS + 1, nFY);
            break;
        case -13:
            jBound = mI2(-nFY, -nFY + paramsH->NUM_BOUNDARY_LAYERS - 1);
            break;
        case 23:
            iBound = mI2(nFX - paramsH->NUM_BOUNDARY_LAYERS + 1, nFX);
            break;
        case -23:
            iBound = mI2(-nFX, -nFX + paramsH->NUM_BOUNDARY_LAYERS - 1);
            break;
        default:
            printf("wrong argument box bce initialization\n");
            break;
    }

    for (int i = iBound.x; i <= iBound.y; i++) {
        for (int j = jBound.x; j <= jBound.y; j++) {
            for (int k = kBound.x; k <= kBound.y; k++) {
                Real3 relMarkerPos = mR3(i * initSpaceX, j * initSpaceY, k * initSpaceZ);

                if ((relMarkerPos.x < paramsH->cMin.x || relMarkerPos.x > paramsH->cMax.x) ||
                    (relMarkerPos.y < paramsH->cMin.y || relMarkerPos.y > paramsH->cMax.y) ||
                    (relMarkerPos.z < paramsH->cMin.z || relMarkerPos.z > paramsH->cMax.z)) {
                    continue;
                }
                posRadBCE.push_back(mR4(relMarkerPos, initSpace0));
            }
        }
    }
}
// =============================================================================

void CreateBCE_On_shell(thrust::host_vector<Real4>& posRadBCE,
                        SimParams* paramsH,
                        std::shared_ptr<chrono::fea::ChElementShellANCF> shell,
                        bool multiLayer,
                        bool removeMiddleLayer,
                        int SIDE) {
    Real initSpace0 = paramsH->MULT_INITSPACE * paramsH->HSML;
    double dx = shell->GetLengthX() / 2 - initSpace0 / 2;
    double dy = shell->GetLengthY() / 2 - initSpace0 / 2;

    double nX = dx / (initSpace0)-floor(dx / (initSpace0));
    double nY = dy / (initSpace0)-floor(dy / (initSpace0));
    int nFX = floor(dx / (initSpace0));
    int nFY = floor(dy / (initSpace0));
    if (nX > 0.5)
        nFX++;
    if (nY > 0.5)
        nFY++;

    int nFZ = SIDE;

    Real initSpaceX = dx / nFX;
    Real initSpaceY = dy / nFY;
    Real initSpaceZ = paramsH->HSML * paramsH->MULT_INITSPACE_Shells;

    int2 iBound = mI2(-nFX, nFX);
    int2 jBound = mI2(-nFY, nFY);
    int2 kBound;
    // If multi-layer BCE is required
    if (SIDE > 0 && multiLayer)  // Do SIDE number layers in one side
        kBound = mI2(0, SIDE);
    else if (SIDE < 0 && multiLayer)  // Do SIDE number layers in the other side
        kBound = mI2(SIDE, 0);
    else if (SIDE == 0 && multiLayer)  // Do 1 layer on each side. Note that there would be 3 layers in total
        kBound = mI2(-1, 1);           // The middle layer would be on the shell
    else                               // IF you do not want multi-layer just use one layer on the shell
        kBound = mI2(0, 0);            // This will create some marker deficiency and reduce the accuracy but look nicer

    for (int i = iBound.x; i <= iBound.y; i++) {
        for (int j = jBound.x; j <= jBound.y; j++) {
            for (int k = kBound.x; k <= kBound.y; k++) {
                Real3 relMarkerPos = mR3(i * initSpaceX, j * initSpaceY, k);

                if (k == 0 && SIDE == 0 && multiLayer && removeMiddleLayer) {
                    // skip the middle layer for this specific case
                    paramsH->MULT_INITSPACE_Shells = 0.5;
                    continue;
                }

                posRadBCE.push_back(mR4(relMarkerPos, initSpace0));
            }
        }
    }
}  // =============================================================================

void CreateBCE_On_ChElementCableANCF(thrust::host_vector<Real4>& posRadBCE,
                                     SimParams* paramsH,
                                     std::shared_ptr<chrono::fea::ChElementCableANCF> cable,
                                     std::vector<int> remove,
                                     bool multiLayer,
                                     bool removeMiddleLayer,
                                     int SIDE) {
    Real initSpace0 = paramsH->MULT_INITSPACE * paramsH->HSML;

    double dx = (cable->GetNodeB()->GetX0() - cable->GetNodeA()->GetX0()).Length();
    double nX = dx / (initSpace0)-floor(dx / (initSpace0));
    int nFX = floor(dx / (initSpace0));
    if (nX > 0.5)
        nFX++;

    Real initSpaceX;
    if (nFX != 0)
        initSpaceX = dx / nFX;
    else
        initSpaceX = dx;

    //  initSpaceX = dx / nFX;

    Real initSpaceZ = initSpace0;
    int2 iBound = mI2(0, nFX);

    for (int i = iBound.x; i <= iBound.y; i++) {
        bool con1 = (remove[1] && (i == iBound.y));
        bool con2 = (remove[0] && (i == iBound.x));
        if (con1 || con2)
            continue;

        Real3 relMarkerPos;
        //    if (multiLayer && removeMiddleLayer) {
        //      // skip the middle layer for this specific case
        //      paramsH->MULT_INITSPACE_Shells = 0.5;
        //      initSpaceZ = paramsH->HSML * paramsH->MULT_INITSPACE_Shells;
        //    }

        if (multiLayer) {
            for (int j = 1; j <= SIDE; j++) {
                relMarkerPos = mR3(i * initSpaceX, j * initSpaceZ, 0);
                posRadBCE.push_back(mR4(relMarkerPos, initSpace0));
                relMarkerPos = mR3(i * initSpaceX, -j * initSpaceZ, 0);
                posRadBCE.push_back(mR4(relMarkerPos, initSpace0));
                relMarkerPos = mR3(i * initSpaceX, 0, j * initSpaceZ);
                posRadBCE.push_back(mR4(relMarkerPos, initSpace0));
                relMarkerPos = mR3(i * initSpaceX, 0, -j * initSpaceZ);
                posRadBCE.push_back(mR4(relMarkerPos, initSpace0));

                double CONST = sqrt(2) / 2;
                //        if (removeMiddleLayer) {
                //          relMarkerPos = mR3(i * initSpaceX, j * initSpaceZ * CONST, j * initSpaceZ * CONST);
                //          posRadBCE.push_back(relMarkerPos);
                //          relMarkerPos = mR3(i * initSpaceX, -j * initSpaceZ * CONST, j * initSpaceZ * CONST);
                //          posRadBCE.push_back(relMarkerPos);
                //          relMarkerPos = mR3(i * initSpaceX, -j * initSpaceZ * CONST, -j * initSpaceZ * CONST);
                //          posRadBCE.push_back(relMarkerPos);
                //          relMarkerPos = mR3(i * initSpaceX, j * initSpaceZ * CONST, -j * initSpaceZ * CONST);
                //          posRadBCE.push_back(relMarkerPos);
                //        }
            }
        }

        if (!removeMiddleLayer) {
            relMarkerPos = mR3(i * initSpaceX, 0, 0);
            posRadBCE.push_back(mR4(relMarkerPos, initSpace0));
        }
    }
}
// =============================================================================

void CreateBCE_On_ChElementShellANCF(thrust::host_vector<Real4>& posRadBCE,
                                     SimParams* paramsH,
                                     std::shared_ptr<chrono::fea::ChElementShellANCF> shell,
                                     std::vector<int> remove,
                                     bool multiLayer,
                                     bool removeMiddleLayer,
                                     int SIDE) {
    Real initSpace0 = paramsH->MULT_INITSPACE * paramsH->HSML;

    double dx = shell->GetLengthX() / 2;
    double dy = shell->GetLengthY() / 2;

    double nX = dx / (initSpace0)-floor(dx / (initSpace0));
    double nY = dy / (initSpace0)-floor(dy / (initSpace0));
    int nFX = floor(dx / (initSpace0));
    int nFY = floor(dy / (initSpace0));
    if (nX > 0.5)
        nFX++;
    if (nY > 0.5)
        nFY++;

    int nFZ = SIDE;

    Real initSpaceX = dx / nFX;
    Real initSpaceY = dy / nFY;
    Real initSpaceZ = paramsH->HSML * paramsH->MULT_INITSPACE_Shells;

    int2 iBound = mI2(-nFX, nFX);
    int2 jBound = mI2(-nFY, nFY);
    int2 kBound;
    // If multi-layer BCE is required
    if (SIDE > 0 && multiLayer)  // Do SIDE number layers in one side
        kBound = mI2(0, SIDE);
    else if (SIDE < 0 && multiLayer)  // Do SIDE number layers in the other side
        kBound = mI2(SIDE, 0);
    else if (SIDE == 0 && multiLayer)  // Do 1 layer on each side. Note that there would be 3 layers in total
        kBound = mI2(-1, 1);           // The middle layer would be on the shell
    else                               // IF you do not want multi-layer just use one layer on the shell
        kBound = mI2(0, 0);            // This will create some marker deficiency and reduce the accuracy but look nicer

    for (int k = kBound.x; k <= kBound.y; k++) {
        for (int j = jBound.x; j <= jBound.y; j++) {
            for (int i = iBound.x; i <= iBound.y; i++) {
                Real3 relMarkerPos = mR3(i * initSpaceX, j * initSpaceY, k);

                if (k == 0 && SIDE == 0 && multiLayer && removeMiddleLayer) {
                    // skip the middle layer for this specific case
                    printf(
                        "---------------paramsH->MULT_INITSPACE_Shells was changed in CreateBCE_On_Mesh to 0.5. "
                        "\n");
                    paramsH->MULT_INITSPACE_Shells = 0.5;
                    continue;
                }

                //        if (std::abs(i) <= 2 && std::abs(j) <= 2) {
                //          // skip the middle layer for this specific case
                //          continue;
                //        }

                // It has to skip puting BCE on the nodes if one of the following conditions is true
                bool con1 = (remove[0] && remove[1] && j == jBound.x);
                bool con2 = (remove[2] && remove[3] && j == jBound.y);
                bool con3 = (remove[1] && remove[2] && i == iBound.y);
                bool con4 = (remove[3] && remove[0] && i == iBound.x);

                if (con1 || con2 || con3 || con4)
                    continue;

                posRadBCE.push_back(mR4(relMarkerPos, initSpace0));
            }
        }
    }
}
// =============================================================================

void LoadBCE_fromFile(thrust::host_vector<Real4>& posRadBCE,  // do not set the
                                                              // size here since
                                                              // you are using
                                                              // push back later
                      std::string fileName) {
    std::string ddSt;
    char buff[256];
    int numBce = 0;
    const int cols = 3;
    std::cout << "  reading BCE data from: " << fileName << " ...\n";
    std::ifstream inMarker;
    inMarker.open(fileName);
    if (!inMarker) {
        std::cout << "   Error! Unable to open file: " << fileName << std::endl;
    }
    getline(inMarker, ddSt);
    Real q[cols];
    while (getline(inMarker, ddSt)) {
        std::stringstream linestream(ddSt);
        for (int i = 0; i < cols; i++) {
            linestream.getline(buff, 50, ',');
            q[i] = atof(buff);
        }
        posRadBCE.push_back(mR4(q[0], q[1], q[2], q[3]));
        numBce++;
    }

    std::cout << "  Loaded BCE data from: " << fileName << std::endl;
}

}  // namespace utils
}  // namespace fsi
}  // namespace chrono
