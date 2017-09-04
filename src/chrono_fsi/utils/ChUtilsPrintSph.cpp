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
// Utility function to print the save fluid, bce, and boundary data into file
// =============================================================================
#include "chrono_fsi/utils/ChUtilsPrintSph.h"
#include "chrono_fsi/ChDeviceUtils.cuh"
#include "chrono_fsi/ChParams.cuh"
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cstring>
#include <thrust/reduce.h>

namespace chrono {
namespace fsi {
namespace utils {
//*******************************************************************************************************************************
void PrintToFile(const thrust::device_vector<Real3>& posRadD,
                 const thrust::device_vector<Real3>& velMasD,
                 const thrust::device_vector<Real4>& rhoPresMuD,
                 const thrust::host_vector<int4>& referenceArray,
                 const thrust::host_vector<int4>& referenceArrayFEA,
                 const std::string& out_dir,
                 bool printToParaview) {
    thrust::host_vector<Real3> posRadH = posRadD;
    thrust::host_vector<Real3> velMasH = velMasD;
    thrust::host_vector<Real4> rhoPresMuH = rhoPresMuD;

    char fileCounter[5];
    static int dumNumChar = -1;
    dumNumChar++;
    sprintf(fileCounter, "%d", dumNumChar);

    //*****************************************************
    const std::string nameFluid = out_dir + std::string("/fluid") + std::string(fileCounter) + std::string(".csv");

    std::ofstream fileNameFluidParticles;
    fileNameFluidParticles.open(nameFluid);
    std::stringstream ssFluidParticles;
    if (printToParaview)
        ssFluidParticles << "x,y,z,v_x,v_y,v_z,|U|,rho(rpx),p(rpy),mu(rpz),type(rpw)\n";

    for (int i = referenceArray[0].x; i < referenceArray[0].y; i++) {
        Real3 pos = posRadH[i];
        Real3 vel = velMasH[i];
        Real4 rP = rhoPresMuH[i];
        Real velMag = length(vel);
        ssFluidParticles << pos.x << ", " << pos.y << ", " << pos.z << ", " << vel.x << ", " << vel.y << ", " << vel.z
                         << ", " << velMag << ", " << rP.x << ", " << rP.y << ", " << rP.z << ", " << rP.w << std::endl;
    }
    fileNameFluidParticles << ssFluidParticles.str();
    fileNameFluidParticles.close();
    //*****************************************************
    const std::string nameFluidBoundaries =
        out_dir + std::string("/fluid_boundary") + std::string(fileCounter) + std::string(".csv");

    std::ofstream fileNameFluidBoundaries;
    fileNameFluidBoundaries.open(nameFluidBoundaries);
    std::stringstream ssFluidBoundaryParticles;
    if (printToParaview)
        ssFluidBoundaryParticles << "x,y,z,v_x,v_y,v_z,|U|,rho(rpx),p(rpy),mu(rpz),type(rpw)\n";

    //		ssFluidBoundaryParticles.precision(20);
    for (int i = referenceArray[0].x; i < referenceArray[1].y; i++) {
        Real3 pos = posRadH[i];
        Real3 vel = velMasH[i];
        Real4 rP = rhoPresMuH[i];
        Real velMag = length(vel);
        ssFluidBoundaryParticles << pos.x << ", " << pos.y << ", " << pos.z << ", " << vel.x << ", " << vel.y << ", "
                                 << vel.z << ", " << velMag << ", " << rP.x << ", " << rP.y << ", " << rP.z << ", "
                                 << rP.w << std::endl;
    }
    fileNameFluidBoundaries << ssFluidBoundaryParticles.str();
    fileNameFluidBoundaries.close();
    //*****************************************************
    const std::string nameBCE = out_dir + std::string("/BCE") + std::string(fileCounter) + std::string(".csv");

    std::ofstream fileNameBCE;
    fileNameBCE.open(nameBCE);
    std::stringstream ssBCE;
    //		ssFluidBoundaryParticles.precision(20);
    if (printToParaview)
        ssBCE << "x,y,z,vx,vy,vz,U,rpx,rpy,rpz,rpw\n";

    int refSize = referenceArray.size();
    if (refSize > 2) {
        for (int i = referenceArray[2].x; i < referenceArray[refSize - 1].y; i++) {
            Real3 pos = posRadH[i];
            Real3 vel = velMasH[i];
            Real4 rP = rhoPresMuH[i];
            Real velMag = length(vel);

            ssBCE << pos.x << ", " << pos.y << ", " << pos.z << ", " << vel.x << ", " << vel.y << ", " << vel.z << ", "
                  << velMag << ", " << rP.x << ", " << rP.y << ", " << rP.z << ", " << rP.w << std::endl;
        }
    }
    fileNameBCE << ssBCE.str();
    fileNameBCE.close();
    //*****************************************************
    const std::string nameBCE_Flex =
        out_dir + std::string("/BCE_Flex") + std::string(fileCounter) + std::string(".csv");

    std::ofstream fileNameBCE_Flex;
    fileNameBCE_Flex.open(nameBCE_Flex);
    std::stringstream ssBCE_Flex;
    //		ssFluidBoundaryParticles.precision(20);

    int refSize_Flex = referenceArrayFEA.size();

    if (refSize_Flex > 0) {
        if (printToParaview)
            ssBCE_Flex << "x,y,z,v_x,v_y,v_z,|U|,rho(rpx),p(rpy),mu(rpz),type(rpw)\n";
        for (int i = referenceArrayFEA[0].x; i < referenceArrayFEA[refSize_Flex - 1].y; i++) {
            Real3 pos = posRadH[i];
            Real3 vel = velMasH[i];
            Real4 rP = rhoPresMuH[i];
            Real velMag = length(vel);

            ssBCE_Flex << pos.x << ", " << pos.y << ", " << pos.z << ", " << vel.x << ", " << vel.y << ", " << vel.z
                       << ", " << velMag << ", " << rP.x << ", " << rP.y << ", " << rP.z << ", " << rP.w << std::endl;
        }
    }
    fileNameBCE_Flex << ssBCE_Flex.str();
    fileNameBCE_Flex.close();
    //*****************************************************
    posRadH.clear();
    velMasH.clear();
    rhoPresMuH.clear();
}

}  // end namespace utils
}  // end namespace fsi
}  // end namespace chrono
