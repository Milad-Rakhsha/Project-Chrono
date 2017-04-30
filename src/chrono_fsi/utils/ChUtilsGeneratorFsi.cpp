// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2014 projectchrono.org
// All right reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Author: Arman Pazouki
// =============================================================================
//
// Utility class for generating BCE markers.//
// =============================================================================

#include "chrono/utils/ChUtilsCreators.h"
#include "chrono/utils/ChUtilsGenerators.h"
#include "chrono/utils/ChUtilsGeometry.h"

#include "chrono_fsi/ChDeviceUtils.cuh"
#include "chrono_fsi/ChFsiTypeConvert.h"
#include "chrono_fsi/custom_math.h"
#include "chrono_fsi/utils/ChUtilsGeneratorFsi.h"
#include "chrono_parallel/physics/ChSystemParallel.h"

#include "chrono_fea/ChElementCableANCF.h"
#include "chrono_fea/ChElementShellANCF.h"
#include "chrono_fea/ChMesh.h"

namespace chrono {
namespace fsi {
namespace utils {

// =============================================================================
// left off here
// TransformToCOG
// This utility function converts a given position and orientation, specified
// with respect to a body's reference frame, into a frame defined with respect
// to the body's centroidal frame.  Note that by default, a body's reference
// frame is the centroidal frame. This is not true for a ChBodyAuxRef.
void TransformBceFrameToCOG(chrono::ChBody* body,
                            const chrono::ChVector<>& pos,
                            const chrono::ChMatrix33<>& rot,
                            chrono::ChFrame<>& frame) {
  frame = chrono::ChFrame<>(pos, rot);
  if (chrono::ChBodyAuxRef* body_ar = dynamic_cast<chrono::ChBodyAuxRef*>(body)) {
    frame = frame >> body_ar->GetFrame_REF_to_COG();
  }
}

chrono::ChVector<> TransformBCEToCOG(chrono::ChBody* body, const chrono::ChVector<>& pos) {
  chrono::ChFrame<> frame;
  TransformBceFrameToCOG(body, pos, chrono::QUNIT, frame);
  return frame.GetPos();
}

chrono::ChVector<> TransformBCEToCOG(chrono::ChBody* body, const Real3& pos3) {
  chrono::ChVector<> pos = ChFsiTypeConvert::Real3ToChVector(pos3);
  return TransformBCEToCOG(body, pos);
}
// =============================================================================
void CreateBceGlobalMarkersFromBceLocalPos(ChFsiDataManager* fsiData,
                                           SimParams* paramsH,
                                           const thrust::host_vector<Real3>& posRadBCE,
                                           std::shared_ptr<chrono::ChBody> body,
                                           chrono::ChVector<> collisionShapeRelativePos,
                                           chrono::ChQuaternion<> collisionShapeRelativeRot,
                                           bool isSolid) {
  if (fsiData->fsiGeneralData.referenceArray.size() < 1) {
    printf(
        "\n\n\n\n Error! fluid need to be initialized before boundary. "
        "Reference array should have two "
        "components \n\n\n\n");
    std::cin.get();
  }
  ::int4 refSize4 = fsiData->fsiGeneralData.referenceArray[fsiData->fsiGeneralData.referenceArray.size() - 1];
  int type = 0;
  if (isSolid) {
    type = refSize4.w + 1;
  }
  if (type < 0) {
    printf(
        "\n\n\n\n Error! reference array type is not correct. It does not "
        "denote boundary or rigid \n\n\n\n");
    std::cin.get();
  } else if (type > 0 && (fsiData->fsiGeneralData.referenceArray.size() - 1 != type)) {
    printf("\n\n\n\n Error! reference array size does not match type \n\n\n\n");
    std::cin.get();
  }

  //#pragma omp parallel for  // it is very wrong to do it in parallel. race
  // condition will occur
  for (int i = 0; i < posRadBCE.size(); i++) {
    chrono::ChVector<> posLoc_collisionShape = ChFsiTypeConvert::Real3ToChVector(posRadBCE[i]);
    chrono::ChVector<> posLoc_body = chrono::ChTransform<>::TransformLocalToParent(
        posLoc_collisionShape, collisionShapeRelativePos, collisionShapeRelativeRot);
    chrono::ChVector<> posLoc_COG = TransformBCEToCOG(body.get(), posLoc_body);
    chrono::ChVector<> posGlob =
        chrono::ChTransform<>::TransformLocalToParent(posLoc_COG, body->GetPos(), body->GetRot());
    fsiData->sphMarkersH.posRadH.push_back(ChFsiTypeConvert::ChVectorToReal3(posGlob));

    chrono::ChVector<> vAbs = body->PointSpeedLocalToParent(posLoc_COG);
    Real3 v3 = ChFsiTypeConvert::ChVectorToReal3(vAbs);
    fsiData->sphMarkersH.velMasH.push_back(v3);

    fsiData->sphMarkersH.rhoPresMuH.push_back(mR4(paramsH->rho0, paramsH->BASEPRES, paramsH->mu0, type));
  }

  // ------------------------
  // Modify number of objects
  // ------------------------

  int numBce = posRadBCE.size();
  fsiData->numObjects.numAllMarkers += numBce;
  if (type == 0) {
    fsiData->numObjects.numBoundaryMarkers += numBce;
    if (fsiData->fsiGeneralData.referenceArray.size() == 1) {
      fsiData->fsiGeneralData.referenceArray.push_back(mI4(refSize4.y, refSize4.y + numBce, 0, 0));
    } else if (fsiData->fsiGeneralData.referenceArray.size() == 2) {
      refSize4.y = refSize4.y + numBce;
      fsiData->fsiGeneralData.referenceArray[1] = refSize4;
    } else {
      printf(
          "Error! reference array size is greater than 2 while marker type "
          "is 0 \n\n");
      std::cin.get();
    }
  } else {
    if (fsiData->fsiGeneralData.referenceArray.size() < 2) {
      printf(
          "Error! Boundary markers are not initialized while trying to "
          "initialize rigid marker!\n\n");
      std::cin.get();
    }
    fsiData->numObjects.numRigid_SphMarkers += numBce;
    fsiData->numObjects.numRigidBodies += 1;
    fsiData->numObjects.startRigidMarkers = fsiData->fsiGeneralData.referenceArray[1].y;
    fsiData->fsiGeneralData.referenceArray.push_back(mI4(refSize4.y, refSize4.y + numBce, type, 1));
    if (fsiData->numObjects.numRigidBodies != fsiData->fsiGeneralData.referenceArray.size() - 2) {
      printf("Error! num rigid bodies does not match reference array size!\n\n");
      std::cin.get();
    }
  }

  //	SetNumObjects(numObjects, fsiGeneralData.referenceArray, numAllMarkers);
}
// =============================================================================

void CreateBceGlobalMarkersFromBceLocalPos_CableANCF(ChFsiDataManager* fsiData,
                                                     SimParams* paramsH,
                                                     const thrust::host_vector<Real3>& posRadBCE,
                                                     std::shared_ptr<chrono::fea::ChElementCableANCF> cable) {
  int type = 2;

  chrono::ChMatrixNM<double, 4, 1> N;
  double dx = (cable->GetNodeB()->GetX0() - cable->GetNodeA()->GetX0()).Length();

  chrono::ChVector<> Element_Axis = (cable->GetNodeB()->GetX0() - cable->GetNodeA()->GetX0()).GetNormalized();
  chrono::ChVector<> Old_axis = ChVector<>(1, 0, 0);
  chrono::ChQuaternion<double> Rotation = (Q_from_Vect_to_Vect(Old_axis, Element_Axis));
  Rotation.Normalize();
  chrono::ChVector<> new_y_axis = Rotation.Rotate(ChVector<>(0, 1, 0));
  chrono::ChVector<> new_z_axis = Rotation.Rotate(ChVector<>(0, 0, 1));
  //  printf(" Rotation Q for this element is = (%f,%f,%f,%f)\n", Rotation.e0, Rotation.e1, Rotation.e2, Rotation.e3);
  //  printf(" new_x_axis element is = (%f,%f,%f)\n", Element_Axis.x, Element_Axis.y, Element_Axis.z);
  //  printf(" new_y_axis element is = (%f,%f,%f)\n", new_y_axis.x, new_y_axis.y, new_y_axis.z);
  //  printf(" new_z_axis element is = (%f,%f,%f)\n", new_z_axis.x, new_z_axis.y, new_z_axis.z);

  chrono::ChVector<> physic_to_natural(1 / dx, 1, 1);
  chrono::ChVector<> nAp = cable->GetNodeA()->GetPos();
  chrono::ChVector<> nBp = cable->GetNodeB()->GetPos();

  chrono::ChVector<> nAv = cable->GetNodeA()->GetPos_dt();
  chrono::ChVector<> nBv = cable->GetNodeB()->GetPos_dt();

  chrono::ChVector<> nAa = cable->GetNodeA()->GetPos_dtdt();
  chrono::ChVector<> nBa = cable->GetNodeB()->GetPos_dtdt();

  int posRadSizeModified = 0;

  printf(" posRadBCE.size()= :%d\n", posRadBCE.size());
  for (int i = 0; i < posRadBCE.size(); i++) {
    //    chrono::ChVector<> posGlob =
    chrono::ChVector<> pos_physical = ChFsiTypeConvert::Real3ToChVector(posRadBCE[i]);
    chrono::ChVector<> pos_natural = pos_physical * physic_to_natural;
    //    printf(" physic_to_natural is = (%f,%f,%f)\n", physic_to_natural.x, physic_to_natural.y, physic_to_natural.z);
    //    printf(" pos_physical is = (%f,%f,%f)\n", pos_physical.x, pos_physical.y, pos_physical.z);
    //    printf(" pos_natural is = (%f,%f,%f)\n", pos_natural.x, pos_natural.y, pos_natural.z);

    cable->ShapeFunctions(N, pos_natural.x);
    chrono::ChVector<> x_dir = (nBp - nAp);
    chrono::ChVector<> Normal;
    //    printf(" N0 =%f, nAp.z= %f, N2=%f, nAp.z=%f\n", N(0), nAp.z, N(2), nBp.z);

    chrono::ChVector<> Correct_Pos =
        N(0) * nAp + N(2) * nBp + new_y_axis * pos_physical.y + new_z_axis * pos_physical.z;

    if ((Correct_Pos.x < paramsH->cMin.x || Correct_Pos.x > paramsH->cMax.x) ||
        (Correct_Pos.y < paramsH->cMin.y || Correct_Pos.y > paramsH->cMax.y) ||
        (Correct_Pos.z < paramsH->cMin.z || Correct_Pos.z > paramsH->cMax.z)) {
      continue;
    }
    //    printf("fsiData->sphMarkersH.posRadH.push_back :%f,%f,%f\n", Correct_Pos.x, Correct_Pos.y, Correct_Pos.z);

    bool addthis = true;
    for (int p = 0; p < fsiData->sphMarkersH.posRadH.size() - 1; p++) {
      if (length(fsiData->sphMarkersH.posRadH[p] - ChFsiTypeConvert::ChVectorToReal3(Correct_Pos)) < 1e-8 &&
          fsiData->sphMarkersH.rhoPresMuH[p].w != -1) {
        addthis = false;
        //        printf("remove this particle %f,%f,%f because of its overlap with a particle at %f,%f,%f\n",
        //               fsiData->sphMarkersH.posRadH[p].x, fsiData->sphMarkersH.posRadH[p].y,
        //               fsiData->sphMarkersH.posRadH[p].z,
        //               Correct_Pos.x, Correct_Pos.y, Correct_Pos.z);

        break;
      }
    }

    // THIS is hardcoding for now

    std::vector<double> box;
    box.resize(6);
    box[0] = -0.0045;
    box[1] = 0.0045;
    box[2] = -0.0045;
    box[3] = 0.0045;
    box[4] = -0.005;
    box[5] = 0.005;
    bool insideBox = true;
    insideBox = Correct_Pos.x > box[0] && Correct_Pos.x < box[1] && Correct_Pos.y > box[2] && Correct_Pos.y < box[3] &&
                Correct_Pos.z > box[4] && Correct_Pos.z < box[5];

    if (addthis && insideBox) {
      fsiData->sphMarkersH.posRadH.push_back(ChFsiTypeConvert::ChVectorToReal3(Correct_Pos));
      chrono::ChVector<> Correct_Vel = N(0) * nAv + N(2) * nBv;
      Real3 v3 = ChFsiTypeConvert::ChVectorToReal3(Correct_Vel);
      fsiData->sphMarkersH.velMasH.push_back(v3);
      fsiData->sphMarkersH.rhoPresMuH.push_back(mR4(paramsH->rho0, paramsH->BASEPRES, paramsH->mu0, type));
      posRadSizeModified++;
    }
  }

  // ------------------------
  // Modify number of objects
  // ------------------------
  int numObjects = fsiData->fsiGeneralData.referenceArray.size();
  int numBce = posRadSizeModified;
  fsiData->numObjects.numAllMarkers += numBce;

  int numRigid = fsiData->numObjects.numRigidBodies;
  fsiData->numObjects.numFlex_SphMarkers += numBce;
  fsiData->numObjects.numFlexBodies1D += 1;
  fsiData->numObjects.startFlexMarkers = fsiData->fsiGeneralData.referenceArray[numRigid + 1].y;
  int start_flex = fsiData->numObjects.startFlexMarkers;

  int4 last = fsiData->fsiGeneralData.referenceArray[fsiData->fsiGeneralData.referenceArray.size() - 1];
  fsiData->fsiGeneralData.referenceArray.push_back(
      mI4(last.y, last.y + numBce, type, fsiData->numObjects.numFlexBodies1D));  // 2: for Shell

  fsiData->fsiGeneralData.referenceArray_FEA.push_back(
      mI4(last.y, last.y + numBce, type, fsiData->numObjects.numFlexBodies1D));  // 2: for Shell

  printf(" push_back Index %d. ", fsiData->fsiGeneralData.referenceArray.size() - 1);
  int4 test = fsiData->fsiGeneralData.referenceArray[fsiData->fsiGeneralData.referenceArray.size() - 1];
  printf(" x=%d, y=%d, z=%d, w=%d\n", test.x, test.y, test.z, test.w);

  if (fsiData->numObjects.numFlexBodies1D !=
      fsiData->fsiGeneralData.referenceArray.size() - 2 - fsiData->numObjects.numRigidBodies) {
    printf("Error! num rigid Flexible does not match reference array size!\n\n");
    std::cin.get();
  }
  numObjects = fsiData->fsiGeneralData.referenceArray.size();
  printf("numObjects : %d\n ", numObjects);
  printf("numObjects.startFlexMarkers  : %d\n ", fsiData->numObjects.startFlexMarkers);
}
// =============================================================================

void CreateBceGlobalMarkersFromBceLocalPos_ShellANCF(ChFsiDataManager* fsiData,
                                                     SimParams* paramsH,
                                                     const thrust::host_vector<Real3>& posRadBCE,
                                                     std::shared_ptr<chrono::fea::ChElementShellANCF> shell) {
  int type = 3;

  chrono::ChMatrixNM<double, 8, 1> N;
  int posRadSizeModified = 0;

  Real dx = shell->GetLengthX();
  Real dy = shell->GetLengthY();
  chrono::ChVector<> physic_to_natural(2 / dx, 2 / dy, 1);
  chrono::ChVector<> nAp = shell->GetNodeA()->GetPos();
  chrono::ChVector<> nBp = shell->GetNodeB()->GetPos();
  chrono::ChVector<> nCp = shell->GetNodeC()->GetPos();
  chrono::ChVector<> nDp = shell->GetNodeD()->GetPos();

  chrono::ChVector<> nAv = shell->GetNodeA()->GetPos_dt();
  chrono::ChVector<> nBv = shell->GetNodeB()->GetPos_dt();
  chrono::ChVector<> nCv = shell->GetNodeC()->GetPos_dt();
  chrono::ChVector<> nDv = shell->GetNodeD()->GetPos_dt();

  chrono::ChVector<> nAa = shell->GetNodeA()->GetPos_dtdt();
  chrono::ChVector<> nBa = shell->GetNodeB()->GetPos_dtdt();
  chrono::ChVector<> nCa = shell->GetNodeC()->GetPos_dtdt();
  chrono::ChVector<> nDa = shell->GetNodeD()->GetPos_dtdt();
  printf(" posRadBCE.size()= :%d\n", posRadBCE.size());
  for (int i = 0; i < posRadBCE.size(); i++) {
    //    chrono::ChVector<> posGlob =
    chrono::ChVector<> pos_physical = ChFsiTypeConvert::Real3ToChVector(posRadBCE[i]);
    chrono::ChVector<> pos_natural = pos_physical * physic_to_natural;
    shell->ShapeFunctions(N, pos_natural.x, pos_natural.y, pos_natural.z);
    chrono::ChVector<> x_dir = (nBp - nAp + nCp - nDp);
    chrono::ChVector<> y_dir = (nCp - nBp + nDp - nAp);
    chrono::ChVector<> Normal;
    Normal.Cross(x_dir, y_dir);
    Normal.Normalize();
    //    printf("GetNormalized :%f,%f,%f\n", Normal.x, Normal.y, Normal.z);

    chrono::ChVector<> Correct_Pos = N(0) * nAp + N(2) * nBp + N(4) * nCp + N(6) * nDp +
                                     Normal * pos_physical.z * paramsH->HSML * paramsH->MULT_INITSPACE_Shells;

    if ((Correct_Pos.x < paramsH->cMin.x || Correct_Pos.x > paramsH->cMax.x) ||
        (Correct_Pos.y < paramsH->cMin.y || Correct_Pos.y > paramsH->cMax.y) ||
        (Correct_Pos.z < paramsH->cMin.z || Correct_Pos.z > paramsH->cMax.z)) {
      continue;
    }
    //    printf("fsiData->sphMarkersH.posRadH.push_back :%f,%f,%f\n", Correct_Pos.x, Correct_Pos.y, Correct_Pos.z);

    // Note that the fluid markers are removed differently
    bool addthis = true;
    for (int p = 0; p < fsiData->sphMarkersH.posRadH.size() - 1; p++) {
      if (length(fsiData->sphMarkersH.posRadH[p] - ChFsiTypeConvert::ChVectorToReal3(Correct_Pos)) < 1e-8 &&
          fsiData->sphMarkersH.rhoPresMuH[p].w != -1) {
        addthis = false;
        //        printf("remove this particle %f,%f,%f because of its overlap with a particle at %f,%f,%f\n",
        //               fsiData->sphMarkersH.posRadH[p].x, fsiData->sphMarkersH.posRadH[p].y,
        //               fsiData->sphMarkersH.posRadH[p].z,
        //               Correct_Pos.x, Correct_Pos.y, Correct_Pos.z);
        break;
      }
    }

    if (addthis) {
      fsiData->sphMarkersH.posRadH.push_back(ChFsiTypeConvert::ChVectorToReal3(Correct_Pos));
      chrono::ChVector<> Correct_Vel = N(0) * nAv + N(2) * nBv + N(4) * nCv + N(6) * nDv;
      Real3 v3 = ChFsiTypeConvert::ChVectorToReal3(Correct_Vel);
      fsiData->sphMarkersH.velMasH.push_back(v3);
      fsiData->sphMarkersH.rhoPresMuH.push_back(mR4(paramsH->rho0, paramsH->BASEPRES, paramsH->mu0, type));
      posRadSizeModified++;
    }
  }
  fsiData->sphMarkersH.rhoPresMuH.size();
  //  printf(" CreateBceGlobalMarkersFromBceLocalPos_ShellANCF : fsiData->sphMarkersH.rhoPresMuH.size() %d. ",
  //         fsiData->sphMarkersH.rhoPresMuH.size());

  // ------------------------
  // Modify number of objects
  // ------------------------
  int numObjects = fsiData->fsiGeneralData.referenceArray.size();
  int numBce = posRadSizeModified;
  fsiData->numObjects.numAllMarkers += numBce;

  int numRigid = fsiData->numObjects.numRigidBodies;
  fsiData->numObjects.numFlex_SphMarkers += numBce;
  fsiData->numObjects.numFlexBodies2D += 1;
  fsiData->numObjects.startFlexMarkers = fsiData->fsiGeneralData.referenceArray[numRigid + 1].y;
  int start_flex = fsiData->numObjects.startFlexMarkers;

  int4 last = fsiData->fsiGeneralData.referenceArray[fsiData->fsiGeneralData.referenceArray.size() - 1];
  fsiData->fsiGeneralData.referenceArray.push_back(
      mI4(last.y, last.y + numBce, type, fsiData->numObjects.numFlexBodies2D));  // 2: for Shell

  fsiData->fsiGeneralData.referenceArray_FEA.push_back(
      mI4(last.y, last.y + numBce, type, fsiData->numObjects.numFlexBodies2D));  // 2: for Shell

  printf(" push_back Index %d. ", fsiData->fsiGeneralData.referenceArray.size() - 1);
  int4 test = fsiData->fsiGeneralData.referenceArray[fsiData->fsiGeneralData.referenceArray.size() - 1];
  printf(" x=%d, y=%d, z=%d, w=%d\n", test.x, test.y, test.z, test.w);

  if (fsiData->numObjects.numFlexBodies2D !=
      fsiData->fsiGeneralData.referenceArray.size() - 2 - fsiData->numObjects.numRigidBodies -
          fsiData->numObjects.numFlexBodies1D) {
    printf("Error! num rigid Flexible does not match reference array size!\n\n");
    std::cin.get();
  }
  numObjects = fsiData->fsiGeneralData.referenceArray.size();
  printf("numObjects : %d\n ", numObjects);
  printf("numObjects.startFlexMarkers  : %d\n ", fsiData->numObjects.startFlexMarkers);
}

// =============================================================================
void CreateBceGlobalMarkersFromBceLocalPosBoundary(ChFsiDataManager* fsiData,
                                                   SimParams* paramsH,
                                                   const thrust::host_vector<Real3>& posRadBCE,
                                                   std::shared_ptr<chrono::ChBody> body,
                                                   chrono::ChVector<> collisionShapeRelativePos,
                                                   chrono::ChQuaternion<> collisionShapeRelativeRot) {
  CreateBceGlobalMarkersFromBceLocalPos(fsiData, paramsH, posRadBCE, body, collisionShapeRelativePos,
                                        collisionShapeRelativeRot, false);
}
// =============================================================================
void AddSphereBce(ChFsiDataManager* fsiData,
                  SimParams* paramsH,
                  std::shared_ptr<chrono::ChBody> body,
                  chrono::ChVector<> relPos,
                  chrono::ChQuaternion<> relRot,
                  Real radius) {
  thrust::host_vector<Real3> posRadBCE;
  CreateBCE_On_Sphere(posRadBCE, radius, paramsH);

  //	if (fsiData->sphMarkersH.posRadH.size() !=
  // fsiData->numObjects.numAllMarkers) {
  //		printf("Error! numMarkers, %d, does not match posRadH.size(),
  //%d\n",
  //				fsiData->numObjects.numAllMarkers,
  // fsiData->sphMarkersH.posRadH.size());
  //		std::cin.get();
  //	}

  CreateBceGlobalMarkersFromBceLocalPos(fsiData, paramsH, posRadBCE, body);

  posRadBCE.clear();
}
// =============================================================================

void AddCylinderBce(ChFsiDataManager* fsiData,
                    SimParams* paramsH,
                    std::shared_ptr<chrono::ChBody> body,
                    chrono::ChVector<> relPos,
                    chrono::ChQuaternion<> relRot,
                    Real radius,
                    Real height) {
  thrust::host_vector<Real3> posRadBCE;
  CreateBCE_On_Cylinder(posRadBCE, radius, height, paramsH);

  //	if (fsiData->sphMarkersH.posRadH.size() !=
  // fsiData->numObjects.numAllMarkers) {
  //		printf("Error! numMarkers, %d, does not match posRadH.size(),
  //%d\n",
  //				fsiData->numObjects.numAllMarkers,
  // fsiData->sphMarkersH.posRadH.size());
  //		std::cin.get();
  //	}

  CreateBceGlobalMarkersFromBceLocalPos(fsiData, paramsH, posRadBCE, body);
  posRadBCE.clear();
}

// =============================================================================
// Arman note, the function in the current implementation creates boundary bce
// (accesses only referenceArray[1])

// Arman thrust::host_vector<uint>& bodyIndex,

// Arman later on, you can remove numObjects since the Finalize function will
// take care of setting up the numObjects

void AddBoxBce(ChFsiDataManager* fsiData,
               SimParams* paramsH,
               std::shared_ptr<chrono::ChBody> body,
               chrono::ChVector<> relPos,
               chrono::ChQuaternion<> relRot,
               const chrono::ChVector<>& size) {
  thrust::host_vector<Real3> posRadBCE;

  CreateBCE_On_Box(posRadBCE, ChFsiTypeConvert::ChVectorToReal3(size), 12, paramsH);
  //	if (fsiData->sphMarkersH.posRadH.size() !=
  // fsiData->numObjects.numAllMarkers) {
  //		printf("Error! numMarkers, %d, does not match posRadH.size(),
  //%d\n",
  //				fsiData->numObjects.numAllMarkers,
  // fsiData->sphMarkersH.posRadH.size());
  //		std::cin.get();
  //	}

  CreateBceGlobalMarkersFromBceLocalPosBoundary(fsiData, paramsH, posRadBCE, body, relPos, relRot);
  posRadBCE.clear();
}

void AddBoxBceYZ(ChFsiDataManager* fsiData,
                 SimParams* paramsH,
                 std::shared_ptr<chrono::ChBody> body,
                 chrono::ChVector<> relPos,
                 chrono::ChQuaternion<> relRot,
                 const chrono::ChVector<>& size) {
  thrust::host_vector<Real3> posRadBCE;

  CreateBCE_On_Box(posRadBCE, ChFsiTypeConvert::ChVectorToReal3(size), 23, paramsH);
  //	if (fsiData->sphMarkersH.posRadH.size() !=
  // fsiData->numObjects.numAllMarkers) {
  //		printf("Error! numMarkers, %d, does not match posRadH.size(),
  //%d\n",
  //				fsiData->numObjects.numAllMarkers,
  // fsiData->sphMarkersH.posRadH.size());
  //		std::cin.get();
  //	}

  CreateBceGlobalMarkersFromBceLocalPosBoundary(fsiData, paramsH, posRadBCE, body, relPos, relRot);
  posRadBCE.clear();
}
void AddBoxBceXZ(ChFsiDataManager* fsiData,
                 SimParams* paramsH,
                 std::shared_ptr<chrono::ChBody> body,
                 chrono::ChVector<> relPos,
                 chrono::ChQuaternion<> relRot,
                 const chrono::ChVector<>& size) {
  thrust::host_vector<Real3> posRadBCE;

  CreateBCE_On_Box(posRadBCE, ChFsiTypeConvert::ChVectorToReal3(size), 13, paramsH);
  //	if (fsiData->sphMarkersH.posRadH.size() !=
  // fsiData->numObjects.numAllMarkers) {
  //		printf("Error! numMarkers, %d, does not match posRadH.size(),
  //%d\n",
  //				fsiData->numObjects.numAllMarkers,
  // fsiData->sphMarkersH.posRadH.size());
  //		std::cin.get();
  //	}

  CreateBceGlobalMarkersFromBceLocalPosBoundary(fsiData, paramsH, posRadBCE, body, relPos, relRot);
  posRadBCE.clear();
}
// =============================================================================

void AddBCE_ShellANCF(ChFsiDataManager* fsiData,
                      SimParams* paramsH,
                      std::vector<std::shared_ptr<chrono::fea::ChElementShellANCF>>* fsiShellsPtr,
                      std::shared_ptr<chrono::fea::ChMesh> my_mesh,
                      bool multiLayer,
                      bool removeMiddleLayer,
                      int SIDE) {
  thrust::host_vector<Real3> posRadBCE;
  int numShells = my_mesh->GetNelements();
  printf("number of shells to be meshed is %d\n", numShells);
  for (int i = 0; i < numShells; i++) {
    auto thisShell = std::dynamic_pointer_cast<fea::ChElementShellANCF>(my_mesh->GetElement(i));
    fsiShellsPtr->push_back(thisShell);
    CreateBCE_On_shell(posRadBCE, paramsH, thisShell, multiLayer, removeMiddleLayer, SIDE);
    CreateBceGlobalMarkersFromBceLocalPos_ShellANCF(fsiData, paramsH, posRadBCE, thisShell);

    posRadBCE.clear();
  }
}

// =============================================================================

void AddBCE_ShellFromMesh(ChFsiDataManager* fsiData,
                          SimParams* paramsH,
                          std::vector<std::shared_ptr<chrono::fea::ChElementShellANCF>>* fsiShellsPtr,
                          std::vector<std::shared_ptr<chrono::fea::ChNodeFEAxyzD>>* fsiNodesPtr,
                          std::shared_ptr<chrono::fea::ChMesh> my_mesh,
                          std::vector<std::vector<int>> elementsNodes,
                          std::vector<std::vector<int>> NodeNeighborElement,
                          bool multiLayer,
                          bool removeMiddleLayer,
                          int SIDE) {
  thrust::host_vector<Real3> posRadBCE;
  int numShells = my_mesh->GetNelements();
  std::vector<int> remove;

  for (int i = 0; i < NodeNeighborElement.size(); i++) {
    auto thisNode = std::dynamic_pointer_cast<fea::ChNodeFEAxyzD>(my_mesh->GetNode(i));
    fsiNodesPtr->push_back(thisNode);
  }

  for (int i = 0; i < numShells; i++) {
    remove.resize(4);
    std::fill(remove.begin(), remove.begin() + 4, 0);
    auto thisShell = std::dynamic_pointer_cast<fea::ChElementShellANCF>(my_mesh->GetElement(i));
    fsiShellsPtr->push_back(thisShell);
    // Look into the nodes of this element
    int myNumNodes = (elementsNodes[i].size() > 4) ? 4 : elementsNodes[i].size();

    for (int j = 0; j < myNumNodes; j++) {
      int thisNode = elementsNodes[i][j] - 1;
      //      printf("Considering elementsNodes[%d][%d]=%d\n", i, j, thisNode);

      // Look into the elements attached to thisNode
      for (int k = 0; k < NodeNeighborElement[thisNode].size(); k++) {
        // If this neighbor element has more than one common node with the previous node this means that we must not
        // add BCEs to this edge anymore. Because that edge has already been given BCE markers
        // The kth element of this node:
        int neighborElement = NodeNeighborElement[thisNode][k];
        if (neighborElement >= i)
          continue;
        //        printf("neighborElement %d\n", neighborElement);

        int JNumNodes = (elementsNodes[neighborElement].size() > 4) ? 4 : elementsNodes[neighborElement].size();

        for (int inode = 0; inode < myNumNodes; inode++) {
          for (int jnode = 0; jnode < JNumNodes; jnode++) {
            if (elementsNodes[i][inode] - 1 == elementsNodes[neighborElement][jnode] - 1 &&
                thisNode != elementsNodes[i][inode] - 1 && i > neighborElement) {
              //              printf("node %d is common between %d and %d\n", elementsNodes[i][inode] - 1, i,
              //              neighborElement);
              remove[inode] = 1;
            }
          }
        }
      }
    }

    //    printf("remove: %d, %d, %d, %d\n", remove[0], remove[1], remove[2], remove[3]);

    CreateBCE_On_ChElementShellANCF(posRadBCE, paramsH, thisShell, remove, multiLayer, removeMiddleLayer, SIDE);
    CreateBceGlobalMarkersFromBceLocalPos_ShellANCF(fsiData, paramsH, posRadBCE, thisShell);
    posRadBCE.clear();
  }
}
// =============================================================================

// =============================================================================

void AddBCE_FromMesh(ChFsiDataManager* fsiData,
                     SimParams* paramsH,
                     std::shared_ptr<chrono::fea::ChMesh> my_mesh,
                     std::vector<std::shared_ptr<chrono::fea::ChNodeFEAxyzD>>* fsiNodesPtr,
                     std::vector<std::shared_ptr<chrono::fea::ChElementCableANCF>>* fsiCablesPtr,
                     std::vector<std::shared_ptr<chrono::fea::ChElementShellANCF>>* fsiShellsPtr,
                     std::vector<std::vector<int>> NodeNeighborElement,
                     std::vector<std::vector<int>> _1D_elementsNodes,
                     std::vector<std::vector<int>> _2D_elementsNodes,
                     bool add1DElem,
                     bool add2DElem,
                     bool multiLayer,
                     bool removeMiddleLayer,
                     int SIDE,
                     int SIDE2D) {
  thrust::host_vector<Real3> posRadBCE;
  int numElems = my_mesh->GetNelements();
  std::vector<int> remove2D;
  std::vector<int> remove1D;

  for (int i = 0; i < my_mesh->GetNnodes(); i++) {
    auto thisNode = std::dynamic_pointer_cast<fea::ChNodeFEAxyzD>(my_mesh->GetNode(i));
    fsiNodesPtr->push_back(thisNode);
  }

  for (int i = 0; i < numElems; i++) {
    ///////////////////////////
    // Check for Cable Elements
    if (_1D_elementsNodes.size() > 0) {
      if (auto thisCable = std::dynamic_pointer_cast<fea::ChElementCableANCF>(my_mesh->GetElement(i))) {
        remove1D.resize(2);
        std::fill(remove1D.begin(), remove1D.begin() + 2, 0);
        fsiCablesPtr->push_back(thisCable);
        // Look into the nodes of this element
        //      for (int i = 0; i < _1D_elementsNodes.size(); i++) {
        //        printf("_1D_elementsNodes[%d][1,2]= %d, %d\n", i, _1D_elementsNodes[i][0], _1D_elementsNodes[i][1]);
        //      }
        //      printf("NodeNeighborElement.size() after adding beams =%d\n", NodeNeighborElement.size());

        int myNumNodes = (_1D_elementsNodes[i].size() > 2) ? 2 : _1D_elementsNodes[i].size();

        for (int j = 0; j < myNumNodes; j++) {
          int thisNode = _1D_elementsNodes[i][j];

          // Look into the elements attached to thisNode
          for (int k = 0; k < NodeNeighborElement[thisNode].size(); k++) {
            // If this neighbor element has more than one common node with the previous node this means that we must not
            // add BCEs to this edge anymore. Because that edge has already been given BCE markers
            // The kth element of this node:
            int neighborElement = NodeNeighborElement[thisNode][k];
            if (neighborElement >= i)
              continue;

            int JNumNodes =
                (_1D_elementsNodes[neighborElement].size() > 2) ? 2 : _1D_elementsNodes[neighborElement].size();

            for (int inode = 0; inode < myNumNodes; inode++) {
              for (int jnode = 0; jnode < JNumNodes; jnode++) {
                if (_1D_elementsNodes[i][inode] == _1D_elementsNodes[neighborElement][jnode] &&
                    thisNode != _1D_elementsNodes[i][inode] - 1) {
                  remove1D[inode] = 1;
                  //                  printf("removing _1D_elementsNodes[%d][%d]=%d\n", i, inode,
                  //                  _1D_elementsNodes[i][inode]);
                }
              }
            }
          }
        }

        if (add1DElem) {
          CreateBCE_On_ChElementCableANCF(posRadBCE, paramsH, thisCable, remove1D, multiLayer, removeMiddleLayer, SIDE);
          CreateBceGlobalMarkersFromBceLocalPos_CableANCF(fsiData, paramsH, posRadBCE, thisCable);
        }
        posRadBCE.clear();
      }
    }
    int Curr_size = _1D_elementsNodes.size();

    ///////////////////////////
    // Check for Shell Elements
    if (_2D_elementsNodes.size() > 0) {
      if (auto thisShell = std::dynamic_pointer_cast<fea::ChElementShellANCF>(my_mesh->GetElement(i))) {
        remove2D.resize(4);
        std::fill(remove2D.begin(), remove2D.begin() + 4, 0);

        fsiShellsPtr->push_back(thisShell);
        // Look into the nodes of this element
        int myNumNodes = (_2D_elementsNodes[i - Curr_size].size() > 4) ? 4 : _2D_elementsNodes[i - Curr_size].size();

        for (int j = 0; j < myNumNodes; j++) {
          int thisNode = _2D_elementsNodes[i - Curr_size][j];
          //          printf("Considering elementsNodes[%d][%d]=%d\n", i - Curr_size, j, thisNode);

          // Look into the elements attached to thisNode
          for (int k = 0; k < NodeNeighborElement[thisNode].size(); k++) {
            // If this neighbor element has more than one common node with the previous node this means that we must not
            // add BCEs to this edge anymore. Because that edge has already been given BCE markers
            // The kth element of this node:
            int neighborElement = NodeNeighborElement[thisNode][k] - Curr_size;
            //            printf("Considering neighbor NodeNeighborElement[%d][%d]=%d\n", thisNode, k, neighborElement);

            if (neighborElement >= i - Curr_size)
              continue;

            int JNumNodes =
                (_2D_elementsNodes[neighborElement].size() > 4) ? 4 : _2D_elementsNodes[neighborElement].size();

            for (int inode = 0; inode < myNumNodes; inode++) {
              for (int jnode = 0; jnode < JNumNodes; jnode++) {
                if (_2D_elementsNodes[i - Curr_size][inode] == _2D_elementsNodes[neighborElement][jnode] &&
                    thisNode != _2D_elementsNodes[i - Curr_size][inode] && i > neighborElement) {
                  remove2D[inode] = 1;
                  //                  printf("removing _2D_elementsNodes[%d][%d]=%d\n", i - Curr_size, inode,
                  //                         _2D_elementsNodes[i - Curr_size][inode]);
                }
              }
            }
          }
        }
        if (add2DElem) {
          CreateBCE_On_ChElementShellANCF(posRadBCE, paramsH, thisShell, remove2D, multiLayer, removeMiddleLayer,
                                          SIDE2D);
          CreateBceGlobalMarkersFromBceLocalPos_ShellANCF(fsiData, paramsH, posRadBCE, thisShell);
        }
        posRadBCE.clear();
      }
    }
    ///////////////////////////
    // Check for break Elements
  }
}
// =============================================================================

void AddBCE_FromFile(ChFsiDataManager* fsiData,
                     SimParams* paramsH,
                     std::shared_ptr<chrono::ChBody> body,
                     std::string dataPath) {
  //----------------------------
  //  chassis
  //----------------------------
  thrust::host_vector<Real3> posRadBCE;

  LoadBCE_fromFile(posRadBCE, dataPath);

  //	if (fsiData->sphMarkersH.posRadH.size() !=
  // fsiData->numObjects.numAllMarkers) {
  //		printf("Error! numMarkers, %d, does not match posRadH.size(),
  //%d\n",
  //				fsiData->numObjects.numAllMarkers,
  // fsiData->sphMarkersH.posRadH.size());
  //		std::cin.get();
  //	}

  CreateBceGlobalMarkersFromBceLocalPos(fsiData, paramsH, posRadBCE, body);
  posRadBCE.clear();
}

// =============================================================================
void CreateSphereFSI(ChFsiDataManager* fsiData,
                     chrono::ChSystem& mphysicalSystem,
                     std::vector<std::shared_ptr<chrono::ChBody>>* fsiBodeisPtr,
                     SimParams* paramsH,
                     std::shared_ptr<chrono::ChMaterialSurface> mat_prop,
                     Real density,
                     chrono::ChVector<> pos,
                     Real radius) {
  //	ChVector<> pos = ChVector<>(-9.5, .20, 3);
  //	Real radius = 0.3;

  auto body = std::make_shared<chrono::ChBody>(new chrono::collision::ChCollisionModelParallel);
  body->SetBodyFixed(false);
  body->SetCollide(true);
  body->SetMaterialSurface(mat_prop);
  body->SetPos(pos);
  double volume = chrono::utils::CalcSphereVolume(radius);
  chrono::ChVector<> gyration = chrono::utils::CalcSphereGyration(radius).Get_Diag();
  double mass = density * volume;
  body->SetMass(mass);
  body->SetInertiaXX(mass * gyration);
  //
  body->GetCollisionModel()->ClearModel();
  chrono::utils::AddSphereGeometry(body.get(), radius);
  body->GetCollisionModel()->BuildModel();
  mphysicalSystem.AddBody(body);
  fsiBodeisPtr->push_back(body);

  AddSphereBce(fsiData, paramsH, body, chrono::ChVector<>(0, 0, 0), chrono::ChQuaternion<>(1, 0, 0, 0), radius);
}
// =============================================================================
void CreateCylinderFSI(ChFsiDataManager* fsiData,
                       chrono::ChSystem& mphysicalSystem,
                       std::vector<std::shared_ptr<chrono::ChBody>>* fsiBodeisPtr,
                       SimParams* paramsH,
                       std::shared_ptr<chrono::ChMaterialSurface> mat_prop,
                       Real density,
                       chrono::ChVector<> pos,
                       chrono::ChQuaternion<> rot,
                       Real radius,
                       Real length) {
  auto body = std::make_shared<chrono::ChBody>(new chrono::collision::ChCollisionModelParallel);
  body->SetBodyFixed(false);
  body->SetCollide(true);
  body->SetMaterialSurface(mat_prop);
  body->SetPos(pos);

  body->SetRot(rot);
  double volume = chrono::utils::CalcCylinderVolume(radius, 0.5 * length);
  chrono::ChVector<> gyration = chrono::utils::CalcCylinderGyration(radius, 0.5 * length).Get_Diag();
  double mass = density * volume;
  body->SetMass(mass);
  body->SetInertiaXX(mass * gyration);
  //
  body->GetCollisionModel()->ClearModel();
  chrono::utils::AddCylinderGeometry(body.get(), radius, 0.5 * length);
  body->GetCollisionModel()->BuildModel();
  mphysicalSystem.AddBody(body);

  fsiBodeisPtr->push_back(body);
  AddCylinderBce(fsiData, paramsH, body, chrono::ChVector<>(0, 0, 0), chrono::ChQuaternion<>(1, 0, 0, 0), radius,
                 length);
}

// =============================================================================
void CreateBoxFSI(ChFsiDataManager* fsiData,
                  chrono::ChSystem& mphysicalSystem,
                  std::vector<std::shared_ptr<chrono::ChBody>>* fsiBodeisPtr,
                  SimParams* paramsH,
                  std::shared_ptr<chrono::ChMaterialSurface> mat_prop,
                  Real density,
                  chrono::ChVector<> pos,
                  chrono::ChQuaternion<> rot,
                  const chrono::ChVector<>& hsize) {
  auto body = std::make_shared<chrono::ChBody>(new chrono::collision::ChCollisionModelParallel);
  body->SetBodyFixed(false);
  body->SetCollide(true);
  body->SetMaterialSurface(mat_prop);
  body->SetPos(pos);
  body->SetRot(rot);
  double volume = chrono::utils::CalcBoxVolume(hsize);
  chrono::ChVector<> gyration = chrono::utils::CalcBoxGyration(hsize).Get_Diag();
  double mass = density * volume;
  body->SetMass(mass);
  body->SetInertiaXX(mass * gyration);
  //
  body->GetCollisionModel()->ClearModel();
  chrono::utils::AddBoxGeometry(body.get(), hsize);
  body->GetCollisionModel()->BuildModel();
  mphysicalSystem.AddBody(body);

  fsiBodeisPtr->push_back(body);
  AddBoxBce(fsiData, paramsH, body, chrono::ChVector<>(0, 0, 0), chrono::ChQuaternion<>(1, 0, 0, 0), hsize);
}

}  // end namespace utils
}  // end namespace fsi
}  // end namespace chrono
