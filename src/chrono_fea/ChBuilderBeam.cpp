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
// Authors: Alessandro Tasora
// =============================================================================

#include "chrono_fea/ChBuilderBeam.h"

namespace chrono {
namespace fea {

void ChBuilderBeam::BuildBeam(std::shared_ptr<ChMesh> mesh,                 ///< mesh to store the resulting elements
                              std::shared_ptr<ChBeamSectionAdvanced> sect,  ///< section material for beam elements
                              const int N,                                  ///< number of elements in the segment
                              const ChVector<> A,                           ///< starting point
                              const ChVector<> B,                           ///< ending point
                              const ChVector<> Ydir                         ///< the 'up' Y direction of the beam
                              ) {
    beam_elems.clear();
    beam_nodes.clear();

    ChMatrix33<> mrot;
    mrot.Set_A_Xdir(B - A, Ydir);

    auto nodeA = std::make_shared<ChNodeFEAxyzrot>(ChFrame<>(A, mrot));
    mesh->AddNode(nodeA);
    beam_nodes.push_back(nodeA);

    for (int i = 1; i <= N; ++i) {
        double eta = (double)i / (double)N;
        ChVector<> pos = A + (B - A) * eta;

        auto nodeB = std::make_shared<ChNodeFEAxyzrot>(ChFrame<>(pos, mrot));
        mesh->AddNode(nodeB);
        beam_nodes.push_back(nodeB);

        auto element = std::make_shared<ChElementBeamEuler>();
        mesh->AddElement(element);
        beam_elems.push_back(element);

        element->SetNodes(beam_nodes[i - 1], beam_nodes[i]);

        element->SetSection(sect);
    }
}

void ChBuilderBeam::BuildBeam(std::shared_ptr<ChMesh> mesh,                 ///< mesh to store the resulting elements
                              std::shared_ptr<ChBeamSectionAdvanced> sect,  ///< section material for beam elements
                              const int N,                                  ///< number of elements in the segment
                              std::shared_ptr<ChNodeFEAxyzrot> nodeA,       ///< starting point
                              std::shared_ptr<ChNodeFEAxyzrot> nodeB,       ///< ending point
                              const ChVector<> Ydir                         ///< the 'up' Y direction of the beam
                              ) {
    beam_elems.clear();
    beam_nodes.clear();

    ChMatrix33<> mrot;
    mrot.Set_A_Xdir(nodeB->Frame().GetPos() - nodeA->Frame().GetPos(), Ydir);

    beam_nodes.push_back(nodeA);

    for (int i = 1; i <= N; ++i) {
        double eta = (double)i / (double)N;
        ChVector<> pos = nodeA->Frame().GetPos() + (nodeB->Frame().GetPos() - nodeA->Frame().GetPos()) * eta;

        std::shared_ptr<ChNodeFEAxyzrot> nodeBi;
        if (i < N) {
            nodeBi = std::make_shared<ChNodeFEAxyzrot>(ChFrame<>(pos, mrot));
            mesh->AddNode(nodeBi);
        } else
            nodeBi = nodeB;  // last node: use the one passed as parameter.

        beam_nodes.push_back(nodeBi);

        auto element = std::make_shared<ChElementBeamEuler>();
        mesh->AddElement(element);
        beam_elems.push_back(element);

        element->SetNodes(beam_nodes[i - 1], beam_nodes[i]);

        ChQuaternion<> elrot = mrot.Get_A_quaternion();
        element->SetNodeAreferenceRot(elrot.GetConjugate() % element->GetNodeA()->Frame().GetRot());
        element->SetNodeBreferenceRot(elrot.GetConjugate() % element->GetNodeB()->Frame().GetRot());

        element->SetSection(sect);
    }
}

void ChBuilderBeam::BuildBeam(std::shared_ptr<ChMesh> mesh,                 ///< mesh to store the resulting elements
                              std::shared_ptr<ChBeamSectionAdvanced> sect,  ///< section material for beam elements
                              const int N,                                  ///< number of elements in the segment
                              std::shared_ptr<ChNodeFEAxyzrot> nodeA,       ///< starting point
                              const ChVector<> B,                           ///< ending point
                              const ChVector<> Ydir                         ///< the 'up' Y direction of the beam
                              ) {
    beam_elems.clear();
    beam_nodes.clear();

    ChMatrix33<> mrot;
    mrot.Set_A_Xdir(B - nodeA->Frame().GetPos(), Ydir);

    beam_nodes.push_back(nodeA);

    for (int i = 1; i <= N; ++i) {
        double eta = (double)i / (double)N;
        ChVector<> pos = nodeA->Frame().GetPos() + (B - nodeA->Frame().GetPos()) * eta;

        auto nodeBi = std::make_shared<ChNodeFEAxyzrot>(ChFrame<>(pos, mrot));
        mesh->AddNode(nodeBi);
        beam_nodes.push_back(nodeBi);

        auto element = std::make_shared<ChElementBeamEuler>();
        mesh->AddElement(element);
        beam_elems.push_back(element);

        element->SetNodes(beam_nodes[i - 1], beam_nodes[i]);

        ChQuaternion<> elrot = mrot.Get_A_quaternion();
        element->SetNodeAreferenceRot(elrot.GetConjugate() % element->GetNodeA()->Frame().GetRot());
        element->SetNodeBreferenceRot(elrot.GetConjugate() % element->GetNodeB()->Frame().GetRot());
        // GetLog() << "Element n." << i << " with rotations: \n";
        // GetLog() << "   Qa=" << element->GetNodeAreferenceRot() << "\n";
        // GetLog() << "   Qb=" << element->GetNodeBreferenceRot() << "\n\n";
        element->SetSection(sect);
    }
}

/////////////////////////////////////////////////////////
//
// ChBuilderBeamANCF

void ChBuilderBeamANCF::BuildBeam(std::shared_ptr<ChMesh> mesh,              ///< mesh to store the resulting elements
                                  std::shared_ptr<ChBeamSectionCable> sect,  ///< section material for beam elements
                                  const int N,                               ///< number of elements in the segment
                                  const ChVector<> A,                        ///< starting point
                                  const ChVector<> B                         ///< ending point
                                  ) {
    beam_elems.clear();
    beam_nodes.clear();

    ChVector<> bdir = (B - A);
    bdir.Normalize();

    auto nodeA = std::make_shared<ChNodeFEAxyzD>(A, bdir);
    mesh->AddNode(nodeA);
    beam_nodes.push_back(nodeA);

    for (int i = 1; i <= N; ++i) {
        double eta = (double)i / (double)N;
        ChVector<> pos = A + (B - A) * eta;

        auto nodeB = std::make_shared<ChNodeFEAxyzD>(pos, bdir);
        mesh->AddNode(nodeB);
        beam_nodes.push_back(nodeB);

        auto element = std::make_shared<ChElementCableANCF>();
        mesh->AddElement(element);
        beam_elems.push_back(element);

        element->SetNodes(beam_nodes[i - 1], beam_nodes[i]);

        element->SetSection(sect);
    }
}

/////////////////////////////////////////////////////////
//
// ChBuilderBeamANCF_modified

void ChBuilderBeamANCF::BuildBeam_FSI(std::shared_ptr<ChMesh> mesh,  ///< mesh to store the resulting elements
                                      std::shared_ptr<ChBeamSectionCable> sect,  ///< section material for beam elements
                                      const int N,                               ///< number of elements in the segment
                                      const ChVector<> A,                        ///< starting point
                                      const ChVector<> B,                        ///< ending point
                                      std::vector<std::vector<int>>& _1D_elementsNodes_mesh,
                                      std::vector<std::vector<int>>& NodeNeighborElement1D_mesh) {
    beam_elems.clear();
    beam_nodes.clear();
    int _1D_elementsNodes_size = _1D_elementsNodes_mesh.size();
    int NodeNeighborElement1D_size = NodeNeighborElement1D_mesh.size();

    _1D_elementsNodes_mesh.resize(N + _1D_elementsNodes_size);
    NodeNeighborElement1D_mesh.resize(N + 1 + NodeNeighborElement1D_size);
    ChVector<> bdir = (B - A);
    bdir.Normalize();

    auto nodeA = std::make_shared<ChNodeFEAxyzD>(A, bdir);
    mesh->AddNode(nodeA);
    beam_nodes.push_back(nodeA);

    for (int i = 0; i < N; i++) {
        double eta = (double)(i + 1) / (double)N;
        ChVector<> pos = A + (B - A) * eta;

        auto nodeB = std::make_shared<ChNodeFEAxyzD>(pos, bdir);
        mesh->AddNode(nodeB);
        beam_nodes.push_back(nodeB);

        auto element = std::make_shared<ChElementCableANCF>();
        mesh->AddElement(element);
        beam_elems.push_back(element);

        element->SetNodes(beam_nodes[i], beam_nodes[i + 1]);

        element->SetSection(sect);

        _1D_elementsNodes_mesh[_1D_elementsNodes_size + i].push_back(_1D_elementsNodes_size + i);
        _1D_elementsNodes_mesh[_1D_elementsNodes_size + i].push_back(i + _1D_elementsNodes_size + 1);

        NodeNeighborElement1D_mesh[_1D_elementsNodes_size + i].push_back(_1D_elementsNodes_size + i);
        NodeNeighborElement1D_mesh[i + _1D_elementsNodes_size + 1].push_back(_1D_elementsNodes_size + i);

        //        printf("Adding nodes %d,%d to the cable element %i\n ", _1D_elementsNodes_size + i,
        //               _1D_elementsNodes_size + i + 1, _1D_elementsNodes_size + i);
        //        printf("Adding element %d to the nodes %d,%d\n ", _1D_elementsNodes_size + i, _1D_elementsNodes_size +
        //        i,
        //               _1D_elementsNodes_size + i + 1);

        //        printf("Added cable element %d with nBp=(%f,%f,%f) from ChBuilderBeamd\n", i, pos.x, pos.y, pos.z);

        //        printf("Added cable element %d with nBp=(%f,%f,%f) from ChBuilderBeamd\n", i, pos.x, pos.y, pos.z);
    }
}
}  // end namespace fea
}  // end namespace chrono
