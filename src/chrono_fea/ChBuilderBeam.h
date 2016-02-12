//
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2013 Project Chrono
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file at the top level of the distribution
// and at http://projectchrono.org/license-chrono.txt.
//
// File authors: Alessandro Tasora

#ifndef CHBUILDERBEAM_H
#define CHBUILDERBEAM_H

#include "chrono_fea/ChMesh.h"
#include "chrono_fea/ChElementBeamEuler.h"
#include "chrono_fea/ChElementBeamANCF.h"

namespace chrono {
namespace fea {

/// Class for an helper object that provides easy functions to create
/// complex beams, for example subdivides a segment in multiple finite
/// elements.
class ChApiFea ChBuilderBeam {
  protected:
    std::vector<std::shared_ptr<ChElementBeamEuler> > beam_elems;
    std::vector<std::shared_ptr<ChNodeFEAxyzrot> > beam_nodes;

  public:
    /// Helper function.
    /// Adds beam FEM elements to the mesh to create a segment beam
    /// from point A to point B, using ChElementBeamEuler type elements.
    /// Before running, each time resets lists of beam_elems and beam_nodes.
    void BuildBeam(std::shared_ptr<ChMesh> mesh,                 ///< mesh to store the resulting elements
                   std::shared_ptr<ChBeamSectionAdvanced> sect,  ///< section material for beam elements
                   const int N,                                  ///< number of elements in the segment
                   const ChVector<> A,                           ///< starting point
                   const ChVector<> B,                           ///< ending point
                   const ChVector<> Ydir                         ///< the 'up' Y direction of the beam
                   );

    /// Helper function.
    /// Adds beam FEM elements to the mesh to create a segment beam
    /// from one existing node to another existing node, using ChElementBeamEuler type elements.
    /// Before running, each time resets lists of beam_elems and beam_nodes.
    void BuildBeam(std::shared_ptr<ChMesh> mesh,                 ///< mesh to store the resulting elements
                   std::shared_ptr<ChBeamSectionAdvanced> sect,  ///< section material for beam elements
                   const int N,                                  ///< number of elements in the segment
                   std::shared_ptr<ChNodeFEAxyzrot> nodeA,       ///< starting point
                   std::shared_ptr<ChNodeFEAxyzrot> nodeB,       ///< ending point
                   const ChVector<> Ydir                         ///< the 'up' Y direction of the beam
                   );

    /// Helper function.
    /// Adds beam FEM elements to the mesh to create a segment beam
    /// from one existing node to a point B, using ChElementBeamEuler type elements.
    /// Before running, each time resets lists of beam_elems and beam_nodes.
    void BuildBeam(std::shared_ptr<ChMesh> mesh,                 ///< mesh to store the resulting elements
                   std::shared_ptr<ChBeamSectionAdvanced> sect,  ///< section material for beam elements
                   const int N,                                  ///< number of elements in the segment
                   std::shared_ptr<ChNodeFEAxyzrot> nodeA,       ///< starting point
                   const ChVector<> B,                           ///< ending point
                   const ChVector<> Ydir                         ///< the 'up' Y direction of the beam
                   );
    
    /// Access the list of elements used by the last built beam.
    /// It can be useful for changing properties afterwards.
    /// This list is reset all times a 'Build...' function is called.
    std::vector<std::shared_ptr<ChElementBeamEuler> >& GetLastBeamElements() { return beam_elems; }

    /// Access the list of nodes used by the last built beam.
    /// It can be useful for adding constraints or changing properties afterwards.
    /// This list is reset all times a 'Build...' function is called.
    std::vector<std::shared_ptr<ChNodeFEAxyzrot> >& GetLastBeamNodes() { return beam_nodes; }
};



/// Class for an helper object that provides easy functions to create
/// complex beams of ChElementBeamANCF class, for example subdivides a segment 
/// in multiple finite elements.

class ChApiFea ChBuilderBeamANCF {
  protected:
    std::vector<std::shared_ptr<ChElementBeamANCF> > beam_elems;
    std::vector<std::shared_ptr<ChNodeFEAxyzD> > beam_nodes;

  public:
    /// Helper function.
    /// Adds beam FEM elements to the mesh to create a segment beam
    /// from point A to point B, using ChElementBeamANCF type elements.
    /// Before running, each time resets lists of beam_elems and beam_nodes.
    void BuildBeam(std::shared_ptr<ChMesh> mesh,                 ///< mesh to store the resulting elements
                   std::shared_ptr<ChBeamSectionCable> sect,     ///< section material for beam elements
                   const int N,                                  ///< number of elements in the segment
                   const ChVector<> A,                           ///< starting point
                   const ChVector<> B                            ///< ending point
                   );
    
    /// Access the list of elements used by the last built beam.
    /// It can be useful for changing properties afterwards.
    /// This list is reset all times a 'Build...' function is called.
    std::vector<std::shared_ptr<ChElementBeamANCF> >& GetLastBeamElements() { return beam_elems; }

    /// Access the list of nodes used by the last built beam.
    /// It can be useful for adding constraints or changing properties afterwards.
    /// This list is reset all times a 'Build...' function is called.
    std::vector<std::shared_ptr<ChNodeFEAxyzD> >& GetLastBeamNodes() { return beam_nodes; }
};


}  // END_OF_NAMESPACE____
}  // END_OF_NAMESPACE____

#endif
