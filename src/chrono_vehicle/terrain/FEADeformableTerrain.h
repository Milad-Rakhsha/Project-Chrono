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
// Authors: Antonio Recuero, Bryan Peterson
// =============================================================================
//
// FEA Deformable terrain. Box of terrain composed of 9-node brick elements which
// can capture moderate deformation (no remeshing). Constitutive behavior given
// by Drucker-Prager.
//
// =============================================================================

#ifndef FEADEFORMABLE_TERRAIN_H
#define FEADEFORMABLE_TERRAIN_H

#include "chrono/physics/ChSystem.h"
#include "chrono_fea/ChMesh.h"
#include "chrono_vehicle/ChApiVehicle.h"
#include "chrono_vehicle/ChTerrain.h"

namespace chrono {
namespace vehicle {

/// @addtogroup vehicle_terrain
/// @{

/// FEA Deformable terrain model.
/// This class implements a terrain made up of isoparametric finite elements. It features
/// Drucker-Prager plasticity and capped Drucker-Prager plasticity.
class CH_VEHICLE_API FEADeformableTerrain : public ChTerrain {
  public:
    /// Construct a default DeformableSoil.
    /// The user is responsible for calling various Set methods before Initialize.
    FEADeformableTerrain(ChSystem* system);  ///< [in/out] pointer to the containing system);

    ~FEADeformableTerrain() {}

    /// Get the terrain height at the specified (x,y) location.
    virtual double GetHeight(double x, double y) const override;

    /// Get the terrain normal at the specified (x,y) location.
    virtual chrono::ChVector<> GetNormal(double x, double y) const override;

    /// Set the properties of the Drucker-Prager FEA soil.
    void SetSoilParametersFEA(double rho,              ///< [in] Soil density
                              double Emod,             ///< [in] Soil modulus of elasticity
                              double nu,               ///< [in] Soil Poisson ratio
                              double yield_stress,     ///< [in] Soil yield stress, for plasticity
                              double hardening_slope,  ///< [in] Soil hardening slope, for plasticity
                              double friction_angle,   ///< [in] Soil internal friction angle
                              double dilatancy_angle   ///< [in] Soil dilatancy angle
                              );

    /// Initialize the terrain system (flat).
    /// This version creates a flat array of points.
    void Initialize(
        const ChVector<>& start_point,                 ///< [in] Base point to build terrain box
        const ChVector<>& terrain_dimension,           ///< [in] terrain dimensions in the 3 directions
        const ChVector<int>& terrain_discretization);  ///< [in] Number of finite elements in the 3 directions

    /// Get the underlying FEA mesh.
    std::shared_ptr<fea::ChMesh> GetMesh() const { return m_mesh; }

  private:
    std::shared_ptr<fea::ChMesh> m_mesh;  ///< soil mesh

    double m_rho;  ///< Soil density
    double m_E;    ///< Soil modulus of elasticity
    double m_nu;   ///< Soil Poisson ratio

    double m_yield_stress;     ///< Yield stress for soil plasticity
    double m_hardening_slope;  ///< Hardening slope for soil plasticity
    double m_friction_angle;   ///< Set friction angle for soil plasticity
    double m_dilatancy_angle;  ///< Set dilatancy angle for soil plasticity
};

/// @} vehicle_terrain

}  // end namespace vehicle
}  // end namespace chrono

#endif
