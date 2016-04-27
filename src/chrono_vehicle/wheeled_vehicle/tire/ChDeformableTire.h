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
// Authors: Radu Serban
// =============================================================================
//
// Base class for a deformable tire (i.e. modeled with an FEA mesh)
//
// =============================================================================

#ifndef CH_DEFORMABLETIRE_H
#define CH_DEFORMABLETIRE_H

#include "chrono/physics/ChBody.h"
#include "chrono/physics/ChLoadContainer.h"
#include "chrono/physics/ChSystemDEM.h"

#include "chrono_fea/ChContactSurfaceMesh.h"
#include "chrono_fea/ChContactSurfaceNodeCloud.h"
#include "chrono_fea/ChLinkDirFrame.h"
#include "chrono_fea/ChLinkPointFrame.h"
#include "chrono_fea/ChMesh.h"
#include "chrono_fea/ChNodeFEAbase.h"
#include "chrono_fea/ChVisualizationFEAmesh.h"

#include "chrono_vehicle/wheeled_vehicle/ChTire.h"

namespace chrono {
namespace vehicle {

/// @addtogroup vehicle_wheeled_tire
/// @{

/// Base class for a deformable tire model.
class CH_VEHICLE_API ChDeformableTire : public ChTire {
  public:
    /// Type of the mesh contact surface.
    enum ContactSurfaceType { NODE_CLOUD, TRIANGLE_MESH };

    /// Construct a deformable tire with the specified name.
    ChDeformableTire(const std::string& name  ///< [in] name of this tire system
                     );

    /// Set the type of contact surface.
    void SetContactSurfaceType(ContactSurfaceType type) { m_contact_type = type; }
    ContactSurfaceType GetContactSurfaceType() const { return m_contact_type; }

    /// Set radius of contact nodes.
    /// This value is relevant only for NODE_CLOUD contact surface type.
    void SetContactNodeRadius(double radius) { m_contact_node_radius = radius; }
    double GetContactNodeRadius() const { return m_contact_node_radius; }

    /// Set thickness of contact faces (radius of swept sphere).
    /// This value is relevant only for TRIANGLE_MESH contact surface type.
    void SetContactFaceThickness(double thickness) { m_contact_face_thickness = thickness; }
    double GetContactFaceThickness() const { return m_contact_face_thickness; }

    /// Set contact material properties.
    /// Alternatively, the contact material coefficients can be set explicitly, using the
    /// function SetContactMaterialCoefficients.
    void SetContactMaterialProperties(float friction_coefficient = 0.6f,    ///< [in] coefficient of friction
                                      float restitution_coefficient = 0.1,  ///< [in] coefficient of restitution
                                      float young_modulus = 2e5f,           ///< [in] Young's modulus of elasticity
                                      float poisson_ratio = 0.3f            ///< [in] Poisson ratio
                                      );

    /// Set contact material coefficients.
    /// Alternatively, physical material properties can be set, using the function
    /// SetContactMaterialProperties.
    void SetContactMaterialCoefficients(float kn,  ///< [in] normal contact stiffness
                                        float gn,  ///< [in] normal contact damping
                                        float kt,  ///< [in] tangential contact stiffness
                                        float gt   ///< [in] tangential contact damping
                                        );

    /// Get the tire contact material.
    std::shared_ptr<ChMaterialSurfaceDEM> GetContactMaterial() const { return m_contact_mat; }

    /// Enable/disable tire pressure.
    void EnablePressure(bool val) { m_pressure_enabled = val; }
    bool IsPressureEnabled() const { return m_pressure_enabled; }

    /// Enable/disable tire contact.
    void EnableContact(bool val) { m_contact_enabled = val; }
    bool IsContactEnabled() const { return m_contact_enabled; }

    /// Enable/disable tire-rim connection.
    void EnableRimConnection(bool val) { m_connection_enabled = val; }
    bool IsRimConnectionEnabled() const { return m_connection_enabled; }

    /// Get a handle to the mesh visualization.
    /// Note that this function can only be invoked after initialization.
    std::shared_ptr<fea::ChVisualizationFEAmesh> GetMeshVisualization() { return m_visualization; }

    /// Get the underlying FEA mesh.
    std::shared_ptr<fea::ChMesh> GetMesh() const { return m_mesh; }

    /// Get the mesh contact surface.
    /// If contact is not enabled, an empty shared pointer is returned.
    std::shared_ptr<fea::ChContactSurface> GetContactSurface() const;

    /// Get the load container associated with this tire.
    std::shared_ptr<ChLoadContainer> GetLoadContainer() const { return m_load_container; }

    /// Set the tire pressure.
    void SetPressure(double pressure) {
        assert(m_pressure > 0);
        m_pressure = pressure;
    }

    /// Get the rim radius (inner tire radius).
    virtual double GetRimRadius() const = 0;

    /// Get the tire width.
    virtual double GetWidth() const = 0;

    /// Get total tire mass.
    double GetMass() const;

    /// Get the tire force and moment.
    /// A ChDeformableTire always returns zero forces and moments if the tire is simulated
    /// together with the associated vehicle (the tire forces are implicitly applied
    /// to the associated wheel through the tire-wheel connections). If the tire is
    /// co-simulated, the tire force and moment encapsulate the tire-terrain forces
    /// as well as the weight of the tire itself.
    virtual TireForce GetTireForce(bool cosim = false  ///< [in] indicate if the tire is co-simulated
                                   ) const override;

    /// Initialize this tire system.
    /// This function creates the tire contact shape and attaches it to the
    /// associated wheel body.
    virtual void Initialize(std::shared_ptr<ChBody> wheel,  ///< [in] associated wheel body
                            VehicleSide side                ///< [in] left/right vehicle side
                            ) override;

  protected:
    /// Return the default tire pressure.
    virtual double GetDefaultPressure() const = 0;

    /// Return list of nodes connected to the rim.
    virtual std::vector<std::shared_ptr<fea::ChNodeFEAbase>> GetConnectedNodes() const = 0;

    /// Create the FEA nodes and elements.
    /// The wheel rotational axis is assumed to be the Y axis.
    virtual void CreateMesh(const ChFrameMoving<>& wheel_frame,  ///< [in] frame of associated wheel
                            VehicleSide side                     ///< [in] left/right vehicle side
                            ) = 0;

    /// Create the ChLoad for applying pressure to the tire.
    /// A derived class must create a load and add it to the underlying load container.
    virtual void CreatePressureLoad() = 0;

    /// Create the contact surface for the tire mesh.
    /// A derived class must create a contact surface and add it to the underlying mesh.
    virtual void CreateContactSurface() = 0;

    /// Create the tire-rim connections.
    /// A derived class must create the various constraints between the tire and the
    /// provided wheel body and add them to the underlying system.
    virtual void CreateRimConnections(std::shared_ptr<ChBody> wheel  ///< [in] associated wheel body
                                      ) = 0;

    std::shared_ptr<fea::ChMesh> m_mesh;                                ///< tire mesh
    std::shared_ptr<ChLoadContainer> m_load_container;                  ///< load container (for pressure load)
    std::vector<std::shared_ptr<fea::ChLinkPointFrame>> m_connections;  ///< tire-wheel point connections
    std::vector<std::shared_ptr<fea::ChLinkDirFrame>> m_connectionsD;   ///< tire-wheel direction connections

    bool m_connection_enabled;  ///< enable tire connections to rim
    bool m_pressure_enabled;    ///< enable internal tire pressure
    bool m_contact_enabled;     ///< enable tire-terrain contact

    double m_pressure; ///< internal tire pressure

    ContactSurfaceType m_contact_type;  ///< type of contact surface model (node cloud or mesh)
    double m_contact_node_radius;       ///< node radius (for node cloud contact surface)
    double m_contact_face_thickness;    ///< face thickness (for mesh contact surface)

    bool m_use_mat_props;   ///< specify contact material using physical properties
    float m_friction;       ///< contact coefficient of friction
    float m_restitution;    ///< contact coefficient of restitution
    float m_young_modulus;  ///< contact material Young modulus
    float m_poisson_ratio;  ///< contact material Poisson ratio
    float m_kn;             ///< normal contact stiffness
    float m_gn;             ///< normal contact damping
    float m_kt;             ///< tangential contact stiffness
    float m_gt;             ///< tangential contact damping

    std::shared_ptr<ChMaterialSurfaceDEM> m_contact_mat;           ///< tire contact material
    std::shared_ptr<fea::ChVisualizationFEAmesh> m_visualization;  ///< tire mesh visualization
};

/// @} vehicle_wheeled_tire

}  // end namespace vehicle
}  // end namespace chrono

#endif
