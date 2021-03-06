//
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2010-2011 Alessandro Tasora
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be 
// found in the LICENSE file at the top level of the distribution
// and at http://projectchrono.org/license-chrono.txt.
//

#ifndef CHLINKFASTCONTACT_H
#define CHLINKFASTCONTACT_H

///////////////////////////////////////////////////
//
//   ChLinkFastContact.h
//
//   Classes for enforcing constraints (contacts)
//   created by collision detection.
//
//   HEADER file for CHRONO,
//	 Multibody dynamics engine
//
// ------------------------------------------------
//             www.deltaknowledge.com
// ------------------------------------------------
///////////////////////////////////////////////////

//***OBSOLETE***  not used anymore - see ChConcactContainer.h and ChContact.h

#include "physics/ChLinkContact.h"
#include "lcp/ChLcpConstraintTwoContactN.h"

namespace chrono
{

// Unique link identifier, for detecting type faster than with rtti.
#define LNK_FASTCONTACT	35

///
/// Class representing an unilateral contact constraint. 
/// Since such objects are automatically generated by the
/// collision detection engine (this type of constraint is 
/// automatically removed from the ChSystem list of links at each
/// step, because their creation/deletion is automated by the
/// collision engine. So, you most often do not need to create any
/// ChLinkFastContact)
///

class ChApi ChLinkFastContact : public ChLinkContact {

	CH_RTTI(ChLinkFastContact,ChLinkContact);

protected:
				//
	  			// DATA
				//

							// the collision pair which generated the link
	ChCollisionPair	collision_pair;

							// the plane of contact (X is normal direction)
	ChMatrix33<float> contact_plane;
	
							// The three scalar constraints, to be feed into the 
							// system solver. They contain jacobians data and special functions.
	ChLcpConstraintTwoContactN  Nx;
	ChLcpConstraintTwoFrictionT Tu;
	ChLcpConstraintTwoFrictionT Tv; 

public:
				//
	  			// CONSTRUCTORS
				//

	ChLinkFastContact ();

	ChLinkFastContact (ChCollisionPair* mpair, 
							ChBody* mbody1, 
							ChBody* mbody2);

	virtual ~ChLinkFastContact ();
	virtual void Copy(ChLinkFastContact* source);
	virtual ChLink* new_Duplicate ();	// always return base link class pointer


				//
	  			// FUNCTIONS
				//

	virtual int GetType	() {return LNK_FASTCONTACT;}

					/// Initialize again this constraint.
	virtual void Reset(ChCollisionPair* mpair, ChBody* mbody1, ChBody* mbody2);


					/// Get the link coordinate system, expressed relative to Body2 (the 'master'
					/// body). This represents the 'main' reference of the link: reaction forces 
					/// are expressed in this coordinate system.
					/// (It is the coordinate system of the contact plane relative to Body2)
	virtual ChCoordsys<> GetLinkRelativeCoords();

					/// Returns the pointer to a contained 3x3 matrix representing the UV and normal
					/// directions of the contact. In detail, the X versor (the 1s column of the 
					/// matrix) represents the direction of the contact normal.
	ChMatrix33<float>* GetContactPlane() {return &contact_plane;};

					/// Returns the ChCollisionPair info of this contact.
	ChCollisionPair* GetCollisionPair() {return &collision_pair;};

	virtual ChVector<> GetContactP1() {return collision_pair.p1; };
	virtual ChVector<> GetContactP2() {return collision_pair.p2; };
	virtual ChVector<float> GetContactNormal()  {return collision_pair.normal; };
	virtual double	   GetContactDistance()  {return collision_pair.norm_dist; };

				//
				// UPDATING FUNCTIONS
				//

					/// Override _all_ time, jacobian etc. updating.
					/// In detail, it computes jacobians, violations, etc. and stores 
					/// results in inner structures.
	virtual void Update (double mtime);


				//
				// LCP INTERFACE
				//

	virtual void InjectConstraints(ChLcpSystemDescriptor& mdescriptor);
	virtual void ConstraintsBiReset();
	virtual void ConstraintsBiLoad_C(double factor=1., double recovery_clamp=0.1, bool do_clamp=false);
	virtual void ConstraintsBiLoad_Ct(double factor=1.);
	//virtual void ConstraintsFbLoadForces(double factor=1.);
	virtual void ConstraintsLoadJacobians();
	virtual void ConstraintsLiLoadSuggestedSpeedSolution();
	virtual void ConstraintsLiLoadSuggestedPositionSolution();
	virtual void ConstraintsLiFetchSuggestedSpeedSolution();
	virtual void ConstraintsLiFetchSuggestedPositionSolution();
	virtual void ConstraintsFetch_react(double factor=1.);


};




//////////////////////////////////////////////////////
//////////////////////////////////////////////////////


} // END_OF_NAMESPACE____

#endif
