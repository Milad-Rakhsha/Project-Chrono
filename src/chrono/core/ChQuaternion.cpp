//
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2010 Alessandro Tasora
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file at the top level of the distribution
// and at http://projectchrono.org/license-chrono.txt.
//

///////////////////////////////////////////////////
//
//   ChQuaternion.cpp
//
//   Math functions for:
//	 - QUATERNIONS
//
// ------------------------------------------------
//             www.deltaknowledge.com
// ------------------------------------------------
///////////////////////////////////////////////////

#include <math.h>

#include "core/ChQuaternion.h"
#include "core/ChMatrix33.h"

namespace chrono {

///////////////////////////////////////////////////
// QUATERNION OPERATIONS

double Qlength(const ChQuaternion<double>& q) {
    return (sqrt(pow(q.e0, 2) + pow(q.e1, 2) + pow(q.e2, 2) + pow(q.e3, 2)));
}

ChQuaternion<double> Qscale(const ChQuaternion<double>& q, double fact) {
    ChQuaternion<double> result;
    result.e0 = q.e0 * fact;
    result.e1 = q.e1 * fact;
    result.e2 = q.e2 * fact;
    result.e3 = q.e3 * fact;
    return result;
}

ChQuaternion<double> Qadd(const ChQuaternion<double>& qa, const ChQuaternion<double>& qb) {
    ChQuaternion<double> result;
    result.e0 = qa.e0 + qb.e0;
    result.e1 = qa.e1 + qb.e1;
    result.e2 = qa.e2 + qb.e2;
    result.e3 = qa.e3 + qb.e3;
    return result;
}

ChQuaternion<double> Qsub(const ChQuaternion<double>& qa, const ChQuaternion<double>& qb) {
    ChQuaternion<double> result;
    result.e0 = qa.e0 - qb.e0;
    result.e1 = qa.e1 - qb.e1;
    result.e2 = qa.e2 - qb.e2;
    result.e3 = qa.e3 - qb.e3;
    return result;
}
// returns the norm two of the quaternion. Eulero's parameters have norm = 1

ChQuaternion<double> Qnorm(const ChQuaternion<double>& q) {
    double invlength;
    invlength = 1 / (Qlength(q));
    return Qscale(q, invlength);
}

// The conjugate of the quaternion [s,v1,v2,v3] is [s,-v1,-v2,-v3]

ChQuaternion<double> Qconjugate(const ChQuaternion<double>& q) {
    ChQuaternion<double> res;
    res.e0 = q.e0;
    res.e1 = -q.e1;
    res.e2 = -q.e2;
    res.e3 = -q.e3;
    return (res);
}

// Returns the product of two quaternions. It is NONcommutative! (like cross
// product in vectors.

ChQuaternion<double> Qcross(const ChQuaternion<double>& qa, const ChQuaternion<double>& qb) {
    ChQuaternion<double> res;
    res.e0 = qa.e0 * qb.e0 - qa.e1 * qb.e1 - qa.e2 * qb.e2 - qa.e3 * qb.e3;
    res.e1 = qa.e0 * qb.e1 + qa.e1 * qb.e0 - qa.e3 * qb.e2 + qa.e2 * qb.e3;
    res.e2 = qa.e0 * qb.e2 + qa.e2 * qb.e0 + qa.e3 * qb.e1 - qa.e1 * qb.e3;
    res.e3 = qa.e0 * qb.e3 + qa.e3 * qb.e0 - qa.e2 * qb.e1 + qa.e1 * qb.e2;
    return (res);
}

// Gets the quaternion from an agle of rotation and an axis,
// defined in _abs_ coords. The axis is supposed to be fixed, i.e.
// it is constant during rotation! And must be normalized!

ChQuaternion<double> Q_from_AngAxis(double angle, const ChVector<double>& axis) {
    ChQuaternion<double> quat;
    double halfang;
    double sinhalf;

    halfang = (angle * 0.5);
    sinhalf = sin(halfang);

    quat.e0 = cos(halfang);
    quat.e1 = axis.x * sinhalf;
    quat.e2 = axis.y * sinhalf;
    quat.e3 = axis.z * sinhalf;
    return (quat);
}

ChQuaternion<double> Q_from_AngZ(double angleZ) {
    return Q_from_AngAxis(angleZ, VECT_Z);
}
ChQuaternion<double> Q_from_AngX(double angleX) {
    return Q_from_AngAxis(angleX, VECT_X);
}
ChQuaternion<double> Q_from_AngY(double angleY) {
    return Q_from_AngAxis(angleY, VECT_Y);
}

ChQuaternion<double> Q_from_NasaAngles(const ChVector<double>& mang) {
    ChQuaternion<double> mq;
    double c1 = cos(mang.z / 2);
    double s1 = sin(mang.z / 2);
    double c2 = cos(mang.x / 2);
    double s2 = sin(mang.x / 2);
    double c3 = cos(mang.y / 2);
    double s3 = sin(mang.y / 2);
    double c1c2 = c1 * c2;
    double s1s2 = s1 * s2;
    mq.e0 = c1c2 * c3 + s1s2 * s3;
    mq.e1 = c1c2 * s3 - s1s2 * c3;
    mq.e2 = c1 * s2 * c3 + s1 * c2 * s3;
    mq.e3 = s1 * c2 * c3 - c1 * s2 * s3;
    return mq;
}
ChVector<double> Q_to_NasaAngles(const ChQuaternion<double>& q1) {
    ChVector<double> mnasa;
    double sqw = q1.e0 * q1.e0;
    double sqx = q1.e1 * q1.e1;
    double sqy = q1.e2 * q1.e2;
    double sqz = q1.e3 * q1.e3;
    // heading
    mnasa.z = atan2(2.0 * (q1.e1 * q1.e2 + q1.e3 * q1.e0), (sqx - sqy - sqz + sqw));
    // bank
    mnasa.y = atan2(2.0 * (q1.e2 * q1.e3 + q1.e1 * q1.e0), (-sqx - sqy + sqz + sqw));
    // attitude
    mnasa.x = asin(-2.0 * (q1.e1 * q1.e3 - q1.e2 * q1.e0));
    return mnasa;
}

void Q_to_AngAxis(ChQuaternion<double>* quat, double* a_angle, ChVector<double>* a_axis) {
    double arg, invsine;
    ChVector<double> vtemp;

    if (quat->e0 < 0.99999999) {
        arg = acos(quat->e0);
        *a_angle = 2 * arg;
        invsine = 1 / (sin(arg));
        vtemp.x = invsine * quat->e1;
        vtemp.y = invsine * quat->e2;
        vtemp.z = invsine * quat->e3;
        *a_axis = Vnorm(vtemp);
    } else {
        a_axis->x = 1;
        a_axis->y = 0;
        a_axis->z = 0;
        *a_angle = 0;
    }
}

//	Gets the four quaternion dq/dt from the vector of angular speed,
// with w specified in _absolute_ coords.

ChQuaternion<double> Qdt_from_Wabs(const ChVector<double>& w, const ChQuaternion<double>& q) {
    ChQuaternion<double> qw;
    double half = 0.5;

    qw.e0 = 0;
    qw.e1 = w.x;
    qw.e2 = w.y;
    qw.e3 = w.z;

    return Qscale(Qcross(qw, q), half);  // {q_dt} = 1/2 {0,w}*{q}
}

//	Gets the four quaternion dq/dt from the vector of angular speed,
// with w specified in _local_ coords.

ChQuaternion<double> Qdt_from_Wrel(const ChVector<double>& w, const ChQuaternion<double>& q) {
    ChQuaternion<double> qw;
    double half = 0.5;

    qw.e0 = 0;
    qw.e1 = w.x;
    qw.e2 = w.y;
    qw.e3 = w.z;

    return Qscale(Qcross(q, qw), half);  // {q_dt} = 1/2 {q}*{0,w_rel}
}

//	Gets the quaternion ddq/dtdt from the vector of angular acceleration
//  with a specified in _absolute_ coords.

ChQuaternion<double> Qdtdt_from_Aabs(const ChVector<double>& a, const ChQuaternion<double>& q, const ChQuaternion<double>& q_dt) {
    ChQuaternion<double> ret;
    ret.Qdtdt_from_Aabs(a, q, q_dt);
    return ret;
}

//	Gets the quaternion ddq/dtdt from the vector of angular acceleration
//  with a specified in _relative_ coords.

ChQuaternion<double> Qdtdt_from_Arel(const ChVector<double>& a, const ChQuaternion<double>& q, const ChQuaternion<double>& q_dt) {
    ChQuaternion<double> ret;
    ret.Qdtdt_from_Arel(a, q, q_dt);
    return ret;
}

// Gets the dquaternion/dt from a quaternion, a speed of
// rotation and an axis, defined in _abs_ coords.

ChQuaternion<double> Qdt_from_AngAxis(const ChQuaternion<double>& quat, double angle_dt, const ChVector<double>& axis) {
    ChVector<double> W;

    W = Vmul(axis, angle_dt);

    return (chrono::Qdt_from_Wabs(W, quat));
}

// Gets the ddquaternion/dtdt from a quaternion, an angular
// acceleration and an axis, defined in _abs_ coords.

ChQuaternion<double> Qdtdt_from_AngAxis(double angle_dtdt, const ChVector<double>& axis, const ChQuaternion<double>& q, const ChQuaternion<double>& q_dt) {
    ChVector<double> Acc;

    Acc = Vmul(axis, angle_dtdt);

    return (chrono::Qdtdt_from_Aabs(Acc, q, q_dt));
}

// Returns TRUE if two quaternions are equal

bool Qequal(const ChQuaternion<double>& qa, const ChQuaternion<double>& qb) {
    return qa == qb;
}

// Returns TRUE if quaternion is not null;

bool Qnotnull(const ChQuaternion<double>& qa) {
    return (qa.e0 != 0) || (qa.e1 != 0) || (qa.e2 != 0) || (qa.e3 != 0);
}

// Given the immaginary (vectorial) {e1 e2 e3} part of a quaternion, tries to find the
// entire quaternion q = {e0, e1, e2, e3}. Also for q_dt and q_dtdt
// Note: singularities may happen!

ChQuaternion<double> ImmQ_complete(ChVector<double>* qimm) {
    ChQuaternion<double> mq;
    mq.e1 = qimm->x;
    mq.e2 = qimm->y;
    mq.e3 = qimm->z;
    mq.e0 = sqrt(1 - mq.e1 * mq.e1 - mq.e2 * mq.e2 - mq.e3 * mq.e3);
    return (mq);
}

ChQuaternion<double> ImmQ_dt_complete(ChQuaternion<double>* mq, ChVector<double>* qimm_dt) {
    ChQuaternion<double> mqdt;
    mqdt.e1 = qimm_dt->x;
    mqdt.e2 = qimm_dt->y;
    mqdt.e3 = qimm_dt->z;
    mqdt.e0 = (-mq->e1 * mqdt.e1 - mq->e2 * mqdt.e2 - mq->e3 * mqdt.e3) / mq->e0;
    return (mqdt);
}

ChQuaternion<double> ImmQ_dtdt_complete(ChQuaternion<double>* mq, ChQuaternion<double>* mqdt, ChVector<double>* qimm_dtdt) {
    ChQuaternion<double> mqdtdt;
    mqdtdt.e1 = qimm_dtdt->x;
    mqdtdt.e2 = qimm_dtdt->y;
    mqdtdt.e3 = qimm_dtdt->z;
    mqdtdt.e0 = (-mq->e1 * mqdtdt.e1 - mq->e2 * mqdtdt.e2 - mq->e3 * mqdtdt.e3 - mqdt->e0 * mqdt->e0 -
                 mqdt->e1 * mqdt->e1 - mqdt->e2 * mqdt->e2 - mqdt->e3 * mqdt->e3) /
                mq->e0;
    return (mqdtdt);
}

////////////////////////////////////////////////////////////
//  ANGLE CONVERSION UTILITIES

ChQuaternion<double> Angle_to_Quat(int angset, const ChVector<double>& mangles) {
    ChQuaternion<double> res;
    ChMatrix33<> Acoord;

    switch (angset) {
        case ANGLESET_EULERO:
            Acoord.Set_A_Eulero(mangles);
            break;
        case ANGLESET_CARDANO:
            Acoord.Set_A_Cardano(mangles);
            break;
        case ANGLESET_HPB:
            Acoord.Set_A_Hpb(mangles);
            break;
        case ANGLESET_RXYZ:
            Acoord.Set_A_Rxyz(mangles);
            break;
        case ANGLESET_RODRIGUEZ:
            Acoord.Set_A_Rodriguez(mangles);
            break;
    }
    res = Acoord.Get_A_quaternion();
    return res;
}

ChVector<double> Quat_to_Angle(int angset, const ChQuaternion<double>& mquat) {
    ChVector<double> res;
    ChMatrix33<> Acoord;

    Acoord.Set_A_quaternion(mquat);

    switch (angset) {
        case ANGLESET_EULERO:
            res = Acoord.Get_A_Eulero();
            break;
        case ANGLESET_CARDANO:
            res = Acoord.Get_A_Cardano();
            break;
        case ANGLESET_HPB:
            res = Acoord.Get_A_Hpb();
            break;
        case ANGLESET_RXYZ:
            res = Acoord.Get_A_Rxyz();
            break;
        case ANGLESET_RODRIGUEZ:
            res = Acoord.Get_A_Rodriguez();
            break;
    }
    return res;
}

ChQuaternion<double> AngleDT_to_QuatDT(int angset, const ChVector<double>& mangles, const ChQuaternion<double>& q) {
    ChQuaternion<double> res;
    ChQuaternion<double> q2;
    ChVector<double> ang1, ang2;

    ang1 = Quat_to_Angle(angset, q);
    ang2 = Vadd(ang1, Vmul(mangles, CH_LOWTOL));
    q2 = Angle_to_Quat(angset, ang2);
    res = Qscale(Qsub(q2, q), (1 / CH_LOWTOL));

    return res;
}

ChQuaternion<double> AngleDTDT_to_QuatDTDT(int angset, const ChVector<double>& mangles, const ChQuaternion<double>& q) {
    ChQuaternion<double> res;
    ChQuaternion<double> qa, qb;
    ChVector<double> ang0, angA, angB;
    double hsquared = CH_LOWTOL;

    ang0 = Quat_to_Angle(angset, q);
    angA = Vsub(ang0, Vmul(mangles, hsquared));
    angB = Vadd(ang0, Vmul(mangles, hsquared));
    qa = Angle_to_Quat(angset, angA);
    qb = Angle_to_Quat(angset, angB);
    res = Qscale(Qadd(Qadd(qa, qb), Qscale(q, -2)), 1 / hsquared);

    return res;
}

ChVector<double> Angle_to_Angle(int setfrom, int setto, const ChVector<double>& mangles) {
    ChVector<double> res;
    ChMatrix33<> Acoord;

    switch (setfrom) {
        case ANGLESET_EULERO:
            Acoord.Set_A_Eulero(mangles);
            break;
        case ANGLESET_CARDANO:
            Acoord.Set_A_Cardano(mangles);
            break;
        case ANGLESET_HPB:
            Acoord.Set_A_Hpb(mangles);
            break;
        case ANGLESET_RXYZ:
            Acoord.Set_A_Rxyz(mangles);
            break;
        case ANGLESET_RODRIGUEZ:
            Acoord.Set_A_Rodriguez(mangles);
            break;
    }

    switch (setto) {
        case ANGLESET_EULERO:
            res = Acoord.Get_A_Eulero();
            break;
        case ANGLESET_CARDANO:
            res = Acoord.Get_A_Cardano();
            break;
        case ANGLESET_HPB:
            res = Acoord.Get_A_Hpb();
            break;
        case ANGLESET_RXYZ:
            res = Acoord.Get_A_Rxyz();
            break;
        case ANGLESET_RODRIGUEZ:
            res = Acoord.Get_A_Rodriguez();
            break;
    }
    return res;
}

// Get the X axis of a coordsystem, given the quaternion which
// represents the alignment of the coordsystem.
ChVector<double> VaxisXfromQuat(const ChQuaternion<double>& quat) {
    ChVector<double> res;
    res.x = (pow(quat.e0, 2) + pow(quat.e1, 2)) * 2 - 1;
    res.y = ((quat.e1 * quat.e2) + (quat.e0 * quat.e3)) * 2;
    res.z = ((quat.e1 * quat.e3) - (quat.e0 * quat.e2)) * 2;
    return res;
}

}  // END_OF_NAMESPACE____

////////
