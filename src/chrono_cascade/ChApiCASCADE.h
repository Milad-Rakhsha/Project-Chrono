//
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2011 Alessandro Tasora
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file at the top level of the distribution
// and at http://projectchrono.org/license-chrono.txt.
//

#ifndef CHAPI_CASCADE_H
#define CHAPI_CASCADE_H

//#pragma warning(disable: 4251)

#include "chrono/core/ChPlatform.h"

// Chrono::Engine unit GPU, version
//
// This is an integer, as 0xaabbccdd where
// for example version 1.2.0 is 0x00010200

#define CH_VERSION_CASCADE_MODULE 0x00010300

// When compiling this library, remember to define CH_API_COMPILE_CASCADE
// (so that the symbols with 'ChApiCASCADE' in front of them will be
// marked as exported). Otherwise, just do not define it if you
// link the library to your code, and the symbols will be imported.

#if defined(CH_API_COMPILE_CASCADE)
#define ChApiCASCADE ChApiEXPORT
#else
#define ChApiCASCADE ChApiIMPORT
#endif

/**
    @defgroup cascade_module CASCADE module
    @brief Tools for interoperation with CAD files. 

    This module allows Finite Element Analysis (FEA) in Chrono::Engine.

    For additional information, see:
    - the [installation guide](@ref module_cascade_installation)
    - the [tutorials](@ref tutorial_root)
*/


namespace chrono {

/// @addtogroup cascade_module
/// @{

/// \brief Namespace with classes for the OpenCASCADE unit.
/// The "cascade" namespace contains tools for interoperation with CAD
/// files. The OpenCASCADE open-source library is used to this end:
/// it can load STEP files saved from most 3D CADs.
namespace cascade {}

/// @}

}



#endif  // END of header