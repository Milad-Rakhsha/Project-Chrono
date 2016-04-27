#pragma once

#include "chrono/core/ChPlatform.h"

// Chrono::Engine parallel module, version
//
// This is an integer, as 0xaabbccdd where
// for example version 1.2.0 is 0x00010200

#define CH_VERSION_PARALLEL_MODULE 0x00010200

// When compiling this library, remember to define CH_API_COMPILE_PARALLEL
// (so that the symbols with 'CH_PARALLEL_API' in front of them will be
// marked as exported). Otherwise, just do not define it if you
// link the library to your code, and the symbols will be imported.

#if defined(CH_API_COMPILE_PARALLEL)
#define CH_PARALLEL_API ChApiEXPORT
#else
#define CH_PARALLEL_API ChApiIMPORT
#endif

// Macros for specifying type alignment
#if (defined __GNUC__) || (defined __INTEL_COMPILER)
#define CHRONO_ALIGN_16 __attribute__((aligned(16)))
#elif defined _MSC_VER
#define CHRONO_ALIGN_16 __declspec(align(16))
#else
#define CHRONO_ALIGN_16
#endif

#if defined _MSC_VER
#define fmax max
#define fmin min
#endif

#if defined(WIN32) || defined(WIN64)
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
#define ELPP_WINSOCK2
#endif

/**
    @defgroup parallel_module PARALLEL module
    @brief Module that enables parallel computation in Chrono 

    This module implements parallel computing algorithms that can be
    used as a faster alternative to the default simulation algorithms
    in Chrono::Engine. This is achieved using OpenMP, CUDA, Thrust, etc.

    For additional information, see:
    - the [installation guide](@ref module_parallel_installation)
    - the [tutorials](@ref tutorial_root)
*/