%{

/* Includes the header in the wrapper code */
#include "serialization/ChArchive.h"

using namespace chrono;

%}

// Trick to disable a macro that stops SWIG
#define CH_CREATE_MEMBER_DETECTOR(GetRTTI)

/* Parse the header file to generate wrappers */
 %include "../chrono/serialization/ChArchive.h"    


