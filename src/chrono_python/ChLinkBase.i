%{

/* Includes the header in the wrapper code */
#include "physics/ChLinkBase.h"

%}

// Forward ref 
//%import "ChPhysicsItem.i" // parent class does not need %import if all .i are included in proper order


/* Parse the header file to generate wrappers */
%include "../chrono/physics/ChLinkBase.h"  






