%{

/* Includes the header in the wrapper code */
#include "physics/ChLink.h"

%}
 
// Tell SWIG about parent class in Python
%import "ChPhysicsItem.i"


/* Parse the header file to generate wrappers */
%include "../chrono/physics/ChLink.h"  







