%{

/* Includes the header in the wrapper code */
#include "physics/ChLinkLinActuator.h"

%}
 
// Tell SWIG about parent class in Python
%import "ChLinkLock.i"



/* Parse the header file to generate wrappers */
%include "../chrono/physics/ChLinkLinActuator.h"  







