%{

/* Includes the header in the wrapper code */
#include "physics/ChShaftsCouple.h"

%}
 
// Forward ref (parent class does not need %import if all .i are included in proper order
//%import "ChPhysicsItem.i"


/* Parse the header file to generate wrappers */
%include "../physics/ChShaftsCouple.h"  







