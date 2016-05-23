#=============================================================================
# CMake configuration file for Chrono FSI module
#
# Cannot be used stand-alone (it is loaded by parent CMake configuration file)
#=============================================================================

option(ENABLE_MODULE_FSI "Enable the Chrono FSI module" OFF)

# Return now if this module is not enabled
if(NOT ENABLE_MODULE_FSI)
  return()
endif()

message(STATUS "==== Chrono FSI module ====")

# ----------------------------------------------------------------------------
# CUDA STUFF , Arman Take care of this
# ----------------------------------------------------------------------------

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})

OPTION(ENABLE_CUDA "Cuda Support" ON)
IF(ENABLE_CUDA)
find_package(CUDA)

SET(CHRONO_INCLUDE_DIRS 
	${CHRONO_INCLUDE_DIRS} 
	"${CUDA_TOOLKIT_ROOT_DIR}/include"
    	"${CUDA_SDK_ROOT_DIR}/common/inc")

SET(SAP_INC_DIR "" CACHE PATH "Where is the SAP include directory?")
INCLUDE_DIRECTORIES(${SAP_INC_DIR})


SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")
option(CUDA_PROPAGATE_HOST_FLAGS "set host flags off" FALSE)
ENDIF()
# ----------------------------------------------------------------------------
# OpenGL
# ----------------------------------------------------------------------------
IF(ENABLE_MODULE_OPENGL)
    INCLUDE_DIRECTORIES(${CH_OPENGL_INCLUDES})
    SET(LIBRARIES ${LIBRARIES} ChronoEngine_opengl)
ENDIF()
# ----------------------------------------------------------------------------
# Generate and install configuration file
# ----------------------------------------------------------------------------

# Generate the configuration header file using substitution variables.

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ChConfigFSI.h.in
               ${PROJECT_BINARY_DIR}/chrono_fsi/ChConfigFSI.h)

install(FILES "${PROJECT_BINARY_DIR}/chrono_fsi/ChConfigFSI.h"
        DESTINATION include/chrono_vehicle)

#-----------------------------------------------------------------------------
# LIST THE FILES THAT MAKE THE FSI FLUID-SOLID INTERACTION LIBRARY

SET(ChronoEngine_FSI_SOURCES
checkPointReduced.cpp 
ChFsiModelData.cpp
CreateBCE.cpp
InitializeSphMarkers.cpp
printToFile.cu
SphInterface.cpp 
VehicleExtraProperties.cpp
UtilsDeviceOperations.cu
#SDKCollisionSystem.cu 
#collideSphereSphere.cu
#FSI_Integrate.cpp
incompressible_collisionSystem.cu
incompressible_collideSphereSphere.cu
incompressible_integrate.cpp
)

SET(ChronoEngine_FSI_HEADERS
checkPointReduced.h
ChFsiModelData.h
CreateBCE.h
custom_cutil_math.h
demo_utils.h
InitializeSphMarkers.h
MyStructs.cuh
printToFile.cu
SPHCudaUtils.h
SphInterface.h
VehicleExtraProperties.h
UtilsDeviceOperations.cuh
#SDKCollisionSystem.cuh
#collideSphereSphere.cuh
#FSI_Integrate.h
incompressible_collisionSystem.cuh
incompressible_collideSphereSphere.cuh
incompressible_integrate.h
)

SOURCE_GROUP("" FILES 
    ${ChronoEngine_FSI_SOURCES} 
    ${ChronoEngine_FSI_HEADERS})

#-----------------------------------------------------------------------------	

#-----------------------------------------------------------------------------	
# Create the ChronoEngine_fsi library

CUDA_ADD_LIBRARY(ChronoEngine_fsi SHARED 
    ${ChronoEngine_FSI_SOURCES}
    ${ChronoEngine_FSI_HEADERS})

SET_TARGET_PROPERTIES(ChronoEngine_fsi PROPERTIES
                      COMPILE_FLAGS "${CH_CXX_FLAGS}"
                      LINK_FLAGS "${CH_LINKERFLAG_SHARED}" 
                      COMPILE_DEFINITIONS "CH_API_COMPILE_FEA")
                          
TARGET_LINK_LIBRARIES(ChronoEngine_fsi
                      ChronoEngine)

ADD_DEPENDENCIES(ChronoEngine_fsi ChronoEngine)

INSTALL(TARGETS ChronoEngine_fsi
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

INSTALL(FILES ${ChronoEngine_FSI_HEADERS}
        DESTINATION include/chrono_fsi)
