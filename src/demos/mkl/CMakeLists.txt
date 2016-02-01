#--------------------------------------------------------------
# List of all Irrlicht demos

SET(DEMOS
  demo_ballinbucket
  demo_MklEngine
  demo_bricksIP
  demo_convergenceIP
)

#--------------------------------------------------------------
# Additional include paths

INCLUDE_DIRECTORIES( ${CH_IRRLICHTINC} )
INCLUDE_DIRECTORIES( ${CH_MKLINC} )
INCLUDE_DIRECTORIES( ${CH_MATLABINC} )

#--------------------------------------------------------------

IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  SET (CH_LINKERFLAG_EXE  "${CH_LINKERFLAG_EXE} -framework IOKit -framework Cocoa -framework OpenGL")
ENDIF()

#--------------------------------------------------------------

# Add executables

MESSAGE(STATUS "Demo programs for MKL module...")

FOREACH(PROGRAM ${DEMOS})
  MESSAGE(STATUS "...add ${PROGRAM}")

  ADD_EXECUTABLE(${PROGRAM}  "${PROGRAM}.cpp")
  SOURCE_GROUP(""  FILES  "${PROGRAM}.cpp")

  SET_TARGET_PROPERTIES(${PROGRAM}  PROPERTIES
    FOLDER demos
    COMPILE_FLAGS "${CH_CXX_FLAGS} ${CH_IRRLICHT_CXX_FLAGS}"
    LINK_FLAGS "${CH_LINKERFLAG_EXE}"
    )

  TARGET_LINK_LIBRARIES(${PROGRAM}
    ChronoEngine
    ChronoEngine_irrlicht
	ChronoEngine_mkl
	ChronoEngine_matlab
    ${CH_IRRLICHTLIB}
    )
    
  ADD_DEPENDENCIES (${PROGRAM} ChronoEngine ChronoEngine_irrlicht ChronoEngine_mkl)

  INSTALL(TARGETS ${PROGRAM} DESTINATION bin)

ENDFOREACH()




