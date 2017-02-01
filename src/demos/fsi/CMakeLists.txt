# ------------------------------------------------------------------------------
# List of demos according their depencies
# ------------------------------------------------------------------------------

# List all FSI demos that use chrono
SET(FSI_DEMOS
# add fluid demos here
)

# List all FSI demos that use chrono-parallel
SET(FSI_PARALLEL_DEMOS
# add fluid demos here
demo_FSI_CylinderDrop_Implicit
demo_FSI_KitchenSink_Implicit
demo_FSI_Compressibility
demo_FSI_DamBreak
demo_FSI_DamBreak_Flexible_Shell
demo_FSI_Poiseuille_flow
demo_FSI_DamBreak_Explicit
demo_FSI_Compressibility_Explicit
demo_FSI_Validation
demo_FSI_DamBreak_Flexible_Cube
)

SET(FSI_PARALLEL_MKL_DEMOS	
demo_AC_model
demo_FSI_MultuPhysics
testMKL
)

# List all FSI demos use vehicle
SET(FSI_VEHICLE_DEMOS
# add fluid demos here
)

# List all FSI demos use parallel and vehicle
SET(FSI_PARALLEL_VEHICLE_DEMOS
# add fluid demos here
)

#--------------------------------------------------------------
SET(LIBRARIES
    ChronoEngine
    ChronoEngine_fsi
)

IF(ENABLE_MODULE_PARALLEL)
	INCLUDE_DIRECTORIES(${CH_PARALLEL_INCLUDES})
	SET(LIBRARIES ${LIBRARIES} ChronoEngine_parallel)
ENDIF()

IF(ENABLE_MODULE_VEHICLE)
SET(LIBRARIES ${LIBRARIES}
    ChronoEngine_vehicle   
)
ENDIF()

IF(ENABLE_MODULE_OPENGL)
    INCLUDE_DIRECTORIES(${CH_OPENGL_INCLUDES})
    SET(LIBRARIES ${LIBRARIES} ChronoEngine_opengl)
ENDIF()

MESSAGE(STATUS "Demo programs for FSI module...")

# Add executables for demos that have no other dependencies
IF(ENABLE_MODULE_PARALLEL)
	INCLUDE_DIRECTORIES(${CH_PARALLEL_INCLUDES})
	
	IF(ENABLE_MODULE_VEHICLE)
		FOREACH(PROGRAM ${FSI_PARALLEL_VEHICLE_DEMOS})
		    MESSAGE(STATUS "...add ${PROGRAM}")

		    CUDA_ADD_EXECUTABLE(${PROGRAM}  "${PROGRAM}.cpp")
		    SOURCE_GROUP(""  FILES "${PROGRAM}.cpp")

		    SET_TARGET_PROPERTIES(${PROGRAM} PROPERTIES
				FOLDER demos
				COMPILE_FLAGS "${CH_CXX_FLAGS} ${CH_PARALLEL_CXX_FLAGS}"
				LINK_FLAGS "${CH_LINKERFLAG_EXE}"
		    )

		    TARGET_LINK_LIBRARIES(${PROGRAM} ${LIBRARIES})
		    ADD_DEPENDENCIES(${PROGRAM} ${LIBRARIES})

		    INSTALL(TARGETS ${PROGRAM} DESTINATION ${CH_INSTALL_DEMO})

		ENDFOREACH()
	ENDIF()

	IF(ENABLE_MODULE_MKL)
        INCLUDE_DIRECTORIES(${CH_MKL_INCLUDES})
       
       FOREACH(PROGRAM ${FSI_PARALLEL_MKL_DEMOS})
            MESSAGE(STATUS "...add ${PROGRAM}")

            CUDA_ADD_EXECUTABLE(${PROGRAM}  "${PROGRAM}.cpp")
            SOURCE_GROUP(""  FILES  "${PROGRAM}.cpp")    
            SET_TARGET_PROPERTIES(${PROGRAM} PROPERTIES 
                FOLDER demos
                COMPILE_FLAGS "${CH_CXX_FLAGS} ${CH_MKL_CXX_FLAGS} ${CH_PARALLEL_CXX_FLAGS}"
                LINK_FLAGS "${CH_LINKERFLAG_EXE} ${CH_MKL_LINK_FLAGS}")
            TARGET_LINK_LIBRARIES(${PROGRAM}
            	${LIBRARIES}
                ChronoEngine
                ChronoEngine_fea
                ChronoEngine_mkl
                ChronoEngine_parallel
                ChronoEngine_fsi)
            TARGET_INCLUDE_DIRECTORIES(${PROGRAM} PUBLIC
			${CH_MKL_INCLUDES})
            ADD_DEPENDENCIES(${PROGRAM}
             	${LIBRARIES}
                ChronoEngine
                ChronoEngine_fea
                ChronoEngine_mkl
                ChronoEngine_parallel
                ChronoEngine_fsi)
            INSTALL(TARGETS ${PROGRAM} DESTINATION bin)
       ENDFOREACH()
    ENDIF()   
    
    
	FOREACH(PROGRAM ${FSI_PARALLEL_DEMOS})
	    MESSAGE(STATUS "...add ${PROGRAM}")

	    CUDA_ADD_EXECUTABLE(${PROGRAM}  "${PROGRAM}.cpp")
	    SOURCE_GROUP(""  FILES "${PROGRAM}.cpp")

	    SET_TARGET_PROPERTIES(${PROGRAM} PROPERTIES
			FOLDER demos
			COMPILE_FLAGS "${CH_CXX_FLAGS} ${CH_PARALLEL_CXX_FLAGS}"
			LINK_FLAGS "${CH_LINKERFLAG_EXE}"
	    )

	    TARGET_LINK_LIBRARIES(${PROGRAM} ${LIBRARIES})
	    ADD_DEPENDENCIES(${PROGRAM} ${LIBRARIES})

	    INSTALL(TARGETS ${PROGRAM} DESTINATION ${CH_INSTALL_DEMO})

	ENDFOREACH()

ELSE()

	IF(ENABLE_MODULE_VEHICLE)
		FOREACH(PROGRAM ${FSI_VEHICLE_DEMOS})
		    MESSAGE(STATUS "...add ${PROGRAM}")

		    CUDA_ADD_EXECUTABLE(${PROGRAM}  "${PROGRAM}.cpp")
		    SOURCE_GROUP(""  FILES "${PROGRAM}.cpp")

		    SET_TARGET_PROPERTIES(${PROGRAM} PROPERTIES
				FOLDER demos
				COMPILE_FLAGS "${CH_CXX_FLAGS}"
				LINK_FLAGS "${CH_LINKERFLAG_EXE}"
		    )

		    TARGET_LINK_LIBRARIES(${PROGRAM} ${LIBRARIES})
		    ADD_DEPENDENCIES(${PROGRAM} ${LIBRARIES})

		    INSTALL(TARGETS ${PROGRAM} DESTINATION ${CH_INSTALL_DEMO})

		ENDFOREACH()
	ENDIF()


	FOREACH(PROGRAM ${FSI_DEMOS})
	    MESSAGE(STATUS "...add ${PROGRAM}")

	    CUDA_ADD_EXECUTABLE(${PROGRAM}  "${PROGRAM}.cpp")
	    SOURCE_GROUP(""  FILES "${PROGRAM}.cpp")

	    SET_TARGET_PROPERTIES(${PROGRAM} PROPERTIES
			FOLDER demos
			COMPILE_FLAGS "${CH_CXX_FLAGS}"
			LINK_FLAGS "${CH_LINKERFLAG_EXE}"
	    )

	    TARGET_LINK_LIBRARIES(${PROGRAM} ${LIBRARIES})
	    ADD_DEPENDENCIES(${PROGRAM} ${LIBRARIES})

	    INSTALL(TARGETS ${PROGRAM} DESTINATION ${CH_INSTALL_DEMO})

	ENDFOREACH()

ENDIF()




