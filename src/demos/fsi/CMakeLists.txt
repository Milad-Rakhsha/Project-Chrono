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
demo_FSI_cylinderDrop
demo_FSI_DamBreak
demo_FSI_CylinderDrop_Implicit
)

SET(FSI_FEA_DEMOS
demo_FSI_Flexible_Shell
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

	IF(ENABLE_MODULE_MKL AND ENABLE_MODULE_FEA)
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
    
   	IF(ENABLE_MODULE_FEA)
		FOREACH(PROGRAM ${FSI_FEA_DEMOS})
		    MESSAGE(STATUS "...add ${PROGRAM}")

		    CUDA_ADD_EXECUTABLE(${PROGRAM}  "${PROGRAM}.cpp")
		    SOURCE_GROUP(""  FILES "${PROGRAM}.cpp")

			SET_TARGET_PROPERTIES(${PROGRAM} PROPERTIES 
                FOLDER demos
                COMPILE_FLAGS "${CH_CXX_FLAGS} ${CH_FEA_CXX_FLAGS}"
                LINK_FLAGS "${CH_LINKERFLAG_EXE} ${CH_FEA_LINK_FLAGS}")
            TARGET_LINK_LIBRARIES(${PROGRAM}
            	${LIBRARIES}
                ChronoEngine
                ChronoEngine_fea
                ChronoEngine_parallel
                ChronoEngine_fsi)
            TARGET_INCLUDE_DIRECTORIES(${PROGRAM} PUBLIC
			${CH_FEA_INCLUDES})
            ADD_DEPENDENCIES(${PROGRAM}
             	${LIBRARIES}
                ChronoEngine
                ChronoEngine_fea
                ChronoEngine_parallel
                ChronoEngine_fsi)
            
		    TARGET_LINK_LIBRARIES(${PROGRAM} ${LIBRARIES})
		    ADD_DEPENDENCIES(${PROGRAM} ${LIBRARIES})
		    INSTALL(TARGETS ${PROGRAM} DESTINATION ${CH_INSTALL_DEMO})

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




