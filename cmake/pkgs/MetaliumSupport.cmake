option(USE_TT_METAL "Enable support for TT-Metalium")

if (USE_TT_METAL)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    add_definitions(-DUSE_TT_METAL)
    #add_definitions(-DPRINT_DEBUG) #Uncomment to Enable Debug Prints
endif()
