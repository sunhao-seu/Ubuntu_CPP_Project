cmake_minimum_required(VERSION 2.8.3)
project(voronoi_segmentation)

set(catkin_RUN_PACKAGES
  pcl_conversions
  pcl_ros
  cv_bridge
  roscpp
  sensor_msgs
  nav_msgs
  roslib
)

set(catkin_BUILD_PACKAGES
	${catkin_RUN_PACKAGES}
)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
	${catkin_BUILD_PACKAGES}
)

# find and include boost, cgal, qt4 library.
find_package( Boost REQUIRED )
if ( NOT Boost_FOUND )
  message(STATUS "This project requires the Boost library, and will not be compiled.")
  return()
endif()

find_package(OpenCV REQUIRED)
if ( NOT OpenCV_FOUND )
  message(STATUS "This project requires the opencv library, and will not be compiled.")
  return()
endif()


find_package(CGAL COMPONENTS Core Boost)
FIND_PACKAGE(Qt4 REQUIRED)

if ( CGAL_FOUND )
  include( ${CGAL_USE_FILE} )
  include( CGAL_CreateSingleSourceCGALProgram )
else()
    message(STATUS "This program requires the CGAL library, and will not be compiled.")
endif()

INCLUDE(${QT_USE_FILE})
ADD_DEFINITIONS(${QT_DEFINITIONS})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# generate a static library called topo_graph_2d;
add_library (topo_graph_2d STATIC  /
  src/VoriConfig.cpp /
  src/VoriGraph.cpp /
  src/TopoGraph.cpp /
  src/RoomDect.cpp /
  src/roomGraph.cpp /
  src/Denoise.cpp /
  src/passageSearch.cpp /
  src/cgal/CgalVoronoi.cpp /
  src/cgal/AlphaShape.cpp /
  src/qt/QImageVoronoi.cpp /
  src/cgal/AlphaShapeRemoval.cpp)



catkin_package(
INCLUDE_DIRS
	common/include
LIBRARIES
CATKIN_DEPENDS
	${catkin_RUN_PACKAGES}
DEPENDS
	OpenCV
	Boost
)

include_directories(
	common/include
	${catkin_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
	${Boost_INCLUDE_DIRS}
  	${CMAKE_CURRENT_SOURCE_DIR}/include
)
#include CGAL path and QT path
include_directories(${CGAL_INCLUDE_DIR} ${QT_INCLUDE_DIR})


message(STATUS "${CMAKE_CURRENT_SOURCE_DIR}")   #print 
message(STATUS "${catkin_LIBRARIES}")   #print 
### add execuable file and related cpp files.
add_executable(${PROJECT_NAME}_node 
src/voronoi_segmentation_function.cpp 
test/voronoi_segmentation_node.cpp
common/src/abstract_voronoi_segmentation.cpp
common/src/voronoi_segmentation.cpp
common/src/wavefront_region_growing.cpp
common/src/contains.cpp
common/src/room_class.cpp
)



###link the target with necessary libraries.
target_link_libraries(${PROJECT_NAME}_node  
  ${catkin_LIBRARIES}
	${Boost_LIBRARIES}
	${OpenCV_LIBRARIES}
	${QT_LIBRARIES} 
	${CGAL_LIBRARIES}
)

target_link_libraries(${PROJECT_NAME}_node
  topo_graph_2d
  ${QT_LIBRARIES} ${CGAL_LIBRARIES} -lboost_unit_test_framework -lboost_filesystem -lboost_system
)





