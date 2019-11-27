//area_segmentation_header
#pragma once
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/transforms.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/common/common.h>
#include <pcl/filters/filter.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/conditional_removal.h>
#include "nav_msgs/OccupancyGrid.h"

#include <string>
#include <iostream>

#include <fstream>
#include <cmath>
#include <cstdio>
#include <stdlib.h>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>

#include "VoriGraph.h"
#include "TopoGraph.h"
#include "cgal/CgalVoronoi.h"
#include "cgal/AlphaShape.h"
#include "qt/QImageVoronoi.h"

#include <boost/filesystem.hpp>
#include <boost/iterator/filter_iterator.hpp>
#include <boost/filesystem/path.hpp>

#include <QApplication>

#include <QMessageBox>

#include "RoomDect.h"

#include "roomGraph.h"
#include "Denoise.h"
#include <sys/stat.h>
#include <sys/types.h>

#include "cgal/AlphaShapeRemoval.h"

using namespace std;

#include <sstream>

#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/transforms.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/common/common.h>
#include <pcl/filters/filter.h>
#include "voronoi_segmentation_function.h"
#include "nav_msgs/OccupancyGrid.h"

#include <ros/package.h>
#include <iostream>
#include <list>
#include <vector>
#include <math.h>
#include <fstream>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/algorithm/string.hpp>

#include <ipa_room_segmentation/voronoi_segmentation.h>

#include <ipa_room_segmentation/room_class.h>
#include <ipa_room_segmentation/abstract_voronoi_segmentation.h>

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

void transform_zxy_xyz(PointCloud &cloud_zxy, PointCloud &cloud_xyz);
void pre_process_cloud_points(PointCloud &cloud_in, PointCloud &cloud_out);
void from_3d_to_2d(PointCloud &cloud_in_3d, PointCloud &cloud_out_2d);
void from_2d_to_occupy_map(PointCloud &cloud_in_2d, nav_msgs::OccupancyGrid &occupancy_map);
void from_occupy_map(PointCloud &cloud_in_2d, nav_msgs::OccupancyGrid &occupancy_map);
// bool from_2d_to_voronoi(PointCloud &two_dim_point_cloud, VoriGraph &voriGraph, VoriConfig *sConfig);
// void from_voronoi_to_area(VoriGraph &voriGraph, VoriConfig *sConfig);
void occupy_map_to_qimage(nav_msgs::OccupancyGrid &occupancy_map, const char *input_name);
void occupy_map_to_cvimage(nav_msgs::OccupancyGrid &occupancy_map, const char *input_name);

void occupy_qimage_to_result(const char *input_name);
