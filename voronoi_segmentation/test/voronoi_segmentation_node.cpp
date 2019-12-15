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

// #include <ros/package.h>
// #include <iostream>
// #include <list>
// #include <vector>
// #include <math.h>
// #include <fstream>
// #include <string>

// #include <opencv2/opencv.hpp>
// #include <opencv2/highgui/highgui.hpp>

// #include <boost/algorithm/string.hpp>

// #include <ipa_room_segmentation/voronoi_segmentation.h>



ros::Publisher pcd_pub;
ros::Publisher occupy_map_pub;
ros::Publisher pub_voronio;


void pcdLoadCallback(const ros::TimerEvent&){
  // PointCloud::Ptr cloud_load(new PointCloud);

///home/wolfhao/Documents/area_segmentation_file/2F_gt.pcd
///home/wolfhao/Documents/area_segmentation_file/PCD/outside_Map_pcd.pcd
  // if (pcl::io::loadPCDFile<pcl::PointXYZ>("/home/wolfhao/Documents/area_segmentation_file/2F_gt.pcd", *cloud_load) == -1){
  //   PCL_ERROR("Couldn't read pcd file\n");
  // }

  // PointCloud::Ptr cloud_load_transform(new PointCloud);
  // PointCloud::Ptr processed_cloud(new PointCloud);
  // PointCloud::Ptr two_dim_point_cloud(new PointCloud);
  // nav_msgs::OccupancyGrid occupancy_map;
  
  // //transform_zxy_xyz(*cloud_load, *cloud_load_transform);
  // *cloud_load_transform = *cloud_load;
  // pre_process_cloud_points(*cloud_load_transform, *processed_cloud);
  // from_3d_to_2d(*processed_cloud, *two_dim_point_cloud);
  // from_2d_to_occupy_map(*two_dim_point_cloud, occupancy_map);
  // std::cout << "before create png" << std::endl;  
  // //occupy_map_to_qimage(occupancy_map, "2dmap.png");
  // occupy_map_to_cvimage(occupancy_map, "2dmap.png");

  // std::cout << "before occupy_qimage_to_result" << std::endl;  


  //occupy_qimage_to_result("/home/wolfhao/Pictures/picture/6.png");
  
  // Pub the point cloud
  // PointCloud cloud = *cloud_load_transform;
  // cloud.header.frame_id = "global";
  // pcd_pub.publish(cloud);
  // occupy_map_pub.publish(occupancy_map);
}


int main(int argc, char** argv)
{
//   // Make a ROS node, which we'll use to publish copies of the data in the CollisionMap and SDF
//   // and Rviz markers that allow us to visualize them.
//   ros::init(argc, argv, "area_segmentation");
//   // Get a handle to the current node
//   ros::NodeHandle nh;
//   // Create the timer to load pcd
//   ros::Timer estimate_update_timer = nh.createTimer(ros::Duration(5.0), pcdLoadCallback);
//   // Make a publisher for point cloud
//   pcd_pub = nh.advertise<PointCloud>("pcd_pre_processed", 1, false);
//   occupy_map_pub = nh.advertise<nav_msgs::OccupancyGrid>("occupy_map", 1, false);
//   // Make a subscriber for point cloud
//  // ros::Subscriber cloud_sub = nh.subscribe<sensor_msgs::PointCloud2>("pcd", 10, pcdCallback);

//  //ros::Subscriber cloud_grid_map = nh.subscribe<PointCloud>("cloud_grid_map", 10, cloudCallback);
  
//   ros::spin();
    if (argc < 2) 
    {
        std::cout << "YOu need input the input photo path as a argument..." << std::endl;
        std::cout << "Press anykey to exit " << std::endl;
        getchar();
        return -1;
    }

    occupy_qimage_to_result(argv[1]);

    return 0;
}