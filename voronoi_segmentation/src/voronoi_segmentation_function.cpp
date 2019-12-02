#include "voronoi_segmentation_function.h"

#define occupy_map_resolution 0.05
// #define occupy_map_width 5000
// #define occupy_map_height 5000
// VoriConfig *sConfig;
#define res 0.05
#define noise_percent 1.5
#define robot_low_height -0.8
#define robot_high_height 0.1

using namespace cv;
using namespace std;

template<typename T>
std::string NumberToString(T Number) {
    std::ostringstream ss;
    ss << Number;
    return ss.str();
}

int nearint(double a) {
    return ceil( a ) - a < 0.5 ? ceil( a ) : floor( a );
}

void transform_zxy_xyz(PointCloud &cloud_zxy, PointCloud &cloud_xyz)
{
    //specially for LOAM output pointcloud
    PointCloud cloud_temp;
    cloud_temp = cloud_zxy;     //without this, the output pointcloud will lose much information, like size...
    for(int i = 0; i < cloud_zxy.points.size(); i ++)
    {
        cloud_temp.points[i].x = cloud_zxy.points[i].z;
        cloud_temp.points[i].y = cloud_zxy.points[i].x;
        cloud_temp.points[i].z = cloud_zxy.points[i].y;
    }
    cloud_xyz = cloud_temp;
}


void pre_process_cloud_points(PointCloud &cloud_in, PointCloud &cloud_out)
{
    //std::cout << "test hahah" << std::endl;

    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(cloud_in, cloud_in, indices);  //remove nan points


    pcl::PointXYZ minPt, maxPt;
    pcl::getMinMax3D(cloud_in, minPt, maxPt);

    float center_x = (maxPt.x - minPt.x)/2;
    float center_y = (maxPt.y - minPt.y)/2;
    float center_z = (maxPt.z - minPt.z)/2;
    // Transform the point cloud
    Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity();
    //transform_2.translation() << -center_x, -center_y, -center_z;
    transform_2.translation() << 0, 0, 0;
    transform_2.rotate (Eigen::AngleAxisf (0.0, Eigen::Vector3f::UnitZ()));
    pcl::transformPointCloud (cloud_in, cloud_in, transform_2);

    //std::cout << "transform2_matrix(): " << transform_2.matrix() << std::endl;
    // std::cout << "pointx before(): " << cloud_load->points[0].x << std::endl;
    // std::cout << "pointx later(): " << processed_cloud->points[1].x << std::endl;
    // std::cout << "pointy before(): " << cloud_load->points[0].y << std::endl;
    // std::cout << "pointy later(): " << processed_cloud->points[1].y << std::endl;
    // std::cout << "pointz before(): " << cloud_load->points[0].z << std::endl;
    // std::cout << "pointz later(): " << processed_cloud->points[1].z << std::endl;

    //remove outlier points
    //基于半径的离群点剔除
		pcl::RadiusOutlierRemoval<pcl::PointXYZ>  rout;
		rout.setInputCloud(cloud_in.makeShared());
		rout.setRadiusSearch(0.5);//设置搜索半径的值
		rout.setMinNeighborsInRadius(5);//设置最小邻居个数，默认是1
		rout.filter(cloud_out);
}

void from_3d_to_2d(PointCloud &cloud_in_3d, PointCloud &cloud_out_2d)
{
    //cloud_out_2d = cloud_in_3d;
    pcl::PointXYZ minPt, maxPt;
    pcl::getMinMax3D(cloud_in_3d, minPt, maxPt);
    float ground_height = minPt.z;
    std::cout << "ground_height " << ground_height <<std::endl;
    std::cout << "floor_height " <<  maxPt.z <<std::endl;
    int cnt = 0;
    for(int i = 0; i < cloud_in_3d.points.size(); i ++)
    {
        if((cloud_in_3d.points[i].z > (robot_low_height) ) && (cloud_in_3d.points[i].z < (robot_high_height)))
        {
            pcl::PointXYZ p;
            p.x = cloud_in_3d.points[i].x;
            p.y = cloud_in_3d.points[i].y; 
            p.z = 0; 
            cloud_out_2d.points.push_back(p);
            cnt ++;          
        }
    }
    cloud_out_2d.width = cnt;
    cloud_out_2d.height = 1;
    cloud_out_2d.is_dense = 1;
}


void from_2d_to_occupy_map(PointCloud &cloud_in_2d, nav_msgs::OccupancyGrid &occupancy_map)
{
    pcl::PointXYZ minPt, maxPt;
    pcl::getMinMax3D(cloud_in_2d, minPt, maxPt);
    // minPt.x = 1000; minPt.y = 1000;
    // maxPt.x = -1000; maxPt.y = -1000; 
    // for(int i = 0; i < cloud_in_2d.points.size(); i ++)
    // {
    //     if(cloud_in_2d.points[i].x < minPt.x)
    //         minPt.x = cloud_in_2d.points[i].x;
    //     if(cloud_in_2d.points[i].y < minPt.y)
    //         minPt.y = cloud_in_2d.points[i].y;
    //     if(cloud_in_2d.points[i].x > maxPt.x)
    //         maxPt.x = cloud_in_2d.points[i].x;
    //     if(cloud_in_2d.points[i].y > maxPt.y)
    //         maxPt.y = cloud_in_2d.points[i].y;
    // }

    //float center_x = (maxPt.x - minPt.x)/2;
    //float center_y = (maxPt.y - minPt.y)/2;
    //+10 
    unsigned int occupy_map_width = ceil( (maxPt.x - minPt.x)/occupy_map_resolution ) +10;
    unsigned int occupy_map_height = ceil( (maxPt.y - minPt.y)/occupy_map_resolution ) + 10;

    occupancy_map.header.frame_id = "global";
    occupancy_map.header.stamp = ros::Time::now();
    occupancy_map.info.resolution = occupy_map_resolution;
    occupancy_map.info.width = occupy_map_width;
    occupancy_map.info.height = occupy_map_height;
    for(int i = 0; i < occupy_map_width; i ++)
    {
        for(int j = 0; j < occupy_map_height; j ++)
        {
            occupancy_map.data.push_back(0);
        }
    }
    // occupancy_map.info.origin.position.x = - occupy_map_width / 2. * occupy_map_resolution;
    // occupancy_map.info.origin.position.y = - occupy_map_height / 2. * occupy_map_resolution;
    // occupancy_map.info.origin.position.z = 0;
    // occupancy_map.info.origin.orientation.x = 0;
    // occupancy_map.info.origin.orientation.y = 0;
    // occupancy_map.info.origin.orientation.z = 0;
    // occupancy_map.info.origin.orientation.w = 1;

    for(int i = 0; i < cloud_in_2d.points.size(); i ++)
    {
        //all points in the map are in forth quadrant, so the [0,0] represent the point at left up corner.
        // the map is actually a matrix...
        unsigned int row_index_map = floor( (cloud_in_2d.points[i].y - minPt.y)/occupy_map_resolution ) +5;
        unsigned int column_index_map = floor( (cloud_in_2d.points[i].x - minPt.x)/occupy_map_resolution ) +5;

        unsigned int map_index = row_index_map * occupy_map_width + column_index_map;
        if( (map_index < occupy_map_width * occupy_map_height) && (map_index > 0) )
            occupancy_map.data[map_index] = 100;
            
    }

}

// bool from_2d_to_voronoi(PointCloud &two_dim_point_cloud, VoriGraph &voriGraph, VoriConfig *sConfig)
// {
//     std::vector<topo_geometry::point> sites;
//     for(int i = 0; i < two_dim_point_cloud.points.size(); i ++)
//     {
//         double point_x = two_dim_point_cloud.points[i].x;
//         double point_y = two_dim_point_cloud.points[i].y;
//         sites.push_back(topo_geometry::point(point_x, point_y));
//     }

//     std::cout << "after sites, two_dim_point_cloud.points.size(): " << two_dim_point_cloud.points.size() << std::endl;
//     bool ret = createVoriGraph( sites, voriGraph, sConfig );
//     return ret;
// }

// void from_voronoi_to_area(VoriGraph &voriGraph, VoriConfig *sConfig)
// {
//     // int remove_alpha_value = 3600;


//     // //QImage alpha = test;
//     // AlphaShapePolygon alphaSP, tem_alphaSP;
//     // //mythink: find the biggest area in the boundary; ???
//     // AlphaShapePolygon::Polygon_2 *poly = alphaSP.performAlpha_biggestArea( alpha, remove_alpha_value, true );
//     // if (poly) {
//     //     cout << "Removing vertices outside of polygon" << endl;
//     //     //mythink: remove the vetices outside the boundary in Voronoi Graph? short edge
//     //     removeOutsidePolygon( voriGraph, *poly );
//     // }
//     // // do alpha-shape algorithm by the alpha parameter calculated before.
//     // // find the edges
//     // AlphaShapePolygon::Polygon_2 *tem_poly = tem_alphaSP.performAlpha_biggestArea( alpha,
//     //                                                                                sConfig->alphaShapeRemovalSquaredSize(),
//     //                                                                                false );
//     // //merge short edges and polygons
//     // voriGraph.joinHalfEdges_jiawei();
//     // cout << "size of Polygons: " << tem_alphaSP.sizeOfPolygons() << endl;

// }

void occupy_map_to_qimage(nav_msgs::OccupancyGrid &occupancy_map, const char *input_name)
{
    int img_height =  occupancy_map.info.height ;
    int img_width = occupancy_map.info.width;

    QImage image(QSize(img_width,img_height),QImage::Format_ARGB32);

    for(int i = 0; i < img_height; i ++)
    {
        for(int j = 0; j < img_width; j ++)
        {
            
            unsigned int map_index = i * img_width + j;
            int pixel_color = occupancy_map.data[map_index];
            if(pixel_color < 50)
                image.setPixel(j, i, qRgb(255, 255, 255));
            else
                image.setPixel(j, i, qRgb(0, 0, 0));
                
        }
    }

    image.save(input_name);

}

void occupy_map_to_cvimage(nav_msgs::OccupancyGrid &occupancy_map, const char *input_name)
{
    int img_height =  occupancy_map.info.height ;
    int img_width = occupancy_map.info.width;

    //QImage image(QSize(img_width,img_height),QImage::Format_ARGB32);
    cv::Mat image(img_height, img_width, CV_8UC1);

    for(int i = 0; i < img_height; i ++)
    {
        for(int j = 0; j < img_width; j ++)
        {
            
            unsigned int map_index = i * img_width + j;
            int pixel_color = occupancy_map.data[map_index];
            if(pixel_color < 50)
                image.at<uchar>(i,j)=255;
            else
                image.at<uchar>(i,j)=0;
                
        }
    }

    int black_threshold = 210;
    //remove the noise from map image
    bool de = DenoiseImg( input_name, input_name, black_threshold, 18, noise_percent );
    if (de)
        cout << "Denoise run successed!!" << endl;

    imwrite(input_name, image);

}

void occupy_qimage_to_result(const char *input_name)
{
    cv::Mat original_img = cv::imread(input_name,0);

    double room_upper_limit_voronoi_ = 1200.0;
    double room_lower_limit_voronoi_ = 2;
    int voronoi_neighborhood_index_ = 310;
    int max_iterations_ = 1000;
    double min_critical_point_distance_factor_ =  1.3;
    double max_area_for_merging_ =  1.3;
    double map_resolution = 0.05;
    cv::Mat segmented_map;

std::cout << "before voronoi " << std::endl;
    VoronoiSegmentation voronoi_segmentation; //voronoi segmentation method
	voronoi_segmentation.segmentMap(original_img, segmented_map, map_resolution, room_lower_limit_voronoi_, room_upper_limit_voronoi_,
		voronoi_neighborhood_index_, max_iterations_, min_critical_point_distance_factor_, max_area_for_merging_, 0);

std::cout << "end voronoi " << std::endl;
}