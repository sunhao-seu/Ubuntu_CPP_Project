#include "ros/ros.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <list>
#include <vector>
#include <queue>
#include <math.h>

#include <ctime>

#include <ipa_room_segmentation/room_class.h>
#include <ipa_room_segmentation/abstract_voronoi_segmentation.h>

class VoronoiSegmentation : public AbstractVoronoiSegmentation
{
protected:

public:

	VoronoiSegmentation();

	//the segmentation-algorithm
	void segmentMap(const cv::Mat& map_to_be_labeled, cv::Mat& segmented_map, double map_resolution_from_subscription,
			double room_area_factor_lower_limit, double room_area_factor_upper_limit, int neighborhood_index, int max_iterations,
			double min_critical_point_distance_factor, double max_area_for_merging, bool display_map=false);

	//void draw_segmented_map(const cv::Mat& segmented_map_to_be_draw, std::vector<Room>& rooms, const char *input_name);
	void draw_segmented_line(const cv::Mat& map_to_be_draw, std::vector<std::vector<cv::Point>>& segment_result, const char *input_name);

	void Find_critical_points(const cv::Mat& map, const cv::Mat& distance_map, std::vector<cv::Point>& critical_points, double map_resolution_from_subscription, int neighborhood_index, int max_iterations);

	void Find_my_node_points(const cv::Mat& voronoi_map_backup, const cv::Mat& distance_map, std::vector<cv::Point>& my_true_node_points);

	void ray_occupy_map_func(cv::Point current_point, cv::Mat& ray_cast_occupy_map,  const cv::Mat& original_map);
	
	void SegmentArea(cv::Point node_point, const cv::Mat& ray_cast_occupy_map, cv::Mat& merged_ray_map, std::vector<cv::Point>& overlap_points, int& last_area_id);
};
