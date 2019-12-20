#include <ipa_room_segmentation/voronoi_segmentation.h>

#include <ipa_room_segmentation/wavefront_region_growing.h>
#include <ipa_room_segmentation/contains.h>

#include <ipa_room_segmentation/timer.h>
#include <set>


#define k_robot_width 0.3
#define k_robot_length 0.5
#define k_filter_node_range 3.1
#define k_select_contour_distance_low_bound k_robot_width
#define k_select_contour_distance_high_bound 2
#define k_wave_step 2
#define k_wave_biggest_step 1000

VoronoiSegmentation::VoronoiSegmentation()
{

}

void VoronoiSegmentation::RayCannySegmentMap(const cv::Mat& original_map, cv::Mat& segmented_map, double map_resolution_from_subscription)
{
	/*
	
	*/
	cv::Mat map_to_be_labeled = original_map.clone();

	//1: Get the distance map
	cv::Mat distance_map; //distance-map of the original-map (used to check the distance of each point to nearest black pixel)
	cv::distanceTransform(map_to_be_labeled, distance_map, CV_DIST_L2, 5);
	cv::convertScaleAbs(distance_map, distance_map);
imwrite("distance_map.png", distance_map);	//  distance to the nearest zero point(black point)(obstacle)()


	//2:Get voronoi graph and dead end points
	cv::Mat voronoi_map = original_map.clone();
	createVoronoiGraph(voronoi_map); //voronoi-map for the segmentation-algorithm
imwrite("voronoi_map.png", voronoi_map);
cv::Mat voronoi_map_backup = voronoi_map.clone();
	std::vector<cv::Point> voronoi_dead_end_points; //variable for node point extraction
	std::vector<cv::Point> voronoi_joint_points; //variable for node point extraction
	FindDeadJointPoints(voronoi_map, voronoi_dead_end_points, voronoi_joint_points);

	//remove the joint points which is too close to dead points. // in the dead-end point's radius
	//if do it, passage cannot be splited like 1.png
	for(int i = 0; i < voronoi_joint_points.size(); i++)
	{
		for(int j = 0; j < voronoi_dead_end_points.size(); j++)
		{
			float distance_joint_dead_points = sqrt( (voronoi_joint_points[i].x - voronoi_dead_end_points[j].x)*(voronoi_joint_points[i].x - voronoi_dead_end_points[j].x)
			+ (voronoi_joint_points[i].y - voronoi_dead_end_points[j].y)*(voronoi_joint_points[i].y - voronoi_dead_end_points[j].y)  );
			float dead_point_radius = (int) distance_map.at<unsigned char>(voronoi_dead_end_points[j].y, voronoi_dead_end_points[j].x);
			if(dead_point_radius > distance_joint_dead_points)
			{
				voronoi_joint_points.erase(voronoi_joint_points.begin()+i);
				i = -1;
				break;
			}
		}	
	}
cv::Mat display_dead_joint_points = voronoi_map.clone();
for (size_t i=0; i<voronoi_dead_end_points.size(); ++i)
	cv::circle(display_dead_joint_points, voronoi_dead_end_points[i], 1, cv::Scalar(70), -1);
for (size_t i=0; i<voronoi_joint_points.size(); ++i)
	cv::circle(display_dead_joint_points, voronoi_joint_points[i], 2, cv::Scalar(130), -1);
imwrite("voronoi_dead_joint_points.png", display_dead_joint_points);
std::cout << "voronoi_dead_end_points.size(): " << voronoi_dead_end_points.size() << std::endl;
std::cout << "voronoi_joint_points.size(): " << voronoi_joint_points.size() << std::endl;




	//3:evenly set particles
	std::vector<cv::Point> ray_points; //variable for node point extraction
	Generate_even_ray_points(map_to_be_labeled, ray_points);
cv::Mat display_test2 = map_to_be_labeled.clone();
for (size_t i=0; i<ray_points.size(); ++i)
	cv::circle(display_test2, ray_points[i], 1, cv::Scalar(128), -1);
imwrite("evenly_sample_points.png", display_test2);
std::cout << "evenly_sample_points.size(): " << ray_points.size() << std::endl;

	//4:ray cast the map
	cv::Mat ray_cast_occupy_map = cv::Mat::zeros(map_to_be_labeled.rows,map_to_be_labeled.cols,CV_32SC1);
	for(int node_point_index = 0; node_point_index < ray_points.size(); node_point_index++)
	{
		cv::Point current_point = ray_points[node_point_index];
		ray_occupy_map_func(current_point, ray_cast_occupy_map, map_to_be_labeled);
	}
imwrite("ray_cast_occupy_map.png", ray_cast_occupy_map);

	
	//5: canny segment
	int low_threshold = (6 > ray_points.size()/100 * 8) ?  ray_points.size()/6 * 4 : 6;
	int high_threshold = ray_points.size()/12 * 8;	//need a adaptive paranmeter
	cv::Mat canny_input_image = cv::imread("ray_cast_occupy_map.png",0); 
	cv::Mat canny_edge_out;
	//need to select parameters adaptively
	Canny(canny_input_image,canny_edge_out,20,200,3,true);		//sobel suanzi; -1,-2,-1;  max :255*4 = 1020//黑白的边缘 高低阈值比值为2:1或3:1最佳(50:150 = 1:3)
imwrite("canny_demo.png", canny_edge_out);

	
	//6:Extract the enclosure space; store it and remove them from initial map
	std::vector<std::vector<cv::Point>> segmented_map_vector;
	std::vector<cv::Point> split_line_points;
	//after_segmented + segmented_map_vector = original_map
	ExtractEnclosureArea(map_to_be_labeled, segmented_map_vector, canny_edge_out, voronoi_dead_end_points, voronoi_joint_points, distance_map, split_line_points);
std::cout << "split_line_points.size(): " << split_line_points.size() << std::endl;

std::cout<< "sub_area number: " << segmented_map_vector.size() << std::endl;


cv::Mat after_first_segmented = map_to_be_labeled.clone();
//cv::circle(display_test_line, sub_region_edge_points[max_distance_index], 3, cv::Scalar(128), -1);
for (size_t i=0; i<split_line_points.size(); i = i+2)
{
	cv::line(after_first_segmented, split_line_points[i], split_line_points[i+1], cv::Scalar(0), 1);
	cv::circle(after_first_segmented, split_line_points[i], 1, cv::Scalar(0), -1);
	cv::circle(after_first_segmented, split_line_points[i+1], 1, cv::Scalar(0), -1);
}
imwrite("after_first_segmented.png", after_first_segmented);



	//generate the main sub-area
	cv::Mat main_sub_area = cv::Mat::zeros(map_to_be_labeled.rows,map_to_be_labeled.cols,CV_8UC1);
// for (size_t i=0; i<voronoi_joint_points.size(); i++)
// {
// 	main_sub_area.at<unsigned char>(voronoi_joint_points[i].y, voronoi_joint_points[i].x) = 255;
// }

std::queue<cv::Point> points_tobe_search_neighbor;
points_tobe_search_neighbor.push(segmented_map_vector[0][9]);
while(points_tobe_search_neighbor.size() > 0)
{
	cv::Point current_point = points_tobe_search_neighbor.front();
	std::vector<cv::Point> points_to_be_merged;

	for(int i = -1; i <= 1; i ++)
	{
		for(int j = -1; j <= 1; j ++)
		{
			//nearest 1 grid
			if((abs(i) + abs(j)) == 1)
			{
				int searching_x = current_point.x + i;
				int searching_y = current_point.y + j;
				if (searching_x >= 0 && searching_y >= 0 && searching_y < after_first_segmented.rows && searching_x < after_first_segmented.cols)
				{
					int after_segmented_pixel = after_first_segmented.at<unsigned char>(searching_y, searching_x);

					if(after_segmented_pixel != 0 && main_sub_area.at<unsigned char>(searching_y, searching_x) == 0)
					{
						main_sub_area.at<unsigned char>(searching_y, searching_x) = 255;
						points_tobe_search_neighbor.push(cv::Point(searching_x, searching_y));					
					}

				}
			}
		}
	}
	points_tobe_search_neighbor.pop();
}
imwrite("main_sub_area.png", main_sub_area);

// cv::Mat test_segmented_map = cv::Mat::zeros(map_to_be_labeled.rows,map_to_be_labeled.cols,CV_8UC3);
// std::vector < cv::Vec3b > already_used_colors;
// for(int i = 0; i < segmented_map_vector.size(); i++)
// {
// 	std::vector<cv::Point> current_sub_area = segmented_map_vector[i];
// 	std::cout<< "size of sub_region" << i << " is "<< current_sub_area.size() << std::endl;

// 	cv::Vec3b color;
// 	bool drawn = false;

// 	int loop_counter = 0;
// 	do
// 	{
// 		loop_counter++;
// 		color[0] = rand() % 255;
// 		color[1] = rand() % 255;
// 		color[2] = rand() % 255;
// 		if (!contains(already_used_colors, color) || loop_counter > 100)
// 		{
// 			drawn = true;
// 			already_used_colors.push_back(color);
// 		}
		

// 	} while (!drawn);

// 	for(int j = 0; j < current_sub_area.size(); j++)
// 	{
// 		int x = current_sub_area[j].x;
// 		int y = current_sub_area[j].y;
// 		//test_segmented_map.at<unsigned char>(y, x) = 255;
// 		test_segmented_map.at<cv::Vec3b>(y, x) = color;
// 	}
// }
// imwrite("a_first_segmented.png", test_segmented_map);


	//7:judge whether continue do segmentation.

}


void VoronoiSegmentation::ExtractEnclosureArea(const cv::Mat& map_to_be_labeled, std::vector<std::vector<cv::Point>>& segmented_map_vector, const cv::Mat& canny_edge_out, const std::vector<cv::Point>&voronoi_dead_end_points, const std::vector<cv::Point>&voronoi_joint_points, const cv::Mat& distance_map, std::vector<cv::Point>& split_line_points)
{
	cv::Mat canny_segmented_map = map_to_be_labeled.clone();	// original_map + canny_split_line
	std::vector<cv::Point> canny_edge_points;
	for (int v = 0; v < map_to_be_labeled.rows; v++)
	{
		for (int u = 0; u < map_to_be_labeled.cols; u++)
		{


			//if free in original map but edge in canny image
			if(canny_edge_out.at<unsigned char>(v, u) != 0 && map_to_be_labeled.at<unsigned char>(v, u) != 0)
			{
				int black_number = 0;
				for(int i = -1; i <= 1; i ++)
				{
					for(int j = -1; j <= 1; j ++)
					{
						if(abs(i) + abs(j) > 0)
						{
							if(map_to_be_labeled.at<unsigned char>(v+i, u+j) == 0)
								black_number ++;
						}
					}
				} 
				if(black_number <=3)
				{
					canny_segmented_map.at<unsigned char>(v, u) = 0;
					canny_edge_points.push_back(cv::Point(u,v));
				}
				
			}
		}
	}
std::cout<< "canny_edge_points number: " << canny_edge_points.size() << std::endl;
imwrite("canny_segmented_map.png", canny_segmented_map);

	for (int dead_points_index = 0; dead_points_index < voronoi_dead_end_points.size(); dead_points_index++)
	{
		cv::Point current_edge_point = voronoi_dead_end_points[dead_points_index];

		int current_x = current_edge_point.x;
		int current_y = current_edge_point.y;

		std::queue<cv::Point> searching_points_queue;
		std::vector<cv::Point> total_connected_points;
		total_connected_points.clear();
		searching_points_queue.push(cv::Point(current_x, current_y));
		total_connected_points.push_back(cv::Point(current_x, current_y));

		//search all points in a region
		while(searching_points_queue.size() > 0)
		{
			cv::Point current_point = searching_points_queue.front();
			if(canny_segmented_map.at<unsigned char>(current_point.y, current_point.x) != 0)
			{
				for(int i = -1; i <= 1; i ++)
				{
					for(int j = -1; j <= 1; j ++)
					{
						//nearest 1 grid
						if((abs(i) + abs(j)) == 1)
						{
							int searching_x = current_point.x + i;
							int searching_y = current_point.y + j;
							if (searching_x >= 0 && searching_y >= 0 && searching_y < canny_segmented_map.rows && searching_x < canny_segmented_map.cols && (canny_segmented_map.at<unsigned char>(searching_y, searching_x) != 0))
							{
								cv::Point searching_point;
								searching_point.x = searching_x;
								searching_point.y = searching_y;
								if(!contains(total_connected_points, searching_point))
								{
									searching_points_queue.push(searching_point);
									total_connected_points.push_back(searching_point);
								}
							}
						}
					}
				}
			}
			searching_points_queue.pop();
		}

		//judge whether there are joint points in the region
		bool sub_area = true;
		for (int joint_points_index = 0; joint_points_index < voronoi_joint_points.size(); joint_points_index++)
		{
			//TODO; need to be modified...prely joint is not stable..joint can be divided
			// analysis the joint point. calculate the fov of the joint point. .example if less than 1/4, then can be segmented..
			if(contains(total_connected_points, voronoi_joint_points[joint_points_index]))
			{
				sub_area = false;
				break;
			}
		}

		if(total_connected_points.size() < 3)	//in case of only 1 point
			sub_area = false;

		//Store the sub_area and black the canny_segmented_map
		if(sub_area)
		{
			segmented_map_vector.push_back(total_connected_points);
			for(int i = 0; i < total_connected_points.size(); i++)
			{
				//map_to_be_labeled.at<unsigned char>(total_connected_points[i].y, total_connected_points[i].x) = 0;
				canny_segmented_map.at<unsigned char>(total_connected_points[i].y, total_connected_points[i].x) = 0;
			}
		}

		
	}
	std::cout << "segmented_map_vector.size(): " << segmented_map_vector.size() << std::endl;


	//Get split line
	for(int i = 0; i < segmented_map_vector.size(); i++)
	{
		std::vector<cv::Point> current_sub_area = segmented_map_vector[i];
		std::vector<cv::Point> sub_region_edge_points;

		//search through and find the edge points around a sub_region distance <=3
		for(int j = 0; j < current_sub_area.size(); j++)
		{
			cv::Point sub_region_i_point = current_sub_area[j];
			for(int canny_edge_index = 0; canny_edge_index < canny_edge_points.size(); canny_edge_index++)
			{
				cv::Point edge_point = canny_edge_points[canny_edge_index];
				float distance = sqrt( (sub_region_i_point.x - edge_point.x) * (sub_region_i_point.x - edge_point.x) +
				(sub_region_i_point.y - edge_point.y) * (sub_region_i_point.y - edge_point.y) );

				if(distance <=3 && ((int) distance_map.at<unsigned char>(edge_point.y, edge_point.x)) >=1)
					sub_region_edge_points.push_back(canny_edge_points[canny_edge_index]);
			}
		}

		float max_distance = 0;
		int max_distance_index = 0;
		std::vector<cv::Point> candidate_line_points;
		// remove the edge points which are too near to obstacle.[these points cannot be split line]
		// and find the edge point with the biggest distance to obstacle.[center of split line.]
		for(int si = 0; si < sub_region_edge_points.size(); si++)
		{
			float distance_to_obstacle = ((int) distance_map.at<unsigned char>(sub_region_edge_points[si].y, sub_region_edge_points[si].x));
			if( (distance_to_obstacle > max_distance) && (map_to_be_labeled.at<unsigned char>(sub_region_edge_points[si].y, sub_region_edge_points[si].x) != 0) )
			{
				max_distance = distance_to_obstacle;
				max_distance_index = si;
			}
			if(distance_to_obstacle >= 1)
			{
				candidate_line_points.push_back(sub_region_edge_points[si]);
			}
		}
		std::cout<< "max_distance of " << i << " is " << max_distance << std::endl;
		//std::cout<< "max_distance_index " << i << " is " << max_distance_index << std::endl;

		int max_distance_center = 0;
		int candidate_index_1=0;
		int candidate_index_2=0;
		//find the point 1 and point2 to make up for split line
		//features of split line: nearly all other points are on one side of the split line
		for(int si = 0; si < candidate_line_points.size(); si++)
		{
			float distance_to_max = sqrt( (candidate_line_points[si].x - candidate_line_points[max_distance_index].x) * (candidate_line_points[si].x - candidate_line_points[max_distance_index].x) +
				(candidate_line_points[si].y - candidate_line_points[max_distance_index].y) * (candidate_line_points[si].y - candidate_line_points[max_distance_index].y) );
			if(distance_to_max >= max_distance && max_distance_center <= distance_to_max )
			{
				//two points determine a segment line... and all other points need in one side
				int x1 = candidate_line_points[si].x;
				int x2 = candidate_line_points[max_distance_index].x;
				int y1 = candidate_line_points[si].y;
				int y2 = candidate_line_points[max_distance_index].y;
				float k,b;
				int positive_count=0;
				int negative_count=0;

				if(x1 == x2)
					k = 10000;
				else
					k = (y2-y1)/(x2-x1);

				b = y2 - k * x2;

				for(int can_points_index = 0; can_points_index < candidate_line_points.size(); can_points_index++)
				{
					int x = candidate_line_points[can_points_index].x;
					int y = candidate_line_points[can_points_index].y;
					float distance_to_line = abs(k*x - y + b)/sqrt(k*k+1);

					if(abs(distance_to_line) > 1 && (y > k*x + b))
						positive_count ++;
					
					if(abs(distance_to_line) > 1 && (y < k*x + b))
						negative_count ++;
				}

				if(positive_count == 0 || negative_count == 0)
				{
					max_distance_center = distance_to_max;
					candidate_index_1 = si;
				}
	
			}
		}

		max_distance_center = 0;
		for(int si = 0; si < candidate_line_points.size(); si++)
		{
			float distance_to_max = sqrt( (candidate_line_points[si].x - candidate_line_points[max_distance_index].x) * (candidate_line_points[si].x - candidate_line_points[max_distance_index].x) +
				(candidate_line_points[si].y - candidate_line_points[max_distance_index].y) * (candidate_line_points[si].y - candidate_line_points[max_distance_index].y) );
			float distance_to_candidate1 = sqrt( (candidate_line_points[si].x - candidate_line_points[candidate_index_1].x) * (candidate_line_points[si].x - candidate_line_points[candidate_index_1].x) +
				(candidate_line_points[si].y - candidate_line_points[candidate_index_1].y) * (candidate_line_points[si].y - candidate_line_points[candidate_index_1].y) );
			
			const double vector_x1 = candidate_line_points[candidate_index_1].x - candidate_line_points[max_distance_index].x;
	 		const double vector_y1 = candidate_line_points[candidate_index_1].y - candidate_line_points[max_distance_index].y;
			const double vector_x2 = candidate_line_points[si].x - candidate_line_points[max_distance_index].x;
	 		const double vector_y2 = candidate_line_points[si].y - candidate_line_points[max_distance_index].y;

			float p2_to_max = sqrt( vector_x2 * vector_x2 + vector_y2 * vector_y2);
			float p1_to_max = sqrt( vector_x1 * vector_x1 + vector_y1 * vector_y1);

			float angle = std::acos((vector_x1 * vector_x2 + vector_y1 * vector_y2) / (p1_to_max * p2_to_max)) * 180.0 / PI;
			if(distance_to_candidate1 >= max_distance && distance_to_max >= max_distance_center && angle > 120)
			{

				//two points determine a segment line... and all other points need in one side
				int x1 = candidate_line_points[si].x;
				int x2 = candidate_line_points[max_distance_index].x;
				int y1 = candidate_line_points[si].y;
				int y2 = candidate_line_points[max_distance_index].y;
				float k,b;
				int positive_count=0;
				int negative_count=0;

				if(x1 == x2)
					k = 10000;
				else
					k = (y2-y1)/(x2-x1);

				b = y2 - k * x2;

				for(int can_points_index = 0; can_points_index < candidate_line_points.size(); can_points_index++)
				{
					int x = candidate_line_points[can_points_index].x;
					int y = candidate_line_points[can_points_index].y;
					float distance_to_line = abs(k*x - y + b)/sqrt(k*k+1);

					if(abs(distance_to_line) > 2 && (y > k*x + b))
						positive_count ++;
					
					if(abs(distance_to_line) > 2 && (y < k*x + b))
						negative_count ++;
				}


				if(positive_count == 0 || negative_count == 0)
				{
					max_distance_center = distance_to_max;
					candidate_index_2 = si;
				}
			}
		}

		split_line_points.push_back(candidate_line_points[candidate_index_1]);
		split_line_points.push_back(candidate_line_points[candidate_index_2]);

	}

cv::Mat display_test_line = map_to_be_labeled.clone();
//cv::circle(display_test_line, sub_region_edge_points[max_distance_index], 3, cv::Scalar(128), -1);
for (size_t i=0; i<split_line_points.size(); i = i+2)
{
	cv::line(display_test_line, split_line_points[i], split_line_points[i+1], cv::Scalar(128), 1);
	cv::circle(display_test_line, split_line_points[i], 2, cv::Scalar(128), -1);
}
imwrite("display_test_line.png", display_test_line);
}




void VoronoiSegmentation::FindDeadJointPoints(const cv::Mat& voronoi_map, std::vector<cv::Point>& voronoi_dead_end_points, std::vector<cv::Point>& voronoi_joint_points)
{
	int voronoi_max_neighbors = 0;
	int max_number = 1;
	for (int v = 1; v < voronoi_map.rows-1; v++)
	{
		for (int u = 1; u < voronoi_map.cols-1; u++)
		{
			if (voronoi_map.at<unsigned char>(v, u) == 127)
			{
				int neighbor_count = 0;	// variable to save the number of neighbors for each point
				// check 3x3 region around current pixel
				for (int row_counter = -1; row_counter <= 1; row_counter++)
				{
					for (int column_counter = -1; column_counter <= 1; column_counter++)
					{
						// don't check the point itself
						if (row_counter == 0 && column_counter == 0)
							continue;

						//check if neighbors are colored with the voronoi-color
						if (voronoi_map.at<unsigned char>(v + row_counter, u + column_counter) == 127)
						{
							neighbor_count++;
						}
					}
				}
				if (neighbor_count < 2)
				{
					voronoi_dead_end_points.push_back(cv::Point(u,v));
				}
				if(neighbor_count > voronoi_max_neighbors)
				{
					voronoi_joint_points.clear();
					voronoi_max_neighbors = neighbor_count;
					max_number = 1;
					
				}
				
				if(neighbor_count >= (voronoi_max_neighbors -1) && neighbor_count >=4)
				{
					max_number ++;
					voronoi_joint_points.push_back(cv::Point(u,v));
				}	
			}
		}
	}

	std::cout << "voronoi_max_neighbors: " << voronoi_max_neighbors<< std::endl;
	std::cout << "max_number: " << max_number<< std::endl;

}


void VoronoiSegmentation::Generate_even_ray_points(const cv::Mat& origial_map, std::vector<cv::Point>& my_generated_points)
{
	cv::Mat temp_map = origial_map.clone();
    for (int v = 0; v < origial_map.rows; v++)
    {
        for (int u = 0; u < origial_map.cols; u++)
        {
            cv::Point current_point;
            current_point.x = u;
            current_point.y = v;
            int current_point_pixel = temp_map.at<unsigned char>(v, u);
			int count_white = 0;
            for(int i = -2; i <= 2; i ++)
            {
                for(int j = -2; j <= 2; j ++)
                {

					int searching_x = current_point.x + i;
					int searching_y = current_point.y + j;
					if (searching_x >= 0 && searching_y >= 0 && searching_y < temp_map.rows && searching_x < temp_map.cols && (temp_map.at<unsigned char>(searching_y, searching_x) != 0))
					{
						count_white++;
					}
                }
            }

			if( count_white >= 20 )
            {
                my_generated_points.push_back(current_point);
				for(int ii = -1; ii <= 1; ii ++)
				{
					for(int jj = -1; jj <= 1; jj ++)
					{

						int searching_x = current_point.x + ii;
						int searching_y = current_point.y + jj;
						temp_map.at<unsigned char>(searching_y, searching_x) = 0;
					}
				}
            } 

        }
    }
    std::cout << "searched " << my_generated_points.size() << " evenly sample points." << std::endl;

}



void VoronoiSegmentation::Sew_segmented_map(const cv::Mat& map_to_be_labeled, cv::Mat& merged_ray_map)
{
	bool merge_map_flag = true;
	int loop_count_down = 100;
	while(merge_map_flag && loop_count_down > 0) 
	{
		loop_count_down --;
		std::vector<cv::Point>  points_tobe_sewed;
		//last_area_id++;
		for (int v = 0; v < merged_ray_map.rows; v++)
		{
			for (int u = 0; u < merged_ray_map.cols; u++)
			{
				unsigned int pixel_merged_map = merged_ray_map.at<int>(v, u);
				unsigned int pixel_origin_map = map_to_be_labeled.at<unsigned char>(v, u);
				if(pixel_origin_map != 0 && pixel_merged_map == 0)
				{
					points_tobe_sewed.push_back(cv::Point(u,v));
					//merged_ray_map.at<int>(v, u) = last_area_id;
				}
			}
		}
		std::cout << "points_tobe_sewed: " << points_tobe_sewed.size() << std::endl;
		if(points_tobe_sewed.size() == 0)
			merge_map_flag = false;

		//for(std::vector<cv::Point>::iterator it = points_tobe_sewed.begin(); it != points_tobe_sewed.end();) 
		for(int sewed_index = 0; sewed_index < points_tobe_sewed.size(); sewed_index ++)
		{
			// int current_x = it->x;
			// int current_y = it->y;
			int current_x = points_tobe_sewed[sewed_index].x;
			int current_y = points_tobe_sewed[sewed_index].y;
			std::vector<int> pixel_id_set;
			int neighbor_number = 0;
			int pixel_id_set_count[9];
			memset(pixel_id_set_count, 0, 9*sizeof(int)); //

			for(int i = -1; i <= 1; i ++)
			{
				for(int j = -1; j <= 1; j ++)
				{
					int searching_x = current_x + i;
					int searching_y = current_y + j;
					if (searching_x >= 0 && searching_y >= 0 && searching_y < merged_ray_map.rows && searching_x < merged_ray_map.cols )
					{
						int searching_pixel = merged_ray_map.at<int>(searching_y, searching_x);
						if(searching_pixel > 0)
						{
							std::vector<int>::iterator iter=std::find(pixel_id_set.begin(),pixel_id_set.end(),searching_pixel);
							if(iter==pixel_id_set.end())
							{
								pixel_id_set.push_back(searching_pixel);
								pixel_id_set_count[neighbor_number] = pixel_id_set_count[neighbor_number] + 1;
								neighbor_number ++;
							}
							else
							{
								int index = std::distance(pixel_id_set.begin(), iter);
								pixel_id_set_count[index] = pixel_id_set_count[index] + 1;
							}
						}
					}
				}
			}

			//if not surrounded by 0, then set the pixel to the  biggest surround pixel...
			int max_surround_pixel_count_index = 0;
			for(int i = 0; i < neighbor_number;i++)
			{
				if(pixel_id_set_count[i] > pixel_id_set_count[max_surround_pixel_count_index])
					max_surround_pixel_count_index = i;
			}
			if(pixel_id_set_count[max_surround_pixel_count_index] > 0)
				merged_ray_map.at<int>(current_y, current_x) = pixel_id_set[max_surround_pixel_count_index];

		}
	}


}



void VoronoiSegmentation::Find_contour_points(const cv::Mat& original_map, const cv::Mat& distance_map, std::vector<cv::Point>& contour_points,  double map_resolution_from_subscription)
{

	for (int v = 0; v < original_map.rows; v++)
	{
		for (int u = 0; u < original_map.cols; u++)
		{
			int distance_to_obstacle = (int) distance_map.at<unsigned char>(v, u);
			float lower_bound_distance = k_select_contour_distance_low_bound / map_resolution_from_subscription;
			float higher_bound_distance = k_select_contour_distance_high_bound / map_resolution_from_subscription;
			if(original_map.at<unsigned char>(v, u) != 0 && distance_to_obstacle > lower_bound_distance && distance_to_obstacle < higher_bound_distance)
			{
				bool insert_point = true;
				for(int i = 0; i < contour_points.size(); i ++)
				{
					float distance_to_existed_point = std::sqrt( (v - contour_points[i].y) * (v - contour_points[i].y) + (u - contour_points[i].x) * (u - contour_points[i].x) );
					if(distance_to_existed_point <= distance_to_obstacle || distance_to_existed_point <= (int) distance_map.at<unsigned char>(contour_points[i].y, contour_points[i].x)
					|| distance_to_existed_point <= k_robot_width/map_resolution_from_subscription)
					{
						insert_point = false;
						break;
					}				
				}
				if(insert_point)
					contour_points.push_back(cv::Point(u,v));

			}
		}
	}
}



void VoronoiSegmentation::SegmentArea(cv::Point node_point, const cv::Mat& ray_cast_occupy_map, cv::Mat& merged_ray_map, std::vector<cv::Point>& overlap_points, int& last_area_id)
{
	//only do wavefront when the point hasn't be allocated a roomid
	if(merged_ray_map.at<int>(node_point.y, node_point.x) == 0)
	{
		int current_area_id = last_area_id +1;
		std::queue<cv::Point> points_tobe_search_neighbor;
		int node_point_vote_number = ray_cast_occupy_map.at<int>(node_point.y, node_point.x);
		points_tobe_search_neighbor.push(node_point);
		while(points_tobe_search_neighbor.size() > 0)
		{
			cv::Point current_point = points_tobe_search_neighbor.front();
			std::vector<cv::Point> points_to_be_merged;
			int spread_size = 0;
			int current_vote_number = ray_cast_occupy_map.at<int>(current_point.y, current_point.x);
			for(int i = -1; i <= 1; i ++)
			{
				for(int j = -1; j <= 1; j ++)
				{
					//nearest 1 grid
					if((abs(i) + abs(j)) == 1)
					{
						int searching_x = current_point.x + i;
						int searching_y = current_point.y + j;
						if (searching_x >= 0 && searching_y >= 0 && searching_y < ray_cast_occupy_map.rows && searching_x < ray_cast_occupy_map.cols )
						{
							int searching_vote_number = ray_cast_occupy_map.at<int>(searching_y, searching_x);
							if(abs(searching_vote_number - current_vote_number) <= k_wave_step && abs(searching_vote_number - node_point_vote_number) <= k_wave_biggest_step && searching_vote_number > 0)
							{
								if(merged_ray_map.at<int>(searching_y, searching_x) == 0)
								{
									//points_tobe_search_neighbor.push(cv::Point(searching_x, searching_y));
									//merged_ray_map.at<int>(searching_y, searching_x) = current_area_id;
									points_to_be_merged.push_back(cv::Point(searching_x, searching_y));
									spread_size++;
									
								}
								else if(merged_ray_map.at<int>(searching_y, searching_x) == current_area_id)
								{
									spread_size ++;
								}
								else
								{
									//points_tobe_search_neighbor.push(cv::Point(searching_x, searching_y));
									overlap_points.push_back(cv::Point(searching_x, searching_y));
								}
								
							}
						}
					}
				}
			}
			points_tobe_search_neighbor.pop();
			if(spread_size >=3)
			{
				for(int i = 0; i < points_to_be_merged.size(); i++)
				{
					points_tobe_search_neighbor.push(points_to_be_merged[i]);
					merged_ray_map.at<int>(points_to_be_merged[i].y, points_to_be_merged[i].x) = current_area_id;
				}
			}
		}
		last_area_id = current_area_id;
	}
}



void VoronoiSegmentation::ray_occupy_map_func(cv::Point current_point, cv::Mat& ray_cast_occupy_map, const cv::Mat& original_map)
{
	//point(x,y);  but in map(y,x)  v < voronoi_map_backup.rows
	//int) distance_map.at<unsigned char>(v, u);
	//cv::Point(u,v)
	int center_y = current_point.y;
	int center_x = current_point.x;
	int image_y_max = original_map.rows;
	int image_x_max = original_map.cols;
	int image_y_min = 0;
	int image_x_min = 0;
	float angle_increment = 0.2;	//1 du   angle / 180.0 * 3.14159265
	std::vector<cv::Point> ray_occupied_points;

	for(float angle = 0; angle < 360; angle = angle + angle_increment)
	{


		if(angle == 0)
		{
			for(int i = center_x; i < image_x_max; i++)
			{
				if(original_map.at<unsigned char>(center_y, i) != 0)
				{
					cv::Point preinsert_point;
					preinsert_point.x = i;
					preinsert_point.y = center_y;
					//ray_occupied_points.push_back(cv::Point(i,center_y));
					if(!contains(ray_occupied_points, preinsert_point))
						ray_occupied_points.push_back(preinsert_point);	
				}
				else break;
			}
		}

		if(angle == 90)
		{
			for(int i = center_y; i >= 0; i--)
			{
				if(original_map.at<unsigned char>(i, center_x) != 0)
				{
					cv::Point preinsert_point;
					preinsert_point.x = center_x;
					preinsert_point.y = i;
					//ray_occupied_points.push_back(cv::Point(i,center_y));
					if(!contains(ray_occupied_points, preinsert_point))
						ray_occupied_points.push_back(preinsert_point);	
				}
				else break;
			}
		}

		if(angle == 180)
		{
			for(int i = center_x; i >= 0; i--)
			{
				if(original_map.at<unsigned char>(center_y, i) != 0)
				{
					cv::Point preinsert_point;
					preinsert_point.x = i;
					preinsert_point.y = center_y;
					//ray_occupied_points.push_back(cv::Point(i,center_y));
					if(!contains(ray_occupied_points, preinsert_point))
						ray_occupied_points.push_back(preinsert_point);	
				}
				else break;
			}
		}

		if(angle == 270)
		{
			for(int i = center_y; i < image_y_max; i++)
			{
				if(original_map.at<unsigned char>(i, center_x) != 0)
				{
					cv::Point preinsert_point;
					preinsert_point.x = center_x;
					preinsert_point.y = i;
					//ray_occupied_points.push_back(cv::Point(i,center_y));
					if(!contains(ray_occupied_points, preinsert_point))
						ray_occupied_points.push_back(preinsert_point);	
				}
				else break;
			}
		}


		
		if(angle > 0 && angle <90)
		{
			//assume the point is origin,, , right x, up y;  y=kx;
			float k = tan(angle / 180.0 * 3.14159265);
			int relative_y_max = center_y;
			int relative_x_max = image_x_max - center_x;
			int relative_y_min = center_y - image_y_max;
			int relative_x_min = 0 - center_x;
			//float x_bound_y = relative_x_max*k;
			// if k <= 1; then set x as master
			if(abs(k) <= 1)
			{
				for(int ix = 0; ix <= relative_x_max; ix ++)
				{
					cv::Point preinsert_point;
					// transform to the image coordinate
					preinsert_point.x = ix + center_x;
					preinsert_point.y = center_y - floor(k * ix);
					if(original_map.at<unsigned char>(preinsert_point.y, preinsert_point.x) != 0)
					{
						if(!contains(ray_occupied_points, preinsert_point))
							ray_occupied_points.push_back(preinsert_point);	
					}
					else break;
					
				}
			}
			if(abs(k) >= 1)
			{
				for(int iy = 0; iy <= relative_y_max; iy ++)
				{
					cv::Point preinsert_point;
					// transform to the image coordinate
					preinsert_point.x = center_x + floor(iy / k);
					preinsert_point.y = center_y - iy;
					if(original_map.at<unsigned char>(preinsert_point.y, preinsert_point.x) != 0)
					{
						if(!contains(ray_occupied_points, preinsert_point))
							ray_occupied_points.push_back(preinsert_point);	
					}
					else break;
				}
			}

		}
		else if (angle > 90 && angle <180)
		{
			//assume the point is origin,, , right x, up y;  y=kx;
			float k = tan(angle / 180.0 * 3.14159265);
			int relative_y_max = center_y;
			int relative_x_max = image_x_max - center_x;
			int relative_y_min = center_y - image_y_max;
			int relative_x_min = 0 - center_x;
			//float x_bound_y = relative_x_max*k;
			// if k <= 1; then set x as master
			if(abs(k) <= 1)
			{
				for(int ix = 0; ix >= relative_x_min; ix --)
				{
					cv::Point preinsert_point;
					// transform to the image coordinate
					preinsert_point.x = center_x + ix;
					preinsert_point.y = center_y - floor(k * ix);
					if(original_map.at<unsigned char>(preinsert_point.y, preinsert_point.x) != 0)
					{
						if(!contains(ray_occupied_points, preinsert_point))
							ray_occupied_points.push_back(preinsert_point);	
					}
					else break;
					
				}
			}
			if(abs(k) >= 1)
			{
				for(int iy = 0; iy <= relative_y_max; iy ++)
				{
					cv::Point preinsert_point;
					// transform to the image coordinate
					preinsert_point.x = center_x + floor(iy / k);
					preinsert_point.y = center_y - iy;
					if(original_map.at<unsigned char>(preinsert_point.y, preinsert_point.x) != 0)
					{
						if(!contains(ray_occupied_points, preinsert_point))
							ray_occupied_points.push_back(preinsert_point);	
					}
					else break;
				}
			}	

		}
		else if (angle > 180 && angle <270)
		{
			//assume the point is origin,, , right x, up y;  y=kx;
			float k = tan(angle / 180.0 * 3.14159265);
			int relative_y_max = center_y;
			int relative_x_max = image_x_max - center_x;
			int relative_y_min = center_y - image_y_max;
			int relative_x_min = 0 - center_x;
			//float x_bound_y = relative_x_max*k;
			// if k <= 1; then set x as master
			if(abs(k) <= 1)
			{
				for(int ix = 0; ix >= relative_x_min; ix --)
				{
					cv::Point preinsert_point;
					// transform to the image coordinate
					preinsert_point.x = center_x + ix;
					preinsert_point.y = center_y - floor(k * ix);
					if(original_map.at<unsigned char>(preinsert_point.y, preinsert_point.x) != 0)
					{
						if(!contains(ray_occupied_points, preinsert_point))
							ray_occupied_points.push_back(preinsert_point);	
					}
					else break;
					
				}
			}
			if(abs(k) >= 1)
			{
				for(int iy = 0; iy >= relative_y_min; iy --)
				{
					cv::Point preinsert_point;
					// transform to the image coordinate
					preinsert_point.x = center_x + floor(iy / k);
					preinsert_point.y = center_y - iy;
					if(original_map.at<unsigned char>(preinsert_point.y, preinsert_point.x) != 0)
					{
						if(!contains(ray_occupied_points, preinsert_point))
							ray_occupied_points.push_back(preinsert_point);	
					}
					else break;
				}
			}
		}
		else if (angle > 270 && angle < 360)
		{
			//assume the point is origin,, , right x, up y;  y=kx;
			float k = tan(angle / 180.0 * 3.14159265);
			int relative_y_max = center_y;
			int relative_x_max = image_x_max - center_x;
			int relative_y_min = center_y - image_y_max;
			int relative_x_min = 0 - center_x;
			//float x_bound_y = relative_x_max*k;
			// if k <= 1; then set x as master
			if(abs(k) <= 1)
			{
				for(int ix = 0; ix <= relative_x_max; ix ++)
				{
					cv::Point preinsert_point;
					// transform to the image coordinate
					preinsert_point.x = center_x + ix;
					preinsert_point.y = center_y - floor(k * ix);
					if(original_map.at<unsigned char>(preinsert_point.y, preinsert_point.x) != 0)
					{
						if(!contains(ray_occupied_points, preinsert_point))
							ray_occupied_points.push_back(preinsert_point);	
					}
					else break;
					
				}
			}
			if(abs(k) >= 1)
			{
				for(int iy = 0; iy >= relative_y_min; iy --)
				{
					cv::Point preinsert_point;
					// transform to the image coordinate
					preinsert_point.x = center_x + floor(iy / k);
					preinsert_point.y = center_y - iy;
					if(original_map.at<unsigned char>(preinsert_point.y, preinsert_point.x) != 0)
					{
						if(!contains(ray_occupied_points, preinsert_point))
							ray_occupied_points.push_back(preinsert_point);	
					}
					else break;
				}
			}
		}

	}

//scan finished,, add the result to result_map
	for(int i = 0; i < ray_occupied_points.size(); i ++)
	{
		ray_cast_occupy_map.at<int>(ray_occupied_points[i].y, ray_occupied_points[i].x) = 1 + ray_cast_occupy_map.at<int>(ray_occupied_points[i].y, ray_occupied_points[i].x);
		//ray_cast_occupy_map.at<unsigned char>(ray_occupied_points[i].y, ray_occupied_points[i].x) = 255;
	}

}


void VoronoiSegmentation::draw_segmented_line(const cv::Mat& map_to_be_draw, std::vector<std::vector<cv::Point>>& segment_result, const char *input_name)
{
	if (map_to_be_draw.type()!=CV_32SC1)
	{
		std::cout << "Error: map_to_be_draw: provided image is not of type CV_32SC1." << std::endl;
		return;
	}

	std::vector <cv::Vec3b> used_colors;

	cv::Mat drawed_segmented_map = cv::Mat::zeros(map_to_be_draw.rows,map_to_be_draw.cols,CV_8UC3);
	for (int row = 0; row < map_to_be_draw.rows; row++)
	{
		for (int column = 1; column < map_to_be_draw.cols; column++)
		{
			if( map_to_be_draw.at<int>(row, column) > 0 )
			{
				cv::Vec3b color;
				color[0] = 255;
				color[1] = 255;
				color[2] = 255;
				drawed_segmented_map.at<cv::Vec3b>(row, column) = color;
			}
		}
	}

	for(int i = 0; i < segment_result.size(); i ++)
	{
		std::vector<cv::Point> segment_line = segment_result[i];
		cv::Vec3b color;
		bool drawn = false;
		int loop_counter = 0;
		do
		{
			loop_counter++;
			color[0] = rand() % 255;
			color[1] = rand() % 255;
			color[2] = rand() % 255;
			if (!contains(used_colors, color) || loop_counter > 100)
			{
				drawn = true;
				used_colors.push_back(color);
			}
		} while (!drawn);

		for(int j = 0; j < segment_line.size(); j ++)
		{
			
			int x = segment_line[j].x;
			int y = segment_line[j].y;
			cv::circle(drawed_segmented_map, segment_line[j], 2, color, -1);
			//drawed_segmented_map.at<cv::Vec3b>(y, x) = color;
		}
	}
	imwrite(input_name, drawed_segmented_map);
}
