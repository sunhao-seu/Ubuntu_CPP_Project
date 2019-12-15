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


void VoronoiSegmentation::segmentMap(const cv::Mat& map_to_be_labeled, cv::Mat& segmented_map, double map_resolution_from_subscription,
		double room_area_factor_lower_limit, double room_area_factor_upper_limit, int neighborhood_index, int max_iterations,
		double min_critical_point_distance_factor, double max_area_for_merging, bool display_map)
{
	//****************Create the Generalized Voronoi-Diagram**********************
	//This function takes a given map and segments it with the generalized Voronoi-Diagram. It takes following steps:
	//	I. It calculates the generalized Voronoi-Diagram using the function createVoronoiGraph.
	//	II. It extracts the critical points, which show the border between two segments. This part takes these steps:
	//		1. Extract node-points of the Voronoi-Diagram, which have at least 3 neighbors.
	//		2. Reduce the leave-nodes (Point on graph with only one neighbor) of the graph until the reduction
	//		   hits a node-Point. This is done to reduce the lines along the real voronoi-graph, coming from the discretisation
	//		   of the contour.
	//		3. Find the critical points in the reduced graph by searching in a specified neighborhood for a local minimum
	//		   in distance to the nearest black pixel. The size of the epsilon-neighborhood is dynamic and goes larger
	//		   in small areas, so they are split into lesser regions.
	//	III. It gets the critical lines, which go from the critical point to its two nearest black pixels and separate the
	//		 regions from each other. This part does following steps:
	//			1. Get the discretized contours of the map and the holes, because these are the possible candidates for
	//			   basis-points.
	//			2. Find the basis-points for each critical-point by finding the two nearest neighbors of the vector from 1.
	//			   Also it saves the angle between the two vectors pointing from the critical-point to its two basis-points.
	//			3. Some critical-lines are too close to each other, so the next part eliminates some of them. For this the
	//			   algorithm checks, which critical points are too close to each other. Then it compares the angles of these
	//			   points, which were calculated in 3., and takes the one with the larger angle, because smaller angles
	//			   (like 90 degree) are more likely to be at edges of the map or are too close to the borders. If they have
	//			   the same angle, the point which comes first in the critical-point-vector is chosen (took good results for
	//			   me, but is only subjective).
	//			4. Draw the critical lines, selected by 3. in the map with color 0.
	//	IV. It finds the segments, which are seperated by the critical lines of III. and fills them with a random colour that
	//		hasn't been already used yet. For this it:
	//			1. It erodes the map with critical lines, so small gaps are closed, and finds the contours of the segments.
	//			   Only contours that are large/small enough are chosen to be drawn.
	//			2. It draws the contours from 1. in a map with a random colour. Contours that belong to holes are not drawn
	//			   into the map.
	//			3. Spread the colour-regions to the last white Pixels, using the watershed-region-spreading function.

	//*********************I. Calculate and draw the Voronoi-Diagram in the given map*****************

double total_area = map_resolution_from_subscription * map_resolution_from_subscription * map_to_be_labeled.rows * map_to_be_labeled.cols;
int max_number_node_points = (int)total_area/0.3;
std::cout << "total_area: " << total_area << std::endl;

	cv::Mat voronoi_map = map_to_be_labeled.clone();
	createVoronoiGraph(voronoi_map); //voronoi-map for the segmentation-algorithm
imwrite("voronoi_map.png", voronoi_map);
	//***************************II. extract the possible candidates for critical Points****************************
	// 1.extract the node-points that have at least three neighbors on the voronoi diagram
	//	node-points are points on the voronoi-graph that have at least 3 neighbors
	// 2.reduce the side-lines along the voronoi-graph by checking if it has only one neighbor until a node-point is reached
	//	--> make it white
	//	repeat a large enough number of times so the graph converges
	std::set<cv::Point, cv_Point_comp> node_points; //variable for node point extraction
	pruneVoronoiGraph(voronoi_map, node_points);
	cv::Mat voronoi_map_backup = voronoi_map.clone();
	// for(int node_remove_loop = 0; node_remove_loop < 10; node_remove_loop ++)
	// {
	// 	if(node_points.size() > max_number_node_points)
	// 	{
	// 		node_points.clear();
	// 		pruneVoronoiGraph(voronoi_map, node_points);
	// 		std::cout << "node_points.size(): " << node_points.size() << std::endl;
	// 	}
	// 	else break;
	// }

std::cout << "node_points.size(): " << node_points.size() << std::endl;
imwrite("pruneVoronoi.png", voronoi_map);
	//3.find the critical points in the previously calculated generalized Voronoi-graph by searching in a specified
	//	neighborhood for the local minimum of distance to the nearest black pixel
	//	critical points need to have at least two neighbors (else they are end points, which would give a very small segment)

	//get the distance transformed map, which shows the distance of every white pixel to the closest zero-pixel
	cv::Mat distance_map; //distance-map of the original-map (used to check the distance of each point to nearest black pixel)
	cv::distanceTransform(map_to_be_labeled, distance_map, CV_DIST_L2, 5);
	cv::convertScaleAbs(distance_map, distance_map);
imwrite("distance_map.png", distance_map);	//  distance to the nearest zero point(black point)(obstacle)()

//********************search in a range(defined by eps), the point with nearest distance to obstacle will be the critical point.************* 
	std::vector<cv::Point> critical_points; //saving-variable for the critical points found on the Voronoi-graph
	Find_critical_points(voronoi_map, distance_map, critical_points, map_resolution_from_subscription, neighborhood_index, max_iterations);

	cv::Mat display = map_to_be_labeled.clone();
	for (size_t i=0; i<critical_points.size(); ++i)
		cv::circle(display, critical_points[i], 2, cv::Scalar(128), -1);
imwrite("voronoi_map_critical_points.png", display);
std::cout << "critical_points.size(): " << critical_points.size() << std::endl;


	std::vector<cv::Point> my_node_points; //variable for node point extraction

	// Find_contour_points(map_to_be_labeled, distance_map, my_node_points, map_resolution_from_subscription);
	// cv::Mat display_test1 = map_to_be_labeled.clone();
	// for (size_t i=0; i<my_node_points.size(); ++i)
	// 	cv::circle(display_test1, my_node_points[i], 1, cv::Scalar(128), -1);
	// imwrite("my_contour_points.png", display_test1);
	// std::cout << "my_contour_points.size(): " << my_node_points.size() << std::endl;

	Generate_even_ray_points(map_to_be_labeled, my_node_points);
	cv::Mat display_test2 = map_to_be_labeled.clone();
	for (size_t i=0; i<my_node_points.size(); ++i)
		cv::circle(display_test2, my_node_points[i], 1, cv::Scalar(128), -1);
	imwrite("evenly_sample_points.png", display_test2);
	std::cout << "evenly_sample_points.size(): " << my_node_points.size() << std::endl;

	Find_my_node_points(voronoi_map_backup, distance_map, my_node_points);	
	cv::Mat display_test = map_to_be_labeled.clone();
	for (size_t i=0; i<my_node_points.size(); ++i)
		cv::circle(display_test, my_node_points[i], 1, cv::Scalar(128), -1);
	imwrite("my_true_node_points.png", display_test);
	std::cout << "my_true_node_points.size(): " << my_node_points.size() << std::endl;

	

	//map_to_be_labeled.convertTo(segmented_map, CV_32SC1, 256, 0); // rescale to 32 int, 255 --> 255*256 = 65280

	//cv::Mat ray_cast_occupy_map = map_to_be_labeled.clone();
	cv::Mat ray_cast_occupy_map = cv::Mat::zeros(map_to_be_labeled.rows,map_to_be_labeled.cols,CV_32SC1);
	for(int node_point_index = 0; node_point_index < my_node_points.size(); node_point_index++)
	{
		cv::Point current_point = my_node_points[node_point_index];
		ray_occupy_map_func(current_point, ray_cast_occupy_map, map_to_be_labeled);
	}

	imwrite("ray_cast_occupy_map.png", ray_cast_occupy_map);

	
	cv::Mat test_map = cv::imread("ray_cast_occupy_map.png",0); 
// 	//cv::Mat test_map = ray_cast_occupy_map.clone();
// 	cv::Mat grad_x, grad_y;
//     cv::Mat abs_grad_x, abs_grad_y, dst;

//     //求x方向梯度
//     Sobel(test_map, grad_x, CV_16S, 1, 0, 1, 1, 0,cv::BORDER_DEFAULT);
//     convertScaleAbs(grad_x, abs_grad_x);
//    // imshow("x方向soble", abs_grad_x);
// imwrite("ray_map_Xsobel.png", abs_grad_x);
//     //求y方向梯度
//     Sobel(test_map, grad_y,CV_16S,0, 1,1, 1, 0, cv::BORDER_DEFAULT);
//     convertScaleAbs(grad_y,abs_grad_y);
//     //imshow("y向soble", abs_grad_y);
// imwrite("ray_map_Ysobel.png", abs_grad_y);
//     //合并梯度

//     addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst);
//     //imshow("整体方向soble", dst);
// 	imwrite("ray_map_sobel.png", dst);

	cv::Mat edge_out;
	Canny(test_map,edge_out,8,200,3,true);		//sobel suanzi; -1,-2,-1;  max :255*4 = 1020//黑白的边缘 高低阈值比值为2:1或3:1最佳(50:150 = 1:3)
    imwrite("canny_demo.png", edge_out);

	cv::Mat canny_split = map_to_be_labeled.clone();

	for (int v = 0; v < ray_cast_occupy_map.rows; v++)
	{
		for (int u = 0; u < ray_cast_occupy_map.cols; u++)
		{
			int count_neighbor_free = 0;
			int current_x = u;
			int current_y = v;
			if(edge_out.at<unsigned char>(current_y, current_x) != 0)
			{
				canny_split.at<unsigned char>(current_y, current_x) = 0;
				for(int i = -1; i <= 1; i ++)
				{
					for(int j = -1; j <= 1; j ++)
					{
						if(abs(i) + abs(j) == 1)
						{
							int searching_x = current_x + i;
							int searching_y = current_y + j;
							if (searching_x >= 0 && searching_y >= 0 && searching_y < ray_cast_occupy_map.rows && searching_x < ray_cast_occupy_map.cols )
							{
								if(ray_cast_occupy_map.at<int>(searching_y, searching_x) != 0)
									count_neighbor_free ++;
							}
						}
						
					}
				}
				
			}
			if(count_neighbor_free >=2)
			{
				ray_cast_occupy_map.at<int>(current_y, current_x) = 0;
			}
		}
	}
	imwrite("canny_ray_cast_occupy_map.png", ray_cast_occupy_map);
	imwrite("canny_split.png", canny_split);

	//test
	std::vector<cv::Point> my_node_points2; //variable for node point extraction

	Generate_even_ray_points(map_to_be_labeled, my_node_points2);
	Find_my_node_points(voronoi_map_backup, distance_map, my_node_points2);	

	cv::Mat ray_cast_occupy_map2 = cv::Mat::zeros(canny_split.rows,canny_split.cols,CV_32SC1);
	for(int node_point_index = 0; node_point_index < my_node_points2.size(); node_point_index++)
	{
		cv::Point current_point = my_node_points2[node_point_index];
		ray_occupy_map_func(current_point, ray_cast_occupy_map2, canny_split);
	}
	imwrite("canny_ray_cast_occupy_map.png", ray_cast_occupy_map2);
	ray_cast_occupy_map = ray_cast_occupy_map2.clone();

	//find the maximum and minimum grid
	unsigned int max_vote = 0;
	unsigned int min_vote = 10000;
	std::vector<int> votes_result;
	for (int v = 0; v < ray_cast_occupy_map.rows; v++)
	{
		for (int u = 0; u < ray_cast_occupy_map.cols; u++)
		{
			unsigned int sum_veto = ray_cast_occupy_map.at<int>(v, u);
			if(sum_veto > 0 && sum_veto <= my_node_points.size())
			{
				if(sum_veto > max_vote)
					max_vote = sum_veto;
				if(sum_veto < min_vote)
					min_vote = sum_veto;

				if(!contains(votes_result, sum_veto))
					votes_result.push_back(sum_veto);
			}
		}
	}
	std::cout << "max veto is: " << max_vote << std::endl;
	std::cout << "min veto is: " << min_vote << std::endl;
	std::cout << "votes_result  is: " << votes_result.size() << std::endl;

	cv::Mat merged_ray_map = cv::Mat::zeros(ray_cast_occupy_map.rows,ray_cast_occupy_map.cols,CV_32SC1);	//each id represent a area
	cv::Mat ray_occupied_map = ray_cast_occupy_map.clone();	
	std::vector<cv::Point> overlap_points;
	int last_area_id = 0;
	// from corner points to center points
	//for(int node_point_index = my_node_points.size()-1; node_point_index >=0; node_point_index--)
	for(int node_point_index = 0; node_point_index < my_node_points.size(); node_point_index++)
	{
		cv::Point current_node_point = my_node_points[node_point_index];
		SegmentArea(current_node_point, ray_cast_occupy_map, merged_ray_map, overlap_points, last_area_id);

	}

	Sew_segmented_map(map_to_be_labeled, merged_ray_map);

std::cout << "overlap_points.size: " << overlap_points.size() << std::endl;
std::cout << "last_area_id: " << last_area_id << std::endl;

	std::vector<Room> rooms; //Vector to save the rooms in this map
	for(int roomid = 1; roomid <= last_area_id; roomid++)
	{
		int current_room_id = roomid;
		int member_count = 0;
		Room current_room(current_room_id); //add the current Contour as a room
		for (int v = 0; v < merged_ray_map.rows; v++)
		{
			for (int u = 0; u < merged_ray_map.cols; u++)
			{
				unsigned int point_roomid = merged_ray_map.at<int>(v, u);
				if(point_roomid == current_room_id)
				{
					current_room.insertMemberPoint(cv::Point(u,v), map_resolution_from_subscription);
					member_count ++;
				}
			}
		}
		if(member_count > 0)
			rooms.push_back(current_room);
		//SegmentArea(current_point, ray_cast_occupy_map, map_to_be_labeled);
	}


	draw_segmented_map(merged_ray_map, rooms, "ray_segmented_map.png");
	std::cout << "Found " << rooms.size() << " rooms.\n";

	Room current_room(last_area_id);
	for (int i = 0; i < overlap_points.size(); i++)
	{
		current_room.insertMemberPoint(overlap_points[i], map_resolution_from_subscription);
	}
	rooms.push_back(current_room);
	draw_segmented_map(merged_ray_map, rooms, "overlap_map.png");




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

void VoronoiSegmentation::Find_critical_points(const cv::Mat& map, const cv::Mat& distance_map, std::vector<cv::Point>& critical_points, double map_resolution_from_subscription, int neighborhood_index, int max_iterations)
{
	cv::Mat voronoi_map = map.clone();
	for (int v = 0; v < voronoi_map.rows; v++)
	{
		for (int u = 0; u < voronoi_map.cols; u++)
		{
			if (voronoi_map.at<unsigned char>(v, u) == 127)		//voronoi vertex flag
			{
				//make the size of the region to be checked dependent on the distance of the current pixel to the closest
				//zero-pixel, so larger areas are split into more regions and small areas into fewer
				int eps = neighborhood_index / (int) distance_map.at<unsigned char>(v, u); //310
				int loopcounter = 0; //if a part of the graph is not connected to the rest this variable helps to stop the loop
				std::vector<cv::Point> temporary_points;	//neighboring-variables, which are different for each point
				std::set<cv::Point, cv_Point_comp> neighbor_points;	//neighboring-variables, which are different for each point
				int neighbor_count = 0;		//variable to save the number of neighbors for each point
				neighbor_points.insert(cv::Point(u,v)); //add the current Point to the neighborhood
				//find every Point along the voronoi graph in a specified neighborhood
				do
				{
					loopcounter++;
					//check every point in the neighborhood for other neighbors connected to it
					for(std::set<cv::Point, cv_Point_comp>::iterator it_neighbor_points = neighbor_points.begin(); it_neighbor_points != neighbor_points.end(); it_neighbor_points++)
					{
						for (int row_counter = -1; row_counter <= 1; row_counter++)
						{
							for (int column_counter = -1; column_counter <= 1; column_counter++)
							{
								if (row_counter == 0 && column_counter == 0)
									continue;

								//check the neighboring points
								//(if it already is in the neighborhood it doesn't need to be checked again)
								const cv::Point& current_neighbor_point = *it_neighbor_points;
								const int nu = current_neighbor_point.x + column_counter;
								const int nv = current_neighbor_point.y + row_counter;
								if (nv >= 0 && nu >= 0 && nv < voronoi_map.rows && nu < voronoi_map.cols &&
									voronoi_map.at<unsigned char>(nv, nu) == 127 && neighbor_points.find(cv::Point(nu, nv))==neighbor_points.end())
								{
									neighbor_count++;
									temporary_points.push_back(cv::Point(nu, nv));
								}
							}
						}
					}
					//go trough every found point after all neighborhood points have been checked and add them to it
					for (int temporary_point_index = 0; temporary_point_index < temporary_points.size(); temporary_point_index++)
					{
						neighbor_points.insert(temporary_points[temporary_point_index]);
						//make the found points white in the voronoi-map (already looked at)
						voronoi_map.at<unsigned char>(temporary_points[temporary_point_index].y, temporary_points[temporary_point_index].x) = 255;
						voronoi_map.at<unsigned char>(v, u) = 255;
					}
					//check if enough neighbors have been checked or checked enough times (e.g. at a small segment of the graph)
				} while (neighbor_count <= eps && loopcounter < max_iterations);
				//check every found point in the neighborhood if it is the local minimum in the distanceMap
				cv::Point current_critical_point_min = cv::Point(u, v);
				cv::Point current_critical_point_max = cv::Point(u, v);
				for(std::set<cv::Point, cv_Point_comp>::iterator it_neighbor_points = neighbor_points.begin(); it_neighbor_points != neighbor_points.end(); it_neighbor_points++)
				{
					if (distance_map.at<unsigned char>(it_neighbor_points->y, it_neighbor_points->x) < distance_map.at<unsigned char>(current_critical_point_min.y, current_critical_point_min.x))
					{
						current_critical_point_min = cv::Point(*it_neighbor_points);
					}
					if (distance_map.at<unsigned char>(it_neighbor_points->y, it_neighbor_points->x) > distance_map.at<unsigned char>(current_critical_point_max.y, current_critical_point_max.x))
					{
						current_critical_point_max = cv::Point(*it_neighbor_points);
					}
				}
				//add the local minimum point to the critical points
				//remove some critical points which is too closet
				bool insert_flag = true;
				for(int i = 0; i < critical_points.size(); i++)
				{
					double vector_px = critical_points[i].x - current_critical_point_min.x;
					double vector_py = critical_points[i].y - current_critical_point_min.y;
					double points_distance = std::sqrt(vector_px*vector_px + vector_py*vector_py);
					if(points_distance < k_robot_width/map_resolution_from_subscription)
					{
						insert_flag = false;
						break;
					}
						
				}
				if(insert_flag)
					critical_points.push_back(current_critical_point_min);


				insert_flag = true;
				for(int i = 0; i < critical_points.size(); i++)
				{
					double vector_px = critical_points[i].x - current_critical_point_max.x;
					double vector_py = critical_points[i].y - current_critical_point_max.y;
					double points_distance = std::sqrt(vector_px*vector_px + vector_py*vector_py);
					if(points_distance < k_robot_width/map_resolution_from_subscription)
					{
						insert_flag = false;
						break;
					}
						
				}
				if(insert_flag)
					critical_points.push_back(current_critical_point_max);
			}
		}
	}
}

void VoronoiSegmentation::Find_my_node_points(const cv::Mat& voronoi_map_backup, const cv::Mat& distance_map, std::vector<cv::Point>& my_true_node_points)
{
	//find special nodes
	std::vector<cv::Point> my_node_points; //variable for node point extraction

	for (int v = 1; v < voronoi_map_backup.rows-1; v++)
	{
		for (int u = 1; u < voronoi_map_backup.cols-1; u++)
		{
			if (voronoi_map_backup.at<unsigned char>(v, u) == 127)
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
						if (voronoi_map_backup.at<unsigned char>(v + row_counter, u + column_counter) == 127)
						{
							neighbor_count++;
						}
					}
				}
				if (neighbor_count > 2)
				{
					my_node_points.push_back(cv::Point(u,v));
				}
			}
		}
	}

	//push the feature points from contour .
	for(int i = 0; i < my_true_node_points.size(); i++)
	{
		my_node_points.push_back(my_true_node_points[i]);
	}

	std::cout << "my_original_node_points.size(): " << my_node_points.size() << std::endl;

	my_true_node_points.clear();

	for(int current_node_index = 0; current_node_index < my_node_points.size(); current_node_index ++)
	{
		bool true_node = true;
		for(int comp_node_index = 0; comp_node_index < my_node_points.size(); comp_node_index ++)
		{
			if(comp_node_index != current_node_index)
			{
				const double vector_x = my_node_points[current_node_index].x - my_node_points[comp_node_index].x;
				const double vector_y = my_node_points[current_node_index].y - my_node_points[comp_node_index].y;
				const double node_point_distance = std::sqrt(vector_x*vector_x + vector_y*vector_y);

				//only exist one node in 7*7 grids
				if(node_point_distance < k_filter_node_range)
				{
					//if two node are too close, remove the points are ranged by others
					if(distance_map.at<unsigned char>(my_node_points[current_node_index].y, my_node_points[current_node_index].x) + node_point_distance <= distance_map.at<unsigned char>(my_node_points[comp_node_index].y, my_node_points[comp_node_index].x))
						true_node = false;
						//if two node are two close, while with the same disatnce to obstacle, then remove the first.
					if(distance_map.at<unsigned char>(my_node_points[current_node_index].y, my_node_points[current_node_index].x) == distance_map.at<unsigned char>(my_node_points[comp_node_index].y, my_node_points[comp_node_index].x)
						&& (current_node_index < comp_node_index)
					)
						true_node = false;
					// remove the points whose radius crossed ..
					// if(distance_map.at<unsigned char>(my_node_points[current_node_index].y, my_node_points[current_node_index].x) == distance_map.at<unsigned char>(my_node_points[comp_node_index].y, my_node_points[comp_node_index].x))
					// {
					// 	bool compare_nodes = true;
					// 	int compare_range = 1;
					// 	do{
					// 		int my_neighbor_count_current = 0;	
					// 		int my_neighbor_count_comp = 0;	// variable to save the number of neighbors for each point
					// 		// check 3x3 region around current pixel
					// 		for (int row_counter = -compare_range; row_counter <= compare_range; row_counter++)
					// 		{
					// 			for (int column_counter = -compare_range; column_counter <= compare_range; column_counter++)
					// 			{
					// 				// don't check the point itself
					// 				if (row_counter == 0 && column_counter == 0)
					// 					continue;

					// 				//check if neighbors are colored with the voronoi-color
					// 				if (voronoi_map_backup.at<unsigned char>(my_node_points[current_node_index].y + row_counter, my_node_points[current_node_index].x + column_counter) == 127)
					// 				{
					// 					my_neighbor_count_current++;
					// 				}
					// 			}
					// 		}
					// 		for (int row_counter = -compare_range; row_counter <= compare_range; row_counter++)
					// 		{
					// 			for (int column_counter = -compare_range; column_counter <= compare_range; column_counter++)
					// 			{
					// 				// don't check the point itself
					// 				if (row_counter == 0 && column_counter == 0)
					// 					continue;

					// 				//check if neighbors are colored with the voronoi-color
					// 				if (voronoi_map_backup.at<unsigned char>(my_node_points[comp_node_index].y + row_counter, my_node_points[comp_node_index].x + column_counter) == 127)
					// 				{
					// 					my_neighbor_count_comp++;
					// 				}
					// 			}
					// 		}
					// 		if(my_neighbor_count_current != my_neighbor_count_comp)
					// 		{
					// 			compare_nodes = false;
					// 			if(my_neighbor_count_current < my_neighbor_count_comp)
					// 				true_node = false;
					// 		}
					// 		else
					// 		{
					// 			compare_range ++;
					// 		}
							
					// 	}while(compare_nodes);
						
					// }
				}
			}
		}
		if(true_node)
			my_true_node_points.push_back(my_node_points[current_node_index]);
	}

std::cout << "true node size: " << my_true_node_points.size() << std::endl;
// 	// add some corner points for ray segment
// 	//find special nodes.  dead end point on voronoi graph.
// 	int dead_end_range = 1;
// 	for (int v = dead_end_range; v < voronoi_map_backup.rows-dead_end_range; v++)
// 	{
// 		for (int u = dead_end_range; u < voronoi_map_backup.cols-dead_end_range; u++)
// 		{
// 			if (voronoi_map_backup.at<unsigned char>(v, u) == 127)
// 			{
// 				int neighbor_count = 0;	// variable to save the number of neighbors for each point
// 				// check 3x3 region around current pixel
// 				for (int row_counter = -dead_end_range; row_counter <= dead_end_range; row_counter++)
// 				{
// 					for (int column_counter = -dead_end_range; column_counter <= dead_end_range; column_counter++)
// 					{
// 						// don't check the point itself
// 						if (row_counter == 0 && column_counter == 0)
// 							continue;

// 						//check if neighbors are colored with the voronoi-color
// 						if (voronoi_map_backup.at<unsigned char>(v + row_counter, u + column_counter) == 127)
// 						{
// 							neighbor_count++;
// 						}
// 					}
// 				}
// 				if (neighbor_count < 2)
// 				{
// 					my_true_node_points.push_back(cv::Point(u,v));
// 				}
// 			}
// 		}
// 	}
// std::cout << "true node size after corner added: " << my_true_node_points.size() << std::endl;


	// cv::Mat display_test = map_to_be_labeled.clone();
	// for (size_t i=0; i<my_true_node_points.size(); ++i)
	// 	cv::circle(display_test, my_true_node_points[i], 2, cv::Scalar(128), -1);
	// imwrite("my_true_node_points.png", display_test);
	// std::cout << "my_true_node_points.size(): " << my_true_node_points.size() << std::endl;

//sort the node points
for(int i = 0; i < my_true_node_points.size(); i++)
{
	for(int j = i+1; j < my_true_node_points.size(); j++)
	{
		bool compare_nodes = true;
		int compare_range = 1;
		if (voronoi_map_backup.at<unsigned char>(my_true_node_points[j].y, my_true_node_points[j].x) != 127)
			compare_nodes = false;

		while(compare_nodes)
		{
			int my_neighbor_count_current = 0;	
			int my_neighbor_count_comp = 0;	// variable to save the number of neighbors for each point
			// check 3x3 region around current pixel
			for (int row_counter = -compare_range; row_counter <= compare_range; row_counter++)
			{
				for (int column_counter = -compare_range; column_counter <= compare_range; column_counter++)
				{
					// don't check the point itself
					if (row_counter == 0 && column_counter == 0)
						continue;

					//check if neighbors are colored with the voronoi-color
					if (voronoi_map_backup.at<unsigned char>(my_true_node_points[i].y + row_counter, my_true_node_points[i].x + column_counter) == 127)
					{
						my_neighbor_count_current++;
					}
				}
			}
			for (int row_counter = -compare_range; row_counter <= compare_range; row_counter++)
			{
				for (int column_counter = -compare_range; column_counter <= compare_range; column_counter++)
				{
					// don't check the point itself
					if (row_counter == 0 && column_counter == 0)
						continue;

					//check if neighbors are colored with the voronoi-color
					if (voronoi_map_backup.at<unsigned char>(my_true_node_points[j].y + row_counter, my_true_node_points[j].x + column_counter) == 127)
					{
						my_neighbor_count_comp++;
					}
				}
			}
			if(my_neighbor_count_current != my_neighbor_count_comp)
			{
				compare_nodes = false;
				if(my_neighbor_count_current < my_neighbor_count_comp)
				{
					cv::Point temp = my_true_node_points[i];
					my_true_node_points[i] = my_true_node_points[j];
					my_true_node_points[j] = temp;
				}
			}
			else
			{
				if(my_neighbor_count_current + my_neighbor_count_comp > 0)
					compare_range ++;
				else
					compare_nodes = false;
				
			}
			
		}
	} 
} 


// draw circle around node_points, without overlap
// 	std::vector < cv::Scalar > already_used_colors; //saving-vector to save the already used coloures

// 	std::vector<Room> rooms; //Vector to save the rooms in this map
// std::vector<cv::Point> drawed_node_points; //variable for node point extraction
// std::vector<float> drawed_node_points_radius;
// 	//1. Erode map one time, so small gaps are closed
// //	cv::erode(voronoi_map_, voronoi_map_, cv::Mat(), cv::Point(-1, -1), 1);
// cv::Mat node_grow_map = map_to_be_labeled.clone();
// 	for (int i = 0; i < my_true_node_points.size(); i++)
// 	{ 
// 		bool draw = true;
// 		bool cut = false;
// 		float circle_radiusi = distance_map.at<unsigned char>(my_true_node_points[i].y, my_true_node_points[i].x);
// 		float cut_radius = 10000;
// 		for (int j = 0; j < drawed_node_points.size(); j++)
// 		{
// 			const double vector_x = my_true_node_points[i].x - drawed_node_points[j].x;
// 			const double vector_y = my_true_node_points[i].y - drawed_node_points[j].y;
// 			const double node_point_distance = std::sqrt(vector_x*vector_x + vector_y*vector_y);

// 			float circle_radiusj = drawed_node_points_radius[j];
// 			if(node_point_distance <= circle_radiusj)
// 				draw = false;
// 			else if(node_point_distance < circle_radiusj + circle_radiusi)
// 			{
// 				cut = true;
// 				if( (node_point_distance - circle_radiusj) < cut_radius)
// 					cut_radius = node_point_distance - circle_radiusj;
// 			}
				
			
// 		}
// 		if(draw && !cut)
// 		{
// 			if(circle_radiusi > k_robot_width/map_resolution_from_subscription)
// 			{
// 				cv::circle(node_grow_map, my_true_node_points[i], circle_radiusi, cv::Scalar(0), 2);
// 				drawed_node_points.push_back(my_true_node_points[i]);
// 				drawed_node_points_radius.push_back(circle_radiusi);
// 			}	
// 		}
			
// 		if(draw && cut)
// 		{
// 			if(cut_radius > k_robot_width/map_resolution_from_subscription)
// 			{
// 				cv::circle(node_grow_map, my_true_node_points[i], cut_radius, cv::Scalar(0), 2);
// 				drawed_node_points.push_back(my_true_node_points[i]);
// 				drawed_node_points_radius.push_back(cut_radius);
// 			}
// 		}
// 	}
// 	imwrite("node_grow_map.png", node_grow_map);
	//std::cout << "my_true_node_points.size(): " << my_true_node_points.size() << std::endl;


}





	//*************III. draw the critical lines from every found critical Point to its two closest zero-pixel****************
	//
	//map to draw the critical lines and fill the map with random colors
	// map_to_be_labeled.convertTo(segmented_map, CV_32SC1, 256, 0); // rescale to 32 int, 255 --> 255*256 = 65280

	// // 1. Get the points of the contour, which are the possible closest points for a critical point
	// //clone the map to extract the contours, because after using OpenCV find-/drawContours
	// //the map will be different from the original one
	// cv::Mat temporary_map_to_extract_the_contours = segmented_map.clone();
	// std::vector < std::vector<cv::Point> > contours;
	// cv::findContours(temporary_map_to_extract_the_contours, contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);

	// // 2. Get the basis-points for each critical-point
	// std::vector<cv::Point> basis_points_1, basis_points_2;
	// std::vector<double> length_of_critical_line;
	// std::vector<double> length_of_minimal_radius;
	// std::vector<double> angles; //the angles between the basis-lines of each critical Point
	// for (int critical_point_index = 0; critical_point_index < critical_points.size(); critical_point_index++)
	// {
	// 	//set inital points and values for the basis points so the distance comparison can be done
	// 	cv::Point basis_point_1 = contours[0][0];
	// 	cv::Point basis_point_2 = contours[0][1];
	// 	//inital values of the first vector from the current critical point to the contour points and for the distance of it
	// 	const cv::Point& critical_point = critical_points[critical_point_index];
	// 	double vector_x_1 = critical_point.x - contours[0][0].x;
	// 	double vector_y_1 = critical_point.y - contours[0][0].y;
	// 	double distance_basis_1 = std::sqrt(vector_x_1*vector_x_1 + vector_y_1*vector_y_1);
	// 	//inital values of the second vector from the current critical point to the contour points and for the distance of it
	// 	double vector_x_2 = critical_point.x - contours[0][1].x;
	// 	double vector_y_2 = critical_point.y - contours[0][1].y;
	// 	double distance_basis_2 = std::sqrt(vector_x_2*vector_x_2 + vector_y_2*vector_y_2);

	// 	//find first basis point
	// 	int basis_vector_1_x, basis_vector_2_x, basis_vector_1_y, basis_vector_2_y;
	// 	for (int c = 0; c < contours.size(); c++)
	// 	{
	// 		for (int p = 0; p < contours[c].size(); p++)
	// 		{
	// 			//calculate the Euclidian distance from the critical Point to the Point on the contour
	// 			const double vector_x = contours[c][p].x - critical_point.x;
	// 			const double vector_y = contours[c][p].y - critical_point.y;
	// 			const double current_distance = std::sqrt(vector_x*vector_x + vector_y*vector_y);
	// 			//compare the distance to the saved distances if it is smaller
	// 			if (current_distance < distance_basis_1)
	// 			{
	// 				distance_basis_1 = current_distance;
	// 				basis_point_1 = contours[c][p];
	// 				basis_vector_1_x = vector_x;
	// 				basis_vector_1_y = vector_y;
	// 			}
	// 		}
	// 	}
	// 	//find second basisPpoint
	// 	for (int c = 0; c < contours.size(); c++)
	// 	{
	// 		for (int p = 0; p < contours[c].size(); p++)
	// 		{
	// 			//calculate the Euclidian distance from the critical point to the point on the contour
	// 			const double vector_x = contours[c][p].x - critical_point.x;
	// 			const double vector_y = contours[c][p].y - critical_point.y;
	// 			const double current_distance = std::sqrt(vector_x*vector_x + vector_y*vector_y);
	// 			//calculate the distance between the current contour point and the first basis point to make sure they
	// 			//are not too close to each other
	// 			const double vector_x_basis = basis_point_1.x - contours[c][p].x;
	// 			const double vector_y_basis = basis_point_1.y - contours[c][p].y;
	// 			const double basis_distance = std::sqrt(vector_x_basis*vector_x_basis + vector_y_basis*vector_y_basis);
	// 			if (current_distance > distance_basis_1 && current_distance < distance_basis_2 &&
	// 				basis_distance > (double) distance_map.at<unsigned char>(critical_point.y, critical_point.x))
	// 			{
	// 				distance_basis_2 = current_distance;
	// 				basis_point_2 = contours[c][p];
	// 				basis_vector_2_x = vector_x;
	// 				basis_vector_2_y = vector_y;
	// 			}
	// 		}
	// 	}
	// 	//calculate angle between the vectors from the critical Point to the found basis-points
	// 	double current_angle = std::acos((basis_vector_1_x * basis_vector_2_x + basis_vector_1_y * basis_vector_2_y) / (distance_basis_1 * distance_basis_2)) * 180.0 / PI;

	// 	//save the critical line with its calculated values
	// 	basis_points_1.push_back(basis_point_1);
	// 	basis_points_2.push_back(basis_point_2);
	// 	length_of_critical_line.push_back(distance_basis_1 + distance_basis_2);
	// 	length_of_minimal_radius.push_back(distance_basis_1);
	// 	angles.push_back(current_angle);
		
	// }

	//find special nodes



// 	//3. Check which critical points should be used for the segmentation. This is done by checking the points that are
// 	//   in a specified distance to each other and take the point with the largest calculated angle, because larger angles
// 	//   correspond to a separation across the room, which is more useful
// 	int critical_line_count = 0;
// 	for (int first_critical_point = 0; first_critical_point < critical_points.size(); first_critical_point++)
// 	{
// 		//reset variable for checking if the line should be drawn
// 		bool draw = true;

		
// 		for (int second_critical_point = 0; second_critical_point < critical_points.size(); second_critical_point++)
// 		{
// 			if (second_critical_point != first_critical_point)
// 			{
// 				//get distance of the two current Points
// 				const double vector_x = critical_points[second_critical_point].x - critical_points[first_critical_point].x;
// 				const double vector_y = critical_points[second_critical_point].y - critical_points[first_critical_point].y;
// 				const double critical_point_distance = std::sqrt(vector_x*vector_x + vector_y*vector_y);
// 				double angle_difference = abs(angles[first_critical_point] - angles[second_critical_point]);
// 				//check if the points are too close to each other corresponding to the distance to the nearest black pixel
// 				//of the current critical point. This is done because critical points at doors are closer to the black region
// 				//and shorter and may be eliminated in the following step. By reducing the checking distance at this point
// 				//it gets better.
// ///////**************************************mainly need to be modified code*********************************
// //draw the local biggest and smallest circles
// 				//local minimal; distance to obstacle < 2m, 
// 				//if the critical point was contained
// 				if( (critical_point_distance + length_of_minimal_radius[first_critical_point]) <= length_of_minimal_radius[second_critical_point])
// 					draw = false;

// 				if( length_of_minimal_radius[first_critical_point] <= k_robot_width/map_resolution_from_subscription)
// 					draw = false;	//even to occupy it.

// 				//for large radius critical points, 
// 				if( critical_point_distance < (length_of_minimal_radius[second_critical_point] + length_of_minimal_radius[first_critical_point]) &&
// 					length_of_minimal_radius[first_critical_point] < length_of_minimal_radius[second_critical_point] && 
// 					(angles[first_critical_point] < angles[second_critical_point]) && length_of_minimal_radius[first_critical_point] > k_robot_width*5/map_resolution_from_subscription)	//assume door's width is  smaller than 3m.
// 					draw = false;

// 				//if two circle are similar ; overlap a lot; corridors
// 				if (critical_point_distance < ((int) distance_map.at<unsigned char>(critical_points[second_critical_point].y, critical_points[second_critical_point].x) * min_critical_point_distance_factor)) //1.7
// 				{
// 					//if one point in neighborhood is found that has a larger angle the actual to-be-checked point shouldn't be drawn
// 					if (angles[first_critical_point] < angles[second_critical_point] && length_of_minimal_radius[first_critical_point] > k_robot_width*3/map_resolution_from_subscription)
// 					{
// 						draw = false;
// 					}
					
// 				}
// 			}
// 		}
// 		//4. draw critical-lines if angle of point is larger than the other
// 		if (draw)
// 		{
// 			if(length_of_minimal_radius[first_critical_point] < k_robot_width*8/map_resolution_from_subscription)
// 			{
// 				cv::line(voronoi_map, critical_points[first_critical_point], basis_points_1[first_critical_point], cv::Scalar(0), 2);
// 				cv::line(voronoi_map, critical_points[first_critical_point], basis_points_2[first_critical_point], cv::Scalar(0), 2);
// 			}
// 			else
// 			{
// 				cv::circle(voronoi_map, critical_points[first_critical_point], length_of_minimal_radius[first_critical_point], cv::Scalar(0), 2);
// 			}
// 			critical_line_count ++;
// 		}
// 	}
// //	if(display_map == true)
// //		cv::imshow("voronoi_map", voronoi_map);
// 	imwrite("voronoi_map_critical_line.png", voronoi_map);
// std::cout << "critical_line_count : " << critical_line_count << std::endl;
// 	//*************error info**********Find the Contours seperated from the critcal lines and fill them with color******************

// 	std::vector < cv::Scalar > already_used_colors; //saving-vector to save the already used coloures

// 	std::vector < cv::Vec4i > hierarchy; //variables for coloring the map

// 	std::vector<Room> rooms; //Vector to save the rooms in this map

// 	//1. Erode map one time, so small gaps are closed
// //	cv::erode(voronoi_map_, voronoi_map_, cv::Mat(), cv::Point(-1, -1), 1);
// 	cv::erode(voronoi_map, voronoi_map, cv::Mat());
// 	cv::erode(voronoi_map, voronoi_map, cv::Mat());
// 	cv::findContours(voronoi_map, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

// 	for (int current_contour = 0; current_contour < contours.size(); current_contour++)
// 	{ //only draw contours that aren't holes
// 		if (hierarchy[current_contour][3] == -1)
// 		{
// 			//calculate area for the contour and check if it is large enough to be a room
// 			double room_area = map_resolution_from_subscription * map_resolution_from_subscription * cv::contourArea(contours[current_contour]);
// 			if (room_area >= room_area_factor_lower_limit && room_area <= room_area_factor_upper_limit)
// 			{
// 				//2. Draw the region with a random color into the map if it is large/small enough
// 				bool drawn = false;
// 				int loop_counter = 0; //counter if the loop gets into a endless loop
// 				do
// 				{
// 					loop_counter++;
// 					int random_number = rand() % 52224 + 13056;
// 					cv::Scalar fill_colour(random_number);
// 					//check if color has already been used
// 					if (!contains(already_used_colors, fill_colour) || loop_counter > 1000)
// 					{
// 						cv::drawContours(segmented_map, contours, current_contour, fill_colour, 1);  
// 						//not for drawing the room, but for the room_id; each pixel denotes a roomid.
// 						//only draw the contours points
// 						already_used_colors.push_back(fill_colour);
// 						Room current_room(random_number); //add the current Contour as a room
// 						for (int point = 0; point < contours[current_contour].size(); point++) //add contour points to room
// 						{
// 							current_room.insertMemberPoint(cv::Point(contours[current_contour][point]), map_resolution_from_subscription);
// 						}
// 						rooms.push_back(current_room);
// 						drawn = true;
// 					}
// 				} while (!drawn);
// 			}
// 		}
// 	}
// 	draw_segmented_map(segmented_map, rooms, "segmented_map_after_contours.png");
// 	std::cout << "Found " << rooms.size() << " rooms.\n";

// 	//3.fill the last white areas with the surrounding color
// 	wavefrontRegionGrowing(segmented_map);
// 	draw_segmented_map(segmented_map, rooms, "segmented_map_before_merge_room.png");

// 	//4.merge the rooms together if neccessary
// 	mergeRooms(segmented_map, rooms, map_resolution_from_subscription, max_area_for_merging, display_map);
// 	std::cout << "after merged Found " << rooms.size() << " rooms.\n";

// 	draw_segmented_map(segmented_map, rooms, "segmented_map_after_merge.png");

// 	std::vector<std::vector<cv::Point>> segment_result;
// 	get_segment_points_set(segmented_map, rooms, segment_result);
// 	std::cout << "segment_result size " << segment_result.size() << " segment line.\n";

// 	draw_segmented_line(segmented_map, segment_result, "segmented_line.png");