#include <ipa_room_segmentation/voronoi_segmentation.h>

#include <ipa_room_segmentation/wavefront_region_growing.h>
#include <ipa_room_segmentation/contains.h>

#include <ipa_room_segmentation/timer.h>
#include <set>


AbstractVoronoiSegmentation::AbstractVoronoiSegmentation()
{

}

bool AbstractVoronoiSegmentation::determineRoomIndexFromRoomID(const std::vector<Room>& rooms, const int room_id, size_t& room_index)
{
	bool found_id = false;
	for (size_t r = 0; r < rooms.size(); r++)
	{
		if (rooms[r].getID() == room_id)
		{
			room_index = r;
			found_id = true;
			break;
		}
	}

	return found_id;
}

void AbstractVoronoiSegmentation::mergeRoomPair(std::vector<Room>& rooms, const int target_index, const int room_to_merge_index, cv::Mat& segmented_map, const double map_resolution)
{
	// integrate room to merge into target and delete merged room
	const int target_id = rooms[target_index].getID();
	const int room_to_merge_id = rooms[room_to_merge_index].getID();
	rooms[target_index].mergeRoom(rooms[room_to_merge_index], map_resolution);
	rooms[room_to_merge_index].setRoomId(rooms[target_index].getID(), segmented_map);
	rooms.erase(rooms.begin()+room_to_merge_index);
	std::sort(rooms.begin(), rooms.end(), sortRoomsAscending);

	// update neighborhood statistics for remaining rooms
	for (size_t i=0; i<rooms.size(); ++i)
	{
		std::vector<int>& neighbor_ids = rooms[i].getNeighborIDs();
		std::vector<int>::iterator it = std::find(neighbor_ids.begin(), neighbor_ids.end(), room_to_merge_id);
		if (it != neighbor_ids.end())
		{
			std::vector<int>::iterator it2 = std::find(neighbor_ids.begin(), neighbor_ids.end(), target_id);
			if (it2 != neighbor_ids.end())
				neighbor_ids.erase(it);
			else
				*it = target_id;
		}

		std::map<int,int>& neighbor_statistics = rooms[i].getNeighborStatistics();
		std::map<int,int>::iterator it3 = neighbor_statistics.find(room_to_merge_id);
		if (it3 != neighbor_statistics.end())
		{
			std::map<int,int>::iterator it4 = neighbor_statistics.find(target_id);
			if (it4 != neighbor_statistics.end())
				it4->second += it3->second;
			else
				neighbor_statistics[target_id] = it3->second;
			neighbor_statistics.erase(it3);
		}
	}
}

void AbstractVoronoiSegmentation::drawVoronoi(cv::Mat &img, const std::vector<std::vector<cv::Point2f> >& facets_of_voronoi, const cv::Scalar voronoi_color, const cv::Mat& eroded_map)
{
	// go trough each facet of the calculated Voronoi-graph and check if it should be drawn.
	for (std::vector<std::vector<cv::Point2f> >::const_iterator current_contour = facets_of_voronoi.begin(); current_contour != facets_of_voronoi.end(); ++current_contour)
	{
		// saving-variable for the last Point that has been looked at
		cv::Point2f last_point = current_contour->back();
		// draw each line of the voronoi-cell
		for (size_t c = 0; c < current_contour->size(); ++c)
		{
			// variable to check, whether a Point is inside a white area or not
			bool inside = true;
			cv::Point2f current_point = current_contour->at(c);
			// only draw lines that are inside the map-contour
			if (((int)current_point.x<0) || ((int)current_point.x >= eroded_map.cols) ||
				((int)current_point.y<0) || ((int)current_point.y >= eroded_map.rows) ||
				eroded_map.at<uchar>((int)current_point.y, (int)current_point.x) == 0 ||
				((int)last_point.x<0) || ((int)last_point.x >= eroded_map.cols) ||
				((int)last_point.y<0) || ((int)last_point.y >= eroded_map.rows) ||
				eroded_map.at<uchar>((int)last_point.y, (int)last_point.x) == 0)
				inside = false;
			if (inside)
			{
				cv::line(img, last_point, current_point, voronoi_color, 1);
			}
			last_point = current_point;
		}
	}
}

//****************Create the Generalized Voronoi-Diagram**********************
// This function is here to create the generalized voronoi-graph in the given map. It does following steps:
//	1. It finds every discretized contour in the given map (they are saved as vector<Point>). Then it takes these
//	   contour-Points and adds them to the OpenCV Delaunay generator from which the voronoi-cells can be generated.
//	2. Finally it gets the boundary-Points of the voronoi-cells with getVoronoiFacetList. It takes these facets
//	   and draws them using the drawVoronoi function. This function draws the facets that only have Points inside
//	   the map-contour (other lines go to not-reachable places and are not necessary to be looked at).
//	3. It returns the map that has the generalized voronoi-graph drawn in.
void AbstractVoronoiSegmentation::createVoronoiGraph(cv::Mat& map_for_voronoi_generation)
{
	cv::Mat map_to_draw_voronoi_in = map_for_voronoi_generation.clone(); //variable to save the given map for drawing in the voronoi-diagram

	cv::Mat temporary_map_to_calculate_voronoi = map_for_voronoi_generation.clone(); //variable to save the given map in the createVoronoiGraph-function

	//apply a closing-operator on the map so bad parts are neglected
	cv::erode(temporary_map_to_calculate_voronoi, temporary_map_to_calculate_voronoi, cv::Mat());
	cv::dilate(temporary_map_to_calculate_voronoi, temporary_map_to_calculate_voronoi, cv::Mat());

	//********************1. Get OpenCV delaunay-traingulation******************************
	cv::Rect rect(0, 0, map_to_draw_voronoi_in.cols, map_to_draw_voronoi_in.rows); //variables to generate the voronoi-diagram, using OpenCVs delaunay-triangulation
	cv::Subdiv2D subdiv(rect);
	std::vector<std::vector<cv::Point> > hole_contours; //variable to save the hole-contours (= black holes inside the white map)
	std::vector<std::vector<cv::Point> > contours; //variables for contour extraction and discretisation
	//hierarchy saves if the contours are hole-contours:
	//hierarchy[{0,1,2,3}]={next contour (same level), previous contour (same level), child contour, parent contour}
	//child-contour = 1 if it has one, = -1 if not, same for parent_contour
	std::vector<cv::Vec4i> hierarchy;

	//get contours of the map
	cv::Mat temp = map_to_draw_voronoi_in.clone();
	cv::findContours(temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
	cv::drawContours(map_to_draw_voronoi_in, contours, -1, cv::Scalar(255), CV_FILLED);
	//put every point of the map-contours into the Delaunay-generator of OpenCV
	for (int current_contour = 0; current_contour < contours.size(); current_contour++)
	{
		for (int current_point = 0; current_point < contours[current_contour].size(); current_point++)
		{
			cv::Point fp = contours[current_contour][current_point];
			subdiv.insert(fp);
		}
		//get the contours of the black holes --> it is necessary to check if points are inside these in drawVoronoi
		if (hierarchy[current_contour][2] == -1 && hierarchy[current_contour][3] != -1)
		{
			hole_contours.push_back(contours[current_contour]);
		}
	}

	//********************2. Get facets and draw voronoi-Graph******************************
	//erode the map so that points near the boundary are not drawn later (see drawVoronoi)
	cv::Mat eroded_map;
	cv::Point anchor(-1, -1);
	cv::erode(temporary_map_to_calculate_voronoi, eroded_map, cv::Mat(), anchor, 2);

	//get the Voronoi regions from the delaunay-subdivision graph
	const cv::Scalar voronoi_color(127); //define the voronoi-drawing colour
	std::vector<std::vector<cv::Point2f> > voronoi_facets; // variables to find the facets and centers of the voronoi-cells
	std::vector<cv::Point2f> voronoi_centers;
	subdiv.getVoronoiFacetList(std::vector<int>(), voronoi_facets, voronoi_centers);

	//draw the voronoi-regions into the map
	drawVoronoi(map_to_draw_voronoi_in, voronoi_facets, voronoi_color, eroded_map);

	//make pixels black, which were black before and were colored by the voronoi-regions
	for (int v = 0; v < map_to_draw_voronoi_in.rows; v++)
	{
		for (int u = 0; u < map_to_draw_voronoi_in.cols; u++)
		{
			if (map_for_voronoi_generation.at<unsigned char>(v, u) == 0)
			{
				map_to_draw_voronoi_in.at<unsigned char>(v, u) = 0;
			}
		}
	}
	map_for_voronoi_generation = map_to_draw_voronoi_in;
}

void AbstractVoronoiSegmentation::pruneVoronoiGraph(cv::Mat& voronoi_map, std::set<cv::Point, cv_Point_comp>& node_points)
{
	// 1.extract the node-points that have at least three neighbors on the voronoi diagram
	//	node-points are points on the voronoi-graph that have at least 3 neighbors
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
				if (neighbor_count > 2)
				{
					node_points.insert(cv::Point(u,v));
				}
			}
		}
	}

	// 2.reduce the side-lines along the voronoi-graph by checking if it has only one neighbor until a node-point is reached
	//	--> make it white
	//	repeat a large enough number of times so the graph converges
	for (int step = 0; step < 500; step++)
	{
		for (int v = 0; v < voronoi_map.rows; v++)
		{
			for (int u = 0; u < voronoi_map.cols; u++)
			{
				// set that the point is a point along the graph and not a side-line
				if (voronoi_map.at<unsigned char>(v, u) == 127)
				{
					int neighbor_count = 0;		//variable to save the number of neighbors for each point
					for (int row_counter = -1; row_counter <= 1; row_counter++)
					{
						for (int column_counter = -1; column_counter <= 1; column_counter++)
						{
							// don't check the point itself
							if (row_counter == 0 && column_counter == 0)
								continue;

							// check the surrounding points
							const int nv = v + row_counter;
							const int nu = u + column_counter;
							if (nv >= 0 && nu >= 0 && nv < voronoi_map.rows && nu < voronoi_map.cols && voronoi_map.at<unsigned char>(nv, nu) == 127)
							{
								neighbor_count++;
							}
						}
					}
					//if the current point is a node point found in the previous step, it belongs to the voronoi-graph
					if (neighbor_count <= 1 && node_points.find(cv::Point(u,v)) == node_points.end())
					{
						//if the Point isn't on the voronoi-graph make it white
						voronoi_map.at<unsigned char>(v, u) = 255;
					}
				}
			}
		}
	}
}

void AbstractVoronoiSegmentation::mergeRooms(cv::Mat& map_to_merge_rooms, std::vector<Room>& rooms, double map_resolution_from_subscription, double max_area_for_merging, bool display_map)
{
	// This function takes the segmented Map from the original Voronoi-segmentation-algorithm and merges rooms together,
	// that are small enough and have only two or one neighbor.

	// 1. go trough every pixel and add points to the rooms with the same ID
	for (int y = 0; y < map_to_merge_rooms.rows; y++)
	{
		for (int x = 0; x < map_to_merge_rooms.cols; x++)
		{
			int current_id = map_to_merge_rooms.at<int>(y, x);
			if (current_id != 0)
			{
				for (size_t current_room = 0; current_room < rooms.size(); current_room++) //add the Points with the same Id as a room to it
				{
					if (rooms[current_room].getID() == current_id) //insert the current point into the corresponding room
					{
						rooms[current_room].insertMemberPoint(cv::Point(x, y), map_resolution_from_subscription);
						break;
					}
				}
			}
		}
	}

	// 2. add the neighbor IDs for every point
	for (int current_room = 0; current_room < rooms.size(); current_room++)
	{
		const int current_id = rooms[current_room].getID();
		std::vector<cv::Point> considered_neighbors;		// storage for already counted neighborhood points
		const std::vector<cv::Point>& current_points = rooms[current_room].getMembers();
		for (int current_point = 0; current_point < current_points.size(); current_point++)
		{
			for (int row_counter = -1; row_counter <= 1; row_counter++)
			{
				for (int col_counter = -1; col_counter <= 1; col_counter++)
				{
					const int label = map_to_merge_rooms.at<int>(current_points[current_point].y + row_counter, current_points[current_point].x + col_counter);

					// collect neighbor IDs
					if (label != 0 && label != current_id)
						rooms[current_room].addNeighborID(label);	//if label ==0, that's wall

					// neighborhood statistics
					cv::Point neighbor_point(current_points[current_point].x + col_counter, current_points[current_point].y + row_counter);
					if (!contains(considered_neighbors, neighbor_point) && label != current_id)
					{
						rooms[current_room].addNeighbor(label);	//the number of neighbors
						considered_neighbors.push_back(neighbor_point);
					}
				}
			}
		}
	}
draw_segmented_map(map_to_merge_rooms, rooms, "a_before.png");

	// 3. merge criteria
	// sort rooms ascending by area
	std::sort(rooms.begin(), rooms.end(), sortRoomsAscending);
	// a) rooms with one neighbor and max. 75% walls around
	for (int current_room_index = 0; current_room_index < rooms.size(); ) //tranverse room
	{
		Room& current_room = rooms[current_room_index];	
		bool merge_rooms = false;
		size_t merge_index = 0;

		// if the room only has one neighbor room; and the area is smaller than max_area and wall_ratio <=0.75, check whether be merged
		if (current_room.getNeighborCount() == 1 && current_room.getArea() < max_area_for_merging && current_room.getWallToPerimeterRatio() <= 0.75)
		{
			// check every room if it should be merged with its neighbor that it shares the most boundary with
			merge_rooms = determineRoomIndexFromRoomID(rooms, current_room.getNeighborWithLargestCommonBorder(), merge_index);	//merge_index: target index  room[merge_index]
		}

		//if find the biggest common border room...then merge the two room;  into  the merge_index
		if (merge_rooms == true)
		{
			//std::cout << "merge " << current_room.getCenter() << ", id=" << current_room.getID() << " into " << rooms[merge_index].getCenter() << ", id=" << rooms[merge_index].getID() << std::endl;
			mergeRoomPair(rooms, merge_index, current_room_index, map_to_merge_rooms, map_resolution_from_subscription);
			current_room_index = 0;	// set 0, repeat the operation
		}
		else
			current_room_index++;
	}
	if (display_map == true)
		cv::imshow("a", map_to_merge_rooms);
//imwrite("a.png", map_to_merge_rooms);
draw_segmented_map(map_to_merge_rooms, rooms, "a.png");

	// b) small rooms
	for (int current_room_index = 0; current_room_index < rooms.size(); )
	{
		Room& current_room = rooms[current_room_index];
		bool merge_rooms = false;
		size_t merge_index = 0;

		const int max_border_neighbor_id = current_room.getNeighborWithLargestCommonBorder();
		if (current_room.getArea() < 2.0 && (double)current_room.getNeighborStatistics()[max_border_neighbor_id]/current_room.getPerimeter() > 0.2)	//0.2 too small?
		{
			// merge with that neighbor that shares the most neighboring pixels
			merge_rooms = determineRoomIndexFromRoomID(rooms, max_border_neighbor_id, merge_index);
			if ((double)rooms[merge_index].getWallToPerimeterRatio() > 0.8) //0.8
				merge_rooms = false;
		}

		if (merge_rooms == true)
		{
			//std::cout << "merge " << current_room.getCenter() << ", id=" << current_room.getID() << " into " << rooms[merge_index].getCenter() << ", id=" << rooms[merge_index].getID() << std::endl;
			mergeRoomPair(rooms, merge_index, current_room_index, map_to_merge_rooms, map_resolution_from_subscription);
			current_room_index = 0;
		}
		else
			current_room_index++;
	}
	if (display_map == true)
		cv::imshow("b", map_to_merge_rooms);
draw_segmented_map(map_to_merge_rooms, rooms, "b.png");

	// c) merge a room with one neighbor that has max. 2 neighbors and sufficient wall ratio (connect parts inside a room)
	for (int current_room_index = 0; current_room_index < rooms.size(); )
	{
		Room& current_room = rooms[current_room_index];
		bool merge_rooms = false;
		size_t merge_index = 0;

		// merge a room with one neighbor that has max. 2 neighbors and sufficient wall ratio (connect parts inside a room)
		const int max_border_neighbor_id = current_room.getNeighborWithLargestCommonBorder();
		if ((current_room.getNeighborCount()==1 || current_room.getPerimeterRatioOfXLargestRooms(1)>0.98) && current_room.getWallToPerimeterRatio() > 0.5 &&
			(double)current_room.getNeighborStatistics()[max_border_neighbor_id]/current_room.getPerimeter() > 0.15) //biggest neighbor's common border larger than 0.15
		{
			// merge with that neighbor that shares the most neighboring pixels
			merge_rooms = determineRoomIndexFromRoomID(rooms, max_border_neighbor_id, merge_index);
			if (rooms[merge_index].getNeighborCount() > 2 && rooms[merge_index].getPerimeterRatioOfXLargestRooms(2)<0.95) // || rooms[merge_index].getWallToPerimeterRatio() < 0.4)
				merge_rooms = false;
		}

		if (merge_rooms == true)
		{
			//std::cout << "merge " << current_room.getCenter() << ", id=" << current_room.getID() << " into " << rooms[merge_index].getCenter() << ", id=" << rooms[merge_index].getID() << std::endl;
			mergeRoomPair(rooms, merge_index, current_room_index, map_to_merge_rooms, map_resolution_from_subscription);
			current_room_index = 0;
		}
		else
			current_room_index++;
	}
	if (display_map == true)
		cv::imshow("c", map_to_merge_rooms);
draw_segmented_map(map_to_merge_rooms, rooms, "c.png");

	// d) merge rooms that share a significant part of their perimeter
	for (int current_room_index = 0; current_room_index < rooms.size(); )
	{
		Room& current_room = rooms[current_room_index];
		bool merge_rooms = false;
		size_t merge_index = 0;

		std::map< int,int,std::greater<int> > neighbor_room_statistics_inverse;	// common border length, room_id
		current_room.getNeighborStatisticsInverse(neighbor_room_statistics_inverse);
//		std::vector<int>& neighbor_ids = current_room.getNeighborIDs();
//		for (size_t n=0; n<neighbor_ids.size(); ++n)
		for (std::map< int,int,std::greater<int> >::iterator it=neighbor_room_statistics_inverse.begin(); it!=neighbor_room_statistics_inverse.end(); ++it)
		{
			if (it->second==0)
				continue;		// skip wall

			const double neighbor_border_ratio = (double)current_room.getNeighborStatistics()[it->second]/current_room.getPerimeter();
			if (neighbor_border_ratio > 0.2 || (neighbor_border_ratio > 0.1 && current_room.getWallToPerimeterRatio() > (1-2*neighbor_border_ratio-0.05) && current_room.getWallToPerimeterRatio() < (1-neighbor_border_ratio)))
			{
				// merge with that neighbor that shares the most neighboring pixels
				merge_rooms = determineRoomIndexFromRoomID(rooms, it->second, merge_index);
				if ((double)rooms[merge_index].getNeighborStatistics()[current_room.getID()]/rooms[merge_index].getPerimeter() <= 0.1)
					merge_rooms = false;
				if (merge_rooms == true)
					break;
			}
		}

		if (merge_rooms == true)
		{
			//std::cout << "merge " << current_room.getCenter() << ", id=" << current_room.getID() << " into " << rooms[merge_index].getCenter() << ", id=" << rooms[merge_index].getID() << std::endl;
			mergeRoomPair(rooms, merge_index, current_room_index, map_to_merge_rooms, map_resolution_from_subscription);
			current_room_index = 0;
		}
		else
			current_room_index++;
	}
	if (display_map == true)
		cv::imshow("d", map_to_merge_rooms);
draw_segmented_map(map_to_merge_rooms, rooms, "d.png");

	// e) largest room neighbor touches > 0.5 perimeter (happens often with furniture)
	for (int current_room_index = 0; current_room_index < rooms.size(); )
	{
		Room& current_room = rooms[current_room_index];
		bool merge_rooms = false;
		size_t merge_index = 0;

		const int max_border_neighbor_id = current_room.getNeighborWithLargestCommonBorder();
		if ((double)current_room.getNeighborStatistics()[max_border_neighbor_id]/current_room.getPerimeter() > 0.4)
		{
			// merge with that neighbor that shares the most neighboring pixels
			merge_rooms = determineRoomIndexFromRoomID(rooms, max_border_neighbor_id, merge_index);
		}

		if (merge_rooms == true)
		{
			//std::cout << "merge " << current_room.getCenter() << ", id=" << current_room.getID() << " into " << rooms[merge_index].getCenter() << ", id=" << rooms[merge_index].getID() << std::endl;
			mergeRoomPair(rooms, merge_index, current_room_index, map_to_merge_rooms, map_resolution_from_subscription);
			current_room_index = 0;
		}
		else
			current_room_index++;
	}
draw_segmented_map(map_to_merge_rooms, rooms, "e.png");

// f) merge the area near the boundary
	for (int current_room_index = 0; current_room_index < rooms.size(); )
	{
		Room& current_room = rooms[current_room_index];
		std::vector<cv::Point> room_members = current_room.getMembers();
		bool merge_rooms = false;
		size_t merge_index = 0;

		const int max_border_neighbor_id = current_room.getNeighborWithLargestCommonBorder();
		for(int i = 0; i < room_members.size(); i ++)
		{
			//avoid the critical line near the boundary of map
			int x = room_members[i].x;
 			int y = room_members[i].y;
			if( (x < 2 || x > (map_to_merge_rooms.cols-3) || y < 2 || y > (map_to_merge_rooms.rows-3)) && ( current_room.getArea() < 5) )
			{
				merge_rooms = determineRoomIndexFromRoomID(rooms, max_border_neighbor_id, merge_index);
			}
		}

		if (merge_rooms == true)
		{
			//std::cout << "merge " << current_room.getCenter() << ", id=" << current_room.getID() << " into " << rooms[merge_index].getCenter() << ", id=" << rooms[merge_index].getID() << std::endl;
			mergeRoomPair(rooms, merge_index, current_room_index, map_to_merge_rooms, map_resolution_from_subscription);
			current_room_index = 0;
		}
		else
			current_room_index++;
	}
draw_segmented_map(map_to_merge_rooms, rooms, "f.png");


}

// f) specially desighn for corridor
// 	for (int current_room_index = 0; current_room_index < rooms.size(); )
// 	{
// 		Room& current_room = rooms[current_room_index];
// 		bool merge_rooms = false;
// 		size_t merge_index = 0;

// 		const int max_border_neighbor_id = current_room.getNeighborWithLargestCommonBorder();

// 		std::vector<cv::Point> room_members = current_room.getMembers();
// 		int current_member_x_min, current_member_x_max, current_member_y_min, current_member_y_max;
// 		int split_line_x_min, split_line_x_max, split_line_y_min, split_line_y_max;
// 		split_line_x_min = current_member_x_min = map_to_merge_rooms.cols;
// 		split_line_y_min = current_member_y_min = map_to_merge_rooms.rows;
// 		split_line_y_max = split_line_x_max = current_member_x_max = current_member_y_max = 0;
// 		for(int j = 0; j < room_members.size(); j ++)
// 		{
// 			int x = room_members[j].x;
// 			int y = room_members[j].y;
// 			if(x > current_member_x_max)
// 				current_member_x_max = x;
// 			if(x < current_member_x_min)
// 				current_member_x_min = x;
// 			if(y > current_member_y_max)
// 				current_member_y_max = y;
// 			if(y < current_member_y_min)
// 				current_member_y_min = y;

// 			for (int row_counter = -1; row_counter <= 1; row_counter++)
// 			{
// 				for (int col_counter = -1; col_counter <= 1; col_counter++)
// 				{
// 					const int label = map_to_merge_rooms.at<int>(current_points[current_point].y + row_counter, current_points[current_point].x + col_counter);

// 					// collect neighbor IDs
// 					if (label == max_border_neighbor_id)
// 					{
// 						int x = current_points[current_point].x + col_counter;
// 						int y = current_points[current_point].y + row_counter;
// 						if(x > split_line_x_max)
// 							split_line_x_max = x;
// 						if(x < split_line_x_min)
// 							split_line_x_min = x;
// 						if(y > split_line_y_max)
// 							split_line_y_max = y;
// 						if(y < split_line_y_min)
// 							split_line_y_min = y;
// 					}
// 				}
// 			}
// 		}

// 		if ( )
// 		{
// 			// merge with that neighbor that shares the most neighboring pixels
// 			merge_rooms = determineRoomIndexFromRoomID(rooms, max_border_neighbor_id, merge_index);
// 		}

// 		if (merge_rooms == true)
// 		{
// 			//std::cout << "merge " << current_room.getCenter() << ", id=" << current_room.getID() << " into " << rooms[merge_index].getCenter() << ", id=" << rooms[merge_index].getID() << std::endl;
// 			mergeRoomPair(rooms, merge_index, current_room_index, map_to_merge_rooms, map_resolution_from_subscription);
// 			current_room_index = 0;
// 		}
// 		else
// 			current_room_index++;
// 	}

void AbstractVoronoiSegmentation::draw_segmented_map(const cv::Mat& segmented_map_to_be_draw, std::vector<Room>& rooms, const char *input_name)
{
	// if (segmented_map_to_be_draw.type()!=CV_32SC1)
	// {
	// 	std::cout << "Error: draw_segmented_map: provided image is not of type CV_32SC1." << std::endl;
	// 	return;
	// }

	std::vector < cv::Vec3b > already_used_colors;
	
	cv::Mat drawed_segmented_map = cv::Mat::zeros(segmented_map_to_be_draw.rows,segmented_map_to_be_draw.cols,CV_8UC3);
	imwrite("test_first_draw.png", drawed_segmented_map);



	cv::Vec3b color;
	//for overlap color
	color[0] = 255;
	color[1] = 0;
	color[2] = 0;
already_used_colors.push_back(color);

	color[0] = 255;
	color[1] = 255;
	color[2] = 255;
already_used_colors.push_back(color);
	for (int row = 0; row < segmented_map_to_be_draw.rows; row++)
	{
		for (int column = 1; column < segmented_map_to_be_draw.cols; column++)
		{
			if( segmented_map_to_be_draw.at<int>(row, column) > 0 )
			{
				drawed_segmented_map.at<cv::Vec3b>(row, column) = color;
			}
		}
	}
	imwrite("test_medium_draw.png", drawed_segmented_map);

	for(int i = 0; i < rooms.size(); i ++)
	{
		cv::Vec3b color;
		bool drawn = false;
		
		if(i == (rooms.size()-1) )
		{
			//for overlap color
			color[0] = 255;
			color[1] = 0;
			color[2] = 0;
		}
		else
		{
			int loop_counter = 0;
			do
			{
				loop_counter++;
				color[0] = rand() % 255;
				color[1] = rand() % 255;
				color[2] = rand() % 255;
				if (!contains(already_used_colors, color) || loop_counter > 100)
				{
					drawn = true;
					already_used_colors.push_back(color);
				}
				

			} while (!drawn);
		}
		
		
		
		
		std::vector<cv::Point> room_members = rooms[i].getMembers();
		if(room_members.size() > 10)
			std::cout << i <<" member size: " << room_members.size() <<std::endl;
		for(int j = 0; j < room_members.size(); j ++)
		{
			int x = room_members[j].x;
			int y = room_members[j].y;
			drawed_segmented_map.at<cv::Vec3b>(y, x) = color;
		}
	}

	imwrite(input_name, drawed_segmented_map);
}


void AbstractVoronoiSegmentation::get_segment_points_set(const cv::Mat& segmented_map, std::vector<Room>& rooms, std::vector<std::vector<cv::Point>>& segment_result)
{

	std::vector<int> room_id_sore;
	for (int current_room = 0; current_room < rooms.size(); current_room++)
	{
		
		const int current_room_id = rooms[current_room].getID();
		int neighbor_id_number = rooms[current_room].getNeighborCount();
		std::vector<int> neighborid_store = rooms[current_room].getNeighborIDs();
		room_id_sore.push_back(current_room_id);

		std::vector<cv::Point> split_line_points[neighbor_id_number];		// storage for splitline points
		const std::vector<cv::Point>& current_points = rooms[current_room].getMembers();
		

		for (int current_point = 0; current_point < current_points.size(); current_point++)
		{
			for (int row_counter = -1; row_counter <= 1; row_counter++)
			{
				for (int col_counter = -1; col_counter <= 1; col_counter++)
				{
					int neighbor_y = current_points[current_point].y + row_counter;
					int neighbor_x = current_points[current_point].x + col_counter;
					if((neighbor_x > 1 && neighbor_x < (segmented_map.cols-3) && neighbor_y > 1 && neighbor_y < (segmented_map.rows-3)))
					{
						const int neighbor_point_id = segmented_map.at<int>(neighbor_y, neighbor_x);

						for(int neighbor_id_count = 0; neighbor_id_count < neighbor_id_number; neighbor_id_count ++)
						{
							if (neighbor_point_id == neighborid_store[neighbor_id_count] && !contains(room_id_sore, neighbor_point_id) && neighbor_point_id != 0)
							{
								// neighborhood statistics
								cv::Point neighbor_point(current_points[current_point].x + col_counter, current_points[current_point].y + row_counter);
								if (!contains(split_line_points[neighbor_id_count], neighbor_point))
								{
									split_line_points[neighbor_id_count].push_back(neighbor_point);
								}
							}
						}
					}
					
				}
			}
		}

		for(int neighbor_id_count = 0; neighbor_id_count < neighbor_id_number; neighbor_id_count ++)
		{
			segment_result.push_back(split_line_points[neighbor_id_count]);
		}

	}
}