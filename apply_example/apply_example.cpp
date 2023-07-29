#include <trimesh.h>
#include <trimesh_algo.h>

#include <iostream>
#include <opencv2/opencv.hpp>

#include "atlas.h"

int main()
{
	clock_t time = clock();
	trimesh::TriMesh::set_verbose(false);

#if 1
	std::string pathPrefix = "D:\\Algorithms\\Atlas\\apply_example\\";
	std::shared_ptr<trimesh::TriMesh> mesh(trimesh::TriMesh::read(pathPrefix + "input_color_mesh.ply"));

	atlas::Atlas atlas;
	std::vector<atlas::Chart> charts;
	atlas::ChartOptions option;
	atlas.computeChart(mesh.get(), option, charts);

	std::cout << "error count: " << atlas.parameterization(charts) << std::endl;

	std::vector<atlas::UV> uvs;
	std::vector<trimesh::TriMesh::Face> facesUvId;
	cv::Mat img;
	atlas.packCharts(charts, mesh.get(), uvs, facesUvId, img);
	cv::imwrite(pathPrefix + "texture.png", img);
	atlas.writeObj(pathPrefix + "origin", mesh->vertices, uvs, mesh->faces, facesUvId);

	std::shared_ptr<trimesh::TriMesh> mesh2(trimesh::TriMesh::read(pathPrefix + "input_simed.ply"));
	std::vector<trimesh::TriMesh::Face> facesUvId2;
	atlas::Atlas::textureBind(mesh.get(), uvs, facesUvId, mesh2.get(), facesUvId2);
	atlas.writeObj(pathPrefix + "simed", mesh2->vertices, uvs, mesh2->faces, facesUvId2);

#endif

	std::cout << "main time: " << clock() - time << std::endl;
	system("pause");
	return 0;
}
