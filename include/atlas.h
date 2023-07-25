#pragma once
#include <trimesh.h>

#include <opencv2/opencv.hpp>
#include <set>

namespace atlas
{
struct UV
{
	UV(float _u = -1, float _v = -1, int _vertexId = -1, int _regionId = -1)
	    : vertexId(_vertexId)
	    , regionId(_regionId)
	    , u(_u)
	    , v(_v){};
	int vertexId;
	int regionId;
	float u, v;
};
struct Chart
{
	int id = -1;
	int seed;
	trimesh::point normal, centroid;
	float area = 0.0f;
	float boundaryLength = 0.0f;
	std::vector<int> faces;
	std::pair<int, int> searchRegion;
	std::shared_ptr<trimesh::TriMesh> mesh; //mesh->flags存储原始顶点id
	std::vector<trimesh::vec2> uvs;
};
struct ChartOptions
{
	float maxChartArea = 0.0f; // Don't grow charts to be larger than this. 0 means no limit.
	float maxBoundaryLength = 0.0f; // Don't grow charts to have a longer boundary than this. 0 means no limit.

	// Weights determine chart growth. Higher weights mean higher cost for that metric.
	float normalDeviationWeight = 2.0f; // Angle between face and average chart normal.
	float roundnessWeight = 0.01f;
	float straightnessWeight = 6.0f;

	float initMaxCost = 1.f; // If total of all metrics * weights > maxCost, don't grow chart. Lower values result in more charts.
	int maxIterations = 2; // Number of iterations of the chart growing and seeding phases. Higher values result in better charts.
	std::vector<float> iterCostLevel = { 0.5f, 1.f, 1.5f, 2.f, FLT_MAX }; //迭代时的Cost等级约束
};

class Atlas
{
public:
	/// \param mesh[in] 非流形面片、孤立点会被删除
	bool computeChart(trimesh::TriMesh* mesh, ChartOptions option, std::vector<Chart>& charts);

	static bool parameterization(Chart& chart);

	/// \return error count
	static int parameterization(std::vector<Chart>& charts);

	/// \brief 纹理打包
	/// \param imgWidth[out], imgHeight[out]:纹理图像的大小，分别对应uv[0], uv[1].
	static bool packCharts(std::vector<Chart>& charts, int& imgWidth, int& imgHeight);

	/// \brief mesh需要是chart对应的mesh，mesh->face顺序会被重写
	static bool packCharts(std::vector<Chart>& charts,
	                       trimesh::TriMesh* mesh,
	                       std::vector<trimesh::vec2>& uvs,
	                       std::vector<trimesh::TriMesh::Face>& facesUvId,
	                       cv::Mat& img);

	/// \brief mesh需要是chart对应的mesh，mesh->face顺序会被重写
	static bool packCharts(std::vector<Chart>& charts,
	                       trimesh::TriMesh* mesh,
	                       std::vector<UV>& uvs,
	                       std::vector<trimesh::TriMesh::Face>& facesUvId,
	                       cv::Mat& img);

	static bool writeObj(std::string name,
	                     std::vector<trimesh::point>& v,
	                     std::vector<UV>& vt,
	                     std::vector<trimesh::TriMesh::Face>& faceId,
	                     std::vector<trimesh::TriMesh::Face>& faceUvId);

	static bool textureBind(trimesh::TriMesh* mesh1,
	                        std::vector<UV>& uvs,
	                        std::vector<trimesh::TriMesh::Face>& facesUvId1,
	                        trimesh::TriMesh* mesh2,
	                        std::vector<trimesh::TriMesh::Face>& facesUvId2);

	/// \brief 移除非流形面片，孤立点
	static bool meshRepair(trimesh::TriMesh* mesh);

	static bool laplaceSmoother(trimesh::TriMesh* mesh, int iterNum);

private:
	float computeCost(Chart& chart, int faceId);
	float computeBoundaryLength(const Chart& chart, int faceId);
	float computeNormalDeviationMetric(const Chart& chart, int faceId);
	float computeRoundnessMetric(const Chart& chart, float newBoundaryLength, float newChartArea);
	float computeStraightnessMetric(const Chart& chart, int faceId);

	bool updateNormal(Chart& chart);
	void planeFit(const std::vector<trimesh::point>& pointSet, trimesh::point& centroid, trimesh::point& normal);
	bool relocateSeeds(std::vector<Chart>& charts);
	bool resetCharts(std::vector<Chart>& charts);
	bool mergeCharts(std::vector<Chart>& charts);
	bool segmentMeshToChart(trimesh::TriMesh* mesh, std::vector<Chart>& charts);

private:
	ChartOptions option_;
	trimesh::TriMesh* mesh_;
	std::vector<int> faceRegions_;
	std::vector<float> faceAreas_;
	std::vector<trimesh::point> faceNormals_;
};
} // namespace atlas
