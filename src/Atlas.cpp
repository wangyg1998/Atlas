#include "atlas.h"

#include <kdtree.h>
#include <poly2tri.h>
#include <trimesh.h>
#include <trimesh_algo.h>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <queue>
#include <stack>

#include "Bff.h"
#include "ConePlacement.h"
#include "Cutter.h"
#include "Generators.h"
#include "HoleFiller.h"
#include "MeshIO.h"
#include "Octree.hpp"
#include "xatlas.h"

namespace atlas
{
static bool trimeshToBmesh(const trimesh::TriMesh& input, bff::Mesh& output)
{
	bff::PolygonSoup soup;
	soup.positions.resize(input.vertices.size());
	for (int i = 0; i < input.vertices.size(); ++i)
	{
		soup.positions[i] = bff::Vector(input.vertices[i][0], input.vertices[i][1], input.vertices[i][2]);
	}
	soup.indices.resize(input.faces.size() * 3);
	for (int i = 0; i < input.faces.size(); ++i)
	{
		const trimesh::TriMesh::Face& f = input.faces[i];
		soup.indices[i * 3] = f[0];
		soup.indices[i * 3 + 1] = f[1];
		soup.indices[i * 3 + 2] = f[2];
	}
	soup.table.construct(soup.positions.size(), soup.indices);
	std::vector<int> isCuttableEdge(soup.table.getSize(), 1);
	std::string error;
	if (!bff::MeshIO::buildMesh(soup, isCuttableEdge, output, error))
	{
		std::cout << error << std::endl;
		return false;
	}
	return true;
}

static bool normalize(bff::Mesh& mesh)
{
	// compute center of mass
	bff::Vector cm;
	int nVertices = 0;
	for (bff::VertexCIter v = mesh.vertices.begin(); v != mesh.vertices.end(); v++)
	{
		cm += v->position;
		nVertices++;
	}
	cm /= nVertices;

	// translate to origin and determine radius
	double radius = 1e-8;
	for (bff::VertexIter v = mesh.vertices.begin(); v != mesh.vertices.end(); v++)
	{
		v->position -= cm;
		radius = std::max(radius, v->position.norm());
	}

	// rescale to unit radius
	for (bff::VertexIter v = mesh.vertices.begin(); v != mesh.vertices.end(); v++)
	{
		v->position /= radius;
	}

	mesh.radius = radius;
	mesh.cm = cm;
	return true;
}

static bool flatten(const trimesh::TriMesh& trimesh,
                    std::vector<trimesh::vec2>& uvs,
                    bool surfaceIsClosed,
                    int nCones,
                    bool flattenToDisk,
                    bool mapToSphere,
                    bool normalizeUVs)
{
	bff::Mesh mesh;
	if (!trimeshToBmesh(trimesh, mesh))
	{
		return false;
	}
	normalize(mesh);

	//补洞
	if (true)
	{
		int nBoundaries = (int)mesh.boundaries.size();
		if (nBoundaries >= 1)
		{
			// mesh has boundaries
			int eulerPlusBoundaries = mesh.eulerCharacteristic() + nBoundaries;
			//std::cout << "eulerPlusBoundaries: " << eulerPlusBoundaries << std::endl;
			if (eulerPlusBoundaries == 2)
			{
				// fill holes if mesh has more than 1 boundary
				if (nBoundaries > 1)
				{
					if (bff::HoleFiller::fill(mesh))
					{
						// all holes were filled
						surfaceIsClosed = true;
					}
				}
			}
			else
			{
				// mesh probably has holes and handles
				bff::HoleFiller::fill(mesh, true);
				bff::Generators::compute(mesh);
			}
		}
		else if (nBoundaries == 0)
		{
			if (mesh.eulerCharacteristic() == 2)
			{
				// mesh is closed
				surfaceIsClosed = true;
			}
			else
			{
				// mesh has handles
				bff::Generators::compute(mesh);
			}
		}
	}

	//参数化
	bff::BFF bff(mesh);
	if (nCones > 0)
	{
		std::vector<bff::VertexIter> cones;
		bff::DenseMatrix coneAngles(bff.data->iN);
		int S = std::min(nCones, (int)mesh.vertices.size() - bff.data->bN);

		if (bff::ConePlacement::findConesAndPrescribeAngles(S, cones, coneAngles, bff.data, mesh) == bff::ConePlacement::ErrorCode::ok)
		{
			if (!surfaceIsClosed || cones.size() > 0)
			{
				bff::Cutter::cut(cones, mesh);
				if (!bff.flattenWithCones(coneAngles, true))
				{
					return false;
				}
			}
		}
	}
	else
	{
		if (surfaceIsClosed)
		{
			if (mapToSphere)
			{
				bff.mapToSphere();
			}
			else
			{
				std::cout << "Surface is closed. Either specify nCones or mapToSphere." << std::endl;
				return false;
			}
		}
		else
		{
			if (flattenToDisk)
			{
				bff.flattenToDisk();
			}
			else
			{
				bff::DenseMatrix u(bff.data->bN);
				if (!bff.flatten(u, true))
				{
					return false;
				}
			}
		}
	}

	//结果拷贝
	uvs.clear();
	uvs.resize(trimesh.vertices.size());
	for (bff::WedgeIter w = mesh.wedges().begin(); w != mesh.wedges().end(); w++)
	{
		if (w->isReal())
		{
			int index = w->vertex()->index;
			trimesh::vec2 uv(w->uv.x, w->uv.y);
			uvs[index] = uv;
		}
	}
	float scale = mesh.radius;
	for (int i = 0; i < uvs.size(); ++i)
	{
		uvs[i] *= scale;
	}
	return true;
}

namespace
{
using namespace std;
using namespace trimesh;

// An "edge" type and ordering functor
typedef pair<int, int> edge;
class OrderEdge
{
public:
	bool operator()(const edge& e1, const edge& e2) const
	{
		if (e1.first < e2.first)
			return true;
		else if (e1.first > e2.first)
			return false;
		else if (e1.second < e2.second)
			return true;
		else
			return false;
	}
};
typedef set<edge, OrderEdge> edgeset;

// A "hole" type - indices of vertices that make up the hole.
typedef vector<int> hole;

// A triple of vertices, together with a quality metric
struct Triple
{
	int v1, v2, v3;
	float quality;
	Triple(int _v1, int _v2, int _v3, float _q)
	    : v1(_v1)
	    , v2(_v2)
	    , v3(_v3)
	    , quality(_q)
	{
	}
	bool operator<(const Triple& rhs) const
	{
		return quality < rhs.quality;
	}
};

#define NO_FACE -1
struct FaceStruct
{
	int v1, v2, v3;
	int n12, n23, n31;
	FaceStruct(int _v1, int _v2, int _v3, int _n12 = NO_FACE, int _n23 = NO_FACE, int _n31 = NO_FACE)
	    : v1(_v1)
	    , v2(_v2)
	    , v3(_v3)
	    , n12(_n12)
	    , n23(_n23)
	    , n31(_n31)
	{
	}
};

struct VertStruct
{
	point p;
	vec norm;
	float& operator[](int i)
	{
		return p[i];
	}
	const float& operator[](int i) const
	{
		return p[i];
	}
	VertStruct()
	{
	}
	VertStruct(const point& _p, const vec& _norm)
	{
		p[0] = _p[0];
		p[1] = _p[1];
		p[2] = _p[2];
		norm[0] = _norm[0];
		norm[1] = _norm[1];
		norm[2] = _norm[2];
	}
};

// Find all the boundary edges
edgeset* find_boundary_edges(const TriMesh* themesh)
{
	//printf("Finding boundary edges... ");
	fflush(stdout);
	edgeset* edges = new edgeset;

	for (size_t f = 0; f < themesh->faces.size(); f++)
	{
		for (int i = 0; i < 3; i++)
		{
			int v1 = themesh->faces[f][i];
			int v2 = themesh->faces[f][NEXT_MOD3(i)];

			// Opposite-pointing edges cancel each other
			if (!edges->erase(make_pair(v2, v1)))
				edges->insert(make_pair(v1, v2));
		}
	}

	//printf("Done.\n");
	return edges;
}

// Find the initial (before hole-filling) neighbors of all the boundary verts
void find_initial_edge_neighbors(const TriMesh* themesh, const edgeset* edges, map<int, set<int>>& initial_edge_neighbors)
{
	size_t nv = themesh->vertices.size(), nf = themesh->faces.size();
	vector<bool> is_edge(nv);
	for (edgeset::const_iterator i = edges->begin(); i != edges->end(); i++)
	{
		is_edge[i->first] = true;
		is_edge[i->second] = true;
	}

	for (size_t i = 0; i < nf; i++)
	{
		int v1 = themesh->faces[i][0];
		int v2 = themesh->faces[i][1];
		int v3 = themesh->faces[i][2];
		if (is_edge[v1])
		{
			initial_edge_neighbors[v1].insert(v2);
			initial_edge_neighbors[v1].insert(v3);
		}
		if (is_edge[v2])
		{
			initial_edge_neighbors[v1].insert(v3);
			initial_edge_neighbors[v1].insert(v1);
		}
		if (is_edge[v3])
		{
			initial_edge_neighbors[v1].insert(v1);
			initial_edge_neighbors[v1].insert(v2);
		}
	}
}

// Find a list of holes, given the boundary edges
vector<hole>* find_holes(edgeset* edges)
{
	//printf("Finding holes... ");
	fflush(stdout);
	vector<hole>* holelist = new vector<hole>;
	edgeset badedges;

	while (!edges->empty())
	{
		// Find an edge at which to start
		const edge& firstedge = *(edges->begin());
		int firstvert = firstedge.first;
		int lastvert = firstedge.second;
		edges->erase(edges->begin());

		// Add the verts as the first two in a new hole
		holelist->push_back(hole());
		hole& newhole = holelist->back();
		newhole.push_back(firstvert);
		newhole.push_back(lastvert);

		// Follow edges to find the rest of this hole
		while (1)
		{
			// Find an edge that starts at lastvert
			edgeset::iterator ei = edges->upper_bound(make_pair(lastvert, -1));
			if (ei == edges->end() || ei->first != lastvert)
			{
				fprintf(stderr, "\nCouldn't find an edge out of vert %d\n", lastvert);
				exit(1);
			}
			int nextvert = ei->second;
			edges->erase(ei);

			// Are we done?
			if (nextvert == firstvert)
				break;

			// Have we encountered this vertex before in this hole?
			// XXX - linear search.  Yuck.
			hole::iterator hi = find(newhole.begin(), newhole.end(), nextvert);
			if (hi != newhole.end())
			{
				// Assuming everything is OK topologically,
				// this could only have been caused if there
				// was a choice of ways to go the last time
				// we encountered this vertex.  Obviously,
				// we chose the wrong way, so find a different
				// way to go.
				edgeset::iterator nei = edges->upper_bound(make_pair(nextvert, -1));
				if (nei == edges->end() || nei->first != nextvert)
				{
					fprintf(stderr, "\nCouldn't find an edge out of vert %d\n", nextvert);
					exit(1);
				}
				// XXX - for paranoia's sake, we should check
				// that nei->second is not in newhole

				// Put the bad edges into "badedges"
				for (hole::iterator tmp = hi; tmp + 1 != newhole.end(); tmp++)
					badedges.insert(make_pair(*tmp, *(tmp + 1)));
				badedges.insert(make_pair(lastvert, nextvert));
				newhole.erase(hi + 1, newhole.end());

				// Take the new edge, and run with it
				lastvert = nei->second;
				newhole.push_back(lastvert);
				edges->erase(nei);
			}
			else
			{
				// All OK.  Add this vert to the hole and go on
				newhole.push_back(nextvert);
				lastvert = nextvert;
			}
		}
		edges->insert(badedges.begin(), badedges.end());
		badedges.clear();
	}

	//printf("Done.\n");
	return holelist;
}

// Compute a quality metric for a potential triangle with three vertices
inline float quality(const TriMesh* themesh, float meanedgelen, const vector<VertStruct>& newverts, int v1, int v2, int v3, bool hack = false)
{
#define VERT(v) ((size_t(v) < themesh->vertices.size()) ? themesh->vertices[v] : newverts[size_t(v) - themesh->vertices.size()].p)
#define NORM(v) ((size_t(v) < themesh->vertices.size()) ? themesh->normals[v] : newverts[size_t(v) - themesh->vertices.size()].norm)

	if (v1 == v2 || v2 == v3 || v3 == v1)
		return -1000.0f;

	const point& p1 = VERT(v1);
	const point& p2 = VERT(v2);
	const point& p3 = VERT(v3);
	vec side1 = p1 - p2, side2 = p2 - p3, side3 = p3 - p1;

	vec norm = side2 CROSS side1;
	normalize(norm);

	float dot1 = norm DOT NORM(v1);
	float dot2 = norm DOT NORM(v2);
	float dot3 = norm DOT NORM(v3);
	if (dot1 < -0.999f || dot2 < -0.999f || dot3 < -0.999f)
		return -1000.0f;

	float len1 = len(side1);
	float len2 = len(side2);
	float len3 = len(side3);

	float maxedgelen = max(max(len1, len2), len3);
	//float minedgelen = min(min(len1,len2),len3);

	normalize(side1);
	normalize(side2);
	normalize(side3);

	float f = dot1 / (1.0f + dot1) + dot2 / (1.0f + dot2) + dot3 / (1.0f + dot3);

	//float d = meanedgelen/(maxedgelen+meanedgelen);
	float d1 = 1.0f + maxedgelen / meanedgelen;
	//d1 = sqrt(d1);
	//float d = 0;

	float a;
	if (hack)
	{
		//a = 0.1f*sqr(2.0f+Dot(side1, side2));
		a = 2.0f + (side1 DOT side2);
		//a = sqrt(1.0f/(1.0f - a));
	}
	else
	{
		a = sqr(1.0f + (side1 DOT side2)) + sqr(1.0f + (side2 DOT side3)) + sqr(1.0f + (side3 DOT side1));
	}

	return f * sqrt(d1) - a * d1;
}

// Fill the given hole, and add the newly-created triangles to newtris
// XXX - FIXME!  This is O(n^2)
void fill_hole(const TriMesh* themesh,
               float meanedgelen,
               const vector<VertStruct>& newverts,
               map<int, set<int>>& initial_edge_neighbors,
               const hole& thehole,
               vector<FaceStruct>& newtris)
{
	vector<bool> used(thehole.size(), false);

	priority_queue<Triple> q;
	for (size_t i = 0; i < thehole.size(); i++)
	{
		int j = (i + 1) % thehole.size();
		int k = (j + 1) % thehole.size();
		float qual = quality(themesh, meanedgelen, newverts, thehole[i], thehole[j], thehole[k], true);
		if (initial_edge_neighbors[thehole[i]].find(thehole[k]) != initial_edge_neighbors[thehole[i]].end())
			qual = -1000.0f;
		q.push(Triple(i, j, k, qual));
	}

	while (!q.empty())
	{
		// Take the highest-quality triple off the queue
		const Triple next = q.top();
		q.pop();

		// Ignore triangles referencing already-used verts
		if (!used[next.v1] && !used[next.v3])
		{
			used[next.v2] = true;

			// Create the new face and push it onto newtris
			newtris.push_back(FaceStruct(thehole[next.v3], thehole[next.v2], thehole[next.v1]));

			// Find next verts forward and back
			int forw = next.v3;
			do
			{
				forw++;
				forw %= thehole.size();
			} while (used[forw]);
			if (forw == next.v1)
				return;
			int back = next.v1;
			do
			{
				back--;
				if (back < 0)
					back += thehole.size();
			} while (used[back]);

			// Insert potential new triangles
			float q13f = quality(themesh, meanedgelen, newverts, thehole[next.v1], thehole[next.v3], thehole[forw], true);
			if (initial_edge_neighbors[thehole[next.v1]].find(thehole[forw]) != initial_edge_neighbors[thehole[next.v1]].end())
				q13f = -2000.0f;
			float qb13 = quality(themesh, meanedgelen, newverts, thehole[back], thehole[next.v1], thehole[next.v3], true);
			if (initial_edge_neighbors[thehole[back]].find(thehole[next.v3]) != initial_edge_neighbors[thehole[back]].end())
				qb13 = -2000.0f;
			q.push(Triple(next.v1, next.v3, forw, q13f));
			q.push(Triple(back, next.v1, next.v3, qb13));
		}
	}
}

} // namespace

void Atlas::holeFill(trimesh::TriMesh* mesh)
{
	if (mesh == nullptr || mesh->faces.empty())
	{
		return;
	}

	mesh->need_normals();
	edgeset* edges = find_boundary_edges(mesh);
	map<int, set<int>> initial_edge_neighbors;
	find_initial_edge_neighbors(mesh, edges, initial_edge_neighbors);
	vector<hole>* holes = find_holes(edges);
	delete edges;
	float mel = mesh->feature_size();

	int maxHoleSize = 0;
	for (int i = 0; i < holes->size(); ++i)
	{
		maxHoleSize = std::max<int>(maxHoleSize, (*holes)[i].size());
	}

	vector<FaceStruct> newtris;
	vector<VertStruct> newverts;
	for (size_t i = 0; i < holes->size(); i++)
	{
		if (maxHoleSize == (*holes)[i].size())
		{
			continue;
		}
		vector<FaceStruct> tmptris;
		fill_hole(mesh, mel, newverts, initial_edge_neighbors, (*holes)[i], tmptris);
		copy(tmptris.begin(), tmptris.end(), back_inserter(newtris));
	}
	delete holes;
	mesh->normals.clear();

	mesh->vertices.reserve(mesh->vertices.size() + newverts.size());
	for (size_t i = 0; i < newverts.size(); i++)
	{
		mesh->vertices.push_back(newverts[i].p);
	}
	mesh->faces.reserve(mesh->faces.size() + newtris.size());
	for (size_t i = 0; i < newtris.size(); i++)
	{
		mesh->faces.push_back(TriMesh::Face(newtris[i].v1, newtris[i].v2, newtris[i].v3));
	}

	return;
}

bool Atlas::laplaceSmoother(trimesh::TriMesh* mesh, int iterNum)
{
	clock_t time = clock();
	mesh->need_neighbors();
	mesh->need_adjacentfaces();
	mesh->normals.clear();
	mesh->need_normals();
	int vertexNum = mesh->vertices.size();
	int faceNum = mesh->faces.size();
	std::vector<trimesh::point> newPosition(vertexNum);
	std::vector<bool> visited;
	while (iterNum--)
	{
#pragma omp parallel for
		for (int i = 0; i < vertexNum; ++i)
		{
			if (mesh->neighbors[i].size() < 2)
			{
				newPosition[i] = mesh->vertices[i];
				continue;
			}
			newPosition[i].set(0.f);
			for (int ring : mesh->neighbors[i])
			{
				newPosition[i] += mesh->vertices[ring];
			}
			newPosition[i] /= static_cast<float>(mesh->neighbors[i].size());
		}

		std::stack<int> searchStack;
		visited.clear();
		visited.resize(vertexNum, false);
		for (int i = 0; i < vertexNum; ++i)
		{
			if (visited[i])
			{
				continue;
			}
			for (int triid : mesh->adjacentfaces[i])
			{
				trimesh::TriMesh::Face& f = mesh->faces[triid];
				trimesh::point triNorm = trimesh::normalized((newPosition[f[1]] - newPosition[f[0]]).cross(newPosition[f[2]] - newPosition[f[0]]));

				if (triNorm.dot(mesh->normals[i]) < 0.f)
				{
					visited[i] = true;
					newPosition[i] = mesh->vertices[i];
					for (int vid : mesh->neighbors[i])
					{
						if (!visited[vid])
						{
							searchStack.push(vid);
						}
					}
					break;
				}
			}
			while (!searchStack.empty())
			{
				int nowV = searchStack.top();
				searchStack.pop();
				if (visited[nowV])
				{
					continue;
				}
				for (int triid : mesh->adjacentfaces[nowV])
				{
					trimesh::TriMesh::Face& f = mesh->faces[triid];
					trimesh::point triNorm = trimesh::normalized((newPosition[f[1]] - newPosition[f[0]]).cross(newPosition[f[2]] - newPosition[f[0]]));
					if (triNorm.dot(mesh->normals[nowV]) < 0.f)
					{
						newPosition[nowV] = mesh->vertices[nowV];
						visited[nowV] = true;
						for (int vid : mesh->neighbors[nowV])
						{
							if (!visited[vid])
							{
								searchStack.push(vid);
							}
						}
						break;
					}
				}
			}
		}
		mesh->vertices.swap(newPosition);
	}
	mesh->normals.clear();
	std::cout << "smooth time: " << clock() - time << std::endl;
	return true;
}

bool Atlas::meshRepair(trimesh::TriMesh* mesh)
{
	clock_t time = clock();
	mesh->need_adjacentfaces();
	std::vector<bool> rmf(mesh->faces.size(), false);
#pragma omp parallel for
	for (int i = 0; i < mesh->faces.size(); ++i)
	{
		trimesh::TriMesh::Face& f = mesh->faces[i];
		for (int j = 0; j < 3; ++j)
		{
			const int& ind1 = f[j];
			const int& ind2 = f[(j + 1) % 3];
			std::vector<int>& ring_faces = mesh->adjacentfaces[ind1];
			for (const int& f_id : ring_faces)
			{
				if (f_id == i)
				{
					continue;
				}
				trimesh::TriMesh::Face& f2 = mesh->faces[f_id];
				if ((f2[0] == ind1 && f2[1] == ind2) || (f2[1] == ind1 && f2[2] == ind2) || (f2[2] == ind1 && f2[0] == ind2))
				{
					rmf[i] = true;
					rmf[f_id] = true;
					break;
				}
			}
		}
	}
	trimesh::remove_faces(mesh, rmf);
	trimesh::remove_unused_vertices(mesh);
	return true;
}

bool Atlas::computeChart(trimesh::TriMesh* mesh, ChartOptions option, std::vector<Chart>& charts)
{
	clock_t time = clock();
	meshRepair(mesh);
	std::vector<trimesh::point> oldVertex = mesh->vertices;
	laplaceSmoother(mesh, 3);

	//数据更新
	charts.clear();
	mesh_ = mesh;
	option_ = option;
	faceRegions_.clear();
	faceRegions_.resize(mesh_->faces.size(), -1);
	faceNormals_.clear();
	faceNormals_.resize(mesh_->faces.size());
	faceAreas_.clear();
	faceAreas_.resize(mesh_->faces.size());
	mesh_->need_across_edge();
	for (int i = 0; i < mesh_->faces.size(); ++i)
	{
		const trimesh::TriMesh::Face& f = mesh_->faces[i];
		faceNormals_[i] = ((mesh_->vertices[f[1]] - mesh_->vertices[f[0]]).cross(mesh_->vertices[f[2]] - mesh_->vertices[f[0]])) * 0.5f;
		faceAreas_[i] = trimesh::length(faceNormals_[i]);
		trimesh::normalize(faceNormals_[i]);
	}

	//随机种子点初始化
	for (int i = 0; i < mesh_->faces.size(); ++i)
	{
		if (faceRegions_[i] > -1)
		{
			continue;
		}
		Chart chart;
		chart.id = charts.size();
		chart.seed = i;
		chart.normal = faceNormals_[i];
		chart.faces.push_back(i);
		faceRegions_[i] = chart.id;
		chart.area += faceAreas_[i];
		chart.boundaryLength += computeBoundaryLength(chart, i);
		std::pair<int, int> searchRegion(0, chart.faces.size());
		while (searchRegion.first < searchRegion.second)
		{
			if (chart.faces.size() < 100 && chart.faces.size() > 1)
			{
				updateNormal(chart);
			}
			for (int j = searchRegion.first; j < searchRegion.second; ++j)
			{
				for (int ring : mesh_->across_edge[chart.faces[j]])
				{
					if (ring < 0 || faceRegions_[ring] > -1 || computeCost(chart, ring) > option_.initMaxCost)
					{
						continue;
					}
					chart.faces.push_back(ring);
					faceRegions_[ring] = chart.id;
					chart.area += faceAreas_[ring];
					chart.boundaryLength = computeBoundaryLength(chart, ring);
				}
			}
			searchRegion.first = searchRegion.second;
			searchRegion.second = chart.faces.size();
		}

		charts.push_back(chart);
	}

	std::cout << "init charts.size(): " << charts.size() << std::endl;

	//迭代优化
	for (int iterNum = 0; iterNum < option_.maxIterations; ++iterNum)
	{
		if (!relocateSeeds(charts))
		{
			break;
		}
		resetCharts(charts);
		for (float costLimit : option_.iterCostLevel)
		{
			for (Chart& chart : charts)
			{
				chart.searchRegion.first = 0;
				chart.searchRegion.second = chart.faces.size();
			}
			int preLeftFaces = mesh_->faces.size(), leftFaces = mesh_->faces.size();
			do
			{
				preLeftFaces = leftFaces;
				leftFaces = mesh_->faces.size();
				for (Chart& chart : charts)
				{
					if (chart.faces.size() < 100 && chart.faces.size() > 5)
					{
						updateNormal(chart);
					}
					for (int j = chart.searchRegion.first; j < chart.searchRegion.second; ++j)
					{
						for (int ring : mesh_->across_edge[chart.faces[j]])
						{
							if (ring < 0 || faceRegions_[ring] > -1 || computeCost(chart, ring) > costLimit)
							{
								continue;
							}
							chart.faces.push_back(ring);
							faceRegions_[ring] = chart.id;
							chart.area += faceAreas_[ring];
							chart.boundaryLength = computeBoundaryLength(chart, ring);
						}
					}
					chart.searchRegion.first = chart.searchRegion.second;
					chart.searchRegion.second = chart.faces.size();
					leftFaces -= chart.faces.size();
				}
			} while (leftFaces > 0 && leftFaces != preLeftFaces);
		}
		mergeCharts(charts);
	}
	std::cout << "final charts.size(): " << charts.size() << std::endl;

	mesh_->vertices.swap(oldVertex);

	//按三角形的数量降序排列
	if (true)
	{
		for (int i = 0; i < charts.size() - 1; i++)
		{
			for (int j = 0; j < charts.size() - 1 - i; j++)
			{
				if (charts[j].faces.size() < charts[j + 1].faces.size())
				{
					std::swap(charts[j], charts[j + 1]);
				}
			}
		}
		for (int i = 0; i < charts.size(); ++i)
		{
			charts[i].id = i;
			const std::vector<int>& faces = charts[i].faces;
			for (int j = 0; j < faces.size(); ++j)
			{
				faceRegions_[faces[j]] = i;
			}
		}
	}

	//分割
	segmentMeshToChart(mesh_, charts);

	if (true)
	{
		std::shared_ptr<trimesh::TriMesh> debug(new trimesh::TriMesh);
		debug->vertices = mesh_->vertices;
		debug->faces = mesh_->faces;
		debug->need_adjacentfaces();
		debug->colors.resize(debug->vertices.size(), trimesh::Color(1.f));
		for (int i = 0; i < debug->vertices.size(); ++i)
		{
			auto af = debug->adjacentfaces[i];
			for (int j = 1; j < af.size(); ++j)
			{
				if (faceRegions_[af[j]] != faceRegions_[af[0]])
				{
					debug->colors[i] = trimesh::Color(1.f, 0.f, 0.f);
				}
			}
		}
		debug->write("D:\\color.ply");
	}

	std::cout << "computeChart time: " << clock() - time << std::endl;
	return true;
}

bool Atlas::packCharts(std::vector<Chart>& charts, trimesh::TriMesh* mesh, std::vector<UV>& uvs, std::vector<trimesh::TriMesh::Face>& facesUvId, cv::Mat& img)
{
	std::vector<trimesh::vec2> originalUvs;
	if (!packCharts(charts, mesh, originalUvs, facesUvId, img))
	{
		return false;
	}
	uvs.clear();
	uvs.resize(originalUvs.size());
	for (int i = 0; i < uvs.size(); ++i)
	{
		uvs[i].u = originalUvs[i][0];
		uvs[i].v = originalUvs[i][1];
	}
	int startId = 0;
	for (int c = 0; c < charts.size(); ++c)
	{
		int endId = startId + charts[c].uvs.size();
		for (int i = startId; i < endId; ++i)
		{
			uvs[i].regionId = c;
		}
		startId += charts[c].uvs.size();
	}
	for (int i = 0; i < facesUvId.size(); ++i)
	{
		trimesh::TriMesh::Face& f = mesh->faces[i];
		trimesh::TriMesh::Face& fuv = facesUvId[i];
		for (int j = 0; j < 3; ++j)
		{
			uvs[fuv[j]].vertexId = f[j];
		}
	}
	return true;
}

bool Atlas::packCharts(std::vector<Chart>& charts,
                       trimesh::TriMesh* mesh,
                       std::vector<trimesh::vec2>& uvs,
                       std::vector<trimesh::TriMesh::Face>& facesUvId,
                       cv::Mat& img)
{
	int imgWidth = 0, imgHeight = 0;
	//检查数据完整性
	for (int c = 0; c < charts.size(); ++c)
	{
		if (charts[c].uvs.empty() || charts[c].uvs.size() != charts[c].mesh->vertices.size() || charts[c].uvs.size() != charts[c].mesh->flags.size())
		{
			return false;
		}
	}
	if (charts.empty() || !packCharts(charts, imgWidth, imgHeight))
	{
		return false;
	}

	//获取偏移量
	uvs.clear();
	facesUvId.clear();
	mesh->faces.clear();
	uvs.reserve(mesh->vertices.size() * 2);
	facesUvId.reserve(mesh->vertices.size() * 3);
	mesh->faces.reserve(mesh->vertices.size() * 3);
	std::vector<std::pair<int, int>> vfOffset;
	for (int c = 0; c < charts.size(); ++c)
	{
		vfOffset.push_back(std::make_pair(uvs.size(), facesUvId.size()));
		uvs.insert(uvs.end(), charts[c].uvs.begin(), charts[c].uvs.end());
		facesUvId.resize(facesUvId.size() + charts[c].mesh->faces.size());
		mesh->faces.resize(mesh->faces.size() + charts[c].mesh->faces.size());
	}

	//整合
	for (int c = 0; c < charts.size(); ++c)
	{
		Chart& chart = charts[c];
		std::vector<unsigned>& vmap = charts[c].mesh->flags;
		for (int i = 0; i < chart.mesh->faces.size(); ++i)
		{
			const trimesh::TriMesh::Face& f = chart.mesh->faces[i];
			mesh->faces[i + vfOffset[c].second] = trimesh::TriMesh::Face(vmap[f[0]], vmap[f[1]], vmap[f[2]]);
			facesUvId[i + vfOffset[c].second] = trimesh::TriMesh::Face(f[0] + vfOffset[c].first, f[1] + vfOffset[c].first, f[2] + vfOffset[c].first);
		}
	}

	//归一化
	if (true)
	{
		trimesh::vec2 minValue(FLT_MAX), maxValue(-FLT_MAX);
		for (int i = 0; i < uvs.size(); ++i)
		{
			minValue[0] = std::min(minValue[0], uvs[i][0]);
			minValue[1] = std::min(minValue[1], uvs[i][1]);
			maxValue[0] = std::max(maxValue[0], uvs[i][0]);
			maxValue[1] = std::max(maxValue[1], uvs[i][1]);
		}
		//留2个像素的边界
		imgWidth = std::floor((maxValue.x - minValue.x) + 5.f);
		imgHeight = std::floor((maxValue.y - minValue.y) + 5.f);
		trimesh::vec2 scale(imgWidth - 0, imgHeight - 0);
		for (int i = 0; i < uvs.size(); ++i)
		{
			uvs[i] = uvs[i] - minValue;
			uvs[i][0] = std::floor(uvs[i][0]) + 2.5f;
			uvs[i][1] = std::floor(uvs[i][1]) + 2.5f;
			uvs[i][0] /= scale[0];
			uvs[i][1] /= scale[1];
		}
	}

	/*opencv的原点在左上角
		* o-------------->u
		* |
		* |
		* |
		* |
		* |
		* \/v
		*/

	/*meshlab的纹理图像原点在左下角
		* /\v
		* |
		* |
		* |
		* |
		* |
		* o-------------->u
		*/

	//写图像
	img = cv::Mat::zeros(imgHeight, imgWidth, CV_8UC3);
	trimesh::vec2 scale(img.rows - 0, img.cols - 0);
	std::vector<std::vector<int>> filled(img.rows, std::vector<int>(img.cols, -10));
	if (mesh->colors.size() == mesh->vertices.size())
	{
		for (int i = 0; i < facesUvId.size(); ++i)
		{
			for (int j = 0; j < 3; ++j)
			{
				int vertexId = mesh->faces[i][j];
				int uvId = facesUvId[i][j];
				int row = uvs[uvId][1] * scale[0];
				int col = uvs[uvId][0] * scale[1];
				row = (img.rows - 1) - row; //原点矫正
				img.at<cv::Vec3b>(row, col)[0] = mesh->colors[vertexId][2] * 255;
				img.at<cv::Vec3b>(row, col)[1] = mesh->colors[vertexId][1] * 255;
				img.at<cv::Vec3b>(row, col)[2] = mesh->colors[vertexId][0] * 255;
				filled[row][col] = 1;
			}
		}
		//todo: 间隙填充
		if (true)
		{
			std::vector<std::pair<int, int>> candidate;
			candidate.reserve(img.rows * img.cols);
			for (int i = 1; i < img.rows - 1; ++i)
			{
				for (int j = 1; j < img.cols - 1; ++j)
				{
					if (filled[i][j] < 0 && (filled[i - 1][j] > 0 || filled[i + 1][j] > 0 || filled[i][j - 1] > 0 || filled[i][j + 1] > 0))
					{
						candidate.push_back(std::make_pair(i, j));
						filled[i][j] = -1;
					}
				}
			}
			std::pair<int, int> searchRegion(0, candidate.size());
			while (searchRegion.first < searchRegion.second)
			{
				for (int i = searchRegion.first; i < searchRegion.second; ++i)
				{
					int row = candidate[i].first, col = candidate[i].second, count = 0;
					trimesh::ivec3 color(0);
					if (filled[row - 1][col] > 0)
					{
						count++;
						color[0] += img.at<cv::Vec3b>(row - 1, col)[0];
						color[1] += img.at<cv::Vec3b>(row - 1, col)[1];
						color[2] += img.at<cv::Vec3b>(row - 1, col)[2];
					}
					if (filled[row + 1][col] > 0)
					{
						count++;
						color[0] += img.at<cv::Vec3b>(row + 1, col)[0];
						color[1] += img.at<cv::Vec3b>(row + 1, col)[1];
						color[2] += img.at<cv::Vec3b>(row + 1, col)[2];
					}
					if (filled[row][col - 1] > 0)
					{
						count++;
						color[0] += img.at<cv::Vec3b>(row, col - 1)[0];
						color[1] += img.at<cv::Vec3b>(row, col - 1)[1];
						color[2] += img.at<cv::Vec3b>(row, col - 1)[2];
					}
					if (filled[row][col + 1] > 0)
					{
						count++;
						color[0] += img.at<cv::Vec3b>(row, col + 1)[0];
						color[1] += img.at<cv::Vec3b>(row, col + 1)[1];
						color[2] += img.at<cv::Vec3b>(row, col + 1)[2];
					}
					color /= static_cast<float>(count);
					img.at<cv::Vec3b>(row, col)[0] = color[0];
					img.at<cv::Vec3b>(row, col)[1] = color[1];
					img.at<cv::Vec3b>(row, col)[2] = color[2];
				}
				for (int i = searchRegion.first; i < searchRegion.second; ++i)
				{
					filled[candidate[i].first][candidate[i].second] = 1;
				}
				for (int i = searchRegion.first; i < searchRegion.second; ++i)
				{
					int row = candidate[i].first, col = candidate[i].second;
					if (row > 1 && row < img.rows - 2 && col > 1 && col < img.cols - 2)
					{
						if (filled[row - 1][col] < -1)
						{
							candidate.push_back(std::make_pair(row - 1, col));
							filled[row - 1][col] = -1;
						}
						if (filled[row + 1][col] < -1)
						{
							candidate.push_back(std::make_pair(row + 1, col));
							filled[row + 1][col] = -1;
						}
						if (filled[row][col - 1] < -1)
						{
							candidate.push_back(std::make_pair(row, col - 1));
							filled[row][col - 1] = -1;
						}
						if (filled[row][col + 1] < -1)
						{
							candidate.push_back(std::make_pair(row, col + 1));
							filled[row][col + 1] = -1;
						}
					}
				}
				searchRegion.first = searchRegion.second;
				searchRegion.second = candidate.size();
			}
		}
	}
	else
	{
		for (int i = 0; i < uvs.size(); ++i)
		{
			int col = uvs[i][0] * scale[1];
			int row = uvs[i][1] * scale[0];
			row = (img.rows - 1) - row;
			img.at<cv::Vec3b>(row, col)[0] = 255;
			img.at<cv::Vec3b>(row, col)[1] = 255;
			img.at<cv::Vec3b>(row, col)[2] = 255;
		}
	}
	return true;
}

struct TrianglePatch
{
	std::vector<int> edgePt[3];
	std::vector<int> innerPt;
	bool valid;
	TrianglePatch()
	    : valid(false){};
};

/// \brief 面片重新三角化
static bool reTriangulation(trimesh::TriMesh* mesh, std::vector<TrianglePatch>& faceVertexPair)
{
	for (int k = 0; k < faceVertexPair.size(); ++k)
	{
		if (faceVertexPair[k].innerPt.empty() && faceVertexPair[k].edgePt[0].empty() && faceVertexPair[k].edgePt[1].empty() &&
		    faceVertexPair[k].edgePt[2].empty())
		{
			continue;
		}
		faceVertexPair[k].valid = true;
		trimesh::TriMesh::Face& f = mesh->faces[k];
		std::shared_ptr<trimesh::TriMesh> cache(new trimesh::TriMesh);
		for (int i = 0; i < 3; ++i)
		{
			cache->vertices.push_back(mesh->vertices[f[i]]);
			cache->flags.push_back(f[i]);
			if (faceVertexPair[k].edgePt[i].size() <= 1)
			{
				for (int vid : faceVertexPair[k].edgePt[i])
				{
					cache->vertices.push_back(mesh->vertices[vid]);
					cache->flags.push_back(vid);
				}
			}
			else //按参数进行排序
			{
				trimesh::point origin = mesh->vertices[f[i]], dest = mesh->vertices[f[(i + 1) % 3]];
				std::vector<std::pair<float, int>> vList;
				for (int vid : faceVertexPair[k].edgePt[i])
				{
					vList.push_back(std::make_pair(0, vid));
					vList.back().first = trimesh::len(mesh->vertices[vid] - origin) / trimesh::len(dest - origin);
				}
				std::sort(vList.begin(), vList.end());
				for (std::pair<float, int> v : vList)
				{
					cache->vertices.push_back(mesh->vertices[v.second]);
					cache->flags.push_back(v.second);
				}
			}
		}
		for (int vid : faceVertexPair[k].innerPt)
		{
			cache->vertices.push_back(mesh->vertices[vid]);
			cache->flags.push_back(vid);
		}
		//投影至二维
		trimesh::point zAxis = trimesh::normalized((mesh->vertices[f[1]] - mesh->vertices[f[0]]).cross(mesh->vertices[f[2]] - mesh->vertices[f[0]]));
		trimesh::point xAxis = trimesh::normalized(mesh->vertices[f[1]] - mesh->vertices[f[0]]);
		trimesh::point yAxis = trimesh::normalized(zAxis.cross(xAxis));
		trimesh::xform xf;
		xf(0, 0) = xAxis[0];
		xf(0, 1) = xAxis[1];
		xf(0, 2) = xAxis[2];
		xf(1, 0) = yAxis[0];
		xf(1, 1) = yAxis[1];
		xf(1, 2) = yAxis[2];
		xf(2, 0) = zAxis[0];
		xf(2, 1) = zAxis[1];
		xf(2, 2) = zAxis[2];
		trimesh::apply_xform(cache.get(), xf);
		std::vector<p2t::Point> pts2d;
		for (int i = 0; i < cache->vertices.size(); ++i)
		{
			pts2d.push_back(p2t::Point(cache->vertices[i].x, cache->vertices[i].y));
		}
		//三角化
		std::vector<p2t::Point*> polyline;
		for (int i = 0; i < cache->vertices.size() - faceVertexPair[k].innerPt.size(); ++i)
		{
			polyline.push_back(&pts2d[i]);
		}
		p2t::CDT cdt(polyline);
		for (int i = cache->vertices.size() - faceVertexPair[k].innerPt.size(); i < pts2d.size(); ++i)
		{
			cdt.AddPoint(&pts2d[i]);
		}
		cdt.Triangulate();
		std::vector<p2t::Triangle*> triangles = cdt.GetTriangles();
		for (p2t::Triangle* triangle : triangles)
		{
			trimesh::TriMesh::Face f;
			for (int i = 0; i < 3; ++i)
			{
				p2t::Point* p = triangle->GetPoint(i);
				for (int j = 0; j < pts2d.size(); ++j)
				{
					if (p == (&pts2d[j]))
					{
						f[i] = j;
						break;
					}
				}
			}
			cache->faces.push_back(f);
		}
		//合并
		for (int i = 0; i < cache->faces.size(); ++i)
		{
			trimesh::TriMesh::Face f = cache->faces[i];
			bool valid = true;
			for (int j = 0; j < 3; ++j)
			{
				f[j] = cache->flags[f[j]];
				if (f[j] < 0 || f[j] >= mesh->vertices.size())
				{
					valid = false;
				}
			}
			if (valid)
			{
				mesh->faces.push_back(f);
			}
		}
	}

	std::vector<bool> rmf(mesh->faces.size(), false);
	for (int k = 0; k < faceVertexPair.size(); ++k)
	{
		if (faceVertexPair[k].valid)
		{
			rmf[k] = true;
		}
	}
	trimesh::remove_faces(mesh, rmf);
	mesh->neighbors.clear();
	mesh->adjacentfaces.clear();
	mesh->flags.clear();
	return true;
}

/// \brief 点到线段的最近点
static trimesh::point closetPoint2Segment(const trimesh::point& queryPoint, const trimesh::point& v0, const trimesh::point& v1)
{
	trimesh::point vec = v1 - v0;
	float t = (queryPoint - v0).dot(vec) / vec.dot(vec);
	t = std::min(std::max(t, 0.f), 1.f);
	return v0 + vec * t;
}

/// \brief 点是否在三角形内
static bool pointInTriangle(const trimesh::point& queryPoint, const trimesh::point& v0, const trimesh::point& v1, const trimesh::point& v2)
{
	trimesh::point a = v0 - queryPoint;
	trimesh::point b = v1 - queryPoint;
	trimesh::point c = v2 - queryPoint;
	trimesh::point normPBC = b.cross(c);
	trimesh::point normPCA = c.cross(a);
	trimesh::point normPAB = a.cross(b);
	if (normPBC.dot(normPCA) < 0.0f)
	{
		return false;
	}
	else if (normPBC.dot(normPAB) < 0.0f)
	{
		return false;
	}
	return true;
}

/// \brief 点到三角形的最小距离及对应的最近点
static std::pair<double, bool> computePoint2TriangleMinDistance(const trimesh::point& queryPoint,
                                                                const trimesh::point& v0,
                                                                const trimesh::point& v1,
                                                                const trimesh::point& v2,
                                                                trimesh::point& closetPoint)
{
	std::pair<double, bool> result(FLT_MAX, false);
	trimesh::point planeNormal = trimesh::normalized((v1 - v0).cross(v2 - v0));
	double distance = planeNormal.dot(queryPoint) - planeNormal.dot(v0);
	closetPoint = queryPoint - planeNormal * distance;

	if (pointInTriangle(closetPoint, v0, v1, v2))
	{
		result.first = distance;
		result.second = true;
	}
	else
	{
		trimesh::point c0 = closetPoint2Segment(closetPoint, v0, v1);
		trimesh::point c1 = closetPoint2Segment(closetPoint, v1, v2);
		trimesh::point c2 = closetPoint2Segment(closetPoint, v2, v0);
		double d0 = trimesh::dist(queryPoint, c0);
		double d1 = trimesh::dist(queryPoint, c1);
		double d2 = trimesh::dist(queryPoint, c2);
		if (d0 < d1 && d0 < d2)
		{
			closetPoint = c0;
		}
		if (d1 < d0 && d1 < d2)
		{
			closetPoint = c1;
		}
		if (d2 < d0 && d2 < d1)
		{
			closetPoint = c2;
		}
		result.first = std::min(std::min(d0, d1), d2);
	}
	result.first = std::abs(result.first);
	return result;
}

/// \brief 点到网格的最近点
static bool getNearestPoint(trimesh::point queryPoint, trimesh::TriMesh* mesh, unibn::Octree<trimesh::point>* ocTree, int& triId, trimesh::point& closetPoint)
{
	double minDist = FLT_MAX;
	std::pair<int, trimesh::point> cache; //面索引及最近点

	int nearestV = ocTree->findNeighbor(queryPoint);
	std::set<int> triSet;
	for (int ring : mesh->neighbors[nearestV])
	{
		for (int ring2 : mesh->neighbors[ring])
		{
			triSet.insert(mesh->adjacentfaces[ring2].begin(), mesh->adjacentfaces[ring2].end());
		}
	}
	for (int tri : triSet)
	{
		trimesh::TriMesh::Face& f = mesh->faces[tri];
		std::pair<double, bool> result;
		result = computePoint2TriangleMinDistance(queryPoint, mesh->vertices[f[0]], mesh->vertices[f[1]], mesh->vertices[f[2]], closetPoint);
		if (result.second)
		{
			triId = tri;
			return true;
		}
		else
		{
			if (result.first < minDist)
			{
				minDist = result.first;
				cache.first = triId;
				cache.second = closetPoint;
			}
		}
	}
	triId = cache.first;
	closetPoint = cache.second;
	return false;
}

/// \brief 获取点到边界的距离并保存至flags
static void getLevelInfo(trimesh::TriMesh* mesh, const std::vector<std::vector<int>>& vertices_uv)
{
	if (mesh->vertices.size() != vertices_uv.size())
	{
		return;
	}
	mesh->need_neighbors();
	mesh->flags.resize(mesh->vertices.size(), 0);
	std::vector<int> vertexList;
	int vertexLevel = 1;
	for (int i = 0; i < mesh->vertices.size(); ++i)
	{
		if (vertices_uv[i].size() > 1)
		{
			vertexList.push_back(i);
			mesh->flags[i] = vertexLevel;
		}
	}
	std::pair<int, int> searchRegion(0, vertexList.size());
	while (searchRegion.second > searchRegion.first)
	{
		vertexLevel++;
		for (int i = searchRegion.first; i < searchRegion.second; ++i)
		{
			for (int ring : mesh->neighbors[vertexList[i]])
			{
				if (mesh->flags[ring] == 0)
				{
					mesh->flags[ring] = vertexLevel;
					vertexList.push_back(ring);
				}
			}
		}
		searchRegion.first = searchRegion.second;
		searchRegion.second = vertexList.size();
	}
	return;
}

#if 0
	bool Atlas::textureBind(trimesh::TriMesh* mesh1, std::vector<trimesh::vec2>& uvs1, std::vector<trimesh::TriMesh::Face>& facesUvId1,
		trimesh::TriMesh* mesh2, std::vector<trimesh::TriMesh::Face>& facesUvId2)
	{
		std::shared_ptr<unibn::Octree<trimesh::point>> ocTree;
		std::vector<std::vector<int>> vertices_uv(mesh1->vertices.size());
		std::vector<TUV> uvs;
		uvs.resize(uvs1.size());
		for (int i = 0; i < uvs1.size(); ++i)
		{
			uvs[i].u = uvs1[i][0];
			uvs[i].v = uvs1[i][1];
		}
		for (int i = 0; i < facesUvId1.size(); ++i)
		{
			const trimesh::TriMesh::Face& f = mesh1->faces[i];
			const trimesh::TriMesh::Face& fuv = facesUvId1[i];
			for (int j = 0; j < 3; ++j)
			{
				uvs[fuv[j]].vertexId = f[j];
				if (std::find(vertices_uv[f[j]].begin(), vertices_uv[f[j]].end(), fuv[j]) == vertices_uv[f[j]].end())
				{
					vertices_uv[f[j]].push_back(fuv[j]);
				}
			}
		}
		ocTree.reset(new unibn::Octree<trimesh::point>);
		ocTree->initialize(mesh1->vertices);
		facesUvId2.clear();
		facesUvId2.resize(mesh2->faces.size());
		for (int i = 0; i < mesh2->faces.size(); ++i)
		{
			trimesh::TriMesh::Face& f = mesh2->faces[i];
			for (int j = 0; j < 3; ++j)
			{
				int vid = ocTree->findNeighbor(mesh2->vertices[f[j]]);
				facesUvId2[i][j] = vertices_uv[vid][0];
			}
		}


		return true;
	}

#else

bool Atlas::textureBind(trimesh::TriMesh* mesh1,
                        std::vector<UV>& uvs,
                        std::vector<trimesh::TriMesh::Face>& facesUvId1,
                        trimesh::TriMesh* mesh2,
                        std::vector<trimesh::TriMesh::Face>& facesUvId2)
{
	clock_t time = clock();
	std::shared_ptr<unibn::Octree<trimesh::point>> ocTree;
	std::vector<std::vector<int>> vertices_uv(mesh1->vertices.size());
	for (int i = 0; i < uvs.size(); ++i)
	{
		int vid = uvs[i].vertexId;
		if (std::find(vertices_uv[vid].begin(), vertices_uv[vid].end(), i) == vertices_uv[vid].end())
		{
			vertices_uv[vid].push_back(i);
		}
	}

	//添加边界点
	std::vector<int> vmap(mesh1->vertices.size(), -1);
	if (true)
	{
		std::vector<TrianglePatch> trianglePatchs(mesh2->faces.size());
		mesh1->need_neighbors();
		mesh1->need_adjacentfaces();
		std::set<std::pair<int, int>> edgeSet; //存储未简化模型的uv边界
		mesh2->need_neighbors();
		mesh2->need_adjacentfaces();
		ocTree.reset(new unibn::Octree<trimesh::point>);
		ocTree->initialize(mesh2->vertices);

		//插入边界点
		for (int i = 0; i < vertices_uv.size(); ++i)
		{
			if (vertices_uv[i].size() < 2)
			{
				continue;
			}
			trimesh::point queryPoint = mesh1->vertices[i];
			trimesh::point closetPoint;
			int triId = -1;
			int vid = ocTree->findNeighbor(queryPoint);
			if (trimesh::dist(mesh1->vertices[i], mesh2->vertices[vid]) < 0.03f) //存在极近点，不再添加新点
			{
				vmap[i] = vid;
				continue;
			}
			if (getNearestPoint(queryPoint, mesh2, ocTree.get(), triId, closetPoint))
			{
				vmap[i] = mesh2->vertices.size();
				trimesh::TriMesh::Face f = mesh2->faces[triId];
				double minDistToSegment = FLT_MAX;
				int edgeId = -1;
				trimesh::point targetPoint;
				for (int j = 0; j < 3; ++j)
				{
					trimesh::point c = closetPoint2Segment(closetPoint, mesh2->vertices[f[j]], mesh2->vertices[f[(j + 1) % 3]]);
					double dist = trimesh::dist(c, closetPoint);
					if (dist < minDistToSegment)
					{
						minDistToSegment = dist;
						edgeId = j;
						targetPoint = c;
					}
				}
				if (minDistToSegment < 0.05f)
				{
					closetPoint = targetPoint;
				}
				if (trimesh::dist(closetPoint, mesh2->vertices[vid]) < 0.03f) //存在极近点，不再添加新点
				{
					vmap[i] = vid;
					continue;
				}
				if (minDistToSegment < 0.05f)
				{
					trianglePatchs[triId].edgePt[edgeId].push_back(mesh2->vertices.size());
					trianglePatchs[triId].valid = true;
					for (int ring : mesh2->adjacentfaces[f[edgeId]])
					{
						trimesh::TriMesh::Face f2 = mesh2->faces[ring];
						for (int j = 0; j < 3; ++j)
						{
							if (f2[j] == f[(edgeId + 1) % 3] && f2[(j + 1) % 3] == f[edgeId])
							{
								trianglePatchs[ring].edgePt[j].push_back(mesh2->vertices.size());
								trianglePatchs[ring].valid = true;
								break;
							}
						}
					}
				}
				else
				{
					trianglePatchs[triId].innerPt.push_back(mesh2->vertices.size());
					trianglePatchs[triId].valid = true;
				}
				mesh2->vertices.push_back(closetPoint);
			}
			else
			{
				if (!mesh1->is_bdy(i))
				{
					//std::cout << "not find, " << trimesh::dist(closetPoint, queryPoint) << " ; ";
				}
			}
		}
		reTriangulation(mesh2, trianglePatchs);

		//拓扑修正
		mesh2->need_neighbors();
		mesh2->need_adjacentfaces();
		//查找边界边
		for (int i = 0; i < mesh1->vertices.size(); ++i)
		{
			if (vertices_uv[i].size() < 2)
			{
				continue;
			}
			for (int ring : mesh1->neighbors[i])
			{
				if (vertices_uv[ring].size() < 2)
				{
					continue;
				}
				if (edgeSet.find(std::make_pair(ring, i)) == edgeSet.end())
				{
					edgeSet.insert(std::make_pair(i, ring));
				}
			}
		}
		//std::cout << "all edge num: " << edgeSet.size() << std::endl;

		int invalidEdgeCount = 0;
		//查找需要折叠的边
		for (auto iter = edgeSet.begin(); iter != edgeSet.end(); ++iter)
		{
			if (vmap[iter->first] < 0 || vmap[iter->second] < 0)
			{
				invalidEdgeCount += 1;
				continue;
			}
			std::pair<int, int> edge(vmap[iter->first], vmap[iter->second]);
			bool connected = false;
			for (int ring : mesh2->neighbors[edge.first])
			{
				if (ring == edge.second)
				{
					connected = true;
					break;
				}
			}
			if (connected)
			{
				continue;
			}
			int v1, v2, v3, v4 = edge.second, tri1 = -1, tri2 = -1;
			std::set<int> vset;
			vset.insert(mesh2->neighbors[edge.second].begin(), mesh2->neighbors[edge.second].end());
			for (int ring : mesh2->adjacentfaces[edge.first])
			{
				trimesh::TriMesh::Face f = mesh2->faces[ring];
				for (int j = 0; j < 3; ++j)
				{
					if (f[j] != edge.first)
					{
						continue;
					}
					if (vset.find(f[(j + 1) % 3]) == vset.end() || vset.find(f[(j + 2) % 3]) == vset.end())
					{
						continue;
					}
					v1 = f[j], v2 = f[(j + 1) % 3], v3 = f[(j + 2) % 3], tri1 = ring;
					break;
				}
				if (tri1 >= 0)
				{
					break;
				}
			}
			for (int ring : mesh2->adjacentfaces[edge.second])
			{
				trimesh::TriMesh::Face f = mesh2->faces[ring];
				for (int j = 0; j < 3; ++j)
				{
					if (f[j] != edge.second)
					{
						continue;
					}
					if (f[(j + 1) % 3] == v3 && f[(j + 2) % 3] == v2)
					{
						tri2 = ring;
						break;
					}
				}
			}
			if (tri1 >= 0 && tri2 >= 0)
			{
				mesh2->faces[tri1] = trimesh::TriMesh::Face(v1, v2, v4);
				mesh2->faces[tri2] = trimesh::TriMesh::Face(v1, v4, v3);
			}
			else
			{
				invalidEdgeCount += 1;
			}
		}
		mesh2->neighbors.clear();
		mesh2->adjacentfaces.clear();
		mesh2->across_edge.clear();
		//std::cout << "invalidEdgeCount: " << invalidEdgeCount << std::endl;
	}

	//纹理坐标绑定
	ocTree->clear();
	ocTree->initialize(mesh1->vertices);
	facesUvId2.clear();
	facesUvId2.resize(mesh2->faces.size());
	int flawCount = 0;
	getLevelInfo(mesh1, vertices_uv);
	for (int i = 0; i < mesh2->faces.size(); ++i)
	{
		const trimesh::TriMesh::Face& f = mesh2->faces[i];
		trimesh::TriMesh::Face& t_faces_uv = facesUvId2[i];
		trimesh::TriMesh::Face oldFace;
		std::multiset<int> regionSet;
		for (int j = 0; j < 3; ++j)
		{
			int vid = ocTree->findNeighbor(mesh2->vertices[f[j]]);
			oldFace[j] = vid;
			t_faces_uv[j] = vertices_uv[vid][0];
			for (int uvid : vertices_uv[vid])
			{
				regionSet.insert(uvs[uvid].regionId);
			}
		}
		if (uvs[t_faces_uv[0]].regionId == uvs[t_faces_uv[1]].regionId && uvs[t_faces_uv[0]].regionId == uvs[t_faces_uv[2]].regionId)
		{
			continue;
		}
		int regionId = -1;
		for (auto iter = regionSet.begin(); iter != regionSet.end(); ++iter)
		{
			if (regionSet.count(*iter) == 3)
			{
				regionId = *iter;
				break;
			}
		}
		if (regionId >= 0)
		{
			for (int j = 0; j < 3; ++j)
			{
				for (int uvid : vertices_uv[oldFace[j]])
				{
					if (uvs[uvid].regionId == regionId)
					{
						t_faces_uv[j] = uvid;
					}
				}
			}
			continue;
		}
		else
		{
			regionId = -1;
			for (int j = 0; j < 3; ++j)
			{
				if (mesh1->flags[oldFace[j]] >= mesh1->flags[oldFace[(j + 1) % 3]] && mesh1->flags[oldFace[j]] >= mesh1->flags[oldFace[(j + 2) % 3]])
				{
					regionId = uvs[vertices_uv[oldFace[j]][0]].regionId;
				}
			}
			for (int j = 0; j < 3; ++j)
			{
				std::vector<uint32_t> indices;
				ocTree->radiusNeighbors(mesh2->vertices[f[j]], 0.3f, indices);

				//冒泡排序
				std::vector<std::pair<int, float>> sortedIndices(indices.size());
				for (int k = 0; k < indices.size(); ++k)
				{
					sortedIndices[k].first = indices[k];
					sortedIndices[k].second = trimesh::dist(mesh2->vertices[f[j]], mesh1->vertices[indices[k]]);
				}
				for (int m = 0; m < sortedIndices.size() - 1; m++)
				{
					for (int n = 0; n < sortedIndices.size() - 1 - m; n++)
					{
						if (sortedIndices[n].second > sortedIndices[n + 1].second)
						{
							std::swap(sortedIndices[n], sortedIndices[n + 1]);
						}
					}
				}
				for (int k = 0; k < indices.size(); ++k)
				{
					indices[k] = sortedIndices[k].first;
				}

				//查找同一个区域内的纹理坐标
				for (int vid : indices)
				{
					for (int uvid : vertices_uv[vid])
					{
						if (uvs[uvid].regionId == regionId)
						{
							t_faces_uv[j] = uvid;
							break;
						}
					}
					if (uvs[t_faces_uv[j]].regionId == regionId)
					{
						break;
					}
				}
			}
			if (uvs[t_faces_uv[0]].regionId == uvs[t_faces_uv[1]].regionId && uvs[t_faces_uv[0]].regionId == uvs[t_faces_uv[2]].regionId)
			{
				continue;
			}
			else
			{
				std::cout << mesh2->faces[i] << ", " << t_faces_uv << ", ";
				for (int j = 0; j < 3; ++j)
				{
					std::cout << mesh1->flags[oldFace[j]] << ", ";
				}
				std::cout << "; ";
				for (int j = 0; j < 3; ++j)
				{
					std::cout << uvs[t_faces_uv[j]].regionId << ", ";
				}
				std::cout << std::endl;
				flawCount += 1; //仍然存在裂痕的面片
				//t_faces_uv[1] = t_faces_uv[0];
				//t_faces_uv[2] = t_faces_uv[0];
			}
		}
	}

	std::cout << "仍然存在裂痕的面片数量: " << flawCount << std::endl;
	std::cout << "textureBind time: " << clock() - time << std::endl;
	return true;
}
#endif

bool Atlas::writeObj(std::string name,
                     std::vector<trimesh::point>& v,
                     std::vector<UV>& vt,
                     std::vector<trimesh::TriMesh::Face>& faceId,
                     std::vector<trimesh::TriMesh::Face>& faceUvId)
{
	std::ofstream out(name + ".obj", std::ios::binary);
	out << "mtllib " + name + ".mtl\n";
	if (!out.is_open() || faceId.size() != faceUvId.size())
	{
		return false;
	}

	for (int i = 0; i < v.size(); i++)
	{
		out << "v " + std::to_string(v[i].x) + " " + std::to_string(v[i].y) + " " + std::to_string(v[i].z) + "\n";
	}
	for (int i = 0; i < vt.size(); i++)
	{
		out << "vt " + std::to_string(vt[i].u) + " " + std::to_string(vt[i].v) + "\n";
	}
	out << "g Group_Global\ns off\nusemtl " + name + "\n";

	for (int i = 0; i < faceId.size(); i++)
	{
		out << "f";
		for (int j = 0; j < 3; j++)
		{
			out << " " + std::to_string(faceId[i][j] + 1) + "/" + std::to_string(faceUvId[i][j] + 1);
		}
		out << "\n";
	}
	out.close();
	out.open(name + ".mtl");
	out << "newmtl " + name + "\n";
	out << "Kd 1.0 1.0 1.0\nKa 0.0 0.0 0.0\nKs 0.0 0.0 0.0\nd 1.0\nNs 0.0\nillum 0\nmap_Kd texture.png\n";
	out << "#EOF";
	out.close();
	return true;
}

bool Atlas::packCharts(std::vector<Chart>& charts, int& imgWidth, int& imgHeight)
{
	clock_t time = clock();
	for (Chart& chart : charts)
	{
		if (chart.uvs.empty() || chart.mesh->faces.empty())
		{
			return false;
		}
	}

	xatlas::Atlas* atlas = xatlas::Create();
	//数据转换
	for (int c = 0; c < charts.size(); ++c)
	{
		Chart& chart = charts[c];
		xatlas::UvMeshDecl meshDecl;
		meshDecl.vertexCount = chart.uvs.size();
		meshDecl.vertexUvData = chart.uvs.data();
		meshDecl.vertexStride = sizeof(float) * 2;
		std::vector<uint32_t> indices(chart.mesh->faces.size() * 3);
		for (int i = 0; i < chart.mesh->faces.size(); ++i)
		{
			indices[i * 3] = chart.mesh->faces[i][0];
			indices[i * 3 + 1] = chart.mesh->faces[i][1];
			indices[i * 3 + 2] = chart.mesh->faces[i][2];
		}
		meshDecl.indexCount = indices.size();
		meshDecl.indexData = indices.data();
		meshDecl.indexFormat = xatlas::IndexFormat::UInt32;
		meshDecl.rotateCharts = true;
		xatlas::AddMeshError::Enum error = xatlas::AddUvMesh_SingleChart(atlas, meshDecl);
		if (error != xatlas::AddMeshError::Success)
		{
			xatlas::Destroy(atlas);
			return false;
		}
	}
	//执行
	xatlas::PackCharts(atlas);
	//复制结果
	imgWidth = atlas->width;
	imgHeight = atlas->height;
	for (int m = 0; m < atlas->meshCount; ++m)
	{
		xatlas::Mesh output = atlas->meshes[m];
		std::vector<trimesh::vec2>& uvs = charts[m].uvs;
		for (int i = 0; i < output.vertexCount; ++i)
		{
			uvs[i][0] = output.vertexArray[i].uv[0];
			uvs[i][1] = output.vertexArray[i].uv[1];
		}
	}

	xatlas::Destroy(atlas);
	std::cout << "packCharts time: " << clock() - time << std::endl;
	return true;
}

bool Atlas::parameterization(Chart& chart)
{
	chart.uvs.clear();
	trimesh::TriMesh* mesh = chart.mesh.get();
	if (!mesh || chart.mesh->faces.empty())
	{
		return false;
	}
	int oldFacesSize = mesh->faces.size();
	holeFill(mesh);
	if (!flatten(*mesh, chart.uvs, false, 0, false, false, false))
	{
		return false;
	}
	mesh->faces.resize(oldFacesSize);
	return true;
}

int Atlas::parameterization(std::vector<Chart>& charts)
{
	clock_t time = clock();
	std::atomic_int errorCount = 0;
#pragma omp parallel for schedule(dynamic)
	for (int c = 0; c < charts.size(); ++c)
	{
		if (!parameterization(charts[c]))
		{
			errorCount++;
		}
	}
	std::cout << "parameterization time: " << clock() - time << std::endl;
	return errorCount;
}

bool Atlas::segmentMeshToChart(trimesh::TriMesh* mesh, std::vector<Chart>& charts)
{
	for (int c = 0; c < charts.size(); ++c)
	{
		Chart& chart = charts[c];
		std::vector<int> vmap(mesh->vertices.size(), -1);
		chart.mesh.reset(new trimesh::TriMesh);
		for (int triId : chart.faces)
		{
			trimesh::TriMesh::Face& f = mesh->faces[triId];
			for (int j = 0; j < 3; ++j)
			{
				if (vmap[f[j]] < 0)
				{
					vmap[f[j]] = chart.mesh->vertices.size();
					chart.mesh->vertices.push_back(mesh->vertices[f[j]]);
					chart.mesh->flags.push_back(f[j]);
				}
			}
			chart.mesh->faces.push_back(trimesh::TriMesh::Face(vmap[f[0]], vmap[f[1]], vmap[f[2]]));
		}
	}
	return true;
}

bool Atlas::mergeCharts(std::vector<Chart>& charts)
{
	clock_t time = clock();
	//重新拟合法向
#pragma omp parallel for
	for (int c = 0; c < charts.size(); ++c)
	{
		updateNormal(charts[c]);
	}
	//按法向偏差合并
	for (int c = charts.size() - 1; c > 0; c--)
	{
		Chart& chart = charts[c];
		std::set<int> adjChartId;
		for (int k = 0; k < chart.faces.size(); ++k)
		{
			for (int ring : mesh_->across_edge[chart.faces[k]])
			{
				if (ring > -1 && faceRegions_[ring] != chart.id)
				{
					adjChartId.insert(faceRegions_[ring]);
				}
			}
		}
		std::pair<float, int> bestChart(FLT_MAX, -1);
		for (int chartId : adjChartId)
		{
			if (chartId < 0 || charts[chartId].faces.empty())
			{
				continue;
			}
			float angle = trimesh::angle(chart.normal, charts[chartId].normal);
			if (angle < bestChart.first)
			{
				bestChart.first = angle;
				bestChart.second = chartId;
			}
		}
		//将size过小的也合并
		if (bestChart.second >= 0 && (bestChart.first < (M_PI_180f * 30.f) || chart.faces.size() < (mesh_->faces.size() / 1000)))
		{
			for (int k = 0; k < chart.faces.size(); ++k)
			{
				faceRegions_[chart.faces[k]] = bestChart.second;
			}
			charts[bestChart.second].faces.insert(charts[bestChart.second].faces.end(), chart.faces.begin(), chart.faces.end());
			chart.faces.clear();
		}
	}
	//删除被合并的Chart
	for (auto iter = charts.begin(); iter != charts.end();)
	{
		if (iter->faces.empty())
		{
			iter = charts.erase(iter);
		}
		else
		{
			++iter;
		}
	}
	//重新分配id
	for (int i = 0; i < charts.size(); ++i)
	{
		charts[i].id = i;
	}
	std::cout << "mergeCharts time: " << clock() - time << std::endl;
	return true;
}

bool Atlas::resetCharts(std::vector<Chart>& charts)
{
	faceRegions_.clear();
	faceRegions_.resize(mesh_->faces.size(), -1);
	for (Chart& chart : charts)
	{
		chart.faces.clear();
		chart.faces.push_back(chart.seed);
		chart.area = faceAreas_[chart.seed];
		faceRegions_[chart.seed] = chart.id;
		chart.boundaryLength = computeBoundaryLength(chart, chart.seed);
		chart.searchRegion = std::make_pair(0, chart.faces.size());
		chart.normal = faceNormals_[chart.seed];
	}
	return true;
}

bool Atlas::relocateSeeds(std::vector<Chart>& charts)
{
	clock_t time = clock();
	struct CandidateSeed
	{
		int faceId;
		float cost, dist;
		CandidateSeed(int _faceId, float _cost, float _dist)
		    : faceId(_faceId)
		    , cost(_cost)
		    , dist(_dist)
		{
		}
	};
	std::atomic_bool anyChange = false;
#pragma omp parallel for schedule(dynamic)
	for (int c = 0; c < charts.size(); ++c)
	{
		Chart& chart = charts[c];
		updateNormal(chart);
		std::vector<CandidateSeed> candidates;
		float minCost = FLT_MAX, minDist = FLT_MAX, cost, dist;
		for (int faceId : chart.faces)
		{
			cost = computeNormalDeviationMetric(chart, faceId);
			dist = trimesh::dist(chart.centroid, mesh_->centroid(faceId));
			if (cost < minCost && dist < minDist)
			{
				candidates.clear();
				minCost = cost;
				minDist = dist;
				candidates.push_back(CandidateSeed(faceId, cost, dist));
			}
			else if (cost < minCost)
			{
				minCost = cost;
				candidates.push_back(CandidateSeed(faceId, cost, dist));
			}
			else if (dist < minDist)
			{
				minDist = dist;
				candidates.push_back(CandidateSeed(faceId, cost, dist));
			}
		}
		for (int i = 0; i < candidates.size() - 1; i++)
		{
			for (int j = 0; j < candidates.size() - 1 - i; j++)
			{
				if (candidates[j].cost > candidates[j + 1].cost)
				{
					std::swap(candidates[j], candidates[j + 1]);
				}
			}
		}
		minDist = FLT_MAX;
		int bestSeed = -1;
		for (int i = 0; i < candidates.size() && i < std::max<int>(i < candidates.size() / 10, 10); ++i)
		{
			if (candidates[i].dist < minDist)
			{
				minDist = candidates[i].dist;
				bestSeed = candidates[i].faceId;
			}
		}
		if (bestSeed != chart.seed)
		{
			chart.seed = bestSeed;
			anyChange = true;
		}
	}
	std::cout << "relocateSeeds time: " << clock() - time << std::endl;
	return anyChange;
}

bool Atlas::updateNormal(Chart& chart)
{
	std::vector<trimesh::point> points;
	points.reserve(chart.faces.size() * 3);
	for (int faceId : chart.faces)
	{
		points.push_back(mesh_->vertices[mesh_->faces[faceId][0]]);
		points.push_back(mesh_->vertices[mesh_->faces[faceId][1]]);
		points.push_back(mesh_->vertices[mesh_->faces[faceId][2]]);
	}
	trimesh::point centroid, normal;
	planeFit(points, centroid, normal);
	if (normal.dot(chart.normal) < 0.f)
	{
		chart.normal = -normal;
	}
	else
	{
		chart.normal = normal;
	}
	chart.centroid = centroid;
	return true;
}

//Fit plane: A(x-x0)+B(y-y0)+C(z-z0)=0
void Atlas::planeFit(const std::vector<trimesh::point>& pointSet, trimesh::point& centroid, trimesh::point& normal)
{
	centroid.set(0.f);
	for (const trimesh::point& p : pointSet)
	{
		centroid += p;
	}
	centroid /= static_cast<float>(pointSet.size());
	Eigen::MatrixXf A(pointSet.size(), 3);
	for (size_t i = 0; i < pointSet.size(); ++i)
	{
		trimesh::point p = pointSet[i] - centroid;
		for (int j = 0; j < 3; ++j)
		{
			A(i, j) = p[j];
		}
	}
	Eigen::MatrixXf B = A.transpose() * A;
	Eigen::EigenSolver<Eigen::MatrixXf> es(B);
	const Eigen::VectorXcf& eigenValues = es.eigenvalues();
	const Eigen::MatrixXcf& eigenVectors = es.eigenvectors();
	for (size_t i = 0; i < 3; ++i)
	{
		if (eigenValues[i].real() < eigenValues[(i + 1) % 3].real() && eigenValues[i].real() < eigenValues[(i + 2) % 3].real())
		{
			normal = trimesh::point(eigenVectors(0, i).real(), eigenVectors(1, i).real(), eigenVectors(2, i).real());
		}
	}
	trimesh::normalize(normal);
	return;
}

float Atlas::computeBoundaryLength(const Chart& chart, int faceId)
{
	float boundartLength = chart.boundaryLength;
	const trimesh::TriMesh::Face& f = mesh_->faces[faceId];
	for (int j = 0; j < 3; ++j)
	{
		int adjFace = mesh_->across_edge[faceId][j];
		float dist = trimesh::dist(mesh_->vertices[f[j]], mesh_->vertices[f[NEXT_MOD3(j)]]);
		if (adjFace > -1 && faceRegions_[adjFace] == chart.id)
		{
			boundartLength -= dist;
		}
		else
		{
			boundartLength += dist;
		}
	}
	return boundartLength;
}

float Atlas::computeNormalDeviationMetric(const Chart& chart, int faceId)
{
	return std::min(1.f - chart.normal.dot(faceNormals_[faceId]), 1.f);
}

float Atlas::computeRoundnessMetric(const Chart& chart, float newBoundaryLength, float newChartArea)
{
	const float oldRoundness = trimesh::sqr(chart.boundaryLength) / chart.area;
	const float newRoundness = trimesh::sqr(newBoundaryLength) / newChartArea;
	return 1.0f - oldRoundness / newRoundness;
}

float Atlas::computeStraightnessMetric(const Chart& chart, int faceId)
{
	float l_out = 0.0f;
	float l_in = 0.0f;
	const trimesh::TriMesh::Face& f = mesh_->faces[faceId];
	for (int j = 0; j < 3; ++j)
	{
		float dist = trimesh::dist(mesh_->vertices[f[j]], mesh_->vertices[f[NEXT_MOD3(j)]]);
		int oppositeFace = mesh_->across_edge[faceId][(j + 2) % 3];
		if (oppositeFace < 0)
		{
			l_out += dist;
		}
		else
		{
			if (faceRegions_[oppositeFace] != chart.id)
			{
				l_out += dist;
			}
			else
			{
				l_in += dist;
			}
		}
	}
	float ratio = (l_out - l_in) / (l_out + l_in);
	return std::min(ratio, 0.0f);
}

float Atlas::computeCost(Chart& chart, int faceId)
{
	float cost = 0.f;
	float newChartArea = chart.area + faceAreas_[faceId];
	float newBoundaryLength = computeBoundaryLength(chart, faceId);
	if (option_.maxChartArea > 0.f && newChartArea > option_.maxChartArea)
	{
		return FLT_MAX;
	}
	if (option_.maxBoundaryLength > 0.f && newBoundaryLength > option_.maxBoundaryLength)
	{
		return FLT_MAX;
	}
	float normalDeviation = computeNormalDeviationMetric(chart, faceId);
	if (normalDeviation > 0.75f)
	{
		return FLT_MAX;
	}
	cost += option_.normalDeviationWeight * normalDeviation;
	cost += option_.roundnessWeight * computeRoundnessMetric(chart, newBoundaryLength, newChartArea);
	cost += option_.straightnessWeight * computeStraightnessMetric(chart, faceId);

	return cost;
}
} // namespace atlas