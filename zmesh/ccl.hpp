#ifndef _ZMESH_CCL_HXX_
#define _ZMESH_CCL_HXX_

#include <cstdint>
#include <limits>
#include <vector>

namespace zmesh::ccl {

template <typename T>
class DisjointSet {
public:
	std::vector<T> ids;
	size_t length;

	DisjointSet (size_t len) {
		length = len;
		ids.resize(len);
	}

	DisjointSet (const DisjointSet &cpy) {
		length = cpy.length;
		ids.resize(length);

		for (int i = 0; i < length; i++) {
			ids[i] = cpy.ids[i];
		}
	}

	T root (T n) {
		T i = ids[n];
		while (i != ids[i]) {
			ids[i] = ids[ids[i]]; // path compression
			i = ids[i];
		}

		return i;
	}

	bool find (T p, T q) {
		return root(p) == root(q);
	}

	void add(T p) {
		if (p >= length) {
			printf("Connected Components Error: Label %lli cannot be mapped to union-find array of length %lu.\n", static_cast<long long int>(p), length);
			throw std::runtime_error("maximum length exception");
		}

		if (ids[p] == 0) {
			ids[p] = p;
		}
	}

	void unify (T p, T q) {
		if (p == q) {
			return;
		}

		T i = root(p);
		T j = root(q);

		if (i == 0) {
			add(p);
			i = p;
		}

		if (j == 0) {
			add(q);
			j = q;
		}

		ids[i] = j;
	}
};

template <typename T>
std::vector<std::vector<T>>
vertex_connected_components_mask(
	const T* faces,
	const uint64_t num_verts,
	const uint64_t num_faces
) {
	T max_label = 0;

	DisjointSet<T> equivalences(num_verts + 2);

	for (uint64_t i = 0; i < num_faces * 3; i += 3) {
		auto f1 = faces[i+0]+1;
		auto f2 = faces[i+1]+1;
		auto f3 = faces[i+2]+1;

		equivalences.unify(f1, f2);
		equivalences.unify(f2, f3);

		max_label = std::max(max_label, f1);
		max_label = std::max(max_label, f2);
		max_label = std::max(max_label, f3);
	}

	std::vector<T> mask(max_label + 1);
	T next_label = 1;

	for (int64_t i = 1; i <= max_label; i++) {
		auto label = equivalences.root(i);

		if (mask[label] == 0) {
			mask[label] = next_label;
			mask[i] = next_label;
			next_label++;
		}
		else {
			mask[i] = mask[label]; 
		}
	}

	equivalences = DisjointSet<T>(0); // clear memory

	std::vector<std::vector<T>> ccl(next_label - 1);

	if (ccl.size() == 0) {
		return ccl;
	}

	for (uint64_t i = 0; i < num_faces * 3; i++) {
		ccl[mask[faces[i]+1] - 1].push_back(faces[i]);
	}

	return ccl;
}

};

#endif
