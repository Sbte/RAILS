#ifndef STLTOOLS_H
#define STLTOOLS_H

#include <utility>
#include <vector>
#include <algorithm>

namespace RAILS
{

static bool eigenvalue_sorter(std::pair<int, double> const &a, std::pair<int, double> const &b)
{
    return std::abs(a.second) > std::abs(b.second);
}

template<class DenseMatrix>
int find_largest_eigenvalues(DenseMatrix const &eigenvalues, std::vector<int> &indices, int N)
{
    std::vector<std::pair<int, double> > index_to_value;
    for (int i = 0; i < eigenvalues.M(); i++)
        index_to_value.push_back(std::pair<int, double>(i, eigenvalues(i)));
    
    std::sort(index_to_value.begin(), index_to_value.end(), eigenvalue_sorter);

    for (int i = 0; i < N; i++)
        indices.push_back(index_to_value[i].first);

    return 0;
}

}

#endif
