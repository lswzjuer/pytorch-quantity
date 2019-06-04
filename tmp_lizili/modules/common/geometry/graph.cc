#include "modules/common/geometry/graph.h"
#include "modules/common/log.h"
namespace roadstar {
namespace common {
template <typename T>
Graph<T>::Graph(const int column, const int row) {
  grids_.resize(column);
  for (auto& grids : grids_) {
    grids.resize(row);
  }
}

template <typename T>
void Graph<T>::set_grid(const int column, const int row, T value) {
  if (column >= grids_.size() || row > grids_[0].size()) {
    AERROR << "index out of range";
    return;
  }
  grids_[column][row] = value;
}

using IntGraph = Graph<int>;
using FloatGraph = Graph<float>;
}  // namespace common
}  // namespace roadstar
