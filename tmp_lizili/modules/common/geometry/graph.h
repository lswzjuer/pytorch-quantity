#ifndef MODULES_COMMON_GEOMETRY_GRAPH_H
#define MODULES_COMMON_GEOMETRY_GRAPH_H
#include <vector>
namespace roadstar {
namespace common {
template <typename T>
class Graph {
 public:
  Graph(const int column, const int row);
  void set_grid(const int column, const int row, T value);

 private:
  std::vector<std::vector<T>> grids_;
};  // class Graph
}  // namespace common
}  // namespace roadstar
#endif  // MODULES_COMMON_GEOMETRY_GRAPH
