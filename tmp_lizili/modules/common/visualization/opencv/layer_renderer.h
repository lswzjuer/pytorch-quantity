#ifndef MODULES_COMMON_VISUALIZATION_OPENCV_LAYER_RENDERER_H_
#define MODULES_COMMON_VISUALIZATION_OPENCV_LAYER_RENDERER_H_

#include <memory>
#include <utility>

namespace cv {
class Mat;
}  // namespace cv
namespace roadstar {
namespace common {
namespace visualization {

class CanvasTransform;

class ILayerRenderer {
 public:
  virtual void Render(const CanvasTransform &trans, cv::Mat *mat) = 0;

  virtual ~ILayerRenderer() = default;
};

template <class T>
class LayerRendererT : public ILayerRenderer {
 public:
  using DataType = T;

  virtual ~LayerRendererT() = default;

  virtual void DoRender(const DataType &, const CanvasTransform &,
                        cv::Mat *) = 0;
};

// thread safe
template <class Renderer>
class TSLayerRenderer : public Renderer {
 public:
  using Renderer::Renderer;
  using DataType = typename Renderer::DataType;

  void Render(const CanvasTransform &trans, cv::Mat *mat) final {
    if (!data_) {
      return;
    }
    auto data = std::atomic_load(&data_);
    Renderer::DoRender(*data, trans, mat);
  }

  void SetData(const DataType &data) {
    std::atomic_store(&data_, std::make_shared<const DataType>(data));
  }

  void SetData(DataType &&data) {
    std::atomic_store(&data_,
                      std::make_shared<const DataType>(std::move(data)));
  }

 private:
  std::shared_ptr<const DataType> data_;
};

}  // namespace visualization
}  // namespace common
}  // namespace roadstar

#endif
