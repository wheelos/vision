#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "vision/brown_camera_model.h"
#include "vision/camera_model.h"
#include "vision/omnidirectional_camera_model.h"
#include "vision/pinhole_camera_model.h"

namespace py = pybind11;
using namespace wheel::vision;

PYBIND11_MODULE(py_camera_model, m) {
  m.doc() = "Python bindings for Wheel Vision Camera Models";

  // 1. 绑定基类 CameraModel (作为接口)
  // 注意：因为是纯虚类，pybind11 需要知道它不仅是 Trampoline
  py::class_<CameraModel, std::shared_ptr<CameraModel>>(m, "CameraModel")
      .def("width", &CameraModel::width)
      .def("height", &CameraModel::height)
      .def("type", &CameraModel::Type)
      .def("name", &CameraModel::Name)
      // 适配器：C++ 指针出参 -> Python 元组返回 (success, pixel_array)
      .def("ray_to_pixel",
           [](CameraModel& self, const Eigen::Vector3f& ray) {
             Eigen::Vector2f pixel;
             bool success = self.RayToPixel(ray, &pixel);
             return std::make_pair(success, pixel);
           })
      .def("pixel_to_ray",
           [](CameraModel& self, const Eigen::Vector2f& pixel) {
             Eigen::Vector3f ray;
             bool success = self.PixelToRay(pixel, &ray);
             return std::make_pair(success, ray);
           })
      .def("apply_image_transform",
           [](CameraModel& self, const Eigen::Matrix3f& transform, int w,
              int h) { self.ApplyImageTransform(transform, w, h); });

  // 2. 绑定 Pinhole 模型
  py::class_<PinholeCameraModel, CameraModel,
             std::shared_ptr<PinholeCameraModel>>(m, "PinholeCameraModel")
      .def(py::init<>())
      .def("init", &PinholeCameraModel::Init);

  // 3. 绑定 Brown 模型
  py::class_<BrownCameraModel, CameraModel, std::shared_ptr<BrownCameraModel>>(
      m, "BrownCameraModel")
      .def(py::init<>())
      .def("init", &BrownCameraModel::Init)
      .def_property_readonly("distort_params",
                             &BrownCameraModel::distort_params);

  // 4. 绑定 Omni 模型
  py::class_<OmnidirectionalCameraModel, CameraModel,
             std::shared_ptr<OmnidirectionalCameraModel>>(
      m, "OmnidirectionalCameraModel")
      .def(py::init<>())
      .def("init", &OmnidirectionalCameraModel::Init);
}
