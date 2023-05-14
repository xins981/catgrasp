#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "collision_manager.h"
#include "common.h"


namespace py = pybind11;



PYBIND11_MODULE(my_cpp, m)
{
  py::class_<CollisionManager>(m, "CollisionManager")
      .def(py::init<>())
      .def("registerMesh", &CollisionManager::registerMesh)
      .def("registerPointCloud", &CollisionManager::registerPointCloud)
      .def("setTransform", &CollisionManager::setTransform)
      .def("isAnyCollision", &CollisionManager::isAnyCollision);

  m.def("augmentGraspPoses", &augmentGraspPoses);
  m.def("filterGraspPose", &filterGraspPose);
  m.def("filterPushPose", &filterPushPose);
  m.def("detectCollisionMove", &detectCollisionMove);
  m.def("makeOccupancyGridFromCloudScan", &makeOccupancyGridFromCloudScan);
  m.def("distObj2Env", &distObj2Env);
}

