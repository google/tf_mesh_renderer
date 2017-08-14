# C++ kernels and generated wrappers for the facedecoder TensorFlow ops.

load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")

cc_library(
    name = "mesh_renderer_kernel",
    srcs = ["mesh_renderer.cc"],
    visibility = ["//tf_mesh_renderer:__pkg__"],
    deps = [
        ":mutex",
        ":stringprintf",
        ":OSMesa_without_tls",
        ":glu_without_tls",
        "//tensorflow/core:framework",
    ],
    alwayslink = 1,
)

tf_gen_op_wrapper_py(
    name = "gen_mesh_renderer",
    out = "gen_mesh_renderer.py",
    visibility = ["//tf_mesh_renderer:__pkg__"],
    deps = [":mesh_renderer_kernel"],
)

cc_library(
    name = "mesh_renderer_grad_kernel",
    srcs = ["mesh_renderer_grad.cc"],
    visibility = ["//tf_mesh_renderer:__pkg__"],
    deps = [
        "tensorflow/core:framework",
    ],
    alwayslink = 1,
)

tf_gen_op_wrapper_py(
    name = "gen_mesh_renderer_grad",
    out = "gen_mesh_renderer_grad.py",
    visibility = ["//tf_mesh_renderer:__pkg__"],
    deps = [":mesh_renderer_grad_kernel"],
)
