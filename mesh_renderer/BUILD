package(default_visibility = ["//visibility:public"])

py_library(
    name = "mesh_renderer",
    srcs = ["mesh_renderer.py"],
    deps = [
        ":rasterize_triangles",
        # "kernels:gen_mesh_renderer",
        # "kernels:gen_mesh_renderer_grad",
        # "kernels:mesh_renderer_grad_kernel",
        # "kernels:mesh_renderer_kernel",
        # "numpy",
        # "tensorflow",
    ],
)

py_test(
    name = "mesh_renderer_test",
    size = "medium",
    timeout = "long",
    srcs = ["mesh_renderer_test.py"],
    data = [
        "//mesh_renderer/test_data:images",
        "//mesh_renderer/kernels:compile_the_thing",
    ],
    deps = [
        ":mesh_renderer",
    ],
)

py_library(
    name = "camera_utils",
    srcs = ["camera_utils.py"],
)

py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
)

py_library(
    name = "rasterize_triangles",
    srcs = ["rasterize_triangles.py"],
    # deps = [
    #     "//research/vision/facedecoder/ops/kernels:gen_rasterize_triangles",
    #     "//research/vision/facedecoder/ops/kernels:gen_rasterize_triangles_grad",
    #     "//mesh_renderer/kernels:rasterize_triangles_grad_kernel",
    #     "//mesh_renderer/kernels:rasterize_triangles_kernel",
    #     "//third_party/py/numpy",
    #     "//third_party/py/tensorflow:tensorflow_google",
    # ],
)

py_test(
    name = "rasterize_triangles_test",
    size = "medium",
    timeout = "long",
    srcs = ["rasterize_triangles_test.py"],
    data = [
        "//mesh_renderer/test_data:images",
        "//mesh_renderer/kernels:compile_the_thing",
    ],
    deps = [
        ":camera_utils",
        ":rasterize_triangles",
        ":test_utils",
    ],
)
