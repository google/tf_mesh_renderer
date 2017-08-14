# Description:
# Useful tensorflow ops for the facedecoder project
# that may be of general interest.

package(default_visibility = ["//visibility:public"])

py_library(
    name = "mesh_renderer",
    srcs = ["mesh_renderer.py"],
    deps = [
        "kernels:gen_mesh_renderer",
        "kernels:gen_mesh_renderer_grad",
        "kernels:mesh_renderer_grad_kernel",
        "kernels:mesh_renderer_kernel",
        "numpy",
        "tensorflow",
        "tf",
    ],
)

py_test(
    name = "mesh_renderer_test",
    size = "medium",
    timeout = "long",
    srcs = ["mesh_renderer_test.py"],
    data = [
        "test_data/Simple_Cube_0.png",
        "test_data/Simple_Cube_1.png",
        "test_data/Simple_Cube_2.png",
        "test_data/Simple_Cube_3.png",
        "test_data/Simple_Cube_4.png",
        "test_data/Simple_Triangle.png",
    ],
    deps = [
        ":mesh_renderer",
        "numpy",
        "tensorflow",
        "tf",
    ],
)
