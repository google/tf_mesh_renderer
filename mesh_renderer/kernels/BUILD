#load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_custom_op_library", "tf_gen_op_wrapper_py")

cc_library(
    name = "rasterize_triangles_impl",
    srcs = ["rasterize_triangles_impl.cc"],
    hdrs = ["rasterize_triangles_impl.h"],
    #deps = ["@org_tensorflow//tensorflow/core:framework"],
    #deps = ["@local_tensorflow_headers//:tensorflow_headers"],
    #copts = ["-Iexternal/local_tensorflow_headers/include"],
)

cc_test(
    name = "rasterize_triangles_impl_test",
    srcs = ["rasterize_triangles_impl_test.cc"],
    data = [
        "//mesh_renderer/test_data:images",
    ],
    deps = [
        ":rasterize_triangles_impl",
        "//third_party:lodepng",
        "@com_google_googletest//:gtest_main",
    ],
    #copts = ["-Iexternal/local_tensorflow_headers/include"],
)

# cc_library(
#     name = "rasterize_triangles_kernel",
#     srcs = ["rasterize_triangles_op.cc"],
#     visibility = ["//mesh_renderer:__subpackages__"],
#     deps = [
#         ":rasterize_triangles_impl",
#     #     "@local_tensorflow_headers//:tensorflow_headers",
#     #     "@local_tensorflow_lib//:tensorflow_framework",
#     #     "@local_nsync_headers//:nsync_headers",
#     #     "@org_tensorflow//tensorflow/core:framework",
#     ],
#     #copts = ["-Iexternal/local_tensorflow_headers"],
#     copts = ["-I/usr/local/google/home/fcole/tensorflow/local/lib/python2.7/site-packages/tensorflow/include",
#     "-I/usr/local/google/home/fcole/tensorflow/local/lib/python2.7/site-packages/tensorflow/include/external/nsync/public",
#     "-L/usr/local/google/home/fcole/tensorflow/local/lib/python2.7/site-packages/tensorflow",
#     "-ltensorflow_framework"],
#     alwayslink = 1,
# )

# tf_gen_op_wrapper_py(
#     name = "gen_rasterize_triangles",
#     out = "gen_rasterize_triangles.py",
#     visibility = ["//mesh_renderer:__subpackages__"],
#     deps = [":rasterize_triangles_kernel"],
# )

# cc_library(
#     name = "rasterize_triangles_grad_kernel",
#     srcs = ["rasterize_triangles_grad.cc"],
#     visibility = ["//mesh_renderer:__subpackages__"],
#     # deps = [
#     #     "@local_tensorflow_headers//:tensorflow_headers",
#     #     "@local_tensorflow_lib//:tensorflow_framework",
#     #     "@local_nsync_headers//:nsync_headers",
#     #     "@org_tensorflow//tensorflow/core:framework",
#     # ],
#     copts = ["-I/usr/local/google/home/fcole/tensorflow/local/lib/python2.7/site-packages/tensorflow/include",
#     "-I/usr/local/google/home/fcole/tensorflow/local/lib/python2.7/site-packages/tensorflow/include/external/nsync/public",
#     "-L/usr/local/google/home/fcole/tensorflow/local/lib/python2.7/site-packages/tensorflow",
#     "-ltensorflow_framework"],
#     #copts = ["-Iexternal/local_tensorflow_headers"],
#     alwayslink = 1,
# )

genrule(
    name = "compile_the_thing",
    srcs = ["rasterize_triangles_grad.cc",
            "rasterize_triangles_op.cc",
            "rasterize_triangles_impl.cc",
            "rasterize_triangles_impl.h"],
    outs = ["rasterize_triangles_kernel.so"],
    cmd = "TF_INC=$$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())');\
           TF_LIB=$$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())'); \
           g++ -std=c++11 -shared $(SRCS) -o $@ -fPIC -I$$TF_INC -I$$TF_INC/external/nsync/public -L$$TF_LIB -ltensorflow_framework -O2",
    visibility = ["//mesh_renderer:__subpackages__"],
)

# tf_gen_op_wrapper_py(
    # name = "gen_rasterize_triangles_grad",
    # out = "gen_rasterize_triangles_grad.py",
    # visibility = ["//mesh_renderer:__subpackages__"],
    # deps = [":rasterize_triangles_grad_kernel"],
# )
#
