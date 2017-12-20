workspace(name = "tf_mesh_renderer")

# local_repository(
#     name = "org_tensorflow",
#     path = "../tensorflow",
# )

new_local_repository(
    name = "local_tensorflow_headers",
    path = "/usr/local/google/home/fcole/tensorflow/lib/python2.7/site-packages/tensorflow/include",
    build_file= "BUILD.tensorflow_headers",
)

new_local_repository(
    name = "local_nsync_headers",
    path = "/usr/local/google/home/fcole/tensorflow/lib/python2.7/site-packages/external/nsync/public",
    build_file= "BUILD.nsync",
)

new_local_repository(
    name = "local_tensorflow_lib",
    path = "/usr/local/google/home/fcole/tensorflow/lib/python2.7/site-packages/tensorflow/libtensorflow_framework.so",
    build_file= "BUILD.tensorflow_lib",
)

# GoogleTest/GoogleMock framework. Used by most unit-tests.
http_archive(
     name = "com_google_googletest",
     urls = ["https://github.com/google/googletest/archive/master.zip"],
     strip_prefix = "googletest-master",
)

# TensorFlow depends on "io_bazel_rules_closure" so we need this here.
# Needs to be kept in sync with the same target in TensorFlow's WORKSPACE file.
# http_archive(
#     name = "io_bazel_rules_closure",
#     sha256 = "110fe68753413777944b473c25eed6368c4a0487cee23a7bac1b13cc49d3e257",
#     strip_prefix = "rules_closure-4af89ef1db659eb41f110df189b67d4cf14073e1",
#     urls = [
#         "http://mirror.bazel.build/github.com/bazelbuild/rules_closure/archive/4af89ef1db659eb41f110df189b67d4cf14073e1.tar.gz",
#         "https://github.com/bazelbuild/rules_closure/archive/4af89ef1db659eb41f110df189b67d4cf14073e1.tar.gz",  # 2017-08-28
#     ],
# )

# # Please add all new TensorFlow Serving dependencies in workspace.bzl.
# load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")
# tf_workspace(tf_repo_name = "org_tensorflow")

# # Specify the minimum required bazel version.
# load("@org_tensorflow//tensorflow:workspace.bzl", "check_version")
# check_version("0.5.4")
