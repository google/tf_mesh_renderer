This is a differentiable, 3D mesh renderer using TensorFlow.

This is not an official Google product.

The interface to the renderer is provided by mesh_renderer.py and
rasterize_triangles.py, which provide TensorFlow Ops that can be added to a
TensorFlow graph. The internals of the renderer are handled by a C++ kernel.

The input to the C++ rendering kernel is a list of 3D vertices and a list of
triangles, where a triangle consists of a list of three vertex ids. The
output of the renderer is a pair of images containing triangle ids and
barycentric weights. Pixel values in the barycentric weight image are the
weights of the pixel center point with respect to the triangle at that pixel
(identified by the triangle id). The renderer provides derivatives of the
barycentric weights of the pixel centers with respect to the vertex
positions.

Any approximation error stems from the assumption that the triangle id at a
pixel does not change as the vertices are moved. This is a reasonable
approximation for small changes in vertex position. Even when the triangle id
does change, the derivatives will be computed by extrapolating the barycentric
weights of a neighboring triangle, which will produce a good approximation if
the mesh is smooth. The main source of error occurs at occlusion boundaries, and
particularly at the edge of an open mesh, where the background appears opposite
the triangle's edge.

How to Build
------------

Follow the instructions to [install TensorFlow using virtualenv](https://www.tensorflow.org/install/install_linux#installing_with_virtualenv).

Build and run tests using Bazel from inside the (tensorflow) virtualenv:

`(tensorflow)$ bazel test ...`
