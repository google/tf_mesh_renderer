#include <algorithm>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace {

// Threshold for a barycentric coordinate triplet's sum, below which the
// coordinates at a pixel are deemed degenerate. Most such degenerate triplets
// in an image will be exactly zero, as this is how pixels outside the mesh
// are rendered.
constexpr float kDegenerateBarycentricCoordinatesCutoff = 0.9f;

// If the area of a triangle is very small in screen space, the corner vertices
// are approaching colinearity, and we should drop the gradient to avoid
// numerical instability (in particular, blowup, as the forward pass computation
// already only has 8 bits of precision).
constexpr float kMinimumTriangleArea = 1e-13;

}  // namespace

namespace research_vision {
namespace facedecoder {

using ::tensorflow::DEVICE_CPU;
using ::tensorflow::OpKernel;
using ::tensorflow::OpKernelConstruction;
using ::tensorflow::OpKernelContext;
using ::tensorflow::PartialTensorShape;
using ::tensorflow::Status;
using ::tensorflow::Tensor;
using ::tensorflow::TensorShape;
using ::tensorflow::errors::InvalidArgument;

REGISTER_OP("MeshRendererGrad")
    .Input("vertices: float32")
    .Input("triangles: int32")
    .Input("barycentric_coordinates: float32")
    .Input("triangle_ids: int32")
    .Input("df_dbarycentric_coordinates: float32")
    .Attr("image_width: int")
    .Attr("image_height: int")
    .Output("df_dvertices: float32");

class MeshRendererGradOp : public OpKernel {
 public:
  explicit MeshRendererGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("image_width", &image_width_));
    OP_REQUIRES(context, image_width_ > 0,
                InvalidArgument("Image width must be > 0, got ", image_width_));

    OP_REQUIRES_OK(context, context->GetAttr("image_height", &image_height_));
    OP_REQUIRES(
        context, image_height_ > 0,
        InvalidArgument("Image height must be > 0, got ", image_height_));
  }

  ~MeshRendererGradOp() override {}

  void Compute(OpKernelContext* context) override {
    const Tensor& vertices_tensor = context->input(0);
    OP_REQUIRES(
        context,
        PartialTensorShape({-1, 3}).IsCompatibleWith(vertices_tensor.shape()),
        InvalidArgument(
            "MeshRendererGrad expects vertices to have shape (-1, 3)."));
    auto vertices_flat = vertices_tensor.flat<float>();
    const unsigned int vertex_count = vertices_flat.size() / 3;
    const float* vertices = vertices_flat.data();

    const Tensor& triangles_tensor = context->input(1);
    OP_REQUIRES(
        context,
        PartialTensorShape({-1, 3}).IsCompatibleWith(triangles_tensor.shape()),
        InvalidArgument("MeshRendererGrad expects triangles to be a matrix."));
    auto triangles_flat = triangles_tensor.flat<int>();
    const int* triangles = triangles_flat.data();

    const Tensor& barycentric_coordinates_tensor = context->input(2);
    OP_REQUIRES(
        context,
        TensorShape({image_height_, image_width_, 3}) ==
            barycentric_coordinates_tensor.shape(),
        InvalidArgument("MeshRendererGrad expects barycentric_coordinates to "
                        "have shape {image_height, image_width, 3}"));
    auto barycentric_coordinates_flat =
        barycentric_coordinates_tensor.flat<float>();
    const float* barycentric_coordinates = barycentric_coordinates_flat.data();

    const Tensor& triangle_ids_tensor = context->input(3);
    OP_REQUIRES(
        context,
        TensorShape({image_height_, image_width_}) ==
            triangle_ids_tensor.shape(),
        InvalidArgument("MeshRendererGrad expected triangle_ids to have shape "
                        " {image_height, image_width}"));
    auto triangle_ids_flat = triangle_ids_tensor.flat<int>();
    const int* triangle_ids = triangle_ids_flat.data();

    // The naming convention we use for all derivatives is d<y>_d<x> ->
    // the partial of y with respect to x.
    const Tensor& df_dbarycentric_coordinates_tensor = context->input(4);
    OP_REQUIRES(
        context,
        TensorShape({image_height_, image_width_, 3}) ==
            df_dbarycentric_coordinates_tensor.shape(),
        InvalidArgument("MeshRendererGrad expects df_dbarycentric_coordinates "
                        "to have shape {image_height, image_width, 3}"));
    auto df_dbarycentric_coordinates_flat =
        df_dbarycentric_coordinates_tensor.flat<float>();
    const float* df_dbarycentric_coordinates =
        df_dbarycentric_coordinates_flat.data();

    Tensor* df_dvertices_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({vertex_count, 3}),
                                            &df_dvertices_tensor));
    auto df_dvertices_flat = df_dvertices_tensor->flat<float>();
    float* df_dvertices = df_dvertices_flat.data();
    std::fill(df_dvertices, df_dvertices + vertex_count * 3, 0.0f);

    // We first loop over each pixel in the output image, and compute
    // dbarycentric_coordinate[0,1,2]/dvertex[0x, 0y, 1x, 1y, 2x, 2y].
    // Next we compute each value above's contribution to
    // df/dvertices, building up that matrix as the output of this iteration.
    for (unsigned int pixel_id = 0; pixel_id < image_height_ * image_width_;
         ++pixel_id) {
      // b0, b1, and b2 are the three barycentric coordinate values
      // rendered at pixel pixel_id.
      const float b0 = barycentric_coordinates[3 * pixel_id];
      const float b1 = barycentric_coordinates[3 * pixel_id + 1];
      const float b2 = barycentric_coordinates[3 * pixel_id + 2];

      if (b0 + b1 + b2 < kDegenerateBarycentricCoordinatesCutoff) {
        continue;
      }

      const float df_db0 = df_dbarycentric_coordinates[3 * pixel_id];
      const float df_db1 = df_dbarycentric_coordinates[3 * pixel_id + 1];
      const float df_db2 = df_dbarycentric_coordinates[3 * pixel_id + 2];

      const int triangle_at_current_pixel = triangle_ids[pixel_id];
      const int* vertices_at_current_pixel =
          &triangles[3 * triangle_at_current_pixel];

      // 'Unrotated' values are those that are invariant to the value of the
      // iteration variable vertex_rotation, described below.
      const int unrotated_v0_id = 3 * vertices_at_current_pixel[0];
      const int unrotated_v1_id = 3 * vertices_at_current_pixel[1];
      const int unrotated_v2_id = 3 * vertices_at_current_pixel[2];

      // Extract the x,y components of the unrotated vertices' normalized
      // device coordinates to compute quantities invariant to the system
      // rotation:
      const float unrotated_v0x = vertices[unrotated_v0_id];
      const float unrotated_v0y = vertices[unrotated_v0_id + 1];
      const float unrotated_v1x = vertices[unrotated_v1_id];
      const float unrotated_v1y = vertices[unrotated_v1_id + 1];
      const float unrotated_v2x = vertices[unrotated_v2_id];
      const float unrotated_v2y = vertices[unrotated_v2_id + 1];

      // px and py, the x and y components of the 3D position represented at the
      // current pixel, are computed by barycentric interpolation of the corner
      // vertex positions:
      const float px =
          b0 * unrotated_v0x + b1 * unrotated_v1x + b2 * unrotated_v2x;
      const float py =
          b0 * unrotated_v0y + b1 * unrotated_v1y + b2 * unrotated_v2y;

      // The derivatives share a common denominator, as the screen space
      // area of the triangle is common to all three vertices.
      // Note this quantity is actually twice the area (i.e. the size of the
      // parallelogram given by the screen space cross product), but we only use
      // the ratio of areas, and we compute all areas this way.
      const float triangle_area =
          unrotated_v0y * unrotated_v1x - unrotated_v0x * unrotated_v1y -
          unrotated_v0y * unrotated_v2x + unrotated_v1y * unrotated_v2x +
          unrotated_v0x * unrotated_v2y - unrotated_v1x * unrotated_v2y;

      if (triangle_area < kMinimumTriangleArea) {
        continue;
      }
      const float triangle_area_sqr = triangle_area * triangle_area;

      // We need to compute the partials of each of the three barycentrics
      // with respect to each of the six (corner vertex, {x,y}) pairs. However,
      // the choice of vertex names 0, 1, and 2 is arbitrary as long as the
      // order is consistent with respect to the clockwise winding. We compute
      // the system with respect to one such choice, and then rotate our
      // reference frame twice to get the derivatives with respect to the other
      // two corner vertices.
      for (int vertex_rotation = 0; vertex_rotation < 3; ++vertex_rotation) {
        const int v0_id = vertices_at_current_pixel[vertex_rotation];
        const int v1_id = vertices_at_current_pixel[(1 + vertex_rotation) % 3];
        const int v2_id = vertices_at_current_pixel[(2 + vertex_rotation) % 3];

        // To compute the derivatives at the current pixel, we need the x and y
        // 3D position components of the three corner vertices for the triangle
        // this pixel represents:
        const float v0x = vertices[3 * v0_id];
        const float v0y = vertices[3 * v0_id + 1];
        const float v1x = vertices[3 * v1_id];
        const float v1y = vertices[3 * v1_id + 1];
        const float v2x = vertices[3 * v2_id];
        const float v2y = vertices[3 * v2_id + 1];

        // We factor out all shared elements of the gradients to create a
        // single multiplier for the vertex v0.
        const float vertex_multiplier =
            (py * (v2x - v1x) + px * (v1y - v2y) + v1x * v2y - v1y * v2x) /
            triangle_area_sqr;

        // Having factored out other components of the gradients, these are the
        // only differing values:
        const float db_dv0_rotated[6] = {(v2y - v1y), (v1x - v2x), (v0y - v2y),
                                         (v2x - v0x), (v1y - v0y), (v0x - v1x)};

        // These indices invert the vertex rotation- i.e. if vertex 0 -> 2,
        // then inverted_v2_idx := 0. They allow us to permute the barycentric
        // coordinates back to their consistent indices with respect to the
        // input tensor.
        const int inverted_v0_idx = (3 - vertex_rotation) % 3;
        const int inverted_v1_idx = (4 - vertex_rotation) % 3;
        const int inverted_v2_idx = (5 - vertex_rotation) % 3;

        // By undoing the rotation, we get the standard (flattened) Jacobian
        // matrix of the three consistently named barycentric coordinates at the
        // pixel with respect to the x and y coordinates of vertex 0:
        const float db_dv0[6] = {db_dv0_rotated[2 * inverted_v0_idx],
                                 db_dv0_rotated[2 * inverted_v0_idx + 1],
                                 db_dv0_rotated[2 * inverted_v1_idx],
                                 db_dv0_rotated[2 * inverted_v1_idx + 1],
                                 db_dv0_rotated[2 * inverted_v2_idx],
                                 db_dv0_rotated[2 * inverted_v2_idx + 1]};

        // We need to compute df/dverts = matmul(df/dbarycentric_coordinates,
        // dbarycentric_coordinates/dvertex_positions) without explicitly
        // forming the large sparse matrix dbarys/dverts. We just computed six
        // values from dbarycentric_coordinates/dvertex_positions above, and so
        // we bin the contributions of those matrix entries to the overall
        // multiplication result, which is the final output tensor.
        df_dvertices[3 * v0_id] +=
            vertex_multiplier *
            (df_db0 * db_dv0[0] + df_db1 * db_dv0[2] + df_db2 * db_dv0[4]);
        df_dvertices[3 * v0_id + 1] +=
            vertex_multiplier *
            (df_db0 * db_dv0[1] + df_db1 * db_dv0[3] + df_db2 * db_dv0[5]);
      }
    }
  }

 private:
  int image_width_;
  int image_height_;
};

REGISTER_KERNEL_BUILDER(Name("MeshRendererGrad").Device(DEVICE_CPU),
                        MeshRendererGradOp);

}  // namespace facedecoder
}  // namespace research_vision
