#include <algorithm>
#include <vector>

#include "mutex.h"
#include "stringprintf.h"
#include "Mesa/include/GL/osmesa.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"

namespace research_vision {
namespace facedecoder {

using ::tensorflow::DEVICE_CPU;
using ::tensorflow::OpKernel;
using ::tensorflow::OpKernelConstruction;
using ::tensorflow::OpKernelContext;
using ::tensorflow::PartialTensorShape;
using ::tensorflow::ResourceBase;
using ::tensorflow::ResourceMgr;
using ::tensorflow::Status;
using ::tensorflow::Tensor;
using ::tensorflow::TensorShape;
using ::tensorflow::TensorShapeUtils;
using ::tensorflow::error::RESOURCE_EXHAUSTED;
using ::tensorflow::errors::Internal;
using ::tensorflow::errors::InvalidArgument;
using ::tensorflow::errors::ResourceExhausted;
using ::tensorflow::mutex;
using ::tensorflow::mutex_lock;

REGISTER_OP("MeshRenderer")
    .Input("vertices: float32")
    .Input("triangles: int32")
    .Attr("image_width: int")
    .Attr("image_height: int")
    .Output("barycentric_coordinates: float32")
    .Output("triangle_ids: int32")
    .Doc(R"doc(
Implements an OpenGL rasterization kernel for rendering mesh geometry.

vertices: 2-D tensor with shape [vertex_count, 3]. The 3-D positions of the mesh
  vertices in Normalized Device Coordinates.
triangles: 2-D tensor with shape [triangle_count, 3]. Each row is a tuple of
  indices into vertices specifying a triangle to be drawn. The triangle has an
  outward facing normal when the given indices appear in a clockwise winding to
  the viewer.
image_width: positive int attribute specifying the width of the output image.
image_height: positive int attribute specifying the height of the output image.
barycentric_coordinates: 3-D tensor with shape [image_height, image_width, 3]
  containing the rendered barycentric coordinate triplet per pixel, before
  perspective correction. The triplet is the zero vector if the pixel is outside
  the mesh boundary. For valid pixels, the ordering of the coordinates
  corresponds to the ordering in triangles.
triangle_ids: 2-D tensor with shape [image_height, image_width]. Contains the
  triangle id value for each pixel in the output image. For pixels within the
  mesh, this is the integer value in the range [0, num_vertices] from triangles.
  For vertices outside the mesh this is 0; 0 can either indicate belonging to
  triangle 0, or being outside the mesh. This ensures all returned triangle ids
  will validly index into the vertex array, enabling the use of tf.gather with
  indices from this tensor. The barycentric coordinates can be used to determine
  pixel validity instead.
)doc");

// Access to OSMesa contexts is managed by TensorFlow's ResourceManager, with
// the underlying OSMesaContext object stored in the render_context variable.
struct ManagedRenderContext : public ResourceBase {
  static Status CreateContext(ManagedRenderContext** ret_context) {
    constexpr int kDepthBufferBitDepth = 16;
    void* render_context = static_cast<void*>(OSMesaCreateContextExt(
        OSMESA_RGBA, kDepthBufferBitDepth, 0, 0, nullptr));
    if (render_context == nullptr) {
      return Status(RESOURCE_EXHAUSTED, "Could not create a Mesa context.");
    }
    *ret_context = new ManagedRenderContext();
    (*ret_context)->render_context = render_context;
    return Status::OK();
  }

  // Once a context has been created, we never explicitly destruct it with
  // OSMesaDestroyContext- doing so inside a tensorflow kernel leaves OSMesa in
  // a bad state due to the destruction of variables between contexts.
  ~ManagedRenderContext() override {}

  string DebugString() override {
    mutex_lock l(render_context_mutex);
    return StringPrintf("Managed Render Context [%p, %p]",
                        &render_context_mutex, render_context);
  }

  mutex render_context_mutex;
  void* render_context;
};

class MeshRendererOp : public OpKernel {
 public:
  explicit MeshRendererOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("image_width", &image_width_));
    OP_REQUIRES(context, image_width_ > 0,
                InvalidArgument("Image width must be > 0, got ", image_width_));

    OP_REQUIRES_OK(context, context->GetAttr("image_height", &image_height_));
    OP_REQUIRES(
        context, image_height_ > 0,
        InvalidArgument("Image height must be > 0, got ", image_height_));

    frame_buffer_ = std::vector<uint8>(image_width_ * image_height_ * 4, 0);
  }

  ~MeshRendererOp() override {}

  void Compute(OpKernelContext* context) override {
    const Tensor& vertices_tensor = context->input(0);
    OP_REQUIRES(
        context,
        PartialTensorShape({-1, 3}).IsCompatibleWith(vertices_tensor.shape()),
        InvalidArgument(
            "MeshRenderer expects vertices to have shape (-1, 3)."));
    // If the following line will not compile, the system float is likely
    // greater than 32 bits.
    auto vertices_flat = vertices_tensor.flat<GLfloat>();
    const GLfloat* vertices = vertices_flat.data();

    const Tensor& triangles_tensor = context->input(1);
    OP_REQUIRES(
        context,
        PartialTensorShape({-1, 3}).IsCompatibleWith(triangles_tensor.shape()),
        InvalidArgument("MeshRenderer expects triangles to be a matrix."));
    auto triangles_flat = triangles_tensor.flat<GLint>();
    const GLint* triangles = triangles_flat.data();
    const int triangle_count = triangles_flat.size() / 3;

    Tensor* barycentric_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       0, TensorShape({image_height_, image_width_, 3}),
                       &barycentric_tensor));

    Tensor* triangle_ids_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                1, TensorShape({image_height_, image_width_}),
                                &triangle_ids_tensor));

    ResourceMgr* context_manager = context->resource_manager();
    OP_REQUIRES(context, context_manager != nullptr,
                Internal("No resource manager was allocated."));

    ManagedRenderContext* render_context_out = nullptr;
    // We need a static mutex here because OSMesa's context creator function is
    // not threadsafe. To compromise between memory usage, stability, and
    // performance, we create only one context per ResourceManager container
    // passed to us.
    static mutex& osmesa_mutex = *new mutex;
    osmesa_mutex.lock();
    Status status = context_manager->LookupOrCreate<ManagedRenderContext>(
        "MeshRendererGlobalContainer", "OSMesaRenderContext",
        &render_context_out, ManagedRenderContext::CreateContext);
    osmesa_mutex.unlock();
    OP_REQUIRES_OK(context, status);

    mutex_lock l(render_context_out->render_context_mutex);

    OP_REQUIRES(context,
                OSMesaMakeCurrent(static_cast<OSMesaContext>(
                                      render_context_out->render_context),
                                  frame_buffer_.data(), GL_UNSIGNED_BYTE,
                                  image_width_, image_height_) == GL_TRUE,
                ResourceExhausted("Could not activate the framebuffer."));

    glDisable(GL_LIGHTING);
    glDepthFunc(GL_LESS);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    RenderTriangleIds(vertices, triangles, triangle_count, triangle_ids_tensor);
    ComputeBarycentricCoordinates(vertices, triangles, triangle_ids_tensor,
                                  barycentric_tensor);
    auto output = triangle_ids_tensor->flat<int32>();
    // The rendered triangle ids disambiguate mesh and background pixels by
    // setting background pixels to an id of -1. However, the python tf.gather
    // interface expects all entries of the tensor to validly index into the
    // vertex buffer, so we must remove those invalid indices:
    for (size_t i = 0; i < output.size(); ++i) {
      if (output(i) == -1) {
        output(i) = 0;
      }
    }
  }

 private:
  // Computes the triplet of barycentric coordinates at each pixel from the
  // rasterized triangle ids.
  //
  // vertices: A flattened 2D array with 3*vertex_count elements.
  //     Each contiguous triplet is the XYZ location of the vertex with that
  //     triplet's id.
  // triangles: A flattened 2D array with 3*triangle_count elements.
  //     Each contiguous triplet is the three vertex ids indexing into vertices
  //     describing one triangle with clockwise winding.
  // triangle_ids_tensor: A 2D Tensor with shape [image_height, image_width].
  //     Each element is the triangle id in the range [0, triangle_count)
  //     present at the corresponding pixel, or 0 if there is no triangle
  //     at the pixel.
  // barycentric_coordinates_tensor: A 3D Tensor with shape [image_height,
  //     image_width, 3], within which the output of this function is returned.
  //     Upon function exit, contains the triplet of barycentric coordinates
  //     at each pixel in the same vertex ordering as triangles.
  void ComputeBarycentricCoordinates(const GLfloat* vertices,
                                    const GLint* triangles,
                                    const Tensor* triangle_ids_tensor,
                                    Tensor* barycentric_coordinates_tensor) {
    auto barycentric_coordinates =
        barycentric_coordinates_tensor->flat<float>();
    const int32* triangle_ids = triangle_ids_tensor->flat<int32>().data();

    int pixel_id = -1;
    // The pixel x and y normalized device coordinates vary between
    // -1.0 and 1.0 across the screen:
    const float y_increment = 2.0f / image_height_;
    const float x_increment = 2.0f / image_width_;
    float py = -1.0f - y_increment / 2.0f;
    for (int iy = 0; iy < image_height_; ++iy) {
      // Reset the current screen space value of px at the start of each row:
      float px = -1.0f - x_increment / 2.0f;
      py += y_increment;
      for (int ix = 0; ix < image_width_; ++ix) {
        px += x_increment;
        const GLint triangle_id = triangle_ids[++pixel_id];
        if (triangle_id == -1) {
          barycentric_coordinates(3 * pixel_id) = 0.0f;
          barycentric_coordinates(3 * pixel_id + 1) = 0.0f;
          barycentric_coordinates(3 * pixel_id + 2) = 0.0f;
          continue;
        }
        const GLint v0_x_id = 3 * triangles[3 * triangle_id];
        const GLint v1_x_id = 3 * triangles[3 * triangle_id + 1];
        const GLint v2_x_id = 3 * triangles[3 * triangle_id + 2];
        const float v0x = vertices[v0_x_id];
        const float v0y = vertices[v0_x_id + 1];
        const float v1x = vertices[v1_x_id];
        const float v1y = vertices[v1_x_id + 1];
        const float v2x = vertices[v2_x_id];
        const float v2y = vertices[v2_x_id + 1];

        // Compute twice the area of two barycentric triangles, as well as the
        // triangle they sit in; the barycentric is the ratio of the triangle
        // areas, so the factor of two does not change the result.
        const float twice_triangle_area =
            (v2x - v0x) * (v1y - v0y) - (v2y - v0y) * (v1x - v0x);
        const float b0 = ((px - v1x) * (v2y - v1y) - (py - v1y) * (v2x - v1x)) /
                         twice_triangle_area;
        const float b1 = ((px - v2x) * (v0y - v2y) - (py - v2y) * (v0x - v2x)) /
                         twice_triangle_area;
        // The three upper triangles partition the lower triangle, so we can
        // compute the third barycentric coordinate using the other two:
        const float b2 = 1.0f - b0 - b1;

        barycentric_coordinates(3 * pixel_id) = b0;
        barycentric_coordinates(3 * pixel_id + 1) = b1;
        barycentric_coordinates(3 * pixel_id + 2) = b2;
      }
    }
  }

  // Uses OpenGL to rasterize the input scene and extract the computed triangle
  // id values per pixel.
  //
  // vertices: A flattened 2D array with 3*vertex_count elements.
  //     Each contiguous triplet is the XYZ location of the vertex with that
  //     triplet's id.
  // triangles: A flattened 2D array with 3*triangle_count elements.
  //     Each contiguous triplet is the three vertex ids indexing into vertices
  //     describing one triangle with clockwise winding.
  // triangle_count: The number of triangles stored in the array triangles.
  // triangle_ids_tensor: A 2D Tensor with shape [image_height, image_width].
  //     Each element is the triangle id in the range [0, triangle_count)
  //     present at the corresponding pixel, or 0 if there is no triangle
  //     at the pixel.
  void RenderTriangleIds(const GLfloat* vertices, const GLint* triangles,
                         int triangle_count, Tensor* triangle_ids_tensor) {
    auto triangle_ids = triangle_ids_tensor->flat<int32>();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glBegin(GL_TRIANGLES);
    for (int ti = 0; ti < triangle_count; ++ti) {
      // We reserve the 0 id for the background:
      const int32 to_encode = ti + 1;
      glColor4ubv(reinterpret_cast<const GLubyte*>(&to_encode));
      for (int vi = 0; vi < 3; ++vi) {
        const GLint vertex_index = 3 * triangles[3 * ti + vi];
        glVertex3fv(&vertices[vertex_index]);
      }
    }
    glEnd();
    glFinish();

    for (size_t i = 0; i < triangle_ids.size(); ++i) {
      triangle_ids(i) =
          *(reinterpret_cast<const int32*>(&frame_buffer_[4 * i])) - 1;
    }
  }

  TF_DISALLOW_COPY_AND_ASSIGN(MeshRendererOp);

  std::vector<uint8> frame_buffer_;
  int image_width_;
  int image_height_;
};

REGISTER_KERNEL_BUILDER(Name("MeshRenderer").Device(DEVICE_CPU),
                        MeshRendererOp);

}  // namespace facedecoder
}  // namespace research_vision
