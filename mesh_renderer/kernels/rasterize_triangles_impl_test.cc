#include <fstream>

#include "gtest/gtest.h"
#include "rasterize_triangles_impl.h"

#include "third_party/lodepng.h"

namespace tf_mesh_renderer {
namespace {

typedef unsigned char uint8;

const int kImageHeight = 480;
const int kImageWidth = 640;

std::string GetRunfilesRelativePath(const std::string& filename) {
  const std::string srcdir = std::getenv("TEST_SRCDIR");
  const std::string test_data = "/tf_mesh_renderer/mesh_renderer/test_data/";
  return srcdir + test_data + filename;
}

void LoadPng(const std::string& filename, std::vector<uint8>* output) {
  unsigned width, height;
  unsigned error = lodepng::decode(*output, width, height, filename.c_str());
  ASSERT_TRUE(error == 0) << "Decoder error: " << lodepng_error_text(error);
}

void SavePng(const std::string& filename, const std::vector<uint8>& image) {
  unsigned error =
      lodepng::encode(filename.c_str(), image, kImageWidth, kImageHeight);
  ASSERT_TRUE(error == 0) << "Encoder error: " << lodepng_error_text(error);
}

void FloatRGBToUint8RGBA(const std::vector<float>& input,
                         std::vector<uint8>* output) {
  output->resize(kImageHeight * kImageWidth * 4);
  for (int y = 0; y < kImageHeight; ++y) {
    for (int x = 0; x < kImageWidth; ++x) {
      for (int c = 0; c < 3; ++c) {
        (*output)[(y * kImageWidth + x) * 4 + c] =
            input[(y * kImageWidth + x) * 3 + c] * 255;
      }
      (*output)[(y * kImageWidth + x) * 4 + 3] = 255;
    }
  }
}

void ExpectImageFileAndImageAreEqual(const std::string& baseline_file,
                                     const std::vector<float>& result,
                                     const std::string& comparison_name,
                                     const std::string& failure_message) {
  std::vector<uint8> baseline_rgba, result_rgba;
  LoadPng(GetRunfilesRelativePath(baseline_file), &baseline_rgba);
  FloatRGBToUint8RGBA(result, &result_rgba);

  const bool images_match = baseline_rgba == result_rgba;

  if (!images_match) {
    const std::string result_output_path = "/tmp/" + comparison_name + "_result.png";
    SavePng(result_output_path, result_rgba);
  }

  EXPECT_TRUE(images_match) << failure_message;
}

class RasterizeTrianglesImplTest : public ::testing::Test {
 protected:
  RasterizeTrianglesImplTest() {
    const int kNumPixels = kImageHeight * kImageWidth;
    const float kClearDepth = 1.0;

    barycentrics_buffer_.resize(kNumPixels * 3);
    triangle_ids_buffer_.resize(kNumPixels);
    z_buffer_.resize(kNumPixels, kClearDepth);
  }

  std::vector<float> barycentrics_buffer_;
  std::vector<int32> triangle_ids_buffer_;
  std::vector<float> z_buffer_;
};

TEST_F(RasterizeTrianglesImplTest, CanRasterizeTriangle) {
  const std::vector<float> vertices = {-0.5, -0.5, 0.8,  0.0, 0.5,
                                       0.3,  0.5,  -0.5, 0.3};
  const std::vector<int32> triangles = {0, 1, 2};

  RasterizeTrianglesImpl(vertices.data(), triangles.data(), 1, kImageWidth,
                         kImageHeight, triangle_ids_buffer_.data(),
                         barycentrics_buffer_.data(), z_buffer_.data());
  ExpectImageFileAndImageAreEqual("Simple_Triangle.png", barycentrics_buffer_,
                                  "triangle", "simple triangle does not match");
}

TEST_F(RasterizeTrianglesImplTest, CanRasterizeTetrahedron) {
  const std::vector<float> vertices = {-0.5, -0.5, 0.8, 0.0, 0.5, 0.3,
                                       0.5,  -0.5, 0.3, 0.0, 0.0, 0.0};
  const std::vector<int32> triangles = {0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3};

  RasterizeTrianglesImpl(vertices.data(), triangles.data(), 4, kImageWidth,
                         kImageHeight, triangle_ids_buffer_.data(),
                         barycentrics_buffer_.data(), z_buffer_.data());

  ExpectImageFileAndImageAreEqual("Simple_Tetrahedron.png",
                                  barycentrics_buffer_, "tetrahedron",
                                  "simple tetrahedron does not match");
}

}  // namespace
}  // namespace tf_mesh_renderer
