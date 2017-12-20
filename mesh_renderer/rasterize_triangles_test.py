# Copyright 2017 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

import test_utils
import camera_utils
import rasterize_triangles


class RenderTest(tf.test.TestCase):

  def setUp(self):
    self.test_data_directory = 'mesh_renderer/test_data/'

    tf.reset_default_graph()
    self.cube_vertex_positions = tf.constant(
        [[-1, -1, 1], [-1, -1, -1], [-1, 1, -1], [-1, 1, 1], [1, -1, 1],
         [1, -1, -1], [1, 1, -1], [1, 1, 1]],
        dtype=tf.float32)
    self.cube_triangles = tf.constant(
        [[0, 1, 2], [2, 3, 0], [3, 2, 6], [6, 7, 3], [7, 6, 5], [5, 4, 7],
         [4, 5, 1], [1, 0, 4], [5, 6, 2], [2, 1, 5], [7, 4, 0], [0, 3, 7]],
        dtype=tf.int32)

  def testRendersSimpleTriangle(self):
    """Directly renders a rasterized triangle's barycentric coordinates.

    Tests only the kernel (rasterize_triangles_module).
    """
    ndc_init = np.array(
        [[-0.5, -0.5, 0.8], [0.0, 0.5, 0.3], [0.5, -0.5, 0.3]],
        dtype=np.float32)

    image_height = 480
    image_width = 640

    normalized_device_coordinates = tf.constant(ndc_init)
    triangles = tf.constant([[0, 1, 2]], dtype=tf.int32)

    rendered_coordinates, _, _ = (
        rasterize_triangles.rasterize_triangles_module.rasterize_triangles(
            normalized_device_coordinates, triangles, image_width,
            image_height))
    rendered_coordinates = tf.concat(
        [rendered_coordinates,
         tf.ones([image_height, image_width, 1])], axis=2)
    with self.test_session() as sess:
      image = rendered_coordinates.eval()
      target_image_name = 'Simple_Triangle.png'
      baseline_image_path = os.path.join(self.test_data_directory,
                                         target_image_name)
      test_utils.expect_image_file_and_render_are_near(
          self, sess, baseline_image_path, image)

  def testRendersSimpleCube(self):
    """Renders a simple cube to test the kernel and python wrapper."""

    tf_float = lambda x: tf.constant(x, dtype=tf.float32)
    # camera position:
    eye = tf_float([[2.0, 3.0, 6.0]])
    center = tf_float([[0.0, 0.0, 0.0]])
    world_up = tf_float([[0.0, 1.0, 0.0]])
    image_width = 640
    image_height = 480

    look_at = camera_utils.look_at(eye, center, world_up)
    perspective = camera_utils.perspective(image_width / image_height,
                                           tf_float([40.0]), tf_float([0.01]),
                                           tf_float([10.0]))

    vertex_rgb = (self.cube_vertex_positions * 0.5 + 0.5)
    vertex_rgba = tf.concat([vertex_rgb, tf.ones([8, 1])], axis=1)

    projection = tf.matmul(perspective, look_at)
    background_value = [0.0, 0.0, 0.0, 0.0]

    rendered = rasterize_triangles.rasterize_triangles(
        tf.expand_dims(self.cube_vertex_positions, axis=0),
        tf.expand_dims(vertex_rgba, axis=0), self.cube_triangles, projection,
        image_width, image_height, background_value)

    with self.test_session() as sess:
      image = sess.run(rendered, feed_dict={})[0,...]
      target_image_name = 'Unlit_Cube_0.png'
      baseline_image_path = os.path.join(self.test_data_directory,
                                         target_image_name)
      test_utils.expect_image_file_and_render_are_near(
          self, sess, baseline_image_path, image)

  def testSimpleTriangleGradientComputation(self):
    """Verifies the Jacobian matrix for a single pixel.

    The pixel is in the center of a triangle facing the camera. This makes it
    easy to check which entries of the Jacobian might not make sense without
    worrying about corner cases.
    """
    image_height = 480
    image_width = 640
    test_pixel_x = 325
    test_pixel_y = 245

    normalized_device_coordinates = tf.placeholder(tf.float32, shape=[3, 3])

    triangles = tf.constant([[0, 1, 2]], dtype=tf.int32)

    barycentric_coordinates, _, _ = (
        rasterize_triangles.rasterize_triangles_module.rasterize_triangles(
            normalized_device_coordinates, triangles, image_width,
            image_height))

    pixels_to_compare = barycentric_coordinates[
        test_pixel_y:test_pixel_y + 1, test_pixel_x:test_pixel_x + 1, :]

    with self.test_session():
      ndc_init = np.array(
          [[-0.5, -0.5, 0.8], [0.0, 0.5, 0.3], [0.5, -0.5, 0.3]],
          dtype=np.float32)
      theoretical, numerical = tf.test.compute_gradient(
          normalized_device_coordinates, (3, 3),
          pixels_to_compare, (1, 1, 3),
          x_init_value=ndc_init,
          delta=4e-2)
      jacobians_match, message = (
          test_utils.check_jacobians_are_nearly_equal(
              theoretical, numerical, 0.01, 0.0, True))
      self.assertTrue(jacobians_match, message)

  def testInternalRenderGradientComputation(self):
    """Isolates and verifies the Jacobian matrix for the custom kernel."""
    image_height = 21
    image_width = 28

    normalized_device_coordinates = tf.placeholder(tf.float32, shape=[8, 3])

    barycentric_coordinates, _, _ = (
        rasterize_triangles.rasterize_triangles_module.rasterize_triangles(
            normalized_device_coordinates, self.cube_triangles, image_width,
            image_height))

    with self.test_session():
      # Precomputed transformation of the simple cube to normalized device
      # coordinates, in order to isolate the rasterization gradient.
      # pyformat: disable
      ndc_init = np.array(
          [[-0.43889722, -0.53184521, 0.85293502],
           [-0.37635487, 0.22206162, 0.90555805],
           [-0.22849123, 0.76811147, 0.80993629],
           [-0.2805393, -0.14092168, 0.71602166],
           [0.18631913, -0.62634289, 0.88603103],
           [0.16183566, 0.08129397, 0.93020856],
           [0.44147962, 0.53497446, 0.85076219],
           [0.53008741, -0.31276882, 0.77620775]],
          dtype=np.float32)
      # pyformat: enable
      theoretical, numerical = tf.test.compute_gradient(
          normalized_device_coordinates, (8, 3),
          barycentric_coordinates, (image_height, image_width, 3),
          x_init_value=ndc_init,
          delta=4e-2)
      jacobians_match, message = (
          test_utils.check_jacobians_are_nearly_equal(
              theoretical, numerical, 0.01, 0.01))
      self.assertTrue(jacobians_match, message)


if __name__ == '__main__':
  tf.test.main()
