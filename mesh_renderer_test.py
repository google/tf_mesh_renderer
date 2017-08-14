"""Tests for mesh_renderer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

import numpy as np
import tensorflow as tf
from tf import transformations

from tensorflow.python.ops import gradient_checker
import mesh_renderer


def _euler_angles_to_rotation_matrix(euler_angles):
  """Computes a static-basis euler angle rotation.

  Args:
    euler_angles: 1D tensor containing the X, Y, and Z rotations in radians,
        respectively.

  Returns:
    A 3x3 column-major tensor containing the matrix stack Z_2*Y_1*X_0.
  """
  sr = tf.sin(euler_angles)
  cr = tf.cos(euler_angles)

  model_rotation = tf.stack(
      [
          cr[2] * cr[1], cr[2] * sr[1] * sr[0] - cr[0] * sr[2],
          sr[2] * sr[0] + cr[2] * cr[0] * sr[1], cr[1] * sr[2],
          cr[2] * cr[0] + sr[2] * sr[1] * sr[0],
          cr[0] * sr[2] * sr[1] - cr[2] * sr[0], -sr[1], cr[1] * sr[0],
          cr[1] * cr[0]
      ],
      axis=0)
  return tf.transpose(tf.reshape(model_rotation, [3, 3]))


def _check_jacobians_are_nearly_equal(theoretical,
                                      numerical,
                                      outlier_relative_error_threshold,
                                      max_outlier_fraction,
                                      include_jacobians_in_error_message=False):
  """Compares two Jacobian matrices, allowing for some fraction of outliers.

  Args:
    theoretical: 2D numpy array containing a Jacobian matrix with entries
        computed via gradient functions. The layout should be as in the output
        of gradient_checker.
    numerical: 2D numpy array of the same shape as theoretical containing a
        Jacobian matrix with entries computed via finite difference
        approximations. The layout should be as in the output
        of gradient_checker.
    outlier_relative_error_threshold: float prescribing the maximum relative
        error (from the finite difference approximation) is tolerated before
        and entry is considered an outlier.
    max_outlier_fraction: float defining the maximum fraction of entries in
        theoretical that may be outliers before the check returns False.
    include_jacobians_in_error_message: bool defining whether the jacobian
        matrices should be included in the return message should the test fail.

  Returns:
    A tuple where the first entry is a boolean describing whether
    max_outlier_fraction was exceeded, and where the second entry is a string
    containing an error message if one is relevant.
  """
  outlier_gradients = np.abs(
      numerical - theoretical) / numerical > outlier_relative_error_threshold
  outlier_fraction = np.count_nonzero(outlier_gradients) / np.prod(
      numerical.shape[:2])
  jacobians_match = outlier_fraction <= max_outlier_fraction

  message = (
      ' %f of theoretical gradients are relative outliers, but the maximum'
      ' allowable fraction is %f ' % (outlier_fraction, max_outlier_fraction))
  if include_jacobians_in_error_message:
    # the gradient_checker convention is the typical Jacobian transposed:
    message += ('\nNumerical Jacobian:\n%s\nTheoretical Jacobian:\n%s' %
                (repr(numerical.T), repr(theoretical.T)))
  return jacobians_match, message


def _get_pil_formatted_image(image):
  """Converts the output of a mesh_renderer call to a numpy array for PIL.

  Args:
    image: a 3D numpy array containing an image using the coordinate scheme
        of mesh_renderer and containing RGBA values in the [0,1] range.

  Returns:
    A 3D numpy array suitable for input to PilImage.fromarray().
  """
  return np.clip(255.0 * image, 0.0, 255.0).astype(np.uint8).copy(order='C')

def _expect_image_file_and_image_are_near(test,
		                          baseline_path,
		                          result_image_bytes_or_numpy,
		                          comparison_name,
		                          images_differ_message,
		                          max_outlier_fraction=0.005,
		                          pixel_error_threshold=0.04,
		                          resize_baseline_image=None):
  """Compares the input image bytes with an image on disk.

  The comparison is soft: the images are considered identical if fewer than
  max_outlier_fraction of the pixels differ by more than pixel_error_threshold
  of the full color value. If the images differ, the function writes the
  baseline and result images into the test's outputs directory.

  Uses ImagesAreNear for the actual comparison.

  Args:
    test: a python unit test instance.
    baseline_path: path to the reference image on disk.
    result_image_bytes_or_numpy: the result image, as either a bytes object
      or a numpy array.
    comparison_name: a string naming this comparison. Names outputs for
      viewing in sponge.
    images_differ_message: the test message to display if the images differ.
    max_outlier_fraction: fraction of pixels that may vary by more than the
      error threshold. 0.005 means 0.5% of pixels.
    pixel_error_threshold: pixel values are considered to differ if their
      difference exceeds this amount. Range is 0.0 - 1.0.
    resize_baseline_image: a (width, height) tuple giving a new size to apply
      to the baseline image, or None.
  """
  try:
    result_image = np.array(
        PilImage.open(io.BytesIO(result_image_bytes_or_numpy)))
  except IOError:
    result_image = result_image_bytes_or_numpy
  baseline_pil_image = PilImage.open(baseline_path)
  if resize_baseline_image:
    baseline_pil_image = baseline_pil_image.resize(resize_baseline_image,
                                                   PilImage.ANTIALIAS)
  baseline_image = np.array(baseline_pil_image)

  images_match, comparison_message = ImagesAreNear(
      baseline_image, result_image, max_outlier_fraction, pixel_error_threshold)

  if not images_match:
    outputs_dir = os.environ["TEST_UNDECLARED_OUTPUTS_DIR"]
    test.assertNotEmpty(outputs_dir)

    baseline_path_split = os.path.splitext(baseline_path)
    test.assertEqual(len(baseline_path_split), 2)
    baseline_format = baseline_path_split[1]
    image_mode = "RGB" if baseline_image.shape[2] == 3 else "RGBA"

    baseline_output_path = os.path.join(
        outputs_dir, comparison_name + "_baseline" + baseline_format)
    PilImage.fromarray(
        baseline_image, mode=image_mode).save(baseline_output_path)

    logging.info("Saving result in format %s %s", image_mode, baseline_format)

    result_output_path = os.path.join(
        outputs_dir, comparison_name + "_result" + baseline_format)
    PilImage.fromarray(result_image, mode=image_mode).save(result_output_path)

  test.assertEqual(baseline_image.shape, result_image.shape)
  test.assertTrue(images_match, msg=images_differ_message + comparison_message)


class RenderTest(tf.test.TestCase):

  def setUp(self):
    self.test_data_directory = 'test_data/'

    tf.reset_default_graph()
    # Set up a basic cube centered at the origin, with edges of length one
    # and vertex normals pointing outwards along the line from the origin to
    # the cube vertices:
    self.unit_edge_length_cube_at_origin_vertex_positions = tf.constant(
        [[-1, -1, 1], [-1, -1, -1], [-1, 1, -1], [-1, 1, 1], [1, -1, 1],
         [1, -1, -1], [1, 1, -1], [1, 1, 1]],
        dtype=tf.float32)
    self.unit_edge_length_cube_at_origin_normals = tf.nn.l2_normalize(
        self.unit_edge_length_cube_at_origin_vertex_positions, dim=1)
    self.unit_edge_length_cube_at_origin_triangles = tf.constant(
        [[0, 1, 2], [2, 3, 0], [3, 2, 6], [6, 7, 3], [7, 6, 5], [5, 4, 7],
         [4, 5, 1], [1, 0, 4], [5, 6, 2], [2, 1, 5], [7, 4, 0], [0, 3, 7]],
        dtype=tf.int32)

  def testRendersSimpleTriangle(self):
    """Directly renders a rasterized triangle's barycentric coordinates."""
    ndc_init = np.array(
        [[-0.5, -0.5, 0.8], [0.0, 0.5, 0.3], [0.5, -0.5, 0.3]],
        dtype=np.float32)

    image_height = 480
    image_width = 640

    normalized_device_coordinates = tf.constant(ndc_init)
    triangles = tf.constant([[0, 1, 2]], dtype=tf.int32)

    rendered_coordinates, _ = mesh_renderer.mesh_renderer_module.mesh_renderer(
        normalized_device_coordinates, triangles, image_width, image_height)
    rendered_coordinates = tf.concat(
        [rendered_coordinates,
         tf.ones([image_height, image_width, 1])], axis=2)
    with self.test_session():
      image = _get_pil_formatted_image(rendered_coordinates.eval())
      target_image_name = 'Simple_Triangle.png'
      baseline_image_path = os.path.join(self.test_data_directory,
          target_image_name)
      test_utils.ImageFileAndImageAreNear(
          self, baseline_image_path, image, target_image_name,
          '%s does not match.' % target_image_name)

  def testRendersSimpleCube(self):
    """Renders a simple cube to test the full forward pass.

    Verifies the functionality of both the custom kernel and the python wrapper.
    """

    # Model space coordinates:
    vertices_model_space = self.unit_edge_length_cube_at_origin_vertex_positions
    normals_model_space = self.unit_edge_length_cube_at_origin_normals

    # rotate the cube for the test:
    model_transform_1 = transformations.euler_matrix(
        math.radians(-20.0), 0.0, math.radians(60.0),
        axes='szyx').astype(np.float32)[:3, :3]

    model_transform_2 = transformations.euler_matrix(
        math.radians(45.0), math.radians(30.0), 0.0,
        axes='szyx').astype(np.float32)[:3, :3]
    model_transforms = np.stack([model_transform_1, model_transform_2])

    vertices_world_space = tf.matmul(
        tf.stack([vertices_model_space, vertices_model_space]),
        model_transforms,
        transpose_b=True)

    normals_world_space = tf.matmul(
        tf.stack([normals_model_space, normals_model_space]),
        model_transforms,
        transpose_b=True)

    triangles = self.unit_edge_length_cube_at_origin_triangles

    # camera position:
    eye = tf.constant([0.0, 0.0, 6.0], dtype=tf.float32)
    center = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
    world_up = tf.constant([0.0, 1.0, 0.0], dtype=tf.float32)
    image_width = 640
    image_height = 480
    light_positions = tf.constant([[[0.0, 0.0, 6.0]], [[0.0, 0.0, 6.0]]])
    light_intensities = tf.ones([2, 1, 3], dtype=tf.float32)
    vertex_diffuse_colors = tf.ones_like(vertices_world_space, dtype=tf.float32)

    rendered = mesh_renderer.mesh_renderer(
        vertices_world_space, triangles, normals_world_space,
        vertex_diffuse_colors, eye, center, world_up, light_positions,
        light_intensities, image_width, image_height)

    with self.test_session() as sess:
      images = sess.run(rendered, feed_dict={})
      for image_id in xrange(images.shape[0]):
        image = _get_pil_formatted_image(images[image_id, :, :, :])
        target_image_name = 'Simple_Cube_%i.png' % image_id
        baseline_image_path = os.path.join(self.test_data_directory,
            target_image_name)
        _expect_image_file_and_image_are_near(
            self, baseline_image_path, image, target_image_name,
            '%s does not match.' % target_image_name)

  def testComplexShading(self):
    """Tests specular highlights, colors, and multiple lights per image."""
    # Model space coordinates:
    vertices_model_space = self.unit_edge_length_cube_at_origin_vertex_positions
    normals_model_space = self.unit_edge_length_cube_at_origin_normals

    # rotate the cube for the test:
    model_transform_1 = transformations.euler_matrix(
        math.radians(-20.0),
        math.radians(30.0),
        math.radians(60.0),
        axes='szyx').astype(np.float32)[:3, :3]

    model_transform_2 = transformations.euler_matrix(
        math.radians(45.0),
        math.radians(20.0),
        math.radians(-18.0),
        axes='szyx').astype(np.float32)[:3, :3]
    model_transforms = np.stack([model_transform_1, model_transform_2])

    vertices_world_space = tf.matmul(
        tf.stack([vertices_model_space, vertices_model_space]),
        model_transforms,
        transpose_b=True)

    normals_world_space = tf.matmul(
        tf.stack([normals_model_space, normals_model_space]),
        model_transforms,
        transpose_b=True)

    triangles = self.unit_edge_length_cube_at_origin_triangles

    # camera position:
    eye = tf.constant([0.0, 0.0, 6.0], dtype=tf.float32)
    center = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
    world_up = tf.constant([0.0, 1.0, 0.0], dtype=tf.float32)
    image_width = 640
    image_height = 480
    light_positions = tf.constant([[[0.0, 0.0, 6.0], [1.0, 2.0, 6.0]],
                                   [[0.0, -2.0, 4.0], [1.0, 3.0, 4.0]]])
    light_intensities = tf.constant(
        [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], [[2.0, 0.0, 1.0], [0.0, 2.0,
                                                                1.0]]],
        dtype=tf.float32)
    # pyformat: disable
    vertex_diffuse_colors = tf.constant(2*[[[1.0, 0.0, 0.0],
                                            [0.0, 1.0, 0.0],
                                            [0.0, 0.0, 1.0],
                                            [1.0, 1.0, 1.0],
                                            [1.0, 1.0, 0.0],
                                            [1.0, 0.0, 1.0],
                                            [0.0, 1.0, 1.0],
                                            [0.5, 0.5, 0.5]]],
                                        dtype=tf.float32)
    vertex_specular_colors = tf.constant(2*[[[0.0, 1.0, 0.0],
                                             [0.0, 0.0, 1.0],
                                             [1.0, 1.0, 1.0],
                                             [1.0, 1.0, 0.0],
                                             [1.0, 0.0, 1.0],
                                             [0.0, 1.0, 1.0],
                                             [0.5, 0.5, 0.5],
                                             [1.0, 0.0, 0.0]]],
                                         dtype=tf.float32)
    # pyformat: enable
    shininess_coefficients = 6.0 * tf.ones([2, 8], dtype=tf.float32)
    ambient_color = tf.constant(
        [[0., 0., 0.], [0.1, 0.1, 0.2]], dtype=tf.float32)
    renders = mesh_renderer.mesh_renderer(
        vertices_world_space, triangles, normals_world_space,
        vertex_diffuse_colors, eye, center, world_up, light_positions,
        light_intensities, image_width, image_height, vertex_specular_colors,
        shininess_coefficients, ambient_color)
    tonemapped_renders = tf.concat(
        [
            mesh_renderer.tone_mapper(renders[:, :, :, 0:3], 0.7),
            renders[:, :, :, 3:4]
        ],
        axis=3)

    # Check that shininess coefficient broadcasting works by also rendering
    # with a scalar shininess coefficient, and ensuring the result is identical:
    broadcasted_renders = mesh_renderer.mesh_renderer(
        vertices_world_space, triangles, normals_world_space,
        vertex_diffuse_colors, eye, center, world_up, light_positions,
        light_intensities, image_width, image_height, vertex_specular_colors,
        6.0, ambient_color)
    tonemapped_broadcasted_renders = tf.concat(
        [
            mesh_renderer.tone_mapper(broadcasted_renders[:, :, :, 0:3], 0.7),
            broadcasted_renders[:, :, :, 3:4]
        ],
        axis=3)

    with self.test_session() as sess:
      images, broadcasted_images = sess.run(
          [tonemapped_renders, tonemapped_broadcasted_renders], feed_dict={})

      for image_id in xrange(images.shape[0]):
        image = _get_pil_formatted_image(images[image_id, :, :, :])
        broadcasted_image = _get_pil_formatted_image(
            broadcasted_images[image_id, :, :, :])
        target_image_name = 'Simple_Cube_%i.png' % (image_id + 3)
        baseline_image_path = os.path.join(self.test_data_directory,
            target_image_name)
        _expect_image_file_and_image_are_near(
            self, baseline_image_path, image, target_image_name,
            '%s does not match.' % target_image_name)
        _expect_image_file_and_image_are_near(
            self, baseline_image_path, broadcasted_image, target_image_name,
            '%s matches without broadcasting, but does not match with'
            'broadcasting.' % target_image_name)

  def testSimpleTriangleGradientComputation(self):
    """Verifies the Jacobian matrix for a single pixel.

    The pixel is in the center of a triangle facing the camera. This makes it
    easy to check which entries of the Jacobian might not make sense without
    worrying about corner cases.
    """
    image_height = 480
    image_width = 640
    test_pixel_x = 320
    test_pixel_y = 240

    normalized_device_coordinates = tf.placeholder(tf.float32, shape=[3, 3])

    triangles = tf.constant([[0, 1, 2]], dtype=tf.int32)

    barycentric_coordinates, _ = (
        mesh_renderer.mesh_renderer_module.mesh_renderer(
            normalized_device_coordinates, triangles, image_width,
            image_height))

    pixels_to_compare = barycentric_coordinates[
        test_pixel_y:test_pixel_y + 1, test_pixel_x:test_pixel_x + 1, :]

    with self.test_session():
      ndc_init = np.array(
          [[-0.5, -0.5, 0.8], [0.0, 0.5, 0.3], [0.5, -0.5, 0.3]],
          dtype=np.float32)
      theoretical, numerical = gradient_checker.compute_gradient(
          normalized_device_coordinates, (3, 3),
          pixels_to_compare, (1, 1, 3),
          x_init_value=ndc_init,
          delta=4e-2)
      jacobians_match, message = _check_jacobians_are_nearly_equal(
          theoretical, numerical, 0.3, 0.0, True)
      self.assertTrue(jacobians_match, message)

  def testInternalRenderGradientComputation(self):
    """Isolates and verifies the Jacobian matrix for the custom kernel."""
    image_height = 21
    image_width = 28

    normalized_device_coordinates = tf.placeholder(tf.float32, shape=[8, 3])

    triangles = self.unit_edge_length_cube_at_origin_triangles

    barycentric_coordinates, _ = (
        mesh_renderer.mesh_renderer_module.mesh_renderer(
            normalized_device_coordinates, triangles, image_width,
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
      theoretical, numerical = gradient_checker.compute_gradient(
          normalized_device_coordinates, (8, 3),
          barycentric_coordinates, (image_height, image_width, 3),
          x_init_value=ndc_init,
          delta=4e-2)
      jacobians_match, message = _check_jacobians_are_nearly_equal(
          theoretical, numerical, 0.3, 0.03)
      self.assertTrue(jacobians_match, message)

  def testFullRenderGradientComputation(self):
    """Verifies the Jacobian matrix for the entire renderer.

    This ensures correct gradients are propagated backwards through the entire
    process, not just through the rasterization kernel. Uses the simple cube
    forward pass.
    """
    image_height = 21
    image_width = 28

    # Model space coordinates:
    vertices_model_space = self.unit_edge_length_cube_at_origin_vertex_positions

    # The test normals are just pointing from the origin to the vertices in
    # model space, because the model space has the cube center as the origin.
    normals_model_space = self.unit_edge_length_cube_at_origin_normals

    # rotate the cube for the test:
    model_transform_1 = transformations.euler_matrix(
        math.radians(-20.0), 0.0, math.radians(60.0),
        axes='szyx').astype(np.float32)[:3, :3]

    model_transform_2 = transformations.euler_matrix(
        math.radians(45.0), math.radians(30.0), 0.0,
        axes='szyx').astype(np.float32)[:3, :3]
    model_transforms = np.stack([model_transform_1, model_transform_2])

    vertices_world_space = tf.matmul(
        tf.stack([vertices_model_space, vertices_model_space]),
        model_transforms,
        transpose_b=True)

    normals_world_space = tf.matmul(
        tf.stack([normals_model_space, normals_model_space]),
        model_transforms,
        transpose_b=True)

    triangles = self.unit_edge_length_cube_at_origin_triangles

    # camera position:
    eye = tf.constant([0.0, 0.0, 6.0], dtype=tf.float32)
    center = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
    world_up = tf.constant([0.0, 1.0, 0.0], dtype=tf.float32)

    # Scene has a single light from the viewer's eye.
    light_positions = tf.expand_dims(tf.stack([eye, eye], axis=0), axis=1)
    light_intensities = tf.ones([2, 1, 3], dtype=tf.float32)

    vertex_diffuse_colors = tf.ones_like(vertices_world_space, dtype=tf.float32)

    rendered = mesh_renderer.mesh_renderer(
        vertices_world_space, triangles, normals_world_space,
        vertex_diffuse_colors, eye, center, world_up, light_positions,
        light_intensities, image_width, image_height)

    with self.test_session():
      theoretical, numerical = gradient_checker.compute_gradient(
          vertices_model_space, (8, 3),
          rendered, (2, image_height, image_width, 4),
          x_init_value=vertices_model_space.eval(),
          delta=1e-3)
      jacobians_match, message = _check_jacobians_are_nearly_equal(
          theoretical, numerical, 0.3, 0.05)
      self.assertTrue(jacobians_match, message)

  def testThatCubeRotates(self):
    """Optimize a simple cube's rotation using pixel loss.

    The rotation is represented as static-basis euler angles. This test checks
    that the computed gradients are useful.
    """
    image_height = 480
    image_width = 640
    initial_euler_angles = [0.0, 40.0, 0.0]

    # Model space coordinates:
    vertices_model_space = self.unit_edge_length_cube_at_origin_vertex_positions

    # The test normals are just pointing from the origin to the vertices in
    # model space, because the model space has the cube center as the origin.
    normals_model_space = self.unit_edge_length_cube_at_origin_normals

    euler_angles = tf.Variable(initial_euler_angles)
    model_rotation = _euler_angles_to_rotation_matrix(euler_angles)

    vertices_world_space = tf.reshape(
        tf.matmul(vertices_model_space, model_rotation, transpose_b=True),
        [1, 8, 3])

    normals_world_space = tf.reshape(
        tf.matmul(normals_model_space, model_rotation, transpose_b=True),
        [1, 8, 3])

    triangles = self.unit_edge_length_cube_at_origin_triangles

    # camera position:
    eye = tf.constant([0.0, 0.0, 6.0], dtype=tf.float32)
    center = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
    world_up = tf.constant([0.0, 1.0, 0.0], dtype=tf.float32)

    vertex_diffuse_colors = tf.ones_like(vertices_world_space, dtype=tf.float32)
    light_positions = tf.reshape(eye, [1, 1, 3])
    light_intensities = tf.ones([1, 1, 3], dtype=tf.float32)

    render = mesh_renderer.mesh_renderer(
        vertices_world_space, triangles, normals_world_space,
        vertex_diffuse_colors, eye, center, world_up, light_positions,
        light_intensities, image_width, image_height)
    render = tf.reshape(render, [image_height, image_width, 4])

    # Pick the desired cube rotation for the test:
    test_model_angles = tf.reshape(
        tf.constant([-20.0, 0.0, 60.0], dtype=tf.float32), [3])
    test_model_rotation = _euler_angles_to_rotation_matrix(test_model_angles)

    desired_vertex_positions = tf.reshape(
        tf.matmul(vertices_model_space, test_model_rotation, transpose_b=True),
        [1, 8, 3])
    desired_normals = tf.reshape(
        tf.matmul(normals_model_space, test_model_rotation, transpose_b=True),
        [1, 8, 3])
    desired_render = mesh_renderer.mesh_renderer(
        desired_vertex_positions, triangles, desired_normals,
        vertex_diffuse_colors, eye, center, world_up, light_positions,
        light_intensities, image_width, image_height)
    desired_render = tf.reshape(desired_render, [image_height, image_width, 4])

    loss = tf.reduce_mean(tf.abs(render - desired_render))
    optimizer = tf.train.MomentumOptimizer(0.7, 0.1)
    grad = tf.gradients(loss, [euler_angles])
    grad, _ = tf.clip_by_global_norm(grad, 1.0)
    opt_func = optimizer.apply_gradients([(grad[0], euler_angles)])

    with tf.Session() as sess:
      sess.run(tf.initialize_all_variables())
      for _ in range(35):
        sess.run([loss, opt_func])
      viewable_final_image = _get_pil_formatted_image(render.eval())
      viewable_desired_image = _get_pil_formatted_image(desired_render.eval())

      target_image_name = 'Simple_Cube_2.png'
      baseline_image_path = os.path.join(self.test_data_directory,
          target_image_name)
      _expect_image_file_and_image_are_near(
          self, baseline_image_path, viewable_desired_image, target_image_name,
          '%s does not match.' % target_image_name)
      _expect_image_file_and_image_are_near(
          self,
          baseline_image_path,
          viewable_final_image,
          target_image_name,
          '%s does not match.' % target_image_name,
          max_outlier_fraction=0.01)


if __name__ == '__main__':
  tf.test.main()
