"""Differentiable 3-D rendering of a textured triangle mesh."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import google3
import tensorflow.google as tf
from google3.research.vision.facedecoder.ops.kernels import gen_mesh_renderer as mesh_renderer_module
from google3.research.vision.facedecoder.ops.kernels import gen_mesh_renderer_grad as mesh_renderer_grad_module

# This epsilon should be smaller than any valid barycentric reweighting factor
# (i.e. the per-pixel reweighting factor used to correct for the effects of
# perspective-incorrect barycentric interpolation). It is necessary primarily
# because the reweighting factor will be 0 for factors outside the mesh, and we
# need to ensure the image color and gradient outside the region of the mesh are
# 0.
_MINIMUM_REWEIGHTING_THRESHOLD = 1e-6

# This epsilon is the minimum absolute value of a homogenous coordinate before
# it is clipped. It should be sufficiently large such that the output of
# the perspective divide step with this denominator still has good working
# precision with 32 bit arithmetic, and sufficiently small so that in practice
# vertices are almost never close enough to a clipping plane to be thresholded.
_MINIMUM_PERSPECTIVE_DIVIDE_THRESHOLD = 1e-6


def _compute_perspective_transform(image_width, image_height, fov_y, near_clip,
                                   far_clip):
  """Computes perspective transformation matrix.

  Switches to left handed coordinate system for NDC at the same time.

  Args:
    image_width: int specifying desired output image width in pixels.
    image_height: int specifying desired output image height in pixels.
    fov_y: float specifying desired output image y field of view.
    near_clip: float specifying near clipping plane distance.
    far_clip: float specifying far clipping plane distance.

  Returns:
    A 4x4 float tensor that maps from right-handed points in eye space to left-
    handed points in clip space.
  """
  aspect_ratio = image_width / image_height
  focal_length_y = 1.0 / math.tan(math.radians(fov_y) / 2.0)
  depth_range = far_clip - near_clip
  p_22 = -(far_clip + near_clip) / depth_range
  p_23 = -2.0 * far_clip * near_clip / depth_range

  perspective_transform = tf.constant(
      [[focal_length_y / aspect_ratio, 0, 0, 0], [0, focal_length_y, 0, 0],
       [0, 0, p_22, p_23], [0, 0, -1, 0]],
      dtype=tf.float32)
  return perspective_transform


def look_at(eye, center, world_up):
  """Computes a camera extrinsics matrix.

  Functionality mimes gluLookAt (glu/include/GLU/glu.h).

  Args:
    eye: 1-D float32 tensor with shape [3] containing the XYZ world space
        position of the camera.
    center: 1-D float32 tensor with shape [3] containing a position along the
        center of the camera's gaze.
    world_up: 1-D float32 tensor with shape [3] specifying the world's up
        direction; the output camera will have no tilt with respect to this
        direction.

  Returns:
    A 4x4 numpy array containing a right-handed camera extrinsics matrix that
    maps points from world space to points in eye space.

  Raises:
    InvalidArgumentError: The arguments specify a degenerate extrinsics matrix.
  """
  vector_degeneracy_cutoff = 1e-6
  forward = center - eye
  forward_norm = tf.norm(forward, ord='euclidean')
  tf.assert_greater(
      forward_norm,
      vector_degeneracy_cutoff,
      message='Camera matrix is degenerate because eye and center are close.')
  forward = tf.divide(forward, forward_norm)

  to_side = tf.cross(forward, world_up)
  to_side_norm = tf.norm(to_side, ord='euclidean')
  tf.assert_greater(
      to_side_norm,
      vector_degeneracy_cutoff,
      message='Camera matrix is degenerate because up and gaze are close or'
      'because up is degenerate.')
  to_side = tf.divide(to_side, to_side_norm)
  cam_up = tf.cross(to_side, forward)

  view_rotation = tf.reshape(
      tf.stack(
          [
              to_side[0], to_side[1], to_side[2], 0, cam_up[0], cam_up[1],
              cam_up[2], 0, -forward[0], -forward[1], -forward[2], 0, 0, 0, 0, 1
          ],
          axis=0), [4, 4])
  # pyformat: disable
  view_translation = tf.reshape(tf.stack([1., 0., 0., -eye[0],
                                          0., 1., 0., -eye[1],
                                          0., 0., 1., -eye[2],
                                          0., 0., 0., 1.], axis=0), [4, 4])
  # pyformat: enable
  camera_matrix = tf.matmul(view_rotation, view_translation)
  return camera_matrix


def rasterizer(vertices,
               vertex_attributes,
               triangles,
               camera_matrix,
               image_width,
               image_height,
               fov_y=40.0,
               near_clip=0.01,
               far_clip=10.0):
  """Rasterizes the input scene and computes interpolated vertex attributes.

  Args:
    vertices: 3-D float32 tensor with shape [batch_size, vertex_count, 3]. Each
        triplet is an xyz position in world space.
    vertex_attributes: 3-D float32 tensor with shape [batch_size, vertex_count,
        attribute_channel_count]. Each attribute will be interpolated across the
        pixels using barycentric interpolation between the vertices of the
        pixel's triangle.
    triangles: 2-D int32 tensor with shape [triangle_count, 3]. Each triplet
        should contain vertex indices describing a triangle such that the
        triangle's normal points toward the viewer if the forward order of the
        triplet defines a clockwise winding of the vertices. Gradients with
        respect to this tensor are not available.
    camera_matrix: 2-D float tensor with shape [4,4] describing camera
        extrinsics.
    image_width: int specifying desired output image width in pixels.
    image_height: int specifying desired output image height in pixels.
    fov_y: float specifying desired output image y field of view in degrees.
    near_clip: float specifying near clipping plane distance.
    far_clip: float specifying far clipping plane distance.

  Returns:
    A tuple (alphas, pixel_attributes). The first, alphas, is a 3-D float32
    tensor with shape [batch_size, image_height, image_width] containing
    the alpha value at each pixel; it is approximately one for mesh pixels and
    0.0 for background pixels. The second, pixel_attributes, is a 4-D float32
    tensor with shape [batch_size, image_height, image_width,
    attribute_channel_count]. It contains the interpolated vertex attributes at
    num_pixel.

  Raises:
    ValueError: An invalid argument to the method is detected.
  """
  if not image_width > 0:
    raise ValueError('Image width must be > 0.')
  if not image_height > 0:
    raise ValueError('Image height must be > 0.')
  if not fov_y > 0.0:
    raise ValueError('Y field of view must be > 0.0.')
  if not near_clip > 0.0:
    raise ValueError('Near clipping plane must be > 0.0.')
  if not far_clip > near_clip:
    raise ValueError(
        'Far clipping plane must be greater than near clipping plane.')
  if len(vertices.shape) != 3:
    raise ValueError('The vertex buffer must be a 3D tensor.')

  batch_size = vertices.shape[0].value
  vertex_count = vertices.shape[1].value

  # We map the coordinates to normalized device coordinates before passing
  # the scene to the rendering kernel to keep as many ops in tensorflow as
  # possible.

  perspective_transform = _compute_perspective_transform(
      image_width, image_height, fov_y, near_clip, far_clip)

  clip_space_transform = tf.matmul(perspective_transform, camera_matrix)

  # Maps vertex object space tensor into ndc coordinates.
  flattened_vertices = tf.reshape(vertices, [-1, 3])
  homogeneous_coord = tf.ones(
      [flattened_vertices.shape[0], 1], dtype=tf.float32)
  vertices_homogeneous = tf.concat([flattened_vertices, homogeneous_coord], 1)

  # Vertices are given in row-major order, but the transformation pipeline is
  # column major:
  clip_space_points = tf.matmul(
      vertices_homogeneous, clip_space_transform, transpose_b=True)

  # Perspective divide, first thresholding the homogeneous coordinate to avoid
  # the possibility of NaNs:
  clip_space_points_xyz = clip_space_points[:, 0:3] * tf.sign(
      clip_space_points[:, 3:4])
  clip_space_points_w = tf.maximum(
      tf.abs(clip_space_points[:, 3:4]), _MINIMUM_PERSPECTIVE_DIVIDE_THRESHOLD)
  normalized_device_coordinates = clip_space_points_xyz / clip_space_points_w

  # Reshapes to render one image at a time:
  normalized_device_coordinates = tf.reshape(normalized_device_coordinates,
                                             [batch_size, -1, 3])

  per_image_uncorrected_barycentric_coordinates = []
  per_image_vertex_ids = []
  for im in xrange(vertices.shape[0]):
    barycentric_coordinates, triangle_ids = mesh_renderer_module.mesh_renderer(
        normalized_device_coordinates[im, :, :], triangles, image_width,
        image_height)
    per_image_uncorrected_barycentric_coordinates.append(
        tf.reshape(barycentric_coordinates, [-1, 3]))

    # Gathers the vertex indices now because the indices don't contain a batch
    # identifier, and reindexes the vertex ids to point to a (batch,vertex_id)
    vertex_ids = tf.gather(triangles, tf.reshape(triangle_ids, [-1]))
    reindexed_ids = tf.add(vertex_ids, im * vertices.shape[1].value)
    per_image_vertex_ids.append(reindexed_ids)

  uncorrected_barycentric_coordinates = tf.concat(
      per_image_uncorrected_barycentric_coordinates, axis=0)
  vertex_ids = tf.concat(per_image_vertex_ids, axis=0)

  # Indexes with each pixel's clip-space triangle's extrema (the pixel's
  # 'corner points') ids to get the relevant properties for deferred shading.
  flattened_vertex_attributes = tf.reshape(vertex_attributes,
                                           [batch_size * vertex_count, -1])
  corner_attributes = tf.gather(flattened_vertex_attributes, vertex_ids)

  # Barycentric interpolation is linear in the reciprocal of the homogeneous
  # W coordinate, so we use these weights to correct for the effects of
  # perspective distortion after rasterization.
  perspective_distortion_weights = tf.reciprocal(clip_space_points[:, 3])
  corner_distortion_weights = tf.gather(perspective_distortion_weights,
                                        vertex_ids)

  # Apply perspective correction to the barycentric coordinates. This step is
  # required since the rasterizer receives normalized-device coordinates (i.e.,
  # after perspective division), so it can't apply perspective correction to the
  # interpolated values.
  weighted_barycentric_coordinates = tf.multiply(
      uncorrected_barycentric_coordinates, corner_distortion_weights)
  barycentric_reweighting_factor = tf.reduce_sum(
      weighted_barycentric_coordinates, axis=1)

  corrected_barycentric_coordinates = tf.divide(
      weighted_barycentric_coordinates,
      tf.expand_dims(
          tf.maximum(barycentric_reweighting_factor,
                     _MINIMUM_REWEIGHTING_THRESHOLD),
          axis=1))

  # Computes the pixel attributes by interpolating the known attributes at the
  # corner points of the triangle interpolated with the barycentric coordinates.
  weighted_vertex_attributes = tf.multiply(
      corner_attributes,
      tf.expand_dims(corrected_barycentric_coordinates, axis=2))
  attributes = tf.reduce_sum(weighted_vertex_attributes, axis=1)
  attribute_images = tf.reshape(attributes,
                                [batch_size, image_height, image_width, -1])

  # Barycentric coordinates should approximately sum to one where there is
  # rendered geometry, but be exactly zero where there is not.
  alphas = tf.clip_by_value(
      tf.reduce_sum(2.0 * corrected_barycentric_coordinates, axis=1), 0.0, 1.0)

  return alphas, attribute_images


def phong_shader(normals,
                 alphas,
                 pixel_positions,
                 light_positions,
                 light_intensities,
                 diffuse_colors=None,
                 camera_position=None,
                 specular_colors=None,
                 shininess_coefficients=None,
                 ambient_color=None):
  """Computes pixelwise lighting from rasterized buffers with the Phong model.

  Args:
    normals: a 4D float32 tensor with shape [batch_size, image_height,
        image_width, 3]. The inner dimension is the world space XYZ normal for
        the corresponding pixel. Should be already normalized.
    alphas: a 3D float32 tensor with shape [batch_size, image_height,
        image_width]. The inner dimension is the alpha value (transparency)
        for the corresponding pixel.
    pixel_positions: a 4D float32 tensor with shape [batch_size, image_height,
        image_width, 3]. The inner dimension is the world space XYZ position for
        the corresponding pixel.
    light_positions: a 3D tensor with shape [batch_size, light_count, 3]. The
        XYZ position of each light in the scene. In the same coordinate space as
        pixel_positions.
    light_intensities: a 3D tensor with shape [batch_size, light_count, 3]. The
        RGB intensity values for each light. Intensities may be above one.
    diffuse_colors: a 4D float32 tensor with shape [batch_size, image_height,
        image_width, 3]. The inner dimension is the diffuse RGB coefficients at
        a pixel in the range [0, 1].
    camera_position: a 1D tensor with shape [3]. The XYZ camera position in the
        scene. If supplied, specular reflections will be computed. If not
        supplied, specular_colors and shininess_coefficients are expected to be
        None. In the same coordinate space as pixel_positions.
    specular_colors: a 4D float32 tensor with shape [batch_size, image_height,
        image_width, 3]. The inner dimension is the specular RGB coefficients at
        a pixel in the range [0, 1]. If None, assumed to be tf.zeros()
    shininess_coefficients: A 3D float32 tensor that is broadcasted to shape
        [batch_size, image_height, image_width]. The inner dimension is the
        shininess coefficient for the object at a pixel. Dimensions that are
        constant can be given length 1, so [batch_size, 1, 1] and [1, 1, 1] are
        also valid input shapes.
    ambient_color: a 2D tensor with shape [batch_size, 3]. The RGB ambient
        color, which is added to each pixel before tone mapping. If None, it is
        assumed to be tf.zeros().
  Returns:
    A 4D float32 tensor of shape [batch_size, image_height, image_width, 4]
    containing the lit RGBA color values for each image at each pixel. Colors
    are in the range [0,1].

  Raises:
    ValueError: An invalid argument to the method is detected.
  """
  batch_size, image_height, image_width = [s.value for s in normals.shape[:-1]]
  light_count = light_positions.shape[1].value
  pixel_count = image_height * image_width
  # Reshape all values to easily do pixelwise computations:
  normals = tf.reshape(normals, [batch_size, -1, 3])
  alphas = tf.reshape(alphas, [batch_size, -1, 1])
  diffuse_colors = tf.reshape(diffuse_colors, [batch_size, -1, 3])
  if camera_position is not None:
    specular_colors = tf.reshape(specular_colors, [batch_size, -1, 3])

  # Ambient component
  output_colors = tf.zeros([batch_size, image_height * image_width, 3])
  if ambient_color is not None:
    output_colors = tf.add(output_colors, tf.expand_dims(ambient_color, axis=1))

  # Diffuse component
  pixel_positions = tf.reshape(pixel_positions, [batch_size, -1, 3])
  per_light_pixel_positions = tf.stack(
      [pixel_positions] * light_count,
      axis=1)  # [batch_size, light_count, pixel_count, 3]
  directions_to_lights = tf.nn.l2_normalize(
      tf.expand_dims(light_positions, axis=2) - per_light_pixel_positions,
      dim=3)  # [batch_size, light_count, pixel_count, 3]
  # The specular component should only contribute when the light and normal
  # face one another (i.e. the dot product is nonnegative):
  normals_dot_lights = tf.clip_by_value(
      tf.reduce_sum(
          tf.expand_dims(normals, axis=1) * directions_to_lights, axis=3), 0.0,
      1.0)  # [batch_size, light_count, pixel_count]
  diffuse_output = tf.expand_dims(
      diffuse_colors, axis=1) * tf.expand_dims(
          normals_dot_lights, axis=3) * tf.expand_dims(
              light_intensities, axis=2)
  diffuse_output = tf.reduce_sum(
      diffuse_output, axis=1)  # [batch_size, pixel_count, 3]
  output_colors = tf.add(output_colors, diffuse_output)

  # Specular component
  if camera_position is not None:
    camera_position = tf.reshape(camera_position, [1, 1, 3])
    mirror_reflection_direction = tf.nn.l2_normalize(
        2.0 * tf.expand_dims(normals_dot_lights, axis=3) * tf.expand_dims(
            normals, axis=1) - directions_to_lights,
        dim=3)
    direction_to_camera = tf.nn.l2_normalize(
        camera_position - pixel_positions, dim=2)
    reflection_direction_dot_camera_direction = tf.reduce_sum(
        tf.expand_dims(direction_to_camera, axis=1) *
        mirror_reflection_direction,
        axis=3)
    # The specular component should only contribute when the reflection is
    # external:
    reflection_direction_dot_camera_direction = tf.clip_by_value(
        tf.nn.l2_normalize(reflection_direction_dot_camera_direction, dim=2),
        0.0, 1.0)
    # The specular component should also only contribute when the diffuse
    # component contributes:
    reflection_direction_dot_camera_direction = tf.where(
        normals_dot_lights != 0.0, reflection_direction_dot_camera_direction,
        tf.zeros_like(
            reflection_direction_dot_camera_direction, dtype=tf.float32))
    # Reshape to support broadcasting the shininess coefficient, which rarely
    # varies per-vertex:
    reflection_direction_dot_camera_direction = tf.reshape(
        reflection_direction_dot_camera_direction,
        [batch_size, light_count, image_height, image_width])
    shininess_coefficients = tf.expand_dims(shininess_coefficients, axis=1)
    specularity = tf.reshape(
        tf.pow(reflection_direction_dot_camera_direction,
               shininess_coefficients),
        [batch_size, light_count, pixel_count, 1])
    specular_output = tf.expand_dims(
        specular_colors, axis=1) * specularity * tf.expand_dims(
            light_intensities, axis=2)
    specular_output = tf.reduce_sum(specular_output, axis=1)
    output_colors = tf.add(output_colors, specular_output)
  rgb_images = tf.reshape(output_colors,
                          [batch_size, image_height, image_width, 3])
  alpha_images = tf.reshape(alphas, [batch_size, image_height, image_width, 1])
  valid_rgb_values = tf.concat(3 * [alpha_images > 0.5], axis=3)
  rgb_images = tf.where(valid_rgb_values, rgb_images,
                        tf.zeros_like(rgb_images, dtype=tf.float32))
  return tf.reverse(tf.concat([rgb_images, alpha_images], axis=3), axis=[1])


def tone_mapper(image, gamma):
  """Applies gamma correction to the input image.

  Tone maps the input image batch in order to make scenes with a high dynamic
  range viewable. The gamma correction factor is computed separately per image,
  but is shared between all provided channels. The exact function computed is:

  image_out = A*image_in^gamma, where A is an image-wide constant computed so
  that the maximum image value is approximately 1. The correction is applied
  to all channels.

  Args:
    image: 4-D float32 tensor with shape [batch_size, image_height,
        image_width, channel_count]. The batch of images to tone map.
    gamma: 0-D float32 nonnegative tensor. Values of gamma below one compress
        relative contrast in the image, and values above one increase it. A
        value of 1 is equivalent to scaling the image to have a maximum value
        of 1.
  Returns:
    4-D float32 tensor with shape [batch_size, image_height, image_width,
    channel_count]. Contains the gamma-corrected images, clipped to the range
    [0, 1].
  """
  batch_size = image.shape[0].value
  corrected_image = tf.pow(image, gamma)
  image_max = tf.reduce_max(
      tf.reshape(corrected_image, [batch_size, -1]), axis=1)
  scaled_image = tf.divide(corrected_image,
                           tf.reshape(image_max, [batch_size, 1, 1, 1]))
  return tf.clip_by_value(scaled_image, 0.0, 1.0)


def mesh_renderer(vertices,
                  triangles,
                  normals,
                  diffuse_colors,
                  camera_position,
                  camera_lookat,
                  camera_up,
                  light_positions,
                  light_intensities,
                  image_width,
                  image_height,
                  specular_colors=None,
                  shininess_coefficients=None,
                  ambient_color=None,
                  fov_y=40.0,
                  near_clip=0.01,
                  far_clip=10.0):
  """Renders an input scene using phong shading, and returns an output image.

  Args:
    vertices: 3-D float32 tensor with shape [batch_size, vertex_count, 3]. Each
        triplet is an xyz position in world space.
    triangles: 2-D int32 tensor with shape [triangle_count, 3]. Each triplet
        should contain vertex indices describing a triangle such that the
        triangle's normal points toward the viewer if the forward order of the
        triplet defines a clockwise winding of the vertices. Gradients with
        respect to this tensor are not available.
    normals: 3-D float32 tensor with shape [batch_size, vertex_count, 3]. Each
        triplet is the xyz vertex normal for its corresponding vertex. Each
        vector is assumed to be already normalized.
    diffuse_colors: 3-D float32 tensor with shape [batch_size,
        vertex_count, 3]. The RGB diffuse reflection in the range [0,1] for
        each vertex.
    camera_position: 1-D tensor with shape [3] specifying the XYZ world space
        camera position.
    camera_lookat: 1-D tensor with shape [3] containing an XYZ point along the
        center of the camera's gaze.
    camera_up: 1-D tensor with shape [3] containing the up direction for the
        camera. The camera will have no tilt with respect to this direction.
    light_positions: a 3-D tensor with shape [batch_size, light_count, 3]. The
        XYZ position of each light in the scene. In the same coordinate space as
        pixel_positions.
    light_intensities: a 3-D tensor with shape [batch_size, light_count, 3]. The
        RGB intensity values for each light. Intensities may be above one.
    image_width: int specifying desired output image width in pixels.
    image_height: int specifying desired output image height in pixels.
    specular_colors: 3-D float32 tensor with shape [batch_size,
        vertex_count, 3]. The RGB specular reflection in the range [0, 1] for
        each vertex.  If supplied, specular reflections will be computed, and
        both specular_colors and shininess_coefficients are expected.
    shininess_coefficients: a 0D-2D float32 tensor with maximum shape
       [batch_size, vertex_count]. The phong shininess coefficient of each
       vertex. A 0D tensor or float gives a constant shininess coefficient
       across all batches and images. A 1D tensor must have shape [batch_size],
       and a single shininess coefficient per image is used.
    ambient_color: a 2D tensor with shape [batch_size, 3]. The RGB ambient
        color, which is added to each pixel in the scene. If None, it is
        assumed to be black.
    fov_y: float specifying desired output image y field of view in degrees.
    near_clip: float specifying near clipping plane distance.
    far_clip: float specifying far clipping plane distance.

  Returns:
    A 4-D float32 tensor of shape [batch_size, image_height, image_width, 4]
    containing the lit RGBA color values for each image at each pixel. RGB
    colors are the intensity values before tonemapping and can be in the range
    [0, infinity]. Clipping to the range [0,1] with tf.clip_by_value is likely
    reasonable for both viewing and training most scenes. More complex scenes
    with multiple lights should tone map color values for display only. One
    simple tonemapping approach is to rescale color values as x/(1+x); gamma
    compression is another common techinque. Alpha values are zero for
    background pixels and near one for mesh pixels.
  Raises:
    ValueError: An invalid argument to the method is detected.
  """
  if len(vertices.shape) != 3:
    raise ValueError('Vertices must have shape [batch_size, vertex_count, 3].')
  if len(normals.shape) != 3:
    raise ValueError('Normals must have shape [batch_size, vertex_count, 3].')
  if len(light_positions.shape) != 3:
    raise ValueError(
        'Light_positions must have shape [batch_size, light_count, 3].')
  if len(light_intensities.shape) != 3:
    raise ValueError(
        'Light_intensities must have shape [batch_size, light_count, 3].')
  if len(diffuse_colors.shape) != 3:
    raise ValueError(
        'vertex_diffuse_colors must have shape [batch_size, vertex_count, 3].')
  if ambient_color is not None and len(ambient_color.shape) != 2:
    raise ValueError('Ambient_color must have shape [batch_size, 3].')
  if camera_position.get_shape().as_list() != [3]:
    raise ValueError('Camera_position must have shape [3]')
  if camera_lookat.get_shape().as_list() != [3]:
    raise ValueError('Camera_lookat must have shape [3]')
  if camera_up.get_shape().as_list() != [3]:
    raise ValueError('Camera_up must have shape [3]')
  if specular_colors is not None and shininess_coefficients is None:
    raise ValueError(
        'Specular colors were supplied without shininess coefficients.')
  if shininess_coefficients is not None and specular_colors is None:
    raise ValueError(
        'Shininess coefficients were supplied without specular colors.')
  if specular_colors is not None:
    # Since a 0-D float32 tensor is accepted, also accept a float.
    if isinstance(shininess_coefficients, float):
      shininess_coefficients = tf.constant(
          shininess_coefficients, dtype=tf.float32)
    if len(specular_colors.shape) != 3:
      raise ValueError('The specular colors must have shape [batch_size, '
                       'vertex_count, 3].')
    if len(shininess_coefficients.shape) > 2:
      raise ValueError('The shininess coefficients must have shape at most'
                       '[batch_size, vertex_count].')
    # If we don't have per-vertex coefficients, we can just reshape the
    # input shininess to broadcast later, rather than interpolating an
    # additional vertex attribute:
    if len(shininess_coefficients.shape) < 2:
      vertex_attributes = tf.concat(
          [normals, vertices, diffuse_colors, specular_colors], axis=2)
    else:
      vertex_attributes = tf.concat(
          [
              normals, vertices, diffuse_colors, specular_colors,
              tf.expand_dims(shininess_coefficients, axis=2)
          ],
          axis=2)
  else:
    vertex_attributes = tf.concat([normals, vertices, diffuse_colors], axis=2)

  camera_matrix = look_at(camera_position, camera_lookat, camera_up)

  alphas, pixel_attributes = rasterizer(
      vertices,
      vertex_attributes,
      triangles,
      camera_matrix,
      image_width,
      image_height,
      fov_y=fov_y,
      near_clip=near_clip,
      far_clip=far_clip)

  # Extract the interpolated vertex attributes from the pixel buffer and
  # supply them to the shader:
  pixel_normals = tf.nn.l2_normalize(pixel_attributes[:, :, :, 0:3], dim=3)
  pixel_positions = pixel_attributes[:, :, :, 3:6]
  diffuse_colors = pixel_attributes[:, :, :, 6:9]
  if specular_colors is not None:
    specular_colors = pixel_attributes[:, :, :, 9:12]
    # Retrieve the interpolated shininess coefficients if necessary, or just
    # reshape our input for broadcasting:
    if len(shininess_coefficients.shape) == 2:
      shininess_coefficients = pixel_attributes[:, :, :, 12]
    else:
      shininess_coefficients = tf.reshape(shininess_coefficients, [-1, 1, 1])

  renders = phong_shader(
      normals=pixel_normals,
      alphas=alphas,
      pixel_positions=pixel_positions,
      light_positions=light_positions,
      light_intensities=light_intensities,
      diffuse_colors=diffuse_colors,
      camera_position=camera_position if specular_colors is not None else None,
      specular_colors=specular_colors,
      shininess_coefficients=shininess_coefficients,
      ambient_color=ambient_color)
  return renders


@tf.RegisterGradient('MeshRenderer')
def _mesh_renderer_grad(op, df_dbarys, _):
  return mesh_renderer_grad_module.mesh_renderer_grad(
      op.inputs[0], op.inputs[1], op.outputs[0], op.outputs[1], df_dbarys,
      op.get_attr('image_width'), op.get_attr('image_height')), None
