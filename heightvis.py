import argparse
import collections
import libtiff
import math
import numpy
import osgrid
import sys

import quadtree

from matplotlib import pyplot as plt

# Radius of the earth in metres (if the earth is modelled as sphere).
EARTH_RADIUS = 6371000

_EsriWorldFile = collections.namedtuple('_EsriWorldFile',
                                        ['x_pixel_size',
                                         'y_rotation',
                                         'x_rotation',
                                         'y_pixel_size',
                                         'top_left_longitude',
                                         'top_left_latitude'])

def _parse_esri_world_file(file_name):
    f = open(file_name)
    try:
        out = _EsriWorldFile(*(float(line.strip()) for line in f.readlines()))
    finally:
        f.close()

    return out


def _load_height_data(height_map_file):
    tif = libtiff.TIFF.open(height_map_file)
    return tif.read_image()


class _SphereMapping(object):
    """A mapping of coordinates of an image onto a latitude/longitude."""

    @classmethod
    def from_world_file(cls, world_file, image_dims):
        assert world_file.x_rotation == 0.0
        assert world_file.y_rotation == 0.0

        return cls((world_file.x_pixel_size,
                        world_file.y_pixel_size),
                   (world_file.top_left_longitude,
                        world_file.top_left_latitude),
                   image_dims)

    def __init__(self, pixel_size, top_left_long_lat, image_dims):
        self.pixel_size = pixel_size
        self.top_left_long_lat = top_left_long_lat
        self.image_dims = image_dims

    def pixel_to_long_lat(self, pix):
        """
        Return a long,lat pair for a given x,y pixel coordinate pair.

        """
        pix = tuple((self.image_dims[i] - pix[i] if pix[i] < 0 else pix[i])
                                                               for i in (0, 1))
        return tuple(self.top_left_long_lat[i] + pix[i] * self.pixel_size[i]
                                                               for i in (0, 1))

    def long_lat_to_pixel(self, long_lat):
        """
        Return a x,y pixel coordinate pair for a given long,lat pair.

        """
        return tuple((long_lat[i] - self.top_left_long_lat[i]) /
                            self.pixel_size[i] for i in (0, 1))

    def __getitem__(self, k):
        """
        Return a new _SphereMapping as if the input array had been thus
        subscripted.

        The subscript is a pair consisting of slices and/or integers. The first
        item refers to the Y axis, and the second item refers to the X axis.
        This is for consistency with array subscripts.
        
        """
        assert len(k) == 2

        k = (k[1], k[0])

        # Make sure the subscript is a slice
        k = tuple((slice(n, n + 1, None) if not isinstance(n, slice) else n)
                                                                    for n in k)

        stride = tuple(k[i].indices(self.image_dims[i])[2] for i in (0, 1))
        tl_pixel = tuple(k[i].indices(self.image_dims[i])[0] for i in (0, 1))
        br_pixel = tuple(k[i].indices(self.image_dims[i])[1] for i in (0, 1))
        img_dims = tuple((br_pixel[i] - tl_pixel[i]) // stride[i]
                                    for i in (0, 1))
        pix_size = tuple(self.pixel_size[i] * stride[i] for i in (0, 1))

        return _SphereMapping(
                          pix_size, self.pixel_to_long_lat(tl_pixel), img_dims)

    def gen_height_map(self, sphere_radius, origin=None):
        """
        Get a height map representing curvature over the region.

        Accept an optional origin, which is the long/lat where the height map
        will be 0. Other points on the height map will fall reduce in value the
        further they get from the origin.

        The value of a given point P is:

            sphere_radius * (cos(theta) - 1)

        Where theta is the angle between the point P and the origin.

        This is an approximatation of the curvature of the sphere which is
        valid for small angles.

        """

        # Obtain an array of pixels' longitudes.
        longs = numpy.repeat(numpy.array([numpy.arange(self.image_dims[0],
                                                        dtype=numpy.float64)]),
                              self.image_dims[1],
                              axis=0)
        longs *= self.pixel_size[0]
        longs += self.top_left_long_lat[0]

        # Also latitudes.
        lats = numpy.repeat(numpy.array([numpy.arange(self.image_dims[1],
                                                        dtype=numpy.float64)]),
                              self.image_dims[0],
                              axis=0).T
        lats *= self.pixel_size[1]
        lats += self.top_left_long_lat[1]

        # Combine the two. For a coordinate (x, y), long_lats[:, y, x] should
        # equal array(self.pixel_to_long_lat((x, y)).
        long_lats = numpy.array([longs, lats])

        # Convert to radians...
        long_lats *= numpy.pi / 180.

        # Plug these values into the spherical law of cosines to obtain the
        # cosine of the angle from the origin. (For the purposes of these
        # comments refer to the angle as theta.)
        if origin is None:
            origin_coords = (self.image_dims[1] // 2,
                             self.image_dims[0] // 2)
        else:
            origin_coords = self.long_lat_to_pixel(origin)
            origin_coords = origin_coords[1], origin_coords[0]
         
        long_diffs = long_lats[0] - long_lats[0][origin_coords]
        cos_angles = (numpy.sin(long_lats[1][origin_coords]) *
                            numpy.sin(long_lats[1, :, :]) +
                      numpy.cos(long_lats[1][origin_coords]) *
                            numpy.cos(long_lats[1, :, :]) *
                            numpy.cos(long_diffs))

        # The height map is then r * (cos(theta) - 1).
        height_map = numpy.array(sphere_radius * (cos_angles - 1.),
                           dtype=numpy.float32)

        return height_map


def _parse_eye_coords(s):
    out = tuple(float(x) for x in s.split())

    if len(out) != 3:
        raise Exception("Invalid eye-coords {!r}".format(s))

    return out


def _parse_lat_long(s):
    out = tuple(float(x) for x in s.split())

    if len(out) != 2:
        raise Exception("Invalid latitude/longitude {!r}".format(s))

    return out[1], out[0]


def _plot_data(visible, view_bounds, height_im, curve_im, sphere_mapping,
               eye_point, bgs=()):
    # Overhead visibility plot, including background data, if available. The
    # visibility info is converted into RGBA with an alpha channel.
    assert len(visible.shape) == 2, "visible should be a greyscale"
    alpha = visible * 0.5
    visible = numpy.array([numpy.zeros(visible.shape)] * 3 + [alpha])
    visible = numpy.transpose(visible, (1, 2, 0))
    fig = plt.figure()
    visible_ax = fig.add_subplot(121)
    for bg in bgs:
        print "Bg extent: {}".format(bg.extent)
        visible_ax.imshow(bg.im, extent=bg.extent)
    left_extent, top_extent = sphere_mapping.pixel_to_long_lat((0, 0))
    right_extent, bottom_extent = sphere_mapping.pixel_to_long_lat((-1, -1))
    visible_ax.imshow(visible,
                      interpolation='nearest',
                      #extent=(left_extent, right_extent,
                      #        bottom_extent, top_extent))
                      extent=view_bounds.to_extent())

    # Dummy data for the profile plot. The profile plot is a side on view
    # showing:
    #   - The terrain (including earth curvature).
    #   - Just earth curvature.
    #   - Line of sight from the eye-point to a piece of terrain.
    x = numpy.arange(0., 1., 0.001)
    y = numpy.array([math.sin(200. * math.pi * i) for i in x])
    profile_ax = fig.add_subplot(122)
    profile_ax.plot(x, y)


    def update_profile(long_lat):
        """Update the profile plot."""
        # Calculate the data for the height and earth curvature lines. Do this
        # by taking 1000 samples on a line between the click location and the
        # eye.
        start_x, start_y = map(float,
                               sphere_mapping.long_lat_to_pixel(long_lat))
        end_x, end_y = eye_point[0, 0], eye_point[1, 0]

        x_data = numpy.arange(0.0, 1.0, 0.001)
        height_y_data = []
        curve_y_data = []
        for k in x_data:
            x = int((1. - k) * start_x + k * end_x)
            y = int((1. - k) * start_y + k * end_y)
            height_y_data.append(height_im[y, x])
            curve_y_data.append(curve_im[y, x])

        profile_ax.clear()
        profile_ax.plot(x_data, numpy.array(height_y_data))
        profile_ax.plot(x_data, numpy.array(curve_y_data))

        # Draw line-of-sight.
        x_data = numpy.array([0., 1.])
        y_data = numpy.array([height_y_data[0], eye_point[2, 0]])
        profile_ax.plot(x_data, y_data)

        fig.canvas.draw()
        
    # Event handling glue.
    def onclick(event):
        print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(
                    event.button, event.x, event.y, event.xdata, event.ydata)
        if event.inaxes == visible_ax:
            update_profile((event.xdata, event.ydata))
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()


_LongLatRectBase = collections.namedtuple('_LongLatRectBase', ('nw', 'se'))
class _LongLatRect(_LongLatRectBase):
    """A rectangle defined in longitude / latitude coordinates."""


    def extend(self, long_lat):
        """Extend a view bounds to include a given point."""
    
        nw, se = list(self.nw), list(self.se)
        # Check western bound
        if long_lat[0] < nw[0]: 
            nw[0] = long_lat[0]
        # Check eastern bound
        if long_lat[0] > se[0]: 
            se[0] = long_lat[0]
        # Check southern bound
        if long_lat[1] < se[1]: 
            se[1] = long_lat[1]
        # Check northern bound
        if long_lat[1] > nw[1]: 
            nw[1] = long_lat[1]

        return _LongLatRect(tuple(nw), tuple(se))


    def to_pixels(self, sphere_mapping):
        """
        Return the bounds mapping to pixels, according to a sphere mapping.

        A pair (nw_pix, se_pix) is returned, which corresponds with the
        north-west and south-east pixel coordinates of the bounding rectangle.

        """
        nw_pix = tuple(map(int, sphere_mapping.long_lat_to_pixel(self.nw)))
        se_pix = tuple(map(int, sphere_mapping.long_lat_to_pixel(self.se)))

        return nw_pix, se_pix


    def to_extent(self):
        """Return a matplotlib compatible (left, right, bottom, top) tuple."""
        return (self.nw[0], self.se[0], self.se[1], self.nw[1])
            

_BackgroundImage = collections.namedtuple('_BackgroundImage',
                                          ('im', 'extent'))

def _get_view_bounds(args):
    """
    Determine the square region to view.
    
    Do this based on the --view-center and --view-size arguments. The returned
    value is a pair (nw, se), indicating the longitude/latitude of the
    north-west and the south-east points of the square, respectively.

    """
    center = _parse_lat_long(args.view_center)
    size = float(args.view_size)

    nw = center[0] - size / 2, center[1] + size / 2
    se = center[0] + size / 2, center[1] - size / 2

    return _LongLatRect(nw, se)


def _restrict_image_to_rect(im, view_bounds, sphere_mapping, sample_factor=1):
    """
    Restrict an image to a particular region.

    The region is determined by view_bounds, a rectangle defined in
    longitude/latitude coordinates. The sphere mapping argument is used to map
    long/lat coordinates to pixel coordinates. An option `sample_factor`
    will down-size the output image by the given factor.

    """

    nw_pix, se_pix = view_bounds.to_pixels(sphere_mapping)

    restricted_im = im[nw_pix[1]:se_pix[1] + 1:sample_factor,
                       nw_pix[0]:se_pix[0] + 1:sample_factor]

    return restricted_im


def _get_visible(args):
    """
    Return an image representing the visible part of the image.

    This calculation uses the --input-file, --view-{center,size}, --world-file,
    --eye-coords, and --scale-factor arguments.

    """
    view_bounds = _get_view_bounds(args)

    # Load the raw height data file.
    height_im = _load_height_data(args.input_file)

    # Obtain the sphere mapping, which maps pixel locations in the height map
    # to long/lat coordinates, and vice versa.
    world_file = _parse_esri_world_file(args.world_file)
    sphere_mapping = _SphereMapping.from_world_file(world_file,
                                                    (height_im.shape[1],
                                                     height_im.shape[0]))

    scale_factor = int(args.scale_factor) if args.scale_factor else 1
    height_im = height_im[::scale_factor, ::scale_factor]
    sphere_mapping = sphere_mapping[::scale_factor, ::scale_factor]

    # Obtain long/lat bounds for the height-map data based on the view bounds
    # extended to include the eye coordinate.
    eye_coords = _parse_eye_coords(args.eye_coords)
    eye_long_lat = eye_coords[1], eye_coords[0]
    eye_height = eye_coords[2]
    height_bounds = view_bounds.extend(eye_long_lat)
    del eye_coords

    # Restrict according to the height-map bounds. Update the sphere-mapping
    # accordingly, and clamp the minimum value (otherwise missing values are
    # mapped to -2**16).
    height_im = _restrict_image_to_rect(height_im,
                                        height_bounds,
                                        sphere_mapping)
    sphere_mapping = _SphereMapping(pixel_size=sphere_mapping.pixel_size,
                                    top_left_long_lat=height_bounds.nw,
                                    image_dims=(height_im.shape[1],
                                                height_im.shape[0]))
    height_im = numpy.maximum(-10. * numpy.ones(height_im.shape), height_im)

    # Offset the height map to account for curvature of the earth.
    #
    # The eye-point is used as the origin for the curve. This is to minimize
    # rays with a negative gradient from the terrain to the eye. Such rays are
    # bad as they are (almost certainly) going to intersect with the grid cell
    # being traced. A better solution for this would be to use bilinear
    # interpolation on the height-map, rather than nearest neighbour, however
    # this hack should get us most of the way there.
    curve_im = sphere_mapping.gen_height_map(EARTH_RADIUS,
                                             origin=eye_long_lat)
    height_im += curve_im

    # Calculate visibility across the view bounds.
    eye_pixel = sphere_mapping.long_lat_to_pixel(eye_long_lat)
    offset_eye_height = (height_im[int(eye_pixel[1]), int(eye_pixel[0])] +
                                                                    eye_height)
    eye_point = numpy.array([list(eye_pixel) + [offset_eye_height]]).T
    height_map = quadtree.HeightMap(height_im)
    visible = height_map.get_visible(
                                    eye_point,
                                    rect=view_bounds.to_pixels(sphere_mapping))

    return visible, view_bounds, height_im, curve_im, sphere_mapping, eye_point


def _load_os_bg(args, pixels_per_degree=3000):
    view_bounds = _get_view_bounds(args)

    tiled_os_map = osgrid.TiledOsMap(args.os_data)
    os_dims = (int(pixels_per_degree * float(args.view_size)),
               int(pixels_per_degree * float(args.view_size)))

    os_im = tiled_os_map.get_image_from_wgs84_rect(view_bounds, os_dims)
    os_bg = _BackgroundImage(im=os_im, extent=view_bounds.to_extent())

    return os_bg


def main():
    parser = argparse.ArgumentParser(
        description='Determine line-of-sight visibility from a geo TIFF')
    parser.add_argument('--input-file', '-i',
                        help='Input TIFF image height data.',
                        required=True)
    parser.add_argument('--scale-factor', '-f',
                        help='Scale the input height down by this (integer) '
                             'factor')
    parser.add_argument('--world-file', '-w', 
                        help='Input ESRI world file (.tfw) for the height '
                             'data',
                        required=True)
    parser.add_argument('--eye-coords', '-e',
                        help='Space separated latitude, longitude and height '
                        'in metres, all in decimal format. Specifies the '
                        'viewpoint',
                       required=True) 
    parser.add_argument('-o', '--os-data',
                        help='OS map tiled zip file')
    parser.add_argument('-c', '--view-center',
                        help='Center of the square region to be viewed as a '
                             'space separated latitude/longitude in degrees.',
                        required=True)
    parser.add_argument('-s', '--view-size',
                        help='Size of the square region to be viewed, in '
                             'degrees.',
                        required=True)

    args = parser.parse_args()

    if args.os_data:
        os_bg = _load_os_bg(args)

    visible, view_bounds, height_im, curve_im, sphere_mapping, eye_point = (
                                                            _get_visible(args))
    
    bgs = (os_bg,) if args.os_data else ()
    _plot_data(visible, view_bounds, height_im, curve_im, sphere_mapping,
               eye_point, bgs=bgs)


if __name__ == '__main__':
    main()
