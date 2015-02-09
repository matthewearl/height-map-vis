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

    def gen_height_map(self, sphere_radius):
        """
        Get a height map representing curvature over the region.

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
        # cosine of the angle from the point in the centre of the image. (For
        # the purposes of these comments refer to the angle as theta.)
        centre_coords = (self.image_dims[1] // 2,
                         self.image_dims[0] // 2)
        long_diffs = long_lats[0] - long_lats[0][centre_coords]
        cos_angles = (numpy.sin(long_lats[1][centre_coords]) *
                            numpy.sin(long_lats[1, :, :]) +
                      numpy.cos(long_lats[1][centre_coords]) *
                            numpy.cos(long_lats[1, :, :]) *
                            numpy.cos(long_diffs))

        # The height map is then r * (cos(theta) - 1).
        height_map = numpy.array(sphere_radius * (cos_angles - 1.),
                           dtype=numpy.float32)

        return height_map

def _parse_eye_coords(s):
    out = tuple(float(x) for x in s.split())

    if len(out) != 3:
        raise Exception("Invalid eye-coords argument {!r}".format(s))

    return out

def _plot_data(visible, height_im, curve_im, sphere_mapping, eye_point,
               bgs=()):
    # Overhead visibility plot, including background data, if available. The
    # visibility info is converted into RGBA with an alpha channel.
    assert len(visible.shape) == 2, "visible should be a greyscale"
    alpha = visible * 0.5
    visible = numpy.array([numpy.zeros(visible.shape)] * 3 + [alpha])
    visible = numpy.transpose(visible, (1, 2, 0))
    left_extent, top_extent = sphere_mapping.pixel_to_long_lat((0, 0))
    right_extent, bottom_extent = sphere_mapping.pixel_to_long_lat((-1, -1))
    fig = plt.figure()
    visible_ax = fig.add_subplot(121)
    for bg in bgs:
        visible_ax.imshow(bg.im, extent=bg.extent)
    visible_ax.imshow(visible,
                      interpolation='nearest',
                      extent=(left_extent, right_extent,
                              bottom_extent, top_extent))

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

_BackgroundImage = collections.namedtuple('_BackgroundImage',
                                          ('im', 'extent'))

def main():
    parser = argparse.ArgumentParser(
        description='Determine line-of-sight visibility from a geo TIFF')
    parser.add_argument('--input-file', '-i',
                        help='Input TIFF image',
                        required=True)
    parser.add_argument('--world-file', '-w', 
                        help='Input ESRI world file (.tfw)',
                        required=True)
    parser.add_argument('--eye-coords', '-e',
                        help='Space separated latitude, longitude and height '
                        'in metres, all in decimal format. Specifies the '
                        'viewpoint',
                       required=True) 
    parser.add_argument('-o', '--os-data',
                        help='OS map tiled zip file')

    args = parser.parse_args()
    world_file = _parse_esri_world_file(args.world_file)

    if args.os_data:
        print "Loading OS map zip"
        tiled_os_map = osgrid.TiledOsMap(args.os_data)
        centre = (-0.35404, 51.818051)
        size = (0.5, 0.5)
        os_dims = (1000, 1000)
        os_extent = (centre[0] - size[0]/2., centre[0] + size[0]/2.,
                     centre[1] - size[1]/2., centre[1] + size[1]/2.)
        os_im = tiled_os_map.get_image_from_wgs84_rect(
                                                  (os_extent[0], os_extent[3]),
                                                  (os_extent[1], os_extent[2]),
                                                  os_dims)
        os_bg = _BackgroundImage(im=os_im, extent=os_extent)

    print "Loading tiff"
    height_im = _load_height_data(args.input_file)
    sphere_mapping = _SphereMapping.from_world_file(world_file,
                                                    (height_im.shape[1],
                                                     height_im.shape[0]))
    height_im = height_im[3200:5200:20, -2000::20]
    height_im = numpy.maximum(-10. * numpy.ones(height_im.shape), height_im)

    print "Offsetting heightmap due to earth curvature"
    sphere_mapping = sphere_mapping[3200:5200:20, -2000::20]
    curve_im = sphere_mapping.gen_height_map(EARTH_RADIUS)
    height_im += curve_im

    print "Building quad tree"
    height_map = quadtree.HeightMap(height_im)

    print "Calculating visibility"
    eye_arg = _parse_eye_coords(args.eye_coords)
    eye_pixel = sphere_mapping.long_lat_to_pixel(
                    (eye_arg[1], eye_arg[0]))
    eye_height = height_im[int(eye_pixel[1]),
                           int(eye_pixel[0])] + eye_arg[2]
    print "Eye is at {}".format(eye_pixel)
    eye_point = numpy.array([list(eye_pixel) + [eye_height]]).T
    visible = height_map.get_visible(eye_point)

    bgs = (os_bg,) if args.os_data else ()
    _plot_data(visible, height_im, curve_im, sphere_mapping, eye_point,
               bgs=bgs)

if __name__ == '__main__':
    main()
