import argparse
import collections
import libtiff
import numpy
import sys

import quadtree


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

    args = parser.parse_args()
    world_file = _parse_esri_world_file(args.world_file)

    print "Loading tiff"
    im = _load_height_data(args.input_file)
    sphere_mapping = _SphereMapping.from_world_file(world_file,
                                                    (im.shape[1],
                                                     im.shape[0]))
    im = im[3200:5200:5, -2000::5]
    im = numpy.maximum(-10. * numpy.ones(im.shape), im)

    print "Offsetting heightmap due to earth curvature"
    sphere_mapping = sphere_mapping[3200:5200:5, -2000::5]
    im += sphere_mapping.gen_height_map(EARTH_RADIUS)

    print "Building quad tree"
    height_map = quadtree.HeightMap(im)

    print "Calculating visibility"
    eye_arg = _parse_eye_coords(args.eye_coords)
    eye_pixel = sphere_mapping.long_lat_to_pixel(
                    (eye_arg[1], eye_arg[0]))
    eye_point = numpy.array([list(eye_pixel) + [eye_arg[2]]]).T
    import pdb; pdb.set_trace()
    visible = height_map.get_visible(eye_point)

    from matplotlib import pyplot as plt
    #plt.ion()
    p = plt.imshow(im * visible, interpolation='nearest')
    #p.write_png("foo.png")
    plt.show()

if __name__ == '__main__':
    main()
