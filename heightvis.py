import argparse
import collections
import libtiff
import numpy
import sys

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

    def __getitem__(self, k):
        """
        Return a new _SphereMapping as if the input array had been thus
        subscripted.

        """
        assert len(k) == 2

        # Make sure the subscript is a slice
        k = tuple((slice(n, n+1, None) if not isinstance(n, slice) else n)
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
        Get a height map with representing curvature over the region.

        """
        # Obtain some useful values used in the calculations below. Angles are
        # converted to radians for use with trig functions.
        centre_coords = (self.image_size[1] // 2,
                         self.image_size[0] // 2)
        centre_long_lat = pixel_to_long_lat((centre_coords[1],
                                             centre_coords[0]))
        centre_long_lat = tuple(x * math.pi / 180. for x in centre_long_lat)
        tl_long_lat = tuple(x * math.pi / 180. for x in self.top_left_long_lat)

        # Generate an array which represents angular distance from the line of
        # longitude that runs through the centre of the image.
        #
        # In actuality this will vary across rows, but for small areas away
        # from the poles duplicating rows is accurate enough.
        x_offs = numpy.repeat(numpy.array([numpy.arange(self.image_dims[0])]),
                              self.image_dims[1],
                              axis=0)
        x_offs -= x_offs[centre_coords] * numpy.ones(x_offs.shape) 
        x_offs *= 2. * (centre_long_lat[0] - tl_long_lat[0]) / (
                    math.cos(centre_long_lat[1]) * x_offs[centre_coords[0], 0])

        # Similarly generate an array which represents angular distance from
        # the line of latitude that runs through the centre of the image.
        y_offs = numpy.repeat(numpy.array([numpy.arange(self.image_dims[1])]),
                              self.image_dims[0],
                              axis=0).T
        y_offs -= y_offs[centre_coords] * numpy.ones(y_offs.shape) 
        y_offs *= 2. * (centre_long_lat[1] - tl_long_lat[1]) / (
                                                y_offs[0, centre_coords[1]])
        
        # From the `x_offs` and `y_offs` obtain an angular distance map from
        # the centre of the image. Use the euclidean norm to approximate this.
        dist_map = numpy.lingalg.norm(numpy.array([x_offs, y_offs]), axis=0)

        # With the angular distance map, take the cosine to determine the
        # required height map.
        height_map = sphere_radius * (numpy.cos(dist_map) -
                                                    numpy.ones(dist_map.shape))

        return height_map
        
def _parse_eye_coords(s):
    out = tuple(float(x) for x in s.split())

    if len(out) != 2:
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
                       required=False) 

    args = parser.parse_args()
    world_file = _parse_esri_world_file(args.world_file)
    im = _load_height_data(args.input_file)

    sphere_mapping = _SphereMapping.from_world_file(world_file,
                                                    (im.shape[1],
                                                     im.shape[0]))
    im = im[3200:5200:2, -2000::2]
    im = numpy.maximum(-10. * numpy.ones(im.shape), im)

    #sphere_mapping = sphere_mapping[3200:5200:2, -2000::2]

    from matplotlib import pyplot as plt
    plt.ion()
    p = plt.imshow(im, interpolation='nearest')
    p.write_png("foo.png")
    plt.show()

if __name__ == '__main__':
    main()
