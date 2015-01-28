import argparse
import collections
import sys

_EsriWorldFile = collections.namedtuple('_EsriWorldFile',
                                        ['x_pixel_size',
                                         'y_rotation',
                                         'x_rotation',
                                         'y_pixel_size',
                                         'top_left_longitude',
                                         'top_right_latitude'])

def _parse_esri_world_file(file_name):
    f = open(file_name)
    try:
        import pdb; pdb.set_trace()
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
                   (world_file.top_left_latitude,
                        world_file.top_left_longitude),
                   image_dims)


    def __init__(self, pixel_size, top_left_lat_long, image_dims):
        self.pixel_size = pixel_size
        self.top_left_lat_long = top_left_lat_long
        self.image_dims = image_dims


    def pixel_to_lat_long(self, pix):
        """
        Return a lat,long pair for a given x,y pixel coordinate pair.

        """
        return tuple(self.top_left_lat_long[i] + pix[i] * self.pixel_size[i]
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

        stride = tuple(k[i].indices(image_dims[i])[2] for i in (0, 1))
        tl_pixel = tuple(k[i].indices(image_dims[i])[0] for i in (0, 1))
        br_pixel = tuple(k[i].indices(image_dims[i])[1] for i in (0, 1))
        img_dims = tuple((br_pixel[i] - tl_pixel[i]) // stride[i]
                                    for i in (0, 1))
        pix_size = tuple(self.pixel_size[i] * stride(i) for i in (0, 1))

        return _SphereMapping(
                    new_pixel_size,
                    tuple(self.pixel_to_lat_long(x) for x in tl_pixel),
                    img_dims)


def main():
    parser = argparse.ArgumentParser(
        description='Determine line-of-sight visibility from a geo TIFF')
    parser.add_argument('--input-file', '-w',
                        help='Input TIFF image',
                        required=True)
    parser.add_argument('--world-file', '-w', 
                        help='Input ESRI world file (.tfw)',
                        required=True)

#    parser.add_argument('--eye-coords', nargs=1, help='Space separated '
#                        'latitude, longitude and height in metres, all in '
#                        'decimal format. Specifies the viewpoint',
#                       required=True)


    args = parser.parse_args()
    world_file = _parse_esri_world_file(args.world_file)
    im = _load_height_data(args.input_file)

    import pdb; pdb.set_trace()

    from matplotlib import pyplot as plt
    plt.imshow(im, interpolation='nearest')
    plt.show()

if __name__ == '__main__':
    main()
