import cv2
import libtiff
import math
import matplotlib.image
import numpy
import os
import StringIO
import tempfile
import zipfile


_OS_MAP_GRID_TILES = (
    ("HL", "HM", "HN", "HO", "HP", "JL", "JM"),
    ("HQ", "HR", "HS", "HT", "HU", "JQ", "JR"),
    ("HV", "HW", "HX", "HY", "HZ", "JV", "JW"),
    ("NA", "NB", "NC", "ND", "NE", "OA", "OB"),
    ("NF", "NG", "NH", "NJ", "NK", "OF", "OG"),
    ("NL", "NM", "NN", "NO", "NP", "OL", "OM"),
    ("NQ", "NR", "NS", "NT", "NU", "OQ", "OR"),
    ("NV", "NW", "NX", "NY", "NZ", "OV", "OW"),
    ("SA", "SB", "SC", "SD", "SE", "TA", "TB"),
    ("SF", "SG", "SH", "SJ", "SK", "TF", "TG"),
    ("SL", "SM", "SN", "SO", "SP", "TL", "TM"),
    ("SQ", "SR", "SS", "ST", "SU", "TQ", "TR"),
    ("SV", "SW", "SX", "SY", "SZ", "TV", "TW"),
)


def _long_lat_to_cartesian(long_lat, a, b):
    """
    Convert long/lat to cartesian coordinates.

    This implements formulas B1 - B5 in Appendix B of `A guide to coordinate
    systems in Great Britain`
    
    """
    long, lat = long_lat
    e2 = 1. - b**2 / a**2
    nu = a / math.sqrt(1. - e2 * math.sin(lat) ** 2)
    H = 0
    return numpy.matrix([(nu + H) * math.cos(lat) * math.cos(long),
                         (nu + H) * math.cos(lat) * math.sin(long),
                         ((1. - e2) * nu + H) * math.sin(lat)]).T


def _cartesian_to_long_lat(v, a, b):
    """
    Convert long/lat to cartesian coordinates.

    This implements formulas B6 - B8 in Appendix B of `A guide to coordinate
    systems in Great Britain`

    """
    x, y, z = numpy.array(v).flatten()
    e2 = 1. - b**2 / a**2
    p = math.sqrt(x**2 + y**2)
    lat = math.atan2(z, p * (1. - e2))
    lat_prev = None
    while lat_prev is None or abs(lat - lat_prev) > 10**-16: 
        lat_prev = lat
        nu = a / math.sqrt(1. - e2 * math.sin(lat) ** 2)
        lat = math.atan2(z + e2 * nu * math.sin(lat), p)

    long = math.atan2(y, x)

    return long, lat


def _helmert_transform(long_lat,
                       from_a, from_b,
                       to_a, to_b,
                       cx, cy, cz, s, rx, ry, rz):
    """
    Convert long/lat between ellipsoids.

    {from,to}_{a,b} are the major/minor semi axes of the ellipsoids being
    converted.

    The other parameters are described here:
        http://en.wikipedia.org/wiki/Helmert_transformation

    """
    long, lat = long_lat

    # Compute the point in cartesian space.
    v = _long_lat_to_cartesian((long, lat), from_a, from_b)

    # Convert rotation parameters from arc-seconds into radians, and s from
    # ppm.
    rx, ry, rz = (math.pi * r / (180. * 3600.) for r in (rx, ry, rz))
    s *= 10**-6

    # Compute R, the rotation matrix, and the translation vector c.
    R = numpy.matrix([[1.0 + s,      -rz,      ry],
                      [rz,       1.0 + s,     -rx],
                      [-ry,           rx, 1.0 + s]])
    c = numpy.matrix([[cx, cy, cz]]).T

    # Transform v.
    v = c + R * v
    
    long, lat = _cartesian_to_long_lat(v, to_a, to_b)

    return long, lat

    
def _wgs84_to_osgb36(long_lat):
    """
    Convert a long/lat in WGS84 coordinates to a long/lat in OSGB36
    coordinates.

    """
    # Ellipsoid paramaters. OSGB36 uses the Airy 1830 ellipsoid, whereas WGS84
    # uses the GSR80 ellipsoid.
    airy1830_a, airy1830_b = 6377563.396, 6356256.909
    gsr80_a, gsr80_b = 6378137.000, 6356752.3141

    # Translation/scale/rotation parameters.
    cx, cy, cz = -446.448, 125.157, -542.06
    s = 20.4894
    rx, ry, rz = -0.1502, -0.247, -0.8421

    return _helmert_transform(long_lat,
                              gsr80_a, gsr80_b,
                              airy1830_a, airy1830_b,
                              cx, cy, cz, s, rx, ry, rz)


def _osgb36_long_lat_to_os_grid(long_lat):
    """
    Convert an OSGB36 long/lat to OS grid northings/eastings.

    This is the implementation described in Appendix C of `A guide to
    coordinate systems in Great Britain`.

    long_lat: OSGB36 longitude/latitude in radians.

    Returns: (E, N), the eastings and northings in OS grid coordinates,
        respectively.

    """
    long, lat = long_lat

    # a, b = major & minor semi-axes
    # F0 = scale factor on central meridian
    # origin_long, origin_lat = True origin latitude, longitude
    # N0, E0 = Northing & easting of true origin, metres
    # e2 = eccentricity squared
    a = 6377563.396
    b = 6356256.909
    F0 = 0.9996012717
    origin_lat = math.pi * 49. / 180.
    origin_long = math.pi * -2. / 180.
    N0 = -100000.
    E0 = 400000.
    e2 = 1. - (b * b)/(a * a)
    n = (a - b)/(a + b)

    nu = a * F0 / (1. - e2 * math.sin(lat)**2.)**0.5
    rho = (a * F0 * (1. - e2) *
                (1. - e2 * math.sin(lat)**2)**-1.5)
    eta2 = nu / rho - 1.

    M1 = (1. + n + (5. / 4) * n**2 + (5. / 4) * n**3) * (lat - origin_lat)
    M2 = ((3. * n + 3. * n**2 + (21. / 8) * n**3) *
          math.sin(lat - origin_lat) *
          math.cos(lat + origin_lat))
    M3 = (((15. / 8) * n**2 + (15. / 8) * n**3) *
          math.sin(2. * (lat - origin_lat)) *
          math.cos(2. * (lat + origin_lat)))
    M4 = ((35. / 24) * n**3 *
          math.sin(3. * (lat - origin_lat)) *
          math.cos(3. * (lat + origin_lat)))
    M = b * F0 * (M1 - M2 + M3 - M4)

    I = M + N0
    II = (nu / 2.) * math.sin(lat) * math.cos(lat)
    III = ((nu / 24.) *
           math.sin(lat) *
           math.cos(lat)**3 *
           (5. - math.tan(lat)**2 + 9. * eta2))
    IIIA = ((nu / 720.) *
            math.sin(lat) *
            math.cos(lat)**5 *
            (61. - 58. * math.tan(lat)**2 + math.tan(lat)**4))
    IV = nu * math.cos(lat)
    V = (nu / 6.) * math.cos(lat)**3 * (nu / rho - math.tan(lat)**2)
    VI = ((nu / 120.) *
          math.cos(lat)**5 *
          (5. - 18. * math.tan(lat)**2 +
           math.tan(lat)**4 +
           14. * eta2 -
           58. * math.tan(lat)**2 * eta2))

    delta_long = long - origin_long

    N = (I +
         II * (delta_long**2) +
         III * (delta_long**4) +
         IIIA * (delta_long**6))
    E = (E0 +
         IV * delta_long +
         V * (delta_long**3) +
         VI * (delta_long**5))

    return E, N


def _wgs84_long_lat_to_os_grid(long_lat):
    """
    Convert an WGS84 long/lat to OS grid northings/eastings.

    long_lat: WGS84 longitude/latitude in radians.

    Returns: (E, N), the eastings and northings in OS grid coordinates,
        respectively.

    """
    osgb36_long_lat = _wgs84_to_osgb36(long_lat)
    return _osgb36_long_lat_to_os_grid(osgb36_long_lat)


def _load_tif_from_non_seekable_file(f):
    """
    Load a tif from a non-seekable file-like object.

    """
    f = StringIO.StringIO(f.read())
    im = matplotlib.image.imread(f, format="tiff")

    return im


class TiledOsMap(object):
    """
    A set of named OS map tiles contained in a zip file.


    """
    DEFAULT_TILE_PATH = "ras250_gb/data"


    def __init__(self, zip_file_name, tile_path=DEFAULT_TILE_PATH):
        self.zip_file = zipfile.ZipFile(zip_file_name, 'r')
        self.tile_path = tile_path


    def _load_tile(self, tile_name):
        """
        Load a tile from the zip file.

        """
        file_path = "{}/{}.tif".format(self.tile_path, tile_name)

        with self.zip_file.open(file_path) as f:
            return _load_tif_from_non_seekable_file(
                                                 self.zip_file.open(file_path))

    def get_image_from_wgs84_rect(self,
                                  north_west_long_lat,
                                  south_east_long_lat, image_dims):
        """
        Map a portion of an OS map into an image, whose bounds are defined by
        minimum and maximum latitudes and longitudes.

        For example:
          get_image_from_wgs84_rect((0, 55), (1, 54), (100, 100))

        Will return a 100x100 pixel OS map where pixel (x, y) corresponds with
        WGS84 longitude (x / 100) degrees and WGS84 latitude (55 - y / 100)
        degrees.

        """
        north_west_long_lat = tuple(math.pi * x / 180. for x in
                                                           north_west_long_lat)
        south_east_long_lat = tuple(math.pi * x / 180. for x in
                                                           south_east_long_lat)

        # Obtain the corners of the rectangle in WGS84 long/lat coordinates.
        long_lat_corners = (
            (north_west_long_lat[0], north_west_long_lat[1]), # NW
            (south_east_long_lat[0], north_west_long_lat[1]), # NE
            (north_west_long_lat[0], south_east_long_lat[1]), # SW
            (south_east_long_lat[0], south_east_long_lat[1]), # SE
        )
        
        # Obtain the corners in OS grid coordinates.
        os_grid_corners = tuple(map(_wgs84_long_lat_to_os_grid,
                                    long_lat_corners))

        # Calculate a set of OS grid tiles that will cover the mapped rectangle.
        os_grid_mins = (min(E for E, N in os_grid_corners),
                        min(N for E, N in os_grid_corners))
        os_grid_maxs = (max(E for E, N in os_grid_corners),
                        max(N for E, N in os_grid_corners))
        west_tile_east_idx = int(os_grid_mins[0] // 100000)
        east_tile_east_idx = int(os_grid_maxs[0] // 100000)
        south_tile_north_idx = int(os_grid_mins[1] // 100000)
        north_tile_north_idx = int(os_grid_maxs[1] // 100000)

        # Pull in imagery for these tiles and stitch them into a single image.
        combined_tiles = numpy.vstack(
            numpy.hstack(self._load_tile(
                _OS_MAP_GRID_TILES[-(1 + north_idx)][east_idx])
                    for east_idx in range(west_tile_east_idx,
                                          east_tile_east_idx + 1))
            for north_idx in reversed(range(south_tile_north_idx,
                                            north_tile_north_idx + 1)))

        # Define a function to convert from OS grid coordinates to image x, y
        # coordinates and use it to calculate the corners in image coordinates.
        northings_per_pixel = 100000. * ((1. + north_tile_north_idx -
                                          south_tile_north_idx) /
                                                       combined_tiles.shape[0])
        eastings_per_pixel = 100000. * ((1. + east_tile_east_idx -
                                          west_tile_east_idx) /
                                                       combined_tiles.shape[1])
        north_west_os_grid_coord = (100000. * west_tile_east_idx,
                                    100000. * (1 + north_tile_north_idx))
        def os_grid_to_pixel_coordinates(grid_coord):
            x = ((grid_coord[0] - north_west_os_grid_coord[0]) /
                                                            eastings_per_pixel)
            y = ((north_west_os_grid_coord[1] - grid_coord[1]) /
                                                           northings_per_pixel)
            return x, y
        src_corners = tuple(map(os_grid_to_pixel_coordinates,
                                  os_grid_corners))

        # Obtain the perspective transform to map long/lat coordinates to image
        # coordinates. 
        import pdb; pdb.set_trace()
        dst_corners = ((0, 0),
                       (image_dims[0], 0),
                       (0, image_dims[1]),
                       image_dims)
        mat = cv2.getPerspectiveTransform(
                                  numpy.array(dst_corners, numpy.float32),
                                  numpy.array(src_corners, numpy.float32))

        # Use the transform to produce the output image.
        out = cv2.warpPerspective(combined_tiles, mat, image_dims,
                                  flags=cv2.WARP_INVERSE_MAP)

        return out


def _test_tile_loading():
    import sys

    print "Opening zip file"
    t = TiledOsMap(sys.argv[1])

    print "Loading file"
    im = t._load_tile("ST")
    im = im[::10, ::10]
    from matplotlib import pyplot as plt
    plt.imshow(im)
    plt.show()
    

def _test_image_from_wgs84_rect():
    import sys

    print "Opening zip file"
    t = TiledOsMap(sys.argv[1])
    se = -0.35404, 51.818051
    nw = -0.387700, 51.832913
    im = t.get_image_from_wgs84_rect(nw, se, (500, 500))

    from matplotlib import pyplot as plt
    plt.imshow(im)
    plt.show()


if __name__ == "__main__":
    _test_image_from_wgs84_rect()

