import math
import numpy


def _helmert_transform(long_lat,
                       from_a, from_b,
                       to_a, to_b,
                       cx, cy, cz, s, rx, ry, rz):

    long, lat = long_lat

    # Compute x = the point in cartesian space.
    x = numpy.matrix([[math.cos(lat) * math.cos(long),
                       math.cos(lat) * math.sin(long),
                       (from_b**2 / from_a**2) * math.sin(lat)]]).T
    x *= from_a / math.sqrt(1. -
            (1. - from_b**2 / from_a**2) * math.sin(lat)**2)

    # Compute R, the rotation matrix, and the translation vector c.
    R = numpy.matrix([[1.0,  -rx,  ry],
                      [rz,   1.0, -rx],
                      [-ry,   rx, 1.0]])
    c = numpy.matrix([[cx, cy, cz]]).T

    # Transform x.
    x = c + (1. + s * 10**-6) * R * x

    # Project back to a latitude / longitude.
    p = numpy.linalg.norm(x[:2, 0])
    lat = math.arctan2(x[2, 0], p * to_b**2 / to_a**2)
    lat_prev = 2. * math.pi
    while abs(lat - lat_prev) > 10**-16: 
        lat, lat_prev = lat_prev, lat
        to_nu = to_a / math.sqrt(1-to_e2*sin(lat_prev)**2)
        lat = arctan2(x[2, 0] + to_e2 * to_nu * math.sin(lat_prev), p)

    
def _grs80_to_airy1830(long_lat):
    """


    """


def _airy1830_long_lat_to_os_grid(long_lat):
    """
    Convert long/lat on an Airy 1830 ellipsoid to OS grid northings/eastings.

    This is the implementation described in Appendix C of `A guide to
    coordinate systems in Great Britain`.

    long_lat: longitude/latitude in radians.

    Returns: N, E.

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

    # nu = transverse radius of curvature
    # rho = meridional radius of curvature
    # eta = ?
    nu = a * F0 / math.sqrt(1. - e2 * math.sin(lat)**2.)
    rho = (a * F0 * (1. - e2) / 
                        (1. - e2 * math.sin(lat) * math.sin(lat))**1.5)
    eta2 = nu / rho - 1

    # Calculate M, the meridian distance.
    Ma = (1. + n + (5. / 4) * n**2 + (5. / 4) * n**3) * (lat - origin_lat)
    Mb = ((3. * n + 3. * n**2 + (21. / 8) * n**3) *
          math.sin(lat - origin_lat) *
          math.cos(lat + origin_lat))
    Mc = (((15. / 8) * n**2 + (15. / 8) * n**3) *
          math.sin(2. * (lat - origin_lat)) *
          math.cos(2. * (lat + origin_lat)))
    Md = ((35. / 24) * n**3 *
          math.sin(3. * (lat - origin_lat)) *
          math.cos(3. * (lat + origin_lat)))
    M = b * F0 * (Ma - Mb + Mc - Md)

    # Apply the Redfearne
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

    N = I + II*(delta_long ** 2) + III*(delta_long ** 4) + IIIA*(delta_long**6)
    E = E0 + IV*delta_long + V*(delta_long**3) + VI*(delta_long**5)

    return E, N


def _wgs84_long_lat_to_os_grid(long_lat):

    long, lat = (x * math.pi / 180. for x in long_lat)


def get_image_from_sphere_mapping(sphere_mapping):
    pass
