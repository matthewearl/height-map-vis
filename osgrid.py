import math
import numpy


def _long_lat_to_cartesian(long_lat, a, b):
    """
    Convert long/lat to cartesian coordinates.

    This implements formulas B1 - B5 in Appendix B of `A guide to coordinate
    systems in Great Britain`
    
    """
    e2 = 1. - a**2 / b**2
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
    e2 = 1. - a**2 / b**2

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

    # Convert rotation parameters from arc-seconds into radians.
    rx, ry, rz = (math.pi * r / (180. * 3600.) for r in (rx, ry, rz))

    # Compute R, the rotation matrix, and the translation vector c.
    R = numpy.matrix([[1.0,  -rz,  ry],
                      [rz,   1.0, -rx],
                      [-ry,   rx, 1.0]])
    c = numpy.matrix([[cx, cy, cz]]).T

    # Transform v.
    v = c + (1. + s * 10**-6) * R * v
    
    long, lat = _cartesian_to_long_lat(v, to_a, to_b)

    return long, lat

    
def _gsr80_to_airy1830(long_lat):
    """
    Convert a long/lat on the GSR80 ellipsoid to a long/lat on the Airy1830
    ellipsoid.

    """
    airy1830_a, airy1830_b = 6377563.396, 6356256.909
    gsr80_a, gsr80_b = 6378137.000, 6356752.3141
    cx, cy, cz = 446.448, 125.157, -542.06
    s = 20.4894
    rx, ry, rz = -0.1502, -0.247, -0.8421

    return _helmert_transform(long_lat,
                              gsr80_a, gsr80_b,
                              airy1830_a, airy1830_b,
                              cx, cy, cz, s, rx, ry, rz)


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
    k


def get_image_from_sphere_mapping(sphere_mapping):
    pass
