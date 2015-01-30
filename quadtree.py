import math
import random

__all__ = (
    'HeightMap',
    'QuadTree',
    'print_quadtree',
)

rays_shot = 0
import collections
stats = collections.defaultdict(int)


USE_PYPY = False
USE_C_QUADTREE = True
TEST_C_QUADTREE = False
PROFILER = False


if TEST_C_QUADTREE:
    assert USE_C_QUADTREE

if USE_PYPY:
    import numpypy
import numpy

if USE_C_QUADTREE:
    import _quadtree


class QuadTree(object):
    """
    A quadtree is a tree representation of an array.

    Each quadtree node represents a rectangular sub-array of an input array.
    If the node represents an array of size > 1, then the node has 2 children.
    Each child represents half of its parent's data. The split is made along
    the longest axis.

    Each node keeps track of the minimum and maximum value of the data it
    represents. As such, if the array is interpreted as a height map, a
    three-dimensional bounding box of the can be easily produced for a given
    node.

    """

    @classmethod
    def make_child(cls, parent, data, mins, maxs):
        return cls(data, mins, maxs)

    def __init__(self, data, mins=None, maxs=None):
        self.data = data

        if mins is None:
            assert maxs is None
            mins = 0, 0
            maxs = data.shape

        self.mins = mins
        self.maxs = maxs

        shape = (maxs[0] - mins[0],
                 maxs[1] - mins[1])
        assert shape[0] > 0
        assert shape[1] > 0

        if shape == (1, 1):
            # Leaf node.
            self.min_val = self.max_val = data[mins]
            self.children = []
            self.split_axis = None
        else:
            # Non-leaf. Split along the largest axis and create children. 
            if shape[0] >= shape[1]:
                self.split_axis = 0
                child_mins = [mins, (mins[0] + shape[0] / 2, mins[1])]
                child_maxs = [(mins[0] + shape[0] / 2, maxs[1]), maxs]
            else:
                self.split_axis = 1
                child_mins = [mins, (mins[0], mins[1] + shape[1] / 2)]
                child_maxs = [(maxs[0], mins[1] + shape[1] / 2), maxs]
            self.children = [
                      self.make_child(self, data, child_mins[i], child_maxs[i])
                                    for i in (0, 1)]
            self.min_val = min(*(c.min_val for c in self.children))
            self.max_val = max(*(c.max_val for c in self.children))

    def get_slice(self):
        return (slice(self.mins[0], self.maxs[0], None),
                slice(self.mins[1], self.maxs[1], None))


def _plane_ray_intersect(plane, ray_origin, ray_dir, max_ray_len=None):
    normal, distance = plane

    # No intersection if the ray is parallel to the plane.
    origin_dot = (normal[0] * ray_origin[0] +
                  normal[1] * ray_origin[1] +
                  normal[2] * ray_origin[2])
    dir_dot = (normal[0] * ray_dir[0] +
               normal[1] * ray_dir[1] +
               normal[2] * ray_dir[2])
    if dir_dot == 0.0:
        return None
    ray_len = (distance - origin_dot) / dir_dot

    # No intersection if ray is shooting away from the plane.
    if ray_len < 0:
        return None

    # No intersection if the plane is too far away.
    if max_ray_len is not None and ray_len > max_ray_len:
        return None

    # Otherwise, project the ray.
    poi = [
        ray_origin[0] + ray_len * ray_dir[0],
        ray_origin[1] + ray_len * ray_dir[1],
        ray_origin[2] + ray_len * ray_dir[2],
    ]

    return poi


def _point_infront_of_plane(plane, point):
    n, d = plane
    return (n[0] * point[0] +
            n[1] * point[1] +
            n[2] * point[2]) >= d

    
def _convex_polyhedron_ray_intersect(planes,
                                     ray_origin,
                                     ray_dir,
                                     max_ray_len=None):
    # First check if the ray starts within the volume.
    for p in planes:
        if not _point_infront_of_plane(p, ray_origin):
            break
    else:
        return True

    # Otherwise, work out whether the ray intersects with the volume.
    for i, plane in enumerate(planes):
        poi = _plane_ray_intersect(plane, ray_origin, ray_dir, max_ray_len)
        if poi is None:
            continue

        # Check that the POI is infront of all the other planes.
        for j,  p in enumerate(planes):
            if j == i:
                continue
            if not _point_infront_of_plane(p, poi):
                break
        else:
            # Loop completed
            return True

    return False
    

def _aa_box_ray_intersect(box,
                          ray_origin,
                          ray_dir,
                          max_ray_len=None):
    """
    Intersect a ray and an axially aligned box.

    box:
        A pair of (3, 1) vectors, describing the vertex of an axially aligned
        box nearest the origin, and furthest from the origin, respectively.
    ray_origin:
        A (3, 1) vector locating the origin of the ray.
    ray_dir:
        A normalized (3, 1) vector giving the ray direction.

    """
    box_mins, box_maxs = box
    
    ray_origin = (float(ray_origin[0, 0]),
                  float(ray_origin[1, 0]),
                  float(ray_origin[2, 0]))
    ray_dir = (float(ray_dir[0, 0]),
               float(ray_dir[1, 0]),
               float(ray_dir[2, 0]))
    planes = [
        ((1.0,0.0,0.0), float(box_mins[0, 0])),
        ((-1.0,0.0,0.0), -float(box_maxs[0, 0])),
        ((0.0,1.0,0.0), float(box_mins[1, 0])),
        ((0.0,-1.0,0.0), -float(box_maxs[1, 0])),
        ((0.0,0.0,1.0), float(box_mins[2, 0])),
        ((0.0,0.0,-1.0), -float(box_maxs[2, 0])),
    ]

    def get_py_res():
        return _convex_polyhedron_ray_intersect(planes, ray_origin, ray_dir,
                                                max_ray_len)

    def get_c_res():
        return _quadtree.convex_polyhedron_ray_intersect(planes, ray_origin,
                                                         ray_dir, max_ray_len)
        
    if USE_C_QUADTREE:
        res = get_c_res()
        if TEST_C_QUADTREE:
            py_res = get_py_res()
            assert py_res == res
    else:
        res = get_py_res()

    return res


class HeightMap(QuadTree):
    """
    A representation of a 3-D volume, defined by a heightmap.

    """

    # Distance above the terrain from which shadow rays will be cast. This is
    # to avoid the shadow ray intersecting with the section being tested.
    RAY_OFFSET = 0.1
    
    @classmethod
    def make_child(cls, parent, data, mins, maxs):
        return cls(data, mins, maxs, min_height=parent.min_height)
        
    def __init__(self, data, *args, **kwargs):
        # Calculate the z-position of the lower bounds of the volume
        # represented by the height map. (The upper bounds are defined by the
        # height map data, and the side bounds are defined by the input array
        # dimensions.)
        if 'min_height' not in kwargs:
            self.min_height = numpy.min(data) - 1.0
        else:
            self.min_height = kwargs['min_height']
            del kwargs['min_height']
        super(HeightMap, self).__init__(data, *args, **kwargs)

    def get_bounding_box(self):
        """
        Return a bounding box for this volume.

        If a point is not in this box, then it is not within the volume.

        """
        return ( numpy.array([[self.mins[1], self.mins[0],
                               self.min_height]]).T,
                 numpy.array([[self.maxs[1], self.maxs[0], self.max_val]]).T
               )

    def get_inscribing_box(self):
        """
        Return an inscribing box for this volume.

        If a point is in this box, then it is within the volume.

        """
        return ( numpy.array([[self.mins[1], self.mins[0],
                               self.min_height]]).T,
                 numpy.array([[self.maxs[1], self.maxs[0], self.min_val]]).T
               )

    def shoot_ray(self, ray_origin, ray_dir, max_ray_len=None):
        global rays_shot
        global stats
        rays_shot += 1
        shape = self.maxs[0] - self.mins[0], self.maxs[1] - self.mins[1]
        stats[shape] += 1 

        if _aa_box_ray_intersect(self.get_inscribing_box(),
                                 ray_origin, ray_dir, max_ray_len):
            return True
        if not _aa_box_ray_intersect(self.get_bounding_box(),
                                     ray_origin, ray_dir, max_ray_len):
            return False

        # If this is a leaf, then the inscribing box should be the same as the
        # bounding box, hence one of the above conditions would have passed and
        # the function would have returned.
        assert self.children

        return any(c.shoot_ray(ray_origin, ray_dir, max_ray_len)
                            for c in self.children)


    def get_visible(self, eye_point):
        visible = numpy.zeros(self.data.shape)
        for r in xrange(self.data.shape[0]):
            print "Row {} / {}".format(r, self.data.shape[0])
            for c in xrange(self.data.shape[1]):
                ray_end = numpy.array([[c + 0.5],
                                        [r + 0.5],
                                        [self.data[r, c] + self.RAY_OFFSET]])
                
                ray_dir = (ray_end - eye_point)
                max_ray_len = math.sqrt(ray_dir[0, 0] ** 2 +
                                        ray_dir[1, 0] ** 2 +
                                        ray_dir[2, 0] ** 2)
                ray_dir /= max_ray_len

                if self.shoot_ray(eye_point, ray_dir, max_ray_len):
                    visible[r, c] = 1.0
                else:
                    visible[r, c] = 0.0
        return visible


def print_quadtree(quadtree, indent=0):
    print " " * indent + "axis={} mins={} maxs={}".format(quadtree.split_axis,
                                                          quadtree.mins,
                                                          quadtree.maxs)
    for child in quadtree.children:
        print_quadtree(child, indent=(indent + 4))


def random_array(shape):
    """numpypy compatible implementation of numpy.random.random()"""
    a = numpy.zeros(shape)
    for r in xrange(shape[0]):
        a[r, :] = [random.random() for i in xrange(shape[1])]
    return a


def test_quadtree():
    print "Generating array"
    d = random_array((100, 100))
    
    def test_recursive(q):
        # Check the stored minimum/maximum values are correct.
        assert q.min_val == numpy.min(d[q.get_slice()])
        assert q.max_val == numpy.max(d[q.get_slice()])

        # Check the data represented by the children, when combined, is the
        # same as the data represented by this node.
        if q.split_axis is not None:
            if q.split_axis == 0:
                stack_fn = numpy.vstack
            elif q.split_axis == 1:
                stack_fn = numpy.hstack
            else:
                assert False

            assert numpy.array_equal(
                          stack_fn([d[cq.get_slice()] for cq in q.children]),
                          d[q.get_slice()])

        # Recursively perform the test on the children.
        for cq in q.children:
            test_recursive(cq)

    print "Building quadtree"
    quadtree = QuadTree(d)
    #print_quadtree(quadtree)
    print "Testing"
    test_recursive(quadtree)

def test_visibility():
    k = 1
    d = numpy.zeros((k*64, k*64))
    d[k*20:k*28, k*25:k*33] = numpy.ones((k*8, k*8))
    d[k*36:k*44, k*25:k*33] = .5 * numpy.ones((k*8, k*8))
    d[k*11:k*19, k*18:k*22] = .5 * numpy.ones((k*8, k*4))
    eye_point = numpy.array([[k*40.5, k*32.5, 2.0]]).T
    
    height_map = HeightMap(d) 

    def print_mat(m):
        def el_to_char(e):
            if e == 0.0:
                return ' '
            else:
                return '{}'.format(e)[0]
        for r in xrange(m.shape[0]):
            print ''.join(el_to_char(m[r, c]) for c in xrange(m.shape[1]))

    print_mat(d)

    v = height_map.get_visible(eye_point)

    print_mat(v)

    #from matplotlib import pyplot as plt
    #plt.imshow(v, interpolation='nearest')
    #plt.show()


if __name__ == "__main__":
    #test_quadtree()
    if PROFILER:
        import cProfile
        cProfile.run('test_visibility()', sort='time')
    else:
        test_visibility()

    print repr(stats)
    print stats
    from pprint import pprint as pp
    for k, v in sorted(stats.iteritems()):
        print "{}: {}".format(k, v)
    print rays_shot / (64. * 64.)


