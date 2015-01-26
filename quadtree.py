#import numpypy
import numpy
import random

__all__ = (
    'QuadTree',
    'print_quadtree',
)

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
    def make_child(cls, data, mins, maxs):
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
                            self.make_child(data, child_mins[i], child_maxs[i])
                                    for i in (0, 1)]
            self.min_val = min(*(c.min_val for c in self.children))
            self.max_val = max(*(c.max_val for c in self.children))

    def get_slice(self):
        return (slice(self.mins[0], self.maxs[0], None),
                slice(self.mins[1], self.maxs[1], None))


def _plane_ray_intersect(plane, ray_origin, ray_dir, max_ray_len=None):
    normal, distance = plane

    # No intersection if the ray is parallel to the plane.
    if normal.T * ray_dir == 0.0:
        return None
    ray_len = ((distance - (normal.T * ray_origin)[0, 0]) /
               (normal.T * ray_dir)[0, 0])

    # No intersection if ray is shooting away from the plane.
    if ray_len < 0:
        return None

    # No intersection if the plane is too far away.
    if max_ray_len is not None and ray_len > max_ray_len:
        return None

    # Otherwise, project the ray.
    poi = ray_origin + ray_len * ray_dir

    return poi


def _point_infront_of_plane(plane, point):
    n, d = plane
    return (n.T * point)[0, 0] >= d

    
def _convex_polyhedron_ray_intersect(planes,
                                     ray_origin,
                                     ray_dir,
                                     max_ray_len=None):
    # First check if the ray starts within the volume.
    if all(_point_infront_of_plane(p, ray_origin) for p in planes):
        return True

    # Otherwise, work out whether the ray intersects with the volume.
    for i, plane in enumerate(planes):
        poi = _plane_ray_intersect(plane, ray_origin, ray_dir, max_ray_len)
        if poi is None:
            continue
        if all(_point_infront_of_plane(p, poi) for p in
                    (p for j, p in enumerate(planes) if j != i)):
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
    planes = []
    for axis in (0, 1, 2):
        min_plane = (numpy.matrix(numpy.zeros((3, 1))), box_mins[axis, 0])
        min_plane[0][axis, 0] = 1.0

        max_plane = (numpy.matrix(numpy.zeros((3, 1))), -box_maxs[axis, 0])
        max_plane[0][axis, 0] = -1.0

        planes.append(min_plane)
        planes.append(max_plane)

    return _convex_polyhedron_ray_intersect(planes, ray_origin, ray_dir,
                                            max_ray_len)

class HeightMap(QuadTree):
    """
    A representation of a 3-D volume, defined by a heightmap.

    """

    # Distance above the terrain from which shadow rays will be cast. This is
    # to avoid the shadow ray intersecting with the section being tested.
    RAY_OFFSET = 0.1
    
    # The z-position of the lower bounds of the volume represented by the
    # height map. (The upper bounds are defined by the height map data, and the
    # side bounds are defined by the input array dimensions.)
    MIN_HEIGHT = -1.0

    def __init__(self, data, *args, **kwargs):
        assert self.MIN_HEIGHT < numpy.min(data)
        super(HeightMap, self).__init__(data, *args, **kwargs)

    def get_bounding_box(self):
        """
        Return a bounding box for this volume.

        If a point is not in this box, then it is not within the volume.

        """
        return ( numpy.matrix([[self.mins[1], self.mins[0], 0.0]]).T,
                 numpy.matrix([[self.maxs[1], self.maxs[0], self.max_val]]).T
               )

    def get_inscribing_box(self):
        """
        Return an inscribing box for this volume.

        If a point is in this box, then it is within the volume.

        """
        return ( numpy.matrix([[self.mins[1], self.mins[0], 0.0]]).T,
                 numpy.matrix([[self.maxs[1], self.maxs[0], self.min_val]]).T
               )

    def shoot_ray(self, ray_origin, ray_dir, max_ray_len=None):
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
                ray_end = numpy.matrix([[c + 0.5],
                                        [r + 0.5],
                                        [self.data[r, c] + self.RAY_OFFSET]])
                
                ray_dir = (ray_end - eye_point)
                max_ray_len = numpy.linalg.norm(ray_dir);
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
    d = numpy.zeros((64, 64))
    d[20:28, 25:33] = numpy.ones((8, 8))
    d[36:44, 25:33] = .5 * numpy.ones((8, 8))
    d[11:19, 18:22] = .5 * numpy.ones((8, 4))
    eye_point = numpy.matrix([[40.5, 32.5, 2.0]]).T
    
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

    from matplotlib import pyplot as plt
    plt.imshow(v, interpolation='nearest')
    plt.show()


if __name__ == "__main__":
    test_quadtree()
    test_visibility()

