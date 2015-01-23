#import numpypy
import numpy
import random

__all__ = (
    'QuadTree',
    'print_quadtree',
)

class QuadTree():
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

def _plane_ray_intersect(plane, ray_origin, ray_dir):
    normal, distance = plane
    # No collision if the ray is parallel to the plane
    if normal.T * ray_dir == 0.0:
        return None
    ray_len = ((normal.T * ray_origin)[0, 0] /
               (normal.T * ray_dir)[0, 0])
    # No collison if ray is shooting away from the plane
    if ray_len < 0:
        return None
    poi = ray_origin + ray_len * ray_dir

    return poi
    
def _convex_polyhedron_ray_intersect(planes, ray_origin, ray_dir):
    for plane, i in enumerate(planes):
        poi = _plane_ray_intersect(plane, ray_origin, ray_dir)
        if poi is None:
            continue
        if all((n.T * poi)[0, 0] > d for n, d in
                    (p for p, j in enumerate(planes) if j != i)):
            return True
    return False
    

def _aa_box_ray_intersect(box_mins, box_maxs, ray_origin, ray_dir):
    """
    Intersect a ray and an axially aligned box.

    box_mins:
        A (3, 1) vector of the point on the box nearest the origin.
    box_maxs:
        A (3, 1) vector of the point on the box furthest from the origin.
    ray_origin:
        A (3, 1) vector locating the origin of the ray.
    ray_dir:
        A normalized (3, 1) vector giving the ray direction.

    """
    planes = []
    for axis in (0, 1, 2):
        min_plane = (numpy.matrix(numpy.zeros((3, 1))), box_mins[axis])
        min_plane[0][axis, 0] = 1.0

        max_plane = (numpy.matrix(numpy.zeros((3, 1))), box_maxs[axis])
        max_plane[0][axis, 0] = -1.0

        planes.append(min_plane)
        planes.append(max_plane)

    return _convex_polyhedron_ray_intersect(planes, ray_origin, ray_dir)


class HeightMap(QuadTree):
    """
    A representation of a 3-D volume, defined by a heightmap.

    """

    def get_bounding_box(self):
        """
        Return a bounding box for this volume.

        If a point is not in this box, then it is not within the volume.

        """
        return ( numpy.matrix([[self.mins[1], self.mins[0], 0.0]]).T,
                 numpy.matrix([[self.maxs[1], self.maxs[0], self.max_val]])).T
               )

    def get_inscribing_box(self):
        """
        Return an inscribing box for this volume.

        If a point is in this box, then it is within the volume.

        """
        return ( numpy.matrix([[self.mins[1], self.mins[0], 0.0]]).T,
                 numpy.matrix([[self.maxs[1], self.maxs[0], self.min_val]])).T
               )

    def shoot_ray(self, ray_origin, ray_dir, stop_quad=None):
        if _aa_box_ray_intersect(*get_inscribing_box(self),
                                 ray_origin, ray_dir):
            return True
        if not _aa_box_ray_intersect(*get_bounding_box(self),
                                     ray_origin, ray_dir):
            return False

        # If this is a leaf, then the inscribing box should be the same as the
        # bounding box, hence one of the above conditions would have passed and
        # the function would have returned.
        assert self.children

        

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
    d = random_array((1000, 1000))
    
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


if __name__ == "__main__":
    test_quadtree()

