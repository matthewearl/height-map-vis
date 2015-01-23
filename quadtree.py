import numpy

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
            self.children = [QuadTree(data, child_mins[i], child_maxs[i])
                                    for i in (0, 1)]
            self.min_val = min(*(c.min_val for c in self.children))
            self.max_val = max(*(c.max_val for c in self.children))

    def get_slice(self):
        return (slice(self.mins[0], self.maxs[0], None),
                slice(self.mins[1], self.maxs[1], None))


def print_quadtree(quadtree, indent=0):
    print " " * indent + "axis={} mins={} maxs={}".format(quadtree.split_axis,
                                                          quadtree.mins,
                                                          quadtree.maxs)
    for child in quadtree.children:
        print_quadtree(child, indent=(indent + 4))


def test_quadtree():
    d = numpy.random.random((200, 200))
    
    def test_recursive(q):
        # Check the stored minimum/maximum values are correct.
        assert q.min_val == numpy.min(d[q.get_slice()])
        assert q.max_val == numpy.max(d[q.get_slice()])

        # Check the data represented by the children, when combined, is the same
        # as the data represented by this node.
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

