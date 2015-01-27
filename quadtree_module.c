#include <assert.h>
#include <Python.h>


static double
_quadtree_dot_product_obj_obj (PyObject *v1, PyObject *v2)
{
    Py_ssize_t  i;
    double      out;

    assert(PyTuple_GET_SIZE(v1) == 3);
    assert(PyTuple_GET_SIZE(v2) == 3);

    out = 0.0;
    for (i = 0; i < 3; i++) {
        out += PyFloat_AS_DOUBLE(PyTuple_GET_ITEM(v1, i)) *
                        PyFloat_AS_DOUBLE(PyTuple_GET_ITEM(v2, i));
    }

    return out;
}


static double
_quadtree_dot_product_obj_arr (PyObject *v1, double *arr)
{
    Py_ssize_t  i;
    double      out;

    assert(PyTuple_GET_SIZE(v1) == 3);

    out = 0.0;
    for (i = 0; i < 3; i++) {
        out += PyFloat_AS_DOUBLE(PyTuple_GET_ITEM(v1, i)) * arr[i];
    }

    return out;

}


static int
_quadtree_point_infront_of_plane_obj (PyObject *plane, PyObject *point)
{
    PyObject *normal;
    double    d;

    normal = PyTuple_GET_ITEM(plane, 0);
    d = PyFloat_AS_DOUBLE(PyTuple_GET_ITEM(plane, 1));

    return _quadtree_dot_product_obj_obj(normal, point) >= d;
}


static int
_quadtree_point_infront_of_plane_arr (PyObject *plane, double *arr)
{
    PyObject *normal;
    double    d;

    normal = PyTuple_GET_ITEM(plane, 0);
    d = PyFloat_AS_DOUBLE(PyTuple_GET_ITEM(plane, 1));

    return _quadtree_dot_product_obj_arr(normal, arr) >= d;
}


static int
_quadtree_plane_ray_intersect (PyObject *plane,
                               PyObject *ray_origin,
                               PyObject *ray_dir,
                               double    max_ray_len,
                               double   *poi)
{
    PyObject *normal;
    double    distance;
    double    origin_dot;
    double    dir_dot;
    double    ray_len;
    int       i;

    normal = PyTuple_GET_ITEM(plane, 0);
    distance = PyFloat_AS_DOUBLE(PyTuple_GET_ITEM(plane, 1));

    origin_dot = _quadtree_dot_product_obj_obj(normal, ray_origin);
    dir_dot = _quadtree_dot_product_obj_obj(normal, ray_dir);

    if (dir_dot == 0.0) {
        return 0;
    }

    ray_len = (distance - origin_dot) / dir_dot;

    if (ray_len < 0.0) {
        return 0;
    }

    if (ray_len > max_ray_len) {
        return 0;
    }
   
    for (i = 0; i < 3; i++) {
        poi[i] = PyFloat_AS_DOUBLE(PyTuple_GET_ITEM(ray_origin, i)) +
                ray_len * PyFloat_AS_DOUBLE(PyTuple_GET_ITEM(ray_dir, i));
    }

    return 1;
}


static PyObject *
_quadtree_convex_polyhedron_ray_intersect (PyObject *self, PyObject *args)
{
    PyObject   *planes;
    PyObject   *plane;
    PyObject   *p;
    PyObject   *ray_origin;
    PyObject   *ray_dir;
    double      max_ray_len;
    Py_ssize_t  i, j;
    double      poi[3];

    if (!PyArg_ParseTuple(args,
                          "OOOd",
                          &planes,
                          &ray_origin,
                          &ray_dir,
                          &max_ray_len)) {
        return NULL;
    }
   
    /* First check if the ray starts within the volume. */
    for (i = 0; i < PyList_GET_SIZE(planes); i++) {
        plane = PyList_GET_ITEM(planes, i);
        if (!_quadtree_point_infront_of_plane_obj(plane, ray_origin)) {
            break;
        }
    }

    if (i == PyList_GET_SIZE(planes)) {
        Py_RETURN_TRUE;
    }

    /* Otherwise, work out whether the ray intersects with the volume. */
    for (i = 0; i < PyList_GET_SIZE(planes); i++) {
        plane = PyList_GET_ITEM(planes, i);
        if (!_quadtree_plane_ray_intersect(plane,
                                           ray_origin,
                                           ray_dir,
                                           max_ray_len,
                                           poi)) {
            continue;
        }

        /* Check that the POI is infront of all the other planes. */
        for (j = 0; j < PyList_GET_SIZE(planes); j++) {
            if (j == i) {
                continue;
            }
            p = PyList_GET_ITEM(planes, j);
            if (!_quadtree_point_infront_of_plane_arr(p, poi)) {
                break;
            }
        }
        if (j == PyList_GET_SIZE(planes)) {
            Py_RETURN_TRUE;
        }
    }
    
    Py_RETURN_FALSE;
}


static PyMethodDef quadtree_methods[] = {
  { "convex_polyhedron_ray_intersect",
                     _quadtree_convex_polyhedron_ray_intersect, METH_VARARGS },
  { NULL, NULL },
};


PyMODINIT_FUNC
init_quadtree (void)
{
    PyObject *m;

    m = Py_InitModule("_quadtree", quadtree_methods);
    if (m == NULL)
        return;

}
 
