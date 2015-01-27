from distutils.core import setup, Extension
setup(name='height-map-vis',
      version='1.0', 
      ext_modules=[Extension('_quadtree', ['quadtree_module.c'])])

