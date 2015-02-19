<h2>Introduction</h2>

Height vis will determinate line of sight visibility information for an object on the earth's surface, using <a href="http://www2.jpl.nasa.gov/srtm/">SRTM</a> topography data.

<h2>Dependencies</h2>

* `numpy`
* `libtiff`
* `matplotlib`
* `opencv2`

<h2>Setup</h2>

1. Install the dependencies (see "Dependencies" below). You may want to setup a virtualenv for this.
2. Run `setup.py install` to build and install the extension module.
3. Download SRTM tile for the area you're interested in, using <a href="http://dwtkns.com/srtm/">the SRTM tile grabber</a>.
4. Optional: Obtain 1:250 000 OS map data from <a href="http://www.ordnancesurvey.co.uk/business-and-government/products/250k-raster.html">the OS website</a>. Skip this step if you're not mapping an area in Britain, or you don't want a map to be overlaid.

<h2>Running</h2>

Here's a sample invocation that plots visibility of <a href="http://en.wikipedia.org/wiki/The_Shard">the Shard</a> from <a href="https://www.google.co.uk/maps/place/Newbury,+West+Berkshire/@51.3927652,-1.326874,12z/data=!4m2!3m1!1s0x487402002f595ba9:0xc6646baff4a75c50">the area surrounding Newbury</a>.

```
python heightvis.py 
    -i data/srtm/srtm_36_02.tif -w data/srtm/srtm_36_02.tfw # SRTM data files,
    -e '51.5045 -0.0865 306'                                # Latitude/longitude/height of the target
    -o data/ras250_gb.zip                                   # 1:250 000 OS map data
    -c '51.397849 -1.343434' -s 0.3                         # Coordinates of view centre, and view size in degrees.

```

(For more options run `python heightvis.py --help`.)

The calculation will take a few minutes to complete, after which you'll be presented with an interactive display. The left hand side is a plot of visibility information: Grey areas are where the target is not visible:

(Insert screenshot)

Clicking on a point on the left hand side will update the right hand side with a cross-sectional profile view

(Insert screenshot)

This shows:
* Blue line: Elevation of the terrain on the line between the clicked piece of terrain and the target.
* Red line: Sight line from the terrain to the target. (If this intersects with the below, then the view is blocked.)
* Green line: Earth curvature on the line between the clicked piece of terrain and the target. This is the offset applied to the raw height data to account for curvature of the earth.

