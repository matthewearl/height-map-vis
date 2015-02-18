<h2>Introduction</h2>

Height vis will determinate line of sight visibility information for an object on the earth's surface, using <a href="http://www2.jpl.nasa.gov/srtm/">SRTM</a> topography data.

<h2>Dependencies</h2>

`numpy`
`libtiff`
`matplotlib`
`opencv2`

<h2>Setup</h2>

1. Install the dependencies (see "Dependencies" below). You may want to setup a virtualenv for this.
2. Run `setup.py install` to build and install the extension module.
3. Download SRTM tile for the area you're interested in, using <a href="http://dwtkns.com/srtm/"> the SRTM tile grabber</a>.
4. Optional: Obtain 1:250 000 OS map data from <a href="http://www.ordnancesurvey.co.uk/business-and-government/products/250k-raster.html">the OS website. Skip this step if you're not mapping an area in Britain, or you don't want a map to be overlaid.

<h2>Running</h2>

Here's a sample invocation that plots visibility of the Shard from the area north of London.

```python heightvis.py 
    -i data/srtm/srtm_36_02.tif -w data/srtm/srtm_36_02.tfw # SRTM data files,
    -e '51.5045 -0.0865 306'                                # Latitude/longitude/height of the target
    -o data/ras250_gb.zip                                   # 1:250 000 OS map data
    -c '51.818051 -0.35404' -s 0.5                          # Coordinates of view centre, and view size in degrees.

```

The calculation will take a few minutes to complete, after which you'll be presented with an interactive display.

