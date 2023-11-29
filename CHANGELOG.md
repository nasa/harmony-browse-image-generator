## v0.0.8
### 2023-11-21
 - Adds support for colortables.
   - HyBIG has the ability to apply a colortable single band input GeoTIFF and it
     follows a series of steps to determine which color information to use. It
     searches each of these locations and uses the first color information
     found.

     1. The input stac `Item`: The `Item`'s `assets` are searched for one with
        the role of `palette`.  If it finds an matching asset entry, it will
        use the data found at the `href` of that asset as the color table to
        apply to the input data.
     2. The HarmonyMessage's `Source`: If the `Source` contains a single
        `variable` and that `variable` contains a `relatedUrls` with
        `urlContentType` of `VisualizationURL` and `type` of `Color Map` the
        color map will be read from that object's `url` value.
     3. The input GeoTIFF's own colormap: If the GeoTIFF contains it's own
        colormap that will be used.
     4. Finally if no color information can be determined the output will use a
        greyscale colormap

     Three and four band input GeoTIFFs are presumed to be RGB[A].

## v0.0.7
### 2023-11-06
 - Memory usage improvements
   - Changes the quantization method from external imgquant library to the
     Pillow built-in `quantize`.
   - Updates commands in the `convert_singleband_to_raster` to use fewer
     temporary arrays as well as using `uint8` arrays instead of `float32`
     arrays.

## v0.0.6
### 2023-10-12
 - scaleExtent input parameter validated for correct order.
 - Tiling is now determined by gridcell count as a proxy for image size.  If an
   image has more than 67108864 (8192 * 8192) cells, the resulting image will be
   tiled and each tile will be broken down into 4096x4096 gridcell tiles.
 - Input parameters have new constraint. If a user supplies a scaleExtent or
   scaleSize in the request, the target CRS must also be included or the
   request will fail.
 - Output dimension calculation no based on input data resolution and the scale
   extent.

## v0.0.5
### 2023-09-12
 - Fixes tiling selection. Fixes bug for large numbers of tiles.

## v0.0.4
### 2023-08-11
- **DAS-1833**
    - Browse images are tiled according to the GIBS ICD document.  That means
      full Earth images that are ~1km in resolution are cut into 10x10degree
      images and returned tiled.

## v0.0.3
### 2023-08-03
- **DAS-1885**
    - PNG images are written as single band paletted images.

## v0.0.2
### 2023-07-25
- **DAS-1835**
    - Uses Harmony Message's parameters, or computes defaults when generating
      and output browse image. When no parameters are passed with the Message,
      the code will resize and reproject to try to make GIBS compatible images.

## v0.0.1
### 2023-07-12
- **DAS-1816**
    - Adds functionality to determine GIBS compatible output parameters, while
      allowing a user to override any parameters in the harmony message.


## v0.0.0
### 2023-05-18
- **DAS-1600**
    - Updates the service to generate PNG images from input geotiffs. The
      service can read single band geotiffs with or without a palette and
      convert them into a 4 band RGBA PNG or 3 band JPEG image. Unpaletted
      geotiffs are rendered in grayscale. The service can also read a 3 or 4
      band (RGB or RGBA) geotiff and convert that to a 4 band RGBA PNG or 3
      band JPEG files.

    - This initial version of the Harmony Browse Image Generator (HyBIG) sets
      up the core infrastructure required for a backend service. It contains a
      HarmonyAdapter class that can be invoked within a Harmony Kubernetes
      cluster.
