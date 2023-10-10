## Unreleased
### 2023-10-09
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
