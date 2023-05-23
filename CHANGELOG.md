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
