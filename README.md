# Phenotyping System for Efficient User-centered Detection On AMIGA

## Setup

```bash
pip install -r requirements.txt
```

## Capture

   For capturing images using the MicaSense Rededge-P, first connect to the built in web server via WiFi or ethernet (IP will vary depending, refer to capture.py and alter it for your use either in the file or in the function call) and run ```python ./capture.py``` to start. Position the reflective panel included with the camera 1 meter below in similar lighting to the capture area, and press enter. Then press enter again once you are ready to survey.

   The default survey time is 5 minutes, this may be altered as well. For implementation into the robot I would recommend making a distance based capture system rather than a timer based one to get a clearer picture of the field, just keep in mind the camera can only handle about 3 pictures per second so do not run the robot / person holding a stick too fast.

   The files are saved on the SD card inserted into the MicaSense camera, and this program expects you to take out and retrieve them manually. The web API does have functionality to access them remotely but this is not implemented in the capture program, please refer to the API [here](https://micasense.github.io/rededge-api/api/http.html#files) if you wish to alter it (you definitely should I was just short on time and had more to code).

   IMPORTANT: Make sure to put the fan on the EXHAUST blowing outward, if you printed the mount I designed (USE PETG NOT PLA AVAILABLE IN LAB I AM BEGGING YOU IT WILL BREAK) just use the zip ties on the little holes in the corner to keep it in place. Additionally you may want to add a cover for the RJ45 breakout board / 2.1mm connection, feel free. I didnt get to that point of testing so I never designed it but the holes on the connection side of the mount were intended for a cover to be added and zip tied on. 

## Image Processing

   Make sure to point the input and output dirs in the tiff_processing file to wherever you send your MicaSense images and wherever you want to predict / hand segment for training. Ideally you would make your own script combining the image processing and prediction steps if you wanted to streamline it, I imagine if you simply create the processed image without saving it, run it through the model, and then just save the mask and pull up the actual image if called upon (which if you're using a purely GPS based UI with a heat map displaying weighted crop / weed ratios you probably don't need to) you will save a lot of time in processing, I just made it this way to create a dataset rather than a tool.
   
   The tiff_processing.py file has some additional functions that you may find useful in a future project. It contains calculation functions for Normalized Difference Vegetation Index (NDVI), Normalized Difference Red Edge Index (NDRE), Green Normalized Difference Vegetation Index (GNDVI), and Optimized Soil Adjusted Vegetation Index (OSAVI), all of which have practical applications in phenotyping. Disease detection, nitrogen levels, canopy coverage, all of these are on the table and the dataset is compatible with all these indexes. Research them further and have fun with it, I want the set to get some more use.

   IMPORTANT: Make sure to set a reference image, it makes the process much faster and as long as your height is consistent with your chosen image it will work very well.

## Segmentation

   Sam V2 is required to run the segmentation tool, make sure [the repo](https://github.com/facebookresearch/sam2) is also installed and is pointed to correctly. I recommend following [this tutorial](https://www.youtube.com/watch?v=32MDGZUV0-M) to make sure you have everything installed, there are some additional installs you have to make to get the model to work.

   IMPORTANT: Make sure to point all the directories saved in segmentation_tool.py to the correct locations

   The tool is really easy to use and works best if you seed a lot of points at the start with a good first image pick. Play around and do what feels right, and feel free to add more classes if desired, it is designed in a way that should allow you to easily slot new ones in if you wanted to say classify grass separately or classify individual weed types. This may affect model performance, but experimentation is good.

## Training / Prediction

   There is a separate readme in unet_train for this section that goes more in depth as it is more complicated to run.

