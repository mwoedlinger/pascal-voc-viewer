# pascal-voc-viewer
Allows to view Pascal VOC bounding boxes and to move unwanted files to another folder.

## Requirements
* `Python 3
* `OpenCV
* `tqdm 
* `numpy

## Usage

```shell
python viewer.py <options>
```

### Options
```
--folder          The input folder. Contains jpg images and xml files with the same basenames.
--out             The output folder where files will be moved to when *s* is pressed.
--class_file      Text file containing the classes and their colors seperated by spaces.
--display_width   The display width in pixels.
--display_height  The display height in pixels.
```

### Controls
The buttons **a** and **d** can be used to navigate to the previous and next image.
By pressing **s** the current image is moved to the folder *out*. **q** exits the program.
