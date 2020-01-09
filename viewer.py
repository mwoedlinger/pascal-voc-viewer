import cv2
from tqdm import tqdm
import numpy as np
import xml.etree.ElementTree as ET
import argparse
import os

def load_class_dict(class_file: str):
    """
    Reads the class label file and returns a tuple consisting of a list of classes, a list of colors
    and a dictionary {class: color}
    :param class_file: The text file containg the class names and colors
    :return: A tuple of classes, colors and a dicitonary {class: color}
    """
    with open(class_file, 'r') as tf:
        text = tf.readlines()
    classes_file = [t.strip() for t in text]

    classes = [l.split()[0] for l in classes_file]
    colors = [[int(l.split()[1]), int(l.split()[2]), int(l.split()[3])] for l in classes_file]
    class_dict = {classes[n]: colors[n] for n in range(len(classes))}

    return classes, colors, class_dict


class xml_parser:
    """
    Reads a Pascal VOC XML file and generates the corresponding label files that contain of a box for the text regions.
    """
    def __init__(self, classes: list, color_dict: dict, xml_filename: str, thickness: int=3):
        # Load the XML file
        self.tree = ET.parse(os.path.join(xml_filename))
        self.root = self.tree.getroot()

        # Extract name and size data
        size = self.root.find('size')
        self.width = int(size.find('width').text)
        self.height = int(size.find('height').text)

        self.thickness = int(thickness*max(self.width/1000, self.height/1000))

        # Extract points
        self.bb_list = self.extract_bounding_boxes()

        """
        self.classes = ['bg', 'cancellation_sum', 'trash_position', 'trash_value', 'cancellation_position',
                        'cancellation_position_value', 'cancellation_person', 'cancellation_person_value',
                        'booking_sum', 'salutation', 'name', 'street', 'place', 'country', 'email', 'iban',
                        'booking_date', 'begin journey', 'cancellation_date', 'cancellation_rate', 'other']
        self.relevant_classes = ['booking_sum', 'cancellation_sum', 'trash_position', 'trash_value',
                                 'cancellation_position', 'cancellation_position_value', 'cancellation_person',
                                 'cancellation_person_value']
        self.class_numbers = {c: n for n, c in enumerate(self.relevant_classes)}
        self.class_numbers['cancellation_person'] = self.class_numbers['cancellation_position']
        self.class_numbers['cancellation_person_value'] = self.class_numbers['cancellation_position_value']
        self.colors = {0: (128, 128, 0), 1: (255, 0, 0), 2: (160, 160, 160), 3: (80, 80, 80), 4: (0, 255, 0), 5: (0, 0, 255)}
        """
        self.classes = classes
        self.color_dict = color_dict



    def extract_bounding_boxes(self) -> list:
        obj_list = []

        for n, obj in enumerate(self.root.iter('object')):
            name = obj.find('name').text
            bndbox = obj.find('bndbox')

            coords = {}
            for c in bndbox:
                coords.update({c.tag: int(c.text)})

            obj_list.append({'name': name, 'coords': coords})

        return obj_list

    def draw_mask(self, thickness: int=3) -> np.array:
        img = np.zeros((self.height, self.width, 3), np.uint8)

        for obj in self.bb_list:
            name = obj['name']
            """
            if name not in self.classes:
                name = 'other'
            """
            if name in self.classes:
                #class_number = self.class_numbers[name]

                coords = obj['coords']
                x_0 = coords['xmin']
                y_0 = coords['ymin']
                x_1 = coords['xmax']
                y_1 = coords['ymax']

                cv2.rectangle(img, (x_0, y_0), (x_1, y_1), self.color_dict[name], -1)#self.colors[self.class_numbers[name]], -1)

        return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Loops through all files in the given folder, plots the bounding boxes and allows the user to move the file to the "out" folder with "m".')
    parser.add_argument('--folder', help='The input folder.', required=True)
    parser.add_argument('--out', help='The output folder', required=True)
    parser.add_argument('--class_file', help='Text file containing the classes and their colors seperated by spaces.', required=True)
    parser.add_argument('--display_width', help='The display width (pixels)', required=False, default=1920)
    parser.add_argument('--display_height', help='The display height (pixels)', required=False, default=1080)
    args = vars(parser.parse_args())

    in_folder = args['folder']
    out_folder = args['out']
    display_height = args['display_height']
    display_width = args['display_width']
    class_file = args['class_file']

    classes, colors, class_dict = load_class_dict(class_file)

    
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
        print('created {} folder'.format(out_folder))

    file_list = []
    for root, directory, files in os.walk(in_folder):
        file_list += [os.path.join(root, f) for f in files]

    xml_list = [f for f in file_list if os.path.basename(f).split('.')[-1] == 'xml']

    n = 0
    pbar = tqdm(total = len(xml_list)-1)

    while(1):
        pbar.n = n
        pbar.total = len(xml_list)-1
        pbar.refresh()
        

        xml_file = xml_list[n]
        png_file = xml_file[:-3]+'jpg'

        png_image = cv2.imread(png_file)
        parser = xml_parser(classes, class_dict, xml_file)
        mask = parser.draw_mask()
        
        img = cv2.addWeighted(png_image, 0.7, mask, 0.3, 0)

        h, w, _ = img.shape

        h_ratio = display_height/h
        w_ratio = display_width/w

        ratio = min(h_ratio, w_ratio)

        if ratio < 1:
            h = int(h*ratio)            
            w = int(w*ratio)

        cv2.namedWindow(png_file, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(png_file, (w, h))
        cv2.imshow(png_file,img)
        key = cv2.waitKey(0)

        if key == 113: #q
            print('quit')
            cv2.destroyAllWindows()
            break
        elif key == 97: #a
            if n == 0:
                n = len(xml_list)-1
            else:
                n -= 1
        elif key == 100: #d
            if n == len(xml_list)-1:
                n = 0
            else:
                n += 1
        elif key == 115: #s
            os.rename(png_file, os.path.join(out_folder, os.path.basename(png_file)))
            os.rename(xml_file, os.path.join(out_folder, os.path.basename(xml_file)))
            #print('## Moved {}'.format(os.path.basename(png_file)))
   
            file_list = []
            for root, directory, files in os.walk(in_folder):
                file_list += [os.path.join(root, f) for f in files]

            xml_list = [f for f in file_list if os.path.basename(f).split('.')[-1] == 'xml']

        cv2.destroyAllWindows()
    pbar.close()
