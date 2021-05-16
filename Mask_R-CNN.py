import sys
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from Mask_RCNN.mrcnn.utils import Dataset
!{sys.executable} -m pip install scikit-image
import skimage


# extract bounding boxes from annotation
def extract_box(file):
    tree = ElementTree.parse(file)
    root = tree.getroot()
    
    # get box coordinates
    for box in root.findall('.//bndbox'):
        xmax = int(box.find('xmax').text)
        xmin = int(box.find('xmin').text)
        ymax = int(box.find('ymax').text)
        ymin = int(box.find('ymin').text)
        box_coords = [xmax, xmin, ymax, ymin]
    
    height = int(root.find('.//size/height').text)
    width = int(root.find('.//size/width').text)
    # return value is a list with box coordinates and the dimensions of pic overall
    return [box_coords, width, height]


print(extract_box('Images/data_training/boat1/000001.xml'))

# need to override the functionality from the github mask RCNN
class ObjectDataset(Dataset):
    
    def load_dataset(self, dataset_dir, is_train=True):
        self.add_class("dataset", 1, "boat")
        self.add_class("dataset", 2, "building")
        self.add_class("dataset", 3, "car")
        self.add_class("dataset", 4, "drone")
        self.add_class("dataset", 5, "person")
        self.add_class("dataset", 6, "horseride")
        self.add_class("dataset", 7, "paraglider")
        self.add_class("dataset", 8, "riding")
        self.add_class("dataset", 9, "group")
        
        # images 
        # folders to skip over
        dir_file_c = {}
        
        for d in listdir(dataset_dir):
            # create count for each file
            dir_file_c[d] = 0
            for f in listdir(d):
                dir_file_c[d]+=1
            dir_file_c[d]/=2
        return dir_file_c
        
    def load_masks(self, image_id):
        return 
        
    def image_reference(self, image_id):
        return
      
print(ObjectDataset().load_dataset('Images/data_training/boat1/000001.xml'))
