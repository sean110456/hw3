import os
import numpy as np
# import cv2
# import matplotlib.pyplot as plt
from mrcnn.config import Config
from mrcnn import model as modellib, utils
#from mrcnn import visualize
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import imgaug
import cv2
import json

# Directory to save logs and trained model
MODEL_DIR = 'logs'


############################################################
#  Configurations
############################################################
class MyConfig(Config):
    """
    Derives from the base Config class and overrides values for the task
    """
    # Give the configuration a recognizable name
    NAME = "hw3"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 20  # background + 20 categories

    # Image resizing
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 320
    IMAGE_MAX_DIM = 448

    # Run all images
    STEPS_PER_EPOCH = 1350


config = MyConfig()
config.display()


############################################################
#  Dataset
############################################################
class MyDataset(utils.Dataset):
    def load_datas(self, json_fname, image_dir):
        """Load the annotations
        """
        coco = COCO(json_fname)  # load training annotations

        # Add classes
        for i in range(len(coco.cats)):
            self.add_class('hw3', i+1, coco.cats[i+1]['name'])

        # Add images
        image_ids = list(coco.imgs.keys())
        for i in image_ids:
            self.add_image(
                'hw3', image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]['width'],
                height=coco.imgs[i]['height'],
                annotations=coco.loadAnns(coco.getAnnIds(imgIds=i)))

    def load_mask(self, image_id):
        """Load instance masks for the given image.
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        annotations = self.image_info[image_id]['annotations']
        class_ids = []
        instance_masks = []
        for annotation in annotations:
            # id conversion
            class_id = self.map_source_class_id(
                'hw3.{}'.format(annotation['category_id']))
            mask = self.annToMask(
                annotation, image_info['height'], image_info['width'])
            class_ids.append(class_id)
            instance_masks.append(mask)
            # No need to deal with iscrowd since the iscrowd is all 0

        mask = np.stack(instance_masks, axis=2).astype(np.bool)
        class_ids = np.array(class_ids, dtype=np.int32)
        return mask, class_ids

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info['source'] == 'hw3':
            return info['path']

# Prepare the train dataset and validation dataset
dataset_train = MyDataset()
dataset_train.load_datas('pascal_train.json','train_images')
dataset_train.prepare()

dataset_val = MyDataset()
dataset_val.load_datas('pascal_train.json','train_images')
dataset_val.prepare()

"""# debug: load_mask
image_ids = np.random.choice(dataset_train._image_ids, 4)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)
    plt.savefig(str(image_id)+'.png')
# """

############################################################
#  Training
############################################################

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

init_with = "imagenet"  # imagenet or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)

elif init_with == "last":
    # Load the last model you trained and continue training
    print('loading: ',model.find_last())
    model.load_weights(model.find_last(), by_name=True)

# Image Augmentation: Right/Left flip 50% of the time
augmentation = imgaug.augmenters.Fliplr(0.5)

# Training - Stage 1
# Train the head branches
print('Training network heads')
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=20,
            layers='heads',
            augmentation=augmentation)

# Training - Stage 2
# Finetune layers from ResNet stage 4 and up
print("Finetune Resnet stage 4 and up")
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=40,
            layers='4+',
            augmentation=augmentation)


# Fine tune all layers
print('Finetune all layers')
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=80,
            layers="all")


############################################################
#  Evaluation
############################################################
from utils import binary_mask_to_rle

class InferenceConfig(MyConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    #DETECTION_MIN_CONFIDENCE = 0

config = InferenceConfig()
config.display()

model = modellib.MaskRCNN(mode="inference", config=config,
                          model_dir=MODEL_DIR)
model.load_weights(model.find_last(), by_name=True)

cocoGt = COCO("test.json")
coco_dt = []

"""# debug: output the image with masks (set imgid)
imgid = 34
image = cv2.imread("test_images/" + cocoGt.loadImgs(ids=imgid)[0]['file_name'])[:,:,::-1]
results = model.detect([image],verbose=1)
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            dataset_train.class_names, r['scores'])
plt.savefig(str(imgid)+'.png')
"""

for imgid in cocoGt.imgs:
    image = cv2.imread("test_images/" + cocoGt.loadImgs(ids=imgid)[0]['file_name'])[:,:,::-1]
    results = model.detect([image])
    r = results[0]
    n_instances = len(r['scores'])
    for i in range(n_instances):
        pred = {}
        pred['image_id'] = imgid
        pred['category_id'] = int(r['class_ids'][i])
        pred['segmentation'] = binary_mask_to_rle(r['masks'][:,:,i])
        pred['score'] = float(r['scores'][i])
        coco_dt.append(pred)
with open('309551082.json', 'w') as f:
    json.dump(coco_dt, f)







