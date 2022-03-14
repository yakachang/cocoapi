import jsonlines
import numpy as np
import skimage.io as io
from pycocotools.coco import COCO

dataDir = '../..'
dataType = 'val2017'

# initialize COCO api for instance annotations
annFile = '{}/annotations/instances_{}.json'.format(dataDir,dataType)
coco = COCO(annFile)

# initialize COCO api for caption annotations
annFile = '{}/annotations/captions_{}.json'.format(dataDir,dataType)
coco_caps = COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
supercategories = set([cat['supercategory'] for cat in cats])

output_file = '{}/annotations/instances_{}_captions.jsonl'.format(dataDir, dataType)
with jsonlines.open(output_file, mode='w') as writer:
    # Get all images containing given categories, select one at random
    # len(imgIds) = 57693
    for category in supercategories:
        catIds = coco.getCatIds(catNms=[category])
        imgIds = coco.getImgIds(catIds=catIds)
        for imgId in imgIds:
            image = coco.getImgIds(imgIds=imgId)
            img = coco.loadImgs(image[np.random.randint(0,len(image))])[0]

            I = io.imread(img['coco_url'])  # Use URL to read image

            # load and display caption annotations
            annIds = coco_caps.getAnnIds(imgIds=img['id'])
            anns = coco_caps.loadAnns(annIds)
            caption_list = []
            for item in anns:
                caption_list.append(item['caption'])
            writer.write({"image_id": img['id'], "captions": caption_list})