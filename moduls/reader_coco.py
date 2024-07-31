from pycocotools import COCO
from skimage import io
from matplotlib import pyplot as plt

anno_file = '/Users/great/Downloads/dataset/coco'
coco = COCO(anno_file)

catIds = coco.getCatIds(catNms=['person'])
print(catIds)
imgIds = coco.getImgIds(catIds=catIds)
print(imgIds)

for i in range(len(imgIds)):
    image = coco.loadImages(imgIds[i])[0]
    I = io.imread = (image['coco_url'])
    plt.show(I)
    anno_id = coco.getAnnIds(imgIds=image['id'], catIds=catIds, iscrowd=None)
    annotation = coco.loadAnns(anno_id)
    coco.showAnns(annotation)
    plt.show()
    