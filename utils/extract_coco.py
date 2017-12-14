from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt

ann_file = '/data/datasets/yanfu/annotations/instances_train2014.json'
coco=COCO(ann_file)

cats = coco.loadCats(coco.getCatIds())
catIds = coco.getCatIds(catNms=['car'])
imgIds = coco.getImgIds(catIds=catIds)
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
I = io.imread(img['coco_url'])
plt.axis('off')
plt.imshow(I)
plt.show()
plt.imshow(I); plt.axis('off')
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
coco.showAnns(anns)
