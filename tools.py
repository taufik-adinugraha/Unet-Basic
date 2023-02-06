import os
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import random

class pipeline():

  def __init__(self, img_dir, mask_dir, store_dir, prefix):
    self.img_dir = img_dir
    self.mask_dir = mask_dir
    self.store_dir = store_dir
    self.prefix = prefix
    self.seed = 1234
    self.split = 0.8
    self.batch_size = 16
    self.img_size = (256, 256)
    self.all_images = [i for i in os.listdir(img_dir) if i.split('.')[-1]=='jpg']
    self.all_masks = [f'{i.split('.')[0]}.png' for i in self.all_images]
    self.add_callbacks = []

  def dataset(self, prep):
    self.prep = prep
    random.seed(self.seed)
    random.shuffle(self.all_images)
    # split into training and validation
    lim = int(self.split * len(self.all_images))
    train_image_files = self.all_images[0:lim]
    train_mask_files = self.all_masks[0:lim]
    valid_image_files = self.all_images[lim:]
    valid_mask_files = self.all_masks[lim:]
    self.train_files = (train_image_files, train_mask_files)
    self.valid_files = (valid_image_files, valid_mask_files)
    
    data = []
    for (file_images, file_masks) in [self.train_files, self.valid_files]:    
      x = []
      y = []
      for (file_img, file_msk) in zip(file_images, file_masks):
        # preprocess the raw images 
        img = cv2.imread(os.path.join(self.img_dir, file_img))
        img = cv2.resize(img, self.img_size)
        # get the masks. Note that masks are png files 
        mask = cv2.imread(os.path.join(self.mask_dir, file_msk))
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        mask = cv2.resize(mask, self.img_size)
        # preprocess the mask 
        mask = np.where(mask==255, 1., 0.)
        # append
        x.append(img)
        y.append(mask)
      # preprocess a batch of images and masks 
      x = np.array(x) / 255.
      x = self.prep(x)
      y = np.array(y)

      data.append((x, y))

    self.train = data[0]
    self.valid = data[1]


  # sample images
  def sample_images(self):
    fig, ax = plt.subplots(7, 3, figsize=(18,12))
    k = 0
    for i in range(7):
      for j in range(3):
        f = self.all_images[k]
        m = self.all_masks[k]
        img = cv2.imread(os.path.join(self.img_dir, f))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        msk = cv2.imread(os.path.join(self.mask_dir, m))
        msk = np.where(msk==255, 255, 0)
        ax[i,j].axis('off')
        ax[i,j].imshow(np.concatenate([img, msk], axis = 1))
        k += 1
    plt.show()
    plt.tight_layout()


  # callbacks
  def customCallbacks(self):
    path = os.path.join(self.store_dir, f"unet_{self.prefix}.h5")
    checkpointer = ModelCheckpoint(
        filepath = path, 
        monitor = 'val_loss',
        # save_freq = int(100 * len(self.train_files) // self.batch_size),
        verbose = 0,
        save_best_only = True, 
        save_weights_only = True
        )
    callbacks = [
                 checkpointer, 
                 evaluation_callback(self.img_size, self.img_dir, self.valid_files, self.prep), 
                 ]
    return callbacks + self.add_callbacks


  # test
  def test(self, filenames, model):
    images = [os.path.join(self.store_dir, i) for i in filenames]
    raws, masks = [], []
    for img in images:
      raw_ori = cv2.imread(img)
      raw_ori = cv2.resize(raw_ori, self.img_size) 
      raw = raw_ori.copy() / 255.
      raw = self.prep(raw)
      # predict the mask 
      pred = model.predict(np.expand_dims(raw, 0))
      msk  = pred.squeeze()
      msk = np.stack((msk,)*3, axis=-1)
      msk[msk >= 0.5] = 1.
      msk[msk < 0.5] = 0.
      raws.append(cv2.cvtColor(raw_ori, cv2.COLOR_RGB2BGR)/255.)
      masks.append(msk)

    # show the mask and the segmented image 
    fig, ax = plt.subplots(len(images), 1, figsize=(18,4*len(images)))
    for i, (raw, msk) in enumerate(zip(raws,masks)):
      out = np.concatenate([raw, msk], axis = 1)
      ax[i].set_axis_off()
      ax[i].imshow(out)
    plt.show()    


# inheritance for training process plot 
class evaluation_callback(Callback):

    def __init__(self, img_size, img_dir, files, prep):
        self.sz = img_size
        self.img_dir = img_dir
        self.files = files
        self.prep = prep

    def on_train_begin(self, logs=None):
        self.i = 0
        self.x = []
        self.loss = []
        self.val_loss = []
        self.iou = []
        self.val_iou = []
        self.logs = []
        
    def on_epoch_end(self, epoch, logs=None):
        self.logs.append(logs)
        self.x.append(self.i)
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.iou.append(logs.get('iou_score'))
        self.val_iou.append(logs.get('val_iou_score'))
        
        # if self.i%25==0:
        #   # test image
        #   fig, ax = plt.subplots(2, 3, figsize=(18,4))
        #   k = 0
        #   for i in range(2):
        #     for j in range(3):
        #       image = self.files[k]
        #       raw_ori = cv2.imread(os.path.join(self.img_dir, image))
        #       raw_ori = cv2.resize(raw_ori, self.sz) 
        #       raw = raw_ori.copy() / 255.
        #       mask = cv2.imread(os.path.join(self.img_dir, 'segmentation', f"{image.split('.')[0]}.png"))
        #       mask = cv2.resize(mask, self.sz) / 255.
        #       mask = np.where(mask==1., 1., 0.)
              
        #       # predict the mask 
        #       raw = self.prep(raw)
        #       pred = self.model.predict(np.expand_dims(raw, 0))          
              
        #       # mask post-processing 
        #       pred_msk  = pred.squeeze()
        #       pred_msk = np.stack((pred_msk,)*3, axis=-1)
        #       pred_msk[pred_msk >= 0.5] = 1.
        #       pred_msk[pred_msk < 0.5] = 0.
              
        #       # show the mask and the segmented image 
        #       raw = cv2.cvtColor(raw_ori, cv2.COLOR_RGB2BGR) / 255.
        #       combined = np.concatenate([raw, mask, pred_msk], axis = 1)
        #       ax[i,j].set_axis_off()
        #       ax[i,j].imshow(combined)

        #       k += 1
        #   fig.suptitle(f'Epoch: {self.i}')
        #   plt.show()

        self.i += 1
