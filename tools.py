import os
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from random import shuffle

class pipeline():

  def __init__(self, img_dir, mask_dir, store_dir):
    self.img_dir = img_dir
    self.mask_dir = mask_dir
    self.store_dir = store_dir
    self.split = 0.8
    self.batch_size = 16
    self.img_size = (256, 256)
    self.all_images = [i for i in os.listdir(img_dir) if i.split('.')[-1]=='jpg']
    self.all_masks = [i for i in os.listdir(mask_dir) if i.split('.')[-1]=='png']

  def dataset(self, prep):
    shuffle(self.all_images)
    # split into training and testing
    lim = int(self.split * len(self.all_images))
    self.train_files = self.all_images[0:lim]
    self.valid_files = self.all_images[lim:]

    data = []
    for files in [self.train_files, self.valid_files]:    
      # variables for collecting batches of inputs and outputs 
      x = []
      y = []
      for f in files:
        # preprocess the raw images 
        img = cv2.imread(os.path.join(self.img_dir, f))
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, self.img_size)
        # get the masks. Note that masks are png files 
        mask = cv2.imread(os.path.join(self.mask_dir, f"{f.split('.')[0]}.png"))
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        mask = cv2.resize(mask, self.img_size)
        # preprocess the mask 
        mask = np.where(mask==255, 1., 0.)
        # append
        x.append(img)
        y.append(mask)
      # preprocess a batch of images and masks 
      x = np.array(x) / 255.
      x = prep(x)
      y = np.array(y)

      data.append((x, y))

    self.train = data[0]
    self.valid = data[1]


  # sample images
  def sample_images(self):
    fig, ax = plt.subplots(5, 3, figsize=(18,8))
    k = 0
    for i in range(5):
      for j in range(3):
        f = self.all_images[k]
        img = cv2.imread(os.path.join(self.img_dir, f))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        msk = cv2.imread(os.path.join(self.mask_dir, f"{f.split('.')[0]}.png"))
        ax[i,j].axis('off')
        ax[i,j].imshow(np.concatenate([img, msk], axis = 1))
        k += 1
    plt.show()
    plt.tight_layout()


  # callbacks
  def customCallbacks(self):
    path = os.path.join(self.store_dir, "unet.h5")
    checkpointer = ModelCheckpoint(
        filepath = path, 
        verbose = 0,
        save_best_only = True, 
        save_weights_only = True
        )
    callbacks = [
                 checkpointer, 
                 evaluation_callback(self.img_size, self.img_dir, self.valid_files), 
                 EarlyStopping(patience=3)
                 ]
    return callbacks

  # test
  def test(self, filenames, model):
    images = [os.path.join(self.store_dir, i) for i in filenames]
    raws, masks = [], []
    for img in images:
      raw = cv2.imread(img)
      raw = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
      raw = cv2.resize(raw, self.img_size) / 255.
      # predict the mask 
      pred = model.predict(np.expand_dims(raw, 0))
      msk  = pred.squeeze()
      msk = np.stack((msk,)*3, axis=-1)
      msk[msk >= 0.5] = 1.
      msk[msk < 0.5] = 0.
      raws.append(raw)
      masks.append(msk)

    # show the mask and the segmented image 
    fig, ax = plt.subplots(1, 2, figsize=(18,6))
    for i, (raw, msk) in enumerate(zip(raws,masks)):
      out = np.concatenate([raw, msk], axis = 1)
      ax[i].set_axis_off()
      ax[i].imshow(out)
    plt.show()    


# inheritance for training process plot 
class evaluation_callback(Callback):

    def __init__(self, img_size, img_dir, files):
        self.sz = img_size
        self.img_dir = img_dir
        self.files = files

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
        self.i += 1
        
        if self.i%10 == 0 | self.i==1 :
          # test image
          fig, ax = plt.subplots(1, 3, figsize=(18,5))
          for i, image in enumerate(self.files[:3]):
            raw = cv2.imread(os.path.join(self.img_dir, image))
            raw = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
            raw = cv2.resize(raw, self.sz) / 255.
            
            # predict the mask 
            pred = self.model.predict(np.expand_dims(raw, 0))          
            
            # mask post-processing 
            msk  = pred.squeeze()
            msk = np.stack((msk,)*3, axis=-1)
            msk[msk >= 0.5] = 1.
            msk[msk < 0.5] = 0.
            
            # show the mask and the segmented image 
            combined = np.concatenate([raw, msk], axis = 1)
            ax[i].set_axis_off()
            ax[i].imshow(combined)
          plt.show()