import os
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt


def image_generator_old(files, batch_size=32, sz=(256, 256)):
  
  while True: 
    
    # extract a random batch 
    batch = np.random.choice(files, size=batch_size)    
    
    # variables for collecting batches of inputs and outputs 
    batch_x = []
    batch_y = []
    
    for f in batch:

        #get the masks. Note that masks are png files 
        mask = Image.open(f'annotations/trimaps/{f[:-4]}.png')
        mask = np.array(mask.resize(sz))

        # preprocess the mask 
        mask = np.where(mask==1, 1, 0)
        
        batch_y.append(mask)

        # preprocess the raw images 
        raw = Image.open(f'images/{f}')
        raw = raw.resize(sz)
        raw = np.array(raw)

        # check the number of channels because some of the images are RGBA or GRAY
        if len(raw.shape) == 2:
          raw = np.stack((raw,)*3, axis=-1)
        else:
          raw = raw[:,:,0:3]

        batch_x.append(raw)

    # preprocess a batch of images and masks 
    batch_x = np.array(batch_x)/255.
    batch_y = np.array(batch_y)
    batch_y = np.expand_dims(batch_y,3)

    yield (batch_x, batch_y)    



def image_generator(img_dir, mask_dir, batch_size=32, img_size=(256, 256)):
  
  while True: 
    
    train_images = [i for i in os.listdir(img_dir) if i.split('.')[-1]=='jpg']
    masks = [i for i in os.listdir(mask_dir) if i.split('.')[-1]=='png']

    # extract a random batch 
    batch = np.random.choice(train_images, size=batch_size)    
    
    # variables for collecting batches of inputs and outputs 
    batch_x = []
    batch_y = []
    
    for f in batch:

        #get the masks. Note that masks are png files 
        mask = cv2.imread(f"{mask_dir}/{f.split('.')[0]}.png")
        mask = cv2.resize(mask, img_size)
        # preprocess the mask 
        mask = np.where(mask==255, 1, 0)

        # preprocess the raw images 
        img = cv2.imread(f'{img_dir}/{f}')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, img_size)

        # append
        batch_y.append(mask)
        batch_x.append(img)

    # preprocess a batch of images and masks 
    batch_x = np.array(batch_x)/255.
    batch_y = np.array(batch_y)

    yield (batch_x, batch_y)   


# sample images
def sample_images(image_gen):
  fig, ax = plt.subplots(5, 3, figsize=(18,15))
  x, y = next(image_gen)
  k = 0
  for i in range(5):
    for j in range(3):
      img = x[k]
      msk = y[k]
      ax[i,j].axis('off')
      ax[i,j].imshow(np.concatenate([img, msk], axis = 1))
      k += 1


# callbacks
def customCallbacks(path, images):
  checkpointer = ModelCheckpoint(filepath=f'{path}/unet.h5', verbose=0, save_best_only=True, save_weights_only=True)
  callbacks = [checkpointer, evaluation_callback(images), EarlyStopping(patience=3)]
  return callbacks


# inheritance for training process plot 
class evaluation_callback(Callback):

    def __init__(self, images):
        self.images = images

    def on_train_begin(self, logs=None):
        self.sz = (256,256)
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.logs = []
        
    def on_epoch_end(self, epoch, logs=None):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('mean_iou'))
        self.val_acc.append(logs.get('val_mean_iou'))
        self.i += 1
        print(f'i={self.i}')
        print(f"loss={logs.get('loss')}, val_loss={logs.get('val_loss')}, MeanIoU={logs.get('mean_io_u')}, val_MeanIoU={logs.get('val_mean_io_u')}")
        
        # test image
        fig, ax = plt.subplots(1, 3, figsize=(18,5))
        for i, image in enumerate(self.images[:3]):
          raw = cv2.imread(f'train/{image}')
          raw = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
          raw = cv2.resize(raw, self.sz)/255.
          # check the number of channels because some of the images are RGBA or GRAY
          if len(raw.shape) == 2:
            raw = np.stack((raw,)*3, axis=-1)
          else:
            raw = raw[:,:,0:3]
          
          #predict the mask 
          pred = self.model.predict(np.expand_dims(raw, 0))
          
          #mask post-processing 
          msk  = pred.squeeze()
          msk = np.stack((msk,)*3, axis=-1)
          msk[msk >= 0.5] = 1 
          msk[msk < 0.5] = 0 
          
          #show the mask and the segmented image 
          # combined = np.concatenate([raw, msk, raw* msk], axis = 1)
          combined = np.concatenate([raw, msk], axis = 1)
          ax[i].set_axis_off()
          ax[i].imshow(combined)
        plt.show()