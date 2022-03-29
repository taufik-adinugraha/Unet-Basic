import numpy as np
from PIL import Image
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt


def image_generator(files, batch_size=32, sz=(256, 256)):
  
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
        mask[mask >= 2] = 0 
        mask[mask != 0 ] = 1
        
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
        fig, ax = plt.subplots(2, 4, figsize=(18,2))
        for i, image in enumerate(self.images):
          raw = Image.open(f'images/{self.image}')
          raw = np.array(raw.resize(self.sz))/255.
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
          combined = np.concatenate([raw, msk, raw* msk], axis = 1)
          ax[i].axis('off')
          ax[i].imshow(combined)
        plt.show()