import tensorflow as tf

class FreezeBackbone(tf.keras.callbacks.Callback):
    def __init__(self, backbone_name, epochs, verbose=True):
        self.backbone_name = backbone_name
        self.epochs  = epochs
        self.verbose = verbose
        
    def on_epoch_begin(self, epoch, logs=None):
        
        if epoch<self.epochs:
            if self.verbose and epoch==0:
                print(f'\nFreezing backbone {self.backbone_name}')
            self.model.get_layer(self.backbone_name).trainable = False
        else:
            if self.verbose:
                print(f'\nEpoch {epoch+1}: Unfreezing backbone\n')
            self.model.get_layer(self.backbone_name).trainable = True