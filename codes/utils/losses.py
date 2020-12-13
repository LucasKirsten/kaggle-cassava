import tensorflow as tf

def smooth_labels(labels, factor=0.1):
	# smooth the labels
	labels *= (1 - factor)
	labels += (factor / labels.shape[1])
	# returned the smoothed labels
	return labels
    
def smooth_categorical_crossentropy(factor=0.1):

    def loss(y_true, y_pred):
        y_true = smooth_labels(y_true, factor)
        return tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    
    return loss