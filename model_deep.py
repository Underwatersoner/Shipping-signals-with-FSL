import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
import analysis_utils as util

def build_siamese_vgg(input_shape, latent_dim):
    """
    Builds the Siamese network using VGG-16 as the backbone.
    """
    backbone = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    backbone.trainable = True # Fine-tuning enabled

    inputs = layers.Input(shape=input_shape)
    x = backbone(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(latent_dim, activation=None)(x) # 512-dim latent vector
    
    return models.Model(inputs, outputs)

@tf.function
def train_step(model, x_support, x_query, y_query, config, optimizer):
    """
    Performs a single training step using the proposed distribution-aware loss.
    """
    with tf.GradientTape() as tape:
        z_support = model(x_support)
        z_query = model(x_query)
        
        # 1. Classification Loss (with Hard Negative Softmax logic)
        # [Implementation of HNS...]
        
        # 2. Boundary Loss (Proposed)
        b_loss = util.boundary_negative_contrastive_loss(z_query, y_query, z_support, config)
        
        total_loss = (config["classification_weight"] * class_loss) + \
                     (config["boundary_loss_weight"] * b_loss)
                     
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return total_loss