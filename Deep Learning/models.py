"""
This file contains various Deep Learning Models.

Each function should return a keras Model.

"""
import keras
import tensorflow as tf
import numpy as np

######################
##  AlexNet         ##
######################

def alex_net(input_shape, nb_classes) -> keras.Model:
    X_input = keras.layers.Input(input_shape)

    X = keras.layers.Conv2D(96,(11,11),strides = 4,name="conv0")(X_input)
    X = keras.layers.BatchNormalization(axis = 3 , name = "bn0")(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.MaxPool2D((3,3),strides = 2, padding = 'same',name = 'max0')(X)

    X = keras.layers.Conv2D(256,(5,5),padding = 'same' , name = 'conv1')(X)
    X = keras.layers.BatchNormalization(axis = 3 ,name='bn1')(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.MaxPool2D((3,3),strides = 2, padding = 'same',name = 'max1')(X)

    X = keras.layers.Conv2D(384, (3,3) , padding = 'same' , name='conv2')(X)
    X = keras.layers.BatchNormalization(axis = 3, name = 'bn2')(X)
    X = keras.layers.Activation('relu')(X)

    X = keras.layers.Conv2D(384, (3,3) , padding = 'same' , name='conv3')(X)
    X = keras.layers.BatchNormalization(axis = 3, name = 'bn3')(X)
    X = keras.layers.Activation('relu')(X)

    X = keras.layers.Conv2D(256, (3,3) , padding = 'same' , name='conv4')(X)
    X = keras.layers.BatchNormalization(axis = 3, name = 'bn4')(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.MaxPool2D((3,3),strides = 2, padding = 'same',name = 'max2')(X)
    
    X = keras.layers.Flatten()(X)
    
    X = keras.layers.Dense(4096, activation = 'relu', name = "fc0")(X)
    X = keras.layers.Dropout(0.5)(X)
    X = keras.layers.Dense(4096, activation = 'relu', name = 'fc1')(X) 
    X = keras.layers.Dropout(0.5)(X)
    X = keras.layers.Dense(nb_classes,activation='softmax',name = 'fc2')(X)
    
    model = keras.Model(inputs = X_input, outputs = X, name='AlexNet')
    
    return model


######################
##  ResNet50        ##
######################

def res_net_50(input_shape, nb_classes) -> keras.Model:
    return  keras.applications.resnet.ResNet50(
        include_top=True, 
        weights=None, 
        input_shape=input_shape, 
        classes=nb_classes
    )


######################
##  EfficientNetB0  ##
######################

def efficient_net_b0(input_shape, nb_classes) -> keras.Model:
    return keras.applications.efficientnet.EfficientNetB0(
        include_top=True,
        weights=None,
        input_shape=input_shape,
        classes=nb_classes
    )


######################
##  VGG16           ##
######################

def vgg_16(input_shape, nb_classes) -> keras.Model:
    return keras.applications.vgg16.VGG16(
        include_top=True,
        weights=None,
        input_shape=input_shape,
        classes=nb_classes
    )



######################
##  1D CNN          ##
######################

def simple_3_layer_1d_cnn(
        input_shape, 
        nb_classes,
        filters: tuple[int, int, int] = (32, 64, 128),
        kernals: tuple[int, int, int] = (2, 2, 2),
        units: tuple[int, int, int] = (512, 256, 128)
):
    X_input = keras.layers.Input(input_shape)

    X = keras.layers.Conv1D(filters[0], (kernals[0]), padding="same", activation="relu")(X_input)
    X = keras.layers.Conv1D(filters[1], (kernals[1]), padding="same", activation="relu")(X)
    X = keras.layers.Conv1D(filters[2], (kernals[2]), padding="same", activation="relu")(X)

    X = keras.layers.Flatten()(X)

    X = keras.layers.Dense(units[0], activation="relu")(X)
    X = keras.layers.Dense(units[1], activation="relu")(X)
    X = keras.layers.Dense(units[2], activation="relu")(X)

    X = keras.layers.Dense(nb_classes, activation="softmax")(X)

    model = keras.Model(inputs=X_input, outputs=X, name="3-Layer-1D-CNN")

    return model

def simple_6_layer_1d_cnn(
        input_shape, 
        nb_classes,
        filters: tuple[int, int, int] = (32, 64, 128),
        kernals: tuple[int, int, int] = (2, 2, 2),
        units: tuple[int, int, int] = (512, 256, 128)
):
    X_input = keras.layers.Input(input_shape)

    X = keras.layers.Conv1D(filters[0], (kernals[0]), padding="same", activation="relu")(X_input)
    X = keras.layers.Conv1D(filters[0], (kernals[0]), padding="same", activation="relu")(X)
    X = keras.layers.Conv1D(filters[1], (kernals[1]), padding="same", activation="relu")(X)
    X = keras.layers.Conv1D(filters[1], (kernals[1]), padding="same", activation="relu")(X)
    X = keras.layers.Conv1D(filters[2], (kernals[2]), padding="same", activation="relu")(X)
    X = keras.layers.Conv1D(filters[2], (kernals[2]), padding="same", activation="relu")(X)
    
    X = keras.layers.Flatten()(X)

    X = keras.layers.Dense(units[0], activation="relu")(X)
    X = keras.layers.Dense(units[1], activation="relu")(X)
    X = keras.layers.Dense(units[2], activation="relu")(X)

    X = keras.layers.Dense(nb_classes, activation="softmax")(X)

    model = keras.Model(inputs=X_input, outputs=X, name="6-Layer-1D-CNN")

    return model



######################
##  TabNet          ##
######################

def tab_net(input_shape, nb_classes) -> keras.Model:
    from Model_TabNet import TabNet

    X_input = keras.layers.Input(input_shape)
    
    X = TabNet(
            num_features=6,
            feature_dim=4,
            output_dim=2,
            num_decision_steps=5,
            relaxation_factor=1.5,
            batch_momentum=0.7,
            virtual_batch_size=4,
            num_classes=4
        )(X_input)

    model = keras.Model(inputs = X_input, outputs = X, name='TabNet')
    
    return model

class TabNet(keras.Model):

    def __init__(self,
                 num_features,
                 feature_dim,
                 output_dim,
                 num_decision_steps,
                 relaxation_factor,
                 batch_momentum,
                 virtual_batch_size,
                 num_classes,
                 epsilon=0.00001):  
        super().__init__()
        self.num_features = num_features
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.num_decision_steps = num_decision_steps
        self.relaxation_factor = relaxation_factor
        self.batch_momentum = batch_momentum
        self.virtual_batch_size = virtual_batch_size
        self.num_classes = num_classes
        self.epsilon = epsilon

        self.input_bn = keras.layers.BatchNormalization(momentum=self.batch_momentum)

        self.indt1 = keras.layers.Dense(units=self.feature_dim * 2, use_bias=False)
        self.inbn1 = keras.layers.BatchNormalization(momentum=self.batch_momentum)

        self.indt2 = keras.layers.Dense(units=self.feature_dim * 2, use_bias=False)
        self.inbn2 = keras.layers.BatchNormalization(momentum=self.batch_momentum)

        self.dt1 = keras.layers.Dense(units=self.feature_dim * 2, use_bias=False)
        self.bn1 = keras.layers.BatchNormalization(momentum=self.batch_momentum)

        self.dt2 = keras.layers.Dense(units=self.feature_dim * 2, use_bias=False)
        self.bn2 = keras.layers.BatchNormalization(momentum=self.batch_momentum)

        self.dt3s = list(map(
                lambda i: keras.layers.Dense(units=self.feature_dim * 2, use_bias=False),
                range(num_decision_steps)))
        self.bn3s = list(map(
                lambda i: keras.layers.BatchNormalization(momentum=self.batch_momentum),
                range(num_decision_steps)))
        
        self.dt4s = list(map(
                lambda i: keras.layers.Dense(units=self.feature_dim * 2, use_bias=False),
                range(num_decision_steps)))
        self.bn4s = list(map(
                lambda i: keras.layers.BatchNormalization(momentum=self.batch_momentum),
                range(num_decision_steps)))
        
        self.attds = list(map(
                lambda i: keras.layers.Dense(units=self.num_features, use_bias=False),
                range(num_decision_steps-1)))
        self.attbns = list(map(
                lambda i: keras.layers.BatchNormalization(momentum=self.batch_momentum),
                range(num_decision_steps-1)))

        self.out = keras.layers.Dense(4, activation="softmax", use_bias=False)


    def call(self, X):

        features = self.input_bn(X)
        batch_size = keras.ops.shape(features)[0]
        
        output_agg = keras.ops.zeros([batch_size, self.output_dim])
        masked_features = features
        mask_values = keras.ops.zeros([batch_size, self.num_features])
        agg_mask_values = keras.ops.zeros([batch_size, self.num_features])
        comp_agg_mask_values = keras.ops.ones([batch_size, self.num_features])
        tot_entropy = 0

        for ni in range(self.num_decision_steps):

            if ni == 0:
                transform1 = self.indt1(masked_features)
                transform1 = self.inbn1(transform1)
                transform1 = glu(transform1, self.feature_dim)

                transform2 = self.indt2(transform1)
                transform2 = self.inbn2(transform2)
                transform2 = (glu(transform2, self.feature_dim) + transform1) * np.sqrt(0.5)    
            else:
                transform1 = self.dt1(masked_features)
                transform1 = self.bn1(transform1)
                transform1 = glu(transform1, self.feature_dim)

                transform2 = self.dt2(transform1)
                transform2 = self.bn2(transform2)
                transform2 = (glu(transform2, self.feature_dim) + transform1) * np.sqrt(0.5)

            transform3 = self.dt3s[ni](transform2)
            transform3 = self.bn3s[ni](transform3)
            transform3 = (glu(transform3, self.feature_dim) + transform2) * np.sqrt(0.5)

            transform4 = self.dt4s[ni](transform3)
            transform4 = self.bn4s[ni](transform4)
            transform4 = (glu(transform4, self.feature_dim) + transform3) * np.sqrt(0.5)

            if ni > 0:
                decision_out = keras.ops.relu(transform4[:, :self.output_dim])
                output_agg += decision_out
                scale_agg = tf.reduce_sum(decision_out, axis=1, keepdims=True) / (self.num_decision_steps - 1)
                agg_mask_values += mask_values * scale_agg

            features_for_coef = (transform4[:, self.output_dim:])

            if ni < self.num_decision_steps - 1:

                mask_values = self.attds[ni](features_for_coef)
                mask_values = self.attbns[ni](mask_values)
                mask_values *= comp_agg_mask_values
                mask_values = keras.ops.sparsemax(mask_values)

                comp_agg_mask_values *= (self.relaxation_factor - mask_values)

                tot_entropy += tf.reduce_mean(
                    tf.reduce_sum(
                        -mask_values * keras.ops.log(mask_values + self.epsilon), axis = 1
                    ) / (self.num_decision_steps - 1)
                )

                masked_features = keras.ops.multiply(mask_values, features)

        out = self.out(output_agg)
        
        return out



def glu(act, n_units):
  """Generalized linear unit nonlinear activation."""
  return act[:, :n_units] * keras.ops.sigmoid(act[:, n_units:])