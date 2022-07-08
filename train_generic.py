#Tensorflow
import warnings
warnings.filterwarnings("ignore")
import tensorflow
from tensorflow.keras import regularizers
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionResNetV2
from pathlib import Path
import json

def train_generic(training_path, validation_path, output_path, devices):

    training_data_path = training_path
    validation_data_path = validation_path

    num_classes = 3
    image_size = 299

    with open('./hyperparameters.json') as in_file:

        hyperparameters = json.load(in_file)
        batch_size = hyperparameters['batch_size']
        epochs = hyperparameters['epochs']
        weight_decay = hyperparameters['weight_decay']
        learnable_layers = hyperparameters['learnable_layers']
        learning_rate = hyperparameters['learning_rate']
        learning_rate_decay = hyperparameters['learning_rate_decay']
        dropout_rate = hyperparameters['dropout_rate']

    training_data_generator = ImageDataGenerator(rescale = 1.0/255.0, rotation_range = 60, width_shift_range = .35, height_shift_range = .15, horizontal_flip = False, fill_mode = 'nearest') #Rescale images to 0 to .255
    validation_data_generator = ImageDataGenerator(rescale = 1.0/255.0)

    training_data_gen = training_data_generator.flow_from_directory(str(training_data_path),  shuffle = True, target_size = (image_size,image_size), batch_size = batch_size, color_mode = 'rgb', class_mode = "categorical")
    train_len = len(training_data_gen.classes)

    validation_data_gen = validation_data_generator.flow_from_directory(str(validation_data_path),  shuffle = True, target_size = (image_size,image_size), batch_size = batch_size, color_mode = 'rgb', class_mode = "categorical")
    val_len = len(validation_data_gen.classes)

    mirrored_strategy = tensorflow.distribute.MirroredStrategy(devices)

    with mirrored_strategy.scope():

        inputs = layers.Input(shape = (image_size, image_size, 3))

        x = tensorflow.keras.models.Sequential()(inputs)

        model = InceptionResNetV2(include_top = False, weights = 'imagenet', input_shape = (image_size,image_size,3), input_tensor = x)#(normalized)
        model.trainable = False
        x = layers.GlobalAveragePooling2D()(model.output)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(num_classes, activation = "softmax")(x)
        final_model = tensorflow.keras.Model(inputs, outputs, name = "Generic_Echo_Model")

        make_trainable = round(len(final_model.layers) * learnable_layers)

        for layer in final_model.layers[-make_trainable:]:

            layer.trainable = True

            if isinstance(layer, layers.Conv2D) or isinstance(layer, layers.Dense):

                layer.add_loss(lambda: regularizers.l2(weight_decay)(layer.kernel))
            
            if hasattr(layer, 'bias_regularizer') and layer.use_bias:

                layer.add_loss(lambda: regularizers.l2(weight_decay)(layer.bias))

        reduce_lr  = ReduceLROnPlateau(monitor = 'val_loss', mode = 'min', patience = 10, factor = learning_rate_decay, min_lr = .000000001, verbose = 2)

        final_model.compile(optimizer = tensorflow.keras.optimizers.Adam(learning_rate = learning_rate), loss = "categorical_crossentropy", metrics = ["accuracy"])

        final_model.fit(training_data_gen, epochs = epochs, steps_per_epoch = train_len//batch_size, validation_data=validation_data_gen, validation_steps = val_len//batch_size, verbose = 1, callbacks = [reduce_lr], use_multiprocessing = True, workers = 64)
        final_model.save(output_path + "/generic_classifier.h5")

    return "Finished training Generic"