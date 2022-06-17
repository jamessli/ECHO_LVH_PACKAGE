#Tensorflow
import tensorflow
from tensorflow.keras import regularizers
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionResNetV2
from pathlib import Path

def train_generic(training_path, validation_path, output_path, devices):

    training_data_path = training_path
    validation_data_path = validation_path

    batch_size = 32 #4Gpus, 64 batches each?
    num_classes = 3
    epochs = 50
    image_size = 299
    weight_decay = .3
    learnable_layers = .5
    learning_rate = ".0005"
    learning_rate_decay = .5
    dropout_rate = .4

    if Path('./hyperparameters.txt').is_file():

        with open('./hyperparameters.txt') as in_file:

            lines = in_file.readlines()

            for line in lines:

                if line == "batch_size":

                    batch_size = int(line.split('=')[-1])

                if line == "epochs":

                    epochs = int(line.split('=')[-1])

                if line == "weight_decay":

                    weight_decay = float(line.split('=')[-1])

                if line == "learnable_layers":

                    learnable_layers = float(line.split('=')[-1])

                if line == "learning_rate":

                    learning_rate = float(line.split('=')[-1])

                if line == "learning_rate_decay":

                    learning_rate_decay = float(line.split('=')[-1])

                if line == "dropout_rate":

                    dropout_rate = float(line.split('=')[-1])

    training_data_generator = ImageDataGenerator(rescale = 1.0/255.0, rotation_range = 60, width_shift_range = .35, height_shift_range = .15, horizontal_flip = False, fill_mode = 'nearest') #Rescale images to 0 to .255
    validation_data_generator = ImageDataGenerator(rescale = 1.0/255.0)

    training_data_gen = training_data_generator.flow_from_directory(str(training_data_path),  shuffle = True, target_size = (image_size,image_size), batch_size = batch_size, color_mode = 'rgb', class_mode = "categorical")
    train_len = len(training_data_gen.classes)
    print("Training data: ", training_data_gen.class_indices)

    validation_data_gen = validation_data_generator.flow_from_directory(str(validation_data_path),  shuffle = True, target_size = (image_size,image_size), batch_size = batch_size, color_mode = 'rgb', class_mode = "categorical")
    val_len = len(validation_data_gen.classes)

    print("Validation data: ", validation_data_gen.class_indices)

    if len(devices) > 1:

        mirrored_strategy = tensorflow.distribute.MirroredStrategy(devices)

        with mirrored_strategy.scope():

            inputs = layers.Input(shape = (image_size, image_size, 3))

            x = tensorflow.keras.models.Sequential()(inputs)

            model = InceptionResNetV2(include_top = True, weights = 'imagenet', input_shape = (image_size,image_size,3), input_tensor = x)#(normalized)
            model.trainable = False
            x = layers.GlobalMaxPooling2D()(model.output)
            x = layers.Dropout(dropout_rate)(x)
            outputs = layers.Dense(num_classes, activation = "softmax")(model.output)
            final_model = tensorflow.keras.Model(inputs, outputs, name = "Inception_ResNet_V2")

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
            final_model.save(output_path + "/GENERIC_multiclassifier.h5")

        return "Finished training Generic"

    else:

        inputs = layers.Input(shape = (image_size, image_size, 3))

        x = tensorflow.keras.models.Sequential()(inputs)

        model = InceptionResNetV2(include_top = True, weights = 'imagenet', input_shape = (image_size,image_size,3), input_tensor = x)#(normalized)
        model.trainable = False
        x = layers.GlobalMaxPooling2D()(model.output)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(num_classes, activation = "softmax")(model.output)
        final_model = tensorflow.keras.Model(inputs, outputs, name = "Inception_ResNet_V2")

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