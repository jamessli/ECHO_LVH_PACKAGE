#Tensorflow
import tensorflow
from tensorflow.keras import regularizers
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionResNetV2

def train_generic(studies_path, output_path, devices):

    mirrored_strategy = tensorflow.distribute.MirroredStrategy(devices)

    #mirrored_strategy = tensorflow.distribute.MirroredStrategy(['/gpu:2', '/gpu:3'])

    training_data_path = "{0}/{1}".format(studies_path, "images_training")
    validation_data_path = "{0}/{1}".format(studies_path, "images_validation")
    
    #TUNE MODEL WITH THIS
    dropout_rate = .4
    batch_size = 32 #4Gpus, 64 batches each?
    num_classes = 3
    epochs = 50
    image_size = 299
    weight_decay = .3

    training_data_generator = ImageDataGenerator(rescale = 1.0/255.0, rotation_range = 60, width_shift_range = .35, height_shift_range = .15, horizontal_flip = False, fill_mode = 'nearest') #Rescale images to 0 to .255
    validation_data_generator = ImageDataGenerator(rescale = 1.0/255.0)
    training_data_gen = training_data_generator.flow_from_directory(str(training_data_path),  shuffle = True, target_size = (image_size,image_size), batch_size = batch_size, color_mode = 'rgb', class_mode = "categorical")

    train_len = len(training_data_gen.classes)

    print("Training data: ", training_data_gen.class_indices)

    validation_data_gen = validation_data_generator.flow_from_directory(str(validation_data_path),  shuffle = True, target_size = (image_size,image_size), batch_size = batch_size, color_mode = 'rgb', class_mode = "categorical")

    val_len = len(validation_data_gen.classes)

    print("Validation data: ", validation_data_gen.class_indices)

    with mirrored_strategy.scope():

        inputs = layers.Input(shape = (image_size, image_size, 3))

        x = tensorflow.keras.models.Sequential()(inputs)

        model = InceptionResNetV2(include_top = True, weights = 'imagenet', input_shape = (image_size,image_size,3), input_tensor = x)#(normalized)
        model.trainable = False
        x = layers.GlobalMaxPooling2D()(model.output)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(num_classes, activation = "softmax")(model.output)
        final_model = tensorflow.keras.Model(inputs, outputs, name = "Inception_ResNet_V2")

        make_trainable = len(final_model.layers)//2

        for layer in final_model.layers[-make_trainable:]:

            layer.trainable = True

            if isinstance(layer, layers.Conv2D) or isinstance(layer, layers.Dense):

                layer.add_loss(lambda: regularizers.l2(weight_decay)(layer.kernel))
            
            if hasattr(layer, 'bias_regularizer') and layer.use_bias:

                layer.add_loss(lambda: regularizers.l2(weight_decay)(layer.bias))

        reduce_lr  = ReduceLROnPlateau(monitor = 'val_loss', mode = 'min', patience = 10, factor = .5, min_lr = .000015625, verbose = 2)

        final_model.compile(optimizer = tensorflow.keras.optimizers.Adam(learning_rate = .0005), loss = "categorical_crossentropy", metrics = ["accuracy"])

        checkpoint_filepath = '../new_model_02/auto_saves_2/'
        model_checkpoint_callback = tensorflow.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=False,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)

        final_model.fit(training_data_gen, epochs = epochs, steps_per_epoch = train_len//batch_size, validation_data=validation_data_gen, validation_steps = val_len//batch_size, verbose = 1, callbacks = [reduce_lr], use_multiprocessing = True, workers = 64)
        final_model.save(output_path + "/GENERIC_multiclassifier.h5")

    return "Finished training Generic"