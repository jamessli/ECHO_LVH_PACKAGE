from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
#Tensorflow
import tensorflow
from tensorflow.keras import regularizers
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

def train_views(training_path, validation_path, output_path, devices):

    mirrored_strategy = tensorflow.distribute.MirroredStrategy(devices)

    training_path_views = training_path
    validation_path_views = validation_path
    model_path = output_path

    batch_size = 32 #4Gpus, 64 batches each?
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


    training_data_generator = ImageDataGenerator(rescale = 1.0/255.0, rotation_range = 45, width_shift_range = .5, height_shift_range = .15, horizontal_flip = False, fill_mode = 'nearest') #Rescale images to 0 to .255
    validation_data_generator = ImageDataGenerator(rescale = 1.0/255.0)

    with mirrored_strategy.scope():

        AP2_model = tensorflow.keras.models.load_model(str(model_path) + "/generic_classifier.h5")
        AP3_model = tensorflow.keras.models.load_model(str(model_path) + "/generic_classifier.h5")
        AP4_model = tensorflow.keras.models.load_model(str(model_path) + "/generic_classifier.h5")
        PSAX_V_model = tensorflow.keras.models.load_model(str(model_path) + "/generic_classifier.h5")
        PSAX_M_model = tensorflow.keras.models.load_model(str(model_path) + "/generic_classifier.h5")
        PLAX_model = tensorflow.keras.models.load_model(str(model_path) + "/generic_classifier.h5")

    model_list = [AP2_model, AP3_model, AP4_model, PSAX_V_model, PSAX_M_model, PLAX_model]

    for view,model in zip(['AP2', 'AP3', 'AP4', 'PSAX_V', 'PSAX_M', 'PLAX'], model_list):

        print("Beginning training for view: ", view)

        training_data_gen = training_data_generator.flow_from_directory(str(training_path_views + '/{0}/'.format(view)),  shuffle = True, target_size = (image_size,image_size), batch_size = batch_size, color_mode = 'rgb', class_mode = "categorical")
        train_len = len(training_data_gen.classes)
        print(view, " training data processed slices: ", training_data_gen.class_indices)

        validation_data_gen = validation_data_generator.flow_from_directory(str(validation_path_views + '/{0}/'.format(view)),  shuffle = True, target_size = (image_size,image_size), batch_size = batch_size, color_mode = 'rgb', class_mode = "categorical")
        val_len = len(validation_data_gen.classes)
        print(view, "validation data processed slices: ", validation_data_gen.class_indices)

        make_trainable = round(len(model.layers)*learnable_layers)

        for layer in model.layers[-make_trainable:]:

            layer.trainable = True

            if isinstance(layer, layers.Conv2D) or isinstance(layer, layers.Dense):

                layer.add_loss(lambda: regularizers.l2(weight_decay)(layer.kernel))
            
            if hasattr(layer, 'bias_regularizer') and layer.use_bias:

                layer.add_loss(lambda: regularizers.l2(weight_decay)(layer.bias))
                
        reduce_lr  = ReduceLROnPlateau(monitor = 'val_loss', mode = 'min', patience = 10, factor = learning_rate_decay, min_lr = .000000001, verbose = 2)

        model.compile(optimizer = tensorflow.keras.optimizers.Adam(learning_rate = learning_rate), loss = "categorical_crossentropy", metrics = ["accuracy"])

        history = model.fit(training_data_gen, epochs = epochs, steps_per_epoch = train_len//batch_size, validation_data=validation_data_gen, validation_steps = val_len//batch_size, verbose = 1, callbacks = [reduce_lr], use_multiprocessing = True, workers = 64)
        model.save(model_path + "/{0}_classifier.h5".format(view))

        print(view, " model training accuracy: ", history.history["accuracy"], " | validation accuracy: ", history.history["val_accuracy"])

    return "Finished training view-based classifiers"