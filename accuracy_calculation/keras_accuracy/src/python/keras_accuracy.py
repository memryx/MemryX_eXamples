# Import all the modules necessary
import argparse
import numpy as np 
import keras
import time
from keras.applications import *
from memryx import NeuralCompiler
import cv2
import glob
import os
import numpy as np
import tensorflow as tf
from memryx import AsyncAccl
import tf_keras
from pathlib import Path
import subprocess
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


################################################################################
# ImageNet Dataset
################################################################################
def prepare_imagenet_dataset(imagenet_path, count):
    """
    Creates imagenet_path directory if not present and runs a bash script to
    download and extract imagenet validation dataset if not present.

    keras_accuracy
    ├── src
        └──  python
             └── get_imagenet_valdata.sh
             └── keras_accuracy.py
    └── assets
        └── ImageNet2012_valdata
              └── images ← downloads here
              └── ground_truth.txt

    """

    if not os.path.isdir(imagenet_path):
        os.mkdir(imagenet_path, mode=0o777)
    jpeg_count = len([filename for filename in os.listdir(imagenet_path)
                         if os.path.splitext(filename)[1] in [".JPEG"]])
    if jpeg_count < count:
        # use bash script to download and extract imagnet dataset
        exit_status  = subprocess.call("./get_imagenet_valdata.sh", shell=True)
        if exit_status  != 0:
            print(f"ImagNet dataset not configured! Try to run bash script get_imagenet_valdata.sh")
            print("Exiting...")
            sys.exit(1)


################################################################################

# Create the parser
parser = argparse.ArgumentParser(description = "Keras Applications accuracy check")

# Add arguments
parser.add_argument('--model_name', 
                    type = str, 
                    help = "Model name( https://keras.io/api/applications/ )")


parser.add_argument('--num_images', 
                    type = int, 
                    default = 50000,
                    help = "Specify the number of images you wish to run inference on.")


parser.add_argument('--backend', 
                    type = str, 
                    default = 'mxa',
                    help = "If you provide 'mxa' as the argument, it will run inference on the MXA. To instead run on the CPU, provide 'cpu'.")


parser.add_argument('--dfp', 
                    help = "Provide path to model dfp. In case you already have a dfp for a model, you may provide that dfp and skip the compilation step.")


# Parse the arguments
args = parser.parse_args()

user_provided_model_name = args.model_name
user_provided_model_dfp = args.dfp
num_images = int(args.num_images)
backend = args.backend


# Create the data paths for the Imagenet dataset
assets_dir         =  os.path.join(Path.cwd().parent.parent, 'assets')
imagenet_path      =  os.path.join(assets_dir, os.path.join("ImageNet2012_valdata", "images"))
ground_truth_path  =  os.path.join(assets_dir, os.path.join("ImageNet2012_valdata", "ground_truth.txt"))    

prepare_imagenet_dataset(imagenet_path, num_images)


# Reference : https://github.com/jake-memryx/KerasClassifiers/blob/main/compile.py
application_library = {
    'densenet': ['DenseNet121', 'DenseNet169', 'DenseNet201'], 
    'efficientnet': ['EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2'], 
    'efficientnet_v2': ['EfficientNetV2B0', 'EfficientNetV2B1', 'EfficientNetV2B2', 'EfficientNetV2B3'], 
    'inception_v3': ['InceptionV3'], 
    'mobilenet': ['MobileNet'], 
    'mobilenet_v2': ['MobileNetV2'], 
    'regnet': ['RegNetX002', 'RegNetX004', 'RegNetX006', 'RegNetX008', 'RegNetX016', 'RegNetY002', 'RegNetY004', 'RegNetY006', 'RegNetY008', 'RegNetY016'], 
    'resnet': ['ResNet50'], 
    'resnet_v2': ['ResNet50V2'], 
    'xception': ['Xception'], 
}


class ImageBatchIterator:
    """
    The ImageBatchIterator class contains functions that reads images in batchs from the Imagenet dataset and processes them before sending it to the
    model for inference. The functions are especially useful while running inference on the CPU.
    """

    def __init__(self, image_paths, batch_size, image_height, image_width, module):

        self.image_paths = image_paths
        self.batch_size = batch_size
        self.index = 0
        self.total_images = len(image_paths)
        self.image_height = image_height
        self.image_width = image_width
        self.module = module

    def __iter__(self):
        return self

    def __next__(self):
      
        if self.index >= self.total_images:
            raise StopIteration

        # Get the current batch of image paths
        batch_paths = self.image_paths[self.index : self.index + self.batch_size]
        batch_images = []

        # Load and preprocess each image in the batch
        for img_path in batch_paths:
            
            image_string = tf.io.read_file(img_path)
            image = tf.image.decode_jpeg(image_string, channels=3)

            # Resize and Center Crop (https://github.com/keras-team/keras/issues/15822#issuecomment-1027178496)
            size = self.image_height

            h, w = tf.shape(image)[0], tf.shape(image)[1]
            ratio = (tf.cast(size, tf.float32) / tf.cast(tf.minimum(h, w), tf.float32))
            h = tf.cast(tf.round(tf.cast(h, tf.float32) * ratio), tf.int32)
            w = tf.cast(tf.round(tf.cast(w, tf.float32) * ratio), tf.int32)
            image = tf.image.resize(image, [h, w])
            
            top, left = (h - size) // 2, (w - size) // 2
            image = tf.image.crop_to_bounding_box(image, top, left, size, size)
        
            # Additional preprocessing based on the model provided
            image = self.module.preprocess_input(image)
            image = np.expand_dims(np.array(image), axis=0)
            batch_images.append(image)

        # Stack images into a batch array
        batch_images = np.vstack(batch_images)
        self.index += self.batch_size

        # Preprocess the batch
        return batch_images


### Create helper functions

def load_images_and_labels():
    
    with open(ground_truth_path, 'r') as f:
        ground_truth = f.read().split('\n')[:-1]

    image_paths = glob.glob(imagenet_path+'/*.JPEG')
    image_paths.sort()
    
    return image_paths, ground_truth


def get_expected_model_input_shape(model_name, module_name):

    if module_name == 'regnet' or module_name == 'efficientnet':

        # Get the input shape expected of the model 
        model_class = getattr(tf_keras.applications, model_name)
        model = model_class()

    else:

        # Get the input shape expected of the model 
        model_class = getattr(tf.keras.applications, model_name)
        model = model_class()

    return model.input_shape[1:]


def get_keras_module_name(model_name):

    # Get the keras preprocessing module name for the model
    keras_preprocessing_module_name = None

    for k,v in application_library.items():

        if model_name in v:
            keras_preprocessing_module_name = k
            break

    if keras_preprocessing_module_name is None:
        raise ValueError('Unknown model. Please refer to https://keras.io/api/applications/ for the list of models.')

    return keras_preprocessing_module_name


def get_keras_module_and_model(module_name, model_name):

    # Note: Regnetx models are present in the older version of keras ie Keras 2
    if module_name == 'regnet' or module_name == 'efficientnet':
        module = getattr(tf_keras.applications, module_name)
        model = getattr(module, model_name)(weights = 'imagenet')   

    else:
        module = getattr(tf.keras.applications, module_name)
        model = getattr(module, model_name)(weights = 'imagenet')

    return module, model
  

def compile_model(model):

    nc = NeuralCompiler(num_chips = 4, models = model, verbose = 1)
    dfp = nc.run()

    return dfp


def get_accuracy(predictions, ground_truth):
    
    top1, top5, total = 0, 0, len(predictions)
    
    for i,pred in enumerate(predictions):
        gt = ground_truth[i]

        classes = [guess[0] for guess in pred]
        if gt in classes:
            top5 += 1
        if gt == classes[0]:
            top1 += 1

    print("Top 1: ({}/{})  {:.2f} % ".format(top1, total, top1/total*100))
    print("Top 5: ({}/{})  {:.2f} % ".format(top5, total, top5/total*100))


def run_inference_cpu(image_paths, ground_truth, new_height, new_width, module, model):

    batch_size = 128  # Set batch size

    # Create the iterator
    image_iterator = ImageBatchIterator(image_paths, batch_size, new_height, new_width, module) 

    # Collect predictions
    cpu_outputs = []

    start = time.time()

    # Iterate through the iterator using a for loop
    for batch in image_iterator:
        
        batch_preds = model.predict(batch)
        cpu_outputs.extend(batch_preds)

    cpu_inference_time = time.time() - start
    cpu_outputs = np.stack([np.squeeze(arr) for arr in cpu_outputs])
    cpu_predictions = module.decode_predictions(cpu_outputs, top=5)

    get_accuracy(cpu_predictions, ground_truth)
    print("CPU Inference time: {:.1f} msec".format(cpu_inference_time*1000))


def run_inference_mxa(image_paths, ground_truth, user_provided_model_name, module, module_name, user_provided_model_dfp, model):

    def process_output(*outputs):
        mxa_outputs.append(np.squeeze(outputs[0], 0))

    def preprocess_images():

        expected_input_shape = get_expected_model_input_shape(user_provided_model_name, module_name)
        new_height, new_width, _ = expected_input_shape

        images = []

        for img_path in image_paths:

            image_string = tf.io.read_file(img_path)
            image = tf.image.decode_jpeg(image_string, channels=3)

            # Resize and Center Crop (https://github.com/keras-team/keras/issues/15822#issuecomment-1027178496)
            size = new_height

            h, w = tf.shape(image)[0], tf.shape(image)[1]
            ratio = (tf.cast(size, tf.float32) / tf.cast(tf.minimum(h, w), tf.float32))
            h = tf.cast(tf.round(tf.cast(h, tf.float32) * ratio), tf.int32)
            w = tf.cast(tf.round(tf.cast(w, tf.float32) * ratio), tf.int32)
            image = tf.image.resize(image, [h, w])
            
            top, left = (h - size) // 2, (w - size) // 2
            image = tf.image.crop_to_bounding_box(image, top, left, size, size)

            # Additional preprocessing based on the model provided
            image = tf.expand_dims(image, 0)
            image = module.preprocess_input(image)
    
            yield np.array(image)
    

    # Check if the user provided a DFP or else compile the model and generate the DFP
    if user_provided_model_dfp:
        dfp = user_provided_model_dfp

    else:
        # Compile model and generate DFP
        dfp = compile_model(model)

    # Test on MXA
    mxa_outputs = []

    accl = AsyncAccl(dfp = dfp)
    start = time.time()
    accl.connect_input(preprocess_images)
    accl.connect_output(process_output)
    accl.wait()

    # Postprocess the outputs
    mxa_outputs = np.stack([arr for arr in mxa_outputs])
    mxa_inference_time = time.time() - start
    mxa_predictions = module.decode_predictions(mxa_outputs, top=5)

    # Display results
    get_accuracy(mxa_predictions, ground_truth)
    print("MXA Inference time: {:.1f} msec".format(mxa_inference_time*1000))


def main():
        
    # Load images and labels
    image_paths, ground_truth = load_images_and_labels()
    image_paths = image_paths[:num_images]
    
    # Get the keras module name and the model 
    module_name = get_keras_module_name(user_provided_model_name)
    module, model = get_keras_module_and_model(module_name, user_provided_model_name)

    # Preprocess data
    expected_input_shape = get_expected_model_input_shape(user_provided_model_name, module_name)
    new_height, new_width, _ = expected_input_shape

    ## Run on cpu
    if backend == 'cpu':
        run_inference_cpu(image_paths, ground_truth, new_height, new_width, module, model)
      
    elif backend == 'mxa':
        run_inference_mxa(image_paths, ground_truth, user_provided_model_name, module, module_name, user_provided_model_dfp, model)
       
    

if __name__=="__main__":
    main()
