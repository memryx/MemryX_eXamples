# classification_resnet50
# ├── README.md
# ├── src
#     └──  python
#          └── get_imagenet_valdata.sh
#          └── preprocess.property
#          └── run_validation.py
# └── assets
#     └── ImageNet2012_valdata
#           └── images
#           └── ground_truth.txt
# └── models
#     └──resnet50_v1.pb
#     └──resnet50_v1.dfp

import os, sys, subprocess, tarfile, glob, argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

import tensorflow as tf
import google.protobuf
from preprocess import pre_process_vgg
from tensorflow.keras.applications.resnet50 import decode_predictions
from memryx import NeuralCompiler, AsyncAccl


################################################################################
# ImageNet Dataset
################################################################################
def prepare_imagenet_dataset(imagenet_path, count):
    """
    Creates imagenet_path directory if not present and runs a bash script to
    download and extract imagenet validation dataset if not present.

    classification_resnet50
    ├── src
        └──  python
             └── get_imagenet_valdata.sh
             └── preprocess.property
             └── run_validation.py
    └── assets
        └── ImageNet2012_valdata
              └── images ← downloads here
              └── ground_truth.txt
    └── models
        └──resnet50_v1.pb
        └──resnet50_v1.dfp

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

class ResNet50_Classification:
    def __init__(self, model_path, dataset_path, count):
        # model exists
        if not os.path.isfile(model_path):
            model_url = 'https://zenodo.org/record/2535873/files/resnet50_v1.pb'
            result = subprocess.run(["wget", model_url], capture_output=True, text=True)

        self.model_path   = model_path
        self.input_shape  = (224,224,3)
        self.dataset_path = dataset_path
        self.count        = count
        self.mxa_outputs  = []
        with open(os.path.join(os.path.dirname(self.dataset_path), 'ground_truth.txt')) as f:
            self.ground_truth = f.read().split('\n')
        self.image_paths = glob.glob(self.dataset_path +'/*.JPEG')[:self.count]

    def val(self, device, batch=10):
        if device == "cpu":
            predictions, groundtruth = self.cpu_run(batch)
        if device == "mxa":
            dfp_name = os.path.basename(self.model_path).split(".")[0]
            dfp_path = os.path.join(os.path.dirname(self.model_path), dfp_name + ".dfp")
            if not os.path.isfile(dfp_path):
                # if dfp does not exist, compile and generated .dfp file
                self.compile_model(dfp_name)
            predictions, groundtruth = self.mxa_run(dfp_path, batch)

        self.get_accuracy(predictions, groundtruth)

    def compile_model(self, dfp_name):
        """
        Compiles the model using the NeuralCompiler API and generates the dfp file
        """
        nc = NeuralCompiler(num_chips=4,
                            models=self.model_path,
                            verbose=1,
                            dfp_fname = os.path.join(os.path.dirname(self.model_path), dfp_name),
                            autocrop=True)
        nc.run()

    def load_tfmodel(self, model_path):
        """
        Helper function to load tf model from model path
        """
        with tf.io.gfile.GFile(model_path, "rb") as f:
            file_data = f.read()
        # for frozen inference graph protobuf
        frozen_model = tf.compat.v1.GraphDef()
        frozen_model.ParseFromString(file_data)
        return frozen_model

    def load_images_and_labels(self, start, end):
        """
        Loads images and their respective labels
        """
        images = []
        labels = []
        for fname in self.image_paths[start:end]:
            img = cv2.imread(fname)
            preprocessed_img = pre_process_vgg(img, self.input_shape)
            preprocessed_img = np.expand_dims(preprocessed_img, 0)
            images.append(preprocessed_img)
            # get ground_truth labels!
            # ILSVRC2012_val_xxxxxxxx -> int(xxxxxxxx) - 1 ; as numbering starts from 1
            label = self.ground_truth[int(os.path.split(fname)[1].split('_')[2].split('.')[0]) - 1]
            labels.append(label)
        return images, labels

    def postprocess(self, outputs):
        outputs = outputs[:,1:]
        preds = decode_predictions(outputs, top=5)
        return preds

    def cpu_run(self, batch=10):
        """
        Generates classification predictions by passing loaded images through loaded TF model
        Runs on CPU.
        """
        # load model
        model      = self.load_tfmodel(self.model_path)

        # Prepare tf session
        input_names = ['input_tensor']
        output_names = ["softmax_tensor", "ArgMax"]
        graph = tf.compat.v1.Graph()
        node_names = [node.name for node in model.node]
        with graph.as_default():
            tf.compat.v1.import_graph_def(model, return_elements=node_names, name="")
            sess = tf.compat.v1.Session()
        fetches = [out_name + ':0' for out_name in output_names]

        # prepare input_images
        labels_lst = []
        predictions = []
        for b in tqdm(range(0, self.count, batch)):
            start = b
            end   = b + batch if (b + batch) <= self.count else b + self.count
            images, labels = self.load_images_and_labels(start=start, end=end)
            images = np.array(images)
            images = np.squeeze(images, 1)
            labels_lst.extend(labels)
            feed_dict = {'input_tensor:0': images}
            # session run!
            outs = sess.run(feed_dict=feed_dict, fetches=fetches)
            # postprocess!
            predictions.extend(self.postprocess(outs[0]))
            start = b

        return predictions, labels_lst

    def get_accuracy(self, predictions, ground_truth):
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


    def mxa_run(self, dfp_path, batch=10):
        """
        Generates classification predictions by passing loaded images through the MXA accelerator,
        using the pre-compiled dfp. Uses Piplined AsyncAccl API.
        """

        images, labels = self.load_images_and_labels(start=0, end=self.count)
        img_iter = iter(images)

        mxa_outputs = []
        def get_frame():
            return next(img_iter, None)

        def process_output(*outputs):
           mxa_outputs.extend(outputs[0])

        accl = AsyncAccl(dfp_path)
        labels_lst = []
        for b in tqdm(range(0, self.count, batch)):
            start = b
            end   = b + batch if (b + batch) <= self.count else b + self.count
            images, labels = self.load_images_and_labels(start=start, end=end)
            labels_lst.extend(labels)
            img_iter = iter(images)
            # MXA run
            accl.connect_input(get_frame)
            accl.connect_output(process_output)
            accl.wait()
            start = b

        mxa_outputs = np.stack([np.squeeze(arr) for arr in mxa_outputs])
        mxa_predictions = self.postprocess(mxa_outputs)

        return mxa_predictions, labels_lst


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run ResNet50 validation with MXA")
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        choices=["mxa", "cpu"],
        default="mxa",
        help='Specify the device to run the validation on (mxa, cpu). Default is "mxa".',
    )

    parser.add_argument(
        "-c",
        "--count",
        type=int,
        default=50000,
        help='Number of ImageNet validation images to run classification accuracy on - default=50000.',
    )

    args = parser.parse_args()
    device = args.device
    count  = args.count

    assets_dir     =  os.path.join(Path.cwd().parent.parent, 'assets')
    models_dir     =  os.path.join(Path.cwd().parent.parent, 'models')
    model_path     =  os.path.join(models_dir, "resnet50_v1.pb")
    imagenet_path =   os.path.join(assets_dir, os.path.join("ImageNet2012_valdata", "images"))
    prepare_imagenet_dataset(imagenet_path, count)

    model = ResNet50_Classification(model_path, imagenet_path, count)
    if device == "cpu":
        print("Using CPU!")
        model.val(device, batch=10)

    if device == "mxa":
        print("Using MXA!")
        model.val(device, batch=1000)
