import argparse
import json
import os
from glob import glob
from pathlib import Path

import memryx as mx
import numpy as np
import onnx
import onnxruntime as ort
import torch
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import LOGGER, TQDM

weights_dir = os.getcwd() / Path("weights")


class MxaDetectionValidator(DetectionValidator):
    """
    The Validator must be a child of BaseValidator which is the parent
    of DetectionValidator. The BaseValidator defines the __call__
    method which we need to override.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Set required attributes
        self.stride = 32
        self.training = False

        model_name = Path(self.args.model).stem
        LOGGER.info(f"\033[32mRunning {model_name} inference on MXA\033[0m")

        # Ensure your paths/naming scheme matches
        self.mxa = mx.SyncAccl(weights_dir / f"{model_name}.dfp")
        self.ort = ort.InferenceSession(weights_dir / f"{model_name}_post.onnx")

    def __call__(self, model):
        model.eval()

        # Create COCO dataloader
        self.data = check_det_dataset(self.args.data)
        self.dataloader = self.get_dataloader(
            self.data.get(self.args.split), self.args.batch
        )

        # Validation Loop
        self.init_metrics((model))
        self.jdict = []
        progress_bar = TQDM(
            self.dataloader, desc=self.get_desc(), total=len(self.dataloader)
        )
   
        for batch in progress_bar:
            batch = self.preprocess(batch)
            preds = self.mxa_detect(batch["img"])
            preds = self.postprocess(preds)
            self.update_metrics(preds, batch)

        # Compute and print stats
        stats = self.get_stats()
        self.finalize_metrics()
        self.print_results()

        # Save predictions and evaluate on pycocotools
        with open(str(self.save_dir / "predictions.json"), "w") as f:
            LOGGER.info(f"Saving {f.name}...")
            json.dump(self.jdict, f)
        stats = self.eval_json(stats)

        return stats

    def mxa_detect(self, batch):
        """
        Detection using MXA accelerator.

        Args:
            batch (torch.Tensor): Input batch of images. (8, 3, 640, 640)

        Returns:
            preds (list): List of length 2.
                preds[0] (torch.Tensor): Predictions. (8, 84, 8400)
                preds[1] (None): Unused fmaps
        Notes:
            Fj in (64, 80) and Fi in (80, 40, 20)
        """
        # Pass images through accelerator
        batch = batch.detach().cpu().numpy()  # (8, 3, 640, 640)
        batch = [img[np.newaxis, ...] for img in batch] # (8, 1, 3, 640, 640)
        accl_out = self.mxa.run(batch)  # (8, 6, Fi, Fi, Fj)

        # Process accl out for onnxruntime
        if isinstance(accl_out[0], np.ndarray):
            accl_out = [accl_out]

        onnx_inps = []  
        batch_size, num_inputs = len(accl_out), len(accl_out[0])  

        for inp_idx in range(num_inputs):
            inp = [
                accl_out[batch_idx][inp_idx].squeeze(axis=0) for batch_idx in range(batch_size)
            ]  
            onnx_inps.append(inp)

        onnx_inp_names = [inp.name for inp in self.ort.get_inputs()]
        input_feed = {k: v for k, v in zip(onnx_inp_names, onnx_inps)}

        # Pass fmaps through onnxruntime
        onnx_out = self.ort.run(None, input_feed)
        out = torch.from_numpy(onnx_out[0])  # (8, 84, 8400)

        preds = [out, None]
        return preds


def dfp_exists(model):
    """Checks that the DFP and post-processing ONNX model exists"""
    model_name = Path(model.ckpt_path).stem
    dfp = (weights_dir / f"{model_name}.dfp").exists()
    post_onnx = (weights_dir / f"{model_name}_post.onnx").exists()
    return dfp and post_onnx


def compile_model(model):
    """Exports model to ONNX and compiles it to DFP."""
    model_name = Path(model.ckpt_path).stem
    # Export to onnx
    model.export(format="onnx", simplify=True, batch=8)
    onnx_model = onnx.load(weights_dir / f"{model_name}.onnx")
    # Compile and save the DFP file
    nc = mx.NeuralCompiler(
        models=onnx_model,
        autocrop=True,
        no_sim_dfp=True,
        dfp_fname=weights_dir / f"{model_name}.dfp",
        verbose=1,
    )
    nc.run()
    # Rename the exported ONNX files
    os.rename(
        weights_dir / "main_graph_crop.onnx",
        weights_dir / f"{model_name}_crop.onnx",
    )
    os.rename(
        weights_dir / "main_graph_post.onnx",
        weights_dir / f"{model_name}_post.onnx",
    )
    # Print file paths
    LOGGER.info(f'Files saved: {glob(f"{weights_dir.stem}/{model_name}*")}')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run YOLOv8 validation with MXA")
    parser.add_argument(
        "-s",
        "--size",
        type=str,
        choices=["n", "s", "m"],
        default="m",
        help='Specify the size of the YOLOv8 model (n, s, m). Default is "m".',
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        choices=["mxa", "cpu"],
        default="mxa",
        help='Specify the device to run the validation on (mxa, cpu). Default is "mxa".',
    )

    args = parser.parse_args()
    size = args.size
    device = args.device
    model_path = weights_dir / f"yolov8{size}.pt"

    # Downloads model if not already available
    model = YOLO(model_path)

    # Uses CUDA if available
    if device == "cpu":
        if torch.cuda.is_available():
            LOGGER.info("Found available GPU, using CUDA instead.")
        model.val()

    # Exports and compiles the model if necessary
    if device == "mxa":
        if not dfp_exists(model):
            compile_model(model)
        model.val(validator=MxaDetectionValidator, batch=8, rect=False)
