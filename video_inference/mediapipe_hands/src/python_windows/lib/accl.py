"""
accl.py
MemryX Runtime API for Python


Copyright (c) 2024 MemryX Inc.
MIT License

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""


import queue
import threading
import multiprocessing
import numpy as np
import sys, os
import tempfile
import struct
import time
from pathlib import Path
import weakref
import numbers
import unittest
from collections import defaultdict, deque
import functools
from abc import ABC
import inspect
import logging
import traceback
import io
import math

from lib.dfp import Dfp
from lib import mxa


logger = logging.getLogger(__name__)
handler = logging.NullHandler()
formatter = logging.Formatter(fmt='[%(asctime)s] [%(threadName)s] [%(levelname)s] %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)
logger.addHandler(handler)

def advance_iter(iterator):
    try:
        res = next(iterator)
    except StopIteration:
        res = None
    return res

def validate_input_callback_result(input_frames):
    if input_frames is None:
        return
    if not isinstance(input_frames, (list, tuple)):
        frame_seq = [input_frames]
    else:
        frame_seq = input_frames
    for x in frame_seq:
        if not isinstance(x, np.ndarray):
            raise TypeError(f"input_callback must return a `np.ndarray` or sequence of `np.ndarray`, got: {type(x)}")

class _PassThroughModel:
    def __init__(self):
        pass

    def predict_for_mxa(self, inputs, format_output=False, output_order=None):
        """ return input as is, kwargs only to match interface """
        if isinstance(inputs, dict):
            return list(inputs.values())
        return inputs

class Accl(ABC):
    def __init__(self, dfp, group_id=0,  **kwargs):

        self._print_lock = multiprocessing.Lock()

        model_id = group_id

        if not isinstance(group_id, int) or group_id < 0:
            raise TypeError("Group ID must be an non-negative integer")

        self._dfp = self._parse_dfp(dfp)
        self._model_id = model_id
        self._group_id = group_id
        self._chip_gen = self._dfp.chip_gen # add check to compare DFP chip gen with H/W when possible
        if self._chip_gen == 'Cascade+':
            self._chip_gen = 3.1
        elif self._chip_gen == 'Cascade':
            self._chip_gen = 3
        else:
            self._chip_gen = 2
        if self._chip_gen not in [3, 3.1]:
            raise ValueError(f"Invalid chip generation: {self._chip_gen}, select from [3, 3.1]")

        self._finalizer = weakref.finalize(self, self._cleanup) # called at program exit to free resources

        self._configure(self._dfp, kwargs.get('inport_mapping', {}), kwargs.get('outport_mapping', {}))

    def _configure(self, dfp, inport_mapping={}, outport_mapping={}):

        #TODO Redesign to work with both cascade, cascade_plus
        for idx_pair, mapping in inport_mapping.items():
            mpu, port = idx_pair
            for k in ['model_index', 'layer_name']:
                self._dfp.input_ports[port][k] = mapping[k]
        for idx_pair, mapping in outport_mapping.items():
            mpu, port = idx_pair
            for k in ['model_index', 'layer_name']:
                self._dfp.output_ports[port][k] = mapping[k]

        self._model_idx_to_out_info = defaultdict(lambda : defaultdict(list))
        self._outlayer_to_port_idx = {}
        for i, info in self._dfp.output_ports.items():
            row,col,z,ch = info['shape']
            if 'model_index' in info:
                idx = info['model_index']
                self._model_idx_to_out_info[idx]['dtypes'].append(np.float32)
                if self._chip_gen > 3:
                    self._model_idx_to_out_info[idx]['shapes'].append([row, col, z, ch])
                else:
                    self._model_idx_to_out_info[idx]['shapes'].append([row, col, ch])
                self._model_idx_to_out_info[idx]['ports'].append(i)
                if 'layer_name' in info:
                    layer_name = info['layer_name']
                    self._model_idx_to_out_info[idx]['layers'].append(layer_name)
                    self._outlayer_to_port_idx[(idx, layer_name)] = i

        self._model_idx_to_in_info = defaultdict(lambda : defaultdict(list))
        self._inlayer_to_port_idx = {}
        for i, info in self._dfp.input_ports.items():
            row,col,z,ch = info['shape']
            if info['data_type'] in ['float'] or info['data_range_enabled']:
                dtype = np.float32
            else:
                dtype = np.uint8
            info['dtype'] = dtype
            if 'model_index' in info:
                idx = info['model_index']
                self._model_idx_to_in_info[idx]['dtypes'].append(dtype)
                if self._chip_gen > 3:
                    self._model_idx_to_in_info[idx]['shapes'].append([row, col, z, ch])
                else:
                    self._model_idx_to_in_info[idx]['shapes'].append([row, col, ch])
                self._model_idx_to_in_info[idx]['ports'].append(i)
                if 'layer_name' in info:
                    layer_name = info['layer_name']
                    self._model_idx_to_in_info[idx]['layers'].append(layer_name)
                    self._inlayer_to_port_idx[(idx, layer_name)] = i

        self._create_models()
        self._init(self._model_id, self._group_id, self._chip_gen)

    def _create_models(self):
        self._models = []
        for m in self._model_idx_to_in_info:
            self._models.append(AcclModel(self._model_id, self._model_idx_to_in_info[m], self._model_idx_to_out_info[m], self))

    def _printl(self, *objects, sep=' ', end='\n', file=sys.stdout, flush=False):
        with self._print_lock:
            print(*objects, sep=sep, end=end, file=file, flush=flush)

    def _parse_dfp(self, dfp):
        self._tmp_dfp_file = None
        if not isinstance(dfp, Dfp):
            self._dfp_path = str(dfp)
            dfp = Dfp(dfp)
        else:
            self._tmp_dfp_file = tempfile.NamedTemporaryFile(dir=os.getcwd(), suffix='.dfp')
            dfp.write(self._tmp_dfp_file.name)
            self._dfp_path = self._tmp_dfp_file.name

        ver = dfp.version
        if ver == 'legacy' or int(ver) < 5:
            raise RuntimeError(f"Unsupported DFP version: {ver}, please recompile to the latest version")
        return dfp

    def _init(self, model_id, group_id, chip_gen, model_idx=0, model_type=3):
        # clean up context from existing instance
        #mxa.close(model_id)
        #time.sleep(0.01)
        #mxa.unlock(group_id)

        if mxa.lock(group_id):
            raise RuntimeError(f'Error getting lock on the accelerator')
        if mxa.open(model_id, group_id, chip_gen):
            raise RuntimeError(f'Error starting the accelerator, ensure that the set `group_id` is connected')

        MIN_MPUS = 1
        MAX_MPUS = 32
        dfp_mpus = self._dfp.num_chips
        hw_mpus = mxa.chip_count(self._group_id)


        if hw_mpus > 0 and dfp_mpus != hw_mpus:
            if chip_gen == 3:
                raise RuntimeError(f"Cascade (EVB1/EVB2) is no longer supported  :-(")
            elif hw_mpus == 4 and dfp_mpus == 2:
                mxa.config_mpu_group(self._group_id, 1)

            elif hw_mpus == 4 and dfp_mpus == 4:
                mxa.config_mpu_group(self._group_id, 0)

            elif hw_mpus == 8 and dfp_mpus == 8:
                mxa.config_mpu_group(self._group_id, 5)
            else:
                raise RuntimeError(f"Input DFP was compiled for {dfp_mpus} chips, but the connected accelerator has {hw_mpus} chips")
        if mxa.download(model_id, self._dfp_path):
            raise RuntimeError(f'Error loading DFP, try resetting the board')
        if mxa.set_stream_enable(model_id, 0):
            raise RuntimeError(f'Error enabling input/output, try resetting the board')

        # temp file no longer needed
        if self._tmp_dfp_file:
            self._tmp_dfp_file.close()

    def _load_dfp(self, model_id, group_id, dfp_path, model_idx=0, model_type=3):
        # TODO - Allow loading new DFP and reconfigure accordingly
        err = mxa.lock(group_id)
        if err:
            self._printl(f'Error getting lock on mxa: {err}')
        err = mxa.open(model_id, group_id, self._chip_gen)
        if err:
            self._printl(f'Error opening mxa: {err}')
        err = mxa.download(model_id, dfp_path)
        if err:
            self._printl(f'Error loading dfp: {err}')
        err = mxa.set_stream_enable(model_id, 0)
        if err:
            self._printl(f'Error enabling stream: {err}')
        mxa.close(model_id)
        mxa.unlock(group_id)

    @property
    def models(self):
        """
        Returns a list of model objects that provide input/output API
        """
        return self._models


    @property
    def input_port_ids(self):
        """
        Returns a list of ids of all active input ports in the accelerator
        """
        all_ports = []
        for model in self._models:
            all_ports.extend(model.input.port_ids)
        return all_ports

    @property
    def output_port_ids(self):
        """
        Returns a list of ids of all active output ports in the accelerator
        """
        all_ports = []
        for model in self._models:
            all_ports.extend(model.output.port_ids)
        return all_ports

    @property
    def mpu_count(self):
        """
        Returns the number of MPUs in the accelerator.
        """
        return self._dfp.num_chips

    @property
    def model_id(self):
        """
        Returns the model id defined in the driver
        """
        return self._model_id

    @property
    def group_id(self):
        """
        Returns the group id defined in the driver
        """
        return self._group_id

    @property
    def chip_gen(self):
        """
        Returns the architecture generation of the accelerator
        """
        return self._chip_gen

    def outport_assignment(self, model_idx=0):
        """
        Returns a dictionary which maps output port ids to model output layer names for
        the model specified by `model_idx`

        Parameters
        ----------
        model_idx: int
            Index of the model whose output port assignment is returned
        """
        self._ensure_valid_model_idx(model_idx)
        return self._models[model_idx].output.port_assignment

    def inport_assignment(self, model_idx=0):
        """
        Returns a dictionary which maps input port ids to model input layer names for
        the model specified by `model_idx`

        Parameters
        ----------
        model_idx: int
            Index of the model whose input port assignment is returned
        """
        self._ensure_valid_model_idx(model_idx)
        return self._models[model_idx].input.port_assignment

    def shutdown(self):
        """
        Shutdown the accelerator to make it available for other processes to use.
        If this function is not called while the current program is running,
        other processes will not be able to access the same accelerator until
        the current program terminates.
        """
        time.sleep(0.05)
        mxa.close(self._model_id)
        time.sleep(0.01)
        mxa.unlock(self._group_id)
        if self._tmp_dfp_file:
            self._tmp_dfp_file.close()

    def _ensure_valid_model_idx(self, model_idx):
        if model_idx not in range(len(self._models)):
            raise IndexError(
                f"Valid model indices are in the range [0, {len(self._models) - 1}], "
                    f"but got: {model_idx}"
                    )

    def _ensure_nd_array(self, data, model_idx):
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Model {model_idx} expected np.ndarray, got: {type(data)}")

    def _ensure_seq_nd_array(self, data, model_idx):
        for item in data:
            if not isinstance(item, np.ndarray):
                raise TypeError(f"Model {model_idx} expected np.ndarray, got: {type(data)}")

    def _ensure_input_len(self, data, target_len, model_idx):
        if len(data) != target_len:
            raise RuntimeError(
                    f"Model {model_idx} has {target_len} inputs, "
                    f"but data is a sequence of length: {len(data)}"
                    )

    def _cleanup(self):
        mxa.close(self._model_id)
        time.sleep(0.01)
        mxa.unlock(self._group_id)
        if self._tmp_dfp_file:
            self._tmp_dfp_file.close()

class AcclModel(ABC):
    def __init__(self, model_id, input_info, output_info, accl):
        self._model_id = model_id
        self._accl = accl
        self.input = AcclModelInput(model_id, input_info, accl, self)
        self.output = AcclModelOutput(model_id, output_info, accl, self)

class AcclModelInput(ABC):
    def __init__(self, model_id, info, accl, model):
        self._model_id = model_id
        self._info = info
        self._accl = accl
        self._model = model
        self._pre_model = _PassThroughModel()

    @property
    def port_assignment(self):
        """
        Returns a dictionary which maps port ids to output layer names
        """
        return {self._info['ports'][i]: self._info['layers'][i] for i in range(len(self._info['ports']))}

    @property
    def port_count(self):
        """
        Returns the number of input ports used by the model
        """
        return len(self._info['ports'])

    @property
    def port_ids(self):
        """
        Returns a list of input port ids used by the model
        """
        return self._info['ports']

    @property
    def port_spec(self):
        """
        Returns a list of (shape, dtype) tuples for each input port used by the model
        """
        return [(shape, dtype) for shape, dtype in zip(self._info['shapes'], self._info['dtypes'])]

    def set_preprocessing(self, model):
        raise RuntimeError("set_preprocessing is not yet supported on Windows!")

    def _stream_ifmaps(self, input_frames):
        if not isinstance(input_frames, (list, tuple)):
            input_frames = [input_frames]
        model_idx = self._accl.models.index(self._model)
        if len(input_frames) != len(self._info['ports']):
            raise RuntimeError(f"Model {model_idx} expects {len(self._info['ports'])} inputs, but input function returned {len(input_frames)} frames")
        for f, p, shape, dtype in zip(input_frames, self._info['ports'], self._info['shapes'], self._info['dtypes']):
            if len(shape) == 4 and shape[-2] == 1 and len(list(f.shape)) == 3:
                shape = shape[:-2] + [shape[-1]]
            if list(f.shape) not in [list(shape), [1] + list(shape)]:
                raise RuntimeError(f"Model {model_idx} input port {p} expects data of shape {shape}, but got {f.shape}")
            if list(f.shape) == [1] + list(shape):
                f = np.squeeze(f, 0)
            if f.dtype != dtype:
                raise RuntimeError(f"Model {model_idx} input port {p} expects data of type {dtype}, but got {f.dtype}")
            # NOTE: explicitly copy the fmap because
            # we cannot trust the user will leave the reference to 'f'
            # alone while the accelerator works on it. Perhaps we can add
            # an 'immutable' flag in the future to avoid this copy.
            mxa.stream_ifmap(self._model_id, p, f.copy())

class AcclModelOutput(ABC):
    def __init__(self, model_id, info, accl, model):
        self._model_id = model_id
        self._info = info
        self._accl = accl
        self._model = model
        self._post_model = _PassThroughModel()

    @property
    def port_assignment(self):
        """
        Returns a dictionary which maps port ids to model output layer names
        """
        return {self._info['ports'][i]: self._info['layers'][i] for i in range(len(self._info['ports']))}

    @property
    def port_count(self):
        """
        Returns the number of output ports used by the model
        """
        return len(self._info['ports'])

    @property
    def port_ids(self):
        """
        Returns a list of output port ids used by the model
        """
        return self._info['ports']

    @property
    def port_spec(self):
        """
        Returns a list of (shape, dtype) tuples for each output port used by the model
        """
        shapes = []
        # TODO fix when DFP stores original model output shapes
        for shape in self._info['shapes']:
            if len(shape) == 4 and shape[-2] == 1:
                shapes.append(shape[:-2] + [shape[-1]])
            else:
                shapes.append(shape)

        return [(shape, dtype) for shape, dtype in zip(shapes, self._info['dtypes'])]

    def set_postprocessing(self, model):
        raise RuntimeError("set_postprocessing is not yet supported on Windows")

    def _stream_ofmaps(self, model_name, outputs, outputs_by_name, timeout=500):
        # returns True if streamed data
        outputs.clear()
        outputs_by_name.clear()
        for idx, p in enumerate(self._info['ports']):
            shape = self._info['shapes'][idx]
            dtype = self._info['dtypes'][idx]
            ofmap = np.zeros(shape).astype(dtype)
            err = mxa.stream_ofmap(self._model_id, p, ofmap, timeout)
            if err: # check specifically for timeout and not others
                logger.debug(f"Model {model_name} stream outputs timed out")
                return False
            #TODO store original model output shapes in the DFP and reshape to those here
            if len(ofmap.shape) == 4 and ofmap.shape[-2] == 1:
                ofmap = np.squeeze(ofmap, -2)
            outputs.append(ofmap)
            outputs_by_name[self._info['layers'][idx]] = ofmap
        logger.debug(f"Model {model_name} stream outputs successful")
        return True

class SyncAccl(Accl):
    """
    This class provides a synchronous API for the MemryX hardware accelerator, which performs input and output
    sequentially per model. The accelerator is abstracted as a collection of models. You can select
    the desired model specifying its index to the member function.

    Parameters
    ----------
    dfp: bytes or string
        Path to dfp or a dfp object (bytearray). The dfp is generated by the NeuralCompiler.
    group_id: int
        The index of the chip group to select.

    Examples
    --------

    .. code-block:: python

        import tensorflow as tf
        import numpy as np
        from memryx import NeuralCompiler, SyncAccl

        # Compile a MobileNet model for testing.
        # Typically, comilation need to be done one time only.
        model = tf.keras.applications.MobileNet()
        nc = NeuralCompiler(models=model)
        dfp = nc.run()

        # Prepare the input data
        img = np.random.rand(224,224,3).astype(np.float32)
        data = tf.keras.applications.mobilenet.preprocess_input(img)

        # Accelerate using the MemryX hardware
        accl = SyncAccl(dfp)
        outputs = accl.run(data) # Run sequential acceleration on the input data.

    .. warning::

        MemryX accelerator is a streaming processor that the user can supply with pipelined input data. Using the synchronous API to perform sequential execution of multiple input frames may result in a significant performance penalty. The user is advised to use the send/receive functions on separate threads or to use the asynchronous API interface.
    """
    # TODO: inport_mapping and outport_mapping will get removed once the DFP branch gets merged

    def __init__(self, dfp, group_id=0, **kwargs):
        super().__init__(dfp, group_id, **kwargs)

    def _create_models(self):
        self._models = []
        for m in self._model_idx_to_in_info:
            self._models.append(SyncAcclModel(self._model_id, self._model_idx_to_in_info[m], self._model_idx_to_out_info[m], self))

    def send(self, data, model_idx=0, timeout=None):
        """
        For the model specified by `model_idx,` this function transfers input data to the accelerator.
        It copies the data to the buffer(s) of the model's input port(s) and returns.
        If there is no space in the buffer(s), the call blocks for a period decided by `timeout.`

        Parameters
        ----------
        data: np.ndarray or sequence of np.ndarray
            Typically the pre-processed input data array(s) of the model
        model_idx: int
            Index of the model to which the data should be sent
        timeout: int
            The number of milliseconds to block if there is no space in the port buffer(s).
            If set to None (default), blocks until there is space, otherwise blocks for
            at most `timeout` milliseconds and raises an error if still there is no space.

        Raises
        ------
        TimeoutError:
            When no data is available at the port buffer(s) after blocking for
            `timeout` (> 0) milliseconds
        """
        m = model_idx
        self._ensure_valid_model_idx(m)
        if not isinstance(data, (list, tuple)):
            data = [data]
        self._ensure_seq_nd_array(data, m)
        inports = self._models[m].input.port_ids
        self._ensure_input_len(data, len(inports), m)
        self._models[m].input.send(data[0], 0, timeout=timeout)
        for i in range(1, len(data)):
            self._models[m].input.send(data[i], i)

    def receive(self, model_idx=0, timeout=None):
        """
        For the model specified by `model_idx,` this function collects the output data from the accelerator.
        It retrieves data from the selected model's output buffer(s).
        If data is unavailable at any of the model output ports,
        the function call gets blocked for the specified `timeout` milliseconds.

        Parameters
        ----------
        model_idx: int
            Index of the model from which the data should be read
        timeout: int
            The number of milliseconds to block if no data is avaialable at the port buffer(s).
            If set to None (default), blocks until data is available, otherwise blocks for
            at most `timeout` milliseconds and raises an error if data is still unavailable.

        Raises
        ------
        TimeoutError:
            When no data is available at the port buffer(s) after blocking for
            `timeout` (> 0) milliseconds

        Example
        -------

        .. code-block:: python

            import tensorflow as tf
            import numpy as np
            from memryx import NeuralCompiler, SyncAccl

            def generate_frame():
                # Prepare the input data
                img = np.random.rand(224,224,3).astype(np.float32)
                return tf.keras.applications.mobilenet.preprocess_input(img)

            # Compile a MobileNet model for testing.
            # Typically, comilation need to be done one time only.
            model = tf.keras.applications.MobileNet()
            nc = NeuralCompiler(models=model)
            dfp = nc.run()

            # Accelerate using the MemryX hardware
            accl = SyncAccl(dfp)

            frame_count = 10
            send_count, recv_count = 0, 0
            while recv_count < frame_count:
                if send_count < frame_count:
                    accl.send(generate_frame())
                    send_count += 1
                try:
                    outputs = accl.receive(timeout=1)
                except TimeoutError:
                    continue # try sending the next frame if output is not ready yet
                recv_count += 1

        """
        m = model_idx
        self._ensure_valid_model_idx(m)
        output_count = len(self._models[m].output.port_ids)
        outputs = []
        first = self._models[m].output.receive(0, timeout=timeout)
        outputs.append(first)
        for i in range(1, output_count):
            outputs.append(self._models[m].output.receive(i))
        if len(outputs) == 1:
            return outputs[0]
        return outputs

    def run(self, inputs, model_idx=0):
        """
        This function combines send and receive in one sequential call.
        It sends the data to the specified model and waits until all the outputs are available to retrieve.

        Parameters
        ----------
            model_idx: int
                Index of the model
            inputs: np.ndarray or a list of np.ndarray or a doubly nested list of np.ndarray
                Typically the pre-processed input data array(s) of the model.
                Each np.ndarray's shape must match the shape expected by the model.
                Multiple sets of inputs can be batched together for greater performance
                compared to running a single set of inputs at a time due to internal pipelining.
                Stacking together multiple sets of inputs into a single np.ndarray is not
                supported.
        """
        self._ensure_valid_model_idx(model_idx)

        data = inputs
        stacked_data = []
        if not isinstance(data, (list, tuple)):
            if not isinstance(data, np.ndarray):
                raise TypeError(f"Expected np.ndarray, got: {type(data)}")
            stacked_data = [[data]] # 1 set, 1 input
        else:
            for item in data:
                if isinstance(item, np.ndarray):
                    stacked_data.append([item])
                elif not isinstance(item, (list, tuple)):
                    raise TypeError(f"Expected sequence of np.ndarray, got: {type(item)}")
                else:
                    for inner_item in item:
                        if isinstance(inner_item, np.ndarray):
                            continue
                        raise TypeError(f"Expected np.ndarray, got: {type(item)}")
                    stacked_data.append(item)

        output_count = len(self._models[model_idx].output.port_ids)
        all_outputs = self.__threaded_run(stacked_data, model_idx)

        unstacked_outputs = []
        for item in all_outputs:
            if len(item) == 1:
                unstacked_outputs.append(item[0])
            else:
                unstacked_outputs.append(item)

        if len(unstacked_outputs) == 1:
            return unstacked_outputs[0]

        return unstacked_outputs

    def __quasi_pipelined_run(self, stacked_data, output_count, model_idx):
        send_count, recv_count = 0, 0
        all_outputs = []
        while recv_count < len(stacked_data):
            if send_count < len(stacked_data):
                self.send(stacked_data[send_count], model_idx)
                send_count += 1
            try:
                out = self._models[model_idx].output.receive(0, timeout=1)
            except TimeoutError:
                continue
            else:
                outputs = [out]
                for i in range(1, output_count):
                    outputs.append(self._models[model_idx].output.receive(i))
                all_outputs.append(outputs)
                recv_count += 1
        return all_outputs

    def shutdown(self):
        """
        Shutdown the accelerator to make it available for other processes to use.
        If this function is not called while the current program is running,
        other processes will not be able to access the same accelerator until
        the current program terminates.
        Note that after calling shutdown, a new instance of SyncAccl must be created
        to re-initialize the accelerator, before it can be run again from the current
        program.
        """
        super().shutdown()

    def __threaded_run(self, stacked_data, model_idx):
        exc = []
        sender = threading.Thread(
                    target=self.__send, args=(stacked_data, model_idx, exc),
                    daemon=True
                    )
        all_outputs = []
        receiver = threading.Thread(
                target=self.__receive, args=(all_outputs, len(stacked_data), model_idx, exc),
                daemon=True
                )
        self._running = True
        receiver.start()
        sender.start()
        time.sleep(0.01)
        try:
            receiver.join()
            sender.join()
        except KeyboardInterrupt as e:
            logger.critical(f"Terminating run due to KeyboardInterrupt")
            self._running = False
            receiver.join()
            sender.join()
        if exc:
            raise exc[0] from None
        return all_outputs

    def __send(self, stacked_data, model_idx, exc):
        i = 0
        while self._running and i < len(stacked_data):
            try:
                data = stacked_data[i]
                self.send(data, model_idx)
            except Exception as e:
                self._running = False
                logger.critical(f"Terminating run due to exception in send: {str(e)}")
                exc.append(e)
                return
            i += 1

    def __receive(self, outputs, out_count, model_idx, exc):
        i = 0
        while self._running and i < out_count:
            try:
                out = self.receive(model_idx, timeout=1000)
            except TimeoutError:
                continue
            except Exception as e:
                exc.append(e)
                return
            if not isinstance(out, list):
                out = [out]
            outputs.append(out)
            i += 1

def cleanup(func):
    def cleanup_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self = args[0]
            self._accl._cleanup()
            raise e from None
    return cleanup_wrapper

class SyncAcclModel(AcclModel):
    def __init__(self, model_id, input_info, output_info, accl):
        super().__init__(model_id, input_info, output_info, accl)
        self.input = SyncAcclModelInput(model_id, input_info, accl, self)
        self.output = SyncAcclModelOutput(model_id, output_info, accl, self)

    def run(self, data, inport_idx, outport_idx):
        self.input.send(data, inport_idx)
        if isinstance(outport_idx, int):
            return self.output.receive(outport_idx)

        try:
            iter(outport_idx)
        except TypeError:
            raise TypeError(f"Expected outport_idx to be an int or iterable of ints but got: {type(outport_idx)}")

        outputs = []
        for port_idx in outport_idx:
            outputs.append(self.output.receive(port_idx))

        return outputs

class SyncAcclModelInput(AcclModelInput):
    def __init__(self, model_id, info, accl, model):
        super().__init__(model_id, info, accl, model)

    def send(self, data, port_idx, timeout=None):
        if not isinstance(port_idx, int):
            raise TypeError(f"Expected port_id to be an int but got: {type(port_idx)}")

        if port_idx not in range(len(self._info['ports'])):
            raise ValueError(f"Input port with idx: {port_idx} is inactive")

        if not isinstance(data, np.ndarray):
            raise TypeError(f"Expected data to be a `np.ndarray` but got: {type(data)}")

        if timeout is not None and not isinstance(timeout, numbers.Real):
            raise TypeError(f"Expected timeout to be a real number but got: {type(timeout)}")

        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative")

        if timeout is None:
            timeout = 0

        shape = list(data.shape)
        reqd_shape = list(self._info['shapes'][port_idx])
        if len(reqd_shape) == 4 and reqd_shape[-2] == 1 and len(shape) == 3:
            reqd_shape = reqd_shape[:-2] + [reqd_shape[-1]]
        if shape not in [reqd_shape, [1] + reqd_shape]:
            raise RuntimeError(
                    f"Port at index: {port_idx} expects data of shape: {reqd_shape}, "
                    f"but got: {shape}"
                    )
            if shape == [1] + reqd_shape:
                data = np.squeeze(data, 0)
        reqd_dtype = self._info['dtypes'][port_idx]
        if reqd_dtype != data.dtype:
            raise TypeError(
                    f"Port at index: {port_idx} expects data of type: {reqd_dtype}, "
                    f"but got: {data.dtype}"
                    )

        p = self._info['ports'][port_idx]

        # NOTE: explicitly copy the 'data' because we cannot
        # trust the user will leave the reference to 'data' alone while the
        # accelerator works on it. Perhaps we can add an 'immutable' flag in
        # the future to avoid this copy.
        err = mxa.stream_ifmap(self._model_id, p, data.copy(), timeout=timeout)

        if err:
            raise TimeoutError(f"Driver stream input function timed out for port idx: {port_idx}")

class SyncAcclModelOutput(AcclModelOutput):
    def __init__(self, model_id, info, accl, model):
        super().__init__(model_id, info, accl, model)

    def receive(self, port_idx, timeout=None):
        if timeout is not None and not isinstance(timeout, numbers.Real):
            raise TypeError(f"Expected timeout to be a real number but got: {type(timeout)}")

        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative")

        if not isinstance(port_idx, int):
            raise TypeError(f"Expected port_idx to be an int but got: {type(port_idx)}")

        if port_idx not in range(len(self._info['ports'])):
            raise ValueError(f"Output port with idx: {port_idx} is inactive")

        if timeout is None:
            timeout = 0

        shape = self._info['shapes'][port_idx]
        ofmap = np.zeros(shape).astype(np.float32)
        p = self._info['ports'][port_idx]
        err = mxa.stream_ofmap(self._model_id, p, ofmap, timeout=timeout)
        #TODO store original model output shapes in the DFP and reshape to those here
        if len(ofmap.shape) == 4 and ofmap.shape[-2] == 1:
            ofmap = np.squeeze(ofmap, -2)
        if err:
            raise TimeoutError(f"Driver stream output function timed out for port at idx: {port_idx}")
        return ofmap

class AsyncAccl(Accl):
    """
    This class provides an asynchronous API to run models on the MemryX hardware accelerator.
    The user provides callback functions to feed data and receive outputs from the accelerator,
    which are then called whenever a model is ready to accept/output data.
    This pipelines execution of the models and allows the accelerator to run at full speed.

    Parameters
    ----------
    dfp: bytes or string
        Path to dfp or a dfp object (bytearray). The dfp is generated by the NeuralCompiler.
    group_id: int
        The index of the chip group to select.

    Examples
    --------

    .. code-block:: python

        import tensorflow as tf
        import numpy as np
        from memryx import NeuralCompiler, AsyncAccl

        # define the callback that will return model input data
        def data_source():
            for i in range(10):
                img = np.random.rand(224,224,3).astype(np.float32)
                data = tf.keras.applications.mobilenet.preprocess_input(img)
                yield data

        # define the callback that will process the outputs of the model
        def output_processor(*outputs):
            logits = np.squeeze(outputs[0], 0)
            preds = tf.keras.applications.mobilenet.decode_predictions(logits)

        # Compile a MobileNet model for testing.
        # Typically, comilation need to be done one time only.
        model = tf.keras.applications.MobileNet()
        nc = NeuralCompiler(models=model)
        dfp = nc.run()

        # Accelerate using the MemryX hardware
        accl = AsyncAccl(dfp)
        accl.connect_input(data_source) # starts asynchronous execution of input generating callback
        accl.connect_output(output_processor) # starts asynchronous execution of output processing callback
        accl.wait() # wait for the accelerator to finish execution

    """
    def __init__(self, dfp, group_id=0, **kwargs):
        super().__init__(dfp, group_id, **kwargs)

    def _create_models(self):
        self._models = []
        for m in self._model_idx_to_in_info:
            self._models.append(AsyncAcclModel(self._model_id, self._model_idx_to_in_info[m], self._model_idx_to_out_info[m], self))

    def connect_input(self, callback, model_idx=0):
        """
        Sets a callback function to execute when this model is ready to start processing
        an input frame.

        Parameters
        ----------
        callback: callable
            This callable is invoked asynchonously whenever the accelerator is ready
            to start processing an input frame through the model specified by `model_idx`.
            `callback` is responsible for generating and returning the next input frame(s) for this
            model. `callback` must not take any arguments and it may return either a
            single np.ndarray if the model has a single input, or a sequence of
            np.ndarrays for multi-input models. The data types of the np.ndarrays must match
            those expected by the model.

            Any exception raised when calling `callback` is taken to signal the end of the
            input stream for this model. Returning None from the `callback` is also taken
            to signal the end of input stream for this model.

            If a pre-processing model was set by the calling the `set_preprocessing`
            method, the outputs of `callback` will first be run through the pre-processing
            model and resulting outputs will be fed to the accelerator.
        model_idx: int
            Index of the model whose input should be connected to `callback`

        Raises
        ------
        RuntimeError: If the signature of `callback` contains any paramters
        """
        self._ensure_valid_model_idx(model_idx)
        self._models[model_idx].input.connect(callback)

    def connect_output(self, callback, model_idx=0):
        """
        Sets a callback function to execute when the outputs of this model are ready.

        Parameters
        ----------
        callback: callable
            This callable is invoked asynchonously whenever the accelerator finishes
            processing an input frame for the model specified by `model_idx`. The outputs of the model are
            passed to this callable according to the port order assigned to the model,
            which is returned by the `outport_assignment` method. The
            signature of `callback` must only consist of parameters that correspond to
            the model output feature maps, no other parameter may be present.

            If a post-processing model was set by the calling the `set_postprocessing`
            method, the outputs of the model will first be run through the post-processing
            model and resulting outputs will be used instead to call `callback`.
        model_idx: int
            Index of the model whose output should be connected to `callback`

        """
        self._ensure_valid_model_idx(model_idx)
        self._models[model_idx].output.connect(callback)

    def stop(self):
        """
        Send a signal to stop each of the models running on the accelerator.
        This call blocks until each of the models stops and cleans up its
        resources.
        """
        for model in self._models:
            model.stop()
        time.sleep(0.01)

    def wait(self):
        """
        Make the main thread wait for the accelerator to finish executing all models.

        Raises
        ------
        RuntimeError: If the any of the model's inputs/outputs are left unconnected
        """
        for i, model in enumerate(self._models):
            if not model.input.connected():
                raise RuntimeError(
                        f"Model {i}'s input is not connected, "
                        f"please call `connect_input(f, {i})` "
                        "where f is the callback function that feeds data to Model {i}"
                        )
            if not model.output.connected():
                raise RuntimeError(
                        f"Model {i}'s output is not connected, "
                        f"please call `connect_output(f, {i})` "
                        "where f is the callback function that "
                        "consumes data output by the Model {i}"
                        )

        try:
            for model in self._models:
                model.wait()
        except KeyboardInterrupt:
            for model in self._models:
                model.stop()

    def shutdown(self):
        """
        Stop all currently running models and shutdown the accelerator
        to make it available for other processes to use. Calling stop() only
        stops the running models, but doesn't allow other processes to use
        the same accelerator while the current program is running.
        Note that after calling shutdown, a new instance of AsyncAccl must be
        created to re-initialize the accelerator, before it can be run again
        from the same program.
        """
        self.stop()
        super().shutdown()

    def set_preprocessing_model(self, model_or_path, model_idx=0):
        """
        Supply the path to a model/file that should be run to pre-process the input feature map.
        This is an optional feature that can be used to automatically run the pre-processing model
        output by the NeuralCompiler

        .. note::

            This function currently does not support PyTorch models

        Parameters
        ----------
        model_or_path: obj or str
            Can be either an already loaded model such as a tf.keras.Model object for Keras,
            or a str path to a model file.
        model_idx: int
            Index of the model on the accelerator whose input feature map should be
            pre-processed by the supplied model
        """
        self._ensure_valid_model_idx(model_idx)
        self._models[model_idx].input.set_preprocessing(model_or_path)

    def set_postprocessing_model(self, model_or_path, model_idx=0):
        """
        Supply the path to a model/file that should be run to post-process the output feature maps
        This is an optional feature that can be used to automatically run the post-processing model
        output by the NeuralCompiler

        .. note::

            This function currently does not support PyTorch models

        Parameters
        ----------
        model_or_path: obj or str
            Can be either an already loaded model such as a tf.keras.Model object for Keras,
            or a string path to a model file.
        model_idx: int
            Index of the model on the accelerator whose output should be
            post-processed by the supplied model

        """
        self._ensure_valid_model_idx(model_idx)
        self._models[model_idx].output.set_postprocessing(model_or_path)

    def _cleanup(self):
        self.stop()
        super()._cleanup()

class AsyncAcclModel(AcclModel):
    def __init__(self, model_id, input_info, output_info, accl):
        super().__init__(model_id, input_info, output_info, accl)
        self.input = AsyncAcclModelInput(model_id, input_info, accl, self)
        self.output = AsyncAcclModelOutput(model_id, output_info, accl, self)

    def wait(self):
        self.input.wait()
        self.output.wait()

    def stop(self):
        self.input.stop()
        self.output.stop()

def graceful_shutdown(func):
    def exc_handling_wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            buf = io.StringIO()
            traceback.print_exc(file=buf)
            logger.critical(buf.getvalue()) # log exception now as other model.stop() may block forever
            buf.close()

            self = args[0]
            self._stop_event.set()

            raise e from None

    return exc_handling_wrapper

class AsyncAcclModelInput(AcclModelInput):
    def __init__(self, model_id, info, accl, model):
        super().__init__(model_id, info, accl, model)
        self._callback = None
        self._thread = None
        self._stop_event = threading.Event()
        self._pre_model = _PassThroughModel()
        self._call_counter = 0
        self._call_counter_lock = threading.Lock()
        self._generator_iter = None
        self._worker_exc_log = []

    def connected(self):
        """
        Return whether a callback was set for execution
        """
        return self._callback is not None

    def connect(self, callback):
        sig = inspect.signature(callback)
        if sig.parameters:
            raise RuntimeError(
                "Input `callback` must not have any parameters other than "
                "the implicit self for bound methods"
                )
        self._callback = callback
        if self._thread is not None:
            self.stop()
            self._stop_event = threading.Event()
        model_idx = self._accl.models.index(self._model)
        thread_name = f'Model {model_idx} input function'
        self._thread = threading.Thread(
                target=self._worker,
                args=(callback, self._worker_exc_log),
                name=thread_name,
                )
        self._thread.start()

    def wait(self):
        self._thread.join()

    def stop(self):
        if self.stopped():
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()

    def stopped(self):
        return self._stop_event.is_set()

    def _get_frames(self, callback):
        if self._generator_iter is not None:
            frames = advance_iter(self._generator_iter)
        else:
            try:
                cb_result = callback()
            except StopIteration: # for the case where the app code calls next(iter)
                return None
            if cb_result is None:
                return None
            if type(cb_result).__name__ == 'generator':
                self._generator_iter = cb_result
                frames = advance_iter(cb_result)
            elif hasattr(cb_result, '__next__'):
                frames = advance_iter(cb_result)
            else:
                frames = cb_result
        validate_input_callback_result(frames)
        return frames

    @graceful_shutdown
    def _worker(self, callback, worker_exc_log):
        model_idx = self._accl.models.index(self._model)
        while not self._stop_event.is_set():
            try:
                input_frames = self._get_frames(callback)
            except Exception as e:
                logger.critical(f'Model {model_idx} input stream terminated due to an exception related to the callback function: {str(e)}')
                worker_exc_log.append(e)
                self._stop_event.set()
                raise e from None

            if input_frames is None:
                self._stop_event.set()
                return

            try:
                input_frames = self._pre_model.predict_for_mxa(input_frames, format_output=True, output_order=self._info['layers'])
            except Exception as e:
                logger.critical(f'Model {model_idx} input stream terminated due to an exception related to the pre-processing function: {str(e)}')
                worker_exc_log.append(e)
                raise RuntimeError(
                        "Failed to run inference on the pre-processing model\n"
                        "Ensure that the pre-processing model passed to `set_preprocessing_model()` "
                        "and the model in the DFP come from the same model passed to the NeuralCompiler"
                        ) from None

            try:
                self._stream_ifmaps(input_frames)
            except Exception as e:
                logger.critical(f'Model {model_idx} input stream terminated due to an exception during streaming inputs: {str(e)}')
                worker_exc_log.append(e)
                raise e from None

            with self._call_counter_lock:
                self._call_counter += 1

class AsyncAcclModelOutput(AcclModelOutput):
    def __init__(self, model_id, info, accl, model):
        super().__init__(model_id, info, accl, model)
        self._callback = None
        self._thread = None
        self._stop_event = threading.Event()
        self._call_counter = 0
        self._call_counter_lock = threading.Lock()
        self._worker_exc_log = []

    def connected(self):
        """
        Return whether a callback was set for execution
        """
        return self._callback is not None

    def connect(self, callback):
        self._callback = callback
        if self._thread is not None:
            self.stop()
            self._stop_event = threading.Event()
        self._model_idx = self._accl.models.index(self._model)
        thread_name = f'Model {self._model_idx} output function'
        self._thread = threading.Thread(
                target=self._worker,
                args=(callback, self._model.input, self._model_idx, self._worker_exc_log),
                name=thread_name,
                )
        self._thread.start()

    def wait(self):
        self._thread.join()

    def stop(self):
        if self.stopped():
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()

    def stopped(self):
        return self._stop_event.is_set()

    @graceful_shutdown
    def _worker(self, callback, model_input, model_idx, exc_log):

        def _all_frames_processed():
            with model_input._call_counter_lock:
                all_done = self._call_counter == model_input._call_counter
            return all_done

        outputs = []
        outputs_by_name = {}
        while not self._stop_event.is_set():
            if model_input.stopped() and _all_frames_processed():
                return

            streamed = self._stream_ofmaps(model_idx, outputs, outputs_by_name)
            if not streamed:
                continue

            try:
                fmaps = self._post_model.predict_for_mxa(outputs_by_name)
            except Exception as e:
                exc_log.append(e)
                model_input.stop()
                raise RuntimeError(
                    "Failed to run inference on the post-processing model\n"
                    "Ensure that the post-processing model passed to `set_postprocessing_model()` "
                    "and the model in the DFP come from the same model passed to the NeuralCompiler"
                    ) from None

            try:
                callback(*fmaps)
            except Exception as e:
                exc_log.append(e)
                logging.critical(f'Model {model_idx} output processing terminated due to an exception in the callback function: {str(e)}')
                model_input.stop()
                raise e from None

            with self._call_counter_lock:
                self._call_counter += 1

        # flush out ofmaps left in the driver output queue
        while self._stream_ofmaps(model_idx, outputs, outputs_by_name):
            pass


class MultiStreamAsyncAccl(Accl):
    """
    This class provides a multi-stream version of the AsyncAccl API. This allows multiple input+output callbacks
    to be associated with a single model.

    Parameters
    ----------
    dfp: bytes or string
        Path to dfp or a dfp object (bytearray). The dfp is generated by the NeuralCompiler.
    group_id: int
        The index of the chip group to select.

    Examples
    --------

    .. code-block:: python

        import numpy as np
        import tensorflow as tf
        from memryx import NeuralCompiler, MultiStreamAsyncAccl

        class Application:
            # define the callback that will return model input data
            def __init__(self):
                self.streams = []
                self.streams_idx = []
                self.outputs = []
                for i in range(2):
                    self.streams.append([np.random.rand(224,224,3).astype(np.float32) for i in range(10)])
                    self.streams_idx.append(0)
                    self.outputs.append([])

            def data_source(self, stream_idx):
                # generate inputs based on stream_idx
                if self.streams_idx[stream_idx] == len(self.streams[stream_idx]):
                    return None
                self.streams_idx[stream_idx]+=1
                return self.streams[stream_idx][self.streams_idx[stream_idx]-1]

            # define the callback that will process the outputs of the model
            def output_processor(self, stream_idx, *outputs):
                logits = np.squeeze(outputs[0], 0)
                preds = tf.keras.applications.mobilenet.decode_predictions(logits)
                # route outputs based on stream_idx
                self.outputs[stream_idx].append(preds)

        # Compile a MobileNet model for testing.
        # Typically, comilation need to be done one time only.
        model = tf.keras.applications.MobileNet()
        nc = NeuralCompiler(models=model,verbose=1)
        dfp = nc.run()

        # Accelerate using the MemryX hardware
        app = Application()
        accl = MultiStreamAsyncAccl(dfp)
        accl.connect_streams(app.data_source, app.output_processor, 2) # starts asynchronous execution of input output callback pair associated with 2 streams
        accl.wait() # wait for the accelerator to finish execution

    """
    def __init__(self, dfp, group_id=0, stream_workers=None, **kwargs):
        if stream_workers is None:
            stream_workers = os.cpu_count() - 1
        stream_workers = max(2, stream_workers)
        self._input_task_pool = InputTaskPool(stream_workers)
        super().__init__(dfp, group_id, **kwargs)
        self._input_tasks = {}

    def _create_models(self):
        self._models = []
        self._stream_idx = []
        for i, m in enumerate(self._model_idx_to_in_info):
            self._models.append(MultiStreamAsyncAcclModel(self._model_id, self._model_idx_to_in_info[m], self._model_idx_to_out_info[m], self, str(i)))
            self._stream_idx.append(0)

    def connect_streams(self, input_callback, output_callback, stream_count, model_idx=0):
        """
        Registers and starts execution of a pair of input and output callback functions that processes `stream_count` number of data sources

        Parameters
        ----------
        input_callback: callable
            A function/bound method that returns the input data to consumed by the model identified by `model_idx`.
            It must have exactly one parameter `stream_idx` which is the index from 0 to `stream_count` - 1 for the model at `model_idx`
            which is used in the application code to distinguish/select the appropriate data source from which the data is returned
        output_callback: callable
            A function/bound method that is called with the output feature maps generated by the model.
            It must have at least 2 parameters: `stream_idx` and either a packed `*fmaps` or fmap0, fmap1, ... depending on the
            number of output feature maps generated by the model
        stream_count: int
            The number of input feature map sources/streams to the model.
        model_idx: int
            The target model for this pair of input and output callback functions.
            Each model will have its own `stream_idx` in the range of 0 to `stream_count - 1`
        """
        if not isinstance(stream_count, int):
            raise TypeError("stream_count must be an integer")
        if stream_count <= 0:
            raise ValueError("stream_count must be positive")
        sig = inspect.signature(input_callback)
        if len(sig.parameters) != 1:
            raise TypeError(
                "input_callback must have exactly 1 parameter (stream index) "
                "other than the implicit self for bound methods"
                )
        sig = inspect.signature(output_callback)
        if len(sig.parameters) < 2:
            raise TypeError(
                "output_callback must have at least 2 parameters (stream index, *fmaps) "
                "other than the implicit self for bound methods"
                )
        self._ensure_valid_model_idx(model_idx)
        for i in range(stream_count):
            stream = Stream(model_idx, self._stream_idx[model_idx], input_callback, output_callback)
            task = InputTask(self._models[model_idx], stream)
            task_key = self._get_task_key(self._stream_idx[model_idx], input_callback, output_callback)
            self._input_tasks[task_key] = task
            self._input_task_pool.add_task(task)
            self._stream_idx[model_idx] += 1
        try:
            self._models[model_idx]._connect_stream(input_callback, output_callback, stream_count)
        except Exception as e:
            self.stop()
            raise e from None

    def set_preprocessing_model(self, model_or_path, model_idx=0):
        """
        Supply the path to a model/file that should be run to pre-process the input feature map.
        This is an optional feature that can be used to automatically run the pre-processing model
        output by the NeuralCompiler

        .. note::

            This function currently does not support PyTorch models

        .. warning::

            This function is currently not available on the ARM platform

        Parameters
        ----------
        model_or_path: obj or str
            Can be either an already loaded model such as a tf.keras.Model object for Keras,
            or a str path to a model file.
        model_idx: int
            Index of the model on the accelerator whose input feature map should be
            pre-processed by the supplied model
        """
        self._ensure_valid_model_idx(model_idx)
        self._models[model_idx].input.set_preprocessing(model_or_path)

    def set_postprocessing_model(self, model_or_path, model_idx=0):
        """
        Supply the path to a model/file that should be run to post-process the output feature maps
        This is an optional feature that can be used to automatically run the post-processing model
        output by the NeuralCompiler

        .. note::

            This function currently does not support PyTorch models

        .. warning::

            This function is currently not available on the ARM platform

        Parameters
        ----------
        model_or_path: obj or str
            Can be either an already loaded model such as a tf.keras.Model object for Keras,
            or a string path to a model file.
        model_idx: int
            Index of the model on the accelerator whose output should be
            post-processed by the supplied model

        """
        self._ensure_valid_model_idx(model_idx)
        self._models[model_idx].output.set_postprocessing(model_or_path)

    def stop(self):
        """
        Sends a signal to stop each of the models running on the accelerator.
        This call blocks until each of the models stops and cleans up its
        resources.
        """
        self._input_task_pool.stop()
        for model in self._models:
            model.stop()

    def wait(self):
        """
        Blocks the application thread until the accelerator finishes executing all models.

        Raises
        ------
        RuntimeError: If the any of the model's inputs/outputs are left unconnected
        """
        try:
            for model in self._models:
                model.wait()
        except KeyboardInterrupt:
            for model in self._models:
                model.stop()
        self._input_task_pool.stop()

    def _get_task_key(self, index, input_callback, output_callback):
        return (index, input_callback, output_callback)

class InputTask:
    def __init__(self, model, stream):
        self.model = model
        self.stream = stream

    def __str__(self):
        return f'Model {self.model.name()} stream {self.stream.index}'

class InputTaskPool:
    def __init__(self, count):
        self._count = count
        self._task_queue = queue.Queue()
        self._tasks = set()
        self._workers = []
        self._lock = threading.Lock()
        for i in range(count):
            worker_name = f'InputWorker-{i}'
            self._workers.append(threading.Thread(target=self._worker_target,
                                                  args=(),
                                                  daemon=True,
                                                  name=worker_name))
        self._stop_event = threading.Event()
        for worker in self._workers:
            worker.start()

    def _worker_target(self):
        while not self._stop_event.is_set():
            try:
                task = self._task_queue.get(timeout=0.5)
            except queue.Empty:
                logger.debug(f'Input worker get task timed out')
                continue
            logger.debug(f'Input worker dequeued {task}')
            if task not in self._tasks:
                continue
            logger.debug(f'Input worker executing {task} callback')
            input_frames = self._execute_task(task)
            logger.debug(f'Input worker received frames from {task} callback')
            self._enqueue_model_input_frames(task, input_frames)
            finished = input_frames is None
            if finished:
                logger.debug(f'{task} ended')
                self._tasks.remove(task)
                continue
            self._task_queue.put(task)

    def _execute_task(self, task):
        stream_idx = task.stream.index

        try:
            input_frames = self._get_frames(task.stream.input_callback, stream_idx)
        except Exception as e:
            logger.critical(f'Model {task.model.name()} input stream {stream_idx} terminated due to an exception related to the callback function: ')
            logger.critical("AssertionError") if isinstance(e, AssertionError) else logger.critical(str(e))
            raise e from None

        if input_frames is None:
            return input_frames

        mpu_input_order = task.model.input._info['layers']

        self._lock.acquire()
        try:
            input_frames = task.model.input._pre_model.predict_for_mxa(input_frames, format_output=True, output_order=mpu_input_order)
        except Exception as e:
            raise RuntimeError(
                "Failed to run inference on the pre-processing model\n"
                "Ensure that the pre-processing model passed to `set_preprocessing_model()` "
                "and the model in the DFP come from the same model passed to the NeuralCompiler"
                ) from None
        self._lock.release()

        return input_frames

    def _enqueue_model_input_frames(self, task, input_frames):
        while not self._stop_event.is_set():
            try:
                task.model.input.put(input_frames, task.stream, timeout=0.5)
            except queue.Full:
                logger.debug(f'Model {task.model.name()} input queue full')
            else:
                logger.debug(f'Model {task.model.name()} input queue got frames')
                return

    def _get_frames(self, callback, *cb_args):
        try:
            cb_result = callback(*cb_args)
        except StopIteration:
            return None
        if cb_result is None:
            return None
        if type(cb_result).__name__ == 'generator':
            raise TypeError("input_callback must not return a generator")
        elif hasattr(cb_result, '__next__'):
            frames = advance_iter(cb_result)
        else:
            frames = cb_result
        validate_input_callback_result(frames)
        return frames

    def add_task(self, task):
        self._task_queue.put(task)
        self._tasks.add(task)
        logger.debug(f'Enqueued {task}')

    def remove_task(self, task):
        if not self.has_task(task):
            raise ValueError("task not found")
        self._tasks.remove(task)

    def has_task(self, task):
        return task in self._tasks

    def stop(self):
        if self.stopped():
            return
        self._stop_event.set()
        logger.debug(f'Input worker pool stop signal sent')
        for worker in self._workers:
            worker.join()

    def stopped(self):
        return self._stop_event.is_set()


class MultiStreamAsyncAcclModel(AcclModel):
    def __init__(self, model_id, input_info, output_info, accl, name):
        super().__init__(model_id, input_info, output_info, accl)
        self._name = name
        self.input = MultiStreamAsyncAcclModelInput(model_id, input_info, accl, self)
        self.output = MultiStreamAsyncAcclModelOutput(model_id, output_info, accl, self)

    def _connect_stream(self, input_callback, output_callback, stream_count):
        self.output.connect(output_callback, stream_count)
        self.input.connect(input_callback, stream_count)

    def wait(self):
        self.input.wait()
        self.output.wait()
        logger.debug(f"Model {self._name} wait ended")

    def stop(self):
        self.input.stop()
        self.output.stop()
        logger.debug(f"Model {self._name} stopped")

    def name(self):
        return self._name

class Stream:
    def __init__(self, model_idx, index, input_callback, output_callback):
        self.model_idx = model_idx
        self.index = index
        self.input_callback = input_callback
        self.output_callback = output_callback

    def __str__(self):
        return f'Stream {self.index}'

class MultiStreamAsyncAcclModelInput(AcclModelInput):
    def __init__(self, model_id, info, accl, model):
        super().__init__(model_id, info, accl, model)
        self._connected = False
        self._stop_event = threading.Event()
        self._call_counter = 0
        self._call_counter_lock = threading.Lock()
        self._merge_queue = queue.Queue(maxsize=10) # TODO: consider a max-size based on frame memory consumption and expose to user?
        self._output_stream_queue = queue.Queue()
        self._pre_model = _PassThroughModel()
        self._thread = threading.Thread(
                target=self._worker,
                args=(self._merge_queue, self._output_stream_queue),
                name=f'Model {model.name()} InputWorker'
                )
        self._stream_log = deque([], maxlen=1000) # for making assertions stream routing in tests
        self._stream_count = 0

    def connected(self):
        """
        Return whether a callback was set for execution
        """
        return self._connected

    def connect(self, input_callback, stream_count):
        """
        Set an input callback
        """
        self._connected = True
        self._stream_count += stream_count
        if not self._thread.is_alive():
            self._thread.start()

    def put(self, input_frames, stream, timeout):
        self._merge_queue.put((input_frames, stream), timeout=timeout)

    def wait(self):
        if self._thread.is_alive():
            self._thread.join()
        logger.debug(f"Model {self._model.name()} input wait ended")

    def stop(self):
        if self.stopped():
            return
        self._stop_event.set()
        logger.debug(f"Model {self._model.name()} input stop signal sent")
        if self._thread.is_alive():
            self._thread.join()
        logger.debug(f"Model {self._model.name()} input stopped")

    def stopped(self):
        return self._stop_event.is_set()

    @graceful_shutdown
    def _worker(self, merge_queue, output_stream_queue):
        done_count = 0
        while not self._stop_event.is_set():
            try:
                input_frames, stream = merge_queue.get(timeout=0.5)
            except queue.Empty:
                logger.debug(f'Model {self._model.name()} get frames timed out')
                continue
            if input_frames is None:
                done_count += 1
                logger.debug(f'Model {self._model.name()} finished stream count: {done_count}')
                if done_count == self._stream_count:
                    self._stop_event.set()
                    logger.debug(f'Model {self._model.name()} finished all streams')
                    break
                continue

            self._stream_log.append(stream.index)
            output_stream_queue.put(stream)

            self._stream_ifmaps(input_frames)
            logger.debug(f'Model {self._model.name()} stream inputs successful')

            with self._call_counter_lock:
                self._call_counter += 1

class MultiStreamAsyncAcclModelOutput(AcclModelOutput):
    def __init__(self, model_id, info, accl, model):
        super().__init__(model_id, info, accl, model)
        self._connected = False
        self._stop_event = threading.Event()
        self._post_model = _PassThroughModel()
        self._call_counter = 0
        self._call_counter_lock = threading.Lock()
        self._thread = threading.Thread(
                target=self._worker,
                args=(model,),
                name=f'Model {model.name()} OutputWorker'
                )
        self._stream_log = deque([], maxlen=1000)
        self._stream_count = 0

    def connected(self):
        """
        Return whether a callback was set for execution
        """
        return self._connected

    def connect(self, output_callback, stream_count):
        self._connected = True
        self._stream_count += stream_count
        if not self._thread.is_alive():
            self._thread.start()

    def wait(self):
        if self._thread.is_alive():
            self._thread.join()
        logger.debug(f"Model {self._model.name()} output wait ended")

    def stop(self):
        if self.stopped():
            return
        self._stop_event.set()
        logger.debug(f"Model {self._model.name()} output stop signal sent")
        if self._thread.is_alive():
            self._thread.join()
        logger.debug(f"Model {self._model.name()} output stopped")

    def stopped(self):
        return self._stop_event.is_set()

    def _all_frames_processed(self):
        with self._model.input._call_counter_lock:
            all_done = self._call_counter == self._model.input._call_counter
        return all_done

    @graceful_shutdown
    def _worker(self, model):
        outputs = []
        outputs_by_name = {}

        while not self._stop_event.is_set():
            if model.input.stopped() and self._all_frames_processed():
                logger.debug(f"Model {model.name()} streamed all outputs")
                return

            streamed = self._stream_ofmaps(model.name(), outputs, outputs_by_name)
            if not streamed:
                continue

            stream = self._model.input._output_stream_queue.get()
            self._stream_log.append(stream.index)

            try:
                fmaps = self._post_model.predict_for_mxa(outputs_by_name)
            except Exception as e:
                raise RuntimeError(
                    "Failed to run inference on the post-processing model\n"
                    "Ensure that the post-processing model passed to `set_postprocessing_model()` "
                    "and the model in the DFP come from the same model passed to the NeuralCompiler"
                    ) from None

            try:
                stream.output_callback(stream.index, *fmaps)
            except Exception as e:
                logger.critical(f'{stream} output processing terminated due to an exception in the callback function:')
                self._model.input.stop()
                raise e from None

            with self._call_counter_lock:
                self._call_counter += 1

        # flush out ofmaps left in the driver output queue
        while self._stream_ofmaps(model.name(), outputs, outputs_by_name):
            pass

class _MultiProcessNumpyFifo:
    def __init__(self, num, shape, dtype=np.float32, fps=None):
        self.shape = shape
        self.num = num
        self.dtype = dtype
        self.fps = fps

        self.size = np.array((1,), dtype=dtype).nbytes*int(np.product([num] + list(shape)))

        # Arr[0] = Head; Arr[1] = Tail; Arr[2] = IsFull;
        self.arr = multiprocessing.Array('i', 3)
        self.rawbuffer = multiprocessing.RawArray('B', self.size)

        self.buffer = np.ndarray([num] + list(shape), dtype=dtype,
                buffer=self.rawbuffer)

    def put(self, data, block=True, timeout=None):
        t1 = time.perf_counter()
        if block:
            if timeout is None:
                timeout = np.inf
            while time.perf_counter() - t1 < timeout and self.__is_full():
                time.sleep(1e-5)

        self.__put(data)

    def __put(self, data):
        if not all([int(s0)==int(s1) for s0,s1 in zip(data.shape,self.shape)]):
            raise ValueError('Data shape {} != {}'.format(data.shape, self.shape))

        if self.__is_full():
            raise queue.Full

        with self.arr.get_lock():
            np.copyto(self.buffer[self.arr[0]], data)

            if self.arr[0]+1 < self.num:
                self.arr[0] += 1
            else:
                self.arr[0] = 0

            if self.arr[0] == self.arr[1]:
                self.arr[2] = 1


    def __is_full(self):
        with self.arr.get_lock():
            return (self.arr[0] == self.arr[1]) and self.arr[2]==1

    def full(self):
        return self.__is_full()

    def get(self, block=True, timeout=None):
        t1 = time.perf_counter()
        if block:
            if timeout is None:
                timeout = np.inf
            while time.perf_counter() - t1 < timeout and self.__is_empty():
                time.sleep(1e-5)

        data = self.__get()

        return data

    def __is_empty(self):
        with self.arr.get_lock():
            return self.arr[0] == self.arr[1] and self.arr[2]==0

    def empty(self):
        return self.__is_empty()

    def __get(self):
        if self.__is_empty():
            raise queue.Empty

        with self.arr.get_lock():
            data = self.buffer[self.arr[1]].copy().astype(self.dtype)

            if self.arr[1]+1 < self.num:
                self.arr[1] += 1
            else:
                self.arr[1] = 0

            self.arr[2] = 0

            return data

class _CounterEvent:
    def __init__(self, n):
        super().__init__()
        self._n = n
        self._value = n
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._event.set() # ready at init to prevent blocking

    def set(self):
        with self._lock:
            self._value -= 1
            if self._value == 0:
                self._event.set()
                self._value = self._n

    def clear(self):
        self._event.clear()

    def is_set(self):
        return self._event.is_set()

    def wait(self, timeout=None):
        return self._event.wait(timeout=timeout)
