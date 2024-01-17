import ctypes
import cupy as cp
import cupyx
import tensorrt as trt


# Following code was written by DEB.
# ================================================
import torch
import numpy as np

from collections import OrderedDict, namedtuple

Binding = namedtuple(
    "Binding", 
    ("name", "dtype", "shape", "data", "ptr")
)
# ================================================


class HostDeviceMem:
    def __init__(self, size, dtype):
        self.size = size
        self.dtype = dtype
        self.host = cupyx.empty_pinned(size, dtype)
        self.device = cp.empty(size, dtype)

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

    @property
    def nbytes(self):
        return self.host.nbytes

    @property
    def hostptr(self):
        return self.host.ctypes.data

    @property
    def devptr(self):
        return self.device.data.ptr

    def copy_htod_async(self, stream):
        self.device.data.copy_from_host_async(self.hostptr, self.nbytes, stream)

    def copy_dtoh_async(self, stream):
        self.device.data.copy_to_host_async(self.hostptr, self.nbytes, stream)


class TRTInference:
    # initialize TensorRT
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')

    def __init__(self, model, batch_size):
        self.model = model
        self.batch_size = batch_size

        # Original code had the following code block
        # not commented.
        # =============================================
        # Load plugin if the model requires one.
        if self.model.PLUGIN_PATH is not None:
            try:
                ctypes.cdll.LoadLibrary(self.model.PLUGIN_PATH)
            except OSError as err:
                raise RuntimeError('Plugin not found') from err
        # =============================================

        # load trt engine or build one if not found
        if not self.model.ENGINE_PATH.exists():
            self.engine = self.model.build_engine(TRTInference.TRT_LOGGER, self.batch_size)
        else:
            runtime = trt.Runtime(TRTInference.TRT_LOGGER)
            with open(self.model.ENGINE_PATH, 'rb') as engine_file:
                self.engine = runtime.deserialize_cuda_engine(engine_file.read())
        if self.engine is None:
            raise RuntimeError('Unable to load the engine file')
        if self.engine.has_implicit_batch_dimension:
            assert self.batch_size <= self.engine.max_batch_size
        self.context = self.engine.create_execution_context()
        self.stream = cp.cuda.Stream()

        # allocate buffers
        self.bindings = []
        self.outputs = []
        self.input = None
        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            size = trt.volume(shape)
            if self.engine.has_implicit_batch_dimension:
                size *= self.batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # allocate host and device buffers
            buffer = HostDeviceMem(size, dtype)
            # append the device buffer to device bindings
            self.bindings.append(buffer.devptr)
            if self.engine.binding_is_input(binding):
                if not self.engine.has_implicit_batch_dimension:
                    msg = "-" * 10 + f"Batch size: {batch_size}, Shape[0]: {shape[0]}" + "-" * 10  # DEB
                    assert self.batch_size == shape[0], msg  # DEB
                    # assert self.batch_size == shape[0], msg  # ORIGINAL
                # expect one input
                self.input = buffer
            else:
                self.outputs.append(buffer)
        assert self.input is not None

        # timing events
        self.start = cp.cuda.Event()
        self.end = cp.cuda.Event()

    def __del__(self):
        if hasattr(self, 'context'):
            self.context.__del__()
        if hasattr(self, 'engine'):
            self.engine.__del__()

    def infer(self):
        self.infer_async()
        return self.synchronize()

    def infer_async(self, from_device=False):
        self.start.record(self.stream)
        if not from_device:
            self.input.copy_htod_async(self.stream)
        if self.engine.has_implicit_batch_dimension:
            self.context.execute_async(batch_size=self.batch_size, bindings=self.bindings,
                                       stream_handle=self.stream.ptr)
        else:
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.ptr)
        for out in self.outputs:
            out.copy_dtoh_async(self.stream)
        self.end.record(self.stream)

    def synchronize(self):
        self.stream.synchronize()
        return [out.host for out in self.outputs]

    def get_infer_time(self):
        self.end.synchronize()
        return cp.cuda.get_elapsed_time(self.start, self.end)


class YOLOv7TRTInference:
    # Initialize TensorRT.
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(TRT_LOGGER, namespace="")

    def __init__(self, 
                 engine_path, 
                 batch_size=1, 
                 device="cpu"):
        self.batch_size = batch_size
        self.device = device

        # Load trt engine.
        runtime = trt.Runtime(YOLOv7TRTInference.TRT_LOGGER)
        self.engine = None
        with open(engine_path, "rb") as engine_file:
            self.engine = runtime.deserialize_cuda_engine(engine_file.read())
        if self.engine is None:
            raise RuntimeError("Unable to load the engine file.")

        # Create execution context.
        self.context = self.engine.create_execution_context()
        self.context.set_binding_shape(
            0, (self.batch_size, 3, 960, 960)
        )

        # Allocate buffers.
        self.bindings = OrderedDict()
        for index in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(index)
            dtype = trt.nptype(self.engine.get_binding_dtype(index))
            shape = tuple(self.context.get_binding_shape(index))
            data = torch.from_numpy(
                np.empty(shape, dtype=np.dtype(dtype))
            ).to(self.device)
            self.bindings[name] = Binding(
                name, 
                dtype, 
                shape, 
                data, 
                int(data.data_ptr())
            )
            self.binding_addrs = OrderedDict(
                (n, d.ptr) for n, d in self.bindings.items()
            )
        
    def infer_async(self, 
                    frame, 
                    device=None):
        # Load frame to device.
        self.binding_addrs["images"] = int(frame.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))

        # Get inference results.
        num_dets = self.bindings["num_dets"].data
        det_boxes = self.bindings["det_boxes"].data
        det_boxes = det_boxes[0, :num_dets[0][0]]
        det_scores = self.bindings["det_scores"].data
        det_scores = det_scores[0, :num_dets[0][0]]
        det_classes = self.bindings["det_classes"].data
        det_classes = det_classes[0, :num_dets[0][0]]

        return det_boxes, det_scores, det_classes

    def infer(self, frame, device=None):
        return self.infer_async(frame=frame, device=device)
