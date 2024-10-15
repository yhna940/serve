import re
from typing import Any

import numpy as np
import onnxruntime as ort
import torch


class ORTInferenceSessionWrapper:
    """Callable class to wrap ONNXRuntime InferenceSession object

    Attributes:
        _session (ort.InferenceSession): ONNXRuntime InferenceSession object
        _io_binding (ort.IOBinding): ONNXRuntime IOBinding object
        _input_metas (dict[str, ort.NodeArg]): Dictionary of input metadata
        _input_names (list[str]): List of input names
        _output_names (list[str]): List of output names
    """

    def __init__(self, session: ort.InferenceSession):
        """Initialize the wrapper with the ONNXRuntime InferenceSession object

        Args:
            session (ort.InferenceSession): ONNXRuntime InferenceSession object
        """
        self._session: ort.InferenceSession = session
        self._io_binding: ort.IOBinding = self._session.io_binding()
        self._input_metas: dict[str, ort.NodeArg] = {
            input.name: input for input in self._session.get_inputs()
        }
        self._input_names: list[str] = list(self._input_metas.keys())
        self._output_names: list[str] = self._session.get_outputs()

    def __getattr__(self, attr: str) -> Any:
        """Bind the attribute to the ONNXRuntime InferenceSession object

        Args:
            attr (str): Attribute name

        Returns:
            Any: Attribute of the ONNXRuntime InferenceSession object
        """
        return getattr(self._session, attr)

    def _parse_cuda_device_id(self, device: str) -> int:
        """Parse the CUDA device ID from the device string

        Args:
            device (str): Device string in the format "cuda" or "cuda:<device_id>"

        Returns:
            int: CUDA device ID
        """
        match_result = re.match("([^:]+)(:[0-9]+)?$", device)
        if match_result is None:
            raise ValueError("Cannot parse device string: " + device)
        if match_result.group(1).lower() != "cuda":
            raise ValueError("Not a cuda device: " + device)
        return 0 if match_result.lastindex == 1 else int(match_result.group(2)[1:])

    def __call__(
        self, data: torch.Tensor | dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Run the inference session with the given input data

        Args:
            data (torch.Tensor | dict[str, torch.Tensor]): Input data for inference
            *args: Additional arguments
            **kwargs: Additional keyword arguments

        Returns:
            torch.Tensor | dict[str, torch.Tensor]: Inference output

        Raises:
            ValueError: Raises ValueError if the input data is a torch.Tensor and the model has multiple inputs
        """
        if isinstance(data, torch.Tensor):
            if len(self._input_names) > 1:
                raise ValueError(
                    "Model has multiple inputs, please provide a dictionary of inputs"
                )
            data = {self._input_names[0].name: data}
        for name, tensor in data.items():
            input_type: str = self._input_metas[name].type
            if "float16" in input_type:
                tensor = tensor.to(torch.float16)
            tensor = tensor.contiguous()
            self._io_binding.bind_input(
                name=name,
                device_type=tensor.device.type,
                device_id=-1
                if tensor.device.type == "cpu"
                else self._parse_cuda_device_id(tensor.device.index),
                element_type=tensor.new_zeros(1, device="cpu").numpy().dtype,
                bufer_ptr=tensor.data_ptr(),
            )
        for name in self._output_names:
            self._io_binding.bind_output(name)

        if "CUDAExecutionProvider" in self._session.get_providers():
            torch.cuda.synchronize()

        # Run the inference
        self._session.run_with_iobinding(self._io_binding)

        outputs: list[torch.Tensor] = list(
            map(
                lambda tensor: torch.from_numpy(
                    tensor if tensor.dtype != np.float16 else tensor.astype(np.float32)
                ),
                self._io_binding.copy_outputs_to_cpu(),
            )
        )
        if len(outputs) == 1:
            return outputs[0]
        else:
            return {name: tensor for name, tensor in zip(self._output_names, outputs)}
