from functools import partial
from typing import Optional, Dict, Tuple, List

import numpy as np

try:
    from hailo_platform import (VDevice, HEF, FormatType, HailoSchedulingAlgorithm)
except Exception as e:
    print(f"Error importing hailo_platform modules: {e}")


class _SyncCompletionInfo:
    """Minimal stand-in matching the completion_info duck-type invoke_async's real
    Hailo-SDK-provided callback argument uses (just an `.exception` attribute), so
    invoke()'s synchronous path can reuse the same __organize_output code path."""

    def __init__(self, exception):
        self._exception = exception

    @property
    def exception(self):
        return self._exception


class Interpreter:
    __target = None

    @staticmethod
    def __get_target():
        if Interpreter.__target is None:
            params = VDevice.create_params()
            # Set the scheduling algorithm to round-robin to activate the scheduler
            params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
            Interpreter.__target = VDevice(params)
        return Interpreter.__target

    def __init__(self, model_path, batch_size=1, target=None,
                 input_type: Optional[str] = None, output_type: Optional[Dict[str, str]] = None):
        self.__model_path = model_path
        self.__configured_infer_model = None
        self.__id_generator = 0
        self.__bindings_list = []
        self.__outputs_datas = {}
        if target is not None:
            self.__target = target
        if input_type is not None:
            self._set_input_type(input_type)
        if output_type is not None:
            self._set_output_type(input_type)
        self.output_type = output_type
        self.hef = HEF(model_path)
        self.__infer_model = self.__get_target().create_infer_model(model_path)
        self.__infer_model.set_batch_size(batch_size)

    def _set_input_type(self, input_type: Optional[str] = None) -> None:
        """
        Set the input type for the HEF model. If the model has multiple inputs,
        it will set the same type of all of them.

        Args:
            input_type (Optional[str]): Format type of the input stream.
        """
        self.__infer_model.input().set_format_type(getattr(FormatType, input_type))

    def _set_output_type(self, output_type_dict: Optional[Dict[str, str]] = None) -> None:
        """
        Set the output type for the HEF model. If the model has multiple outputs,
        it will set the same type for all of them.

        Args:
            output_type_dict (Optional[dict[str, str]]): Format type of the output stream.
        """
        for output_name, output_type in output_type_dict.items():
            self.__infer_model.output(output_name).set_format_type(
                getattr(FormatType, output_type)
            )

    def get_vstream_info(self) -> Tuple[list, list]:

        """
        Get information about input and output stream layers.

        Returns:
            Tuple[list, list]: List of input stream layer information, List of
                               output stream layer information.
        """
        return (
            self.hef.get_input_vstream_infos(),
            self.hef.get_output_vstream_infos()
        )

    def get_input_details(self):
        """
        Get the input details of the model.

        Returns:
            The input details of the model.
        """
        return self.hef.get_input_vstream_infos() # Assumes one input

    def _get_output_type_str(self, output_info) -> str:
        if self.output_type is None:
            return str(output_info.format.type).split(".")[1].lower()
        else:
            self.output_type[output_info.name].lower()

    def configure_model(self):
        self.__configured_infer_model = self.__infer_model.configure()

    def shutdown_configured_model(self):
        if self.__is_not_configured_model():
            return
        self.__configured_infer_model = None

    def __is_not_configured_model(self):
        return self.__configured_infer_model is None

    def _create_bindings(self, configured_infer_model) -> object:
        """
        Create bindings for input and output buffers.

        Args:
            configured_infer_model: The configured inference model.

        Returns:
            object: Bindings object with input and output buffers.
        """
        if self.output_type is None:
            output_buffers = {
                output_info.name: np.empty(
                    self.__infer_model.output(output_info.name).shape,
                    dtype=(getattr(np, self._get_output_type_str(output_info)))
                )
                for output_info in self.hef.get_output_vstream_infos()
            }
        else:
            output_buffers = {
                name: np.empty(
                    self.__infer_model.output(name).shape,
                    dtype=(getattr(np, self.output_type[name].lower()))
                )
                for name in self.output_type
            }
        return configured_infer_model.create_bindings(
            output_buffers=output_buffers
        )

    def set_input(self, images_batch):
        bindings_list = []
        for frame in images_batch:
            bindings = self._create_bindings(self.__configured_infer_model)
            bindings.input().set_buffer(np.array(frame))
            bindings_list.append(bindings)
        self.__bindings_list = bindings_list

    def get_output(self, job_id=-1) -> List:
        if job_id < 0:
            job_id = self.__get_last_id()
        results = self.__outputs_datas[job_id]
        del self.__outputs_datas[job_id]
        return results

    def __get_last_id(self):
        return self.__id_generator - 1

    def __save_results(self, results):
        current_id = self.__id_generator
        self.__outputs_datas[current_id] = results
        self.__id_generator += 1
        return current_id

    def __organize_output(self, completion_info, callback=None):
        job_id = -1
        if completion_info.exception is not None:
            results = []
            for i, bindings in enumerate(self.__bindings_list):
                # If the model has a single output, return the output buffer.
                # Else, return a dictionary of output buffers, where the keys are the output names.
                if len(bindings._output_names) == 1:
                    result = bindings.output().get_buffer()
                else:
                    result = {
                        name: np.expand_dims(
                            bindings.output(name).get_buffer(), axis=0
                        )
                        for name in bindings._output_names
                    }
                results.append(result)
            job_id = self.__save_results(results)
        if callback is not None:
            callback(completion_info=completion_info, job_id=job_id)

    def invoke(self):
        self.__configured_infer_model.run(self.__bindings_list, 10000)
        return self.__organize_output(_SyncCompletionInfo(None))

    def invoke_async(self, callback):
        self.__configured_infer_model.wait_for_async_ready(timeout_ms=10000)
        job = self.__configured_infer_model.run_async(
            self.__bindings_list,
            partial(
                self.__organize_output,
                callback=callback
            )
        )
        job.wait(10000)  # Wait for the last job


