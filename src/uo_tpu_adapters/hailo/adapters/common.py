# Lint as: python3
# Copyright 2024 Mahomed2904
from typing import List

from PIL.Image import Image

from uo_tpu_adapters.hailo.utils import hailotpu


def input_details(interpreter: hailotpu.Interpreter):
    """Gets a model's input details by specified key.

  Args:
    interpreter: The ``Interpreter`` holding the model.
    key (int): The index position of an input tensor.
  Returns:
    The input details.
  """
    input_details_ = interpreter.get_input_details()
    print("Input size:", input_details_)
    return input_details_[0]


def input_size(interpreter: hailotpu.Interpreter):
    """Gets a model's input size as (width, height) tuple.

  Args:
    interpreter: The ``Interpreter`` holding the model.
  Returns:
    The input tensor size as (width, height) tuple.
  """
    height, width, _ = input_details(interpreter).shape
    return width, height


def set_input(interpreter: hailotpu.Interpreter, data: List[Image]):
    """Copies data to a model's input tensor.

  Args:
    interpreter: The ``hailo.Interpreter`` to update.
    data: A list of images .
  """
    interpreter.set_input(data)


def set_resized_input(interpreter: hailotpu.Interpreter, batch_size: int, resize):
    """Copies a resized and properly zero-padded image to a model's input tensor.

  Args:
    interpreter: The ``libs.hailo.core.interpreter.Interpreter`` to update.
    batch_size (int): The original image size as (width, height) tuple.
    resize: A function that takes a (width, height) tuple, and returns an
      image resized to those dimensions.

  Returns:
    The resized tensor with zero-padding as tuple
    (resized_tensor, resize_ratio).
  """
    width, height = input_size(interpreter)
    result = []
    scales = []
    for i in range(batch_size):
        resized_image, scale = resize(i, (width, height))
        result.append(resized_image)
        scales.append(scale)
    interpreter.set_input(result)
    return result, scales
