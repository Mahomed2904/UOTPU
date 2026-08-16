# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions to work with a detection model."""

from pycoral.adapters import common

from uo_tpu_adapters.common import Object, BBox


def get_objects(interpreter,
                score_threshold=-float('inf'),
                image_scale=(1.0, 1.0)):
  """Gets results from a detection model as a list of detected objects.

  Args:
    interpreter: The ``tf.lite.Interpreter`` to query for results.
    score_threshold (float): The score threshold for results. All returned
      results have a score greater-than-or-equal-to this value.
    image_scale (float, float): Scaling factor to apply to the bounding boxes as
      (x-scale-factor, y-scale-factor), where each factor is from 0 to 1.0.

  Returns:
    A list of :obj:`Object` objects, which each contains the detected object's
    id, score, and bounding box as :obj:`BBox`.
  """
  # If a model has signature, we use the signature output tensor names to parse
  # the results. Otherwise, we parse the results based on some assumption of the
  # output tensor order and size.
  # pylint: disable=protected-access
  signature_list = interpreter._get_full_signature_list()
  # pylint: enable=protected-access
  if signature_list:
    if len(signature_list) > 1:
      raise ValueError('Only support model with one signature.')
    signature = signature_list[next(iter(signature_list))]
    count = int(interpreter.tensor(signature['outputs']['output_0'])()[0])
    scores = interpreter.tensor(signature['outputs']['output_1'])()[0]
    class_ids = interpreter.tensor(signature['outputs']['output_2'])()[0]
    boxes = interpreter.tensor(signature['outputs']['output_3'])()[0]
  elif common.output_tensor(interpreter, 3).size == 1:
    boxes = common.output_tensor(interpreter, 0)[0]
    class_ids = common.output_tensor(interpreter, 1)[0]
    scores = common.output_tensor(interpreter, 2)[0]
    count = int(common.output_tensor(interpreter, 3)[0])
  else:
    scores = common.output_tensor(interpreter, 0)[0]
    boxes = common.output_tensor(interpreter, 1)[0]
    count = (int)(common.output_tensor(interpreter, 2)[0])
    class_ids = common.output_tensor(interpreter, 3)[0]

  width, height = common.input_size(interpreter)
  image_scale_x, image_scale_y = image_scale
  sx, sy = width / image_scale_x, height / image_scale_y

  def make(i):
    ymin, xmin, ymax, xmax = boxes[i]
    return Object(
        id=int(class_ids[i]),
        score=float(scores[i]),
        bbox=BBox(xmin=xmin, ymin=ymin, xmax=xmax,
                  ymax=ymax).scale(sx, sy).map(int))

  return [make(i) for i in range(len(scores)) if scores[i] >= score_threshold]
