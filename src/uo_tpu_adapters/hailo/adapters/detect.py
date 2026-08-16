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

from uo_tpu_adapters.common import Object, BBox
import uo_tpu_adapters.common as common

from uo_tpu_adapters.hailo.core.interpreter import Interpreter


def __make(i, boxes, classes, scores, sx, sy):
    ymin, xmin, ymax, xmax = boxes[i]
    return Object(
        id=int(classes[i]),
        score=float(scores[i]),
        bbox=BBox(xmin=xmin, ymin=ymin, xmax=xmax,
                  ymax=ymax).scale(sx, sy).map(int))


def get_objects(interpreter: Interpreter,
                job_id=-1,
                score_threshold=-float('inf'),
                images_scale=(1.0, 1.0)):
    """Gets results from a detection model as a list of detected objects.

  Args:
    job_id: Job ID of the output
    interpreter: The ``libs.hailo.core.interpreter.Interpreter`` to query for results.
    score_threshold (float): The score threshold for results. All returned
      results have a score greater-than-or-equal-to this value.
    images_scale (list[tuple]): Scaling factor to apply to the bounding boxes as
      (x-scale-factor, y-scale-factor), where each factor is from 0 to 1.0.

  Returns:
    A list of :obj:`Object` objects, which each contains the detected object's
    id, score, and bounding box as :obj:`BBox`.
  """
    results = []
    for j, image_results in enumerate(interpreter.get_output(job_id)):
        boxes, scores, classes = [], [], []
        num_detections = 0
        for i, detection in enumerate(image_results):
            if len(detection) == 0:
                continue

            for det in detection:
                bbox, score = det[:4], det[4]

                if score >= score_threshold:
                    boxes.append(bbox)
                    scores.append(score)
                    classes.append(i)
                    num_detections += 1

        width, height = common.input_size(interpreter)
        image_scale_x, image_scale_y = images_scale[j]
        sx, sy = width / image_scale_x, height / image_scale_y
        results.append([__make(i, boxes, classes, scores, sx, sy)
                        for i in range(len(scores)) if scores[i] >= score_threshold])

    return results
