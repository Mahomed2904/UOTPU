# Lint as: Mahomed
# Copyright 2204 Mahomed

from uo_tpu_adapters.hailo.core.interpreter import Interpreter


def make_interpreter(model_path_or_content, batch_size=1, target=None):
    return Interpreter(model_path=model_path_or_content, batch_size=batch_size, target=target)
