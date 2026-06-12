import os
try:
    import paddle.inference as paddle_infer
except ImportError:
    import paddle.fluid as fluid
    paddle_infer = fluid.core

def load_model(params_file, model_file, use_gpu=False, use_mkl=False, mkl_thread_num=4):
    try:
        config = paddle_infer.Config(model_file, params_file)
    except AttributeError:
        config = paddle_infer.AnalysisConfig(model_file, params_file)

    if use_gpu:
        # Set GPU memory (MB) and Device ID
        config.enable_use_gpu(100, 0)
    else:
        config.disable_gpu()
    if use_mkl and not use_gpu:
        config.enable_mkldnn()
        config.set_cpu_math_library_num_threads(mkl_thread_num)
    try:
        config.disable_glog_info()
    except AttributeError:
        pass
    try:
        config.enable_memory_optim()
    except AttributeError:
        pass

    # Enable IR optimization passes such as OP fusion
    try:
        config.switch_ir_optim(True)
    except AttributeError:
        pass
    # Disable feed/fetch OP; must be set when using ZeroCopy interface
    try:
        config.switch_use_feed_fetch_ops(False)
    except AttributeError:
        pass
    predictor = paddle_infer.create_predictor(config)
    return predictor

predictor = load_model('models/paddle/__params__', 'models/paddle/__model__', use_gpu=False)

import numpy as np
data = np.random.rand(1, 3, 260, 260).astype('float32')

input_names = predictor.get_input_names()
try:
    input_tensor = predictor.get_input_handle(input_names[0])
except AttributeError:
    input_tensor = predictor.get_input_tensor(input_names[0])

input_tensor.copy_from_cpu(data)

try:
    predictor.run()
except AttributeError:
    predictor.zero_copy_run()

output_names = predictor.get_output_names()
try:
    output_tensor = predictor.get_output_handle(output_names[0])
except AttributeError:
    output_tensor = predictor.get_output_tensor(output_names[0])

result = output_tensor.copy_to_cpu()

print(result.shape)