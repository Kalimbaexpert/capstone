import tensorrt as trt
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
    config = builder.create_builder_config()
    config.max_workspace_size = (1 << 30 )
    builder.max_batch_size = 1 
    config.set_flag(trt.BuilderFlag.FP16)
    # builder.fp16_mode = True 
    # if builder.platform_has_fast_fp16 :
    #     builder.fp16_mode = True
    with open('models/human-pose-estimation-3d.onnx', 'rb') as model: 
        parser.parse(model.read()) 
    engine = builder.build_engine(network) 
    with open('models/human-pose-estimation-3d.trt', "wb") as f: 
        f.write(engine.serialize())