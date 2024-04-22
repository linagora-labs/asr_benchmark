import asr_benchmark.benchmark.interfaces as interfaces

def get_model(config):
    backend = config["backend"]
    if backend == "transformers":
        import asr_benchmark.benchmark.interfaces_transformers as interfaces_transformers
        model = interfaces_transformers.TransformersModel(config)
    elif backend == "transformers-facebook":
        import asr_benchmark.benchmark.interfaces_transformers as interfaces_transformers
        model = interfaces_transformers.TransformersFacebookModel(config)
    elif backend == "transformers-bofenghuang":
        import asr_benchmark.benchmark.interfaces_transformers as interfaces_transformers
        model = interfaces_transformers.TransformersBofenghuangModel(config)
    elif backend == "intel-transformers":
        import asr_benchmark.benchmark.interfaces_transformers as interfaces_transformers
        model = interfaces_transformers.IntelTransformersModel(config)
    elif backend == "faster-whisper":
        import asr_benchmark.benchmark.interfaces_whisper as interfaces_whisper
        model = interfaces_whisper.FasterWhisperModel(config)
    elif backend == "openai":
        import asr_benchmark.benchmark.interfaces_whisper as interfaces_whisper
        model = interfaces_whisper.OpenAIModel(config)
    elif backend == "http-api":
        model = interfaces.HttpAPIModel(config)
    elif backend == "linto-stt":
        import asr_benchmark.benchmark.interfaces_whisper as interfaces_whisper
        model = interfaces_whisper.LintoSttModel(config)
    elif backend == "nemo":
        import asr_benchmark.benchmark.interfaces_nemo as interfaces_nemo
        model = interfaces_nemo.NemoModel(config)
    else:
        raise ValueError(f"Invalid backend: {backend}")
    return model