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
    elif backend == "transformers-voxtral-realtime":
        import asr_benchmark.benchmark.interfaces_transformers as interfaces_transformers
        model = interfaces_transformers.TransformersVoxtralRealtimeModel(config)
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
    elif backend in ("linto-stt", "linto-stt-whisper"):
        import asr_benchmark.benchmark.interfaces_lintostt as interfaces_lintostt
        model = interfaces_lintostt.LintoSttWhisperModel(config)
    elif backend == "linto-stt-nemo":
        import asr_benchmark.benchmark.interfaces_lintostt as interfaces_lintostt
        model = interfaces_lintostt.LintoSttNemoModel(config)
    elif backend == "nemo":
        import asr_benchmark.benchmark.interfaces_nemo as interfaces_nemo
        model = interfaces_nemo.NemoModel(config)
    elif backend == "moss":
        import asr_benchmark.benchmark.interfaces_moss as interfaces_moss
        model = interfaces_moss.MossTranscribeDiarizeModel(config)
    elif backend == "transformers-voxtral":
        import asr_benchmark.benchmark.interfaces_voxtral as interfaces_voxtral
        model = interfaces_voxtral.TransformersVoxtralModel(config)
    elif backend == "vllm":
        import asr_benchmark.benchmark.interfaces_vllm as interfaces_vllm
        model = interfaces_vllm.VllmTranscriptionModel(config)
    elif backend == "gemma3n":
        import asr_benchmark.benchmark.interfaces_gemma3n as interfaces_gemma3n
        model = interfaces_gemma3n.Gemma3nModel(config)
    elif backend == "qwen3-omni":
        import asr_benchmark.benchmark.interfaces_qwen3omni as interfaces_qwen3omni
        model = interfaces_qwen3omni.Qwen3OmniModel(config)
    else:
        raise ValueError(f"Invalid backend: {backend}")
    return model