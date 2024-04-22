
import logging
import os
import torch
import json
import ssak.utils.vad
import nemo.collections.asr as nemo_asr
logging.getLogger('nemo_logging').setLevel(logging.ERROR)
from asr_benchmark.utils.benchmark import load_audio
from asr_benchmark.benchmark.interfaces import Model

class NemoModel(Model):
    
    def __init__(self, config) -> None:
        super().__init__(config)
        model_type = nemo_asr.models.EncDecCTCModelBPE
        if "hybrid" in self.config['model']:
             model_type = nemo_asr.models.EncDecHybridRNNTCTCBPEModel
        elif "rnnt" in self.config['model']:
            model_type = nemo_asr.models.EncDecRNNTModel
        elif "canary" in self.config['model']:
            model_type = nemo_asr.models.EncDecMultiTaskModel
        self.model_type = model_type


    def load(self) -> None:
        logging.getLogger("nemo_logger").setLevel(logging.ERROR)
        model_type = nemo_asr.models.EncDecCTCModelBPE
        if self.config['model'].endswith(".nemo"):
            self.model = model_type.restore_from(self.config['model'], map_location=self.config['device'])
        else:
            self.model = model_type.from_pretrained(model_name=self.config['model'], map_location=self.config['device'])
        if self.model_type == nemo_asr.models.EncDecHybridRNNTCTCBPEModel:
            self.model.change_decoding_strategy(decoder_type=self.config['decoder'])
        elif self.model_type == nemo_asr.models.EncDecMultiTaskModel:
            decode_cfg = self.model.cfg.decoding
            decode_cfg.beam.beam_size = 1
            self.model.change_decoding_strategy(decode_cfg)
    
    def load_audio(self, audio: str, start=0.0, duration=None):
        return load_audio(audio, start=start, duration=duration)
    
    def transcribe(self, audio: str) -> str:
        if self.config['vad'] and self.config['vad'] in ['auditok','silero', 'pyannote']:
            audio, _ = ssak.utils.vad.remove_non_speech(audio, method=self.config['vad'])
        output = dict()
        if isinstance(self.model, nemo_asr.models.EncDecMultiTaskModel):
            predicted_text = self.model.transcribe(
                audio,
                duration=None,
                task="asr",
                source_lang="fr",
                target_lang= "fr",
                pnc="no",
                answer="na",
                verbose=False
            )
            return predicted_text[0]
        elif isinstance(self.model, nemo_asr.models.EncDecHybridRNNTCTCBPEModel):
            result = self.model.transcribe(audio, verbose=False)
            output['text'] = result[0][0].text
            return output
        result = self.model.transcribe(audio, verbose=False)
        output['text'] = result[0].text
        return output

    def transcribe_batch(self, data: str) -> str:
        with open("tmp.jsonl", "w", encoding="utf-8") as f:
            for i in data:
                f.write(json.dumps(i, ensure_ascii=False)+"\n")
        import nemo.collections.asr as nemo_asr
        batch_size = int(self.config.get('batch_size', 16))
        if isinstance(self.model, nemo_asr.models.EncDecMultiTaskModel):
            predicted_text = self.model.transcribe(
                "tmp.jsonl",
                duration=None,
                task="asr",
                source_lang="fr",
                target_lang= "fr",
                pnc="no",
                answer="na",
                batch_size=batch_size,  # batch size to run the inference with
                num_workers=4
            )
            return predicted_text
        elif isinstance(self.model, nemo_asr.models.EncDecHybridRNNTCTCBPEModel):
            result_full = self.model.transcribe("tmp.jsonl", batch_size=batch_size, num_workers=4)
            return result_full[0]
        result = self.model.transcribe("tmp.jsonl", batch_size=batch_size, num_workers=4)
        os.remove("tmp.jsonl")
        return result

    def can_output_word_timestamps(self):
        return True
    
    def cleanup(self):
        torch.cuda.empty_cache()

    def add_defaults_to_config(self, config):
        config['vad'] = config.get('vad', 'false')
        config['device'] = config.get('device', 'cuda')
        config['decoder'] = config.get('decoder', 'ctc')
        return super().add_defaults_to_config(config)

    def get_metadata(self):
        metadata = super().get_metadata()
        if self.config['model'].endswith(".nemo"):
            metadata['model'] = self.config['model'].split("/")[-1].replace(".nemo", "").replace("_","-")
        else:
            metadata['model'] = self.config['model'].replace("_","-")
        if "batch_size" in metadata:
            del metadata['batch_size']
        return metadata
    
    def get_folder_name(self):
        tot_config = self.config.copy()
        if tot_config['model'].endswith(".nemo"):
            model = tot_config['model'].split("/")[-1].replace(".nemo", "").replace("_","-")
        else:
            model = tot_config['model'].replace(".nemo", "").replace("_","-").replace("/","-")
        name = f"nemo_{model}_device-{tot_config['device']}"
        if self.model_type == nemo_asr.models.EncDecHybridRNNTCTCBPEModel:
            name += f"_decoder-{tot_config['decoder']}"
        elif self.model_type == nemo_asr.models.EncDecRNNTModel:
            name += f"_decoder-rnnt"
        elif self.model_type == nemo_asr.models.EncDecCTCModelBPE:
            name += f"_decoder-ctc"
        if tot_config['compute_rtf']:
            name += f"_vad-{tot_config['vad']}"
        else:
            name += f"_vad-false"
        name = name.replace("/", "-")
        name += "_rtf" if tot_config['compute_rtf'] else ""
        return name