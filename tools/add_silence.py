import random
import argparse
from pathlib import Path
from pydub import AudioSegment
from pydub.generators import WhiteNoise

def add_silence(file_path, output_dir=None, number_of_silence=3, silence_duration=20000):
    fp = Path(file_path)
    if fp.suffix in (".wav", ".mp3", ".flac"):
        if fp.suffix == ".wav":
            sound = AudioSegment.from_wav(file_path)
        elif fp.suffix == ".mp3":
            sound = AudioSegment.from_mp3(file_path)
        elif fp.suffix == ".flac":
            sound = AudioSegment.from_file(file_path)
        for i in range(number_of_silence):
            silence = WhiteNoise(sound.frame_rate).to_audio_segment(duration=silence_duration*1000, volume=-30)
            # silence = AudioSegment.silent(duration=silence_duration*1000, frame_rate=sound.frame_rate)  #duration in milliseconds
            random_position = random.randint(0, len(sound))
            sound = sound[:random_position] + silence + sound[random_position:]
        parent = fp.parent
        extension = fp.suffix
        basename = fp.stem
        if output_dir:
            out = Path(output_dir) / fp.name
            print(out)
            sound.export(str(out), format=extension[1:])
        else:
            out = parent / (basename + "_silenced" + extension)
            print(out)
            sound.export(str(out), format=extension[1:])

if __name__=="__main__":
    # copy all files from input_dir to output_dir
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Input directory")
    parser.add_argument("--output_dir", help="Output directory")
    parser.add_argument("--number_of_silence", default=12, help="Number of silence to add")
    parser.add_argument("--silence_duration", default=5, help="Duration of silence in seconds")
    parser.add_argument("--number_of_files", default=None, help="Number of files to process")
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        if output_dir.is_dir():
            print(f"Output directory {output_dir} already exists, delete it")
        output_dir.mkdir(parents=True, exist_ok=False)
    if input_dir.is_file():
        add_silence(str(input_dir), str(output_dir) if output_dir else None, args.number_of_silence, args.silence_duration)
    else:
        for i, file in enumerate(input_dir.iterdir()):
            add_silence(str(file), str(output_dir) if output_dir else None, args.number_of_silence, args.silence_duration)
            if args.number_of_files and i+1 >= int(args.number_of_files):
                break
