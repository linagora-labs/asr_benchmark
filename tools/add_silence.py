import os
import random
import argparse
from pydub import AudioSegment
from pydub.generators import WhiteNoise

def add_silence(file_path, output_dir=None, number_of_silence=3, silence_duration=20000):
    if file_path.endswith(".wav") or file_path.endswith(".mp3") or file_path.endswith(".flac"):
        if file_path.endswith(".wav"):
            sound = AudioSegment.from_wav(file_path)
        elif file_path.endswith(".mp3"):
            sound = AudioSegment.from_mp3(file_path)
        elif file_path.endswith(".flac"):
            sound = AudioSegment.from_file(file_path)
        for i in range(number_of_silence):
            silence = WhiteNoise(sound.frame_rate).to_audio_segment(duration=silence_duration*1000, volume=-30)
            # silence = AudioSegment.silent(duration=silence_duration*1000, frame_rate=sound.frame_rate)  #duration in milliseconds
            random_position = random.randint(0, len(sound))
            sound = sound[:random_position] + silence + sound[random_position:]
        file = file_path.split("/")[-1]
        path = os.path.join(*file_path.split("/")[:-1])
        extension = "."+file.split(".")[-1]
        basename = file.split(".")[0]
        if output_dir:
            print(os.path.join(output_dir, file))
            sound.export(os.path.join(output_dir, file), format=extension[1:])
        else:
            print(os.path.join(path, basename+"_silenced.wav"+extension))
            sound.export(os.path.join(path, basename+"_silenced"+extension), format=extension[1:])

if __name__=="__main__":
    # copy all files from input_dir to output_dir
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Input directory")
    parser.add_argument("--output_dir", help="Output directory")
    parser.add_argument("--number_of_silence", default=12, help="Number of silence to add")
    parser.add_argument("--silence_duration", default=5, help="Duration of silence in seconds")
    parser.add_argument("--number_of_files", default=None, help="Number of files to process")
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    if output_dir:
        if os.path.isdir(output_dir):
            print(f"Output directory {output_dir} already exists, delete it")
        os.makedirs(output_dir, exist_ok=False)
    if os.path.isfile(input_dir):
        add_silence(input_dir, output_dir, args.number_of_silence, args.silence_duration)
    else:
        for i, file in enumerate(os.listdir(input_dir)):
            add_silence(os.path.join(input_dir, file), output_dir, args.number_of_silence, args.silence_duration)
            if args.number_of_files and i+1 >= int(args.number_of_files):
                break