import argparse
from pydub import AudioSegment
from tqdm import tqdm



if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Concatenate multiple audios files into one')
    parser.add_argument('input_files', help="Input manifest" , nargs='+', type=str)
    parser.add_argument('output_file', type=str, default="merged.wav")
    args = parser.parse_args()
    
    sound = None
    for audio in tqdm(args.input_files, desc="Loading files"):
        if sound is None:
            sound = AudioSegment.from_file(audio)
        else:
            wav = AudioSegment.from_file(audio)
            sound = sound + wav


    # simple export
    output_file = sound.export(args.output_file, format="wav")