import asyncio
import json
import time
import torchaudio
import librosa
import os
import subprocess
import ssak.utils.audio


def load_audio(fname, return_format="librosa", start=0.0, duration=None):
    end = start + duration if duration else None
    waveform = ssak.utils.audio.load_audio(fname, start=start, end=end, sample_rate=16000, mono=True, return_format="torch" if return_format=="file" else return_format)
    if return_format == "file" or return_format=="torch":
        waveform = waveform.unsqueeze(0)
    if return_format == "file":
        torchaudio.save("tmp.wav", waveform, sample_rate=16000)
        return "tmp.wav"
    else:
        return waveform

def get_audio_duration(file_path):
    y, sr = librosa.load(file_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    return duration

def get_data(input_file):
    all_data = []
    with open(input_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("/") or line.startswith("#"):
                continue
            row = json.loads(line)
            basename = os.path.basename(row['audio_filepath']).split('.')[0]
            if 'id' not in row:
                if 'name' not in row:
                    row['id'] = basename
                    row['name'] = row.get('dataset', 'unknown')
                else:
                    row['id'] = f"{row['name']}_{basename}"
                if float(row.get("offset", 0.0)) > 0.0:
                    row['id'] += f"_{row['offset']}"
            all_data.append(row) 
    return all_data


def merge_timestamps(sentence_a, sentence_b):
    from collections import Counter
    # Convert words to lists while preserving order
    words_a = [word['word'] for word in sentence_a]
    words_b = [word['word'] for word in sentence_b]

    # Count occurrences of words
    count_a = Counter(words_a)
    count_b = Counter(words_b)

    # Get words that appear in both sentences
    common_words = []
    seen = Counter()
    for word in words_a:
        if word in count_b and seen[word] < min(count_a[word], count_b[word]):
            common_words.append(word)
            seen[word] += 1


    # Filter sentences to keep only words in common
    filtered_a = [word for word in sentence_a if word['word'] in common_words]
    filtered_b = [word for word in sentence_b if word['word'] in common_words]

    # Merge timestamps and emission times, aligning them correctly
    result = []
    for word_a, word_b in zip(filtered_a, filtered_b):
        if word_a['word'] == word_b['word']:
            result.append({**word_a, **word_b})
    return result

async def send_data(websocket, stream, logger, stream_config):
    """Asynchronously load and send data to the WebSocket."""
    duration = 0
    try:
        while True:
            data = stream.read(int(stream_config['stream_duration'] * 2 * 16000))
            duration += stream_config['stream_duration']
            if stream_config['audio_file'] and not data:
                logger.debug("Audio file finished")
                break

            if stream_config['vad']:
                import auditok
                audio_events = auditok.split(
                    data,
                    min_dur=0.2,
                    max_silence=0.3,
                    energy_threshold=65,
                    sampling_rate=16000,
                    sample_width=2,
                    channels=1
                )
                audio_events = list(audio_events)
                if len(audio_events) == 0:
                    logger.debug(f"Full silence for chunk: {duration - stream_config['stream_duration']:.1f}s --> {duration:.1f}s")
                    if stream_config['stream_wait']>0:
                        await asyncio.sleep(stream_config['stream_wait'])
                    continue
            await websocket.send(data)
            logger.debug(f"Sent audio chunk: {duration - stream_config['stream_duration']:.1f}s --> {duration:.1f}s")
            if stream_config['stream_wait']>0:
                await asyncio.sleep(stream_config['stream_wait'])

    except asyncio.CancelledError:  # handle server errors and ctrl+c...
        logger.debug("Data sending task cancelled.")
    except Exception as e:          # handle data loading errors
        logger.error(f"Error in data sending: {e}")
    logger.debug(f"Waiting before sending EOF")
    await asyncio.sleep(5)
    logger.debug(f"Sending EOF")
    await websocket.send('{"eof" : 1}')

async def _linstt_streaming(
    audio_file,
    ws_api = "ws://localhost:8080/streaming",
    language = None,
    apply_vad = False,
    stream_duration = 0.5,
    stream_wait = 0.5,
    compute_latency = False,
):
    import websockets
    import logging
    logger = logging.getLogger(__name__)
    stream_config = {"language": language, "sample_rate": 16000, "vad": apply_vad, "stream_duration": stream_duration, "stream_wait": stream_wait}
    subprocess.run(["ffmpeg", "-y", "-i", audio_file, "-acodec", "pcm_s16le", "-ar", str(stream_config['sample_rate']), "-ac", "1", "tmp.wav"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stream = open("tmp.wav", "rb")
    stream_config["audio_file"] = audio_file
    text = ""
    partial = None
    duration = 0
    latencies = []
    async with websockets.connect(ws_api, ping_interval=None, ping_timeout=None) as websocket:
        if language is not None:
            config = {"config" : {"sample_rate": stream_config['sample_rate'], "language": stream_config['language']}}
        else: 
            config = {"config" : {"sample_rate": stream_config['sample_rate']}}
        await websocket.send(json.dumps(config))
        send_task = asyncio.create_task(send_data(websocket, stream, logger, stream_config))
        try:
            partial_latencies = []
            stream_start = time.time()
            while True:
                res = await websocket.recv()
                current_time = time.time() - stream_start
                message = json.loads(res)
                if message is None:
                    logger.debug("\n Received None")
                    continue
                if "text" in message.keys():
                    latencies.extend(partial_latencies)
                    partial_latencies = []
                    sentence = message["text"]
                    logger.debug(f'Final (after {duration:.1f}s): "{sentence}"')
                    if text:
                        text += "\n"
                    for w in sentence:
                        if w[0] is not None:
                            text += w[2]
                elif "partial" in message.keys():
                    partial = message["partial"]
                    logger.debug(f'Partial (after {duration:.1f}s): "{partial}"')
                    for i, p in enumerate(partial):
                        word = p[2].strip()
                        if i<len(partial_latencies) and word.lower()!=partial_latencies[i]["word"]:
                            partial_latencies[i]["partial"] = round(current_time,3)
                            partial_latencies[i]["start"] = round(p[0],3)
                            partial_latencies[i]["end"] = round(p[1],3)
                            partial_latencies[i]["word"] = word.lower()
                        elif i>=len(partial_latencies):
                            partial_latencies.append({"word": word.lower(), "start": round(p[0],3), "end": round(p[1],3), "partial": round(current_time,3)})
        except asyncio.CancelledError:  # handle ctrl+c...
            logger.debug("Message processing thread stopped as websocket was closed.")
        except websockets.exceptions.ConnectionClosedOK:
            logger.debug("Websocket closed")
        except websockets.exceptions.ConnectionClosedError as e:
            logger.error(f"Websocket closed with error: {e}")
        except Exception as e:
            raise Exception(f"Error in message processing {message} {type(message)}: {e}")
        finally:
            await websocket.close()
    latencies = sorted(latencies, key=lambda x: x["start"])
    return text, latencies

def linstt_streaming(*kargs, **kwargs):
    text, latencies = asyncio.run(_linstt_streaming(*kargs, **kwargs))
    return text, latencies