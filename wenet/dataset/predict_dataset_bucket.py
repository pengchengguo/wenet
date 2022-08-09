import argparse
import codecs
import copy
import logging
import random
import math
import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import torchaudio.sox_effects as sox_effects
import yaml
from PIL import Image
from PIL.Image import BICUBIC
from torch.utils.data import Dataset, DataLoader

from wenet.dataset.wav_distortion import distort_wav_conf
from wenet.utils.common import IGNORE_ID

torchaudio.set_audio_backend("sox_io")


def _spec_augmentation(
    x, warp_for_time=False, num_t_mask=2, num_f_mask=2, max_t=50, max_f=10, max_w=80
):
    """Deep copy x and do spec augmentation then return it

    Args:
        x: input feature, T * F 2D
        num_t_mask: number of time mask to apply
        num_f_mask: number of freq mask to apply
        max_t: max width of time mask
        max_f: max width of freq mask
        max_w: max width of time warp

    Returns:
        augmented feature
    """
    y = np.copy(x)
    max_frames = y.shape[0]
    max_freq = y.shape[1]

    # time warp
    if warp_for_time and max_frames > max_w * 2:
        center = random.randrange(max_w, max_frames - max_w)
        warped = random.randrange(center - max_w, center + max_w) + 1

        left = Image.fromarray(x[:center]).resize((max_freq, warped), BICUBIC)
        right = Image.fromarray(x[center:]).resize(
            (max_freq, max_frames - warped), BICUBIC
        )
        y = np.concatenate((left, right), 0)
    # time mask
    for i in range(num_t_mask):
        start = random.randint(0, max_frames - 1)
        length = random.randint(1, max_t)
        end = min(max_frames, start + length)
        y[start:end, :] = 0
    # freq mask
    for i in range(num_f_mask):
        start = random.randint(0, max_freq - 1)
        length = random.randint(1, max_f)
        end = min(max_freq, start + length)
        y[:, start:end] = 0
    return y


def _spec_substitute(x, max_t=20, num_t_sub=3):
    """Deep copy x and do spec substitute then return it

    Args:
        x: input feature, T * F 2D
        max_t: max width of time substitute
        num_t_sub: number of time substitute to apply

    Returns:
        augmented feature
    """
    y = np.copy(x)
    max_frames = y.shape[0]
    for i in range(num_t_sub):
        start = random.randint(0, max_frames - 1)
        length = random.randint(1, max_t)
        end = min(max_frames, start + length)
        # only substitute the earlier time chosen randomly for current time
        pos = random.randint(0, start)
        y[start:end, :] = y[start - pos : end - pos, :]
    return y


def _waveform_distortion(waveform, distortion_methods_conf):
    """Apply distortion on waveform

    This distortion will not change the length of the waveform.

    Args:
        waveform: numpy float tensor, (length,)
        distortion_methods_conf: a list of config for ditortion method.
            a method will be randomly selected by 'method_rate' and
            apply on the waveform.

    Returns:
        distorted waveform.
    """
    r = random.uniform(0, 1)
    acc = 0.0
    for distortion_method in distortion_methods_conf:
        method_rate = distortion_method["method_rate"]
        acc += method_rate
        if r < acc:
            distortion_type = distortion_method["name"]
            distortion_conf = distortion_method["params"]
            point_rate = distortion_method["point_rate"]
            return distort_wav_conf(
                waveform, distortion_type, distortion_conf, point_rate
            )
    return waveform


# add speed perturb when loading wav
# return augmented, sr
def _load_wav_with_speed(wav_file, speed):
    """Load the wave from file and apply speed perpturbation

    Args:
        wav_file: input feature, T * F 2D

    Returns:
        augmented feature
    """
    if speed == 1.0:
        wav, sr = torchaudio.load(wav_file)
    else:
        sample_rate = torchaudio.backend.sox_io_backend.info(wav_file).sample_rate
        # get torchaudio version
        ta_no = torchaudio.__version__.split(".")
        ta_version = 100 * int(ta_no[0]) + 10 * int(ta_no[1])

        if ta_version < 80:
            # Note: deprecated in torchaudio>=0.8.0
            E = sox_effects.SoxEffectsChain()
            E.append_effect_to_chain("speed", speed)
            E.append_effect_to_chain("rate", sample_rate)
            E.set_input_file(wav_file)
            wav, sr = E.sox_build_flow_effects()
        else:
            # Note: enable in torchaudio>=0.8.0
            wav, sr = sox_effects.apply_effects_file(
                wav_file, [["speed", str(speed)], ["rate", str(sample_rate)]]
            )

    return wav, sr


def _extract_feature(
    batch, speed_perturb, wav_distortion_conf, feature_extraction_conf
):
    """Extract acoustic fbank feature from origin waveform.

    Speed perturbation and wave amplitude distortion is optional.

    Args:
        batch: a list of tuple (wav id , wave path).
        speed_perturb: bool, whether or not to use speed pertubation.
        wav_distortion_conf: a dict , the config of wave amplitude distortion.
        feature_extraction_conf:a dict , the config of fbank extraction.

    Returns:
        (keys, feats, labels)
    """
    keys = []
    feats = []
    lengths = []
    wav_dither = wav_distortion_conf["wav_dither"]
    wav_distortion_rate = wav_distortion_conf["wav_distortion_rate"]
    distortion_methods_conf = wav_distortion_conf["distortion_methods"]
    if speed_perturb:
        speeds = [1.0, 1.1, 0.9]
        weights = [1, 1, 1]
        speed = random.choices(speeds, weights, k=1)[0]
        # speed = random.choice(speeds)
    for i, x in enumerate(batch):
        try:
            wav = x[1]
            value = wav.strip().split(",")
            # 1 for general wav.scp, 3 for segmented wav.scp
            assert len(value) == 1 or len(value) == 3
            wav_path = value[0]
            sample_rate = torchaudio.backend.sox_io_backend.info(wav_path).sample_rate
            if "resample" in feature_extraction_conf:
                resample_rate = feature_extraction_conf["resample"]
            else:
                resample_rate = sample_rate
            if speed_perturb:
                if len(value) == 3:
                    logging.error(
                        "speed perturb does not support segmented wav.scp now"
                    )
                assert len(value) == 1
                waveform, sample_rate = _load_wav_with_speed(wav_path, speed)
            else:
                # value length 3 means using segmented wav.scp
                # incluede .wav, start time, end time
                if len(value) == 3:
                    start_frame = int(float(value[1]) * sample_rate)
                    end_frame = int(float(value[2]) * sample_rate)
                    waveform, sample_rate = torchaudio.backend.sox_io_backend.load(
                        filepath=wav_path,
                        num_frames=end_frame - start_frame,
                        frame_offset=start_frame,
                    )
                else:
                    waveform, sample_rate = torchaudio.load(wav_path)
            waveform = waveform * (1 << 15)
            if resample_rate != sample_rate:
                waveform = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=resample_rate
                )(waveform)

            if wav_distortion_rate > 0.0:
                r = random.uniform(0, 1)
                if r < wav_distortion_rate:
                    waveform = waveform.detach().numpy()
                    waveform = _waveform_distortion(waveform, distortion_methods_conf)
                    waveform = torch.from_numpy(waveform)
            mat = kaldi.fbank(
                waveform,
                num_mel_bins=feature_extraction_conf["mel_bins"],
                frame_length=feature_extraction_conf["frame_length"],
                frame_shift=feature_extraction_conf["frame_shift"],
                dither=wav_dither,
                energy_floor=0.0,
                sample_frequency=resample_rate,
            )
            mat = mat.detach().numpy()
            feats.append(mat)
            keys.append(x[0])
            lengths.append(mat.shape[0])
        except (Exception) as e:
            print(e)
            logging.warn("read utterance {} error".format(x[0]))
            pass
    # Sort it because sorting is required in pack/pad operation
    order = np.argsort(lengths)[::-1]
    sorted_keys = [keys[i] for i in order]
    sorted_feats = [feats[i] for i in order]
    labels = [x[2].split() for x in batch]
    labels = [np.fromiter(map(int, x), dtype=np.int32) for x in labels]
    sorted_labels = [labels[i] for i in order]
    return sorted_keys, sorted_feats, sorted_labels


def pad_sequence(sequences, batch_first=True, padding_value=0, padding_max_len=None):
    max_size = sequences[0].shape
    trailing_dims = max_size[1:]

    if padding_max_len is not None:
        max_len = padding_max_len
    else:
        max_len = max([s.shape[0] for s in sequences])

    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_sequences = torch.full(out_dims, fill_value=padding_value)

    for i, seq in enumerate(sequences):
        length = seq.shape[0] if seq.shape[0] <= max_len else max_len
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_sequences[i, :length, ...] = seq[:length]
        else:
            out_sequences[:length, i, ...] = seq[:length]

    return out_sequences


class CollateFunc(object):
    """Collate function for AudioDataset"""

    def __init__(
        self,
        feature_dither=0.0,
        speed_perturb=False,
        spec_aug=False,
        spec_aug_conf=None,
        spec_sub=False,
        spec_sub_conf=None,
        feature_extraction_conf=None,
        wav_distortion_conf=None,
    ):
        self.wav_distortion_conf = wav_distortion_conf
        self.feature_extraction_conf = feature_extraction_conf
        self.spec_aug = spec_aug
        self.feature_dither = feature_dither
        self.speed_perturb = speed_perturb
        self.spec_aug = spec_aug
        self.spec_aug_conf = spec_aug_conf
        self.spec_sub = spec_sub
        self.spec_sub_conf = spec_sub_conf

    def __call__(self, batch, max_tgt_len=30):
        keys, xs, ys = _extract_feature(
            batch,
            self.speed_perturb,
            self.wav_distortion_conf,
            self.feature_extraction_conf,
        )

        train_flag = True
        if ys is None:
            train_flag = False

        # optional feature dither d ~ (-a, a) on fbank feature
        # a ~ (0, 0.5)
        if self.feature_dither != 0.0:
            a = random.uniform(0, self.feature_dither)
            xs = [x + (np.random.random_sample(x.shape) - 0.5) * a for x in xs]

        # optinoal spec substitute
        if self.spec_sub:
            xs = [_spec_substitute(x, **self.spec_sub_conf) for x in xs]

        # optinoal spec augmentation
        if self.spec_aug:
            xs = [_spec_augmentation(x, **self.spec_aug_conf) for x in xs]

        xs_pad = pad_sequence(
            [torch.from_numpy(x).float() for x in xs],
            batch_first=True,
            padding_value=0.0,
            padding_max_len=1200,
        )

        xs_lengths = torch.from_numpy(
            np.array([x.shape[0] for x in xs], dtype=np.int32)
        )

        if train_flag:
            ys_pad = pad_sequence(
                [torch.from_numpy(y).int() for y in ys],
                batch_first=True,
                padding_value=IGNORE_ID,
                padding_max_len=max_tgt_len,
            )

            ys_lengths = torch.from_numpy(
                np.array([y.shape[0] for y in ys], dtype=np.int32)
            )
        else:
            ys_pad = None
            ys_lengths = None
        return keys, xs_pad, ys_pad, xs_lengths, ys_lengths


class AudioDataset(Dataset):
    def __init__(
        self,
        data_file,
        max_length=10240,
        min_length=0,
        token_max_length=200,
        token_min_length=1,
    ):
        # load all samples
        data = []
        with codecs.open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                arr = line.strip().split("\t")
                if len(arr) != 7:
                    continue
                key = arr[0].split(":")[1]
                tokenid = arr[5].split(":")[1]
                output_dim = int(arr[6].split(":")[1].split(",")[1])
                wav_path = ":".join(arr[1].split(":")[1:])
                duration = int(float(arr[2].split(":")[1]) * 1000 / 10)
                data.append((key, wav_path, duration, tokenid))
                self.output_dim = output_dim

        tot_sample = 0
        self.batches = []

        for i in range(len(data)):
            length = data[i][2]
            token_length = len(data[i][3].split())
            # remove too lang or too short utt for both input and output
            if length > max_length or length < min_length:
                continue
            elif token_length > token_max_length or token_length < token_min_length:
                continue
            else:
                tot_sample += 1
                # (uttid, wav_path, tokens)
                self.batches.append((data[i][0], data[i][1], data[i][3]))

        logging.warning(
            "total utts: {}, remove too long/short utts: {}".format(
                tot_sample, len(data) - tot_sample
            )
        )

        self.sos = self.output_dim - 1
        self.eos = self.output_dim - 1

    def __getitem__(self, idx):
        return self.batches[idx]

    def __len__(self):
        return len(self.batches)
