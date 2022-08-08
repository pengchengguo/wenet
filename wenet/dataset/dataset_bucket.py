import argparse
import codecs
import copy
import logging
import random
import math
import numpy as np
import yaml

from PIL import Image
from PIL.Image import BICUBIC
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import torchaudio.sox_effects as sox_effects

from wenet.dataset.wav_distortion import distort_wav_conf
from wenet.utils.common import IGNORE_ID

torchaudio.set_audio_backend("sox_io")


def _spec_augmentation(
    x,
    warp_for_time=False,
    num_t_mask=2,
    num_f_mask=2,
    max_t=50,
    max_f=10,
    max_w=80,
):
    """ Deep copy x and do spec augmentation then return it

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


def _waveform_distortion(waveform, distortion_methods_conf):
    """ Apply distortion on waveform

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
    """ Load the wave from file and apply speed perpturbation

    Args:
        wav_file: input feature, T * F 2D

    Returns:
        augmented feature
    """
    if speed == 1.0:
        wav, sr = torchaudio.load(wav_file)
    else:
        sample_rate = torchaudio.backend.sox_io_backend.info(
            wav_file
        ).sample_rate
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
    """ Extract acoustic fbank feature from origin waveform.

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

    for i, x in enumerate(batch):
        try:
            wav = x[1]
            value = wav.strip().split(",")
            # 1 for general wav.scp, 3 for segmented wav.scp
            assert len(value) == 1 or len(value) == 3
            wav_path = value[0]
            sample_rate = torchaudio.backend.sox_io_backend.info(
                wav_path
            ).sample_rate
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
                    (
                        waveform,
                        sample_rate,
                    ) = torchaudio.backend.sox_io_backend.load(
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
                    waveform = _waveform_distortion(
                        waveform, distortion_methods_conf
                    )
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
            logging.warning("read utterance {}: {} error".format(x[0], x[1]))
            pass

    # Sort it because sorting is required in pack/pad operation
    order = np.argsort(lengths)[::-1]
    sorted_keys = [keys[i] for i in order]
    sorted_feats = [feats[i] for i in order]
    labels = [x[2].split() for x in batch]
    labels = [np.fromiter(map(int, x), dtype=np.int32) for x in labels]
    sorted_labels = [labels[i] for i in order]

    return sorted_keys, sorted_feats, sorted_labels


def pad_sequence(
    sequences,
    batch_first=True,
    padding_value=0,
    padding_max_len=None,
):
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
        length = seq.shape[0]
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_sequences[i, :length, ...] = seq
        else:
            out_sequences[:length, i, ...] = seq

    return out_sequences


class CollateFunc(object):
    """ Collate function for AudioDataset
    """

    def __init__(
        self,
        wav_distortion_conf=None,
        feature_extraction_conf=None,
        feature_dither=0.0,
        speed_perturb=False,
        spec_aug=False,
        spec_aug_conf=None,
    ):
        """
        Args:
            raw_wav:
                    True if input is raw wav and feature extraction is needed.
                    False if input is extracted feature
        """
        self.wav_distortion_conf = wav_distortion_conf
        self.feature_extraction_conf = feature_extraction_conf
        self.feature_dither = feature_dither
        self.speed_perturb = speed_perturb
        self.spec_aug = spec_aug
        self.spec_aug_conf = spec_aug_conf

    def __call__(self, batch):
        data = batch[0][0]
        max_src_len = batch[0][1]
    
        keys, xs, ys = _extract_feature(
            data,
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

        # optinoal spec augmentation
        if self.spec_aug:
            xs = [_spec_augmentation(x, **self.spec_aug_conf) for x in xs]

        xs_pad = pad_sequence(
            [torch.from_numpy(x).float() for x in xs],
            batch_first=True,
            padding_value=0.0,
            padding_max_len=max_src_len,
        )

        xs_lengths = torch.from_numpy(
            np.array([x.shape[0] for x in xs], dtype=np.int32)
        )

        if train_flag:
            ys_pad = pad_sequence(
                [torch.from_numpy(y).int() for y in ys],
                batch_first=True,
                padding_value=IGNORE_ID,
                padding_max_len=30,
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
        frame_bucket_limit="200,300",
        batch_bucket_limit="220,200",
        batch_factor=0.2,
        sort=True,
    ):
        """Bucket dataset for loading audio data.

        Attributes::
            data_file: input data file
                Plain text data file, each line contains following 7 fields,
                which is split by '\t':
                    utt:utt1
                    feat:tmp/data/file1.wav or feat:tmp/data/fbank.ark:30
                    feat_shape: 4.95(in seconds) or feat_shape:495,80(495 is in frames)
                    text:i love you
                    token: i <space> l o v e <space> y o u
                    tokenid: int id of this token
                    token_shape: M,N    # M is the number of token, N is vocab size
            max_length: drop utterance which is greater than max_length(10ms)
            min_length: drop utterance which is less than min_length(10ms)
            token_max_length: drop utterance which is greater than token_max_length,
                especially when use char unit for english modeling
            token_min_length: drop utterance which is less than token_max_length
            batch_type: static or dynamic, see max_frames_in_batch(dynamic)
            batch_size: number of utterances in a batch,
               it's for static batch size.
            max_frames_in_batch: max feature frames in a batch,
               when batch_type is dynamic, it's for dynamic batch size.
               Then batch_size is ignored, we will keep filling the
               batch until the total frames in batch up to max_frames_in_batch.
            sort: whether to sort all data, so the utterance with the same
               length could be filled in a same batch.
            raw_wav: use raw wave or extracted featute.
                if raw wave is used, dynamic waveform-level augmentation could be used
                and the feature is extracted by torchaudio.
                if extracted featute(e.g. by kaldi) is used, only feature-level
                augmentation such as specaug could be used.
        """
        frame_bucket_limit = [int(i) for i in frame_bucket_limit.split(",")]
        batch_bucket_limit = [
            int(int(i) * batch_factor) for i in batch_bucket_limit.split(",")
        ]
        assert len(frame_bucket_limit) == len(batch_bucket_limit)
        bucket_select_dict = self.bucket_init(frame_bucket_limit)

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
                
        if sort:
            data = sorted(data, key=lambda x: x[2])

        tot_sample = 0
        self.batches = []
        caches = {}  # caches to store data
        for idx, max_frame in enumerate(frame_bucket_limit):
            # caches[idx]: [data, num_sentence, max_frame]
            caches[idx] = [[], 0, max_frame]

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
                bucket_idx = bucket_select_dict[length]
                caches[bucket_idx][0].append((data[i][0], data[i][1], data[i][3]))
                caches[bucket_idx][1] += 1

                if caches[bucket_idx][1] >= batch_bucket_limit[bucket_idx]:
                    self.batches.append((caches[bucket_idx][0], caches[bucket_idx][2]))
                    caches[bucket_idx] = [[], 0, frame_bucket_limit[bucket_idx]]
        
        logging.warning(
            "total utts: {}, remove too long/short utts: {}".format(tot_sample, len(data) - tot_sample)
        )
        # handle the left samples which are not able to form a complete batch
        for key, value in caches.items():
            if len(value[0]) != 0:
                repeat_time = math.ceil(batch_bucket_limit[key] / len(value[0]))
                data_expand = value[0] * repeat_time
                self.batches.append((data_expand[: batch_bucket_limit[key]], value[2]))

        del caches

        self.sos = self.output_dim - 1
        self.eos = self.output_dim - 1


    def bucket_init(self, frame_bucket_limit):
        bucket_select_dict = {}
        for idx, _ in enumerate(frame_bucket_limit):
            low = 0 if idx == 0 else frame_bucket_limit[idx - 1] + 1
            high = frame_bucket_limit[idx] + 1
            bucket_select_dict.update(dict([[i, idx] for i in range(low, high)]))

        return bucket_select_dict

    def __getitem__(self, idx):
        return self.batches[idx]

    def __len__(self):
        return len(self.batches)


if __name__ == "__main__":
    torch.manual_seed(777)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="config file")
    parser.add_argument("--data_file", help="input data file")
    args = parser.parse_args()

    with open(args.config_file, "r") as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    # Init dataset and data loader
    collate_conf = configs["collate_conf"]
    collate_func = CollateFunc(**collate_conf)
    dataset_conf = configs["dataset_conf"]
    dataset = AudioDataset(
        args.data_file,
        max_length=dataset_conf["max_length"],
        min_length=dataset_conf["min_length"],
        token_max_length=dataset_conf["token_max_length"],
        token_min_length=dataset_conf["token_min_length"],
        frame_bucket_limit=dataset_conf["frame_bucket_limit"],
        batch_bucket_limit=dataset_conf["batch_bucket_limit"],
        batch_factor=dataset_conf["batch_factor"],
    )

    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        sampler=None,
        num_workers=0,
        collate_fn=collate_func,
    )

    for i, batch in enumerate(data_loader):
        keys = batch[0]
        xs_pad = batch[1]
        ys_pad = batch[2]
        xs_lengths = batch[3]
        ys_lengths = batch[4]
        print(xs_pad.shape)
        print(ys_pad.shape)
