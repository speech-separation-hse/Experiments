import random
from pathlib import Path
from typing import Union

import audio_reader

import torch

from torch.utils.data import DataLoader, Dataset


class DatasetMixer:
    def __init__(self,
                 path_to_video: Path,
                 n_samples: int,
                 path_to_features: Union[Path, None] = None,
                 reader: Union[audio_reader.AudioReader, None] = None,
                 deterministic: bool = True,
                 cache_directory: Union[Path, None] = None):
        self.path_to_video = path_to_video
        self.speakers: list[Path] = list(path_to_video.iterdir())
        self.n_samples = n_samples
        self.path_to_features = path_to_features
        self.reader = reader if reader else audio_reader.get_default_audio_reader()
        self.deterministic = deterministic
        self.cache_directory = cache_directory
        if self.cache_directory:
            self.cache_directory.mkdir(exist_ok=True)
        assert not (self.cache_directory and not self.deterministic), 'Could not cache while not deterministic'

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index) -> dict:
        if self.cache_directory:
            cache_filename = self.cache_directory / f'{index}.pt'
            if cache_filename.is_file():
                output = torch.load(cache_filename)
                return output

        if self.deterministic:
            random.seed(index)

        first_speaker_path, second_speaker_path = random.sample(self.speakers, 2)

        first_speaker_video_path = random.choice(list(first_speaker_path.rglob('*.mp4')))
        second_speaker_video_path = random.choice(list(second_speaker_path.rglob('*.mp4')))

        output = {'first_video_path': first_speaker_video_path, 'second_video_path': second_speaker_video_path}

        first_audio = self.reader.load(first_speaker_video_path)
        second_audio = self.reader.load(second_speaker_video_path)

        first_audio_duration = self.reader.length(first_audio)
        second_audio_duration = self.reader.length(second_audio)
        min_duration = min(first_audio_duration, second_audio_duration)
        first_audio_offset = random.uniform(0, first_audio_duration - min_duration)
        second_audio_offset = random.uniform(0, second_audio_duration - min_duration)
        output.update({
            'duration': min_duration,
            'first_audio_offset': first_audio_offset,
            'second_audio_offset': second_audio_offset
        })

        first_audio = self.reader.slice(audio=first_audio, offset=first_audio_offset, duration=min_duration)
        second_audio = self.reader.slice(audio=second_audio, offset=second_audio_offset, duration=min_duration)
        mix_audio = self.reader.overlay(first_audio, second_audio)
        output.update({
            'first_audio': self.reader.to_tensor(first_audio),
            'second_audio': self.reader.to_tensor(second_audio),
            'mix_audio': self.reader.to_tensor(mix_audio)
        })

        if self.path_to_features:
            first_video_features_path = self.path_to_features / f'{first_speaker_path.stem}_{first_speaker_video_path.stem}.npy'
            first_video_features = torch.load(first_video_features_path, map_location='cpu')
            second_video_features_path = self.path_to_features / f'{second_speaker_path.stem}_{second_speaker_video_path.stem}.npy'
            second_video_features = torch.load(second_video_features_path, map_location='cpu')
            output.update({
                'first_video_features': first_video_features,
                'second_video_features': second_video_features
            })

        if self.cache_directory:
            torch.save(output, cache_filename)

        return output


if __name__ == '__main__':
    path_to_video = Path('video')
    path_to_features = Path('features')

    dataset = DatasetMixer(path_to_video=path_to_video, n_samples=30, path_to_features=path_to_features, cache_directory=Path('caches'))
    result = dataset[10]

    result2 = dataset[10]

    assert result.keys() == result2.keys()

    import timeit

    times = timeit.repeat(lambda: dataset[10], repeat=10, number=1)

    print(min(times))
