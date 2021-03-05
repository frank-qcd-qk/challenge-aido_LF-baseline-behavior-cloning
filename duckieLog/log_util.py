import pickle
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Dict

import numpy as np

SCHEMA_VERSION = "1.0.0"


@dataclass
class Step:
    obs: np.ndarray = None
    reward: float = None
    action: List[float] = field(default_factory=list)
    done: bool = False
    obstacle: float = None


@dataclass
class Episode:
    steps: List[Step] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    version: str = SCHEMA_VERSION


class Reader:
    def __init__(self, log_file):
        self._log_file = open(log_file, 'rb')

    def read(self):
        end = False
        Observation = []
        Linear = []
        Angular = []

        while not end:
            try:
                log = pickle.load(self._log_file)
                for entry in log:
                    step = entry['step']
                    Observation.append(step[0])
                    action = step[1]
                    Linear.append(action[0])
                    Angular.append(action[1])

            except EOFError:
                end = True

        return Observation, Linear, Angular

    def modern_read(self):
        episode_data = None
        episode_index = 0
        end = False
        Observation = []
        Linear = []
        Angular = []
        while True:
            if episode_data is None:
                try:
                    episode_data = pickle.load(self._log_file)
                    episode_index = 0
                except EOFError:
                    print("End of log file!")
                    print("Size: ", len(Observation), " ", len(Linear), " ", len(Angular))
                    return Observation, Linear, Angular
            try:
                step = episode_data.steps[episode_index]
                episode_index += 1
                Observation.append(step.obs)
                Linear.append(step.action[0])
                Angular.append(step.action[1])
            except IndexError:
                episode_data = None
                continue

    def close(self):
        self._log_file.close()


class Logger:
    def __init__(self, env, log_file):
        self.env = env
        self.episode = Episode(version=SCHEMA_VERSION)
        self.episode_count = 0

        self._log_file = open(log_file, 'wb')
        # we log the data in a multithreaded fashion
        self._multithreaded_recording = ThreadPoolExecutor(4)
        # self.recording = []

    def get_step_length(self):
        return len(self.episode.steps)

    def log(self, step: Step, info: Dict):
        if self.episode.metadata is None:
            self.episode.metadata = info
        self.episode.steps.append(step)

    def reset_episode(self):
        self.episode = Episode(version=SCHEMA_VERSION)

    def on_episode_done(self):
        length = self.get_step_length()
        print(f"episode {self.episode_count} done, total length {length} writing to file")
        # The next file cause all episodes to be written to the same pickle FP. (Overwrite first?)
        # self._multithreaded_recording.submit(lambda: self._commit(self.episode))
        self._commit(self.episode)
        self.episode = Episode(version=SCHEMA_VERSION)
        self.episode_count += 1

    def _commit(self, episode):
        # we use pickle to store our data
        # pickle.dump(self.recording, self._log_file)
        pickle.dump(episode, self._log_file)
        self._log_file.flush()
        # del self.recording[:]
        # self.recording.clear()

    def close(self):
        self._multithreaded_recording.shutdown()
        self._log_file.close()