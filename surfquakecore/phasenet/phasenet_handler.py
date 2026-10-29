# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: phasenet_handler.py
# Program: surfQuake & ISP
# Date: January 2024
# Purpose: Phase Picking toolbox manager
# Author: Roberto Cabieces & Thiago C. Junqueira
#  Email: rcabdia@roa.es
# --------------------------------------------------------------------

import os
import time
import pandas as pd
import numpy as np
from surfquakecore.config.global_config import read_stream
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import shutil
from obspy import UTCDateTime
from collections import namedtuple
from scipy.interpolate import interp1d
from datetime import datetime, timedelta
from surfquakecore import model_dir
import warnings

# Adjust the warning behavior globally
warnings.simplefilter("ignore")

tf.compat.v1.disable_eager_execution()


def py_func_decorator(output_types=None, output_shapes=None, name=None):
    def decorator(func):
        def call(*args):
            nonlocal output_shapes

            flat_output_types = tf.nest.flatten(output_types)
            flat_values = tf.numpy_function(func, inp=args, Tout=flat_output_types, name=name)

            if output_shapes is not None:
                for v, s in zip(flat_values, output_shapes):
                    v.set_shape(s)

            return tf.nest.pack_sequence_as(output_types, flat_values)

        return call

    return decorator


N_CHANNEL = 3
DTYPE = "float32"
# component letter -> input channel of the network ("H", hydrophone, is ignored: the network has 3 channels)
COMP2IDX = {"1": 0, "X": 0, "E": 0, "2": 1, "Y": 1, "N": 1, "Z": 2, "H": 3}


class PhasenetISP:
    def __init__(self, files, batch_size=3, highpass_filter=0.5, min_p_prob=0.3, min_s_prob=0.3,
                 min_peak_distance=50, amplitude=False, output=None, device="auto", verbose=False):
        """

        Main class to initialize the picker

        :param files: Dictionary with kewords addressing to seismograms file path and their corresponding metadata (i.e. sampling rate).
        :type SurfProject: required (see Project section)

        :param batch_size: Number of files (days) processed at once (larger batch size uses more memory)
        :type int:

        :param highpass_filter: Lower corner frequency of highpass filter to be applied to the raw seismogram. Set to 0 to do not apply any pre-filter
        :type float:

        :param min_p_prob: Probability threshold for P pick
        :type float:

        :param min_s_prob: Probability threshold for S pick
        :type float:

        :param min_peak_distance: Minimum peak distance
        :type float:

        :param amplitude: if return amplitude value
        :type bool:

        :param output: directory to create (optional)
        :type str:

        :param device: "auto" (GPU if available, otherwise CPU), "cpu" or "gpu"
        :type str:

        :param verbose: print the number of P and S picks of every file
        :type bool:

        :returns:
        :rtype: :class:`surfquakecore.phasenet.phasenet_handler.PhasenetISP`

            """
        self.project = PhasenetUtils.convert2dataframe(files)
        self.files = self.project['fname']

        self.batch_size = batch_size
        self.highpass_filter = highpass_filter
        self.min_p_prob = min_p_prob
        self.min_s_prob = min_s_prob
        self.min_peak_distance = min_peak_distance
        self.amplitude = amplitude
        self.device = device
        self.verbose = verbose
        self.data_reader = None

        if output is not None:
            os.makedirs(output, exist_ok=True)

    def phasenet(self):
        self.data_reader = PhasenetReader(
            data_list=self.files,
            amplitude=self.amplitude,
            highpass_filter=self.highpass_filter
        )

        return self.predict()

    def predict(self):
        batch_size = self.batch_size
        n_files = self.data_reader.num_data
        n_batches = -(-n_files // batch_size)  # ceil

        dataset = self.data_reader.dataset(batch_size)
        batch = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
        model = PhasenetUNet(batch)

        session_config = tf.compat.v1.ConfigProto()
        session_config.gpu_options.allow_growth = True

        gpus = tf.config.list_physical_devices('GPU')
        if self.device == "cpu":
            session_config.device_count['GPU'] = 0
        elif self.device == "gpu" and not gpus:
            raise RuntimeError("device='gpu' requested but no GPU was found")
        using_gpu = self.device != "cpu" and len(gpus) > 0

        print(f"[PhaseNet] {n_files} files | device: {'GPU' if using_gpu else 'CPU'} | batch size: {batch_size} "
              f"| min prob P/S: {self.min_p_prob}/{self.min_s_prob}", flush=True)
        start = time.time()

        with tf.compat.v1.Session(config=session_config) as session:
            picks = []
            amplitudes = [] if self.amplitude else None

            saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
            saver.restore(session, tf.train.latest_checkpoint(model_dir))

            for n_batch in range(1, n_batches + 1):
                if self.amplitude:
                    pred_batch, amplitude_batch, file_batch, t0_batch, station_batch = session.run(
                        [model.preds, batch[1], batch[2], batch[3], batch[4]],
                        feed_dict={model.is_training: False})
                else:
                    pred_batch, file_batch, t0_batch, station_batch = session.run(
                        [model.preds, batch[1], batch[2], batch[3]],
                        feed_dict={model.is_training: False})

                picks_aux = PhasenetUtils.extract_picks(
                    preds=pred_batch,
                    fnames=file_batch,
                    station_ids=station_batch,
                    t0=t0_batch,
                    config={'min_p_prob': self.min_p_prob,
                            'min_s_prob': self.min_s_prob,
                            'min_peak_distance': self.min_peak_distance})

                picks.extend(picks_aux)

                if self.amplitude:
                    amplitudes.extend(PhasenetUtils.extract_amplitude(amplitude_batch, picks_aux))

                # progress
                if self.verbose:
                    for p in picks_aux:
                        print(f"    {os.path.basename(p.fname)}: {len(p.p_idx[0])} P, {len(p.s_idx[0])} S picks")
                done = min(n_batch * batch_size, n_files)
                elapsed = time.time() - start
                eta = elapsed / done * (n_files - done)
                print(f"[PhaseNet] {done}/{n_files} files ({100 * done / n_files:.0f}%) | "
                      f"elapsed {timedelta(seconds=int(elapsed))} | remaining ~{timedelta(seconds=int(eta))}",
                      flush=True)

        return self.project.merge(self.picks2df(picks, amplitudes), how='inner', on='fname')

    @staticmethod
    def picks2df(picks, amplitudes):
        _picks = pd.DataFrame(picks)

        if amplitudes is not None:
            _amplitudes = pd.DataFrame(amplitudes)
            _picks = _picks.assign(p_amp=_amplitudes['p_amp'], s_amp=_amplitudes['s_amp'])

        return _picks


class PhasenetReader:
    def __init__(self, data_list, amplitude=True, highpass_filter=0.5):
        self.data_list = data_list
        self.amplitude = amplitude
        self.highpass_filter = highpass_filter
        self.num_data = len(self.data_list)
        self._first = None
        self.X_shape = self.get_data_shape()

    def __len__(self):
        return self.num_data

    def get_data_shape(self):
        # The length of the first file is the input length of the network. It is kept to not read it twice.
        self._first = self.read(self.data_list[0])

        return self._first["data"].shape

    def read(self, fname):
        # Read and preprocess the seismic data
        st = read_stream(fname)
        st = st.detrend("spline", order=2, dspline=5 * st[0].stats.sampling_rate)
        st = st.merge(fill_value=0)

        if self.highpass_filter > 0:
            st = st.filter("highpass", freq=self.highpass_filter)

        # Only the first trace is used, trimmed to 24h
        tr = self.ensure_24(st[0])
        t0 = tr.stats.starttime.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

        # Fill the data array according to the component
        data = np.zeros([len(tr.data), N_CHANNEL], dtype=DTYPE)
        comp = tr.stats.channel[-1].upper()
        idx = COMP2IDX.get(comp)
        if idx is not None and idx < N_CHANNEL:
            data[:, idx] = tr.data.astype(DTYPE)
        else:
            print(f"Warning: unknown or extra component '{comp}' in {tr.id}")

        return {"data": data[:, np.newaxis, :], "t0": t0}

    @staticmethod
    def ensure_24(tr):
        """Trim the trace to the UTC day that contains most of its samples (the first day if tied)."""
        start, end = tr.stats.starttime, tr.stats.endtime
        first_day = UTCDateTime(year=start.year, month=start.month, day=start.day)

        day, best = first_day, -1
        for k in range(int((end - first_day) // 86400) + 1):
            day_k = first_day + 86400 * k
            overlap = min(end, day_k + 86400) - max(start, day_k)  # seconds of the trace inside this day
            if overlap > best:
                day, best = day_k, overlap

        check_endtime = UTCDateTime(year=day.year, month=day.month, day=day.day,
                                    hour=23, minute=59, second=59, microsecond=999999)

        tr.trim(starttime=day, endtime=check_endtime, pad=False, nearest_sample=True, fill_value=0)

        return tr

    def __getitem__(self, i):
        base_name = self.data_list[i]

        if i == 0 and self._first is not None:
            meta, self._first = self._first, None
        else:
            meta = self.read(base_name)

        data = meta["data"]

        # data and amplitude are cut / zero-padded to the length of the first file
        sample = np.zeros(self.X_shape, dtype=DTYPE)
        sample[: data.shape[0], ...] = PhasenetUtils.normalize_long(data)[: self.X_shape[0], ...]
        sample[~np.isfinite(sample)] = 0

        station_id = os.path.basename(base_name).rstrip("*")

        if self.amplitude:
            raw_amp = np.zeros(self.X_shape, dtype=DTYPE)
            raw_amp[: data.shape[0], ...] = data[: self.X_shape[0], ...]
            return sample, raw_amp, base_name, meta["t0"], station_id
        else:
            return sample, base_name, meta["t0"], station_id

    def dataset(self, batch_size, num_parallel_calls=2):
        if self.amplitude:
            output_types = (DTYPE, DTYPE, "string", "string", "string")
            output_shapes = (self.X_shape, self.X_shape, None, None, None)
        else:
            output_types = (DTYPE, "string", "string", "string")
            output_shapes = (self.X_shape, None, None, None)

        dataset = PhasenetUtils.dataset_map(self, output_types, output_shapes, num_parallel_calls)

        return dataset.batch(batch_size).prefetch(batch_size * 2)


class PhasenetUNet:
    """
    PhaseNet U-Net, inference only.
    The layer names are the variable names stored in the checkpoint: do not rename them.
    """
    depths = 5
    filters_root = 8
    kernel_size = [7, 1]
    pool_size = [4, 1]
    n_class = 3

    def __init__(self, input_batch):
        self.is_training = tf.compat.v1.placeholder(dtype=tf.bool, name="is_training")
        self.preds = self.build(input_batch[0])

    def conv_bn_relu(self, net, filters, conv_name, bn_name, strides=(1, 1), transpose=False, use_bias=False):
        conv = tf.compat.v1.layers.conv2d_transpose if transpose else tf.compat.v1.layers.conv2d
        net = conv(net, filters=filters, kernel_size=self.kernel_size, strides=strides, padding="same",
                   use_bias=use_bias, name=conv_name)
        net = tf.compat.v1.layers.batch_normalization(net, training=self.is_training, name=bn_name)

        return tf.nn.relu(net)

    def build(self, net):
        convs = [None] * self.depths  # output of each depth, used by the skip connections

        with tf.compat.v1.variable_scope("Input"):
            net = self.conv_bn_relu(net, self.filters_root, "input_conv", "input_bn", use_bias=True)

        # down sample layers
        for depth in range(self.depths):
            with tf.compat.v1.variable_scope("DownConv_%d" % depth):
                filters = 2 ** depth * self.filters_root
                net = self.conv_bn_relu(net, filters, f"down_conv1_{depth + 1}", f"down_bn1_{depth + 1}")
                convs[depth] = net

                if depth < self.depths - 1:
                    net = self.conv_bn_relu(net, filters, f"down_conv3_{depth + 1}", f"down_bn3_{depth + 1}",
                                            strides=self.pool_size)

        # up sample layers
        for depth in range(self.depths - 2, -1, -1):
            with tf.compat.v1.variable_scope("UpConv_%d" % depth):
                filters = 2 ** depth * self.filters_root
                net = self.conv_bn_relu(net, filters, f"up_conv0_{depth + 1}", f"up_bn0_{depth + 1}",
                                        strides=self.pool_size, transpose=True)
                net = PhasenetUtils.crop_and_concat(convs[depth], net)  # skip connection
                net = self.conv_bn_relu(net, filters, f"up_conv1_{depth + 1}", f"up_bn1_{depth + 1}")

        with tf.compat.v1.variable_scope("Output"):
            net = tf.compat.v1.layers.conv2d(net, filters=self.n_class, kernel_size=(1, 1), padding="same",
                                             name="output_conv")

        return tf.nn.softmax(net)


class PhasenetUtils:
    @staticmethod
    def crop_and_concat(net1, net2):
        """
        the size(net1) <= size(net2)
        """
        # dynamic shape
        chn1 = net1.get_shape().as_list()[-1]
        chn2 = net2.get_shape().as_list()[-1]
        net1_shape = tf.shape(net1)
        net2_shape = tf.shape(net2)

        offsets = [0, (net2_shape[1] - net1_shape[1]) // 2, (net2_shape[2] - net1_shape[2]) // 2, 0]
        size = [-1, net1_shape[1], net1_shape[2], -1]
        net2_resize = tf.slice(net2, offsets, size)

        out = tf.concat([net1, net2_resize], 3)
        out.set_shape([None, None, None, chn1 + chn2])

        return out

    @staticmethod
    def extract_picks(preds, fnames, station_ids, t0, config):
        record = namedtuple("phase", ["fname", "station_id", "t0", "p_idx", "p_prob", "s_idx", "s_prob"])
        picks = []

        for i, pred in enumerate(preds):
            p_idx, p_prob, s_idx, s_prob = [], [], [], []

            for j in range(pred.shape[1]):
                p_idx_, p_prob_ = PhasenetUtils.detect_peaks(pred[:, j, 1], mph=config['min_p_prob'],
                                                             mpd=config['min_peak_distance'])
                s_idx_, s_prob_ = PhasenetUtils.detect_peaks(pred[:, j, 2], mph=config['min_s_prob'],
                                                             mpd=config['min_peak_distance'])
                p_idx.append(p_idx_)
                p_prob.append(p_prob_)
                s_idx.append(s_idx_)
                s_prob.append(s_prob_)

            # strings come from TensorFlow as bytes
            picks.append(record(fnames[i].decode(), station_ids[i].decode(), t0[i].decode(),
                                p_idx, p_prob, s_idx, s_prob))

        return picks

    @staticmethod
    def extract_amplitude(data, picks, window_p=10, window_s=5, dt=0.01):
        record = namedtuple("amplitude", ["p_amp", "s_amp"])
        window_p = int(window_p / dt)
        window_s = int(window_s / dt)
        amps = []

        for i, (da, pi) in enumerate(zip(data, picks)):
            p_amp, s_amp = [], []

            for j in range(da.shape[1]):
                amp = np.max(np.abs(da[:, j, :]), axis=-1)
                tmp = []

                for k in range(len(pi.p_idx[j]) - 1):
                    tmp.append(np.max(amp[pi.p_idx[j][k]:min(pi.p_idx[j][k] + window_p, pi.p_idx[j][k + 1])]))

                if len(pi.p_idx[j]) >= 1:
                    tmp.append(np.max(amp[pi.p_idx[j][-1]:pi.p_idx[j][-1] + window_p]))

                p_amp.append(tmp)
                tmp = []

                for k in range(len(pi.s_idx[j]) - 1):
                    tmp.append(np.max(amp[pi.s_idx[j][k]:min(pi.s_idx[j][k] + window_s, pi.s_idx[j][k + 1])]))

                if len(pi.s_idx[j]) >= 1:
                    tmp.append(np.max(amp[pi.s_idx[j][-1]:pi.s_idx[j][-1] + window_s]))

                s_amp.append(tmp)

            amps.append(record(p_amp, s_amp))

        return amps

    @staticmethod
    def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising', kpsh=False, valley=False):

        x = np.atleast_1d(x).astype('float64')
        if x.size < 3:
            return np.array([], dtype=int), np.array([])
        if valley:
            x = -x
            if mph is not None:
                mph = -mph

        # find indices of all peaks
        dx = x[1:] - x[:-1]

        # handle NaN's
        indnan = np.where(np.isnan(x))[0]

        if indnan.size:
            x[indnan] = np.inf
            dx[np.where(np.isnan(dx))[0]] = np.inf

        ine, ire, ife = np.array([[], [], []], dtype=int)

        if not edge:
            ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
        else:
            if edge.lower() in ['rising', 'both']:
                ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
            if edge.lower() in ['falling', 'both']:
                ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]

        ind = np.unique(np.hstack((ine, ire, ife)))

        # handle NaN's
        if ind.size and indnan.size:
            # NaN's and values close to NaN's cannot be peaks
            ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))), invert=True)]

        # first and last values of x cannot be peaks
        if ind.size and ind[0] == 0:
            ind = ind[1:]

        if ind.size and ind[-1] == x.size - 1:
            ind = ind[:-1]

        # remove peaks < minimum peak height
        if ind.size and mph is not None:
            ind = ind[x[ind] >= mph]

        # remove peaks - neighbors < threshold
        if ind.size and threshold > 0:
            dx = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
            ind = np.delete(ind, np.where(dx < threshold)[0])

        # detect small peaks closer than minimum peak distance
        if ind.size and mpd > 1:
            ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
            idel = np.zeros(ind.size, dtype=bool)

            for i in range(ind.size):
                if not idel[i]:
                    # keep peaks with the same height if kpsh is True
                    idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                           & (x[ind[i]] > x[ind] if kpsh else True)
                    idel[i] = 0  # Keep current peak

            # remove the small peaks and sort back the indices by their occurrence
            ind = np.sort(ind[~idel])

        return ind, x[ind]

    @staticmethod
    def normalize_long(data, axis=(0,), window=3000):
        """
        data: nt, nch
        """
        nt, nar, nch = data.shape
        if window is None:
            window = nt

        shift = window // 2

        # std in slide windows
        data_pad = np.pad(data, ((window // 2, window // 2), (0, 0), (0, 0)), mode="reflect")
        t = np.arange(0, nt, shift, dtype="int")
        std = np.zeros([len(t) + 1, nar, nch])
        mean = np.zeros([len(t) + 1, nar, nch])

        for i in range(1, len(std)):
            std[i, :] = np.std(data_pad[i * shift: i * shift + window, :, :], axis=axis)
            mean[i, :] = np.mean(data_pad[i * shift: i * shift + window, :, :], axis=axis)

        t = np.append(t, nt)

        std[-1, ...], mean[-1, ...] = std[-2, ...], mean[-2, ...]
        std[0, ...], mean[0, ...] = std[1, ...], mean[1, ...]

        # normalize data with interplated std
        t_interp = np.arange(nt, dtype="int")
        std_interp = interp1d(t, std, axis=0, kind="slinear")(t_interp)
        mean_interp = interp1d(t, mean, axis=0, kind="slinear")(t_interp)
        tmp = np.sum(std_interp, axis=(0, 1))
        std_interp[std_interp == 0] = 1.0
        data = (data - mean_interp) / std_interp

        # dropout effect of < 3 channel
        nonzero = np.count_nonzero(tmp)
        if (nonzero < 3) and (nonzero > 0):
            data *= 3.0 / nonzero

        return data

    @staticmethod
    def dataset_map(iterator, output_types, output_shapes=None, num_parallel_calls=None):
        dataset = tf.data.Dataset.range(len(iterator))

        @py_func_decorator(output_types, output_shapes)
        def index_to_entry(idx):
            return iterator[idx]

        return dataset.map(index_to_entry, num_parallel_calls=num_parallel_calls)

    @staticmethod
    def write_nlloc_format(dataframe, output, starttime=None, endtime=None):
        """
        Write a pandas DataFrame to a file in NonLinLoc Phase file format.

        Parameters:
            dataframe (pd.DataFrame): The input data with the required columns.
            output (str): Directory where nll_picks.txt is written.
            starttime (str or UTCDateTime, optional): Start time of the range.
                If str, it should follow the format "%Y-%m-%d %H:%M:%S".
            endtime (str or UTCDateTime, optional): End time of the range.
                If str, it should follow the format "%Y-%m-%d %H:%M:%S".
        """

        # Convert starttime and endtime to pandas-compatible datetime objects
        if starttime:
            starttime = pd.Timestamp(str(starttime)) if isinstance(starttime, UTCDateTime) else pd.Timestamp(starttime)
        if endtime:
            endtime = pd.Timestamp(str(endtime)) if isinstance(endtime, UTCDateTime) else pd.Timestamp(endtime)

        dataframe = dataframe.copy()  # do not modify the table of the caller
        dataframe['date_time'] = pd.to_datetime(dataframe['date_time'])

        # If starttime and endtime are provided, filter the dataframe
        if starttime:
            dataframe = dataframe[dataframe['date_time'] >= starttime]
        if endtime:
            dataframe = dataframe[dataframe['date_time'] <= endtime]

        output_file = os.path.join(output, "nll_picks.txt")
        # Write to NLLoc format
        with open(output_file, 'w') as file:

            # Write the header line
            header = ("Station_name\tInstrument\tComponent\tP_phase_onset\tP_phase_descriptor\t"
                      "First_Motion\tDate\tHour_min\tSeconds\tErr\tErrMag\tCoda_duration\tAmplitude\tPeriod\n")
            file.write(header)

            for _, row in dataframe.iterrows():
                station = row['station'].ljust(6)  # Station name, left-justified, 6 chars
                instrument = "?".ljust(4)  # Placeholder for Instrument
                component = row['channel'].ljust(4)  # Placeholder for Component
                p_phase_onset = "?"  # Placeholder for P phase onset
                phase_descriptor = row['phase'].ljust(6)  # Phase descriptor (e.g., P, S)
                first_motion = "?"  # Placeholder for First Motion
                date = f"{row['date']}"  # Date in yyyymmdd format
                hour_min = f"{row['date_time'].hour:02}{row['date_time'].minute:02}"  # hhmm
                seconds = f"{row['date_time'].second + row['date_time'].microsecond / 1e6:07.4f}"  # ss.ssss
                err = "GAU"  # Error type (GAU)

                if row['weight'] > 0.95:
                    weight = 2.00E-02
                elif 0.95 >= row['weight'] > 0.9:
                    weight = 4.00E-02
                elif 0.9 >= row['weight'] > 0.8:
                    weight = 7.00E-02
                elif 0.8 >= row['weight'] > 0.7:
                    weight = 1.00E-1
                elif 0.7 >= row['weight'] > 0.6:
                    weight = 1.50E-1
                else:
                    weight = 5.00E-1

                err_mag = f"{weight:.2e}"  # Error magnitude in seconds
                coda_duration = "-1.00e+00"  # Placeholder for Coda duration
                amplitude = f"{row['amplitude']:.2e}"  # Amplitude
                period = "-1.00e+00"  # Placeholder for Period

                # Construct the line
                line = (
                    f"{station} {instrument} {component} {p_phase_onset} {phase_descriptor} {first_motion} "
                    f"{date} {hour_min} {seconds} {err} {err_mag} {coda_duration} {amplitude} {period}\n"
                )
                file.write(line)

            # Add a blank line at the end for NLLoc format compliance
            file.write("\n")

    @staticmethod
    def convert2real(picks, pick_dir: str, clean_output_folder=False):
        """
        :param picks: picks is output from method split_picks in mseedutils
        :param pick_dir: directory outpur where phases are storaged
        :return:
        """

        dates = picks['date'].unique()

        if clean_output_folder:
            for path in os.listdir(pick_dir):
                fpath = os.path.join(pick_dir, path)

                if os.path.isfile(fpath) or os.path.islink(fpath):
                    os.unlink(fpath)
                else:
                    shutil.rmtree(fpath)

        for date in dates:
            pickpath = os.path.join(pick_dir, date)

            os.makedirs(pickpath, exist_ok=True)

        for (date, fname), savedata in picks.groupby(['date', 'fname'], sort=False):
            savedata = savedata.sort_values('tt')
            with open(os.path.join(pick_dir, date, fname + '.txt'), 'w') as f:
                f.write(savedata.to_string(header=False, index=False, columns=['tt', 'weight', 'amplitude']))

    @staticmethod
    def save_original_picks(original_picks, original_p_dir):
        """

        :param original_picks: picking output from phasenet (method split_picks in mseedutils)
        :param original_p_dir: output to storage original_picks
        :return:
        """

        pick_path = os.path.join(original_p_dir, "original_picks.csv")
        original_picks.to_csv(pick_path, index=False)

    @staticmethod
    def split_picks(picks):

        """
        :param picks: A DataFrame with all pick information
        :type picks: Pandas DataFrame
        """

        prob_threshold = 0.3
        columns = ['date', 'fname', 'year', 'month', 'day', 'net', 'station', 'flag', 'tt', 'date_time',
                   'weight', 'amplitude', 'phase', 'channel']
        has_amp = 'p_amp' in picks.columns and 's_amp' in picks.columns

        rows = []
        for _, row in picks.iterrows():
            date_start = datetime.strptime(row['t0'], "%Y-%m-%dT%H:%M:%S.%f")
            network, station, channel = row['stats'].network, row['stats'].station, row['stats'].channel
            sampling_interval = 1 / row['stats'].sampling_rate
            ss = (date_start.hour * 3600 + date_start.minute * 60 + date_start.second
                  + date_start.microsecond / 1000000)

            for phase in ("P", "S"):
                key = phase.lower()
                idx, prob = row[key + '_idx'][0], row[key + '_prob'][0]
                amp = row[key + '_amp'][0] if has_amp else None

                for j in range(len(prob)):
                    if float(prob[j]) >= prob_threshold:
                        # idx = pick samples from file waveform starttime, date_start = datetime of the starttime
                        delta_time = int(idx[j]) * sampling_interval
                        t_pick = date_start + timedelta(seconds=delta_time)
                        rows.append([date_start.strftime("%Y%m%d"), network + '.' + station + '.' + phase,
                                     date_start.year, date_start.month, date_start.day, network, station, 1,
                                     delta_time + ss, t_pick.strftime("%Y-%m-%dT%H:%M:%S.%f"), prob[j],
                                     float(amp[j]) * 2080 * 25 if has_amp else 0, phase, channel])

        split_picks_ = pd.DataFrame(rows, columns=columns)

        # keep the column types the output always had
        return split_picks_.astype({c: object for c in ('year', 'month', 'day', 'flag')})

    @staticmethod
    def print_summary(picks, n_files, output_dir, elapsed):
        """
        Print a short report at the end of the run.

        :param picks: output of split_picks
        :param n_files: number of files that were processed
        :param output_dir: directory where the results were saved
        :param elapsed: total time in seconds
        """
        line = "=" * 60
        print(f"\n{line}\n PhaseNet summary\n{line}")
        print(f" Files processed     : {n_files}")
        print(f" Stations with picks : {len(picks[['net', 'station']].drop_duplicates())}")
        print(f" P picks             : {int((picks['phase'] == 'P').sum())}")
        print(f" S picks             : {int((picks['phase'] == 'S').sum())}")
        print(f" Total time          : {timedelta(seconds=int(elapsed))}")
        print(f" Results saved in    : {output_dir}")
        print("   original_picks.csv, nll_picks.txt and one folder per day (REAL format)")
        print(line)

    @staticmethod
    def convert2dataframe(path_project):
        project_converted = []
        _names = path_project.keys()

        for name in _names:
            for i in range(len(path_project[name])):
                project_converted.append({
                    'id': name,
                    'fname': path_project[name][i][0],
                    'stats': path_project[name][i][1]
                })

        return pd.DataFrame.from_dict(project_converted)