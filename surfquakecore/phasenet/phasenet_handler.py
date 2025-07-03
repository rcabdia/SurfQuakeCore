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
import pandas as pd
import numpy as np
import tensorflow as tf
import shutil
import time as _time
import datetime
from obspy import read, UTCDateTime, Stream
from collections import namedtuple
from scipy.interpolate import interp1d
from datetime import datetime, timedelta
import json
from surfquakecore import model_dir
import warnings

# Adjust the warning behavior globally
warnings.simplefilter("ignore")


tf.compat.v1.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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


class Config:
    seed = 123
    use_seed = True
    n_channel = 3
    n_class = 3
    sampling_rate = 100
    dt = 1.0 / sampling_rate
    X_shape = [3000, 1, n_channel]
    Y_shape = [3000, 1, n_class]
    min_event_gap = 3 * sampling_rate
    label_shape = "gaussian"
    label_width = 30
    dtype = "float32"

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class PhasenetISP:
    def __init__(self, files, batch_size=3, highpass_filter=0.5, min_p_prob=0.3, min_s_prob=0.3,
                 min_peak_distance=50, amplitude=False, plot_figure=False, save_prob=False, output=None):
        """

        Main class to initialize the picker

        :param files: Dictionary with kewords addressing to seismograms file path and their corresponding metadata (i.e. sampling rate).
        :type SurfProject: required (see Project section)

        :param batch_size: Determines the number of samples in each batch (larger batch size uses more memory but can provide more accurate updates)
        :type float:

        :param highpass_filter: Lower corner frequency of highpass filter to be applied to the raw seismogram. Set to 0 to do not apply any pre-filter
        :type float:

        :param min_p_prob: Probability threshold for P pick
        :type float:

        :param min_s_prob: Probability threshold for S pick
        :type float:

        :param min_peak_distance: Minimum peak distance
        :type float:

        :param amplitude: if return amplitude value
        :type float:

        :returns:
        :rtype: :class:`surfquakecore.phasenet.phasenet_handler.PhasenetISP`

            """
        files_ = PhasenetUtils.convert2dataframe(files)

        self.batch_size = batch_size
        self.highpass_filter = highpass_filter
        self.min_p_prob = min_p_prob
        self.min_s_prob = min_s_prob
        self.min_peak_distance = min_peak_distance

        self.model_path = model_dir
        self.file_path = ""
        self.hdf5_group = "data"
        self.result_path = "results"

        self.project = files_
        self.files = files_['fname']
        self.hdf5_file = ""
        self.result_fname = "picks"
        self.stations = ""

        self.amplitude = amplitude
        self.plot_figure = plot_figure
        self.save_prob = save_prob

        self.data_reader = None

        if output is not None:
            # check if output dir exists otherwise try to crate it
            if os.path.isdir(output):
                pass
            else:
                try:
                    os.makedirs(output)
                except Exception as error:
                    print("An exception occurred:", error)

    def phasenet(self):
        with tf.compat.v1.name_scope('create_inputs'):
            self.data_reader = PhasenetReader(
                data_dir=self.file_path,
                data_list=self.files,
                hdf5_file=self.hdf5_file,
                hdf5_group=self.hdf5_group,
                amplitude=self.amplitude,
                highpass_filter=self.highpass_filter
            )

        return self.predict()

    def predict(self):
        batch_size = self.batch_size
        dataset = self.data_reader.dataset(batch_size)
        batch = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()

        config = PhasenetModel(X_shape=self.data_reader.X_shape)

        model = PhasenetUNet(config=config, input_batch=batch, mode='pred')

        session_config = tf.compat.v1.ConfigProto()
        session_config.gpu_options.allow_growth = True

        with tf.compat.v1.Session(config=session_config) as session:
            picks = []
            amplitudes = [] if self.amplitude else None

            saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=5)
            init = tf.compat.v1.global_variables_initializer()
            session.run(init)

            latest_check_point = tf.train.latest_checkpoint(self.model_path)
            saver.restore(session, latest_check_point)

            for _ in range(0, self.data_reader.num_data, batch_size):
                if self.amplitude:
                    pred_batch, x_batch, amplitude_batch, file_batch, t0_batch, station_batch = session.run(
                        [model.preds, batch[0], batch[1], batch[2], batch[3], batch[4]],
                        feed_dict={model.drop_rate: 0, model.is_training: False},
                    )
                else:
                    pred_batch, x_batch, file_batch, t0_batch, station_batch = session.run(
                        [model.preds, batch[0], batch[1], batch[2], batch[3]],
                        feed_dict={model.drop_rate: 0, model.is_training: False},
                    )

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
                    amplitudes_aux = PhasenetUtils.extract_amplitude(amplitude_batch, picks_aux)
                    amplitudes.extend(amplitudes_aux)

            picks_ = self.picks2df(picks, amplitudes)

            return self.project.merge(picks_, how='inner', on='fname')

    @staticmethod
    def picks2df(picks, amplitudes):
        if amplitudes is None:
            _picks = pd.DataFrame(picks)

        else:
            _picks = pd.DataFrame(picks)
            _amplitudes = pd.DataFrame(amplitudes)
            _picks[['p_amp', 's_amp']] = ["", ""]

            _picks = _picks.assign(p_amp=_amplitudes['p_amp'], s_amp=_amplitudes['s_amp'])

        return _picks


class PhasenetReader:
    def __init__(self, amplitude=True, config=Config(), **kwargs):
        self.buffer = {}
        self.n_channel = config.n_channel
        self.n_class = config.n_class
        self.X_shape = config.X_shape
        self.Y_shape = config.Y_shape
        self.dt = config.dt
        self.dtype = config.dtype
        self.label_shape = config.label_shape
        self.label_width = config.label_width
        self.config = config
        self.amplitude = amplitude

        if "highpass_filter" in kwargs:
            self.highpass_filter = kwargs["highpass_filter"]

        #if format == "mseed":
        self.data_dir = kwargs["data_dir"]
        self.data_list = kwargs["data_list"]
        self.num_data = len(self.data_list)
        #else:
        #    raise Exception(f"{format} not support!")

        self.X_shape = self.get_data_shape()

    def __len__(self):
        return self.num_data

    def get_data_shape(self):
        base_name = self.data_list[0]
        meta = self.read(os.path.join(self.data_dir, base_name))

        return meta["data"].shape

    from obspy import read, Stream
    import numpy as np

    def read(self, fname):
        # Read and preprocess the seismic data
        st = read(fname)
        print('FNAME:', fname)
        st = st.detrend("spline", order=2, dspline=5 * st[0].stats.sampling_rate)
        st = st.merge(fill_value=0)

        if self.highpass_filter > 0:
            st = st.filter("highpass", freq=self.highpass_filter)

        # Ensure 24h length
        tr_raw = st[0]
        tr = self.ensure_24(tr_raw)
        st = Stream(traces=tr)

        # Component to index mapping
        comp2idx = {
            "1": 0, "X": 0, "E": 0,
            "2": 1, "Y": 1, "N": 1,
            "Z": 2,
            "H": 3  # Hydrophone, optional 4th channel
        }

        # Prepare metadata
        t0 = st[0].stats.starttime.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        nt = len(st[0].data)
        data = np.zeros([nt, self.config.n_channel], dtype=self.dtype)

        # Fill data array based on component
        for tr in st:
            # Convert component letter to uppercase
            comp = tr.stats.channel[-1].upper()
            idx = comp2idx.get(comp)
            if idx is not None and idx < self.config.n_channel:
                data[:, idx] = tr.data.astype(self.dtype)
            else:
                print(f"Warning: unknown or extra component '{comp}' in {tr.id}")

        data = data[:, np.newaxis, :]
        return {"data": data, "t0": t0}

    @staticmethod
    def ensure_24(tr):
        random_list = np.random.choice(len(tr), 100)
        times_posix = tr.times(type="timestamp")
        days_prob = times_posix[random_list.tolist()]
        days_prob_max = days_prob.tolist()
        max_prob = max(set(days_prob_max), key=days_prob_max.count)
        year = int(datetime.utcfromtimestamp(max_prob).strftime('%Y'))
        month = int(datetime.utcfromtimestamp(max_prob).strftime('%m'))
        day = int(datetime.utcfromtimestamp(max_prob).strftime('%d'))

        check_starttime = UTCDateTime(year=year, month=month, day=day, hour=00, minute=00, microsecond=00)
        check_endtime = UTCDateTime(year=year, month=month, day=day, hour=23, minute=59, second=59, microsecond=999999)

        tr.trim(starttime=check_starttime, endtime=check_endtime, pad=False, nearest_sample=True, fill_value=0)
        print("TR: ", tr)
        return tr

    def __getitem__(self, i):
        base_name = self.data_list[i]
        meta = self.read(os.path.join(self.data_dir, base_name))
        # print('META: ', self.data_dir, base_name)
        if meta == -1:
            return np.zeros(self.X_shape, dtype=self.dtype), base_name

        raw_amp = np.zeros(self.X_shape, dtype=self.dtype)
        raw_amp[: meta["data"].shape[0], ...] = meta["data"][: self.X_shape[0], ...]
        sample = np.zeros(self.X_shape, dtype=self.dtype)
        sample[: meta["data"].shape[0], ...] = PhasenetUtils.normalize_long(meta["data"])[: self.X_shape[0], ...]

        # print("T0: ", meta["t0"])
        if "t0" in meta:
            t0 = meta["t0"]
        else:
            t0 = "1970-01-01T00:00:00.000"

        if "station_id" in meta:
            station_id = meta["station_id"].split("/")[-1].rstrip("*")
        else:
            station_id = os.path.basename(base_name).rstrip("*")

        if np.isnan(sample).any() or np.isinf(sample).any():
            sample[np.isnan(sample)] = 0
            sample[np.isinf(sample)] = 0

        if self.amplitude:
            return sample[: self.X_shape[0], ...], raw_amp[: self.X_shape[0], ...], base_name, t0, station_id
        else:
            return sample[: self.X_shape[0], ...], base_name, t0, station_id

    def dataset(self, batch_size, num_parallel_calls=2, shuffle=False, drop_remainder=False):
        if self.amplitude:
            dataset = PhasenetUtils.dataset_map(self,
                                                output_types=(self.dtype, self.dtype, "string", "string", "string"),
                                                output_shapes=(self.X_shape, self.X_shape, None, None, None),
                                                num_parallel_calls=num_parallel_calls, shuffle=shuffle,)
        else:
            dataset = PhasenetUtils.dataset_map(self,
                                                output_types=(self.dtype, "string", "string", "string"),
                                                output_shapes=(self.X_shape, None, None, None),
                                                num_parallel_calls=num_parallel_calls, shuffle=shuffle,)
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder).prefetch(batch_size * 2)
        return dataset


class PhasenetModel:
    batch_size = 20
    depths = 5
    filters_root = 8
    kernel_size = [7, 1]
    pool_size = [4, 1]
    dilation_rate = [1, 1]
    class_weights = [1.0, 1.0, 1.0]
    loss_type = "cross_entropy"
    weight_decay = 0.0
    optimizer = "adam"
    momentum = 0.9
    learning_rate = 0.01
    decay_step = 1e9
    decay_rate = 0.9
    drop_rate = 0.0
    summary = True

    X_shape = [3000, 1, 3]
    n_channel = X_shape[-1]
    Y_shape = [3000, 1, 3]
    n_class = Y_shape[-1]

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def update_args(self, args):
        for k, v in vars(args).items():
            setattr(self, k, v)


class PhasenetUNet:
    def __init__(self, config=PhasenetModel(), input_batch=None, mode='train'):
        self.depths = config.depths
        self.filters_root = config.filters_root
        self.kernel_size = config.kernel_size
        self.dilation_rate = config.dilation_rate
        self.pool_size = config.pool_size
        self.X_shape = config.X_shape
        self.Y_shape = config.Y_shape
        self.n_channel = config.n_channel
        self.n_class = config.n_class
        self.class_weights = config.class_weights
        self.batch_size = config.batch_size
        self.loss_type = config.loss_type
        self.weight_decay = config.weight_decay
        self.optimizer = config.optimizer
        self.learning_rate = config.learning_rate
        self.decay_step = config.decay_step
        self.decay_rate = config.decay_rate
        self.momentum = config.momentum
        self.global_step = tf.compat.v1.get_variable(name="global_step", initializer=0, dtype=tf.int32)
        self.summary_train = []
        self.summary_valid = []
        self.precision = []
        self.recall = []
        self.f1 = []
        self.train_op = None
        self.learning_rate_node = None
        self.loss = None
        self.preds = None
        self.logits = None
        self.representation = None
        self.initializer = None
        self.regularizer = None
        self.drop_rate = None
        self.is_training = None
        self.input_batch = None
        self.X = None
        self.Y = None

        self.build(input_batch, mode=mode)

    def build(self, input_batch=None, mode='train'):
        self.add_placeholders(input_batch, mode)
        self.add_prediction_op()

        if mode in ["train", "valid", "test"]:
            self.add_loss_op()
            self.add_training_op()
            self.summary_train = tf.compat.v1.summary.merge(self.summary_train)
            self.summary_valid = tf.compat.v1.summary.merge(self.summary_valid)

        return 0

    def add_placeholders(self, input_batch=None, mode="train"):
        if input_batch is None:
            self.X = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, None, None, self.X_shape[-1]], name='X')
            self.Y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, None, None, self.n_class], name='y')
        else:
            self.X = input_batch[0]

            if mode in ["train", "valid", "test"]:
                self.Y = input_batch[1]

            self.input_batch = input_batch

        self.is_training = tf.compat.v1.placeholder(dtype=tf.bool, name="is_training")
        self.drop_rate = tf.compat.v1.placeholder(dtype=tf.float32, name="drop_rate")

    def add_prediction_op(self):
        if self.weight_decay > 0:
            weight_decay = tf.constant(self.weight_decay, dtype=tf.float32, name="weight_constant")
            self.regularizer = tf.keras.regularizers.l2(l=0.5 * weight_decay)
        else:
            self.regularizer = None

        self.initializer = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg",
                                                                           distribution="uniform")

        # down sample layers
        convs = [None] * self.depths  # store output of each depth

        with tf.compat.v1.variable_scope("Input"):
            net = self.X
            net = tf.compat.v1.layers.conv2d(net, filters=self.filters_root, kernel_size=self.kernel_size,
                                             activation=None, padding='same', dilation_rate=self.dilation_rate,
                                             kernel_initializer=self.initializer, kernel_regularizer=self.regularizer,
                                             name="input_conv")

            net = tf.compat.v1.layers.batch_normalization(net, training=self.is_training, name="input_bn")
            net = tf.nn.relu(net, name="input_relu")
            net = tf.compat.v1.layers.dropout(net, rate=self.drop_rate, training=self.is_training, name="input_dropout")

        for depth in range(0, self.depths):
            with tf.compat.v1.variable_scope("DownConv_%d" % depth):
                filters = int(2 ** depth * self.filters_root)

                net = tf.compat.v1.layers.conv2d(net, filters=filters, kernel_size=self.kernel_size, activation=None,
                                                 use_bias=False, padding='same', dilation_rate=self.dilation_rate,
                                                 kernel_initializer=self.initializer,
                                                 kernel_regularizer=self.regularizer,
                                                 name="down_conv1_{}".format(depth + 1))

                net = tf.compat.v1.layers.batch_normalization(net, training=self.is_training,
                                                              name="down_bn1_{}".format(depth + 1))

                net = tf.nn.relu(net, name="down_relu1_{}".format(depth + 1))
                net = tf.compat.v1.layers.dropout(net, rate=self.drop_rate, training=self.is_training,
                                                  name="down_dropout1_{}".format(depth + 1))
                convs[depth] = net

                if depth < self.depths - 1:
                    net = tf.compat.v1.layers.conv2d(net, filters=filters, kernel_size=self.kernel_size,
                                                     strides=self.pool_size, activation=None, use_bias=False,
                                                     padding='same',  dilation_rate=self.dilation_rate,
                                                     kernel_initializer=self.initializer,
                                                     kernel_regularizer=self.regularizer,
                                                     name="down_conv3_{}".format(depth + 1))

                    net = tf.compat.v1.layers.batch_normalization(net, training=self.is_training,
                                                                  name="down_bn3_{}".format(depth + 1))

                    net = tf.nn.relu(net, name="down_relu3_{}".format(depth + 1))
                    net = tf.compat.v1.layers.dropout(net, rate=self.drop_rate, training=self.is_training,
                                                      name="down_dropout3_{}".format(depth + 1))

        # up layers
        for depth in range(self.depths - 2, -1, -1):
            with tf.compat.v1.variable_scope("UpConv_%d" % depth):
                filters = int(2 ** depth * self.filters_root)
                net = tf.compat.v1.layers.conv2d_transpose(net, filters=filters, kernel_size=self.kernel_size,
                                                           strides=self.pool_size, activation=None, use_bias=False,
                                                           padding="same", kernel_initializer=self.initializer,
                                                           kernel_regularizer=self.regularizer,
                                                           name="up_conv0_{}".format(depth + 1))

                net = tf.compat.v1.layers.batch_normalization(net, training=self.is_training,
                                                              name="up_bn0_{}".format(depth + 1))
                net = tf.nn.relu(net, name="up_relu0_{}".format(depth + 1))
                net = tf.compat.v1.layers.dropout(net, rate=self.drop_rate, training=self.is_training,
                                                  name="up_dropout0_{}".format(depth + 1))

                # skip connection
                net = PhasenetUtils.crop_and_concat(convs[depth], net)

                net = tf.compat.v1.layers.conv2d(net, filters=filters, kernel_size=self.kernel_size, activation=None,
                                                 use_bias=False, padding='same',  dilation_rate=self.dilation_rate,
                                                 kernel_initializer=self.initializer,
                                                 kernel_regularizer=self.regularizer,
                                                 name="up_conv1_{}".format(depth + 1))

                net = tf.compat.v1.layers.batch_normalization(net, training=self.is_training,
                                                              name="up_bn1_{}".format(depth + 1))

                net = tf.nn.relu(net, name="up_relu1_{}".format(depth + 1))
                net = tf.compat.v1.layers.dropout(net, rate=self.drop_rate, training=self.is_training,
                                                  name="up_dropout1_{}".format(depth + 1))

        # Output Map
        with tf.compat.v1.variable_scope("Output"):
            net = tf.compat.v1.layers.conv2d(net, filters=self.n_class, kernel_size=(1, 1), activation=None,
                                             padding='same', kernel_initializer=self.initializer,
                                             kernel_regularizer=self.regularizer, name="output_conv")

            output = net

        with tf.compat.v1.variable_scope("representation"):
            self.representation = convs[-1]

        with tf.compat.v1.variable_scope("logits"):
            self.logits = output
            tmp = tf.compat.v1.summary.histogram("logits", self.logits)
            self.summary_train.append(tmp)

        with tf.compat.v1.variable_scope("preds"):
            self.preds = tf.nn.softmax(output)
            tmp = tf.compat.v1.summary.histogram("preds", self.preds)
            self.summary_train.append(tmp)

    def add_loss_op(self):
        if self.loss_type == "cross_entropy":
            with tf.compat.v1.variable_scope("cross_entropy"):
                flat_logits = tf.reshape(self.logits, [-1, self.n_class], name="logits")
                flat_labels = tf.reshape(self.Y, [-1, self.n_class], name="labels")

                if (np.array(self.class_weights) != 1).any():
                    class_weights = tf.constant(np.array(self.class_weights, dtype=np.float32), name="class_weights")
                    weight_map = tf.multiply(flat_labels, class_weights)
                    weight_map = tf.reduce_sum(input_tensor=weight_map, axis=1)
                    loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits, labels=flat_labels)
                    weighted_loss = tf.multiply(loss_map, weight_map)
                    loss = tf.reduce_mean(input_tensor=weighted_loss)
                else:
                    loss = tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
                                                                                               labels=flat_labels))

        elif self.loss_type == "IOU":
            with tf.compat.v1.variable_scope("IOU"):
                eps = 1e-7
                loss = 0

                for i in range(1, self.n_class):
                    intersection = eps + tf.reduce_sum(input_tensor=self.preds[:, :, :, i] * self.Y[:, :, :, i],
                                                       axis=[1, 2])
                    union = eps + tf.reduce_sum(input_tensor=self.preds[:, :, :, i], axis=[1, 2]) + \
                        tf.reduce_sum(input_tensor=self.Y[:, :, :, i], axis=[1, 2])
                    loss += 1 - tf.reduce_mean(input_tensor=intersection / union)

        elif self.loss_type == "mean_squared":
            with tf.compat.v1.variable_scope("mean_squared"):
                flat_logits = tf.reshape(self.logits, [-1, self.n_class], name="logits")
                flat_labels = tf.reshape(self.Y, [-1, self.n_class], name="labels")

                with tf.compat.v1.variable_scope("mean_squared"):
                    loss = tf.compat.v1.losses.mean_squared_error(labels=flat_labels, predictions=flat_logits)
        else:
            raise ValueError("Unknown loss function: " + self.loss_type)

        tmp = tf.compat.v1.summary.scalar("train_loss", loss)
        self.summary_train.append(tmp)
        tmp = tf.compat.v1.summary.scalar("valid_loss", loss)
        self.summary_valid.append(tmp)

        if self.weight_decay > 0:
            with tf.compat.v1.name_scope('weight_loss'):
                tmp = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
                weight_loss = tf.add_n(tmp, name="weight_loss")

            self.loss = loss + weight_loss
        else:
            self.loss = loss

    def add_training_op(self):
        if self.optimizer == "momentum":
            self.learning_rate_node = tf.compat.v1.train.exponential_decay(learning_rate=self.learning_rate,
                                                                           global_step=self.global_step,
                                                                           decay_steps=self.decay_step,
                                                                           decay_rate=self.decay_rate, staircase=True)
            optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=self.learning_rate_node,
                                                             momentum=self.momentum)

        elif self.optimizer == "adam":
            self.learning_rate_node = tf.compat.v1.train.exponential_decay(learning_rate=self.learning_rate,
                                                                           global_step=self.global_step,
                                                                           decay_steps=self.decay_step,
                                                                           decay_rate=self.decay_rate, staircase=True)

            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate_node)

        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

        tmp = tf.compat.v1.summary.scalar("learning_rate", self.learning_rate_node)
        self.summary_train.append(tmp)

    def add_metrics_op(self):
        with tf.compat.v1.variable_scope("metrics"):
            y = tf.argmax(input=self.Y, axis=-1)
            confusion_matrix = tf.cast(tf.math.confusion_matrix(labels=tf.reshape(y, [-1]),
                                       predictions=tf.reshape(self.preds, [-1]),
                                       num_classes=self.n_class, name='confusion_matrix'), dtype=tf.float32)

            c = tf.constant(1e-7, dtype=tf.float32)
            precision_p = (confusion_matrix[1, 1] + c) / (tf.reduce_sum(input_tensor=confusion_matrix[:, 1]) + c)
            recall_p = (confusion_matrix[1, 1] + c) / (tf.reduce_sum(input_tensor=confusion_matrix[1, :]) + c)
            f1_p = 2*precision_p*recall_p/(precision_p+recall_p)

            tmp1 = tf.compat.v1.summary.scalar("train_precision_p", precision_p)
            tmp2 = tf.compat.v1.summary.scalar("train_recall_p", recall_p)
            tmp3 = tf.compat.v1.summary.scalar("train_f1_p", f1_p)
            self.summary_train.extend([tmp1, tmp2, tmp3])

            tmp1 = tf.compat.v1.summary.scalar("valid_precision_p", precision_p)
            tmp2 = tf.compat.v1.summary.scalar("valid_recall_p", recall_p)
            tmp3 = tf.compat.v1.summary.scalar("valid_f1_p", f1_p)
            self.summary_valid.extend([tmp1, tmp2, tmp3])

            precision_s = (confusion_matrix[2, 2] + c) / (tf.reduce_sum(input_tensor=confusion_matrix[:, 2]) + c)
            recall_s = (confusion_matrix[2, 2] + c) / (tf.reduce_sum(input_tensor=confusion_matrix[2, :]) + c)
            f1_s = 2 * precision_s * recall_s / (precision_s + recall_s)

            tmp1 = tf.compat.v1.summary.scalar("train_precision_s", precision_s)
            tmp2 = tf.compat.v1.summary.scalar("train_recall_s", recall_s)
            tmp3 = tf.compat.v1.summary.scalar("train_f1_s", f1_s)
            self.summary_train.extend([tmp1, tmp2, tmp3])

            tmp1 = tf.compat.v1.summary.scalar("valid_precision_s", precision_s)
            tmp2 = tf.compat.v1.summary.scalar("valid_recall_s", recall_s)
            tmp3 = tf.compat.v1.summary.scalar("valid_f1_s", f1_s)
            self.summary_valid.extend([tmp1, tmp2, tmp3])

            self.precision = [precision_p, precision_s]
            self.recall = [recall_p, recall_s]
            self.f1 = [f1_p, f1_s]

    def train_on_batch(self, sess, inputs_batch, labels_batch, summary_writer, drop_rate=0.0):
        feed = {self.X: inputs_batch,
                self.Y: labels_batch,
                self.drop_rate: drop_rate,
                self.is_training: True}

        _, step_summary, step, loss = sess.run([self.train_op, self.summary_train, self.global_step, self.loss],
                                               feed_dict=feed)

        summary_writer.add_summary(step_summary, step)
        return loss

    def valid_on_batch(self, sess, inputs_batch, labels_batch, summary_writer):
        feed = {self.X: inputs_batch,
                self.Y: labels_batch,
                self.drop_rate: 0,
                self.is_training: False}

        step_summary, step, loss, preds = sess.run([self.summary_valid, self.global_step, self.loss, self.preds],
                                                   feed_dict=feed)

        summary_writer.add_summary(step_summary, step)

        return loss, preds


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
    def crop_only(net1, net2):
        """
        the size(net1) <= size(net2)
        """
        net1_shape = net1.get_shape().as_list()
        net2_shape = net2.get_shape().as_list()

        offsets = [0, (net2_shape[1] - net1_shape[1]) // 2, (net2_shape[2] - net1_shape[2]) // 2, 0]
        size = [-1, net1_shape[1], net1_shape[2], -1]
        net2_resize = tf.slice(net2, offsets, size)

        return net2_resize

    @staticmethod
    def extract_picks(preds, fnames=None, station_ids=None, t0=None, config=None):
        if preds.shape[-1] == 4:
            record = namedtuple("phase", ["fname", "station_id", "t0", "p_idx", "p_prob", "s_idx", "s_prob", "ps_idx",
                                "ps_prob"])
        else:
            record = namedtuple("phase", ["fname", "station_id", "t0", "p_idx", "p_prob", "s_idx", "s_prob"])

        picks = []

        for i, pred in enumerate(preds):

            if config is None:
                mph_p, mph_s, mpd = 0.3, 0.3, 50
            else:
                mph_p, mph_s, mpd = config['min_p_prob'], config['min_s_prob'], config['min_peak_distance']

            if fnames is None:
                fname = f"{i:04d}"
            else:
                if isinstance(fnames[i], str):
                    fname = fnames[i]
                else:
                    fname = fnames[i].decode()

            if station_ids is None:
                station_id = f"{i:04d}"
            else:
                if isinstance(station_ids[i], str):
                    station_id = station_ids[i]
                else:
                    station_id = station_ids[i].decode()

            if t0 is None:
                start_time = "1970-01-01T00:00:00.000"
            else:
                if isinstance(t0[i], str):
                    start_time = t0[i]
                else:
                    start_time = t0[i].decode()

            p_idx, p_prob, s_idx, s_prob = [], [], [], []

            for j in range(pred.shape[1]):
                p_idx_, p_prob_ = PhasenetUtils.detect_peaks(pred[:, j, 1], mph=mph_p, mpd=mpd, show=False)
                s_idx_, s_prob_ = PhasenetUtils.detect_peaks(pred[:, j, 2], mph=mph_s, mpd=mpd, show=False)
                p_idx.append(p_idx_)
                p_prob.append(p_prob_)
                s_idx.append(s_idx_)
                s_prob.append(s_prob_)

            if pred.shape[-1] == 4:
                ps_idx, ps_prob = PhasenetUtils.detect_peaks(pred[:, 0, 3], mph=0.3, mpd=mpd, show=False)
                picks.append(record(fname, station_id, start_time, p_idx, p_prob, s_idx, s_prob,
                                    ps_idx, ps_prob))
            else:
                picks.append(record(fname, station_id, start_time, list(p_idx), list(p_prob), list(s_idx),
                                    list(s_prob)))

        return picks

    @staticmethod
    def extract_amplitude(data, picks, window_p=10, window_s=5, config=None):
        record = namedtuple("amplitude", ["p_amp", "s_amp"])
        dt = 0.01 if config is None else config['dt']
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
    def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising', kpsh=False, valley=False, show=False, ax=None,
                     title=True):

        x = np.atleast_1d(x).astype('float64')
        if x.size < 3:
            return np.array([], dtype=int)
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

        if show:
            if indnan.size:
                x[indnan] = np.nan
            if valley:
                x = -x

                if mph is not None:
                    mph = -mph

            PhasenetUtils._plot(x, mph, mpd, threshold, edge, valley, ax, ind, title)

        return ind, x[ind]

    @staticmethod
    def _plot(x, mph, mpd, threshold, edge, valley, ax, ind, title):
        """Plot results of the detect_peaks function, see its help."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print('matplotlib is not available.')
        else:
            if ax is None:
                _, ax = plt.subplots(1, 1, figsize=(8, 4))
                no_ax = True
            else:
                no_ax = False

            ax.plot(x, 'b', lw=1)

            if ind.size:
                label = 'valley' if valley else 'peak'
                label = label + 's' if ind.size > 1 else label
                ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8, label='%d %s' % (ind.size, label))
                ax.legend(loc='best', framealpha=.5, numpoints=1)

            ax.set_xlim(-.02 * x.size, x.size * 1.02 - 1)
            ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
            yrange = ymax - ymin if ymax > ymin else 1
            ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)
            ax.set_xlabel('Data #', fontsize=14)
            ax.set_ylabel('Amplitude', fontsize=14)

            if title:
                if not isinstance(title, str):
                    mode = 'Valley detection' if valley else 'Peak detection'
                    title = "%s (mph=%s, mpd=%d, threshold=%s, edge='%s')" % (mode, str(mph), mpd, str(threshold), edge)

                ax.set_title(title)

            if no_ax:
                plt.show()

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
    def dataset_map(iterator, output_types, output_shapes=None, num_parallel_calls=None, name=None, shuffle=False):
        dataset = tf.data.Dataset.range(len(iterator))

        if shuffle:
            dataset = dataset.shuffle(len(iterator), reshuffle_each_iteration=True)

        @py_func_decorator(output_types, output_shapes, name=name)
        def index_to_entry(idx):
            return iterator[idx]

        return dataset.map(index_to_entry, num_parallel_calls=num_parallel_calls)

    @staticmethod
    def write_nlloc_format(dataframe, output, starttime=None, endtime=None):
        """
        Write a pandas DataFrame to a file in NonLinLoc Phase file format.

        Parameters:
            dataframe (pd.DataFrame): The input data with the required columns.
            output_file (str): Path to the output file.
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

        # If starttime and endtime are provided, filter the dataframe
        if starttime or endtime:
            dataframe['date_time'] = pd.to_datetime(dataframe['date_time'])
            if starttime:
                dataframe = dataframe[dataframe['date_time'] >= starttime]
            if endtime:
                dataframe = dataframe[dataframe['date_time'] <= endtime]

        dataframe['date_time'] = pd.to_datetime(dataframe['date_time'])

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
                elif row['weight'] <= 0.95 and row['weight'] > 0.9:
                    weight = 4.00E-02
                elif row['weight'] <= 0.9 and row['weight'] > 0.8:
                    weight = 7.00E-02
                elif row['weight'] <= 0.8 and row['weight'] > 0.7:
                    weight = 1.50E-01
                elif row['weight'] <= 0.7 and row['weight'] > 0.6:
                    weight = 1.00E-01
                else:
                    weight = 5.00E-01

                err_mag = f"{weight:.2e}"  # Error magnitude in seconds
                coda_duration = "-1.00e+00"  # Placeholder for Coda duration
                amplitude = f"{row['amplitude']:.2e}"  # Amplitude
                period = "-1.00e+00"  # Placeholder for Period

                # err_mag = f"{row['weight']:.2E}"  # Error magnitude in seconds
                # coda_duration = "-1.00E+00"  # Placeholder for Coda duration
                # amplitude = f"{row['amplitude']:.2E}"  # Amplitude
                # period = "-1.00E+00"  # Placeholder for Period

                # Construct the line
                line = (
                    f"{station} {instrument} {component} {p_phase_onset} {phase_descriptor} {first_motion} "
                    f"{date} {hour_min} {seconds} {err} {err_mag} {coda_duration} {amplitude} {period}\n"
                )
                file.write(line)

            # Add a blank line at the end for NLLoc format compliance
            file.write("\n")

    @staticmethod
    def convert2real(picks, pick_dir: str, clean_output_folder = False):
        """
        :param picks: picks is output from method split_picks in mseedutils
        :param pick_dir: directory outpur where phases are storaged
        :return:
        """

        dates = picks['date'].unique()
        fnames = picks['fname'].unique()

        if clean_output_folder:
            for path in os.listdir(pick_dir):
                fpath = os.path.join(pick_dir, path)

                if os.path.isfile(fpath) or os.path.islink(fpath):
                    os.unlink(fpath)
                else:
                    shutil.rmtree(fpath)

        for date in dates:
            pickpath = os.path.join(pick_dir, date)

            try:
                os.mkdir(pickpath)
            except OSError as e:
                print(e)

        start = _time.time()

        for date in dates:
            pickpath = os.path.join(pick_dir, date)

            for fname in fnames:
                fname_new = fname + '.txt'
                pickfile = os.path.join(pickpath, fname_new)

                savedata = picks[(picks['date'] == date) & (picks['fname'] == fname)]

                savedata = savedata.sort_values('tt')

                if len(savedata.index) > 0:
                    with open(pickfile, 'w') as f:
                        data_aux = savedata.to_string(header=False, index=False, columns=['tt', 'weight', 'amplitude'])
                        f.write(data_aux)

        print('TIME SAVING ALL FILES: ', _time.time() - start)


    @staticmethod
    def save_original_picks(original_picks, original_p_dir):
        """

        :param original_picks: picking output from phasenet (method split_picks in mseedutils)
        :param original_p_dir: output to storage original_picks
        :return:
        """

        print('saving_picks_original_format')
        pick_path = os.path.join(original_p_dir, "original_picks.csv")
        original_picks.to_csv(pick_path, index=False)


    @staticmethod
    def split_picks(picks):

        """
        :param picks: A DataFrame with all pick information
        :type picks: Pandas DataFrame
        """

        print('get_picks & converting to REAL associator format')
        prob_threshold = 0.3
        columns = ['date', 'fname', 'year', 'month', 'day', 'net', 'station', 'flag', 'tt', 'date_time',
                   'weight', 'amplitude', 'phase', 'channel']
        split_picks_ = pd.DataFrame(columns=columns)

        stats = picks['stats']
        t0 = picks['t0']
        ppick_tmp = picks['p_idx']
        spick_tmp = picks['s_idx']
        pprob_tmp = picks['p_prob']
        sprob_tmp = picks['s_prob']

        if 'p_amp' in picks.columns:
            p_amp = picks['p_amp']
        else:
            p_amp = []

        if 's_amp' in picks.columns:
            s_amp = picks['s_amp']
        else:
            s_amp = []

        start = _time.time()

        for i in np.arange(0, len(ppick_tmp)):
            ppick = []
            spick = []
            pprob = []
            sprob = []
            pamp = []
            samp = []
            pamp_um = []
            samp_um = []
            split_aux_p = []
            split_aux_s = []
            t0[i] = datetime.strptime(t0[i], "%Y-%m-%dT%H:%M:%S.%f")
            date_start = t0[i]
            year = t0[i].year
            month = t0[i].month
            day = t0[i].day
            station = stats[i].station
            network = stats[i].network
            channel = stats[i].channel

            ss = t0[i].hour*3600 + t0[i].minute*60 + t0[i].second + t0[i].microsecond/1000000

            samplingrate = 1/stats[i].sampling_rate

            if len(ppick_tmp[i][0]) > 0:
                ppick_um = ppick_tmp[i][0][:]
                pprob_um = pprob_tmp[i][0][:]

                if len(p_amp) > 0:
                    pamp_um = p_amp[i][0][:]

                for j in np.arange(0, len(ppick_um)):
                    if ppick_um[j] != ',':
                        ppick.append(ppick_um[j])

                for j in np.arange(0, len(pprob_um)):
                    if pprob_um[j] != ',':
                        pprob.append(pprob_um[j])

                if len(p_amp) > 0:
                    for j in np.arange(0, len(pamp_um)):
                        if pamp_um[j] != ',':
                            pamp.append(pamp_um[j])

                for j in np.arange(0, len(pprob)):
                    # ppick = pick samples from file waveform starttime
                    # t0 = datetime of the starttime
                    if float(pprob[j]) >= prob_threshold and j <= len(ppick)-1:
                        fname = network + '.' + station + '.' + 'P'
                        delta_time = int(ppick[j])*samplingrate
                        tp = delta_time+ss
                        t_pick = date_start + timedelta(seconds=delta_time)
                        t_pick_string = t_pick.strftime("%Y-%m-%dT%H:%M:%S.%f")
                        amp = float(pamp[j])*2080*25 if len(p_amp) > 0 else 0
                        split_aux_p.append([str(year)+"{:02d}".format(month)+"{:02d}".format(day), fname, year, month, day, network, station,
                                            1, tp, t_pick_string, pprob[j], amp, "P", channel])

                split_picks_ = pd.concat([split_picks_, pd.DataFrame(split_aux_p, columns=columns)], ignore_index=True)

            if len(spick_tmp[i][0]) > 0:
                spick_um = spick_tmp[i][0][:]
                sprob_um = sprob_tmp[i][0][:]

                if len(s_amp) > 0:
                    samp_um = s_amp[i][0][:]

                for j in np.arange(0, len(spick_um)):
                    if spick_um[j] != ',':
                        spick.append(spick_um[j])

                for j in np.arange(0, len(sprob_um)):
                    if sprob_um[j] != ',':
                        sprob.append(sprob_um[j])

                if len(s_amp) > 0:
                    for j in np.arange(0, len(samp_um)):
                        if samp_um[j] != ',':
                            samp.append(samp_um[j])

                for j in np.arange(0, len(sprob)):
                    if float(sprob[j]) >= prob_threshold and j <= len(spick)-1:
                        fname = network + '.' + station + '.' + 'S'
                        delta_time = int(spick[j]) * samplingrate
                        tp = delta_time+ss
                        amp = float(samp[j]) * 2080 * 25 if len(s_amp) > 0 else 0
                        t_pick = date_start + timedelta(seconds=delta_time)
                        t_pick_string = t_pick.strftime("%Y-%m-%dT%H:%M:%S.%f")
                        split_aux_s.append([str(year)+"{:02d}".format(month)+"{:02d}".format(day), fname, year, month, day, network, station, 1,
                                            tp, t_pick_string, sprob[j], amp, "S", channel])

                split_picks_ = pd.concat([split_picks_, pd.DataFrame(split_aux_s, columns=columns)], ignore_index=True)

        print('LEN TOTAL: ', len(ppick_tmp))
        print('START: ', _time.time()-start)

        return split_picks_

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

    def calc_timestamp(self, timestamp, sec):
        timestamp = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%f") + timedelta(seconds=sec)
        return timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    def save_picks_json(self, picks, output_dir, dt=0.01, amps=None, fname=None):
        if fname is None:
            fname = "picks.json"

        picks_ = []
        if amps is None:
            for pick in picks:
                for idxs, probs in zip(pick.p_idx, pick.p_prob):
                    for idx, prob in zip(idxs, probs):
                        picks_.append(
                            {
                                "id": pick.station_id,
                                "timestamp": self.calc_timestamp(pick.t0, float(idx) * dt),
                                "prob": prob.astype(float),
                                "type": "p",
                            }
                        )
                for idxs, probs in zip(pick.s_idx, pick.s_prob):
                    for idx, prob in zip(idxs, probs):
                        picks_.append(
                            {
                                "id": pick.station_id,
                                "timestamp": self.calc_timestamp(pick.t0, float(idx) * dt),
                                "prob": prob.astype(float),
                                "type": "s",
                            }
                        )
        else:
            for pick, amplitude in zip(picks, amps):
                for idxs, probs, amps in zip(pick.p_idx, pick.p_prob, amplitude.p_amp):
                    for idx, prob, amp in zip(idxs, probs, amps):
                        picks_.append(
                            {
                                "id": pick.station_id,
                                "timestamp": self.calc_timestamp(pick.t0, float(idx) * dt),
                                "prob": prob.astype(float),
                                "amp": amp.astype(float),
                                "type": "p",
                            }
                        )
                for idxs, probs, amps in zip(pick.s_idx, pick.s_prob, amplitude.s_amp):
                    for idx, prob, amp in zip(idxs, probs, amps):
                        picks_.append(
                            {
                                "id": pick.station_id,
                                "timestamp": self.calc_timestamp(pick.t0, float(idx) * dt),
                                "prob": prob.astype(float),
                                "amp": amp.astype(float),
                                "type": "s",
                            }
                        )
        with open(os.path.join(output_dir, fname), "w") as fp:
            json.dump(picks_, fp)

        return 0

