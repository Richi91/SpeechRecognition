# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 12:30:50 2015

@author: Richard Kurle
"""

import theano
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream
from fuel.transformers import Mapping
from fuel.datasets.hdf5 import H5PYDataset
from theano import tensor
from blocks.bricks.recurrent import LSTM, GatedRecurrent, SimpleRecurrent
from blocks.bricks import NDimensionalSoftmax
from blocks.extensions.saveload import Checkpoint
from blocks.algorithms import AdaDelta, GradientDescent, CompositeRule, StepClipping, Momentum
from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.model import Model
from utils import ShuffleDim, AddInputNoise, createLogger
import conf
import ctc_cost as ctc
from model import SimpleSpeechRecognizer, SpeechRecognizer
from utils import PhonemeErrorRate, PhonemeErrorRateFramewise ,categorical_crossentropy_batch
from blocks.graph import apply_noise, apply_dropout
from blocks.filter import VariableFilter
import logging
from blocks.extensions.stopping import FinishIfNoImprovementAfter
from blocks.extensions import predicates
from blocks.extensions.training import TrackTheBest
from blocks.roles import INPUT, WEIGHT

#logger = logging.getLogger(__name__)
#logger = createLogger('log.log')


#***************** DataSet *******************
sources=[conf.input_theano, conf.input_mask_theano, conf.target_theano]
if conf.task=='CTC':
    sources.append(conf.target_mask_theano)
    path_to_dataset=conf.path_to_dataset_CTC
elif conf.task=='framewise':
    path_to_dataset=conf.path_to_dataset_framewise
else: 
    raise ValueError, conf.task

train_set = H5PYDataset(path_to_dataset, which_sets=('train',),sources=tuple(sources))
val_set = H5PYDataset(path_to_dataset, which_sets=('val',),sources=tuple(sources))
test_set = H5PYDataset(path_to_dataset, which_sets=('test',),sources=tuple(sources))

scheme_train = ShuffledScheme(examples=train_set.num_examples, batch_size=conf.batch_size)
scheme_val = SequentialScheme(examples=val_set.num_examples, batch_size=conf.batch_size)
scheme_test = SequentialScheme(examples=test_set.num_examples, batch_size=conf.batch_size)

stream_train = DataStream(dataset=train_set, iteration_scheme=scheme_train)
stream_val = DataStream(dataset=val_set, iteration_scheme=scheme_val)
stream_test = DataStream(dataset=test_set, iteration_scheme=scheme_test)

stream_train = Mapping(data_stream = stream_train, mapping=ShuffleDim)
stream_val = Mapping(data_stream = stream_val, mapping=ShuffleDim)
stream_test = Mapping(data_stream = stream_test, mapping=ShuffleDim)

stream_train = AddInputNoise(data_stream = stream_train, std=conf.std_input_noise)


epoch_iterator=stream_train.get_epoch_iterator(as_dict = True)
test_batch=next(epoch_iterator)
num_features=test_batch[conf.input_theano].shape[-1]


if conf.transition=='GRU':
    transition=GatedRecurrent
elif conf.transition=='LSTM':
    transition=LSTM
elif conf.transition=='RNN':
    transition=SimpleRecurrent
else:
    raise ValueError, conf.transition



num_classes = 39 if conf.mapTo39Phonemes_Training else 61
if conf.task=='CTC': 
    num_classes+=1

# ********* inputs + targets + masks **********
x = tensor.tensor3(name=conf.input_theano)
x_m = tensor.matrix(name=conf.input_mask_theano)
y = tensor.lmatrix(name=conf.target_theano)
y_m = tensor.matrix(name=conf.target_mask_theano)

# ******************* Model *******************
recognizer = SimpleSpeechRecognizer(transition=transition,
                dims_transition=conf.dims_transition,
                num_features=num_features, num_classes=num_classes)

#recognizer = SpeechRecognizer(
#    num_features=num_features, dims_bottom=[],
#    dims_bidir=conf.dims_transition, dims_top=[num_classes],
#    bidir_trans=GatedRecurrent, bottom_activation=None)


# ******************* output *******************
y_hat = recognizer.apply(x,x_m)
y_hat.name = 'outputs'
y_hat_softmax = NDimensionalSoftmax().apply(y_hat, extra_ndim = y_hat.ndim - 2)
y_hat_softmax.name = 'outputs_softmax'

# there is a cost function for monitoring and for training, because one is more stable to compute
# gradients and seems also to be more memory efficient, but does not compute the true cost.
if conf.task=='CTC':
    cost_train = ctc.pseudo_cost(y, y_hat, y_m, x_m).mean()
    cost_train.name = "cost_train"
    
    cost_monitor = ctc.cost(y, y_hat_softmax, y_m, x_m).mean()
    cost_monitor.name = "cost_monitor"
elif conf.task=='framewise':
    cost_train = categorical_crossentropy_batch().apply(y_hat_softmax, y, x_m)
    cost_train.name='cost'
    cost_monitor = cost_train
else:
    raise ValueError, conf.task


recognizer.initialize()
cg = ComputationGraph([cost_train, y_hat, x_m, y, y_m])


weights = VariableFilter(roles=[WEIGHT])(cg.variables)
cg = apply_noise(cg, weights, conf.weight_noise)



#************* training algorithm *************
model = Model(cost_train)
if conf.step_rule=='AdaDelta':
    step_rule = AdaDelta()
elif conf.step_rule=='Momentum':
    step_rule = Momentum(learning_rate=conf.learning_rate, momentum=conf.momentum)
else:
    raise('step_rule not known: {}'.format(conf.step_rule))

step_rule = CompositeRule([step_rule, StepClipping(conf.step_clipping)])
algorithm = GradientDescent(cost=cost_train, parameters=cg.parameters, step_rule = step_rule)


#***************** main loop ****************
train_monitor = TrainingDataMonitoring([cost_monitor], prefix="train")
val_monitor = DataStreamMonitoring([cost_monitor], stream_val, prefix="val")

if conf.task=='CTC':
    PER_Monitor = PhonemeErrorRate
elif conf.task=='framewise':
    PER_Monitor = PhonemeErrorRateFramewise
else:
    raise ValueError, conf.task
per_val_monitor = PER_Monitor(stream_val, theano.function([x,x_m],y_hat_softmax), 
                             before_first_epoch=True,after_epoch=True, prefix='valPER')
per_test_monitor = PER_Monitor(stream_test, theano.function([x,x_m],y_hat_softmax), 
                             before_first_epoch=False, every_n_epochs=5, prefix='testPER')

checkpoint=Checkpoint(conf.path_to_model, after_training=False)
checkpoint.add_condition(['after_epoch'], predicate=predicates.OnLogRecord('valid_log_p_best_so_far'))
extensions=[val_monitor,
            train_monitor,
            per_val_monitor,
            per_test_monitor,
            Timing(),
            FinishAfter(after_n_epochs=conf.max_epochs),
            checkpoint,
            Printing(),
            TrackTheBest(record_name='val_monitor',notification_name='valid_log_p_best_so_far'),
            FinishIfNoImprovementAfter(notification_name='valid_log_p_best_so_far', epochs=conf.epochs_early_stopping),
            ]
main_loop = MainLoop(
    algorithm = algorithm,
    data_stream = stream_train,
    model = model,
    extensions=extensions,
)

main_loop.run()