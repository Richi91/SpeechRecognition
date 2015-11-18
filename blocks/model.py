from theano import tensor
from toolz import merge
from blocks.bricks import Tanh, Maxout, Linear, FeedforwardSequence, Bias, Initializable, MLP, Sequence
from blocks.bricks.attention import SequenceContentAttention, AttentionRecurrent
from blocks.bricks.base import application, lazy
from blocks.bricks.lookup import LookupTable
from blocks.bricks.parallel import Fork
from blocks.bricks.recurrent import GatedRecurrent, Bidirectional, LSTM, SimpleRecurrent, RecurrentStack
from blocks.bricks.sequence_generators import LookupFeedback, Readout, SoftmaxEmitter, SequenceGenerator, TrivialFeedback, FakeAttentionRecurrent, BaseSequenceGenerator
from blocks.roles import add_role, WEIGHT
from blocks.utils import shared_floatx_nans
from picklable_itertools.extras import equizip
#import utils
from blocks.utils import dict_union, dict_subset
from blocks.bricks import NDimensionalSoftmax
import numpy as np
import numpy
import theano
import logging
from theano import tensor

from blocks.bricks import (
    Bias, Identity, Initializable, MLP, Tanh)
from blocks.bricks.attention import SequenceContentAttention
from blocks.bricks.base import application
from blocks.bricks.recurrent import (
    BaseRecurrent, RecurrentStack)
from blocks.bricks.sequence_generators import (
    SequenceGenerator, Readout,
    SoftmaxEmitter, LookupFeedback)
from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter
from blocks.roles import OUTPUT
from blocks.search import BeamSearch
from blocks.serialization import load_parameter_values
from blocks.initialization import Orthogonal, Uniform,Constant







class RecurrentWithFork(Initializable):

    @lazy(allocation=['input_dim'])
    def __init__(self, transition, input_dim, hidden_dim,
                 rec_weights_init, ff_weights_init, biases_init, **kwargs):
        super(RecurrentWithFork, self).__init__(**kwargs)
        self.rec_weights_init=rec_weights_init
        self.ff_weights_init=ff_weights_init
        self.biases_init=biases_init
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim

        self.transition=transition
        self.transition.dim=self.hidden_dim
        self.transition.weights_init=self.rec_weights_init
        self.transition.bias_init=self.biases_init


        self.fork = Fork(
            [name for name in self.transition.apply.sequences if name != 'mask'],
             prototype=Linear())
        self.fork.input_dim = self.input_dim
        self.fork.output_dims = [self.transition.apply.brick.get_dim(name)
                                 for name in self.fork.output_names]
        self.fork.weights_init=self.ff_weights_init
        self.fork.biases_init=self.biases_init

        self.children = [transition, self.fork]

#    def _push_allocation_config(self):#
#        #super(RecurrentWithFork, self)._push_allocation_config()
#        self.transition.dim=self.hidden_dim
#        self.fork.input_dim = self.input_dim
#        self.fork.output_dims = [self.transition.apply.brick.get_dim(name)
#                                 for name in self.fork.output_names]

#    def _push_initialization_config(self):
#        #super(RecurrentWithFork, self)._push_initialization_config()
#        self.fork.weights_init=self.ff_weights_init
#        self.fork.biases_init=self.biases_init
#        self.transition.weights_init=self.rec_weights_init
#        self.transition.bias_init=self.biases_init

    @application(inputs=['input_', 'mask'])
    def apply(self, input_, mask=None, **kwargs):
        states=self.transition.apply(
            mask=mask, **dict_union(self.fork.apply(input_, as_dict=True), kwargs))
        # I don't know, why blocks returns a list [states, cell] for LSTM
        # but just states (no list) for GRU or normal RNN. We only want LSTM's states.
        # cells should not be visible from outside.
        return states[0] if isinstance(states,list) else states

    @apply.property('outputs')
    def apply_outputs(self):
        return self.transition.apply.states




class DeepBidirectional(Initializable):
    def __init__(self, transition, dim_input, dims_hidden,
                 rec_weights_init, ff_weights_init, biases_init, **kwargs):
        super(DeepBidirectional, self).__init__(**kwargs)

        for layer_num, (input_dim, hidden_dim) in enumerate(
                zip([dim_input] + list(2 * np.array(dims_hidden)), dims_hidden)):
            bidir = Bidirectional(
                RecurrentWithFork(
                    transition=transition(dim=hidden_dim, activation=Tanh()),
                    input_dim=input_dim, hidden_dim=hidden_dim,
                    rec_weights_init=rec_weights_init,
                    ff_weights_init=ff_weights_init, biases_init=biases_init,
                    name='with_fork'),
                name='bidir{}'.format(layer_num))
            self.children.append(bidir)

    @application(outputs=['encoded'])
    def apply(self, input_, mask):
        for bidir in self.children:
            input_ = bidir.apply(input_, mask)
        return input_




class SpeechRecognizer(Initializable):
    """
    :parameter: num_features: number of features or input dimensionality
    :parameter: dims_bottom: list of dims for MLP before bidirectional
    :parameter: dims_bidir: list of dims for RNN in bidirectional
    :parameter: dims_top: list of dims for MLP after bidirectional
    :parameter: bidir_trans: transition of bidirectional (e.g. GatedRecurrent or LSTM)
    :parameter: bottom_activation: Nonlinearity for bot MLP
    :parameter: top_activation: Nonlinearity for top MLP. The last activation an output dim equal
     to the number of classes is always None
    :parameter:
    """
    def __init__(self, num_features, dims_bottom=[], dims_bidir=[250], dims_top=[],
                 bidir_trans=GatedRecurrent, bottom_activation=None, top_activation=None,
                 **kwargs):

        if bottom_activation is None:
            bottom_activation = Tanh()
        if top_activation is None:
            top_activation = Tanh()

        super(SpeechRecognizer, self).__init__(**kwargs)

        # TODO: think about putting this into conf ?
        # Owant to use rthogonal in  LSTM only for the recurrent weights (W_states),
        # but 1. blocks concats all 4 recurrents matrices to one. Does Orthogonal Init
        # know this and do the correct init? 2. peepholes (vectors/diag-mats) are initialized
        # with the same weight_init ..... cant init vector with Orthogonal.
        # For now, don't use Orthogonal.

        # TODO: Maybe implement LSTM by myself without cells as output and possibility to init orthogonal
        # self.rec_weights_init = Orthogonal(scale=1.0)
        self.rec_weights_init = Uniform(mean=0, width=0.01)
        self.ff_weights_init = Uniform(mean=0, width=0.01)
        self.biases_init = Constant(0.0)

        self.bidir_trans = bidir_trans


        # *********** Bottom: Before BiRNN ************
        if dims_bottom:
            bottom = MLP([bottom_activation] * len(dims_bottom),
                         [num_features] + dims_bottom,
                         weights_init=self.ff_weights_init,
                         biases_init=self.biases_init,
                         name="bottom",
                         )

        else:
            bottom = Identity(name='bottom')

        # ************ Middle: Deep BiRNN *************
        middle = DeepBidirectional(transition=self.bidir_trans, dims_hidden=dims_bidir,
                        dim_input=dims_bottom[-1] if len(dims_bottom) else num_features,
                        rec_weights_init=self.rec_weights_init,
                        ff_weights_init=self.ff_weights_init,
                        biases_init=self.biases_init,
                        )

        # ************ Top: after BiRNN ***************
        if dims_top: # last non-linearity is None (linear), because we will use softmax outside of this class
            top = MLP([bottom_activation]*(len(dims_top)-1) + [None],[2 * dims_bidir[-1]] + dims_top,
                      weights_init=self.ff_weights_init,
                      biases_init=self.biases_init,
                      name="top",
                      )
        else:
            top = Identity(name='top')

        # Remember child bricks
        self.bottom = bottom
        self.middle = middle
        self.top = top
        self.children = [bottom, middle, top]


    @application(inputs=['sequence', 'mask'],
                 outputs=['output'])
    def apply(self, sequence, mask):
        bottom_processed = self.bottom.apply(sequence)
        middle_processed = self.middle.apply(
            input_=bottom_processed,
            mask=mask)
        output = self.top.apply(middle_processed)
        return output



class SimpleSpeechRecognizer(Initializable):
    """
    Initializable, does nothing more than combining
    class DeepBidirectional and an MLP as output
    Parameters
    ----------
    
    transition: transition of bidirectional (e.g. GatedRecurrent or LSTM)
    dims_transition: list of dims for RNN in bidirectional
    num_features: number of features or input dimensionality
    num_classes
    
    """
    def __init__(self, transition, dims_transition, 
                 num_features, num_classes, **kwargs):
                     
        super(SimpleSpeechRecognizer, self).__init__(**kwargs)
        # TODO: think about putting this into conf ?
        # Owant to use rthogonal in  LSTM only for the recurrent weights (W_states),
        # but 1. blocks concats all 4 recurrents matrices to one. Does Orthogonal Init
        # know this and do the correct init? 2. peepholes (vectors/diag-mats) are initialized
        # with the same weight_init ..... cant init vector with Orthogonal.
        # For now, don't use Orthogonal.
        # TODO: Maybe implement LSTM by myself

        # self.rec_weights_init = Orthogonal(scale=1.0)
        self.rec_weights_init = Uniform(mean=0, width=0.01)
        self.ff_weights_init = Uniform(mean=0, width=0.01)
        self.biases_init = Constant(0.0)
        self.transition = transition
        
        # ************ Deep BiRNN *************
        self.dblstm = DeepBidirectional(
                        transition=self.transition, 
                        dims_hidden=dims_transition,
                        dim_input=num_features,
                        rec_weights_init=self.rec_weights_init,
                        ff_weights_init=self.ff_weights_init,
                        biases_init=self.biases_init,)

        # ************ Output ***************
        self.output = MLP(
                        [None],[2 * dims_transition[-1]]+[num_classes],
                        weights_init=self.ff_weights_init,
                        biases_init=self.biases_init,
                        name="top",)

        # Remember child bricks
        self.children = [self.dblstm, self.output]


    @application(inputs=['sequence', 'mask'],
                 outputs=['output'])
    def apply(self, sequence, mask):
        blstm_processed = self.dblstm.apply(
            input_=sequence, mask=mask)
        return self.output.apply(blstm_processed)



