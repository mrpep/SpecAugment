import tensorflow as tf

class SpecAugment(tf.keras.layers.Layer):
  """ SpecAugment layer based on https://arxiv.org/abs/1904.08779. Implements in pure tensorflow the
  time and frequency masking. Time warping is not implemented. The augmentation is only applied during
  training.
  It expects as input a tensor (probably a spectrogram) of shape [BATCH_SIZE, T_BINS, F_BINS]

  Args:
  f_gaps: list of 2 elements [min_gaps,max_gaps]. For each spectrogram N gaps are generated in the frequency
    axis, being N drawn from an uniform distribution [min_gaps,max_gaps]. If min_gaps is set to 0, some
    spectrograms won't have gaps in the frequency domain.
  t_gaps: same as f_gaps but applied in the time axis.
  f_gap_size: list of 2 elements [min_size,max_size]. For each gap, N consecutive frequency bins will be masked,
    being N drawn from an uniform distribution [min_gaps,max_gaps].
  t_gap_size: same as f_gap_size but applied in the frequency axis.
  """
  def __init__(self,f_gaps = [0,4],t_gaps = [0,4],f_gap_size=[5,15],t_gap_size=[5,15],name=None):
    super(SpecAugment, self).__init__(name=name)
    self.f_gaps = f_gaps
    self.t_gaps = t_gaps
    self.f_gap_size = f_gap_size
    self.t_gap_size = t_gap_size

  def build(self,input_shape):
    self.input_shape_list = input_shape.as_list()
    self.mask = tf.Variable(initial_value=tf.keras.initializers.Ones()(shape=self.input_shape_list[1:]),dtype=tf.float32,trainable=False)
    self.f_max = self.input_shape_list[-1]
    self.t_max = self.input_shape_list[-2]

  def call(self,x,training=None):
    def make_gap(x_i):
      self.mask.assign(tf.ones(self.input_shape_list[1:]))
      n_fgaps = tf.random.uniform(minval=self.f_gaps[0],maxval=self.f_gaps[1],shape=[1,],dtype=tf.int32)
      n_tgaps = tf.random.uniform(minval=self.t_gaps[0],maxval=self.t_gaps[1],shape=[1,],dtype=tf.int32)
      f_lens = tf.random.uniform(minval=self.f_gap_size[0],maxval=self.f_gap_size[1],dtype=tf.int32,shape=n_fgaps)
      t_lens = tf.random.uniform(minval=self.t_gap_size[0],maxval=self.t_gap_size[1],dtype=tf.int32,shape=n_tgaps)
      f_starts = tf.random.uniform(minval=0,maxval=self.f_max-tf.reduce_max(f_lens),dtype=tf.int32,shape=n_fgaps)
      t_starts = tf.random.uniform(minval=0,maxval=self.t_max-tf.reduce_max(t_lens),dtype=tf.int32,shape=n_tgaps)

      def apply_f_gaps():
        #Frequency gaps
        indexs_f_gaps_fdim = tf.ragged.range(f_starts,f_starts+f_lens)
        indexs_f_gaps_fdim = indexs_f_gaps_fdim.merge_dims(0,1)
        indexs_f_gaps_fdim = tf.repeat(indexs_f_gaps_fdim,self.t_max)
        indexs_f_gaps_tdim = tf.tile(tf.range(self.t_max),[tf.reduce_sum(f_lens)])
        indexs_f = tf.transpose(tf.stack([indexs_f_gaps_tdim,indexs_f_gaps_fdim]))
        self.mask.scatter_nd_update(indices = indexs_f,updates=tf.zeros(tf.shape(indexs_f)[0],dtype=tf.float32))

      def apply_t_gaps():
        #Time gaps
        indexs_t_gaps_tdim = tf.ragged.range(t_starts,t_starts+t_lens)
        indexs_t_gaps_tdim = indexs_t_gaps_tdim.merge_dims(0,1)
        indexs_t_gaps_tdim = tf.repeat(indexs_t_gaps_tdim,self.f_max)
        indexs_t_gaps_fdim = tf.tile(tf.range(self.f_max),[tf.reduce_sum(t_lens)])
        indexs_t = tf.transpose(tf.stack([indexs_t_gaps_tdim,indexs_t_gaps_fdim]))
        self.mask.scatter_nd_update(indices = indexs_t,updates=tf.zeros(tf.shape(indexs_t)[0],dtype=tf.float32))

      def return_unchanged_mask():
          pass

      tf.cond(n_fgaps>0,apply_f_gaps,return_unchanged_mask)
      tf.cond(n_tgaps>0,apply_t_gaps,return_unchanged_mask)

      return x_i*self.mask

    if training:
      return tf.map_fn(elems = x,fn = make_gap)
    else:
      return x