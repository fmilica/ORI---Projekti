	�-���s@�-���s@!�-���s@	2��9��?2��9��?!2��9��?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�-���s@M�O���?Ah��|?s@Yq���h�?*	      X@2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat���Q��?!     @?@)F%u��?1    �;@:Preprocessing2F
Iterator::ModelHP�sע?!�����*C@)}гY���?1�����/;@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate;�O��n�?!     �2@)-C��6�?1������*@:Preprocessing2S
Iterator::Model::ParallelMap'�����?!�����J&@)'�����?1�����J&@:Preprocessing2X
!Iterator::Model::ParallelMap::Zip���_vO�?!VUUUU�N@)ŏ1w-!?1������@:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��_�Lu?!������@)��_�Lu?1������@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap��_vO�?!     �6@)��H�}m?1      @:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor��H�}m?!      @)��H�}m?1      @:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	M�O���?M�O���?!M�O���?      ��!       "      ��!       *      ��!       2	h��|?s@h��|?s@!h��|?s@:      ��!       B      ��!       J	q���h�?q���h�?!q���h�?R      ��!       Z	q���h�?q���h�?!q���h�?JCPU_ONLY