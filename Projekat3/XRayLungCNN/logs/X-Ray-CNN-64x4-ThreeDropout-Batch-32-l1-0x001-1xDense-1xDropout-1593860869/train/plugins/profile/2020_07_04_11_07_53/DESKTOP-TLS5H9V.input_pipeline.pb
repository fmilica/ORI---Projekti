		��g�Av@	��g�Av@!	��g�Av@	�� ��@�?�� ��@�?!�� ��@�?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$	��g�Av@�=�U��?A���S=v@Y���Q��?*	�����yZ@2F
Iterator::Model�V-�?!�$x%��@@)F%u��?1q����8@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat6�;Nё�?!���yX:@)a��+e�?14C�k7@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate?W[���?!cf�x�<@)$����ۗ?1�{ 6@:Preprocessing2S
Iterator::Model::ParallelMap�&S��?!�N�/!@)�&S��?1�N�/!@:Preprocessing2X
!Iterator::Model::ParallelMap::Zip/n���?!��Cm��P@)� �	�?1��6��@:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicelxz�,C|?!_��e�@)lxz�,C|?1_��e�@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap�5�;Nѡ?!z��w;n@@)HP�s�r?1?��C�_@:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensora��+ei?!4C�k@)a��+ei?14C�k@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�=�U��?�=�U��?!�=�U��?      ��!       "      ��!       *      ��!       2	���S=v@���S=v@!���S=v@:      ��!       B      ��!       J	���Q��?���Q��?!���Q��?R      ��!       Z	���Q��?���Q��?!���Q��?JCPU_ONLY