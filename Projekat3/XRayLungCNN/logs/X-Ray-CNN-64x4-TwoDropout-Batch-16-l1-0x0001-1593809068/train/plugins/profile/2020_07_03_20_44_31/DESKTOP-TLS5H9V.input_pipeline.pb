	y�&1�d@y�&1�d@!y�&1�d@	�x1A��?�x1A��?!�x1A��?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$y�&1�d@�c�ZB�?A�ׁsF�d@Y�St$��?*	�����X@2F
Iterator::Model��\m���?!����aHC@)z6�>W�?1\��2�;@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat2�%䃞?!��vA%�>@)��<,Ԛ?1���0$<;@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate���&�?!��Lp3@)46<�R�?1�*�U?�&@:Preprocessing2S
Iterator::Model::ParallelMap��_�L�?!�Y7�"�%@)��_�L�?1�Y7�"�%@:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�q����?!v�)�Y7 @)�q����?1v�)�Y7 @:Preprocessing2X
!Iterator::Model::ParallelMap::Zip�c�ZB�?!y{��N@)� �	�?1"�w�  @:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor��H�}m?!����@)��H�}m?1����@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap��_vO�?!Ag�bt6@)�����g?1݁���@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�c�ZB�?�c�ZB�?!�c�ZB�?      ��!       "      ��!       *      ��!       2	�ׁsF�d@�ׁsF�d@!�ׁsF�d@:      ��!       B      ��!       J	�St$��?�St$��?!�St$��?R      ��!       Z	�St$��?�St$��?!�St$��?JCPU_ONLY