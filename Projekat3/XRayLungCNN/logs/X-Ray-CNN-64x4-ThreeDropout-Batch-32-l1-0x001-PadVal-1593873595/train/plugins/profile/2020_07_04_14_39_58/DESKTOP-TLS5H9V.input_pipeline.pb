	���(\&r@���(\&r@!���(\&r@	G�`�`�?G�`�`�?!G�`�`�?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$���(\&r@�6�[ �?A��(#r@Y~��k	��?*	hffff�W@2F
Iterator::Model��ׁsF�?! >���D@)�!��u��?1�8�v=@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat)\���(�?!�9�H�<@)�(��0�?1?�b�r�9@:Preprocessing2S
Iterator::Model::ParallelMap�+e�X�?!Tn�wp�'@)�+e�X�?1Tn�wp�'@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate���&�?!t����3@)46<�R�?1C^�*��&@:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�q����?!��J�gQ @)�q����?1��J�gQ @:Preprocessing2X
!Iterator::Model::ParallelMap::Zipy�&1��?!��eo�IM@)y�&1�|?1��eo�I@:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor�����g?!�t�c�D@)�����g?1�t�c�D@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMapj�t��?!p�zR}6@)Ǻ���f?1�gQ�Sn@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�6�[ �?�6�[ �?!�6�[ �?      ��!       "      ��!       *      ��!       2	��(#r@��(#r@!��(#r@:      ��!       B      ��!       J	~��k	��?~��k	��?!~��k	��?R      ��!       Z	~��k	��?~��k	��?!~��k	��?JCPU_ONLY