	�J�={@�J�={@!�J�={@	�0�cyz�?�0�cyz�?!�0�cyz�?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�J�={@��C�l�?A�����7{@Yb��4�8�?*	�����9a@2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat_)�Ǻ�?!�$����A@)��_vO�?1��U�Y?@:Preprocessing2F
Iterator::Model��+e�?!�T�_^�A@)P�s��?1����9@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::ConcatenateJ+��?!��p!��1@)�J�4�?1e�I	b(@:Preprocessing2S
Iterator::Model::ParallelMap�]K�=�?!�3��M#@)�]K�=�?1�3��M#@:Preprocessing2X
!Iterator::Model::ParallelMap::Zip��ͪ�ն?!����.P@)Ǻ����?1��۰dA @:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice� �	�?!�q.s�Y@)� �	�?1�q.s�Y@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap���B�i�?!4ʏ�5@)��_�Lu?1��sHM0@:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor��ZӼ�t?!�"B��@)��ZӼ�t?1�"B��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��C�l�?��C�l�?!��C�l�?      ��!       "      ��!       *      ��!       2	�����7{@�����7{@!�����7{@:      ��!       B      ��!       J	b��4�8�?b��4�8�?!b��4�8�?R      ��!       Z	b��4�8�?b��4�8�?!b��4�8�?JCPU_ONLY