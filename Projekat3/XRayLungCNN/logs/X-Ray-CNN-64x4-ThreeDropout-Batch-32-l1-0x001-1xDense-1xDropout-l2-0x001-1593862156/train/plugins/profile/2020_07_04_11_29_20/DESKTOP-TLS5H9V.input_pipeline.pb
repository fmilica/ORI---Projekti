	'�WKx@'�WKx@!'�WKx@	�&�*�3�?�&�*�3�?!�&�*�3�?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$'�WKx@:#J{�/�?AA��ǘHx@Y�rh��|�?*	������f@2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat�t�V�?!M��:9'P@)`vOj�?1�N�i1SO@:Preprocessing2F
Iterator::Model��y�):�?!��2_?i3@)�]K�=�?1��YF�-@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate�&S��?!-#����#@)�{�Pk�?1����"@:Preprocessing2S
Iterator::Model::ParallelMap;�O��n�?!mU��@);�O��n�?1mU��@:Preprocessing2X
!Iterator::Model::ParallelMap::Zipz�,C��?!^3(�%T@)��ǘ���?1�ؒ�@:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�g��s�u?!j1S�m@)�g��s�u?1j1S�m@:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor��H�}m?!�#�!h�?)��H�}m?1�#�!h�?:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap�g��s��?!j1S�m'@)�~j�t�h?1�q��,�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	:#J{�/�?:#J{�/�?!:#J{�/�?      ��!       "      ��!       *      ��!       2	A��ǘHx@A��ǘHx@!A��ǘHx@:      ��!       B      ��!       J	�rh��|�?�rh��|�?!�rh��|�?R      ��!       Z	�rh��|�?�rh��|�?!�rh��|�?JCPU_ONLY