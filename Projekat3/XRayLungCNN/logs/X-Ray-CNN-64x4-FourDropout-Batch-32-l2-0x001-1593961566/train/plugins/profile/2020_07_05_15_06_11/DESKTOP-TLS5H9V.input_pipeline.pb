	�{�Pޅ@�{�Pޅ@!�{�Pޅ@	�3����?�3����?!�3����?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�{�Pޅ@{�G�z�?A�ܵ�|܅@Y o�ŏ�?*	      W@2F
Iterator::Model8��d�`�?!7�"�u�E@)��j+���?1�n0E>?@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeatB>�٬��?!�)�Y7�>@)-C��6�?1к���;@:Preprocessing2S
Iterator::Model::ParallelMap46<�R�?!��L�'@)46<�R�?1��L�'@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate%u��?!�|���/@)M�O��?1h�`�|�%@:Preprocessing2X
!Iterator::Model::ParallelMap::Zip$���~��?!�g�`�|L@)_�Q�{?1|���g@:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceHP�s�r?!E>�S�@)HP�s�r?1E>�S�@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap/n���?!o0E>�3@)�����g?1L�Ϻ�	@:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor��_vOf?!�u�)�Y@)��_vOf?1�u�)�Y@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	{�G�z�?{�G�z�?!{�G�z�?      ��!       "      ��!       *      ��!       2	�ܵ�|܅@�ܵ�|܅@!�ܵ�|܅@:      ��!       B      ��!       J	 o�ŏ�? o�ŏ�?! o�ŏ�?R      ��!       Z	 o�ŏ�? o�ŏ�?! o�ŏ�?JCPU_ONLY