	Ǻ���x@Ǻ���x@!Ǻ���x@	j$L�?j$L�?!j$L�?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$Ǻ���x@�A`��"�?A)��0�x@Y��ׁsF�?*	������X@2F
Iterator::ModelbX9�Ȧ?!�,�#QF@)T㥛� �?1ˏ��C�?@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat�v��/�?!o�E0��<@)p_�Q�?1��:��9@:Preprocessing2S
Iterator::Model::ParallelMap9��v���?!!��0*@)9��v���?1!��0*@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate	�^)ː?!�2ys0@)������?1ˢ;E'@:Preprocessing2X
!Iterator::Model::ParallelMap::Ziplxz�,C�?!S�OܮK@)���_vO~?1��G�f�@:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicea2U0*�s?!O�n�A@)a2U0*�s?1O�n�A@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap�N@aÓ?!S���[3@)�����g?1ˢ;E@:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensorǺ���f?!��ǫ�w@)Ǻ���f?1��ǫ�w@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�A`��"�?�A`��"�?!�A`��"�?      ��!       "      ��!       *      ��!       2	)��0�x@)��0�x@!)��0�x@:      ��!       B      ��!       J	��ׁsF�?��ׁsF�?!��ׁsF�?R      ��!       Z	��ׁsF�?��ׁsF�?!��ׁsF�?JCPU_ONLY