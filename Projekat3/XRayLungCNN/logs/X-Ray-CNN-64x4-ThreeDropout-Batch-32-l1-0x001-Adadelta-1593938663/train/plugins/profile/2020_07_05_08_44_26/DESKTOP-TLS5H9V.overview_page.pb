�	`vOQ{@`vOQ{@!`vOQ{@	���Sw<�?���Sw<�?!���Sw<�?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$`vOQ{@P��n��?A�����M{@Yv�����?*23333SY@)      =2F
Iterator::Model�0�*�?!��I�UD@)�Zd;�?1�:r�>@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat� �	��?!�Ӓ�g>@))\���(�?1�Ì�%;@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate�5�;Nё?!�f<-1@)tF��_�?1}-�~'@:Preprocessing2S
Iterator::Model::ParallelMap'�����?!�x�B %@)'�����?1�x�B %@:Preprocessing2X
!Iterator::Model::ParallelMap::Zipڬ�\mŮ?!�5��M@)/n���?1��Sd�_!@:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�I+�v?!i�h=��@)�I+�v?1i�h=��@:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensorF%u�k?!�}�
@)F%u�k?1�}�
@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMapˡE����?!�\�ʦ<4@)a��+ei?1Ť$S{@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	P��n��?P��n��?!P��n��?      ��!       "      ��!       *      ��!       2	�����M{@�����M{@!�����M{@:      ��!       B      ��!       J	v�����?v�����?!v�����?R      ��!       Z	v�����?v�����?!v�����?JCPU_ONLY2black"�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: 