# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: object_detection/protos/train.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from vision.object_detection.protos import optimizer_pb2 as object__detection_dot_protos_dot_optimizer__pb2
from vision.object_detection.protos import preprocessor_pb2 as object__detection_dot_protos_dot_preprocessor__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='object_detection/protos/train.proto',
  package='object_detection.protos',
  syntax='proto2',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n#object_detection/protos/train.proto\x12\x17object_detection.protos\x1a\'object_detection/protos/optimizer.proto\x1a*object_detection/protos/preprocessor.proto\"\xb1\t\n\x0bTrainConfig\x12\x16\n\nbatch_size\x18\x01 \x01(\r:\x02\x33\x32\x12M\n\x19\x64\x61ta_augmentation_options\x18\x02 \x03(\x0b\x32*.object_detection.protos.PreprocessingStep\x12\x1c\n\rsync_replicas\x18\x03 \x01(\x08:\x05\x66\x61lse\x12,\n\x1dkeep_checkpoint_every_n_hours\x18\x04 \x01(\x02:\x05\x31\x30\x30\x30\x30\x12\x35\n\toptimizer\x18\x05 \x01(\x0b\x32\".object_detection.protos.Optimizer\x12$\n\x19gradient_clipping_by_norm\x18\x06 \x01(\x02:\x01\x30\x12\x1e\n\x14\x66ine_tune_checkpoint\x18\x07 \x01(\t:\x00\x12#\n\x19\x66ine_tune_checkpoint_type\x18\x16 \x01(\t:\x00\x12T\n\x1c\x66ine_tune_checkpoint_version\x18\x1c \x01(\x0e\x32*.object_detection.protos.CheckpointVersion:\x02V1\x12,\n\x19\x66rom_detection_checkpoint\x18\x08 \x01(\x08:\x05\x66\x61lseB\x02\x18\x01\x12\x31\n\"load_all_detection_checkpoint_vars\x18\x13 \x01(\x08:\x05\x66\x61lse\x12\x38\n*run_fine_tune_checkpoint_dummy_computation\x18\x1e \x01(\x08:\x04true\x12\x14\n\tnum_steps\x18\t \x01(\r:\x01\x30\x12\x1f\n\x13startup_delay_steps\x18\n \x01(\x02:\x02\x31\x35\x12\x1f\n\x14\x62ias_grad_multiplier\x18\x0b \x01(\x02:\x01\x30\x12\"\n\x1aupdate_trainable_variables\x18\x19 \x03(\t\x12\x18\n\x10\x66reeze_variables\x18\x0c \x03(\t\x12 \n\x15replicas_to_aggregate\x18\r \x01(\x05:\x01\x31\x12%\n\x14\x62\x61tch_queue_capacity\x18\x0e \x01(\x05:\x03\x31\x35\x30\x42\x02\x18\x01\x12&\n\x17num_batch_queue_threads\x18\x0f \x01(\x05:\x01\x38\x42\x02\x18\x01\x12&\n\x17prefetch_queue_capacity\x18\x10 \x01(\x05:\x01\x35\x42\x02\x18\x01\x12)\n\x1amerge_multiple_label_boxes\x18\x11 \x01(\x08:\x05\x66\x61lse\x12$\n\x15use_multiclass_scores\x18\x18 \x01(\x08:\x05\x66\x61lse\x12%\n\x17\x61\x64\x64_regularization_loss\x18\x12 \x01(\x08:\x04true\x12$\n\x13max_number_of_boxes\x18\x14 \x01(\x05:\x03\x31\x30\x30\x42\x02\x18\x01\x12\'\n\x19unpad_groundtruth_tensors\x18\x15 \x01(\x08:\x04true\x12%\n\x16retain_original_images\x18\x17 \x01(\x08:\x05\x66\x61lse\x12\x1b\n\x0cuse_bfloat16\x18\x1a \x01(\x08:\x05\x66\x61lse\x12\"\n\x13summarize_gradients\x18\x1b \x01(\x08:\x05\x66\x61lse*0\n\x11\x43heckpointVersion\x12\x0b\n\x07UNKNOWN\x10\x00\x12\x06\n\x02V1\x10\x01\x12\x06\n\x02V2\x10\x02'
  ,
  dependencies=[object__detection_dot_protos_dot_optimizer__pb2.DESCRIPTOR,object__detection_dot_protos_dot_preprocessor__pb2.DESCRIPTOR,])

_CHECKPOINTVERSION = _descriptor.EnumDescriptor(
  name='CheckpointVersion',
  full_name='object_detection.protos.CheckpointVersion',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='UNKNOWN', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='V1', index=1, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='V2', index=2, number=2,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1353,
  serialized_end=1401,
)
_sym_db.RegisterEnumDescriptor(_CHECKPOINTVERSION)

CheckpointVersion = enum_type_wrapper.EnumTypeWrapper(_CHECKPOINTVERSION)
UNKNOWN = 0
V1 = 1
V2 = 2



_TRAINCONFIG = _descriptor.Descriptor(
  name='TrainConfig',
  full_name='object_detection.protos.TrainConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='batch_size', full_name='object_detection.protos.TrainConfig.batch_size', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=32,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='data_augmentation_options', full_name='object_detection.protos.TrainConfig.data_augmentation_options', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='sync_replicas', full_name='object_detection.protos.TrainConfig.sync_replicas', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='keep_checkpoint_every_n_hours', full_name='object_detection.protos.TrainConfig.keep_checkpoint_every_n_hours', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(10000),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='optimizer', full_name='object_detection.protos.TrainConfig.optimizer', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='gradient_clipping_by_norm', full_name='object_detection.protos.TrainConfig.gradient_clipping_by_norm', index=5,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='fine_tune_checkpoint', full_name='object_detection.protos.TrainConfig.fine_tune_checkpoint', index=6,
      number=7, type=9, cpp_type=9, label=1,
      has_default_value=True, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='fine_tune_checkpoint_type', full_name='object_detection.protos.TrainConfig.fine_tune_checkpoint_type', index=7,
      number=22, type=9, cpp_type=9, label=1,
      has_default_value=True, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='fine_tune_checkpoint_version', full_name='object_detection.protos.TrainConfig.fine_tune_checkpoint_version', index=8,
      number=28, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='from_detection_checkpoint', full_name='object_detection.protos.TrainConfig.from_detection_checkpoint', index=9,
      number=8, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=b'\030\001', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='load_all_detection_checkpoint_vars', full_name='object_detection.protos.TrainConfig.load_all_detection_checkpoint_vars', index=10,
      number=19, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='run_fine_tune_checkpoint_dummy_computation', full_name='object_detection.protos.TrainConfig.run_fine_tune_checkpoint_dummy_computation', index=11,
      number=30, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='num_steps', full_name='object_detection.protos.TrainConfig.num_steps', index=12,
      number=9, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='startup_delay_steps', full_name='object_detection.protos.TrainConfig.startup_delay_steps', index=13,
      number=10, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(15),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='bias_grad_multiplier', full_name='object_detection.protos.TrainConfig.bias_grad_multiplier', index=14,
      number=11, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='update_trainable_variables', full_name='object_detection.protos.TrainConfig.update_trainable_variables', index=15,
      number=25, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='freeze_variables', full_name='object_detection.protos.TrainConfig.freeze_variables', index=16,
      number=12, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='replicas_to_aggregate', full_name='object_detection.protos.TrainConfig.replicas_to_aggregate', index=17,
      number=13, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='batch_queue_capacity', full_name='object_detection.protos.TrainConfig.batch_queue_capacity', index=18,
      number=14, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=150,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=b'\030\001', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='num_batch_queue_threads', full_name='object_detection.protos.TrainConfig.num_batch_queue_threads', index=19,
      number=15, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=8,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=b'\030\001', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='prefetch_queue_capacity', full_name='object_detection.protos.TrainConfig.prefetch_queue_capacity', index=20,
      number=16, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=5,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=b'\030\001', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='merge_multiple_label_boxes', full_name='object_detection.protos.TrainConfig.merge_multiple_label_boxes', index=21,
      number=17, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='use_multiclass_scores', full_name='object_detection.protos.TrainConfig.use_multiclass_scores', index=22,
      number=24, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='add_regularization_loss', full_name='object_detection.protos.TrainConfig.add_regularization_loss', index=23,
      number=18, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='max_number_of_boxes', full_name='object_detection.protos.TrainConfig.max_number_of_boxes', index=24,
      number=20, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=100,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=b'\030\001', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='unpad_groundtruth_tensors', full_name='object_detection.protos.TrainConfig.unpad_groundtruth_tensors', index=25,
      number=21, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='retain_original_images', full_name='object_detection.protos.TrainConfig.retain_original_images', index=26,
      number=23, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='use_bfloat16', full_name='object_detection.protos.TrainConfig.use_bfloat16', index=27,
      number=26, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='summarize_gradients', full_name='object_detection.protos.TrainConfig.summarize_gradients', index=28,
      number=27, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=150,
  serialized_end=1351,
)

_TRAINCONFIG.fields_by_name['data_augmentation_options'].message_type = object__detection_dot_protos_dot_preprocessor__pb2._PREPROCESSINGSTEP
_TRAINCONFIG.fields_by_name['optimizer'].message_type = object__detection_dot_protos_dot_optimizer__pb2._OPTIMIZER
_TRAINCONFIG.fields_by_name['fine_tune_checkpoint_version'].enum_type = _CHECKPOINTVERSION
DESCRIPTOR.message_types_by_name['TrainConfig'] = _TRAINCONFIG
DESCRIPTOR.enum_types_by_name['CheckpointVersion'] = _CHECKPOINTVERSION
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

TrainConfig = _reflection.GeneratedProtocolMessageType('TrainConfig', (_message.Message,), {
  'DESCRIPTOR' : _TRAINCONFIG,
  '__module__' : 'object_detection.protos.train_pb2'
  # @@protoc_insertion_point(class_scope:object_detection.protos.TrainConfig)
  })
_sym_db.RegisterMessage(TrainConfig)


_TRAINCONFIG.fields_by_name['from_detection_checkpoint']._options = None
_TRAINCONFIG.fields_by_name['batch_queue_capacity']._options = None
_TRAINCONFIG.fields_by_name['num_batch_queue_threads']._options = None
_TRAINCONFIG.fields_by_name['prefetch_queue_capacity']._options = None
_TRAINCONFIG.fields_by_name['max_number_of_boxes']._options = None
# @@protoc_insertion_point(module_scope)
