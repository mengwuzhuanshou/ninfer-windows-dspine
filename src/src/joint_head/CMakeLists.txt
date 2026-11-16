# The released Clef joint schema head as an FP32 host reference.
#
# Deliberately minimal: the head's weights come from a sidecar, never from a .ninfer artifact, so it
# touches none of the backbone's operator registry, and the encoder takes its tokenizer through an
# interface rather than linking the CUDA side. The only dependency is the header-only JSON library the
# request type is made of. That is what lets `ninfer-decide-serve --selftest` be a CPU-only check on a
# box whose GPU path is under investigation.
add_library(ninfer_joint_head STATIC
  weights.cpp
  head.cpp
  student.cpp
  encoder.cpp
)
ninfer_internal_includes(ninfer_joint_head)
# PUBLIC, not PRIVATE: encoder.h puts nlohmann's ordered_json in the request type callers hand in, so
# the header needs the include path, not just the implementation.
target_link_libraries(ninfer_joint_head PUBLIC ninfer::json)
