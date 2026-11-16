add_executable(ninfer
  cli/main.cpp
  cli/options.cpp)
ninfer_internal_includes(ninfer)
target_link_libraries(ninfer PRIVATE
  ninfer_engine
  ninfer_product_logging
  ninfer_product_prompt_input)

add_executable(ninfer-serve serve/main.cpp)
ninfer_internal_includes(ninfer-serve)
target_link_libraries(ninfer-serve PRIVATE ninfer_serve ninfer_product_logging)

add_executable(ninfer-perplexity
  perplexity/main.cpp
  perplexity/corpus.cpp
  perplexity/evaluation.cpp)
ninfer_internal_includes(ninfer-perplexity)
target_link_libraries(ninfer-perplexity PRIVATE
  ninfer_engine ninfer_product_logging ninfer::json)

# Prefill-only hidden-state export over HTTP (CausalScoring purpose).
add_executable(ninfer-hidden-serve hidden_serve/main.cpp)
ninfer_internal_includes(ninfer-hidden-serve)
target_link_libraries(ninfer-hidden-serve PRIVATE
  ninfer_engine ninfer::json ninfer::httplib Threads::Threads)

# Decision-head host reference. --selftest replays stored hidden-state dumps through the FP32 head, and
# --encode-check rebuilds the same dumps' token ids with the engine's BPE frontend. Both run on the CPU:
# the parity checks are not confounded by the CUDA path, and the card stays off.
# --serve (serve.cpp) loads a real artifact and answers POST /v1/decide from either head, so the
# decision latency is measured in the process that would serve it.
add_executable(ninfer-decide-serve
  decide_serve/main.cpp
  decide_serve/serve.cpp)
ninfer_internal_includes(ninfer-decide-serve)
target_link_libraries(ninfer-decide-serve PRIVATE
  ninfer_joint_head ninfer_engine ninfer::json ninfer::httplib Threads::Threads)
