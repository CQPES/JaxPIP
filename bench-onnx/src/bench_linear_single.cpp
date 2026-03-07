#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <random>
#include <string>
#include <vector>

std::string strip_suffix(std::string s, const std::string &suffix) {
  if (s.length() >= suffix.length() &&
      s.compare(s.length() - suffix.length(), suffix.length(), suffix) == 0) {
    return s.substr(0, s.length() - suffix.length());
  }
  return s;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: ./bench_linear_single.x <model_path>" << std::endl;
    return 1;
  }

  std::string model_path = argv[1];

  std::string filename = model_path.substr(model_path.find_last_of("/\\") + 1);

  std::string base_name = strip_suffix(filename, ".sim.onnx");
  base_name = strip_suffix(base_name, ".onnx");

  size_t last_und = base_name.find_last_of("_");
  std::string system = (last_und != std::string::npos)
                           ? base_name.substr(0, last_und)
                           : base_name;
  std::string order =
      (last_und != std::string::npos) ? base_name.substr(last_und + 1) : "0";

  int n_loops = 1000;
  const int n_warmup = 100;

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "JAXPIP_Benchmark");
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetInterOpNumThreads(1);
  session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

  auto t_start_init = std::chrono::high_resolution_clock::now();
  Ort::Session session(env, model_path.c_str(), session_options);
  auto t_end_init = std::chrono::high_resolution_clock::now();
  double init_time_s =
      std::chrono::duration<double>(t_end_init - t_start_init).count();

  Ort::AllocatorWithDefaultOptions allocator;
  auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
  const char *input_names[] = {input_name_ptr.get()};
  auto out0_name_ptr = session.GetOutputNameAllocated(0, allocator);
  auto out1_name_ptr = session.GetOutputNameAllocated(1, allocator);
  const char *output_names[] = {out0_name_ptr.get(), out1_name_ptr.get()};

  Ort::TypeInfo input_type_info = session.GetInputTypeInfo(0);
  std::vector<int64_t> input_shape =
      input_type_info.GetTensorTypeAndShapeInfo().GetShape();

  std::vector<double> input_values(input_shape[0] * 3, 1.0);
  Ort::MemoryInfo mem_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<double>(
      mem_info, input_values.data(), input_values.size(), input_shape.data(),
      input_shape.size());

  for (int i = 0; i < n_warmup; ++i) {
    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names,
                               &input_tensor, 1, output_names, 2);
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < n_loops; ++i) {
    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names,
                               &input_tensor, 1, output_names, 2);
  }
  auto t2 = std::chrono::high_resolution_clock::now();

  double run_time_s = std::chrono::duration<double>(t2 - t1).count() / n_loops;

  std::cout << "system,order,init,run" << std::endl;
  std::cout << system << "," << order << "," << std::fixed
            << std::setprecision(8) << init_time_s << "," << run_time_s
            << std::endl;

  return 0;
}