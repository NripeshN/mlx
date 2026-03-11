// Copyright © 2024 Apple Inc.

#include "mlx/backend/vulkan/primitives_utils.h"

namespace mlx::core {

namespace {

array reverse_kernel_1d(const array& wt, Stream s) {
  const auto& shape = wt.shape();
  const auto& strides = wt.strides();
  return as_strided(
      wt,
      shape,
      {strides[0], -strides[1], strides[2]},
      static_cast<size_t>((shape[1] - 1) * strides[1]),
      s);
}

array dilate_input_1d(const array& in, int dilation, Stream s) {
  if (dilation == 1) {
    return in;
  }

  auto expanded = expand_dims(in, 2, s);
  auto padded =
      pad(expanded,
          std::vector<std::pair<int, int>>{
              {0, 0}, {0, 0}, {0, dilation - 1}, {0, 0}},
          array(0, in.dtype()),
          "constant",
          s);
  auto reshaped =
      reshape(padded, {in.shape(0), in.shape(1) * dilation, in.shape(2)}, s);
  return slice(
      reshaped,
      {0, 0, 0},
      {in.shape(0), (in.shape(1) - 1) * dilation + 1, in.shape(2)},
      s);
}

bool try_eval_conv1d_vulkan(
    const std::vector<array>& inputs,
    array& out,
    const std::vector<int>& kernel_strides,
    const std::vector<int>& padding_lo,
    const std::vector<int>& padding_hi,
    const std::vector<int>& kernel_dilation,
    const std::vector<int>& input_dilation,
    int groups,
    bool flip,
    Stream s) {
  if (inputs.size() != 2 || inputs[0].ndim() != 3 || inputs[1].ndim() != 3 ||
      out.ndim() != 3) {
    return false;
  }

  if (kernel_strides.size() != 1 || padding_lo.size() != 1 ||
      padding_hi.size() != 1 || kernel_dilation.size() != 1 ||
      input_dilation.size() != 1) {
    return false;
  }

  if (groups <= 0) {
    return false;
  }

  auto in = inputs[0];
  auto wt = inputs[1];

  const int batch = in.shape(0);
  const int in_len = in.shape(1);
  const int in_channels = in.shape(2);
  const int out_channels = wt.shape(0);
  const int kernel = wt.shape(1);
  const int channels_per_group = wt.shape(2);
  const int out_len = out.shape(1);

  if (in_channels != channels_per_group * groups ||
      out_channels % groups != 0) {
    return false;
  }

  if (batch == 0 || out_len == 0 || out_channels == 0) {
    return true;
  }

  in = dilate_input_1d(in, input_dilation[0], s);
  in =
      pad(in,
          std::vector<std::pair<int, int>>{
              {0, 0}, {padding_lo[0], padding_hi[0]}, {0, 0}},
          array(0, in.dtype()),
          "constant",
          s);

  auto wt_work = flip ? reverse_kernel_1d(wt, s) : wt;

  const auto& in_strides = in.strides();
  auto patches = as_strided(
      in,
      {batch, out_len, kernel, in_channels},
      {
          in_strides[0],
          in_strides[1] * kernel_strides[0],
          in_strides[1] * kernel_dilation[0],
          in_strides[2],
      },
      0,
      s);

  const int out_channels_per_group = out_channels / groups;
  std::vector<array> group_outputs;
  group_outputs.reserve(groups);

  for (int g = 0; g < groups; ++g) {
    auto patches_g = slice(
        patches,
        {0, 0, 0, g * channels_per_group},
        {batch, out_len, kernel, (g + 1) * channels_per_group},
        s);
    patches_g =
        reshape(patches_g, {batch * out_len, kernel * channels_per_group}, s);

    auto wt_g = slice(
        wt_work,
        {g * out_channels_per_group, 0, 0},
        {(g + 1) * out_channels_per_group, kernel, channels_per_group},
        s);
    wt_g =
        reshape(wt_g, {out_channels_per_group, kernel * channels_per_group}, s);
    wt_g = transpose(wt_g, {1, 0}, s);

    auto out_g = matmul(patches_g, wt_g, s);
    group_outputs.push_back(
        reshape(out_g, {batch, out_len, out_channels_per_group}, s));
  }

  auto result = group_outputs.size() == 1
      ? group_outputs[0]
      : concatenate(std::move(group_outputs), 2, s);
  eval(result);
  copy_gpu(result, out, CopyType::General, s);
  return true;
}

} // namespace

void Convolution::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (try_eval_conv1d_vulkan(
          inputs,
          out,
          kernel_strides_,
          padding_lo_,
          padding_hi_,
          kernel_dilation_,
          input_dilation_,
          groups_,
          flip_,
          stream())) {
    return;
  }

  eval_cpu_fallback_with_state_on_stream<Convolution>(
      inputs, out, stream(), state());
}

} // namespace mlx::core
