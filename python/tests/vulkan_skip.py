vulkan_skip = {
    # Segmented matmul too slow
    "TestBlas.test_segmented_mm",
    # Conv2d with large filters is slow
    "TestConv.test_conv2d_large_filter_small_channels",
}
