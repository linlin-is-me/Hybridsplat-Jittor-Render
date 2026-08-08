#include "../ops.h"

#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <algorithm>

namespace flashgs {
namespace {

static uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

} // namespace

void sort_gaussian(int num_rendered,
    int width, int height, int block_x, int block_y,
	char* list_sorting_space, size_t sorting_size,
	uint64_t* gaussian_keys_unsorted, uint32_t* gaussian_values_unsorted,
	uint64_t* gaussian_keys_secondary, uint32_t* gaussian_values_secondary,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted, cudaStream_t stream)
{
	dim3 grid((width + block_x - 1) / block_x, (height + block_y - 1) / block_y, 1);
	// First establish a deterministic secondary order. DeviceRadixSort is
	// stable, so the following stable primary-key sort produces the exact
	// lexicographic order (tile-depth key, Gaussian index).
	auto status = cub::DeviceRadixSort::SortPairs(
		list_sorting_space, sorting_size,
		gaussian_values_unsorted, gaussian_values_secondary,
		gaussian_keys_unsorted, gaussian_keys_secondary,
		num_rendered, 0, sizeof(uint32_t) * 8, stream);
    if (status != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(status));
    }
	status = cub::DeviceRadixSort::SortPairs(
		list_sorting_space, sorting_size,
		gaussian_keys_secondary, gaussian_keys_sorted,
		gaussian_values_secondary, gaussian_values_sorted,
		num_rendered, 0, 32 + getHigherMsb(grid.x * grid.y), stream);
    if (status != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

size_t get_sort_buffer_size(int num_rendered, cudaStream_t stream)
{
	size_t secondary_sort_size = 0;
	cub::DeviceRadixSort::SortPairs<uint32_t, uint64_t>(
		nullptr, secondary_sort_size,
		nullptr, nullptr,
		nullptr, nullptr, num_rendered, 0, sizeof(uint32_t) * 8, stream);
    size_t primary_sort_size = 0;
	cub::DeviceRadixSort::SortPairs<uint64_t, uint32_t>(
		nullptr, primary_sort_size,
		nullptr, nullptr,
		nullptr, nullptr, num_rendered, 0, sizeof(uint64_t) * 8, stream);
    return std::max(secondary_sort_size, primary_sort_size);
}

} // namespace flashgs
