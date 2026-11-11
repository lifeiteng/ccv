#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include "3rdparty/dsfmt/dSFMT.h"
#include "3rdparty/sqlite3/sqlite3.h"

TEST_SETUP()
{
	ccv_nnc_init();
}

static const int tmcq_formats[9] = {
	CCV_NNC_QX_8I_ROWWISE_TMCQ_N16, CCV_NNC_QX_8I_ROWWISE_TMCQ_N18, CCV_NNC_QX_8I_ROWWISE_TMCQ_N20,
	CCV_NNC_QX_8I_ROWWISE_TMCQ_N22, CCV_NNC_QX_8I_ROWWISE_TMCQ_N24, CCV_NNC_QX_8I_ROWWISE_TMCQ_N26,
	CCV_NNC_QX_8I_ROWWISE_TMCQ_N28, CCV_NNC_QX_8I_ROWWISE_TMCQ_N30, CCV_NNC_QX_8I_ROWWISE_TMCQ_N32,
};

// Independent reference of the code value: q8 = min((bytesum(cw * 0x83DCD12D) >> 2) - 127, 127).
static int _test_tmcq_q8(const unsigned cw)
{
	const uint32_t x = (uint32_t)((uint64_t)cw * 0x83DCD12Dull);
	int bs = 0, i;
	for (i = 0; i < 4; i++)
		bs += (x >> (8 * i)) & 0xff;
	const int q8 = (bs >> 2) - 127;
	return q8 > 127 ? 127 : q8;
}

static int _test_tmcq_bit(const uint8_t* const data, const size_t bit)
{
	return (data[bit / 8] >> (bit % 8)) & 1;
}

// cw_t = sum_j bit[(E_t - 15 + j) mod T] << j with E_t = (t + 1) n / 16, straight from the definition.
static void _test_tmcq_decode(const uint8_t* const payload, const int rows, const int cols, const int n, int* const q8)
{
	const int T = 16 * n;
	const int tile_cols = cols / 16;
	int g, t, j;
	for (g = 0; g < rows / 16 * tile_cols; g++)
	{
		const int tr = g / tile_cols, tc = g % tile_cols;
		for (t = 0; t < 256; t++)
		{
			const int end = (t + 1) * n / 16;
			unsigned cw = 0;
			for (j = 0; j < 15; j++)
			{
				const int p = ((end - 15 + j) % T + T) % T;
				cw |= (unsigned)_test_tmcq_bit(payload, (size_t)g * T + p) << j;
			}
			q8[(tr * 16 + t / 16) * cols + tc * 16 + t % 16] = _test_tmcq_q8(cw);
		}
	}
}

static size_t _test_tmcq_scale_offset(const int rows, const int cols, const int n)
{
	const size_t payload = ((size_t)rows / 16 * (cols / 16) * 16 * n + 7) / 8;
	return (payload + 127) & -128;
}

static double _test_tmcq_load_scale(const uint8_t* const scales, const int datatype, const int i)
{
	float v;
	if (datatype == CCV_16F)
	{
		ccv_half_precision_to_float((const uint16_t*)scales + i, &v, 1);
		return v;
	} else if (datatype == CCV_16BF) {
		ccv_bfloat_to_float((const uint16_t*)scales + i, &v, 1);
		return v;
	} else if (datatype == CCV_32F)
		return ((const float*)scales)[i];
	return ((const double*)scales)[i];
}

static void _test_tmcq_to_double(const void* const data, const int datatype, const int count, double* const out)
{
	int i;
	for (i = 0; i < count; i++)
	{
		float v;
		if (datatype == CCV_16F)
			ccv_half_precision_to_float((const uint16_t*)data + i, &v, 1), out[i] = v;
		else if (datatype == CCV_16BF)
			ccv_bfloat_to_float((const uint16_t*)data + i, &v, 1), out[i] = v;
		else if (datatype == CCV_32F)
			out[i] = ((const float*)data)[i];
		else
			out[i] = ((const double*)data)[i];
	}
}

static void _test_tmcq_from_float(const float* const values, const int datatype, const int count, void* const out)
{
	int i;
	if (datatype == CCV_16F)
		ccv_float_to_half_precision(values, (uint16_t*)out, count);
	else if (datatype == CCV_16BF)
		ccv_float_to_bfloat(values, (uint16_t*)out, count);
	else if (datatype == CCV_32F)
		memcpy(out, values, sizeof(float) * count);
	else
		for (i = 0; i < count; i++)
			((double*)out)[i] = values[i];
}

TEST_CASE("TMCQ code values follow the multiply-codebook definition")
{
	REQUIRE_EQ(_test_tmcq_q8(0), -127, "zero state has byte sum 0");
	// 0x7fff * 0x83DCD12D = 0x41EDE4B9AED3 -> low 32 bits 0xE4B9AED3, bytes e4 + b9 + ae + d3 = 798.
	REQUIRE_EQ(_test_tmcq_q8(0x7fff), (798 >> 2) - 127, "all-ones state");
	REQUIRE_EQ(_test_tmcq_q8(1), ((0x83 + 0xdc + 0xd1 + 0x2d) >> 2) - 127, "unit state is the multiplier bytes");
	int s, min_q = 127, max_q = -127;
	for (s = 0; s < 32768; s++)
	{
		const int q = _test_tmcq_q8(s);
		min_q = ccv_min(min_q, q);
		max_q = ccv_max(max_q, q);
	}
	REQUIRE(min_q >= -127 && max_q <= 127, "codes stay in the int8 contract");
	REQUIRE_EQ(ccv_nnc_8i_rowwise_x_tmcq_rate(CCV_NNC_QX_8I_ROWWISE_TMCQ_N16), 16, "lowest rate");
	REQUIRE_EQ(ccv_nnc_8i_rowwise_x_tmcq_rate(CCV_NNC_QX_8I_ROWWISE_TMCQ_N32), 32, "highest rate");
	REQUIRE_EQ(CCV_NNC_QX_8I_ROWWISE_TMCQ_N16, 13, "codec ids follow IQ1_M");
	REQUIRE_EQ(CCV_NNC_QX_8I_ROWWISE_TMCQ_N32, 21, "codec ids are contiguous");
}

TEST_CASE("TMCQ CPU decode of fixed payloads writes tiles back in row-major order")
{
	enum { rows = 32, cols = 512, count = rows * cols };
	int f;
	for (f = 0; f < 9; f++)
	{
		const int n = ccv_nnc_8i_rowwise_x_tmcq_rate(tmcq_formats[f]);
		const size_t scale_offset = _test_tmcq_scale_offset(rows, cols, n);
		const size_t size = scale_offset + rows * sizeof(float);
		REQUIRE_EQ(ccv_nnc_8i_rowwise_x_data_size(tmcq_formats[f], CCV_32F, count, cols), size, "size is the tile payload plus aligned row scales");
		REQUIRE_EQ(ccv_nnc_tensor_data_size_without_padding(ccv_nnc_tensor_8i_rowwise_x(CPU_TENSOR_NHWC(32F, rows, cols), tmcq_formats[f])), size, "tensor allocation agrees");
		uint8_t* const packed = cccalloc(size, 1);
		int i;
		// A structured payload: tile g's bytes are (g * 37 + byte * 11) ^ 0x5a.
		const size_t tile_bytes = 2 * n;
		for (i = 0; i < (int)(scale_offset); i++)
			packed[i] = i < rows / 16 * cols / 16 * (int)tile_bytes ? (uint8_t)(((i / tile_bytes) * 37 + (i % tile_bytes) * 11) ^ 0x5a) : 0;
		for (i = 0; i < rows; i++)
			((float*)(packed + scale_offset))[i] = (i % 2 ? -1 : 1) * (i + 1) / 64.f;
		int* const q8 = ccmalloc(sizeof(int) * count);
		_test_tmcq_decode(packed, rows, cols, n, q8);
		float* const expected = ccmalloc(sizeof(float) * count);
		float* const actual = ccmalloc(sizeof(float) * count);
		for (i = 0; i < count; i++)
			expected[i] = (float)((double)((float*)(packed + scale_offset))[i / cols] * q8[i]);
		ccv_nnc_dequantize_8i_rowwise_x(packed, CCV_32F, CCV_TENSOR_CPU_MEMORY, size, cols, tmcq_formats[f], actual, count);
		REQUIRE_ARRAY_EQ(float, expected, actual, count, "tail-biting windows, codebook and tile order match for n = %d", n);
		// All-zero tiles decode to -127 everywhere.
		memset(packed, 0, scale_offset);
		ccv_nnc_dequantize_8i_rowwise_x(packed, CCV_32F, CCV_TENSOR_CPU_MEMORY, size, cols, tmcq_formats[f], actual, count);
		for (i = 0; i < count; i++)
			expected[i] = (float)((double)((float*)(packed + scale_offset))[i / cols] * -127);
		REQUIRE_ARRAY_EQ(float, expected, actual, count, "zero payload is the -127 code");
		ccfree(actual);
		ccfree(expected);
		ccfree(q8);
		ccfree(packed);
	}
}

TEST_CASE("TMCQ CPU decode of random payloads matches the definition bit for bit in every precision")
{
	enum { rows = 48, cols = 768, count = rows * cols };
	const int datatypes[4] = {CCV_16F, CCV_16BF, CCV_32F, CCV_64F};
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 1013);
	int f, d, i;
	int* const q8 = ccmalloc(sizeof(int) * count);
	uint8_t* const decoded = ccmalloc(sizeof(double) * count);
	uint8_t* const expected = ccmalloc(sizeof(double) * count);
	for (f = 0; f < 9; f++)
	{
		const int n = ccv_nnc_8i_rowwise_x_tmcq_rate(tmcq_formats[f]);
		const size_t scale_offset = _test_tmcq_scale_offset(rows, cols, n);
		for (d = 0; d < 4; d++)
		{
			const size_t size = ccv_nnc_8i_rowwise_x_data_size(tmcq_formats[f], datatypes[d], count, cols);
			REQUIRE_EQ(size, scale_offset + rows * CCV_GET_DATA_TYPE_SIZE(datatypes[d]), "trailing scales use the source precision");
			uint8_t* const packed = ccmalloc(size);
			for (i = 0; i < (int)scale_offset; i++)
				packed[i] = (uint8_t)(dsfmt_genrand_uint32(&dsfmt) & 0xff);
			float scale_f[rows];
			for (i = 0; i < rows; i++)
				scale_f[i] = (float)(dsfmt_genrand_open_close(&dsfmt) - 0.5) / 8;
			_test_tmcq_from_float(scale_f, datatypes[d], rows, packed + scale_offset);
			_test_tmcq_decode(packed, rows, cols, n, q8);
			for (i = 0; i < count; i++)
			{
				const double v = _test_tmcq_load_scale(packed + scale_offset, datatypes[d], i / cols) * q8[i];
				const float vf = (float)v;
				if (datatypes[d] == CCV_64F)
					((double*)expected)[i] = v;
				else
					_test_tmcq_from_float(&vf, datatypes[d], 1, expected + i * CCV_GET_DATA_TYPE_SIZE(datatypes[d]));
			}
			ccv_nnc_dequantize_8i_rowwise_x(packed, datatypes[d], CCV_TENSOR_CPU_MEMORY, size, cols, tmcq_formats[f], decoded, count);
			REQUIRE_ARRAY_EQ(uint8_t, expected, decoded, count * CCV_GET_DATA_TYPE_SIZE(datatypes[d]), "random payload decodes exactly for n = %d, datatype %d", n, d);
			ccfree(packed);
		}
	}
	ccfree(expected);
	ccfree(decoded);
	ccfree(q8);
}

static void _test_tmcq_values(dsfmt_t* const dsfmt, float* const values, const int rows, const int cols)
{
	int i, j;
	for (i = 0; i < rows; i++)
	{
		const double row_scale = 0.02 * (1 + i % 5);
		for (j = 0; j < cols; j++)
		{
			// Sum of uniforms, a rough Gaussian, plus a few outliers.
			double v = 0;
			int k;
			for (k = 0; k < 4; k++)
				v += dsfmt_genrand_open_close(dsfmt) - 0.5;
			if ((i * 131 + j * 17) % 251 == 0)
				v *= 6;
			values[i * cols + j] = (float)(v * row_scale);
		}
	}
}

static double _test_tmcq_randn(dsfmt_t* const dsfmt)
{
	const double u = dsfmt_genrand_open_close(dsfmt);
	const double v = dsfmt_genrand_open_close(dsfmt);
	return sqrt(-2 * log(u)) * cos(2 * M_PI * v);
}

static double _test_tmcq_weighted_sse(const float* const x, const float* const xhat, const int rows, const int cols, const float* const imatrix)
{
	double sse = 0;
	int i, j;
	for (i = 0; i < rows; i++)
		for (j = 0; j < cols; j++)
		{
			const double d = (double)x[i * cols + j] - xhat[i * cols + j];
			sse += (imatrix ? imatrix[j] : 1) * d * d;
		}
	return sse;
}

static double _test_tmcq_hessian_error(const float* const x, const float* const xhat, const int rows, const int cols, const float* const hessian)
{
	double total = 0;
	double* const e = ccmalloc(sizeof(double) * cols);
	int i, j, k;
	for (i = 0; i < rows; i++)
	{
		for (j = 0; j < cols; j++)
			e[j] = (double)x[i * cols + j] - xhat[i * cols + j];
		for (j = 0; j < cols; j++)
		{
			double v = 0;
			for (k = 0; k < cols; k++)
				v += hessian[j * cols + k] * e[k];
			total += e[j] * v;
		}
	}
	ccfree(e);
	return total;
}

// Nearest baseline: walk each tile once and greedily take the fresh bits whose code is closest to the
// target (starting from state 0), keep the decoded codes, and give each row its optimal weighted scale.
static void _test_tmcq_nearest(const float* const x, const int rows, const int cols, const int n, const float* const imatrix, const float* const hessian, float* const xhat)
{
	const int T = 16 * n;
	const size_t scale_offset = _test_tmcq_scale_offset(rows, cols, n);
	uint8_t* const payload = cccalloc(scale_offset, 1);
	double cb = 0;
	int s;
	for (s = 0; s < 32768; s++)
		cb += (double)_test_tmcq_q8(s) * _test_tmcq_q8(s);
	cb = sqrt(cb / 32768);
	double* const scale = ccmalloc(sizeof(double) * rows);
	int i, j, g, t;
	for (i = 0; i < rows; i++)
	{
		double sum = 0, wsum = 0;
		for (j = 0; j < cols; j++)
		{
			const double w = hessian ? hessian[j * cols + j] : imatrix ? imatrix[j] : 1;
			sum += w * x[i * cols + j] * x[i * cols + j];
			wsum += w;
		}
		scale[i] = sqrt(sum / wsum) / cb;
	}
	const int tile_cols = cols / 16;
	for (g = 0; g < rows / 16 * tile_cols; g++)
	{
		const int tr = g / tile_cols, tc = g % tile_cols;
		unsigned state = 0;
		int prev_end = 0;
		for (t = 0; t < 256; t++)
		{
			const int r = tr * 16 + t / 16;
			const double y = scale[r] > 0 ? x[r * cols + tc * 16 + t % 16] / scale[r] : 0;
			const int end = (t + 1) * n / 16, f = end - prev_end;
			unsigned best = 0;
			double best_d = DBL_MAX;
			unsigned h;
			for (h = 0; h < (1u << f); h++)
			{
				const unsigned next = (state >> f) | (h << (15 - f));
				const double d = fabs(y - _test_tmcq_q8(next));
				if (d < best_d)
					best_d = d, best = next;
			}
			state = best;
			int p;
			for (p = prev_end; p < end; p++)
				if ((state >> (14 - (end - 1 - p))) & 1)
					payload[((size_t)g * T + p) / 8] |= 1 << (((size_t)g * T + p) % 8);
			prev_end = end;
		}
	}
	int* const q8 = ccmalloc(sizeof(int) * rows * cols);
	_test_tmcq_decode(payload, rows, cols, n, q8);
	for (i = 0; i < rows; i++)
	{
		double num = 0, den = 0;
		for (j = 0; j < cols; j++)
		{
			double hx = 0, hq = 0;
			if (hessian)
			{
				int k;
				for (k = 0; k < cols; k++)
					hx += hessian[j * cols + k] * x[i * cols + k], hq += hessian[j * cols + k] * q8[i * cols + k];
			} else {
				const double w = imatrix ? imatrix[j] : 1;
				hx = w * x[i * cols + j];
				hq = w * q8[i * cols + j];
			}
			num += q8[i * cols + j] * hx;
			den += q8[i * cols + j] * hq;
		}
		const double sr = den > 0 ? num / den : 0;
		for (j = 0; j < cols; j++)
			xhat[i * cols + j] = (float)(sr * q8[i * cols + j]);
	}
	ccfree(q8);
	ccfree(scale);
	ccfree(payload);
}

TEST_CASE("TMCQ quantize then dequantize stays within the rate's error bound")
{
	enum { rows = 32, cols = 512, count = rows * cols };
	// Upper bounds on sum (x - x_hat)^2 / sum x^2 for rates n = 16 .. 32 (n / 16 bits per weight).
	const double bounds[9] = {0.30, 0.27, 0.235, 0.21, 0.185, 0.165, 0.15, 0.135, 0.125};
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 2027);
	float* const values = ccmalloc(sizeof(float) * count);
	float* const decoded = ccmalloc(sizeof(float) * count);
	_test_tmcq_values(&dsfmt, values, rows, cols);
	double energy = 0;
	int i, f;
	for (i = 0; i < count; i++)
		energy += (double)values[i] * values[i];
	double previous = DBL_MAX;
	for (f = 0; f < 9; f++)
	{
		const size_t size = ccv_nnc_8i_rowwise_x_data_size(tmcq_formats[f], CCV_32F, count, cols);
		uint8_t* const packed = ccmalloc(size + 16);
		memset(packed, 0xa5, size + 16);
		REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, count, cols, tmcq_formats[f], 0, 0, 0, 0, packed, size - 1), 0, "short output is rejected");
		for (i = 0; i < size + 16; i++)
			REQUIRE_EQ(packed[i], 0xa5, "rejected output stays unchanged");
		REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, count, cols, tmcq_formats[f], 0, 0, 0, 0, packed, size), size, "quantizer returns exact size");
		for (i = 0; i < 16; i++)
			REQUIRE_EQ(packed[size + i], 0xa5, "quantizer preserves output guard bytes");
		ccv_nnc_dequantize_8i_rowwise_x(packed, CCV_32F, CCV_TENSOR_CPU_MEMORY, size, cols, tmcq_formats[f], decoded, count);
		const double relative = _test_tmcq_weighted_sse(values, decoded, rows, cols, 0) / energy;
		const int n = ccv_nnc_8i_rowwise_x_tmcq_rate(tmcq_formats[f]);
		float* const nearest = ccmalloc(sizeof(float) * count);
		_test_tmcq_nearest(values, rows, cols, n, 0, 0, nearest);
		const double nearest_relative = _test_tmcq_weighted_sse(values, nearest, rows, cols, 0) / energy;
		ccfree(nearest);
		printf("\n\tTMCQ n = %d relative error %.6f (greedy nearest %.6f)", n, relative, nearest_relative);
		REQUIRE(relative < bounds[f], "n = %d relative error %g within %g", n, relative, bounds[f]);
		REQUIRE(relative < nearest_relative, "trellis beats the greedy nearest walk for n = %d", n);
		REQUIRE(relative < previous, "error falls as the rate rises");
		previous = relative;
		ccfree(packed);
	}
	ccfree(decoded);
	ccfree(values);
}

TEST_CASE("TMCQ imatrix and Hessian paths beat the nearest baseline on their objectives")
{
	enum { rows = 32, cols = 256, count = rows * cols, samples = 1024 };
	const int formats[3] = {CCV_NNC_QX_8I_ROWWISE_TMCQ_N16, CCV_NNC_QX_8I_ROWWISE_TMCQ_N22, CCV_NNC_QX_8I_ROWWISE_TMCQ_N32};
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 77);
	float* const values = ccmalloc(sizeof(float) * count);
	float* const decoded = ccmalloc(sizeof(float) * count);
	float* const nearest = ccmalloc(sizeof(float) * count);
	float* const hessian = ccmalloc(sizeof(float) * cols * cols);
	float imatrix[cols];
	_test_tmcq_values(&dsfmt, values, rows, cols);
	// Correlated activations: an AR(0.9) Gaussian walk along the channels with log-normal channel gains.
	double* const a = ccmalloc(sizeof(double) * samples * cols);
	double sigma[cols];
	int i, j, k, f;
	for (j = 0; j < cols; j++)
		sigma[j] = exp(0.5 * _test_tmcq_randn(&dsfmt));
	for (i = 0; i < samples; i++)
	{
		double v = _test_tmcq_randn(&dsfmt);
		for (j = 0; j < cols; j++)
		{
			if (j > 0)
				v = 0.9 * v + sqrt(1 - 0.81) * _test_tmcq_randn(&dsfmt);
			a[i * cols + j] = v * sigma[j];
		}
	}
	for (j = 0; j < cols; j++)
		for (k = 0; k < cols; k++)
		{
			double v = 0;
			for (i = 0; i < samples; i++)
				v += a[i * cols + j] * a[i * cols + k];
			hessian[j * cols + k] = (float)(v / samples);
		}
	for (j = 0; j < cols; j++)
		imatrix[j] = hessian[j * cols + j];
	ccfree(a);
	for (f = 0; f < 3; f++)
	{
		const int n = ccv_nnc_8i_rowwise_x_tmcq_rate(formats[f]);
		const size_t size = ccv_nnc_8i_rowwise_x_data_size(formats[f], CCV_32F, count, cols);
		uint8_t* const packed = ccmalloc(size);
		REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, count, cols, formats[f], 0, 0, 0, 0, packed, size), size, "plain quantization succeeds");
		ccv_nnc_dequantize_8i_rowwise_x(packed, CCV_32F, CCV_TENSOR_CPU_MEMORY, size, cols, formats[f], decoded, count);
		const double plain_w = _test_tmcq_weighted_sse(values, decoded, rows, cols, imatrix);
		const double plain_h = _test_tmcq_hessian_error(values, decoded, rows, cols, hessian);
		REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, count, cols, formats[f], imatrix, cols, 0, 0, packed, size), size, "imatrix quantization succeeds");
		ccv_nnc_dequantize_8i_rowwise_x(packed, CCV_32F, CCV_TENSOR_CPU_MEMORY, size, cols, formats[f], decoded, count);
		const double imatrix_w = _test_tmcq_weighted_sse(values, decoded, rows, cols, imatrix);
		const double imatrix_h = _test_tmcq_hessian_error(values, decoded, rows, cols, hessian);
		_test_tmcq_nearest(values, rows, cols, n, imatrix, 0, nearest);
		const double nearest_w = _test_tmcq_weighted_sse(values, nearest, rows, cols, imatrix);
		REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, count, cols, formats[f], 0, 0, hessian, cols * cols, packed, size), size, "Hessian quantization succeeds");
		ccv_nnc_dequantize_8i_rowwise_x(packed, CCV_32F, CCV_TENSOR_CPU_MEMORY, size, cols, formats[f], decoded, count);
		const double hessian_h = _test_tmcq_hessian_error(values, decoded, rows, cols, hessian);
		_test_tmcq_nearest(values, rows, cols, n, 0, hessian, nearest);
		const double nearest_h = _test_tmcq_hessian_error(values, nearest, rows, cols, hessian);
		printf("\n\tTMCQ n = %d imatrix objective: plain %.6g imatrix %.6g nearest %.6g; Hessian objective: plain %.6g imatrix %.6g hessian %.6g nearest %.6g", n, plain_w, imatrix_w, nearest_w, plain_h, imatrix_h, hessian_h, nearest_h);
		REQUIRE(imatrix_w < nearest_w, "imatrix trellis beats the nearest baseline for n = %d", n);
		REQUIRE(imatrix_w < plain_w, "imatrix changes the objective for n = %d", n);
		REQUIRE(hessian_h < nearest_h, "Hessian trellis beats the nearest baseline for n = %d", n);
		REQUIRE(hessian_h < imatrix_h, "Hessian feedback beats the diagonal for n = %d", n);
		ccfree(packed);
	}
	ccfree(hessian);
	ccfree(nearest);
	ccfree(decoded);
	ccfree(values);
}

TEST_CASE("TMCQ quantizes every precision with identical codes and source-precision scales")
{
	enum { rows = 32, cols = 256, count = rows * cols };
	const int datatypes[4] = {CCV_16F, CCV_16BF, CCV_32F, CCV_64F};
	float values[count];
	int i, d;
	// Multiples of 1/64 below 4 in magnitude are exact in every precision, so all four see the same input.
	for (i = 0; i < count; i++)
		values[i] = i < cols ? 0 : (float)(((i * 7919) % 255) - 127) / 64;
	const int format = CCV_NNC_QX_8I_ROWWISE_TMCQ_N20;
	const int n = ccv_nnc_8i_rowwise_x_tmcq_rate(format);
	const size_t scale_offset = _test_tmcq_scale_offset(rows, cols, n);
	uint8_t* const reference = ccmalloc(scale_offset);
	int* const q8 = ccmalloc(sizeof(int) * count);
	for (d = 0; d < 4; d++)
	{
		const size_t element_size = CCV_GET_DATA_TYPE_SIZE(datatypes[d]);
		uint8_t* const source = ccmalloc(element_size * count);
		uint8_t* const decoded = ccmalloc(element_size * count);
		_test_tmcq_from_float(values, datatypes[d], count, source);
		const size_t size = ccv_nnc_8i_rowwise_x_data_size(format, datatypes[d], count, cols);
		REQUIRE_EQ(size, scale_offset + rows * element_size, "size includes source-precision scales");
		uint8_t* const packed = ccmalloc(size);
		REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(source, datatypes[d], CCV_TENSOR_CPU_MEMORY, count, cols, format, 0, 0, 0, 0, packed, size), size, "quantizes datatype %d", d);
		if (d == 0)
			memcpy(reference, packed, scale_offset);
		else
			REQUIRE_ARRAY_EQ(uint8_t, reference, packed, scale_offset, "codes are fitted in double and match across precisions");
		ccv_nnc_dequantize_8i_rowwise_x(packed, datatypes[d], CCV_TENSOR_CPU_MEMORY, size, cols, format, decoded, count);
		_test_tmcq_decode(packed, rows, cols, n, q8);
		double* const out = ccmalloc(sizeof(double) * count);
		_test_tmcq_to_double(decoded, datatypes[d], count, out);
		double err = 0, energy = 0;
		for (i = 0; i < count; i++)
		{
			const double scale = _test_tmcq_load_scale(packed + scale_offset, datatypes[d], i / cols);
			const double v = scale * q8[i];
			const double expected = datatypes[d] == CCV_64F ? v : datatypes[d] == CCV_32F ? (float)v : out[i];
			if (datatypes[d] == CCV_64F || datatypes[d] == CCV_32F)
				REQUIRE_EQ(out[i], expected, "decoded value is stored scale times code");
			err += (out[i] - values[i]) * (out[i] - values[i]);
			energy += (double)values[i] * values[i];
		}
		for (i = 0; i < cols; i++)
			REQUIRE_EQ(out[i], 0, "zero row decodes to exact zero");
		REQUIRE(err / energy < 0.2, "datatype %d error %g is bounded", d, err / energy);
		ccfree(out);
		ccfree(packed);
		ccfree(decoded);
		ccfree(source);
	}
	ccfree(q8);
	ccfree(reference);
}

TEST_CASE("TMCQ imatrix and Hessian slices quantize like independent calls")
{
	enum { rows = 48, cols = 256, count = rows * cols, slices = 3, slice_rows = rows / slices };
	const int format = CCV_NNC_QX_8I_ROWWISE_TMCQ_N18;
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 5);
	float* const values = ccmalloc(sizeof(float) * count);
	float* const imatrix = ccmalloc(sizeof(float) * cols * slices);
	float* const hessian = ccmalloc(sizeof(float) * cols * cols * slices);
	_test_tmcq_values(&dsfmt, values, rows, cols);
	int i, j, s, h;
	for (i = 0; i < cols * slices; i++)
		imatrix[i] = i % 13 == 0 ? 0 : 0.1f + (float)dsfmt_genrand_open_close(&dsfmt);
	for (s = 0; s < slices; s++)
		for (i = 0; i < cols; i++)
			for (j = 0; j < cols; j++)
				hessian[(s * cols + i) * cols + j] = i == j ? 1 + s + (i % 5) : 0.3f / (1 + abs(i - j) + s);
	const size_t size = ccv_nnc_8i_rowwise_x_data_size(format, CCV_32F, count, cols);
	const size_t slice_size = ccv_nnc_8i_rowwise_x_data_size(format, CCV_32F, slice_rows * cols, cols);
	uint8_t* const packed = ccmalloc(size);
	uint8_t* const slice_packed = ccmalloc(slice_size);
	float* const whole = ccmalloc(sizeof(float) * count);
	float* const sliced = ccmalloc(sizeof(float) * count);
	for (h = 0; h < 2; h++)
	{
		const float* const im = h ? 0 : imatrix;
		const float* const hs = h ? hessian : 0;
		REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, count, cols, format, im, im ? cols * slices : 0, hs, hs ? cols * cols * slices : 0, packed, size), size, "sliced quantization succeeds");
		ccv_nnc_dequantize_8i_rowwise_x(packed, CCV_32F, CCV_TENSOR_CPU_MEMORY, size, cols, format, whole, count);
		for (s = 0; s < slices; s++)
		{
			REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(values + s * slice_rows * cols, CCV_32F, CCV_TENSOR_CPU_MEMORY, slice_rows * cols, cols, format, im ? im + s * cols : 0, im ? cols : 0, hs ? hs + s * cols * cols : 0, hs ? cols * cols : 0, slice_packed, slice_size), slice_size, "independent slice quantizes");
			ccv_nnc_dequantize_8i_rowwise_x(slice_packed, CCV_32F, CCV_TENSOR_CPU_MEMORY, slice_size, cols, format, sliced + s * slice_rows * cols, slice_rows * cols);
		}
		REQUIRE_ARRAY_EQ(float, whole, sliced, count, "%s slices match independent quantization", h ? "Hessian" : "imatrix");
	}
	ccfree(sliced);
	ccfree(whole);
	ccfree(slice_packed);
	ccfree(packed);
	ccfree(hessian);
	ccfree(imatrix);
	ccfree(values);
}

TEST_CASE("TMCQ rejects invalid shapes, H256 and slices that split tile rows")
{
	enum { cols = 256 };
	static float values[64 * 512];
	static uint8_t output[64 * 512];
	static float hessian[3 * cols * cols];
	float imatrix[2 * 512];
	int i;
	for (i = 0; i < 64 * 512; i++)
		values[i] = sinf(i * 0.1f);
	for (i = 0; i < 2 * 512; i++)
		imatrix[i] = 1;
	for (i = 0; i < 3 * cols; i++)
		hessian[(i / cols) * cols * cols + (i % cols) * cols + i % cols] = 1;
	const int format = CCV_NNC_QX_8I_ROWWISE_TMCQ_N24;
	// Rows not a multiple of 16, K not a multiple of 256.
	REQUIRE_EQ(ccv_nnc_8i_rowwise_x_data_size(format, CCV_32F, 15 * cols, cols), 0, "15 rows have no size");
	REQUIRE_EQ(ccv_nnc_8i_rowwise_x_data_size(format, CCV_32F, 16 * 128, 128), 0, "K = 128 has no size");
	REQUIRE_EQ(ccv_nnc_8i_rowwise_x_data_size(format, CCV_32F, 16 * 272, 272), 0, "K = 272 has no size");
	REQUIRE_EQ(ccv_nnc_8i_rowwise_x_data_size(format | CCV_NNC_QX_8I_ROWWISE_HADAMARD_256, CCV_32F, 16 * cols, cols), 0, "H256 has no size");
	REQUIRE(ccv_nnc_8i_rowwise_x_data_size(format, CCV_32F, 16 * cols, cols) > 0, "16 x 256 is valid");
	memset(output, 0xa5, sizeof(output));
	REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, 15 * cols, cols, format, 0, 0, 0, 0, output, sizeof(output)), 0, "15 rows must fail");
	REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, 24 * cols, cols, format, 0, 0, 0, 0, output, sizeof(output)), 0, "24 rows must fail");
	REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, 16 * 128, 128, format, 0, 0, 0, 0, output, sizeof(output)), 0, "K = 128 must fail");
	REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, 16 * 384, 384, format, 0, 0, 0, 0, output, sizeof(output)), 0, "K = 384 must fail");
	REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, 16 * cols, cols, format | CCV_NNC_QX_8I_ROWWISE_HADAMARD_256, 0, 0, 0, 0, output, sizeof(output)), 0, "H256 must fail");
	REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, 16 * cols, cols, format | CCV_NNC_QX_8I_ROWWISE_HADAMARD_256, imatrix, cols, 0, 0, output, sizeof(output)), 0, "H256 with imatrix must fail");
	REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, 16 * 512, 512, format, imatrix, 2 * 512, 0, 0, output, sizeof(output)), 0, "imatrix slices of 8 rows must fail");
	REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, 48 * cols, cols, format, 0, 0, hessian, 2 * cols * cols, output, sizeof(output)), 0, "Hessian slices of 24 rows must fail");
	REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, 16 * cols, cols, format, 0, 0, 0, 0, output, ccv_nnc_8i_rowwise_x_data_size(format, CCV_32F, 16 * cols, cols) - 1), 0, "short output must fail");
	for (i = 0; i < (int)sizeof(output); i++)
		REQUIRE_EQ(output[i], 0xa5, "rejected calls leave the output untouched");
	REQUIRE(ccv_nnc_quantize_8i_rowwise_x(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, 32 * 512, 512, format, imatrix, 2 * 512, 0, 0, output, sizeof(output)) > 0, "imatrix slices of 16 rows are fine");
	REQUIRE(ccv_nnc_quantize_8i_rowwise_x(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, 48 * cols, cols, format, 0, 0, hessian, 3 * cols * cols, output, sizeof(output)) > 0, "Hessian slices of 16 rows are fine");
}

TEST_CASE("TMCQ metadata and payload survive tensor persistence")
{
	enum { rows = 32, cols = 256 };
	ccv_nnc_tensor_t* const source = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, cols), 0);
	int i, f;
	for (i = 0; i < rows * cols; i++)
		source->data.f32[i] = sinf(i * 0.37f) * (1 + i % 3);
	sqlite3* db = 0;
	REQUIRE_EQ(sqlite3_open(":memory:", &db), SQLITE_OK, "in-memory database opens");
	for (f = 0; f < 9; f += 4)
	{
		ccv_nnc_tensor_t* const packed = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(source->info, tmcq_formats[f]), 0);
		const size_t size = ccv_nnc_tensor_data_size_without_padding(packed->info);
		REQUIRE_EQ(size, ccv_nnc_8i_rowwise_x_data_size(tmcq_formats[f], CCV_32F, rows * cols, cols), "tensor size matches");
		REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(source->data.f32, CCV_32F, CCV_TENSOR_CPU_MEMORY, rows * cols, cols, tmcq_formats[f], 0, 0, 0, 0, packed->data.u8, size), size, "allocated tensor quantizes");
		char name[32];
		snprintf(name, sizeof(name), "tmcq_%d", f);
		REQUIRE_EQ(ccv_nnc_tensor_write(packed, db, name, 0), CCV_IO_FINAL, "TMCQ tensor writes");
		ccv_nnc_tensor_t* restored = 0;
		REQUIRE_EQ(ccv_nnc_tensor_read(db, name, 0, 0, 0, &restored), CCV_IO_FINAL, "TMCQ tensor reads");
		REQUIRE_EQ(restored->info.datatype, packed->info.datatype, "source precision persists");
		REQUIRE_EQ(restored->info.reserved, tmcq_formats[f], "codec persists");
		REQUIRE_EQ(ccv_nnc_tensor_data_size_without_padding(restored->info), size, "size persists");
		REQUIRE_EQ(memcmp(restored->data.u8, packed->data.u8, size), 0, "payload and trailing scales persist");
		float a[rows * cols], b[rows * cols];
		ccv_nnc_dequantize_8i_rowwise_x(packed->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY, size, cols, tmcq_formats[f], a, rows * cols);
		ccv_nnc_dequantize_8i_rowwise_x(restored->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY, size, cols, restored->info.reserved, b, rows * cols);
		REQUIRE_ARRAY_EQ(float, a, b, rows * cols, "restored tensor decodes the same");
		ccv_nnc_tensor_free(restored);
		ccv_nnc_tensor_free(packed);
	}
	sqlite3_close(db);
	ccv_nnc_tensor_free(source);
}

// Straightforward Viterbi over one tile in processing order t' = 0..255, t = (t' + rotation) mod 256.
// start < 0: free ends. bp is 256 x 2^14 indexed by t'. Returns the end state.
static int _test_tmcq_pass(const double* const y, const double* const w, const int n, const int rotation, const int start, uint8_t* const bp, double* const cost_out)
{
	double* cost = ccmalloc(sizeof(double) * 32768);
	double* next = ccmalloc(sizeof(double) * 32768);
	int s, step;
	for (s = 0; s < 32768; s++)
		cost[s] = start < 0 || s == start ? 0 : INFINITY;
	for (step = 0; step < 256; step++)
	{
		const int t = (step + rotation) % 256;
		const int f = (t + 1) * n / 16 - t * n / 16;
		for (s = 0; s < 32768; s++)
		{
			const int low = s & ((1 << (15 - f)) - 1);
			int h, best_h = 0;
			for (h = 1; h < (1 << f); h++)
				if (cost[(low << f) | h] < cost[(low << f) | best_h])
					best_h = h;
			const double d = y[t] - _test_tmcq_q8(s);
			next[s] = cost[(low << f) | best_h] + w[t] * (d * d);
			bp[step * 16384 + low] = best_h;
		}
		double* const swap = cost;
		cost = next;
		next = swap;
	}
	int end = start;
	if (start < 0)
		for (s = 1, end = 0; s < 32768; s++)
			if (cost[s] < cost[end])
				end = s;
	*cost_out = cost[end];
	ccfree(next);
	ccfree(cost);
	return end;
}

static int _test_tmcq_trace(const uint8_t* const bp, const int n, const int rotation, const int last, const int first, int s, int* const states)
{
	int step;
	for (step = last; step >= first; step--)
	{
		const int t = (step + rotation) % 256;
		const int f = (t + 1) * n / 16 - t * n / 16;
		if (states)
			states[t] = s;
		const int low = s & ((1 << (15 - f)) - 1);
		s = (low << f) | bp[step * 16384 + low];
	}
	return s;
}

// Tail-biting tile fit: mid = 0 is the v1 two-pass (anchor s_255), mid = 1 the v1.1 mid anchor (s_127).
static double _test_tmcq_fit_tile(const double* const y, const double* const w, const int n, const int mid, int* const states)
{
	uint8_t* const bp = ccmalloc(256 * 16384);
	double cost;
	int s_star = _test_tmcq_pass(y, w, n, 0, -1, bp, &cost);
	if (mid)
		s_star = _test_tmcq_trace(bp, n, 0, 255, 128, s_star, 0);
	const int rotation = mid ? 128 : 0;
	_test_tmcq_pass(y, w, n, rotation, s_star, bp, &cost);
	const int s = _test_tmcq_trace(bp, n, rotation, 255, 0, s_star, states);
	assert(s == s_star);
	(void)s;
	ccfree(bp);
	return cost;
}

// imatrix-path TMCQ encoder written out from the spec, with either anchor. Writes the 32F output layout
// and returns the summed Viterbi tile costs.
static double _test_tmcq_encode(const float* const x, const int rows, const int cols, const int n, const float* const imatrix, const int mid, uint8_t* const output)
{
	const int T = 16 * n;
	const size_t scale_offset = _test_tmcq_scale_offset(rows, cols, n);
	memset(output, 0, scale_offset);
	double cb = 0;
	int s, i, j, k, tr, tc, t;
	for (s = 0; s < 32768; s++)
		cb += (double)_test_tmcq_q8(s) * _test_tmcq_q8(s);
	cb = sqrt(cb / 32768);
	double* const wc = ccmalloc(sizeof(double) * cols);
	double sum_wc = 0;
	for (k = 0; k < cols; k++)
		sum_wc += (wc[k] = imatrix ? ccv_max((double)imatrix[k], 0) : 1);
	int* const q = ccmalloc(sizeof(int) * rows * cols);
	double total = 0;
	for (tr = 0; tr < rows / 16; tr++)
	{
		double scale[16];
		for (i = 0; i < 16; i++)
		{
			double sum = 0;
			for (k = 0; k < cols; k++)
				sum += wc[k] * ((double)x[(tr * 16 + i) * cols + k] * x[(tr * 16 + i) * cols + k]);
			const double sr = sum_wc > 0 ? sqrt(sum / sum_wc) / cb : 0;
			scale[i] = sr > 0 && isfinite(sr) ? sr : 0;
		}
		for (tc = 0; tc < cols / 16; tc++)
		{
			double y[256], w[256];
			for (i = 0; i < 16; i++)
				for (j = 0; j < 16; j++)
				{
					y[i * 16 + j] = scale[i] > 0 ? x[(tr * 16 + i) * cols + tc * 16 + j] / scale[i] : 0;
					w[i * 16 + j] = wc[tc * 16 + j] * (scale[i] * scale[i]);
				}
			int states[256];
			total += _test_tmcq_fit_tile(y, w, n, mid, states);
			const size_t g = (size_t)tr * (cols / 16) + tc;
			int prev_end = 0;
			for (t = 0; t < 256; t++)
			{
				const int end = (t + 1) * n / 16;
				int p;
				for (p = prev_end; p < end; p++)
					if ((states[t] >> (14 - (end - 1 - p))) & 1)
						output[(g * T + p) / 8] |= 1 << ((g * T + p) % 8);
				prev_end = end;
				q[(tr * 16 + t / 16) * cols + tc * 16 + t % 16] = _test_tmcq_q8(states[t]);
			}
		}
		for (i = 0; i < 16; i++)
		{
			double num = 0, den = 0;
			const int r = tr * 16 + i;
			for (k = 0; k < cols; k++)
			{
				num += (double)q[r * cols + k] * (wc[k] * x[r * cols + k]);
				den += (double)q[r * cols + k] * (wc[k] * (double)q[r * cols + k]);
			}
			const double sr = num / den;
			((float*)(output + scale_offset))[r] = (float)(den > 0 && isfinite(sr) ? sr : scale[i]);
		}
	}
	ccfree(q);
	ccfree(wc);
	return total;
}

TEST_CASE("TMCQ mid anchor matches an independent encoder and costs no more than the v1 two-pass fit")
{
	enum { rows = 64, cols = 512, count = rows * cols };
	const int formats[2] = {CCV_NNC_QX_8I_ROWWISE_TMCQ_N16, CCV_NNC_QX_8I_ROWWISE_TMCQ_N24};
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 4242);
	float* const values = ccmalloc(sizeof(float) * count);
	float* const decoded = ccmalloc(sizeof(float) * count);
	float imatrix[cols];
	_test_tmcq_values(&dsfmt, values, rows, cols);
	int i, f;
	for (i = 0; i < cols; i++)
		imatrix[i] = 0.2f + (float)dsfmt_genrand_open_close(&dsfmt) * 2;
	for (f = 0; f < 2; f++)
	{
		const int n = ccv_nnc_8i_rowwise_x_tmcq_rate(formats[f]);
		const size_t size = ccv_nnc_8i_rowwise_x_data_size(formats[f], CCV_32F, count, cols);
		uint8_t* const packed = ccmalloc(size);
		uint8_t* const mid = ccmalloc(size);
		uint8_t* const v1 = ccmalloc(size);
		REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, count, cols, formats[f], imatrix, cols, 0, 0, packed, size), size, "imatrix quantization succeeds");
		const double mid_cost = _test_tmcq_encode(values, rows, cols, n, imatrix, 1, mid);
		const double v1_cost = _test_tmcq_encode(values, rows, cols, n, imatrix, 0, v1);
		REQUIRE_ARRAY_EQ(uint8_t, packed, mid, size, "library output matches the spec 4.2 encoder for n = %d", n);
		ccv_nnc_dequantize_8i_rowwise_x(mid, CCV_32F, CCV_TENSOR_CPU_MEMORY, size, cols, formats[f], decoded, count);
		const double mid_sse = _test_tmcq_weighted_sse(values, decoded, rows, cols, imatrix);
		ccv_nnc_dequantize_8i_rowwise_x(v1, CCV_32F, CCV_TENSOR_CPU_MEMORY, size, cols, formats[f], decoded, count);
		const double v1_sse = _test_tmcq_weighted_sse(values, decoded, rows, cols, imatrix);
		printf("\n\tTMCQ n = %d trellis cost mid %.6g v1 %.6g (%+.3f%%), weighted SSE after refit mid %.6g v1 %.6g (%+.3f%%)", n, mid_cost, v1_cost, (mid_cost / v1_cost - 1) * 100, mid_sse, v1_sse, (mid_sse / v1_sse - 1) * 100);
		REQUIRE(mid_cost <= v1_cost, "mid anchor trellis cost %g is no more than v1 %g for n = %d", mid_cost, v1_cost, n);
		REQUIRE(mid_sse <= v1_sse, "mid anchor weighted error %g is no more than v1 %g for n = %d", mid_sse, v1_sse, n);
		ccfree(v1);
		ccfree(mid);
		ccfree(packed);
	}
	ccfree(decoded);
	ccfree(values);
}

// FIPS 180-4 SHA-256, for the golden payload digests below.
static void _test_sha256(const uint8_t* const data, const size_t length, char hex[65])
{
	static const uint32_t k[64] = {
		0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
		0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
		0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
		0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
		0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
		0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
		0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
		0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
	};
	uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
	const size_t padded = (length + 9 + 63) / 64 * 64;
	uint8_t* const m = cccalloc(padded, 1);
	memcpy(m, data, length);
	m[length] = 0x80;
	const uint64_t bits = (uint64_t)length * 8;
	int i;
	for (i = 0; i < 8; i++)
		m[padded - 1 - i] = (uint8_t)(bits >> (8 * i));
	size_t c;
#define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
	for (c = 0; c < padded; c += 64)
	{
		uint32_t w[64];
		for (i = 0; i < 16; i++)
			w[i] = ((uint32_t)m[c + 4 * i] << 24) | ((uint32_t)m[c + 4 * i + 1] << 16) | ((uint32_t)m[c + 4 * i + 2] << 8) | m[c + 4 * i + 3];
		for (i = 16; i < 64; i++)
		{
			const uint32_t s0 = ROTR(w[i - 15], 7) ^ ROTR(w[i - 15], 18) ^ (w[i - 15] >> 3);
			const uint32_t s1 = ROTR(w[i - 2], 17) ^ ROTR(w[i - 2], 19) ^ (w[i - 2] >> 10);
			w[i] = w[i - 16] + s0 + w[i - 7] + s1;
		}
		uint32_t a = h[0], b = h[1], cc = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
		for (i = 0; i < 64; i++)
		{
			const uint32_t t1 = hh + (ROTR(e, 6) ^ ROTR(e, 11) ^ ROTR(e, 25)) + ((e & f) ^ (~e & g)) + k[i] + w[i];
			const uint32_t t2 = (ROTR(a, 2) ^ ROTR(a, 13) ^ ROTR(a, 22)) + ((a & b) ^ (a & cc) ^ (b & cc));
			hh = g, g = f, f = e, e = d + t1, d = cc, cc = b, b = a, a = t1 + t2;
		}
		h[0] += a, h[1] += b, h[2] += cc, h[3] += d, h[4] += e, h[5] += f, h[6] += g, h[7] += hh;
	}
#undef ROTR
	for (i = 0; i < 8; i++)
		snprintf(hex + i * 8, 9, "%08x", h[i]);
	ccfree(m);
}

// Portable generator for the golden inputs: xorshift32 from 0x12345678, u = (x >> 8) - 2^23, v = u / 2^23.
static uint32_t _test_tmcq_xorshift(uint32_t* const state)
{
	uint32_t x = *state;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	return *state = x;
}

static float _test_tmcq_golden_uniform(uint32_t* const state)
{
	return (float)((int32_t)(_test_tmcq_xorshift(state) >> 8) - 8388608) / 8388608.f;
}

TEST_CASE("TMCQ golden vectors for the QuantStar reference")
{
	// Inputs (all float32, generated in this order from one xorshift32 stream seeded 0x12345678):
	//   W[16][256] = uniform();
	//   imatrix[256] = ((x >> 9) + 2^21) / 2^21 for x = xorshift();
	//   A[512][256] = uniform(), H[j][k] = (float)(sum_s A[s][j] A[s][k] / 512) with s ascending in double.
	// Encoder v1.1 (spec 4.2, mid anchor). Digests cover the whole 32F output: payload, zero padding to 128 bytes and the float row scales.
	// Input digests (little-endian float32): W 2457d29c78ab867fd09946a2ca8b1c18baf38290b88174728302da8bdb2354e7,
	// imatrix c277f046520b74ffd8434a5ece854a98715bfba8bee87f654e377a7ed0568598,
	// H 3686ed3dd6195ab6b88eab8fd4859f7d14072dfe8ea63c519cfbf792e62f1612.
	enum { rows = 16, cols = 256, count = rows * cols, samples = 512 };
	static const char* const expected_digest[2][3] = {
		{ // imatrix: n = 16, 22, 32.
			"c954f755f0fddb77c4bdf849617e6dddb8345a784cad85e0f02dba04bc241755",
			"478f34b56dd77db82a30447abcf664462484983d1b98a67919032f0355e44aca",
			"479e4648f1b25ba149953a292113dba0b65e6ea9a31dc4d19bf9ef80839e237b",
		},
		{ // Hessian: n = 16, 22, 32.
			"7cc5958fe6c0ef66e42ad39fb8439d83b203828ee14bae0bd87e150b846c072e",
			"1fa49211009d64d79a9ed471a582708fa7941e2f01d53e45f9fe61bd3fb435e4",
			"cc38aad17ddf1c2ac75dd85284fd1edb0ba4f6f497107beaf45d02faa3ecc285",
		},
	};
	const int formats[3] = {CCV_NNC_QX_8I_ROWWISE_TMCQ_N16, CCV_NNC_QX_8I_ROWWISE_TMCQ_N22, CCV_NNC_QX_8I_ROWWISE_TMCQ_N32};
	char abc[65];
	_test_sha256((const uint8_t*)"abc", 3, abc);
	REQUIRE(strcmp(abc, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad") == 0, "SHA-256 self check");
	uint32_t state = 0x12345678;
	float* const values = ccmalloc(sizeof(float) * count);
	float imatrix[cols];
	float* const hessian = ccmalloc(sizeof(float) * cols * cols);
	float* const a = ccmalloc(sizeof(float) * samples * cols);
	int i, j, k, f, path;
	for (i = 0; i < count; i++)
		values[i] = _test_tmcq_golden_uniform(&state);
	for (i = 0; i < cols; i++)
		imatrix[i] = (float)((_test_tmcq_xorshift(&state) >> 9) + 2097152) / 2097152.f;
	for (i = 0; i < samples * cols; i++)
		a[i] = _test_tmcq_golden_uniform(&state);
	for (j = 0; j < cols; j++)
		for (k = 0; k < cols; k++)
		{
			double v = 0;
			int s;
			for (s = 0; s < samples; s++)
				v += (double)a[s * cols + j] * a[s * cols + k];
			hessian[j * cols + k] = (float)(v / samples);
		}
	ccfree(a);
	for (path = 0; path < 2; path++)
		for (f = 0; f < 3; f++)
		{
			const size_t size = ccv_nnc_8i_rowwise_x_data_size(formats[f], CCV_32F, count, cols);
			uint8_t* const packed = ccmalloc(size);
			REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, count, cols, formats[f], path ? 0 : imatrix, path ? 0 : cols, path ? hessian : 0, path ? cols * cols : 0, packed, size), size, "golden input quantizes");
			char digest[65];
			_test_sha256(packed, size, digest);
			const int n = ccv_nnc_8i_rowwise_x_tmcq_rate(formats[f]);
			const size_t scale_offset = _test_tmcq_scale_offset(rows, cols, n);
			printf("\n\tTMCQ golden %s n = %d size %zu sha256 %s first bytes", path ? "hessian" : "imatrix", n, size, digest);
			for (i = 0; i < 16; i++)
				printf(" %02x", packed[i]);
			printf(" scale[0] %.9g", ((float*)(packed + scale_offset))[0]);
			REQUIRE(strcmp(digest, expected_digest[path][f]) == 0, "golden digest for %s n = %d", path ? "hessian" : "imatrix", n);
			ccfree(packed);
		}
	ccfree(hessian);
	ccfree(values);
}

#include "case_main.h"
