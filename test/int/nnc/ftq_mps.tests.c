#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#include <nnc/ccv_nnc_codebook.h>
#include <nnc/mps/ccv_nnc_mps.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <nnc/ccv_nnc_trellis_builtin_table.inc>
#include "trellis_table_alt.inc"
#include "3rdparty/sqlite3/sqlite3.h"

// FTQ (CCV_NNC_QX_8I_ROWWISE_FTQ_N16). The reference below lays tiles out by adding up tile lengths and decodes a
// tile from the spec (128 pairs, pair j reads the 16-bit LSB-first window ending at floor((j + 1) T / 128), wrapping
// inside the tile), independently of the codec and the kernels. Any bit string is a valid code, so codes are random
// bytes.

TEST_SETUP()
{
	ccv_nnc_init();
}

static uint32_t ftq_rng_next(uint32_t* const state)
{
	uint32_t x = *state;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	*state = x;
	return x;
}

static ccv_nnc_tensor_param_t ftq_params(const int type, const int rows, const int cols)
{
	ccv_nnc_tensor_param_t params = {
		.type = type,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = CCV_16F,
		.dim = { rows, cols, 0 },
	};
	return params;
}

static int ftq_format(const int n)
{
	return CCV_NNC_QX_8I_ROWWISE_FTQ_BLOCK_FIRST + (n - 8) / 2;
}

// Start bit and length of every tile (tile row major), by summing lengths: every tile has high (16 n) bits.
static void ftq_ref_layout(const int rows, const int cols, const uint32_t high, const uint64_t drop, uint64_t* const starts, uint32_t* const lengths)
{
	(void)drop;
	const uint64_t count = (uint64_t)rows / 16 * (cols / 16);
	uint64_t bit = 0, t;
	for (t = 0; t < count; t++)
	{
		starts[t] = bit;
		lengths[t] = high;
		bit += high;
	}
}

static void ftq_ref_decode_matrix(const uint8_t* const code, const int rows, const int cols, const uint32_t high, const uint64_t drop, const int8_t* const table, const int log2_entries, int8_t* const q8)
{
	const uint64_t count = (uint64_t)rows * cols / 256;
	uint64_t* const starts = (uint64_t*)ccmalloc(sizeof(uint64_t) * count);
	uint32_t* const lengths = (uint32_t*)ccmalloc(sizeof(uint32_t) * count);
	ftq_ref_layout(rows, cols, high, drop, starts, lengths);
	const int strips = cols / 16;
	uint64_t t;
	for (t = 0; t < count; t++)
	{
		const uint64_t start = starts[t];
		const uint64_t T = lengths[t];
		int j, b;
		for (j = 0; j < 128; j++)
		{
			const uint64_t end = (uint64_t)(j + 1) * T / 128;
			uint32_t state = 0;
			for (b = 0; b < 16; b++)
			{
				const uint64_t p = start + (end + T - 16 + b) % T;
				state |= (uint32_t)((code[p >> 3] >> (p & 7)) & 1) << b;
			}
			const uint32_t k = (state * 0x83DCD12Du) >> (32 - log2_entries);
			const int row = (int)(t / strips) * 16 + j / 8;
			const int col = (int)(t % strips) * 16 + 2 * (j % 8);
			q8[(size_t)row * cols + col] = table[2 * k];
			q8[(size_t)row * cols + col + 1] = table[2 * k + 1];
		}
	}
	ccfree(lengths);
	ccfree(starts);
}

// A CCVFTQSR blob: header ("CCVFTQSR", rows, cols, high_bits, drop_count, table, table_crc), random code bits,
// random FP16 scales.
static void ftq_fill(uint8_t* const data, const size_t size, const int format, const int rows, const int cols, const uint32_t table, const uint32_t table_crc, uint32_t seed)
{
	const ccv_nnc_8i_rowwise_x_ftq_schedule_t schedule = ccv_nnc_8i_rowwise_x_ftq_schedule(format, rows, cols);
	memset(data, 0, size);
	memcpy(data, "CCVFTQSR", 8);
	const uint32_t header[6] = { (uint32_t)rows, (uint32_t)cols, schedule.high_bits, (uint32_t)schedule.drop_count, table, table_crc };
	memcpy(data + 8, header, sizeof(header));
	size_t i;
	for (i = 0; i < (schedule.code_bits + 7) / 8; i++)
		data[32 + i] = (uint8_t)(ftq_rng_next(&seed) >> 24);
	if (schedule.code_bits % 8)
		data[32 + schedule.code_bits / 8] &= (uint8_t)((1u << (schedule.code_bits % 8)) - 1);
	float* const scales = (float*)ccmalloc(sizeof(float) * rows);
	int r;
	for (r = 0; r < rows; r++)
		scales[r] = (float)((ftq_rng_next(&seed) >> 8) & 0xffff) / 65536.0f * 0.02f + 1e-4f;
	ccv_float_to_half_precision(scales, (uint16_t*)(data + schedule.scale_offset), rows);
	ccfree(scales);
}

static const uint32_t ftq_builtin_table = 9u << 8; // id 0 (the built-in table), 2^9 pairs

// (rows, cols, n): small grids (32 x 256 n16, 272 x 256 n16, a single tile row 16 x 256 n16 and 16 x 4352 n32),
// odd word counts per tile (n18, n = 2 * 9 words), n24, the real 0.8B / 4B projection sizes at n20, rates 36 - 48,
// and the rates above 3 bpw (n56, n64; the lane windows span more than 64 bits).
static const int ftq_cases[][3] = {
	{ 32, 256, 16 }, { 272, 256, 16 }, { 16, 256, 16 }, { 16, 4352, 32 }, { 48, 768, 18 }, { 64, 1024, 24 }, { 528, 1024, 20 }, { 3584, 1024, 20 }, { 9216, 2560, 20 },
	// Rates above TMCQ's (to 3 bpw, high_bits to 768).
	{ 528, 1024, 36 }, { 64, 1024, 40 }, { 9216, 2560, 36 }, { 9216, 2560, 40 }, { 9216, 2560, 48 }, { 16, 4352, 48 },
	// Above 3 bpw (high_bits 800 - 1024).
	{ 272, 512, 50 }, { 48, 768, 54 }, { 3584, 1024, 56 }, { 9216, 2560, 56 }, { 528, 1024, 62 }, { 3584, 1024, 64 }, { 9216, 2560, 64 }, { 16, 4352, 64 },
};

TEST_CASE("ftq schedule has full word-aligned tiles and a header outside the TMCQ budget")
{
	int c;
	for (c = 0; c < (int)(sizeof(ftq_cases) / sizeof(ftq_cases[0])); c++)
	{
		const int rows = ftq_cases[c][0], cols = ftq_cases[c][1], n = ftq_cases[c][2];
		const int format = ftq_format(n);
		const ccv_nnc_8i_rowwise_x_ftq_schedule_t schedule = ccv_nnc_8i_rowwise_x_ftq_schedule(format, rows, cols);
		const size_t size = ccv_nnc_8i_rowwise_x_data_size(format, CCV_16F, (size_t)rows * cols, cols);
		const uint64_t tiles = (uint64_t)rows / 16 * (cols / 16);
		REQUIRE_EQ(schedule.high_bits, 16u * n, "case %d high_bits is 16 n", c);
		REQUIRE_EQ((int)schedule.drop_count, 0, "case %d drops no bit", c);
		REQUIRE_EQ(schedule.code_bits, tiles * 16 * n, "case %d code bits", c);
		REQUIRE_EQ(schedule.scale_offset, (32 + tiles * 2 * n + 127) & ~(uint64_t)127, "case %d scale offset", c);
		REQUIRE_EQ(size, (size_t)schedule.scale_offset + 2 * rows, "case %d size", c);
		if (n <= 32)
		{
			// The header is outside the TMCQ_N<n> budget: 32 bytes plus the padding to the next 128.
			const int tmcq = CCV_NNC_QX_8I_ROWWISE_TMCQ_N16 + (n - 16) / 2;
			const size_t tmcq_size = ccv_nnc_8i_rowwise_x_data_size(tmcq, CCV_16F, (size_t)rows * cols, cols);
			REQUIRE_EQ(size - tmcq_size, (size_t)(((32 + tiles * 2 * n + 127) & ~(uint64_t)127) - ((tiles * 2 * n + 127) & ~(uint64_t)127)), "case %d size is TMCQ plus the header", c);
		}
		const uint64_t count = schedule.tile_count;
		uint64_t* const starts = (uint64_t*)ccmalloc(sizeof(uint64_t) * count);
		uint32_t* const lengths = (uint32_t*)ccmalloc(sizeof(uint32_t) * count);
		ftq_ref_layout(rows, cols, schedule.high_bits, schedule.drop_count, starts, lengths);
		REQUIRE_EQ(starts[count - 1] + lengths[count - 1], schedule.code_bits, "case %d tiles fill the code", c);
		uint64_t t, misaligned = 0;
		for (t = 0; t < count; t++)
			misaligned += starts[t] % 32 != 0;
		REQUIRE_EQ((int)misaligned, 0, "case %d every tile starts on a 32-bit word", c);
		ccfree(lengths);
		ccfree(starts);
	}
	REQUIRE_EQ(ccv_nnc_8i_rowwise_x_data_size(CCV_NNC_QX_8I_ROWWISE_FTQ_N20, CCV_32F, 16 * 256, 256), 0, "FTQ is FP16 only");
	REQUIRE_EQ(CCV_NNC_QX_8I_ROWWISE_FTQ_BLOCK_FIRST, 32, "the static-rate FTQ block starts at 32");
	REQUIRE_EQ(CCV_NNC_QX_8I_ROWWISE_FTQ_BLOCK_LAST, 95, "and ends at 95");
	REQUIRE_EQ(CCV_NNC_QX_8I_ROWWISE_FTQ_VR_BLOCK_FIRST, 96, "the variable-rate FTQ block starts at 96");
	REQUIRE_EQ(CCV_NNC_QX_8I_ROWWISE_FTQ_VR_BLOCK_LAST, 159, "and ends at 159");
	REQUIRE_EQ(CCV_NNC_QX_8I_ROWWISE_FTQ_N16, 36, "FTQ codec ids are 32 + (n - 8) / 2");
	REQUIRE_EQ(CCV_NNC_QX_8I_ROWWISE_FTQ_N64, 60, "FTQ codec ids are 32 + (n - 8) / 2");
	REQUIRE(!ccv_nnc_8i_rowwise_x_is_ftq(CCV_NNC_QX_8I_ROWWISE_FTQ_BLOCK_FIRST) && !ccv_nnc_8i_rowwise_x_is_ftq(CCV_NNC_QX_8I_ROWWISE_FTQ_N16 - 1), "n8 .. n14 are not open");
	REQUIRE(!ccv_nnc_8i_rowwise_x_is_ftq(CCV_NNC_QX_8I_ROWWISE_FTQ_N64 + 1) && !ccv_nnc_8i_rowwise_x_is_ftq(CCV_NNC_QX_8I_ROWWISE_FTQ_BLOCK_LAST), "n66 .. n134 are not open");
	REQUIRE(!ccv_nnc_8i_rowwise_x_is_tiled(CCV_NNC_QX_8I_ROWWISE_TMCQ_N32 + 1), "code 31 is empty");
	REQUIRE_EQ(ccv_nnc_8i_rowwise_x_ftq_rate(CCV_NNC_QX_8I_ROWWISE_FTQ_N50), 50, "n50");
	REQUIRE_EQ(ccv_nnc_8i_rowwise_x_ftq_rate(CCV_NNC_QX_8I_ROWWISE_FTQ_N64), 64, "n64");
}

TEST_CASE("cpu ftq decode matches the reference with the built-in and a registered table")
{
	// A registered 1024-pair table (slot from the registry, header id 1, log2 10, its crc32).
	int8_t* const big = (int8_t*)ccmalloc(2048);
	uint32_t seed = 0xc0ffeeu;
	int i;
	for (i = 0; i < 2048; i++)
		big[i] = (int8_t)((int)(ftq_rng_next(&seed) % 255) - 127);
	const int slot = ccv_nnc_codebook_register(big, 1024);
	REQUIRE(slot >= 1, "the 1024-pair table registers");
	uint32_t big_crc;
	ccv_nnc_codebook_get(slot, 0, &big_crc);
	static const int cases[][3] = { { 32, 256, 16 }, { 272, 256, 16 }, { 48, 768, 18 }, { 64, 1024, 24 }, { 528, 1024, 20 }, { 528, 1024, 40 }, { 9216, 2560, 48 } };
	int c, which;
	for (c = 0; c < (int)(sizeof(cases) / sizeof(cases[0])); c++)
		for (which = 0; which < 2; which++)
		{
			const int rows = cases[c][0], cols = cases[c][1], n = cases[c][2];
			const int format = ftq_format(n) | (which ? (slot << CCV_NNC_CODEBOOK_SLOT_SHIFT) : 0);
			ccv_nnc_tensor_t* const hq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_CPU_MEMORY, rows, cols), format), 0);
			const size_t size = ccv_nnc_tensor_data_size_without_padding(hq->info);
			ftq_fill(hq->data.u8, size, format, rows, cols, which ? (1u | (10u << 8)) : ftq_builtin_table, which ? big_crc : 0, 0x1700u + c);
			const ccv_nnc_8i_rowwise_x_ftq_schedule_t schedule = ccv_nnc_8i_rowwise_x_ftq_schedule(format, rows, cols);
			int8_t* const q8 = (int8_t*)ccmalloc((size_t)rows * cols);
			ftq_ref_decode_matrix(hq->data.u8 + 32, rows, cols, schedule.high_bits, schedule.drop_count, which ? big : ccv_nnc_trellis_builtin_table, which ? 10 : 9, q8);
			float* const scales = (float*)ccmalloc(sizeof(float) * rows);
			ccv_half_precision_to_float((uint16_t*)(hq->data.u8 + schedule.scale_offset), scales, rows);
			uint16_t* const out = (uint16_t*)ccmalloc(sizeof(uint16_t) * rows * cols);
			ccv_nnc_dequantize_8i_rowwise_x(hq->data.u8, CCV_16F, CCV_TENSOR_CPU_MEMORY, size, cols, format, out, (size_t)rows * cols);
			size_t k, mismatches = 0;
			for (k = 0; k < (size_t)rows * cols; k++)
			{
				const float v = (float)q8[k] * scales[k / cols];
				uint16_t e;
				ccv_float_to_half_precision(&v, &e, 1);
				mismatches += out[k] != e;
			}
			REQUIRE_EQ((int)mismatches, 0, "FTQ %dx%d n%d %s table CPU decode", rows, cols, n, which ? "registered" : "built-in");
			ccfree(out);
			ccfree(scales);
			ccfree(q8);
			ccv_nnc_tensor_free(hq);
		}
	ccfree(big);
}

// One rejection case: a valid FTQ blob (528 x 1024 n20, a registered 1024-pair table unless table is 0), with
// `mutate` applied, must make the CPU decode return -1.
typedef void (*ftq_mutation_f)(uint8_t* const data, const ccv_nnc_8i_rowwise_x_ftq_schedule_t* const schedule, int* const format, const int slot);

static void ftq_bad_magic(uint8_t* const d, const ccv_nnc_8i_rowwise_x_ftq_schedule_t* const s, int* const f, const int slot) { d[7] = '6'; }
static void ftq_bad_rows(uint8_t* const d, const ccv_nnc_8i_rowwise_x_ftq_schedule_t* const s, int* const f, const int slot) { d[8] ^= 16; }
static void ftq_bad_cols(uint8_t* const d, const ccv_nnc_8i_rowwise_x_ftq_schedule_t* const s, int* const f, const int slot) { d[12] ^= 1; }
static void ftq_bad_high(uint8_t* const d, const ccv_nnc_8i_rowwise_x_ftq_schedule_t* const s, int* const f, const int slot) { d[16] ^= 1; }
static void ftq_bad_drop(uint8_t* const d, const ccv_nnc_8i_rowwise_x_ftq_schedule_t* const s, int* const f, const int slot) { d[20] ^= 1; }
static void ftq_bad_reserved(uint8_t* const d, const ccv_nnc_8i_rowwise_x_ftq_schedule_t* const s, int* const f, const int slot) { d[26] = 1; }
static void ftq_bad_log2(uint8_t* const d, const ccv_nnc_8i_rowwise_x_ftq_schedule_t* const s, int* const f, const int slot) { d[25] = 11; }
static void ftq_bad_crc(uint8_t* const d, const ccv_nnc_8i_rowwise_x_ftq_schedule_t* const s, int* const f, const int slot) { d[28] ^= 1; }
static void ftq_id_without_slot(uint8_t* const d, const ccv_nnc_8i_rowwise_x_ftq_schedule_t* const s, int* const f, const int slot) { *f = ccv_nnc_codebook_with_slot(*f, 0); }
static void ftq_unregistered_slot(uint8_t* const d, const ccv_nnc_8i_rowwise_x_ftq_schedule_t* const s, int* const f, const int slot) { *f = ccv_nnc_codebook_with_slot(*f, 250); }
static void ftq_id0_with_crc(uint8_t* const d, const ccv_nnc_8i_rowwise_x_ftq_schedule_t* const s, int* const f, const int slot)
{
	*f = ccv_nnc_codebook_with_slot(*f, 0);
	const uint32_t t[2] = { 9u << 8, 1 };
	memcpy(d + 24, t, sizeof(t));
}
static void ftq_id0_log2_10(uint8_t* const d, const ccv_nnc_8i_rowwise_x_ftq_schedule_t* const s, int* const f, const int slot)
{
	*f = ccv_nnc_codebook_with_slot(*f, 0);
	const uint32_t t[2] = { 10u << 8, 0 };
	memcpy(d + 24, t, sizeof(t));
}
static void ftq_id0_with_slot(uint8_t* const d, const ccv_nnc_8i_rowwise_x_ftq_schedule_t* const s, int* const f, const int slot)
{
	const uint32_t t[2] = { 9u << 8, 0 };
	memcpy(d + 24, t, sizeof(t));
}
static void ftq_negative_scale(uint8_t* const d, const ccv_nnc_8i_rowwise_x_ftq_schedule_t* const s, int* const f, const int slot)
{
	d[s->scale_offset + 1] |= 0x80;
}
static void ftq_infinite_scale(uint8_t* const d, const ccv_nnc_8i_rowwise_x_ftq_schedule_t* const s, int* const f, const int slot)
{
	d[s->scale_offset + 2] = 0x00;
	d[s->scale_offset + 3] = 0x7c;
}
static void ftq_hadamard(uint8_t* const d, const ccv_nnc_8i_rowwise_x_ftq_schedule_t* const s, int* const f, const int slot) { *f |= CCV_NNC_QX_8I_ROWWISE_HADAMARD_256; }

TEST_CASE("cpu ftq decode rejects every broken container, also without asserts")
{
	int8_t* const big = (int8_t*)ccmalloc(2048);
	uint32_t seed = 0xbadu;
	int i;
	for (i = 0; i < 2048; i++)
		big[i] = (int8_t)((int)(ftq_rng_next(&seed) % 255) - 127);
	const int slot = ccv_nnc_codebook_register(big, 1024);
	REQUIRE(slot >= 1, "the table registers");
	uint32_t crc;
	ccv_nnc_codebook_get(slot, 0, &crc);
	const int rows = 528, cols = 1024, n = 20;
	const int base = ftq_format(n) | (slot << CCV_NNC_CODEBOOK_SLOT_SHIFT);
	const ccv_nnc_8i_rowwise_x_ftq_schedule_t schedule = ccv_nnc_8i_rowwise_x_ftq_schedule(base, rows, cols);
	const size_t size = ccv_nnc_8i_rowwise_x_data_size(base, CCV_16F, (size_t)rows * cols, cols);
	uint8_t* const blob = (uint8_t*)ccmalloc(size);
	uint8_t* const work = (uint8_t*)ccmalloc(size);
	uint16_t* const out = (uint16_t*)ccmalloc(sizeof(uint16_t) * rows * cols);
	ftq_fill(blob, size, base, rows, cols, 1u | (10u << 8), crc, 0x4242u);
	REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(blob, CCV_16F, CCV_TENSOR_CPU_MEMORY, size, cols, base, out, (size_t)rows * cols), 0, "the valid container decodes");
	static const struct {
		const char* name;
		ftq_mutation_f mutate;
	} cases[] = {
		{ "magic", ftq_bad_magic }, { "rows", ftq_bad_rows }, { "cols", ftq_bad_cols }, { "high_bits", ftq_bad_high },
		{ "drop_count", ftq_bad_drop }, { "table bits 16 - 31", ftq_bad_reserved }, { "log2 entries", ftq_bad_log2 },
		{ "table crc", ftq_bad_crc }, { "a table id without a slot", ftq_id_without_slot }, { "an unregistered slot", ftq_unregistered_slot },
		{ "id 0 with a crc", ftq_id0_with_crc }, { "id 0 with log2 10", ftq_id0_log2_10 }, { "id 0 with a slot", ftq_id0_with_slot },
		{ "a negative scale", ftq_negative_scale }, { "an infinite scale", ftq_infinite_scale },
		{ "H256", ftq_hadamard },
	};
	int c;
	for (c = 0; c < (int)(sizeof(cases) / sizeof(cases[0])); c++)
	{
		memcpy(work, blob, size);
		int format = base;
		cases[c].mutate(work, &schedule, &format, slot);
		REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(work, CCV_16F, CCV_TENSOR_CPU_MEMORY, size, cols, format, out, (size_t)rows * cols), -1, "FTQ with a broken %s is rejected", cases[c].name);
	}
	// The code is always whole bytes (16 n bits a tile) and the padding after it must be zero; the FP16-only
	// rule is checked on the same small container.
	{
		const int r2 = 16, c2 = 256, n2 = 16;
		const int f2 = ftq_format(n2);
		const ccv_nnc_8i_rowwise_x_ftq_schedule_t s2 = ccv_nnc_8i_rowwise_x_ftq_schedule(f2, r2, c2);
		const size_t z2 = ccv_nnc_8i_rowwise_x_data_size(f2, CCV_16F, (size_t)r2 * c2, c2);
		REQUIRE_EQ(s2.code_bits % 8, 0, "the code is whole bytes");
		REQUIRE(32 + s2.code_bits / 8 < s2.scale_offset, "padding follows the code");
		uint8_t* const b2 = (uint8_t*)ccmalloc(z2);
		ftq_fill(b2, z2, f2, r2, c2, ftq_builtin_table, 0, 7);
		REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(b2, CCV_16F, CCV_TENSOR_CPU_MEMORY, z2, c2, f2, out, (size_t)r2 * c2), 0, "the small container decodes");
		b2[32 + s2.code_bits / 8] = 1;
		REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(b2, CCV_16F, CCV_TENSOR_CPU_MEMORY, z2, c2, f2, out, (size_t)r2 * c2), -1, "FTQ with non-zero padding is rejected");
		b2[32 + s2.code_bits / 8] = 0;
		// Not FP16 output and a short input are rejected too.
		REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(b2, CCV_32F, CCV_TENSOR_CPU_MEMORY, z2, c2, f2, out, (size_t)r2 * c2), -1, "FTQ decodes to FP16 only");
		ccfree(b2);
	}
	ccfree(out);
	ccfree(work);
	ccfree(blob);
	ccfree(big);
}

TEST_CASE("tmcq takes a registered codebook slot and other codecs reject one, also without asserts")
{
	int8_t* const table = (int8_t*)ccmalloc(1024);
	int i;
	for (i = 0; i < 1024; i++)
		table[i] = (int8_t)((i * 31) % 255 - 127);
	const int slot = ccv_nnc_codebook_register(table, 512);
	REQUIRE(slot >= 1, "the table registers");
	const int rows = 32, cols = 256;
	const int tmcq = CCV_NNC_QX_8I_ROWWISE_TMCQ_N20;
	const size_t size = ccv_nnc_8i_rowwise_x_data_size(tmcq, CCV_16F, (size_t)rows * cols, cols);
	REQUIRE(size > 0, "TMCQ has a size");
	REQUIRE_EQ(ccv_nnc_8i_rowwise_x_data_size(tmcq | (slot << CCV_NNC_CODEBOOK_SLOT_SHIFT), CCV_16F, (size_t)rows * cols, cols), size, "TMCQ with a slot has the same size");
	uint8_t* const blob = (uint8_t*)cccalloc(size, 1);
	uint16_t* const out = (uint16_t*)ccmalloc(sizeof(uint16_t) * rows * cols);
	REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(blob, CCV_16F, CCV_TENSOR_CPU_MEMORY, size, cols, tmcq, out, (size_t)rows * cols), 0, "TMCQ decodes");
	REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(blob, CCV_16F, CCV_TENSOR_CPU_MEMORY, size, cols, tmcq | (slot << CCV_NNC_CODEBOOK_SLOT_SHIFT), out, (size_t)rows * cols), 0, "TMCQ with a slot decodes");
	REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(blob, CCV_16F, CCV_TENSOR_CPU_MEMORY, size, cols, tmcq | (250 << CCV_NNC_CODEBOOK_SLOT_SHIFT), out, (size_t)rows * cols), -1, "TMCQ with an unregistered slot is rejected");
	REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(blob, CCV_16F, CCV_TENSOR_CPU_MEMORY, size, cols, CCV_NNC_QX_8I_ROWWISE_Q4_K | (slot << CCV_NNC_CODEBOOK_SLOT_SHIFT), out, (size_t)rows * cols), -1, "a codebook slot on a non-trellis codec is rejected");
#ifdef HAVE_MPS
	if (ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS))
	{
		// Allocated with a valid format, then given an unregistered slot, or a slot on a codec that takes none.
		const int bad[2] = { tmcq | (250 << CCV_NNC_CODEBOOK_SLOT_SHIFT), CCV_NNC_QX_8I_ROWWISE_Q4_K | (slot << CCV_NNC_CODEBOOK_SLOT_SHIFT) };
		const int good[2] = { tmcq | (slot << CCV_NNC_CODEBOOK_SLOT_SHIFT), CCV_NNC_QX_8I_ROWWISE_Q4_K };
		int b;
		for (b = 0; b < 2; b++)
		{
			ccv_nnc_tensor_t* const gq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_GPU_MEMORY, rows, cols), good[b]), 0);
			gq->info.reserved = bad[b];
			ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, 1, cols), 0);
			ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, 1, rows), 0);
			REQUIRE_EQ(ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gq), TENSOR_LIST(gb), 0), CCV_NNC_EXEC_INVALID, "the GPU GEMM rejects case %d", b);
			ccv_nnc_tensor_free(gb);
			ccv_nnc_tensor_free(ga);
			ccv_nnc_tensor_free(gq);
		}
	}
#endif
	ccfree(out);
	ccfree(blob);
	ccfree(table);
}

#ifdef HAVE_MPS

TEST_CASE("mps ftq int8 decode matches the reference bit for bit")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	int c;
	for (c = 0; c < (int)(sizeof(ftq_cases) / sizeof(ftq_cases[0])); c++)
	{
		const int rows = ftq_cases[c][0], cols = ftq_cases[c][1], n = ftq_cases[c][2];
		const int format = ftq_format(n);
		ccv_nnc_tensor_t* const hq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_CPU_MEMORY, rows, cols), format), 0);
		ccv_nnc_tensor_t* const gq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_GPU_MEMORY, rows, cols), format), 0);
		const size_t size = ccv_nnc_tensor_data_size_without_padding(hq->info);
		ftq_fill(hq->data.u8, size, format, rows, cols, ftq_builtin_table, 0, 0x5170u + c);
		const ccv_nnc_8i_rowwise_x_ftq_schedule_t schedule = ccv_nnc_8i_rowwise_x_ftq_schedule(format, rows, cols);
		int8_t* const q8 = (int8_t*)ccmalloc((size_t)rows * cols);
		ftq_ref_decode_matrix(hq->data.u8 + 32, rows, cols, schedule.high_bits, schedule.drop_count, ccv_nnc_trellis_builtin_table, 9, q8);
		const size_t out_scale_offset = ((size_t)rows * cols + 127) & ~(size_t)127;
		const size_t out_bytes = out_scale_offset + rows * 2;
		ccv_nnc_tensor_t* const gout = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, (int)((out_bytes + 3) / 4)), 0);
		ccv_nnc_tensor_t* const hout = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, (int)((out_bytes + 3) / 4)), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hq), TENSOR_LIST(gq), 0);
		ccv_nnc_mps_dequantize_8i_rowwise_x_to_8i_rowwise(gq->data.u8, CCV_16F, size, cols, format, gout->data.u8, (size_t)rows * cols, 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gout), TENSOR_LIST(hout), 0);
		REQUIRE_EQ(memcmp(hout->data.u8, q8, (size_t)rows * cols), 0, "FTQ %dx%d n%d codes", rows, cols, n);
		REQUIRE_EQ(memcmp(hout->data.u8 + out_scale_offset, hq->data.u8 + schedule.scale_offset, rows * 2), 0, "FTQ %dx%d n%d scales", rows, cols, n);
		ccfree(q8);
		ccv_nnc_tensor_free(hout);
		ccv_nnc_tensor_free(gout);
		ccv_nnc_tensor_free(gq);
		ccv_nnc_tensor_free(hq);
	}
}

static float ftq_activation_value(const int row, const int k)
{
	return (float)(((row * 37 + k * 11) % 33) - 16) / 128.0f;
}

static float ftq_bias_value(const int col)
{
	return (float)((col * 7) % 13 - 6) / 64.0f;
}

// Resident 8I_ROWWISE weights from the reference codes and the blob's scales.
static ccv_nnc_tensor_t* ftq_rowwise_reference(const ccv_nnc_tensor_t* const hq, const int format, const int rows, const int cols)
{
	const ccv_nnc_8i_rowwise_x_ftq_schedule_t schedule = ccv_nnc_8i_rowwise_x_ftq_schedule(format, rows, cols);
	ccv_nnc_tensor_t* const hr = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(ftq_params(CCV_TENSOR_CPU_MEMORY, rows, cols)), 0);
	const size_t r_scale = ((size_t)rows * cols + 127) & ~(size_t)127;
	memset(hr->data.u8, 0, r_scale);
	ftq_ref_decode_matrix(hq->data.u8 + 32, rows, cols, schedule.high_bits, schedule.drop_count, ccv_nnc_trellis_builtin_table, 9, (int8_t*)hr->data.u8);
	memcpy(hr->data.u8 + r_scale, hq->data.u8 + schedule.scale_offset, rows * 2);
	return hr;
}

TEST_CASE("mps ftq gemm from 4 rows equals the gemm on its decoded 8i rowwise weights")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	const uint64_t old_flags = ccv_nnc_flags();
	static const int cases[][3] = { { 528, 1024, 22 }, { 272, 256, 16 }, { 9216, 2560, 20 }, { 528, 1024, 36 }, { 9216, 2560, 40 }, { 9216, 2560, 48 }, { 3584, 1024, 56 }, { 9216, 2560, 64 } };
	int c;
	for (c = 0; c < (int)(sizeof(cases) / sizeof(cases[0])); c++)
	{
		const int rows = cases[c][0], cols = cases[c][1], n = cases[c][2];
		const int format = ftq_format(n);
		ccv_nnc_tensor_t* const hq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_CPU_MEMORY, rows, cols), format), 0);
		ccv_nnc_tensor_t* const gq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_GPU_MEMORY, rows, cols), format), 0);
		ftq_fill(hq->data.u8, ccv_nnc_tensor_data_size_without_padding(hq->info), format, rows, cols, ftq_builtin_table, 0, 0x9a9au + c);
		ccv_nnc_tensor_t* const hr = ftq_rowwise_reference(hq, format, rows, cols);
		ccv_nnc_tensor_t* const gr = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(ftq_params(CCV_TENSOR_GPU_MEMORY, rows, cols)), 0);
		ccv_nnc_tensor_param_t bias_params = ftq_params(CCV_TENSOR_CPU_MEMORY, rows, 0);
		bias_params.dim[1] = 0;
		ccv_nnc_tensor_t* const hbias = ccv_nnc_tensor_new(0, bias_params, 0);
		bias_params.type = CCV_TENSOR_GPU_MEMORY;
		ccv_nnc_tensor_t* const gbias = ccv_nnc_tensor_new(0, bias_params, 0);
		int j;
		for (j = 0; j < rows; j++)
		{
			const float v = ftq_bias_value(j);
			ccv_float_to_half_precision(&v, (uint16_t*)hbias->data.u8 + j, 1);
		}
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hq, hr, hbias), TENSOR_LIST(gq, gr, gbias), 0);
		// Pass 0: default routing; pass 1: no ANE. From 4 rows FTQ decodes and takes the quantized scaled GEMM
		// like TMCQ; resident 8I_ROWWISE keeps its GEMV up to 8 rows, so its reference runs on at least 9 rows
		// (independent in the integer GEMM) and the first M rows are compared.
		static const int ms[] = { 4, 8, 9, 64 };
		int pass, mi, with_bias, i, k;
		for (pass = 0; pass < 2; pass++)
		{
			if (pass == 1)
				ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_ANE);
			for (mi = 0; mi < (int)(sizeof(ms) / sizeof(ms[0])); mi++)
			{
				const int m = ms[mi];
				const int mr = m < 9 ? 9 : m;
				ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_CPU_MEMORY, mr, cols), 0);
				for (i = 0; i < mr; i++)
					for (k = 0; k < cols; k++)
					{
						const float v = ftq_activation_value(i, k);
						ccv_float_to_half_precision(&v, (uint16_t*)ha->data.u8 + (size_t)i * cols + k, 1);
					}
				ccv_nnc_tensor_t* const ham = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_CPU_MEMORY, m, cols), 0);
				memcpy(ham->data.u8, ha->data.u8, (size_t)m * cols * 2);
				ccv_nnc_tensor_t* const gar = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, mr, cols), 0);
				ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, m, cols), 0);
				ccv_nnc_tensor_t* const gb1 = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, m, rows), 0);
				ccv_nnc_tensor_t* const gb2 = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, mr, rows), 0);
				ccv_nnc_tensor_t* const hb1 = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_CPU_MEMORY, m, rows), 0);
				ccv_nnc_tensor_t* const hb2 = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_CPU_MEMORY, mr, rows), 0);
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ham, ha), TENSOR_LIST(ga, gar), 0);
				for (with_bias = 0; with_bias < 2; with_bias++)
				{
					int r1, r2;
					if (with_bias)
					{
						r1 = ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gq, gbias), TENSOR_LIST(gb1), 0);
						r2 = ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(gar, gr, gbias), TENSOR_LIST(gb2), 0);
					} else {
						r1 = ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gq), TENSOR_LIST(gb1), 0);
						r2 = ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(gar, gr), TENSOR_LIST(gb2), 0);
					}
					REQUIRE_EQ(r1, CCV_NNC_EXEC_SUCCESS, "FTQ GEMM case=%d pass=%d M=%d bias=%d runs", c, pass, m, with_bias);
					REQUIRE_EQ(r2, CCV_NNC_EXEC_SUCCESS, "8I_ROWWISE GEMM case=%d pass=%d M=%d bias=%d runs", c, pass, m, with_bias);
					ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb1, gb2), TENSOR_LIST(hb1, hb2), 0);
					REQUIRE_EQ(memcmp(hb1->data.u8, hb2->data.u8, (size_t)m * rows * 2), 0, "FTQ GEMM case=%d pass=%d M=%d bias=%d equals the 8I_ROWWISE GEMM", c, pass, m, with_bias);
				}
				ccv_nnc_tensor_free(hb2);
				ccv_nnc_tensor_free(hb1);
				ccv_nnc_tensor_free(gb2);
				ccv_nnc_tensor_free(gb1);
				ccv_nnc_tensor_free(ga);
				ccv_nnc_tensor_free(gar);
				ccv_nnc_tensor_free(ham);
				ccv_nnc_tensor_free(ha);
			}
		}
		if (!(old_flags & CCV_NNC_DISABLE_MFA_ANE))
			ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_ANE);
		ccv_nnc_tensor_free(gbias);
		ccv_nnc_tensor_free(hbias);
		ccv_nnc_tensor_free(gr);
		ccv_nnc_tensor_free(hr);
		ccv_nnc_tensor_free(gq);
		ccv_nnc_tensor_free(hq);
	}
}

TEST_CASE("mps ftq fused gemv for 1 to 3 rows matches the reference bit for bit")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	// One-hot activation rows make every output fp16(fp32(q8 * scale) [+ bias]) whatever the summation
	// order. Rates from n16 to n64, including an odd word count per tile (n54, 27 words), the
	// largest tiles (16 x 4352 n64) and the 4B size.
	static const int cases[][3] = { { 64, 1024, 24 }, { 528, 1024, 20 }, { 272, 256, 16 }, { 32, 256, 16 }, { 16, 4352, 32 }, { 64, 1024, 40 }, { 16, 4352, 48 }, { 9216, 2560, 36 }, { 9216, 2560, 48 }, { 272, 512, 54 }, { 3584, 1024, 56 }, { 16, 4352, 64 }, { 9216, 2560, 64 } };
	int c;
	for (c = 0; c < (int)(sizeof(cases) / sizeof(cases[0])); c++)
	{
		const int rows = cases[c][0], cols = cases[c][1], n = cases[c][2];
		const int format = ftq_format(n);
		ccv_nnc_tensor_t* const hq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_CPU_MEMORY, rows, cols), format), 0);
		ccv_nnc_tensor_t* const gq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_GPU_MEMORY, rows, cols), format), 0);
		ftq_fill(hq->data.u8, ccv_nnc_tensor_data_size_without_padding(hq->info), format, rows, cols, ftq_builtin_table, 0, 0x77u + c);
		const ccv_nnc_8i_rowwise_x_ftq_schedule_t schedule = ccv_nnc_8i_rowwise_x_ftq_schedule(format, rows, cols);
		int8_t* const q8 = (int8_t*)ccmalloc((size_t)rows * cols);
		ftq_ref_decode_matrix(hq->data.u8 + 32, rows, cols, schedule.high_bits, schedule.drop_count, ccv_nnc_trellis_builtin_table, 9, q8);
		float* const scales = (float*)ccmalloc(sizeof(float) * rows);
		ccv_half_precision_to_float((uint16_t*)(hq->data.u8 + schedule.scale_offset), scales, rows);
		ccv_nnc_tensor_param_t bias_params = ftq_params(CCV_TENSOR_CPU_MEMORY, rows, 0);
		bias_params.dim[1] = 0;
		ccv_nnc_tensor_t* const hbias = ccv_nnc_tensor_new(0, bias_params, 0);
		bias_params.type = CCV_TENSOR_GPU_MEMORY;
		ccv_nnc_tensor_t* const gbias = ccv_nnc_tensor_new(0, bias_params, 0);
		int j;
		for (j = 0; j < rows; j++)
		{
			const float v = ftq_bias_value(j);
			ccv_float_to_half_precision(&v, (uint16_t*)hbias->data.u8 + j, 1);
		}
		float* const bias = (float*)ccmalloc(sizeof(float) * rows);
		ccv_half_precision_to_float((uint16_t*)hbias->data.u8, bias, rows);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hq, hbias), TENSOR_LIST(gq, gbias), 0);
		int m;
		for (m = 1; m <= 3; m++)
		{
			ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_CPU_MEMORY, m, cols), 0);
			ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, m, cols), 0);
			ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, m, rows), 0);
			ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_CPU_MEMORY, m, rows), 0);
			int c0, mismatches = 0;
			for (c0 = 0; c0 < cols; c0 += m)
			{
				const int with_bias = (c0 / m) & 1;
				memset(ha->data.u8, 0, (size_t)m * cols * 2);
				int i;
				for (i = 0; i < m; i++)
				{
					const float one = 1;
					ccv_float_to_half_precision(&one, (uint16_t*)ha->data.u8 + (size_t)i * cols + (c0 + i) % cols, 1);
				}
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(ga), 0);
				int status;
				if (with_bias)
					status = ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gq, gbias), TENSOR_LIST(gb), 0);
				else
					status = ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gq), TENSOR_LIST(gb), 0);
				REQUIRE_EQ(status, CCV_NNC_EXEC_SUCCESS, "FTQ GEMV M=%d runs", m);
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb), TENSOR_LIST(hb), 0);
				for (i = 0; i < m; i++)
					for (j = 0; j < rows; j++)
					{
						float v = (float)q8[(size_t)j * cols + (c0 + i) % cols] * scales[j];
						if (with_bias)
							v += bias[j];
						uint16_t e;
						ccv_float_to_half_precision(&v, &e, 1);
						mismatches += ((uint16_t*)hb->data.u8)[(size_t)i * rows + j] != e;
					}
			}
			REQUIRE_EQ(mismatches, 0, "FTQ %dx%d n%d GEMV M=%d one-hot outputs", rows, cols, n, m);
			ccv_nnc_tensor_free(hb);
			ccv_nnc_tensor_free(gb);
			ccv_nnc_tensor_free(ga);
			ccv_nnc_tensor_free(ha);
		}
		ccfree(bias);
		ccfree(scales);
		ccfree(q8);
		ccv_nnc_tensor_free(gbias);
		ccv_nnc_tensor_free(hbias);
		ccv_nnc_tensor_free(gq);
		ccv_nnc_tensor_free(hq);
	}
}

TEST_CASE("ftq quantize round trip: CPU, GPU int8 and fused GEMV decodes agree and stay within the error bound")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	// 48 rows (three tile rows) x 512 columns of rough Gaussians with row scales 0.02 .. 0.10 and a few outliers.
	enum { rows = 48, cols = 512, count = rows * cols };
	static const int rates[6] = { 16, 20, 32, 48, 56, 64 };
	// Upper bounds on sum (x - x_hat)^2 / sum x^2 per rate.
	static const double bounds[6] = { 0.24, 0.18, 0.09, 0.045, 0.038, 0.035 };
	float* const values = (float*)ccmalloc(sizeof(float) * count);
	uint16_t* const half = (uint16_t*)ccmalloc(sizeof(uint16_t) * count);
	uint16_t* const out = (uint16_t*)ccmalloc(sizeof(uint16_t) * count);
	uint32_t state = 0x6a09e667u;
	int i, j, r;
	for (i = 0; i < rows; i++)
		for (j = 0; j < cols; j++)
		{
			double v = 0;
			int k;
			for (k = 0; k < 4; k++)
				v += (ftq_rng_next(&state) >> 8) / 16777216.0 - 0.5;
			if ((i * 131 + j * 17) % 251 == 0)
				v *= 6;
			values[i * cols + j] = (float)(v * 0.02 * (1 + i % 5));
		}
	ccv_float_to_half_precision(values, half, count);
	ccv_half_precision_to_float(half, values, count);
	double energy = 0;
	for (i = 0; i < count; i++)
		energy += (double)values[i] * values[i];
	double previous = DBL_MAX;
	for (r = 0; r < 6; r++)
	{
		const int n = rates[r];
		const int format = ftq_format(n);
		ccv_nnc_tensor_t* const hq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_CPU_MEMORY, rows, cols), format), 0);
		ccv_nnc_tensor_t* const gq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_GPU_MEMORY, rows, cols), format), 0);
		const size_t size = ccv_nnc_tensor_data_size_without_padding(hq->info);
		REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(half, CCV_16F, CCV_TENSOR_CPU_MEMORY, count, cols, format, 0, 0, 0, 0, hq->data.u8, size), size, "FTQ n%d quantizes to the blob size", n);
		const ccv_nnc_8i_rowwise_x_ftq_schedule_t schedule = ccv_nnc_8i_rowwise_x_ftq_schedule(format, rows, cols);
		uint32_t header[6];
		memcpy(header, hq->data.u8 + 8, sizeof(header));
		REQUIRE(memcmp(hq->data.u8, "CCVFTQSR", 8) == 0 && header[0] == rows && header[1] == cols && header[2] == 16 * n && header[3] == 0 && header[4] == ftq_builtin_table && header[5] == 0, "FTQ n%d header", n);
		// CPU decode equals stored scale times the reference codes, bit for bit.
		int8_t* const q8 = (int8_t*)ccmalloc(count);
		ftq_ref_decode_matrix(hq->data.u8 + 32, rows, cols, schedule.high_bits, schedule.drop_count, ccv_nnc_trellis_builtin_table, 9, q8);
		float* const scales = (float*)ccmalloc(sizeof(float) * rows);
		ccv_half_precision_to_float((uint16_t*)(hq->data.u8 + schedule.scale_offset), scales, rows);
		REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(hq->data.u8, CCV_16F, CCV_TENSOR_CPU_MEMORY, size, cols, format, out, count), 0, "FTQ n%d CPU decode accepts the blob", n);
		int mismatches = 0;
		double err = 0;
		for (i = 0; i < count; i++)
		{
			const float v = (float)q8[i] * scales[i / cols];
			uint16_t e;
			ccv_float_to_half_precision(&v, &e, 1);
			mismatches += out[i] != e;
			float o;
			ccv_half_precision_to_float(out + i, &o, 1);
			err += ((double)o - values[i]) * ((double)o - values[i]);
		}
		REQUIRE_EQ(mismatches, 0, "FTQ n%d CPU decode is scale times the reference codes", n);
		const double relative = err / energy;
		printf("\n\tFTQ n = %d relative error %.6f", n, relative);
		REQUIRE(relative < bounds[r], "FTQ n%d relative error %g within %g", n, relative, bounds[r]);
		REQUIRE(relative < previous, "error falls as the rate rises");
		previous = relative;
		// GPU int8 decode.
		const size_t out_scale_offset = ((size_t)count + 127) & ~(size_t)127;
		const size_t out_bytes = out_scale_offset + rows * 2;
		ccv_nnc_tensor_t* const gout = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, (int)((out_bytes + 3) / 4)), 0);
		ccv_nnc_tensor_t* const hout = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, (int)((out_bytes + 3) / 4)), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hq), TENSOR_LIST(gq), 0);
		ccv_nnc_mps_dequantize_8i_rowwise_x_to_8i_rowwise(gq->data.u8, CCV_16F, size, cols, format, gout->data.u8, count, 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gout), TENSOR_LIST(hout), 0);
		REQUIRE_EQ(memcmp(hout->data.u8, q8, count), 0, "FTQ n%d GPU int8 codes", n);
		REQUIRE_EQ(memcmp(hout->data.u8 + out_scale_offset, hq->data.u8 + schedule.scale_offset, rows * 2), 0, "FTQ n%d GPU int8 scales", n);
		ccv_nnc_tensor_free(hout);
		ccv_nnc_tensor_free(gout);
		// Fused GEMV for 1 to 3 rows with one-hot activations: every output is fp16(q8 * scale).
		int m;
		for (m = 1; m <= 3; m++)
		{
			ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_CPU_MEMORY, m, cols), 0);
			ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, m, cols), 0);
			ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, m, rows), 0);
			ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_CPU_MEMORY, m, rows), 0);
			int c0;
			mismatches = 0;
			for (c0 = 0; c0 < cols; c0 += m)
			{
				memset(ha->data.u8, 0, (size_t)m * cols * 2);
				for (i = 0; i < m; i++)
				{
					const float one = 1;
					ccv_float_to_half_precision(&one, (uint16_t*)ha->data.u8 + (size_t)i * cols + (c0 + i) % cols, 1);
				}
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(ga), 0);
				REQUIRE_EQ(ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gq), TENSOR_LIST(gb), 0), CCV_NNC_EXEC_SUCCESS, "FTQ GEMV M=%d runs", m);
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb), TENSOR_LIST(hb), 0);
				for (i = 0; i < m; i++)
					for (j = 0; j < rows; j++)
						mismatches += ((uint16_t*)hb->data.u8)[(size_t)i * rows + j] != out[(size_t)j * cols + (c0 + i) % cols];
			}
			REQUIRE_EQ(mismatches, 0, "FTQ n%d GEMV M=%d one-hot outputs equal the CPU decode", n, m);
			ccv_nnc_tensor_free(hb);
			ccv_nnc_tensor_free(gb);
			ccv_nnc_tensor_free(ga);
			ccv_nnc_tensor_free(ha);
		}
		ccfree(scales);
		ccfree(q8);
		ccv_nnc_tensor_free(gq);
		ccv_nnc_tensor_free(hq);
	}
	ccfree(out);
	ccfree(half);
	ccfree(values);
}

TEST_CASE("mps ftq rejects an unregistered codebook and index select")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	const int rows = 32, cols = 256;
	// A slot nothing registered: the dispatch has no table to bind.
	const int slot = 250;
	REQUIRE(ccv_nnc_codebook_get(slot, 0, 0) == 0, "slot 250 is not registered");
	const int format = CCV_NNC_QX_8I_ROWWISE_FTQ_N20;
	ccv_nnc_tensor_t* const gq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_GPU_MEMORY, rows, cols), format | (slot << CCV_NNC_CODEBOOK_SLOT_SHIFT)), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, 1, cols), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, 1, rows), 0);
	REQUIRE_EQ(ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gq), TENSOR_LIST(gb), 0), CCV_NNC_EXEC_INVALID, "an unregistered codebook slot is rejected");
	ccv_nnc_tensor_t* const gp = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_GPU_MEMORY, rows, cols), format), 0);
	if (ccv_nnc_cmd_ok(CCV_NNC_INDEX_SELECT_FORWARD, CCV_NNC_BACKEND_MPS))
	{
		ccv_nnc_tensor_t* const hindices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 1), 0);
		hindices->data.i32[0] = 3;
		ccv_nnc_tensor_t* const gindices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 1), 0);
		ccv_nnc_tensor_t* const gout = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, cols), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hindices), TENSOR_LIST(gindices), 0);
		REQUIRE_EQ(ccv_nnc_cmd_exec(CMD_INDEX_SELECT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gp, gindices), TENSOR_LIST(gout), 0), CCV_NNC_EXEC_INVALID, "FTQ tables do not support IndexSelect");
		ccv_nnc_tensor_free(gout);
		ccv_nnc_tensor_free(gindices);
		ccv_nnc_tensor_free(hindices);
	}
	ccv_nnc_tensor_free(gp);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gq);
}

// An external pair table for FTQ: its pairs (2 * 2^log2 int8), the registry slot, and the header fields that name it
// (file-local id 1, log2 entries, crc32). expand > 0 replicates the built-in table (entry i = T[i >> expand]); 0 is a random table.
typedef struct {
	int8_t* pairs;
	int log2;
	int slot;
	uint32_t table;
	uint32_t crc;
} ftq_table_t;

static ftq_table_t ftq_table_new(const int log2, const int expand, uint32_t seed)
{
	ftq_table_t t;
	const int entries = 1 << log2;
	t.pairs = (int8_t*)ccmalloc(2 * entries);
	int i;
	for (i = 0; i < entries; i++)
		if (expand > 0)
		{
			t.pairs[2 * i] = ccv_nnc_trellis_builtin_table[2 * (i >> expand)];
			t.pairs[2 * i + 1] = ccv_nnc_trellis_builtin_table[2 * (i >> expand) + 1];
		} else {
			t.pairs[2 * i] = (int8_t)((int)(ftq_rng_next(&seed) % 255) - 127);
			t.pairs[2 * i + 1] = (int8_t)((int)(ftq_rng_next(&seed) % 255) - 127);
		}
	t.log2 = log2;
	t.slot = ccv_nnc_codebook_register(t.pairs, entries);
	ccv_nnc_codebook_get(t.slot, 0, &t.crc);
	t.table = 1u | ((uint32_t)log2 << 8);
	return t;
}

// One-hot GEMV outputs (M = 1 .. 3) against fp16(fp32(q8 * scale) [+ bias]); returns the number of mismatches.
static int ftq_onehot_mismatches(ccv_nnc_tensor_t* const gq, ccv_nnc_tensor_t* const gbias, const int8_t* const q8, const float* const scales, const float* const bias, const int rows, const int cols, const int m)
{
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_CPU_MEMORY, m, cols), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, m, cols), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, m, rows), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_CPU_MEMORY, m, rows), 0);
	int c0, i, j, mismatches = 0;
	for (c0 = 0; c0 < cols; c0 += m)
	{
		const int with_bias = (c0 / m) & 1;
		memset(ha->data.u8, 0, (size_t)m * cols * 2);
		for (i = 0; i < m; i++)
		{
			const float one = 1;
			ccv_float_to_half_precision(&one, (uint16_t*)ha->data.u8 + (size_t)i * cols + (c0 + i) % cols, 1);
		}
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(ga), 0);
		const int status = with_bias ?
			ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gq, gbias), TENSOR_LIST(gb), 0) :
			ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gq), TENSOR_LIST(gb), 0);
		if (status != CCV_NNC_EXEC_SUCCESS)
		{
			mismatches = -1;
			break;
		}
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb), TENSOR_LIST(hb), 0);
		for (i = 0; i < m; i++)
			for (j = 0; j < rows; j++)
			{
				float v = (float)q8[(size_t)j * cols + (c0 + i) % cols] * scales[j];
				if (with_bias)
					v += bias[j];
				uint16_t e;
				ccv_float_to_half_precision(&v, &e, 1);
				mismatches += ((uint16_t*)hb->data.u8)[(size_t)i * rows + j] != e;
			}
	}
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(ha);
	return mismatches;
}

TEST_CASE("mps ftq with external codebooks matches the reference and the built-in table")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	const uint64_t old_flags = ccv_nnc_flags();
	// The built-in table replicated to 1024 / 2048 pairs (decodes as the built-in table), then random 512 / 1024 / 2048 tables.
	ftq_table_t tables[5];
	tables[0] = ftq_table_new(10, 1, 0);
	tables[1] = ftq_table_new(11, 2, 0);
	tables[2] = ftq_table_new(9, 0, 0x1234u);
	tables[3] = ftq_table_new(10, 0, 0x5678u);
	tables[4] = ftq_table_new(11, 0, 0x9abcu);
	int ti;
	for (ti = 0; ti < 5; ti++)
		REQUIRE(tables[ti].slot >= 1, "table %d registers", ti);
	// (rows, cols, n, one-hot GEMV): small grids, the 4B size, and rates above 3 bpw.
	static const int cases[][4] = { { 64, 1024, 24, 1 }, { 528, 1024, 20, 1 }, { 272, 256, 16, 1 }, { 9216, 2560, 20, 0 }, { 64, 1024, 56, 1 }, { 272, 512, 64, 1 } };
	int c;
	for (c = 0; c < (int)(sizeof(cases) / sizeof(cases[0])); c++)
	{
		const int rows = cases[c][0], cols = cases[c][1], n = cases[c][2], onehot = cases[c][3];
		const int base = ftq_format(n);
		const ccv_nnc_8i_rowwise_x_ftq_schedule_t schedule = ccv_nnc_8i_rowwise_x_ftq_schedule(base, rows, cols);
		// The built-in table weights (slot 0) share the code with every table case.
		ccv_nnc_tensor_t* const h0 = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_CPU_MEMORY, rows, cols), base), 0);
		ccv_nnc_tensor_t* const g0 = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_GPU_MEMORY, rows, cols), base), 0);
		const size_t size = ccv_nnc_tensor_data_size_without_padding(h0->info);
		ftq_fill(h0->data.u8, size, base, rows, cols, ftq_builtin_table, 0, 0xe0e0u + c);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(h0), TENSOR_LIST(g0), 0);
		float* const scales = (float*)ccmalloc(sizeof(float) * rows);
		ccv_half_precision_to_float((uint16_t*)(h0->data.u8 + schedule.scale_offset), scales, rows);
		ccv_nnc_tensor_param_t bias_params = ftq_params(CCV_TENSOR_CPU_MEMORY, rows, 0);
		bias_params.dim[1] = 0;
		ccv_nnc_tensor_t* const hbias = ccv_nnc_tensor_new(0, bias_params, 0);
		bias_params.type = CCV_TENSOR_GPU_MEMORY;
		ccv_nnc_tensor_t* const gbias = ccv_nnc_tensor_new(0, bias_params, 0);
		float* const bias = (float*)ccmalloc(sizeof(float) * rows);
		int j;
		for (j = 0; j < rows; j++)
		{
			const float v = ftq_bias_value(j);
			ccv_float_to_half_precision(&v, (uint16_t*)hbias->data.u8 + j, 1);
		}
		ccv_half_precision_to_float((uint16_t*)hbias->data.u8, bias, rows);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hbias), TENSOR_LIST(gbias), 0);
		const size_t out_scale_offset = ((size_t)rows * cols + 127) & ~(size_t)127;
		const size_t out_bytes = out_scale_offset + rows * 2;
		ccv_nnc_tensor_t* const gout = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, (int)((out_bytes + 3) / 4)), 0);
		ccv_nnc_tensor_t* const hout = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, (int)((out_bytes + 3) / 4)), 0);
		int8_t* const q8 = (int8_t*)ccmalloc((size_t)rows * cols);
		for (ti = 0; ti < 5; ti++)
		{
			const ftq_table_t t = tables[ti];
			const int format = base | (t.slot << CCV_NNC_CODEBOOK_SLOT_SHIFT);
			ccv_nnc_tensor_t* const hq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_CPU_MEMORY, rows, cols), format), 0);
			ccv_nnc_tensor_t* const gq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_GPU_MEMORY, rows, cols), format), 0);
			memcpy(hq->data.u8, h0->data.u8, size);
			const uint32_t header_table[2] = { t.table, t.crc };
			memcpy(hq->data.u8 + 24, header_table, sizeof(header_table));
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hq), TENSOR_LIST(gq), 0);
			ftq_ref_decode_matrix(hq->data.u8 + 32, rows, cols, schedule.high_bits, schedule.drop_count, t.pairs, t.log2, q8);
			// GPU int8 decode.
			ccv_nnc_mps_dequantize_8i_rowwise_x_to_8i_rowwise(gq->data.u8, CCV_16F, size, cols, format, gout->data.u8, (size_t)rows * cols, 0);
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gout), TENSOR_LIST(hout), 0);
			REQUIRE_EQ(memcmp(hout->data.u8, q8, (size_t)rows * cols), 0, "FTQ %dx%d n%d table %d GPU codes", rows, cols, n, ti);
			// CPU decode with the same table.
			uint16_t* const out = (uint16_t*)ccmalloc(sizeof(uint16_t) * rows * cols);
			ccv_nnc_dequantize_8i_rowwise_x(hq->data.u8, CCV_16F, CCV_TENSOR_CPU_MEMORY, size, cols, format, out, (size_t)rows * cols);
			size_t k, cpu_mismatches = 0;
			for (k = 0; k < (size_t)rows * cols; k++)
			{
				const float v = (float)q8[k] * scales[k / cols];
				uint16_t e;
				ccv_float_to_half_precision(&v, &e, 1);
				cpu_mismatches += out[k] != e;
			}
			ccfree(out);
			REQUIRE_EQ((int)cpu_mismatches, 0, "FTQ %dx%d n%d table %d CPU decode", rows, cols, n, ti);
			// Fused GEMV, one-hot.
			int m;
			if (onehot)
				for (m = 1; m <= 3; m++)
					REQUIRE_EQ(ftq_onehot_mismatches(gq, gbias, q8, scales, bias, rows, cols, m), 0, "FTQ %dx%d n%d table %d GEMV M=%d one-hot", rows, cols, n, ti, m);
			// GEMM from 4 rows against resident 8I_ROWWISE of the reference codes (at least 9 rows), and for the
			// replicated built-in tables every M against the built-in table bit for bit.
			ccv_nnc_tensor_t* const hr = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(ftq_params(CCV_TENSOR_CPU_MEMORY, rows, cols)), 0);
			memset(hr->data.u8, 0, out_scale_offset);
			memcpy(hr->data.u8, q8, (size_t)rows * cols);
			memcpy(hr->data.u8 + out_scale_offset, h0->data.u8 + schedule.scale_offset, rows * 2);
			ccv_nnc_tensor_t* const gr = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(ftq_params(CCV_TENSOR_GPU_MEMORY, rows, cols)), 0);
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hr), TENSOR_LIST(gr), 0);
			static const int ms[] = { 1, 2, 3, 4, 8, 9, 64 };
			int pass, mi, i;
			for (pass = 0; pass < 2; pass++)
			{
				if (pass == 1)
					ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_ANE);
				for (mi = 0; mi < (int)(sizeof(ms) / sizeof(ms[0])); mi++)
				{
					const int mm = ms[mi];
					const int mr = mm < 9 ? 9 : mm;
					ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_CPU_MEMORY, mr, cols), 0);
					for (i = 0; i < mr; i++)
						for (k = 0; k < (size_t)cols; k++)
						{
							const float v = ftq_activation_value(i, (int)k);
							ccv_float_to_half_precision(&v, (uint16_t*)ha->data.u8 + (size_t)i * cols + k, 1);
						}
					ccv_nnc_tensor_t* const ham = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_CPU_MEMORY, mm, cols), 0);
					memcpy(ham->data.u8, ha->data.u8, (size_t)mm * cols * 2);
					ccv_nnc_tensor_t* const gar = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, mr, cols), 0);
					ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, mm, cols), 0);
					ccv_nnc_tensor_t* const gb1 = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, mm, rows), 0);
					ccv_nnc_tensor_t* const gb0 = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, mm, rows), 0);
					ccv_nnc_tensor_t* const gb2 = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, mr, rows), 0);
					ccv_nnc_tensor_t* const hb1 = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_CPU_MEMORY, mm, rows), 0);
					ccv_nnc_tensor_t* const hb0 = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_CPU_MEMORY, mm, rows), 0);
					ccv_nnc_tensor_t* const hb2 = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_CPU_MEMORY, mr, rows), 0);
					ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ham, ha), TENSOR_LIST(ga, gar), 0);
					REQUIRE_EQ(ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gq, gbias), TENSOR_LIST(gb1), 0), CCV_NNC_EXEC_SUCCESS, "FTQ table %d GEMM M=%d runs", ti, mm);
					REQUIRE_EQ(ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, g0, gbias), TENSOR_LIST(gb0), 0), CCV_NNC_EXEC_SUCCESS, "FTQ built-in GEMM M=%d runs", mm);
					REQUIRE_EQ(ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(gar, gr, gbias), TENSOR_LIST(gb2), 0), CCV_NNC_EXEC_SUCCESS, "8I_ROWWISE GEMM M=%d runs", mr);
					ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb1, gb0, gb2), TENSOR_LIST(hb1, hb0, hb2), 0);
					if (mm >= 4)
						REQUIRE_EQ(memcmp(hb1->data.u8, hb2->data.u8, (size_t)mm * rows * 2), 0, "FTQ %dx%d n%d table %d pass %d GEMM M=%d equals 8I_ROWWISE", rows, cols, n, ti, pass, mm);
					if (ti < 2)
						REQUIRE_EQ(memcmp(hb1->data.u8, hb0->data.u8, (size_t)mm * rows * 2), 0, "FTQ %dx%d n%d replicated table %d pass %d M=%d equals the built-in table", rows, cols, n, ti, pass, mm);
					ccv_nnc_tensor_free(hb2);
					ccv_nnc_tensor_free(hb0);
					ccv_nnc_tensor_free(hb1);
					ccv_nnc_tensor_free(gb2);
					ccv_nnc_tensor_free(gb0);
					ccv_nnc_tensor_free(gb1);
					ccv_nnc_tensor_free(ga);
					ccv_nnc_tensor_free(gar);
					ccv_nnc_tensor_free(ham);
					ccv_nnc_tensor_free(ha);
				}
			}
			if (!(old_flags & CCV_NNC_DISABLE_MFA_ANE))
				ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_ANE);
			ccv_nnc_tensor_free(gr);
			ccv_nnc_tensor_free(hr);
			ccv_nnc_tensor_free(gq);
			ccv_nnc_tensor_free(hq);
		}
		ccfree(q8);
		ccfree(bias);
		ccfree(scales);
		ccv_nnc_tensor_free(hout);
		ccv_nnc_tensor_free(gout);
		ccv_nnc_tensor_free(gbias);
		ccv_nnc_tensor_free(hbias);
		ccv_nnc_tensor_free(g0);
		ccv_nnc_tensor_free(h0);
	}
	for (ti = 0; ti < 5; ti++)
		ccfree(tables[ti].pairs);
}


// Quadrant-folded tables (FTQ table bits 16 - 17 = 1, CCV_NNC_CODEBOOK_FOLD). The reference expands the 512 base pairs
// to the explicit 2048-pair table (pair i: base pair i & 511, the first value negated by bit 9 of i, the second by bit
// 10) and looks it up with the index i = ((h >> 16) & 511) | bit 15 of h << 9 | bit 31 of h << 10, h = state *
// 0x83DCD12D, written independently of the codec.
static void ftq_ref_decode_fold(const uint8_t* const code, const int rows, const int cols, const uint32_t high, const uint64_t drop, const int8_t* const base, int8_t* const q8)
{
	int8_t expanded[4096];
	int i;
	for (i = 0; i < 2048; i++)
	{
		const int a = base[2 * (i & 511)], b = base[2 * (i & 511) + 1];
		expanded[2 * i] = (int8_t)((i >> 9) & 1 ? -a : a);
		expanded[2 * i + 1] = (int8_t)((i >> 10) & 1 ? -b : b);
	}
	const uint64_t count = (uint64_t)rows * cols / 256;
	uint64_t* const starts = (uint64_t*)ccmalloc(sizeof(uint64_t) * count);
	uint32_t* const lengths = (uint32_t*)ccmalloc(sizeof(uint32_t) * count);
	ftq_ref_layout(rows, cols, high, drop, starts, lengths);
	const int strips = cols / 16;
	uint64_t t;
	for (t = 0; t < count; t++)
	{
		const uint64_t start = starts[t];
		const uint64_t T = lengths[t];
		int j, b;
		for (j = 0; j < 128; j++)
		{
			const uint64_t end = (uint64_t)(j + 1) * T / 128;
			uint32_t state = 0;
			for (b = 0; b < 16; b++)
			{
				const uint64_t p = start + (end + T - 16 + b) % T;
				state |= (uint32_t)((code[p >> 3] >> (p & 7)) & 1) << b;
			}
			const uint32_t h = state * 0x83DCD12Du;
			const uint32_t k = ((h >> 16) & 511) | (((h >> 15) & 1) << 9) | ((h >> 31) << 10);
			const int row = (int)(t / strips) * 16 + j / 8;
			const int col = (int)(t % strips) * 16 + 2 * (j % 8);
			q8[(size_t)row * cols + col] = expanded[2 * k];
			q8[(size_t)row * cols + col + 1] = expanded[2 * k + 1];
		}
	}
	ccfree(lengths);
	ccfree(starts);
}

static int8_t* ftq_fold_base_new(uint32_t seed)
{
	int8_t* const base = (int8_t*)ccmalloc(1024);
	int i;
	for (i = 0; i < 1024; i++)
		base[i] = (int8_t)(ftq_rng_next(&seed) % 128);
	return base;
}

TEST_CASE("codebook registry keeps folded and plain tables apart")
{
	int8_t* const base = ftq_fold_base_new(0x0f01u);
	const int plain = ccv_nnc_codebook_register(base, 512);
	const int folded = ccv_nnc_codebook_register_with_flags(base, 512, CCV_NNC_CODEBOOK_FOLD);
	REQUIRE(plain >= 1 && folded >= 1 && plain != folded, "the same content plain and folded are two slots");
	REQUIRE_EQ(ccv_nnc_codebook_flags(plain), 0, "plain slot has no flags");
	REQUIRE_EQ(ccv_nnc_codebook_flags(folded), CCV_NNC_CODEBOOK_FOLD, "folded slot is folded");
	REQUIRE_EQ(ccv_nnc_codebook_fold_slot(plain), folded, "the folded slot of the plain one");
	REQUIRE_EQ(ccv_nnc_codebook_fold_slot(folded), -1, "a folded slot does not fold again");
	REQUIRE_EQ(ccv_nnc_codebook_fold_slot(0), -1, "the built-in table does not fold");
	REQUIRE_EQ(ccv_nnc_codebook_flags(0), 0, "built-in has no flags");
	REQUIRE_EQ(ccv_nnc_codebook_flags(251), -1, "unregistered slot");
	int8_t negative[1024];
	memcpy(negative, base, 1024);
	negative[5] = -3;
	REQUIRE_EQ(ccv_nnc_codebook_register_with_flags(negative, 512, CCV_NNC_CODEBOOK_FOLD), -1, "a folded table has no negative value");
	REQUIRE(ccv_nnc_codebook_fold_slot(ccv_nnc_codebook_register(negative, 512)) == -1, "a table with a negative value does not fold");
	int8_t big[2048];
	memset(big, 1, sizeof(big));
	REQUIRE_EQ(ccv_nnc_codebook_register_with_flags(big, 1024, CCV_NNC_CODEBOOK_FOLD), -1, "a folded table has 512 pairs");
	REQUIRE_EQ(ccv_nnc_codebook_register_with_flags(base, 512, 2), -1, "unknown flags");
	ccfree(base);
}

TEST_CASE("ftq quadrant-folded tables match the explicit 2048-pair reference on CPU, GPU, GEMV and GEMM")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	const uint64_t old_flags = ccv_nnc_flags();
	int8_t* const base = ftq_fold_base_new(0x0f17u);
	const int slot = ccv_nnc_codebook_register_with_flags(base, 512, CCV_NNC_CODEBOOK_FOLD);
	REQUIRE(slot >= 1, "folded table registers");
	uint32_t crc = 0;
	ccv_nnc_codebook_get(slot, 0, &crc);
	const uint32_t table_field = 1u | (9u << 8) | (1u << 16);
	// (rows, cols, n, one-hot GEMV): small grids, the 4B size at n20, n40 and n64, and n56.
	static const int cases[][4] = { { 64, 1024, 24, 1 }, { 272, 256, 16, 1 }, { 9216, 2560, 20, 0 }, { 9216, 2560, 40, 0 }, { 64, 1024, 56, 1 }, { 272, 512, 64, 1 }, { 9216, 2560, 64, 0 } };
	int c;
	for (c = 0; c < (int)(sizeof(cases) / sizeof(cases[0])); c++)
	{
		const int rows = cases[c][0], cols = cases[c][1], n = cases[c][2], onehot = cases[c][3];
		const int format = ftq_format(n) | (slot << CCV_NNC_CODEBOOK_SLOT_SHIFT);
		const ccv_nnc_8i_rowwise_x_ftq_schedule_t schedule = ccv_nnc_8i_rowwise_x_ftq_schedule(format, rows, cols);
		ccv_nnc_tensor_t* const hq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_CPU_MEMORY, rows, cols), format), 0);
		ccv_nnc_tensor_t* const gq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_GPU_MEMORY, rows, cols), format), 0);
		const size_t size = ccv_nnc_tensor_data_size_without_padding(hq->info);
		ftq_fill(hq->data.u8, size, format, rows, cols, table_field, crc, 0xf0f0u + c);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hq), TENSOR_LIST(gq), 0);
		int8_t* const q8 = (int8_t*)ccmalloc((size_t)rows * cols);
		ftq_ref_decode_fold(hq->data.u8 + 32, rows, cols, schedule.high_bits, schedule.drop_count, base, q8);
		size_t k, negatives = 0;
		for (k = 0; k < (size_t)rows * cols; k++)
			negatives += q8[k] < 0;
		REQUIRE(negatives > (size_t)rows * cols / 4, "the reference flips signs (%zu negative)", negatives);
		float* const scales = (float*)ccmalloc(sizeof(float) * rows);
		ccv_half_precision_to_float((uint16_t*)(hq->data.u8 + schedule.scale_offset), scales, rows);
		// CPU decode.
		uint16_t* const out = (uint16_t*)ccmalloc(sizeof(uint16_t) * rows * cols);
		REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(hq->data.u8, CCV_16F, CCV_TENSOR_CPU_MEMORY, size, cols, format, out, (size_t)rows * cols), 0, "FTQ fold CPU decode runs");
		size_t cpu_mismatches = 0;
		for (k = 0; k < (size_t)rows * cols; k++)
		{
			const float v = (float)q8[k] * scales[k / cols];
			uint16_t e;
			ccv_float_to_half_precision(&v, &e, 1);
			cpu_mismatches += out[k] != e;
		}
		ccfree(out);
		REQUIRE_EQ((int)cpu_mismatches, 0, "FTQ fold %dx%d n%d CPU decode", rows, cols, n);
		// GPU int8 decode.
		const size_t out_scale_offset = ((size_t)rows * cols + 127) & ~(size_t)127;
		const size_t out_bytes = out_scale_offset + rows * 2;
		ccv_nnc_tensor_t* const gout = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, (int)((out_bytes + 3) / 4)), 0);
		ccv_nnc_tensor_t* const hout = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, (int)((out_bytes + 3) / 4)), 0);
		ccv_nnc_mps_dequantize_8i_rowwise_x_to_8i_rowwise(gq->data.u8, CCV_16F, size, cols, format, gout->data.u8, (size_t)rows * cols, 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gout), TENSOR_LIST(hout), 0);
		REQUIRE_EQ(memcmp(hout->data.u8, q8, (size_t)rows * cols), 0, "FTQ fold %dx%d n%d GPU codes", rows, cols, n);
		ccv_nnc_tensor_param_t bias_params = ftq_params(CCV_TENSOR_CPU_MEMORY, rows, 0);
		bias_params.dim[1] = 0;
		ccv_nnc_tensor_t* const hbias = ccv_nnc_tensor_new(0, bias_params, 0);
		bias_params.type = CCV_TENSOR_GPU_MEMORY;
		ccv_nnc_tensor_t* const gbias = ccv_nnc_tensor_new(0, bias_params, 0);
		float* const bias = (float*)ccmalloc(sizeof(float) * rows);
		int j;
		for (j = 0; j < rows; j++)
		{
			const float v = ftq_bias_value(j);
			ccv_float_to_half_precision(&v, (uint16_t*)hbias->data.u8 + j, 1);
		}
		ccv_half_precision_to_float((uint16_t*)hbias->data.u8, bias, rows);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hbias), TENSOR_LIST(gbias), 0);
		int m;
		if (onehot)
			for (m = 1; m <= 3; m++)
				REQUIRE_EQ(ftq_onehot_mismatches(gq, gbias, q8, scales, bias, rows, cols, m), 0, "FTQ fold %dx%d n%d GEMV M=%d one-hot", rows, cols, n, m);
		// GEMV (M = 1 .. 3, the 4B shapes) and GEMM from 4 rows against resident 8I_ROWWISE of the reference codes;
		// GEMV rows sum in another order, so the GEMV is compared within FP16 rounding.
		ccv_nnc_tensor_t* const hr = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(ftq_params(CCV_TENSOR_CPU_MEMORY, rows, cols)), 0);
		memset(hr->data.u8, 0, out_scale_offset);
		memcpy(hr->data.u8, q8, (size_t)rows * cols);
		memcpy(hr->data.u8 + out_scale_offset, hq->data.u8 + schedule.scale_offset, rows * 2);
		ccv_nnc_tensor_t* const gr = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(ftq_params(CCV_TENSOR_GPU_MEMORY, rows, cols)), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hr), TENSOR_LIST(gr), 0);
		static const int ms[] = { 1, 3, 4, 8, 9, 64 };
		int pass, mi, i;
		for (pass = 0; pass < 2; pass++)
		{
			if (pass == 1)
				ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_ANE);
			for (mi = 0; mi < (int)(sizeof(ms) / sizeof(ms[0])); mi++)
			{
				const int mm = ms[mi];
				const int mr = mm < 9 ? 9 : mm;
				ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_CPU_MEMORY, mr, cols), 0);
				for (i = 0; i < mr; i++)
					for (k = 0; k < (size_t)cols; k++)
					{
						const float v = ftq_activation_value(i, (int)k);
						ccv_float_to_half_precision(&v, (uint16_t*)ha->data.u8 + (size_t)i * cols + k, 1);
					}
				ccv_nnc_tensor_t* const ham = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_CPU_MEMORY, mm, cols), 0);
				memcpy(ham->data.u8, ha->data.u8, (size_t)mm * cols * 2);
				ccv_nnc_tensor_t* const gar = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, mr, cols), 0);
				ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, mm, cols), 0);
				ccv_nnc_tensor_t* const gb1 = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, mm, rows), 0);
				ccv_nnc_tensor_t* const gb2 = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, mr, rows), 0);
				ccv_nnc_tensor_t* const hb1 = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_CPU_MEMORY, mm, rows), 0);
				ccv_nnc_tensor_t* const hb2 = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_CPU_MEMORY, mr, rows), 0);
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ham, ha), TENSOR_LIST(ga, gar), 0);
				REQUIRE_EQ(ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gq, gbias), TENSOR_LIST(gb1), 0), CCV_NNC_EXEC_SUCCESS, "FTQ fold GEMM M=%d runs", mm);
				REQUIRE_EQ(ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(gar, gr, gbias), TENSOR_LIST(gb2), 0), CCV_NNC_EXEC_SUCCESS, "8I_ROWWISE GEMM M=%d runs", mr);
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb1, gb2), TENSOR_LIST(hb1, hb2), 0);
				if (mm >= 4) {
					REQUIRE_EQ(memcmp(hb1->data.u8, hb2->data.u8, (size_t)mm * rows * 2), 0, "FTQ fold %dx%d n%d pass %d GEMM M=%d equals 8I_ROWWISE", rows, cols, n, pass, mm);
				} else {
					// The fused GEMV against an FP64 sum of the reference codes: within two FP16 roundings of the
					// result plus FP32 summation error (bounded by the sum of |terms|).
					float* const f1 = (float*)ccmalloc(sizeof(float) * mm * rows);
					ccv_half_precision_to_float((uint16_t*)hb1->data.u8, f1, (size_t)mm * rows);
					float* const fa = (float*)ccmalloc(sizeof(float) * mm * cols);
					ccv_half_precision_to_float((uint16_t*)ham->data.u8, fa, (size_t)mm * cols);
					int bad = 0;
					for (i = 0; i < mm && !bad; i++)
						for (j = 0; j < rows && !bad; j++)
						{
							double sum = 0, magnitude = 0;
							for (k = 0; k < (size_t)cols; k++)
							{
								const double term = (double)q8[(size_t)j * cols + k] * fa[(size_t)i * cols + k];
								sum += term;
								magnitude += fabs(term);
							}
							const double expected = sum * scales[j] + bias[j];
							const double err = fabs((double)f1[(size_t)i * rows + j] - expected);
							if (err > fabs(expected) * (2.0 / 2048) + magnitude * scales[j] * 1e-6 + 1e-6)
								bad = 1;
						}
					ccfree(fa);
					ccfree(f1);
					REQUIRE(!bad, "FTQ fold %dx%d n%d GEMV M=%d within rounding of the FP64 sum", rows, cols, n, mm);
				}
				ccv_nnc_tensor_free(hb2);
				ccv_nnc_tensor_free(hb1);
				ccv_nnc_tensor_free(gb2);
				ccv_nnc_tensor_free(gb1);
				ccv_nnc_tensor_free(ga);
				ccv_nnc_tensor_free(gar);
				ccv_nnc_tensor_free(ham);
				ccv_nnc_tensor_free(ha);
			}
		}
		if (!(old_flags & CCV_NNC_DISABLE_MFA_ANE))
			ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_ANE);
		ccfree(bias);
		ccfree(scales);
		ccfree(q8);
		ccv_nnc_tensor_free(gr);
		ccv_nnc_tensor_free(hr);
		ccv_nnc_tensor_free(gbias);
		ccv_nnc_tensor_free(hbias);
		ccv_nnc_tensor_free(hout);
		ccv_nnc_tensor_free(gout);
		ccv_nnc_tensor_free(gq);
		ccv_nnc_tensor_free(hq);
	}
	ccfree(base);
}

TEST_CASE("cpu ftq decode rejects a fold field that does not match the table, also without asserts")
{
	int8_t* const base = ftq_fold_base_new(0x0f23u);
	const int plain = ccv_nnc_codebook_register(base, 512);
	const int folded = ccv_nnc_codebook_register_with_flags(base, 512, CCV_NNC_CODEBOOK_FOLD);
	uint32_t crc = 0;
	ccv_nnc_codebook_get(folded, 0, &crc);
	const int rows = 64, cols = 1024;
	const int base_format = ftq_format(24);
	const size_t size = ccv_nnc_8i_rowwise_x_data_size(base_format, CCV_16F, (size_t)rows * cols, cols);
	uint8_t* const data = (uint8_t*)ccmalloc(size);
	uint16_t* const out = (uint16_t*)ccmalloc(sizeof(uint16_t) * rows * cols);
	// (slot, table field, expected): good folded, the fold bit with a plain slot, a plain field with a folded slot,
	// fold with log2 10, fold with id 0, structure 2, bits 18 - 31.
	const struct { int slot; uint32_t field; int ok; const char* what; } cases[] = {
		{ folded, 1u | (9u << 8) | (1u << 16), 0, "folded" },
		{ plain, 1u | (9u << 8) | (1u << 16), -1, "fold bit, plain slot" },
		{ folded, 1u | (9u << 8), -1, "plain field, folded slot" },
		{ folded, 1u | (10u << 8) | (1u << 16), -1, "fold with log2 10" },
		{ folded, (9u << 8) | (1u << 16), -1, "fold with id 0" },
		{ folded, 1u | (9u << 8) | (2u << 16), -1, "structure 2" },
		{ folded, 1u | (9u << 8) | (1u << 16) | (1u << 18), -1, "bit 18" },
		{ folded, 1u | (9u << 8) | (1u << 16) | (1u << 31), -1, "bit 31" },
	};
	int i;
	for (i = 0; i < (int)(sizeof(cases) / sizeof(cases[0])); i++)
	{
		const int format = base_format | (cases[i].slot << CCV_NNC_CODEBOOK_SLOT_SHIFT);
		ftq_fill(data, size, format, rows, cols, cases[i].field, crc, 0x77u);
		REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(data, CCV_16F, CCV_TENSOR_CPU_MEMORY, size, cols, format, out, (size_t)rows * cols), cases[i].ok, "FTQ %s", cases[i].what);
	}
	// A folded slot names a table for FTQ only.
	const int tmcq_format = CCV_NNC_QX_8I_ROWWISE_TMCQ_N20 | (folded << CCV_NNC_CODEBOOK_SLOT_SHIFT);
	REQUIRE_EQ(ccv_nnc_8i_rowwise_x_data_size(tmcq_format, CCV_16F, (size_t)rows * cols, cols) > 0 ? ccv_nnc_dequantize_8i_rowwise_x(data, CCV_16F, CCV_TENSOR_CPU_MEMORY, ccv_nnc_8i_rowwise_x_data_size(tmcq_format, CCV_16F, (size_t)rows * cols, cols), cols, tmcq_format, out, (size_t)rows * cols) : -1, -1, "TMCQ with a folded slot");
	ccfree(out);
	ccfree(data);
	ccfree(base);
}

TEST_CASE("an ftq checkpoint with a folded table reads into the folded slot and other schedules fail to load")
{
	int8_t* const base = ftq_fold_base_new(0x0f31u);
	const int plain = ccv_nnc_codebook_register(base, 512);
	REQUIRE(plain >= 1, "base table registers");
	uint32_t crc = 0;
	ccv_nnc_codebook_get(plain, 0, &crc);
	const int rows = 64, cols = 1024;
	const int file_format = ftq_format(24) | (4 << CCV_NNC_CODEBOOK_SLOT_SHIFT); // file-local id 4
	sqlite3* db = 0;
	REQUIRE_EQ(sqlite3_open(":memory:", &db), SQLITE_OK, "in-memory database opens");
	REQUIRE_EQ(ccv_nnc_tensor_write_codebook(db, plain, 4), CCV_IO_FINAL, "the base table writes as id 4");
	// header: 0 as filled, 1 a non-zero drop_count, 2 high_bits 16 n - 1 (schedules FTQ rejects).
	const struct { const char* name; uint32_t field; int folded; int header; } tensors[] = {
		{ "w_fold", 4u | (9u << 8) | (1u << 16), 1, 0 },
		{ "w_plain", 4u | (9u << 8), 0, 0 },
		{ "w_bad_structure", 4u | (9u << 8) | (3u << 16), -1, 0 },
		{ "w_drop", 4u | (9u << 8), -1, 1 },
		{ "w_short_high", 4u | (9u << 8), -1, 2 },
	};
	const int tensor_count = (int)(sizeof(tensors) / sizeof(tensors[0]));
	int i;
	for (i = 0; i < tensor_count; i++)
	{
		ccv_nnc_tensor_t* const t = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_CPU_MEMORY, rows, cols), file_format), 0);
		ftq_fill(t->data.u8, ccv_nnc_tensor_data_size_without_padding(t->info), file_format, rows, cols, tensors[i].field, crc, 0x99u + i);
		if (tensors[i].header)
		{
			uint32_t v;
			memcpy(&v, t->data.u8 + (tensors[i].header == 1 ? 20 : 16), 4);
			v = tensors[i].header == 1 ? 1 : v - 1;
			memcpy(t->data.u8 + (tensors[i].header == 1 ? 20 : 16), &v, 4);
		}
		REQUIRE_EQ(ccv_nnc_tensor_write(t, db, tensors[i].name, 0), CCV_IO_FINAL, "%s writes", tensors[i].name);
		ccv_nnc_tensor_free(t);
	}
	for (i = 0; i < tensor_count; i++)
	{
		ccv_nnc_tensor_t* restored = 0;
		const int result = ccv_nnc_tensor_read(db, tensors[i].name, 0, 0, 0, &restored);
		if (tensors[i].folded < 0)
		{
			REQUIRE(result != CCV_IO_FINAL && !restored, "%s is rejected", tensors[i].name);
			continue;
		}
		REQUIRE_EQ(result, CCV_IO_FINAL, "%s reads", tensors[i].name);
		const int slot = ccv_nnc_codebook_slot_of(restored->info.reserved);
		REQUIRE_EQ(ccv_nnc_codebook_flags(slot), tensors[i].folded ? CCV_NNC_CODEBOOK_FOLD : 0, "%s slot flags", tensors[i].name);
		if (!tensors[i].folded)
			REQUIRE_EQ(slot, plain, "%s takes the plain slot", tensors[i].name);
		uint16_t* const out = (uint16_t*)ccmalloc(sizeof(uint16_t) * rows * cols);
		REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(restored->data.u8, CCV_16F, CCV_TENSOR_CPU_MEMORY, ccv_nnc_tensor_data_size_without_padding(restored->info), cols, restored->info.reserved, out, (size_t)rows * cols), 0, "%s decodes", tensors[i].name);
		ccfree(out);
		ccv_nnc_tensor_free(restored);
	}
	sqlite3_close(db);
	ccfree(base);
}

// Tile-row boundaries of the int8 decode, which runs in blocks of 2 neighbouring tiles x 16 rows: 1, 3, 13 and 17 tile
// rows (16 x 256, 48 x 256, 208 x 768, 272 x 512), at rates with three (n <= 48) and four (n > 48) words per lane.
// GPU int8 decode, CPU decode and the one-hot fused GEMV (M = 1 .. 3) must equal
// the reference bit for bit, for the built-in table and a quadrant-folded one.
TEST_CASE("mps ftq decode is exact across odd tile-row counts with the built-in and a folded table")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	int8_t* const base = ftq_fold_base_new(0xb0d7u);
	const int fold_slot = ccv_nnc_codebook_register_with_flags(base, 512, CCV_NNC_CODEBOOK_FOLD);
	REQUIRE(fold_slot >= 1, "folded table registers");
	uint32_t fold_crc = 0;
	ccv_nnc_codebook_get(fold_slot, 0, &fold_crc);
	static const int shapes[][2] = { { 16, 256 }, { 48, 256 }, { 208, 768 }, { 272, 512 } };
	static const int rates[] = { 16, 22, 48, 56, 64 };
	int s, r, folded;
	for (s = 0; s < (int)(sizeof(shapes) / sizeof(shapes[0])); s++)
		for (r = 0; r < (int)(sizeof(rates) / sizeof(rates[0])); r++)
			for (folded = 0; folded < 2; folded++)
			{
				const int rows = shapes[s][0], cols = shapes[s][1], n = rates[r];
				const int format = ftq_format(n) | ((folded ? fold_slot : 0) << CCV_NNC_CODEBOOK_SLOT_SHIFT);
				const ccv_nnc_8i_rowwise_x_ftq_schedule_t schedule = ccv_nnc_8i_rowwise_x_ftq_schedule(format, rows, cols);
				ccv_nnc_tensor_t* const hq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_CPU_MEMORY, rows, cols), format), 0);
				ccv_nnc_tensor_t* const gq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_GPU_MEMORY, rows, cols), format), 0);
				const size_t size = ccv_nnc_tensor_data_size_without_padding(hq->info);
				ftq_fill(hq->data.u8, size, format, rows, cols, folded ? 1u | (9u << 8) | (1u << 16) : ftq_builtin_table, folded ? fold_crc : 0, 0xb0d0u + s * 16 + r * 2 + folded);
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hq), TENSOR_LIST(gq), 0);
				int8_t* const q8 = (int8_t*)ccmalloc((size_t)rows * cols);
				if (folded)
					ftq_ref_decode_fold(hq->data.u8 + 32, rows, cols, schedule.high_bits, schedule.drop_count, base, q8);
				else
					ftq_ref_decode_matrix(hq->data.u8 + 32, rows, cols, schedule.high_bits, schedule.drop_count, ccv_nnc_trellis_builtin_table, 9, q8);
				float* const scales = (float*)ccmalloc(sizeof(float) * rows);
				ccv_half_precision_to_float((uint16_t*)(hq->data.u8 + schedule.scale_offset), scales, rows);
				// CPU decode.
				uint16_t* const out = (uint16_t*)ccmalloc(sizeof(uint16_t) * rows * cols);
				REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(hq->data.u8, CCV_16F, CCV_TENSOR_CPU_MEMORY, size, cols, format, out, (size_t)rows * cols), 0, "FTQ %dx%d n%d fold %d CPU decode runs", rows, cols, n, folded);
				size_t k, cpu_mismatches = 0;
				for (k = 0; k < (size_t)rows * cols; k++)
				{
					const float v = (float)q8[k] * scales[k / cols];
					uint16_t e;
					ccv_float_to_half_precision(&v, &e, 1);
					cpu_mismatches += out[k] != e;
				}
				ccfree(out);
				REQUIRE_EQ((int)cpu_mismatches, 0, "FTQ %dx%d n%d fold %d CPU decode", rows, cols, n, folded);
				// GPU int8 decode, codes and scales.
				const size_t out_scale_offset = ((size_t)rows * cols + 127) & ~(size_t)127;
				const size_t out_bytes = out_scale_offset + rows * 2;
				ccv_nnc_tensor_t* const gout = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, (int)((out_bytes + 3) / 4)), 0);
				ccv_nnc_tensor_t* const hout = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, (int)((out_bytes + 3) / 4)), 0);
				ccv_nnc_mps_dequantize_8i_rowwise_x_to_8i_rowwise(gq->data.u8, CCV_16F, size, cols, format, gout->data.u8, (size_t)rows * cols, 0);
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gout), TENSOR_LIST(hout), 0);
				REQUIRE_EQ(memcmp(hout->data.u8, q8, (size_t)rows * cols), 0, "FTQ %dx%d n%d fold %d GPU codes", rows, cols, n, folded);
				REQUIRE_EQ(memcmp(hout->data.u8 + out_scale_offset, hq->data.u8 + schedule.scale_offset, rows * 2), 0, "FTQ %dx%d n%d fold %d GPU scales", rows, cols, n, folded);
				// One-hot fused GEMV.
				ccv_nnc_tensor_param_t bias_params = ftq_params(CCV_TENSOR_CPU_MEMORY, rows, 0);
				bias_params.dim[1] = 0;
				ccv_nnc_tensor_t* const hbias = ccv_nnc_tensor_new(0, bias_params, 0);
				bias_params.type = CCV_TENSOR_GPU_MEMORY;
				ccv_nnc_tensor_t* const gbias = ccv_nnc_tensor_new(0, bias_params, 0);
				float* const bias = (float*)ccmalloc(sizeof(float) * rows);
				int j, m;
				for (j = 0; j < rows; j++)
				{
					const float v = ftq_bias_value(j);
					ccv_float_to_half_precision(&v, (uint16_t*)hbias->data.u8 + j, 1);
				}
				ccv_half_precision_to_float((uint16_t*)hbias->data.u8, bias, rows);
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hbias), TENSOR_LIST(gbias), 0);
				for (m = 1; m <= 3; m++)
					REQUIRE_EQ(ftq_onehot_mismatches(gq, gbias, q8, scales, bias, rows, cols, m), 0, "FTQ %dx%d n%d fold %d GEMV M=%d one-hot", rows, cols, n, folded, m);
				ccfree(bias);
				ccv_nnc_tensor_free(gbias);
				ccv_nnc_tensor_free(hbias);
				ccv_nnc_tensor_free(hout);
				ccv_nnc_tensor_free(gout);
				ccfree(scales);
				ccfree(q8);
				ccv_nnc_tensor_free(gq);
				ccv_nnc_tensor_free(hq);
			}
	ccfree(base);
}

#endif

#include "case_main.h"
