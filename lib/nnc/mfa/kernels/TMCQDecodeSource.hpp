#ifndef MFA_TMCQ_DECODE_SOURCE_HPP_
#define MFA_TMCQ_DECODE_SOURCE_HPP_

// Metal helpers shared by every kernel that decodes CCV_NNC_QX_8I_ROWWISE_TMCQ_N* weights.
// Codec definition: see CCV_NNC_QX_8I_ROWWISE_TMCQ_N16 in ccv_nnc.h. A tile is 16 * n bits = n / 2
// little-endian 32-bit words (n is even), so every tile starts on a word boundary. Weight t's codeword is
// the 15-bit window ending at E_t = floor((t + 1) n / 16); weight 0's window wraps from the tile's end.
static const char* const ccv_nnc_mfa_tmcq_decode_source = R"(
// int8 code of a 15-bit TMCQ codeword: bytesum(cw * 0x83DCD12D) / 4 - 127, clamped to 127.
// s holds two 16-bit lanes, each the sum of one byte pair. s * 0x10001 puts lo + hi lanes (the byte
// sum) in bits 16 .. 31, and >> 18 is that sum / 4. Over all 32768 codewords the byte sum is at most
// 1005 (checked exhaustively), so the code is at most 124 and the clamp never applies.
inline int tmcq_q8(uint cw)
{
  const uint x = cw * 0x83DCD12Du;
  const uint s = (x & 0x00ff00ffu) + ((x >> 8) & 0x00ff00ffu);
  return int((s * 0x10001u) >> 18) - 127;
}

// The 32 bits of the little-endian pair (lo, hi) starting at bit o (0 <= o < 32).
// (hi << 1) << (31 - o) is hi << (32 - o) without an undefined 32-bit shift when o == 0.
inline uint tmcq_funnel(uint lo, uint hi, uint o)
{
  return (lo >> o) | ((hi << 1) << (31u - o));
}

// int8 codes of row r (weights 16 r ... 16 r + 15) of a tile, as q[0 .. 3] = columns 0-3, ..., 12-15.
// The row's 16 windows start at bits first + k_j, k_j = floor((j + 1) n / 16) - floor(n / 16), with
// first = r n + floor(n / 16) - 15, and end by (r + 1) n: at most n + 14 <= 46 bits. Three words from
// the first window's word cover them; shifting that 96-bit value right by first % 32 leaves a 64-bit
// (a, b) where window j is bits [k_j, k_j + 15) with k_j <= 30 known at compile time. Only row 0
// starts before the tile (its first window is the tail-biting one); its word index wraps to the tile's
// end. No other window crosses the tile's end, so the wrapped third word is never used.
inline void tmcq_decode_row16(device const uint* tile, uint n, uint r, thread int4* q)
{
  const int words = int(n >> 1);
  const int first = int(r * n + (n >> 4)) - 15;
  const int w = first >> 5; // -1 only for row 0
  const uint s0 = uint(first) & 31u;
  const int i0 = w < 0 ? w + words : w;
  const int i1 = w + 1 >= words ? w + 1 - words : w + 1;
  const int i2 = w + 2 >= words ? w + 2 - words : w + 2;
  const uint w0 = tile[i0];
  const uint w1 = tile[i1];
  const uint w2 = tile[i2];
  const uint a = tmcq_funnel(w0, w1, s0);
  const uint b = tmcq_funnel(w1, w2, s0);
  int c[16];
  #pragma clang loop unroll(full)
  for (uint j = 0; j < 16u; ++j)
  {
    const uint k = (((j + 1u) * n) >> 4) - (n >> 4);
    const uint window = k + 15u <= 32u ? (a >> k) : ((a >> k) | (b << (32u - k)));
    c[j] = tmcq_q8(window & 0x7fffu);
  }
  q[0] = int4(c[0], c[1], c[2], c[3]);
  q[1] = int4(c[4], c[5], c[6], c[7]);
  q[2] = int4(c[8], c[9], c[10], c[11]);
  q[3] = int4(c[12], c[13], c[14], c[15]);
}
)";

#endif
