#ifndef MFA_TMCQ_DECODE_SOURCE_HPP_
#define MFA_TMCQ_DECODE_SOURCE_HPP_

// Metal helpers shared by every kernel that decodes CCV_NNC_QX_8I_ROWWISE_TMCQ_N* weights.
// Codec definition: see CCV_NNC_QX_8I_ROWWISE_TMCQ_N16 in ccv_nnc.h. A tile is 16 * n bits = n / 2
// little-endian 32-bit words (n is even), so every tile starts on a word boundary and any 15-bit
// window, including the tail-biting one that wraps past the end of the tile, lies inside two words.
static const char* const ccv_nnc_mfa_tmcq_decode_source = R"(
// int8 code of a 15-bit TMCQ codeword: bytesum(cw * 0x83DCD12D) / 4 - 127, clamped to 127.
inline int tmcq_q8(uint cw)
{
  const uint x = cw * 0x83DCD12Du;
  const int bs = int(x & 0xffu) + int((x >> 8) & 0xffu) + int((x >> 16) & 0xffu) + int(x >> 24);
  return min((bs >> 2) - 127, 127);
}

// E_t = floor((t + 1) * n / 16): one past the last bit of weight t's codeword window.
inline uint tmcq_end(uint n, uint t)
{
  return ((t + 1) * n) >> 4;
}

// The 15-bit LSB-first window ending at bit `end` (exclusive) of a tile of `words` 32-bit words,
// wrapping inside the tile.
inline uint tmcq_window(device const uint* tile, uint words, uint end)
{
  const uint bits = words * 32;
  const uint start = (end + bits - 15) % bits;
  const uint w = start >> 5;
  const ulong pair = ulong(tile[w]) | (ulong(tile[(w + 1) % words]) << 32);
  return uint(pair >> (start & 31)) & 0x7fffu;
}

// int8 code of weight t (row t / 16, column t % 16 inside the 16 x 16 tile).
inline int tmcq_decode(device const uint* tile, uint n, uint t)
{
  return tmcq_q8(tmcq_window(tile, n >> 1, tmcq_end(n, t)));
}
)";

#endif
