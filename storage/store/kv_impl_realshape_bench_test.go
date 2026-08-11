package store

// Benchmarks at the REAL store shape measured on production bsc-v5 tier1
// (uniswap_v2:store_pool, from the quicksave "key_count"/"total_size_bytes" logs):
//
//	key_count        = 3,373,228
//	total_size_bytes = 357,562,168   -> ~106 bytes per entry
//	key              = Hex::encode(address) -> 40 lowercase hex chars
//	value            => ~66 bytes (106 - 40)
//
// This differs from kv_impl_bench_test.go's synthetic grid in two ways that both
// matter, and in opposite directions:
//
//  1. 34x more keys, 15x smaller values. High key count with tiny values is the
//     worst case for Go map per-entry overhead and the best case for bbolt's
//     packed B+tree pages — it should favour mmap on memory.
//  2. Keys are pseudo-random hex, not sequential "key:%08d". Sequential inserts
//     are bbolt's best case (few page splits at FillPercent=1.0); random order is
//     its worst — it should favour memory on merge speed.
//
// Run with:
//
//	go test -run='^$' -bench=BenchmarkRealShape -benchmem -benchtime=5x -count=2 ./storage/store/
//	SUBSTREAMS_STORE_BACKEND=mmap go test -run='^$' -bench=BenchmarkRealShape -benchmem -benchtime=5x -count=2 ./storage/store/
//
// -benchtime=Nx is required: the timed Merge is sub-second while each iteration's
// setup rebuilds a 3.4M-key store, so letting Go pick the iteration count by
// wall-clock would run for a very long time.

import (
	"fmt"
	"runtime"
	"testing"
)

const (
	realStoreKeys = 3_373_228 // uniswap_v2:store_pool on BSC
	realValueSize = 66        // 106 B/entry - 40 B hex key
	realKeyHexLen = 40
)

// makeHexKeys returns n deterministic 40-char lowercase hex keys in
// pseudo-random order, mimicking Hex::encode(address). Deterministic (fixed
// seed, no rand source) so the two backend runs see byte-identical input.
func makeHexKeys(n int) []string {
	const hexDigits = "0123456789abcdef"
	keys := make([]string, n)
	buf := make([]byte, realKeyHexLen)
	// splitmix64: cheap, well-distributed, reproducible.
	var state uint64 = 0x9E3779B97F4A7C15
	next := func() uint64 {
		state += 0x9E3779B97F4A7C15
		z := state
		z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
		z = (z ^ (z >> 27)) * 0x94D049BB133111EB
		return z ^ (z >> 31)
	}
	for i := range n {
		// 40 hex chars = 160 bits = 2.5 x uint64; refresh every 16 nibbles.
		var r uint64
		for j := range realKeyHexLen {
			if j%16 == 0 {
				r = next()
			}
			buf[j] = hexDigits[r&0xF]
			r >>= 4
		}
		keys[i] = string(buf)
	}
	return keys
}

// makeHexKV builds the store contents for the given keys.
func makeHexKV(keys []string, valueSize int) map[string][]byte {
	kv := make(map[string][]byte, len(keys))
	val := make([]byte, valueSize)
	for i := range val {
		val[i] = byte(i % 256)
	}
	for _, k := range keys {
		v := make([]byte, valueSize)
		copy(v, val)
		kv[k] = v
	}
	return kv
}

// BenchmarkRealShape_Merge merges a realistically small partial into the real
// 3.4M-key FullKV. store_pool uses updatePolicy: set and is written once per pool
// creation, so a 1000-block segment touches only a handful of keys — hence the
// small partial sizes. The partial keys are FRESH (not present in the full store),
// which is the real case for a pool store and the one that forces bbolt page
// splits, unlike the synthetic grid where the partial fully overlapped the store.
//
// The store is hydrated ONCE per sub-benchmark and merged into repeatedly, rather
// than rebuilt per iteration. Rebuilding cost ~10s of setup per iteration, capping
// us at 5 iterations — far too few to average out GC of a 3.4M-key map. The first
// attempt produced 100x spread inside a single config (346us..3.8ms for the same
// partial size), which is GC landing randomly in the timed section. Amortising the
// setup buys ~100 iterations, enough for that noise to average rather than dominate.
//
// Each iteration merges a FRESH batch of keys, so every merge inserts. The store
// therefore grows: at partial=5000 x 100 iterations that is +500k keys on 3.37M,
// about 15%. Keys wrap if more iterations run than the pool allows, after which
// merges become updates instead of inserts — keep -benchtime <= 100x.
func BenchmarkRealShape_Merge(b *testing.B) {
	partialSizes := []int{50, 500, 5_000}
	const freshPool = 500_000 // 100 iterations x the largest partial

	allKeys := makeHexKeys(realStoreKeys + freshPool)
	storeKeys := allKeys[:realStoreKeys]
	freshKeys := allKeys[realStoreKeys:]
	fullKV := makeHexKV(storeKeys, realValueSize)
	// Guard: a key generator collision would silently shrink the store and make
	// the whole comparison a measurement of the wrong size.
	if len(fullKV) != realStoreKeys {
		b.Fatalf("key collision: want %d distinct keys, got %d", realStoreKeys, len(fullKV))
	}

	for _, partialN := range partialSizes {
		b.Run(fmt.Sprintf("store=3.4M_keys/partial=%d", partialN), func(b *testing.B) {
			impl := newBenchKVImpl(b)
			if _, err := impl.Load(mapToIter(fullKV)); err != nil {
				b.Fatal(err)
			}
			full := newBenchFullKVFromImpl(impl)

			maxBatches := freshPool / partialN
			batch := 0

			b.ReportAllocs()
			b.SetBytes(int64(partialN) * int64(realValueSize))
			b.ResetTimer()

			for b.Loop() {
				b.StopTimer()
				off := (batch % maxBatches) * partialN
				partial := newBenchPartialStore(makeHexKV(freshKeys[off:off+partialN], realValueSize))
				batch++
				b.StartTimer()

				if err := full.Merge(partial); err != nil {
					b.Fatal(err)
				}
			}

			b.StopTimer()
			impl.Close()
		})
	}
}

// BenchmarkRealShape_Load measures hydrating the real store, and reports the
// RETAINED heap afterwards (heap_inuse_MB) — the number behind the "unpredictable
// RAM spike" the mmap backend is meant to remove. b.ReportAllocs' B/op counts
// total allocations, which is not the same as what stays resident.
func BenchmarkRealShape_Load(b *testing.B) {
	allKeys := makeHexKeys(realStoreKeys)
	fullKV := makeHexKV(allKeys, realValueSize)

	b.Run("store=3.4M_keys", func(b *testing.B) {
		b.ReportAllocs()
		var heapAfter uint64

		for b.Loop() {
			b.StopTimer()
			impl := newBenchKVImpl(b)
			b.StartTimer()

			if _, err := impl.Load(mapToIter(fullKV)); err != nil {
				b.Fatal(err)
			}

			b.StopTimer()
			runtime.GC()
			var m runtime.MemStats
			runtime.ReadMemStats(&m)
			heapAfter = m.HeapInuse
			impl.Close()
			b.StartTimer()
		}

		// Subtract the source map, which is the harness's own baseline and is not
		// part of what the store implementation retains.
		b.ReportMetric(float64(heapAfter)/1e6, "heap_inuse_MB")
	})
}
