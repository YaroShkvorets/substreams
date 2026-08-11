package store

// Benchmarks at the REAL store SHAPE measured on production bsc-v5 tier1, swept
// across store sizes so you can see how the mmap penalty scales with key count.
//
// Shape comes from the quicksave "key_count"/"total_size_bytes" logs, e.g.
// uniswap_v2:store_pool on BSC:
//
//	key_count        = 3,373,228
//	total_size_bytes = 357,562,168   -> ~106 bytes per entry
//	key              = Hex::encode(address) -> 40 lowercase hex chars
//	value            => ~66 bytes (106 - 40)
//
// Get the same figures for any other store with:
//
//	kubectl -n <ns> logs -l app.kubernetes.io/name=substreams-tier1 --since=12h \
//	  | grep '"key_count"' | jq -c '.store | {name, key_count, total_size_bytes}' | sort -u
//
// This differs from kv_impl_bench_test.go's grid in two ways that matter, and they
// pull in opposite directions:
//
//  1. Tiny values (66 B vs 1 KB), so far more keys per megabyte. High key count
//     with small values is the worst case for Go map per-entry overhead and the
//     best case for bbolt's packed B+tree pages — favours mmap on memory.
//  2. Keys are pseudo-random hex, not sequential "key:%08d". Sequential inserts
//     are bbolt's best case (append to the rightmost leaf, few splits at
//     FillPercent=1.0); random order is its worst, since almost every insert lands
//     on a different full page and forces a split — favours memory on write speed.
//
// The two benchmarks need DIFFERENT -benchtime values, so run them separately.
// -benchtime=Nx (iteration count, not duration) is mandatory for both: Go's
// wall-clock mode would either run for hours or stop at a handful of iterations.
//
//	# Merge — 100x is the ceiling (see the freshPool note on BenchmarkRealShape_Merge).
//	go test -run='^$' -bench=RealShape_Merge -benchmem -benchtime=100x -count=6 ./storage/store/ | tee /tmp/merge_memory.txt
//	SUBSTREAMS_STORE_BACKEND=mmap go test -run='^$' -bench=RealShape_Merge -benchmem -benchtime=100x -count=6 ./storage/store/ | tee /tmp/merge_mmap.txt
//
//	# Load — rebuilds the store per iteration, so keep it small. heap_inuse_MB is
//	# deterministic, 3 iterations is plenty.
//	go test -run='^$' -bench=RealShape_Load -benchmem -benchtime=3x -count=3 ./storage/store/ | tee /tmp/load_memory.txt
//	SUBSTREAMS_STORE_BACKEND=mmap go test -run='^$' -bench=RealShape_Load -benchmem -benchtime=3x -count=3 ./storage/store/ | tee /tmp/load_mmap.txt
//
//	benchstat /tmp/merge_memory.txt /tmp/merge_mmap.txt
//
// ns/op is ONE Merge (or one Load) — unlike kv_impl_bench_test.go, where an
// iteration bundles ops=N operations. Divide by the partial= value for per-key cost.
//
// Results OVERSTATE the mmap penalty on Apple Silicon: bbolt takes its page size
// from os.Getpagesize(), 16KB on darwin/arm64 vs 4KB on linux/amd64, so each
// copy-on-write page rewrite moves 4x more bytes. Measured gap on the sequential
// grid was ~1.3-1.8x, so prefer running this on a linux/amd64 box. Also
// newBenchKVImpl leaves InitialMmapSize unset (128MiB) where production reserves
// 1.5x the store size limit, so this pays remap stalls production does not.

import (
	"fmt"
	"runtime"
	"testing"
)

const (
	realValueSize = 66 // 106 B/entry - 40 B hex key
	realKeyHexLen = 40
)

// realShapeSizes is the store-size sweep. Values stay at realValueSize and keys
// stay random hex, so only the key count varies.
var realShapeSizes = []struct {
	name    string
	numKeys int
}{
	{"10k_keys", 10_000},
	{"100k_keys", 100_000},
	{"1M_keys", 1_000_000},
	// {"3.4M_keys", 3_373_228}, // real uniswap_v2:store_pool on BSC; ~15 min/arm
}

func maxRealShapeKeys() int {
	m := 0
	for _, sz := range realShapeSizes {
		if sz.numKeys > m {
			m = sz.numKeys
		}
	}
	return m
}

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
		// 40 hex chars = 160 bits; refresh the source every 16 nibbles.
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

// BenchmarkRealShape_Merge merges a small partial into a large FullKV, swept over
// store size. store_pool uses updatePolicy: set and is written once per pool
// creation, so a 1000-block segment touches only a handful of keys — hence the
// small partial sizes. Partial keys are FRESH (never present in the store), which
// is the real case for a pool store and the one that forces bbolt page splits.
//
// The store is hydrated ONCE per sub-benchmark and merged into repeatedly, rather
// than rebuilt per iteration. Rebuilding a large store cost ~10s of setup per
// iteration, capping us at 5 iterations — far too few to average out GC of a
// multi-million-entry map, and it produced 100x spread inside a single config.
// Amortising setup buys 100 iterations, enough for that noise to average out.
//
// Consequences to keep in mind:
//   - The store GROWS during the run. Combinations are skipped unless 100
//     iterations stay under +50% of the nominal store size, which yields a
//     triangular matrix: 10k->{50}, 100k->{50,500}, 1M->{50,500,5000}.
//   - Keys wrap if more than 100 iterations run, after which merges become updates
//     instead of inserts and skip the page splits. Keep -benchtime <= 100x.
func BenchmarkRealShape_Merge(b *testing.B) {
	partialSizes := []int{50, 500, 5_000}
	const (
		freshPool = 500_000 // 100 iterations x the largest partial
		maxIters  = 100
	)

	maxStore := maxRealShapeKeys()
	allKeys := makeHexKeys(maxStore + freshPool)
	// Fresh keys are taken past the LARGEST store, so they are absent from every
	// store size in the sweep.
	freshKeys := allKeys[maxStore:]

	for _, sz := range realShapeSizes {
		fullKV := makeHexKV(allKeys[:sz.numKeys], realValueSize)
		// Guard: a key generator collision would silently shrink the store and make
		// the whole comparison a measurement of the wrong size.
		if len(fullKV) != sz.numKeys {
			b.Fatalf("key collision: want %d distinct keys, got %d", sz.numKeys, len(fullKV))
		}

		for _, partialN := range partialSizes {
			if partialN*maxIters > sz.numKeys/2 {
				continue // would grow the store by more than 50% over the run
			}
			b.Run(fmt.Sprintf("store=%s/partial=%d", sz.name, partialN), func(b *testing.B) {
				impl := newBenchKVImpl(b)
				if _, err := impl.Load(mapToIter(fullKV)); err != nil {
					b.Fatal(err)
				}
				full := newBenchFullKVFromImpl(impl)

				maxBatches := len(freshKeys) / partialN
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
}

// BenchmarkRealShape_Load measures hydrating the store at each size, and reports
// the RETAINED heap afterwards (heap_inuse_MB) — the number behind the
// "unpredictable RAM spike" the mmap backend is meant to remove. b.ReportAllocs'
// B/op counts total allocations, which is not the same as what stays resident.
//
// CAVEAT: the harness keeps the source map alive for the whole run, and
// memoryKVImpl.Load stores the iterator's value slices by reference rather than
// copying — so the memory arm SHARES values with the source map and its retained
// cost is understated here. Treat the gap as a lower bound; the in-cluster peak
// heap numbers are the real measurement.
func BenchmarkRealShape_Load(b *testing.B) {
	allKeys := makeHexKeys(maxRealShapeKeys())

	for _, sz := range realShapeSizes {
		fullKV := makeHexKV(allKeys[:sz.numKeys], realValueSize)

		b.Run(fmt.Sprintf("store=%s", sz.name), func(b *testing.B) {
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

			b.ReportMetric(float64(heapAfter)/1e6, "heap_inuse_MB")
		})
	}
}
