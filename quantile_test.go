// Copyright 2013 Sean Treadway, SoundCloud Ltd. All rights reserved.  Use of
// this source code is governed by a BSD-style license that can be found in the
// LICENSE file.

package quantile

import (
	"math/rand"
	"runtime"
	"sort"
	"testing"
	"testing/quick"
)

func withinError(t *testing.T, fn Estimate, q, e float64) func(N uint32) bool {
	return func(N uint32) bool {
		n := int(N % 1000000) + 1
		est := New(fn)
		obs := make([]float64, 0, n)

		for i := 0; i < n; i++ {
			s := rand.NormFloat64()*1.0 + 0.0
			obs = append(obs, s)
			est.Add(s)
		}

		sort.Float64Slice(obs).Sort()

		// "v" the estimate
		estimate := est.Get(q)

		// A[⌈(φ − ε)n⌉] ≤ v ≤ A[⌈(φ + ε)n⌉]
		// The bounds of the estimate
		lower := int((q-e)*float64(n)) - 1
		upper := int((q+e)*float64(n)) + 1

		// actual v
		exact := int(q * float64(n))

		min := obs[0]
		if lower > 0 {
			min = obs[lower]
		}

		max := obs[len(obs)-1]
		if upper < len(obs) {
			max = obs[upper]
		}

		t.Logf("delta: %d ex: %f min: %f (%f) max: %f (%f) est: %f n: %d l: %d",
			upper-lower, obs[exact], min, obs[0], max, obs[len(obs)-1], estimate, n, est.items)

		fits := (min <= estimate && estimate <= max)

		if !fits {
			for cur := est.head; cur != nil; cur = cur.next {
				t.Log(cur)
			}
		}

		return fits
	}
}

func TestErrorKnownd(t *testing.T) {
	if err := quick.Check(withinError(t, Known(0.99, 0.0001), 0.99, 0.0001), nil); err != nil {
		t.Error(err)
	}
}

func TestErrorUnknowned(t *testing.T) {
	if err := quick.Check(withinError(t, Unknown(0.0001), 0.99, 0.0001), nil); err != nil {
		t.Error(err)
	}
}

func BenchmarkQuantileEstimator(b *testing.B) {
	est := New(Known(0.01, 0.001), Known(0.05, 0.01), Known(0.50, 0.01), Known(0.99, 0.001))

	// Warmup
	b.StopTimer()
	for i := 0; i < 10000; i++ {
		est.Add(rand.NormFloat64()*1.0 + 0.0)
	}
	b.StartTimer()

	var pre runtime.MemStats
	runtime.ReadMemStats(&pre)

	for i := 0; i < b.N; i++ {
		est.Add(rand.NormFloat64()*1.0 + 0.0)
	}

	var post runtime.MemStats
	runtime.ReadMemStats(&post)

	b.Logf("allocs: %d items: %d 0.01: %f 0.50: %f 0.99: %f", post.TotalAlloc-pre.TotalAlloc, est.items, est.Get(0.01), est.Get(0.50), est.Get(0.99))
}
