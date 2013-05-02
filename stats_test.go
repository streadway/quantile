package sensor

import (
	"math/rand"
	"sort"
	"testing"
)

func normDistSlice(size int, stddev, mean float64) []float64 {
	res := make([]float64, 0, size)
	for i := 0; i < size; i++ {
		res = append(res, rand.NormFloat64()*stddev+mean)
	}
	return res
}

func percentile(pct float64, samples []float64) float64 {
	s := sort.Float64Slice(samples)
	s.Sort()
	return s[int(float64(len(s))*pct)]
}

func Test99Percentile(t *testing.T) {
	z99 := 2.33
	e := 0.1
	p := percentile(0.99, normDistSlice(100000, 1.0, 0.0))

	if p+e < z99 || p-e > z99 {
		t.Fatalf("99th percentile (%f) of normal distribution doesn't match the z-score of %f within an error %f", p, z99, e)
	}
}
