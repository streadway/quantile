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
	p := percentile(0.99, normDistSlice(1000000, 1.0, 0.0))

	if p+e < z99 || p-e > z99 {
		t.Fatalf("99th percentile (%f) of normal distribution doesn't match the z-score of %f within an error %f", p, z99, e)
	}
}

type item struct {
	v     float64
	rank  int
	delta int
	next  *item
}

type quantile struct {
	q float64 // phi like 0.50 (mean) or 0.99th
	e float64 // bias like 0.1 or 0.001
}

type quantileEstimator struct {
	head         *item
	items        int
	observations int

	targets []quantile
	buffer  []float64
}

func newBiasedQuantileEstimator(targets ...quantile) *quantileEstimator {
	return &quantileEstimator{
		targets: targets,
		buffer:  make([]float64, 0, 500),
	}
}

func (est *quantileEstimator) minError(rank int, n int) int {
	min := float64(n + 1)

	for _, quantile := range est.targets {
		var err float64

		if rank < int(quantile.q*float64(n)) {
			err = 2 * quantile.e * (float64(n - rank)) / (1 - quantile.q)
		} else {
			err = 2 * quantile.e * float64(rank) / quantile.q
		}

		if err < min {
			min = err
		}
	}

	return int(min)
}

func (est *quantileEstimator) observe(v float64, rank, delta int, next *item) *item {
	est.observations++
	est.items++

	return &item{
		v:     v,
		rank:  rank,
		delta: delta,
		next:  next,
	}
}

// merges the batch
func (est *quantileEstimator) update(batch []float64) {
	if est.head == nil {
		est.head = est.observe(batch[0], 1, 0, nil)
		batch = batch[1:]
	}

	cur := est.head
	rank := 0

	for _, v := range batch {
		// min
		if v < est.head.v {
			est.head = est.observe(v, 1, 0, est.head)
			cur = est.head
			continue
		}

		// cursor
		for cur.next != nil && cur.next.v < v {
			rank += cur.rank
			cur = cur.next
		}

		// max
		if cur.next == nil {
			cur.next = est.observe(v, 1, 0, nil)
			continue
		}

		cur.next = est.observe(v, 1, est.minError(rank, est.observations)-1, cur.next)
	}
}

func (est *quantileEstimator) flush() {
	sort.Float64Slice(est.buffer).Sort()
	est.update(est.buffer)
	est.buffer = est.buffer[0:0]
	est.compress()
}

func (est *quantileEstimator) Update(s float64) {
	est.buffer = append(est.buffer, s)
	if len(est.buffer) == cap(est.buffer) {
		est.flush()
	}
}

func (est *quantileEstimator) compress() {
	rank := 0
	cur := est.head
	for cur != nil && cur.next != nil {
		if cur.rank+cur.next.rank+cur.next.delta <= est.minError(rank, est.observations) {
			// merge with previous
			cur.v = cur.next.v
			cur.rank += cur.next.rank
			cur.delta = cur.next.delta
			cur.next = cur.next.next

			est.items--
		}
		rank += cur.rank
		cur = cur.next
	}
}

func (est *quantileEstimator) Query(q float64) float64 {
	cur := est.head
	if cur == nil {
		return 0
	}

	est.flush()

	quartile := int(q * float64(est.observations))
	maxrank := quartile + est.minError(quartile, est.observations)/2

	var rank int
	for cur.next != nil {
		rank += cur.rank
		if rank+cur.next.rank+cur.next.delta > maxrank {
			return cur.v
		}
		cur = cur.next
	}
	return cur.v
}

func TestQE(t *testing.T) {
	BenchmarkQuantileEstimator(&testing.B{N: 100000})
}

func BenchmarkQuantileEstimator(b *testing.B) {
	est := newBiasedQuantileEstimator(quantile{0.01, 0.001}, quantile{0.05, 0.01}, quantile{0.50, 0.01}, quantile{0.99, 0.001})
	for i := 0; i < b.N+2; i++ {
		est.Update(rand.NormFloat64()*1.0 + 0.0)
	}

	println(est.items)
	println(est.Query(0.01))
	println(est.Query(0.50))
	println(est.Query(0.99))
}
