package sensor

import (
	"math/rand"
	"runtime"
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

type item struct {
	v     float64
	rank  int
	delta int
	next  *item
}

type Estimate struct {
	Quantile float64 // phi like 0.50 (mean) or 0.99th
	Error    float64 // epsilon like 0.1 or 0.001
}

type target struct {
	q  float64 // from Estimate
	e  float64 // from Estimate
	f1 float64 // cached coefficient q*n <= rank <= n
	f2 float64 // cached coefficient 0 <= rank <= q*n
}

type quantileEstimator struct {
	head         *item
	items        int
	observations int

	targets []target
	buffer  []float64
	pool    chan *item
}

func newBiasedQuantileEstimator(quantiles ...Estimate) *quantileEstimator {
	targets := make([]target, 0, len(quantiles))
	for _, est := range quantiles {
		targets = append(targets, target{
			q:  est.Quantile,
			e:  est.Error,
			f1: 2 * est.Error / (1 - est.Quantile),
			f2: 2 * est.Error / est.Quantile,
		})
	}
	return &quantileEstimator{
		targets: targets,
		buffer:  make([]float64, 0, 512),
		pool:    make(chan *item, 4096),
	}
}

func (est *quantileEstimator) minError(rank int, n int) int {
	min := float64(n + 1)

	for _, t := range est.targets {
		var err float64

		if rank < int(t.q*float64(n)) {
			err = t.f2 * float64(n-rank)
		} else {
			err = t.f1 * float64(rank)
		}

		if err < min {
			min = err
		}
	}

	return int(min)
}

func (est *quantileEstimator) observe(v float64, rank, delta int, next *item) *item {
	//return est.observeAlloc(v, rank, delta, next)
	est.observations++
	est.items++

	select {
	case old := <-est.pool:
		old.v = v
		old.rank = rank
		old.delta = delta
		old.next = next
		return old
	default:
		return &item{
			v:     v,
			rank:  rank,
			delta: delta,
			next:  next,
		}
	}
	panic("unreachable")
}

func (est *quantileEstimator) reuse(old *item) {
	est.items--
	select {
	case est.pool <- old:
	default:
	}
}

func (est *quantileEstimator) observeAlloc(v float64, rank, delta int, next *item) *item {
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
	// initial case
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

		// cursor (possibility to fuse compress here)
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
			// merge with previous/head
			removed := cur.next

			cur.v = removed.v
			cur.rank += removed.rank
			cur.delta = removed.delta
			cur.next = removed.next

			est.reuse(removed)
		}
		rank += cur.rank
		cur = cur.next
	}
}

func (est *quantileEstimator) Query(q float64) float64 {
	est.flush()

	cur := est.head
	if cur == nil {
		return 0
	}

	quantile := int(q * float64(est.observations))
	maxrank := quantile + est.minError(quantile, est.observations)/2
	rank := 0

	for cur.next != nil {
		if rank+cur.next.rank+cur.next.delta > maxrank {
			return cur.v
		}
		rank += cur.rank
		cur = cur.next
	}
	return cur.v
}

func TestQE(t *testing.T) {
	BenchmarkQuantileEstimator(&testing.B{N: 100000})
}

func BenchmarkQuantileEstimator(b *testing.B) {
	est := newBiasedQuantileEstimator(Estimate{0.01, 0.001}, Estimate{0.05, 0.01}, Estimate{0.50, 0.01}, Estimate{0.99, 0.001})
	b.StopTimer()

	for i := 0; i < 10000; i++ {
		est.Update(rand.NormFloat64()*1.0 + 0.0)
	}

	b.StartTimer()

	println(b.N)

	var pre runtime.MemStats
	runtime.ReadMemStats(&pre)

	for i := 0; i < b.N; i++ {
		est.Update(rand.NormFloat64()*1.0 + 0.0)
	}
	var post runtime.MemStats
	runtime.ReadMemStats(&post)

	println("alloc:", post.TotalAlloc-pre.TotalAlloc)

	println(est.items)
	println(est.Query(0.01))
	println(est.Query(0.50))
	println(est.Query(0.99))
}
