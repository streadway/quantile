package sensor

import (
	"math"
	"math/rand"
	"runtime"
	"sort"
	"testing"
	"testing/quick"
)

type item struct {
	v     float64
	rank  float64
	delta float64
	next  *item
}

type Estimate struct {
	Quantile float64 // phi like 0.50 (mean) or 0.99th
	Error    float64 // epsilon like 0.1 or 0.001
}

type target struct {
	q  float64 // from Estimate
	f1 float64 // cached coefficient for fi  q*n <= rank <= n
	f2 float64 // cached coefficient for fii 0 <= rank <= q*n
}

type quantileEstimator struct {
	// linked list datastructure, bookeeping in observe/recycle
	head  *item
	items int

	// avoids conversion during invariant checks
	observations float64

	targets []target
	buffer  []float64
	pool    chan *item
}

func newBiasedQuantileEstimator(quantiles ...Estimate) *quantileEstimator {
	targets := make([]target, 0, len(quantiles))
	for _, est := range quantiles {
		targets = append(targets, target{
			q:  est.Quantile,
			f1: 2 * est.Error / est.Quantile,
			f2: 2 * est.Error / (1 - est.Quantile),
		})
	}

	return &quantileEstimator{
		targets: targets,
		buffer:  make([]float64, 0, 512),
		pool:    make(chan *item, 1024),
	}
}

// targetted
func (est *quantileEstimator) invariant(rank float64, n float64) float64 {
	min := (n + 1)

	for _, t := range est.targets {
		var err float64

		if rank <= math.Floor(t.q*n) {
			err = t.f2 * (n - rank)
		} else {
			err = t.f1 * rank
		}

		if err < min {
			min = err
		}
	}

	return math.Floor(min)
}

func (est *quantileEstimator) observe(v float64, rank, delta float64, next *item) *item {
	est.observations++
	est.items++

	// reuse or allocate
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

func (est *quantileEstimator) recycle(old *item) {
	est.items--
	select {
	case est.pool <- old:
	default:
	}
}

// merges the batch
func (est *quantileEstimator) update(batch []float64) {
	// initial data
	if est.head == nil {
		est.head = est.observe(batch[0], 1, 0, nil)
		batch = batch[1:]
	}

	rank := 0.0
	cur := est.head
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

		cur.next = est.observe(v, 1, est.invariant(rank, est.observations)-1, cur.next)
	}
}

func (est *quantileEstimator) compress() {
	rank := 0.0
	cur := est.head
	for cur != nil && cur.next != nil {
		if cur.rank+cur.next.rank+cur.next.delta <= est.invariant(rank, est.observations) {
			// merge with previous/head
			removed := cur.next

			cur.v = removed.v
			cur.rank += removed.rank
			cur.delta = removed.delta
			cur.next = removed.next

			est.recycle(removed)
		}
		rank += cur.rank
		cur = cur.next
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

func (est *quantileEstimator) Query(q float64) float64 {
	est.flush()

	cur := est.head
	if cur == nil {
		return 0
	}

	midrank := math.Floor(q * est.observations)
	maxrank := midrank + math.Floor(est.invariant(midrank, est.observations)/2)

	rank := 0.0
	for cur.next != nil {
		rank += cur.rank
		if rank+cur.next.rank+cur.next.delta > maxrank {
			return cur.v
		}
		cur = cur.next
	}
	return cur.v
}

func TestErrorBounds(t *testing.T) {
	f := func(N uint32) bool {
		q := 0.99
		e := 0.0001
		n := int(N) % 1000000
		est := newBiasedQuantileEstimator(Estimate{q, e})
		obs := make([]float64, 0, n)

		for i := 0; i < n; i++ {
			s := rand.NormFloat64()*1.0 + 0.0
			obs = append(obs, s)
			est.Update(s)
		}

		sort.Float64Slice(obs).Sort()

		// "v" the estimate
		estimate := est.Query(q)

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

	if err := quick.Check(f, nil); err != nil {
		t.Error(err)
	}
}

func BenchmarkQuantileEstimator(b *testing.B) {
	est := newBiasedQuantileEstimator(Estimate{0.01, 0.001}, Estimate{0.05, 0.01}, Estimate{0.50, 0.01}, Estimate{0.99, 0.001})

	// Warmup
	b.StopTimer()
	for i := 0; i < 10000; i++ {
		est.Update(rand.NormFloat64()*1.0 + 0.0)
	}
	b.StartTimer()

	var pre runtime.MemStats
	runtime.ReadMemStats(&pre)

	for i := 0; i < b.N; i++ {
		est.Update(rand.NormFloat64()*1.0 + 0.0)
	}

	var post runtime.MemStats
	runtime.ReadMemStats(&post)

	b.Logf("allocs: %d items: %d 0.01: %f 0.50: %f 0.99: %f", post.TotalAlloc-pre.TotalAlloc, est.items, est.Query(0.01), est.Query(0.50), est.Query(0.99))
}
