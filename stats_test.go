package sensor

import (
	"container/list"
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
	v           float64
	rank, delta int
}

type quantile struct {
	q float64 // phi like 0.50 (mean) or 0.99th
	e float64 // bias like 0.1 or 0.001
}

type quantileEstimator struct {
	samples      *list.List // <rank>
	targets      []quantile
	buffer       []float64
	observations int
}

func newBiasedQuantileEstimator(targets ...quantile) *quantileEstimator {
	return &quantileEstimator{
		targets: targets,
		samples: list.New(),
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

// merges the batch
func (est *quantileEstimator) update(batch []float64) {
	for _, v := range batch {
		rank := 0

		e := est.samples.Front()
		if e == nil {
			e = est.samples.PushFront(&item{v, 1, 0})
			est.observations++
			continue
		}

		for e != nil {
			cur := e.Value.(*item)
			if v < cur.v {
				break
			}
			rank = rank + cur.rank
			e = e.Next()
		}

		if e == nil {
			est.samples.PushBack(&item{v, 1, est.minError(rank, est.observations) - 1})
		} else {
			est.samples.InsertBefore(&item{v, 1, est.minError(rank, est.observations) - 1}, e)
		}

		est.observations++
	}
}

func (est *quantileEstimator) flush() {
	//sort.Float64Slice(est.buffer).Sort()
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
	if est.samples.Len() < 2 {
		panic("only compress filled samples")
	}

	pre := est.samples.Front()
	cur := pre.Next()

	for rank := 1; cur != nil; pre, cur = cur, cur.Next() {
		p, c := pre.Value.(*item), cur.Value.(*item)
		if p.rank+c.rank+c.delta <= est.minError(rank, est.observations) {
			c.rank += p.rank
			est.samples.Remove(pre)
		} else {
			rank += p.rank
		}

	}
}

func (est *quantileEstimator) Query(q float64) float64 {
	est.flush()

	want := int(q * float64(est.observations))
	rank := 0
	if pre := est.samples.Front(); pre != nil {
		cur := pre.Next()
		for cur != nil {
			p, c := pre.Value.(*item), cur.Value.(*item)
			rank += p.rank
			if rank+c.rank+c.delta > want+est.minError(want, est.observations)/2 {
				return p.v
			}
			pre, cur = cur, cur.Next()
		}
		return est.samples.Back().Value.(*item).v
	}
	return 0
}

func BenchmarkQuantileEstimator(b *testing.B) {
	est := newBiasedQuantileEstimator(quantile{0.01, 0.01}, quantile{0.05, 0.01}, quantile{0.50, 0.01}, quantile{0.99, 0.001})
	b.SetBytes(int64(8 * b.N + 2))
	for i := 0; i < b.N + 2; i++ {
		est.Update(rand.NormFloat64()*1.0+0.0)
	}

	println(est.samples.Len())
	println(est.Query(0.01))
	println(est.Query(0.50))
	println(est.Query(0.99))
}
