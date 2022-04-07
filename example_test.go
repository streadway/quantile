package quantile

import (
	"fmt"
	"time"
)

var rpcs *Estimator

func observeSeconds(est *Estimator, begin time.Time) {
	est.Add(float64(time.Now().Sub(begin)) / float64(time.Second))
}

func Work() {
	defer observeSeconds(rpcs, time.Now())

	// Dance your cares away,
	// Worry's for another day.
	// Let the music play,
}

func ExampleEstimator() {
	// We know we want to query the 95th and 99th, with the 95th a little less accurately.
	rpcs = New(Known(0.95, 0.005), Known(0.99, 0.001))

	Work()
	Work()

	// Report the percentiles
	fmt.Println("95th: ", rpcs.Get(0.95))
	fmt.Println("99th: ", rpcs.Get(0.99))
}
