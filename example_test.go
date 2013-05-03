package quantile_test

import (
	quantile "." // import fully qualified in your code
	"fmt"
	"time"
)

var rpcs *quantile.Estimator

func observeSeconds(est *quantile.Estimator, begin time.Time) {
	est.Update(float64(time.Now().Sub(begin)) / float64(time.Second))
}

func Work() {
	defer observeSeconds(rpcs, time.Now())

	// Dance your cares away, 
	// Worry's for another day. 
	// Let the music play, 
}

func ExampleEstimator() {
	// We know we want to query the 95th and 99th, with the 95th a little less accurately.
	rpcs = quantile.New(quantile.Target(0.95, 0.005),quantile.Target(0.99, 0.001))

	Work()
	Work()

	// Report the percentiles
	fmt.Println("95th: ", rpcs.Query(0.95))
	fmt.Println("99th: ", rpcs.Query(0.99))
}
