package main

import (
	"fmt"

	tg "github.com/galeone/tfgo"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

func main() {

	// Import model.
	model := tg.LoadModel("EXPORT", []string{"serve"}, &tf.SessionOptions{})

	// Create input tensors.
	deviceInput, _ := tf.NewTensor([]string{"desktop"})
	hourInput, _ := tf.NewTensor([]int64{19})
	domainInput, _ := tf.NewTensor([]string{"foo.com"})

	// Predict.
	results := model.Exec(
		[]tf.Output{
			model.Op("linear/head/predictions/probabilities", 0),
		}, map[tf.Output]*tf.Tensor{
			model.Op("device_type_placeholder", 0): deviceInput,
			model.Op("hour_placeholder", 0):        hourInput,
			model.Op("domain_placeholder", 0):      domainInput,
		},
	)
	predictions := results[0].Value().([][]float32)
	fmt.Println(predictions)
}
