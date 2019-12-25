package recCore

import (
	"fmt"
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
	"sort"
)

func main() {
	data := []float64{3, 6, 2, 7, 4, 5, 9, 8}
	matrix := mat.NewDense(4, 2, data)
	//fmt.Println(matrix) 34 23 -14 29
	arr := matrix.ColView(0)
	arr1 := matrix.ColView(1)
	res := mat.Dot(arr, arr1)
	res1 := mat.Sum(arr)
	res2 := mat.Sum(arr1)
	az := []int{0, 1, 2, 3}
	for _, key := range az {
		fmt.Println(3 * arr1.AtVec(key))
	}
	fmt.Println(res, res1, res2)
	//lez := arr.(*mat.VecDense).Len()
	arrz := arr.(*mat.VecDense)
	arrz.ScaleVec(5, arrz)
	//var kk mat.VecDense
	//kk.ScaleVec(5,arrz)
	fmt.Println(arrz)

	src := rand.New(rand.NewSource(1))
	var uni = distuv.Uniform{11, 55, src}
	unif := testUniform(uni)
	fmt.Println(unif)

}

func testUniform(u distuv.Uniform) []float64 {
	const (
		n = 1e5
	)
	x := make([]float64, n)
	generateSamples(x, u)
	sort.Float64s(x)
	fmt.Println(x)
	return x
}

//type Randers interface {
//	Rand() float64
//}
func generateSamples(x []float64, r distuv.Rander) {
	for i := range x {
		x[i] = r.Rand()
	}
}
