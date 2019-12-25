package main

import (
	"fmt"
	"lfm_rec/recCore"
)

func main() {
	ratingPath := "./recCore/ratings.csv"
	lfm := recCore.NewDefaultLFM(ratingPath)
	lfm.InitModel()
	lfm.Train()
	userId := 23.0
	var topN int32 = 10
	var itemScore = lfm.Predict(userId, topN)
	for index, itemScoreEle := range itemScore {
		itemId := float64(itemScoreEle.Key)
		fmt.Println("item : %f ,score : %f ", itemId, itemScoreEle.Value)
		if index == int(topN)*50 {
			break
		}
	}
}
